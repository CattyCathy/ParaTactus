using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using NUnit.Framework;
using ParaTactus.Audio;
using ParaTactus.Decoding;
using ParaTactus.Tracking;

namespace ParaTactus.Tests
{
    /// <summary>
    /// Turns osu! beatmaps into labelled training data for a beat-level reader of the model's activation.
    /// </summary>
    /// <remarks>
    /// The reader this is for does one job the heuristic post-processor cannot: deciding the metrical level. Whether a
    /// run of doubled gaps is a passage where the tracker lost every other beat or a passage where the music genuinely
    /// halved is not answerable from the beat positions - the gaps have the same shape either way - and every rule
    /// written to decide it has been wrong on some track and right on another. The model's own activation separates the
    /// two cases but not by a threshold that holds across tracks, which is what a learned mapping is for.
    ///
    /// The labels come from a beatmap's uninherited timing points. Those are the grid the mapper snapped the objects
    /// to, so they are the author's own statement of where the beats are, and they are free of the ambiguity this is
    /// trying to resolve: a timing point says what the beat is and there is no second reading of it. Two things are
    /// deliberately not used. Inherited points, whose beat length is negative, are the slider and scroll speed and say
    /// nothing about the beat. Hit objects are placed on subdivisions as well as on beats, so a run of them forty
    /// milliseconds apart is a snapping choice and not a tempo.
    ///
    /// What this measures before any of that is turned into training data is whether the labels and the model agree at
    /// all. If the activation does not rise at the grid's beats, then either the grid is not the beat or the model is
    /// not reading it, and a network trained on the pair would be learning the disagreement.
    ///
    /// Set <c>OSUTEST_CORPUS</c> and optionally <c>OSUTEST_DATASET</c> for where to write.
    /// </remarks>
    [Explicit("Needs OSUTEST_CORPUS pointing at a folder of .osu files with their audio, and the model.")]
    public class TrainingDataProbe
    {
        private const string defaultModel = "beat-this-final0-int8";

        [OneTimeSetUp]
        public void SetUp() => AudioTestEnvironment.Initialise();

        [Test]
        public void BuildLabelsAndReportWhetherTheModelAgreesWithThem()
        {
            string corpus = Environment.GetEnvironmentVariable("OSUTEST_CORPUS");

            if (string.IsNullOrEmpty(corpus) || !Directory.Exists(corpus))
                Assert.Ignore("Set OSUTEST_CORPUS to a folder holding .osu files and their audio.");

            string modelPath = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                               ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", $"{defaultModel}.onnx");

            if (!File.Exists(modelPath))
                Assert.Ignore($"no model at {modelPath}");

            // Every beatmap in the folder, not one per set: a set's difficulties are usually the same song with more or
            // fewer objects, and for labels taken from the timing points they are nearly the same example twice.
            var maps = new List<(string Map, string Audio)>();

            foreach (string map in Directory.EnumerateFiles(corpus, "*.osu", SearchOption.AllDirectories))
            {
                string folder = Path.GetDirectoryName(map);
                string audio = folder == null ? null : audioFor(map, folder);

                if (audio != null)
                    maps.Add((map, audio));
            }

            if (maps.Count == 0)
                Assert.Ignore($"no .osu file with audio beside it in {corpus}");

            TestContext.Out.WriteLine($"{maps.Count} beatmap(s) with audio in {corpus}");
            TestContext.Out.WriteLine($"model: {Path.GetFileName(modelPath)}");
            TestContext.Out.WriteLine("");

            var summaries = new List<(string Name, int Beats, double Period, double AtBeat, double OffBeat, double Lag, double BeatRate, double AtOne, double AtTwo, double AtFour)>();

            foreach ((string map, string audio) in maps)
            {
                try
                {
                    summaries.Add(measure(map, audio, modelPath));
                }
                catch (Exception error)
                {
                    TestContext.Out.WriteLine($"  {Path.GetFileNameWithoutExtension(map)}: {error.Message}");
                }
            }

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("the label grid against the model's activation, per beatmap:");
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine($"  {"beatmap",-42} {"period",7} {"at beat",8} {"off beat",9} {"rise",7} {"lag",6} {"beat/s",7} {"pos 1x",7} {"2x",6} {"4x",6}");

            foreach (var row in summaries.OrderBy(s => s.BeatRate))
            {
                TestContext.Out.WriteLine($"  {Trim(row.Name),-42} {row.Period,6:0}ms {row.AtBeat,7:+0.00;-0.00;0.00} {row.OffBeat,8:+0.00;-0.00;0.00} {row.AtBeat - row.OffBeat,6:+0.00;-0.00;0.00} {row.Lag,5:0}ms {row.BeatRate,6:0.00} {row.AtOne,6:0}% {row.AtTwo,5:0}% {row.AtFour,5:0}%");
            }

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("  'at beat' is the activation at the grid's beats, 'off beat' at the points halfway between");
            TestContext.Out.WriteLine("  two of them, both as a difference from the track's own mean, and 'rise' is the gap between");
            TestContext.Out.WriteLine("  them. 'lag' is the shift, within a tenth of a second, at which the two line up best.");
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("  'pos 1x', '2x' and '4x' are the share of the grid's beats that land on a frame the model is");
            TestContext.Out.WriteLine("  positive about, with each beat made one, two and four times as long. A map's timing points");
            TestContext.Out.WriteLine("  are the grid its objects snap to and a dense map snaps to a quarter of the beat, so the level");
            TestContext.Out.WriteLine("  a player taps can be an octave or two above what the map says. The column that comes out");
            TestContext.Out.WriteLine("  highest is the level the model reads, and that is the level the labels should be built at.");
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("  A rise near zero, or every share near zero, means the activation does not know where these");
            TestContext.Out.WriteLine("  beats are and labels built from this grid would teach the network the disagreement.");
        }

        private static (string, int, double, double, double, double, double, double, double, double) measure(string map, string audio, string modelPath)
        {
            List<(double Time, double BeatLength)> timing = uninherited(map);

            if (timing.Count == 0)
                throw new InvalidOperationException("no uninherited timing points");

            float[] samples = BassAudioDecoder.DecodeMono(audio, AnalysisAudio.SampleRate);

            // The streaming tracker rather than the whole-file call, because the streaming path is the one the player
            // uses and the activation it assembles is the one a reader in the player would see.
            using var tracker = new StreamingBeatTracker(modelPath);
            tracker.Add(samples);
            tracker.Flush();

            float[] activation = tracker.Activation.ToArray();
            double[] grid = gridFor(timing, activation.Length * 1000.0 / BeatThisBeatTracker.FramesPerSecond);

            double mean = 0;

            foreach (float value in activation)
                mean += value;

            mean /= Math.Max(1, activation.Length);

            // The shift at which the beats and the activation line up best, searched inside a tenth of a second. The
            // activation is taken as a difference from its own mean first, or the search is won by whichever shift
            // happens to sit in the loudest part of the track rather than by the alignment. Wider than a tenth of a
            // second and the answer is a different beat: both carry the beat rate, so their agreement repeats at one
            // beat's displacement and a wide search finds a second peak a beat away that is just as tall.
            //
            // Every reading of "at the beat" below is taken at this shift rather than at zero, and that is not a
            // refinement. A frame is 20ms and the model reads a beat about 40ms after the audio has it - measured as a
            // constant across tempi from 130 to 333 BPM - so a reading taken at zero lands two frames early, which on a
            // 180ms passage is a third of the way to the next beat. The first version of this measured at zero and
            // reported the two fastest beatmaps as disagreeing with their labels when the misalignment was its own.
            int bestLag = 0;
            double best = double.NegativeInfinity;

            for (int lag = -8; lag <= 8; lag++)
            {
                double total = 0;

                foreach (double beat in grid)
                {
                    int frame = (int)Math.Round(beat * BeatThisBeatTracker.FramesPerSecond / 1000.0) + lag;

                    if (frame >= 0 && frame < activation.Length)
                        total += activation[frame] - mean;
                }

                if (total > best)
                {
                    best = total;
                    bestLag = lag;
                }
            }

            var atBeat = new List<double>();
            var offBeat = new List<double>();

            for (int i = 0; i < grid.Length; i++)
            {
                atBeat.Add(valueAtFrame(activation, grid[i], bestLag) - mean);

                if (i + 1 < grid.Length)
                    offBeat.Add(valueAtFrame(activation, (grid[i] + grid[i + 1]) / 2, bestLag) - mean);
            }

            atBeat.Sort();
            offBeat.Sort();

            double atMiddle = atBeat.Count == 0 ? 0 : atBeat[atBeat.Count / 2];
            double offMiddle = offBeat.Count == 0 ? 0 : offBeat[offBeat.Count / 2];

            // The share of the grid's beats that land on a frame the model is positive about, which is the plainest
            // question that can be asked of the pair and the one that decides whether they belong together at all. A
            // label grid built from a map's beats should raise the model almost everywhere: a beat the music has is a
            // beat this model reports. A share well under one means the grid and the model are describing different
            // things, and no amount of training on the pair would fix that.
            int positive = grid.Count(beat => valueAtFrame(activation, beat, bestLag) > 0);
            double share = grid.Length == 0 ? 0 : 100.0 * positive / grid.Length;

            // The beat rate the grid itself runs at, which is what says whether it is the tactus or a subdivision of
            // it. A grid at four to eight beats a second is a snap resolution, not something a player taps.
            double period = timing[0].BeatLength;
            var lengths = new SortedDictionary<int, int>();

            foreach (var point in timing)
            {
                int bucket = (int)(Math.Round(point.BeatLength / 20) * 20);
                lengths[bucket] = lengths.TryGetValue(bucket, out int seen) ? seen + 1 : 1;
            }

            double dominantLength = lengths.OrderByDescending(pair => pair.Value).First().Key;

            // Which octave of the map's own grid the model actually reads. A map's timing points are the grid its
            // objects snap to, and a dense map snaps to a quarter of the beat as readily as to the beat, so the level
            // a player taps can be one or two octaves above what the timing points say. Asking the model instead of
            // guessing: the level it reads is the level whose beats land on the frames it is positive about.
            double until = activation.Length * 1000.0 / BeatThisBeatTracker.FramesPerSecond;
            double atOne = shareAt(activation, gridFor(timing, until, 1));
            double atTwo = shareAt(activation, gridFor(timing, until, 2));
            double atFour = shareAt(activation, gridFor(timing, until, 4));

            return (
                Path.GetFileNameWithoutExtension(map),
                grid.Length,
                dominantLength,
                atMiddle,
                offMiddle,
                bestLag * 1000.0 / BeatThisBeatTracker.FramesPerSecond,
                60000.0 / dominantLength,
                atOne,
                atTwo,
                atFour);
        }

        /// <summary>The activation at a time, as a difference from the track's mean.</summary>
        private static double activationAt(float[] activation, double milliseconds, double mean)
        {
            return valueAtFrame(activation, milliseconds, 0) - mean;
        }

        /// <summary>The activation at a time, or at the frame a shift away from it.</summary>
        private static double valueAtFrame(float[] activation, double milliseconds, int shift)
        {
            int frame = (int)Math.Round(milliseconds * BeatThisBeatTracker.FramesPerSecond / 1000.0) + shift;

            return frame < 0 || frame >= activation.Length ? 0 : activation[frame];
        }

        /// <summary>
        /// A beatmap's uninherited timing points, in time order.
        /// </summary>
        /// <remarks>
        /// Inherited points are skipped rather than read as a tempo: their beat length is negative, they carry the
        /// slider speed, and a parse that treats them as tempo sees hundreds of changes in a track that holds one.
        /// </remarks>
        private static List<(double Time, double BeatLength)> uninherited(string map)
        {
            var points = new List<(double, double)>();

            bool inTiming = false;

            foreach (string raw in File.ReadLines(map))
            {
                string line = raw.Trim();

                if (line.StartsWith("[", StringComparison.Ordinal))
                {
                    inTiming = line.Equals("[TimingPoints]", StringComparison.OrdinalIgnoreCase);
                    continue;
                }

                if (!inTiming || line.Length == 0)
                    continue;

                string[] fields = line.Split(',');

                if (fields.Length < 2)
                    continue;

                if (!double.TryParse(fields[0], NumberStyles.Float, CultureInfo.InvariantCulture, out double time)
                    || !double.TryParse(fields[1], NumberStyles.Float, CultureInfo.InvariantCulture, out double length))
                {
                    continue;
                }

                if (length > 0)
                    points.Add((time, length));
            }

            points.Sort((a, b) => a.Item1.CompareTo(b.Item1));

            return points;
        }

        /// <summary>The beats a map's timing points imply, from its first one to the end of the audio.</summary>
        /// <param name="timing">The uninherited timing points.</param>
        /// <param name="until">The end of the audio, in milliseconds.</param>
        /// <param name="scale">
        /// How much longer to make each beat than the timing point says. One is the map's own grid, two is half its rate
        /// and four a quarter, which is how the grid is moved onto the level a player taps when the map's own is a finer
        /// snap resolution than the music's beat.
        /// </param>
        private static double[] gridFor(List<(double Time, double BeatLength)> timing, double until, double scale = 1)
        {
            var grid = new List<double>();

            for (int i = 0; i < timing.Count; i++)
            {
                double length = timing[i].BeatLength * scale;
                double end = i + 1 < timing.Count ? timing[i + 1].Time : Math.Max(until, timing[i].Time);

                for (double time = timing[i].Time; time < end && grid.Count < 5_000_000; time += length)
                    grid.Add(time);
            }

            grid.Sort();

            return grid.ToArray();
        }

        /// <summary>
        /// The share of a grid's beats that land on a frame the model is positive about.
        /// </summary>
        /// <remarks>
        /// The plainest question that can be asked of the pair, and the one that decides whether they belong together at
        /// all. A grid of beats the music actually has should raise the model almost everywhere, because a beat the
        /// music has is a beat this model reports. A share well under one means the grid and the model are describing
        /// different things, and training on the pair would be teaching the network the disagreement.
        /// </remarks>
        private static double shareAt(float[] activation, double[] grid)
        {
            if (grid.Length == 0)
                return 0;

            int positive = grid.Count(beat => valueAtFrame(activation, beat, 0) > 0);

            return 100.0 * positive / grid.Length;
        }

        private static string Trim(string name) => name.Length <= 46 ? name : name.Substring(0, 45) + "...";

        private static string audioFor(string map, string folder)
        {
            foreach (string line in File.ReadLines(map))
            {
                if (!line.StartsWith("AudioFilename", StringComparison.Ordinal))
                    continue;

                int colon = line.IndexOf(':');

                if (colon < 0)
                    continue;

                string name = line[(colon + 1)..].Trim();

                if (name.Length == 0)
                    continue;

                string candidate = Path.Combine(folder, name);

                if (File.Exists(candidate))
                    return candidate;

                string byName = Directory.EnumerateFiles(folder, "*", SearchOption.TopDirectoryOnly)
                                         .FirstOrDefault(f => string.Equals(Path.GetFileName(f), Path.GetFileName(name), StringComparison.OrdinalIgnoreCase));

                if (byName != null)
                    return byName;
            }

            return null;
        }
    }
}
