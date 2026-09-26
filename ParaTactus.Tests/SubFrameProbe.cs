using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using NUnit.Framework;
using ParaTactus;
using ParaTactus.Decoding;

namespace ParaTactus.Tests
{
    /// <summary>
    /// Measures one beatmap in one inference pass, for the questions that need an answer inside a minute.
    /// </summary>
    /// <remarks>
    /// <see cref="BeatmapCorpusProbe"/> answers how the readings compare across a folder and takes an hour to do it,
    /// because it decodes and runs the model twice per beatmap. A change to the peak pass is a question about one
    /// number - how far the beats it places sit from the map's own - and asking the whole folder for it is the wrong
    /// instrument. This decodes once, runs the model once, and reads the same activation through every peak pass that
    /// is worth comparing:
    ///
    ///   frames      the model's peak frames, no merge and no refinement, which is where the 20ms quantisation is;
    ///   merged      the frames merged into beats at their mean, which is the published postprocessor;
    ///   refined     the merged beats with a single-frame peak read between its frames, which is what ships.
    ///
    /// The point of having all three is that the two changes are separate claims and a single before-and-after cannot
    /// tell which of them earned the improvement. `merged` against `frames` is the fixing of an integer division that
    /// dropped the fraction at every step of a merged group; `refined` against `merged` is the parabola.
    ///
    /// Set <c>OSUTEST_CORPUS</c> to a folder of .osu files with their audio, and optionally <c>OSUTEST_MAP</c> to the
    /// part of one file name, which defaults to the first beatmap found. <c>OSUTEST_MODEL</c> overrides the model.
    /// </remarks>
    [Explicit("Needs OSUTEST_CORPUS pointing at a folder of .osu files with their audio, and the model.")]
    public class SubFrameProbe
    {
        private const string defaultModel = "beat-this-final0-int8";

        [OneTimeSetUp]
        public void SetUp() => AudioTestEnvironment.Initialise();

        [Test]
        public void CompareThePeakPassesOnOneBeatmap()
        {
            string corpus = Environment.GetEnvironmentVariable("OSUTEST_CORPUS");

            if (string.IsNullOrEmpty(corpus) || !Directory.Exists(corpus))
                Assert.Ignore("Set OSUTEST_CORPUS to a folder holding .osu files and their audio.");

            string modelPath = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                               ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", $"{defaultModel}.onnx");

            if (!File.Exists(modelPath))
                Assert.Ignore($"no model at {modelPath}");

            string wanted = Environment.GetEnvironmentVariable("OSUTEST_MAP");
            string map = Directory.EnumerateFiles(corpus, "*.osu", SearchOption.AllDirectories)
                                  .FirstOrDefault(f => wanted == null
                                                       || Path.GetFileName(f).Contains(wanted, StringComparison.OrdinalIgnoreCase));

            if (map == null)
                Assert.Ignore($"no .osu file in {corpus} matching {wanted}");

            string folder = Path.GetDirectoryName(map);
            string audio = folder == null ? null : audioFor(map, folder);

            if (audio == null)
                Assert.Ignore($"no audio beside {map}");

            List<TimingPoint> timing = timingPoints(map);
            Assert.That(timing.Count, Is.GreaterThanOrEqualTo(1), "the map declares no tempo");

            TestContext.Out.WriteLine($"beatmap: {Path.GetFileName(map)}");
            TestContext.Out.WriteLine($"audio:   {Path.GetFileName(audio)}");
            TestContext.Out.WriteLine($"tempo:   {(timing.Count <= 1 ? $"{60000 / timing[0].BeatLength:0} BPM held" : $"{60000 / timing.Max(p => p.BeatLength):0}-{60000 / timing.Min(p => p.BeatLength):0} BPM, {timing.Count} points")}");

            float[] samples = BassAudioDecoder.DecodeMono(audio, AnalysisAudio.SampleRate);
            (float[] activations, _) = BeatThisBeatTracker.Activations(samples, modelPath);

            double[] frames = BeatThisBeatTracker.FramePeaks(activations)
                                                 .Select(f => BeatThisBeatTracker.FrameToMilliseconds(f))
                                                 .ToArray();

            double[] merged = BeatThisBeatTracker.MergedPeaks(activations)
                                                 .Select(f => BeatThisBeatTracker.FrameToMilliseconds(f))
                                                 .ToArray();

            double[] refined = BeatThisBeatTracker.RawPeaks(activations)
                                                  .Select(f => BeatThisBeatTracker.FrameToMilliseconds(f))
                                                  .ToArray();

            double[] search = BeatSequenceSearch.Frames(activations, null)
                                                .Select(f => BeatThisBeatTracker.FrameToMilliseconds(f))
                                                .ToArray();

            double until = new[] { frames, merged, refined, search }.Max(set => set.Length > 0 ? set[^1] : 0);
            double[] grid = beatmapGrid(timing, until);

            TestContext.Out.WriteLine($"the map has {grid.Length} beats up to {until:0}ms");
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("  reading                                              count   rate   median      p90  covered   settled  scatter");
            report("the model's peak frames, as frames", frames, grid);
            report("the frames merged into beats at their mean", merged, grid);
            report("merged, then a lone peak read between frames", refined, grid);
            report("the tempo search over the activation", search, grid);

            // How many peaks came from more than one frame. If the model reports a beat as one sharp frame every time,
            // the merge step is doing nothing at all and the two readings above are the same array twice, which is
            // worth knowing rather than inferring from two identical columns.
            var widths = new SortedDictionary<int, int>();

            foreach (int width in BeatThisBeatTracker.MergedWidths(activations))
                widths[width] = widths.TryGetValue(width, out int seen) ? seen + 1 : 1;

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("how many frames each merged peak was made of:");
            TestContext.Out.WriteLine("  " + string.Join("  ", widths.Select(pair => $"{pair.Key} frame{(pair.Key == 1 ? "" : "s")}: {pair.Value}")));
            // The interval a beat map's own beats sit at, which is the scale everything above has to be read against. A
            // reading that puts beats at half this is not more precise, it is at the wrong metrical level.
            double mapInterval = grid.Length > 1 ? grid[1] - grid[0] : 0;

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine($"the map's own beat is {mapInterval:0}ms, so a pulse twice as often would be {mapInterval / 2:0}ms");

            // Which of the two failures it is, which decides what can be done about it. A signed offset that sits at
            // roughly one value all track long is a phase error, and a phase error is a constant that can be measured
            // and removed. An offset that walks in one direction is a tempo error, and a tempo error cannot be removed
            // by moving the beats - only by finding the tempo the beats are actually at. A offset that jumps around is
            // the peaks landing on the wrong events, which is neither.
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("the signed offset of the closest beat in each minute, in the model's own reading:");
            TestContext.Out.WriteLine("  minute        beats   median offset      spread");

            double[] reading = refined;

            for (int minute = 0; minute * 60_000 <= until; minute++)
            {
                double from = minute * 60_000;
                double to = from + 60_000;
                double[] signed = reading.Where(b => b >= from && b < to)
                                         .Select(b => signedDistance(grid, b))
                                         .OrderBy(v => v)
                                         .ToArray();

                if (signed.Length < 4)
                    continue;

                double middle = signed[signed.Length / 2];
                double spread = signed[Math.Min(signed.Length - 1, (int)(signed.Length * 0.75))]
                                - signed[Math.Min(signed.Length - 1, (int)(signed.Length * 0.25))];

                TestContext.Out.WriteLine($"  {minute,3}:00  {signed.Length,10}  {middle,12:+0;-0;0}ms  {spread,9:0}ms");
            }
        }

        /// <summary>
        /// Writes a window of a track with the map's beat and the tracker's beat marked by different clicks.
        /// </summary>
        /// <remarks>
        /// The point of this is to hear the answer instead of deriving it, and to hear it against the audio the
        /// tracker actually saw. Every number in this suite is a distance from the beatmap's own grid, which assumes
        /// the map is right about where the music's beats are - true of a well-made map and not something this code
        /// can check. If the tracker's clicks and the map's clicks land on top of each other and both land on the
        /// music, the reading is right; if the model's clicks sound early against the music, the error is in the
        /// tracker, and the map's clicks are what say so.
        ///
        /// The two are separable by ear: the map's beat is a click with a rising edge and the tracker's is a tick
        /// with the phase flipped, so they are opposite in a waveform view as well as different in timbre. A pair that
        /// arrives together is a beat both readings agree on; a tick on its own is a beat only the tracker believes
        /// in, and a click on its own is one it missed.
        ///
        /// Set <c>OSUTEST_CORPUS</c>, optionally <c>OSUTEST_MAP</c>, and:
        ///   <c>OSUTEST_FROM</c>     milliseconds into the track to start, default 60000
        ///   <c>OSUTEST_LENGTH</c>   how much to write, default 20000
        ///   <c>OSUTEST_OUT</c>      where to write the .wav, default beside the corpus
        ///   <c>OSUTEST_RATE</c>     the rate to write at, default 44100
        /// </remarks>
        [Test]
        public void WriteAWindowWithBothPulsesAudible()
        {
            string corpus = Environment.GetEnvironmentVariable("OSUTEST_CORPUS");

            if (string.IsNullOrEmpty(corpus) || !Directory.Exists(corpus))
                Assert.Ignore("Set OSUTEST_CORPUS to a folder holding .osu files and their audio.");

            string modelPath = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                               ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", $"{defaultModel}.onnx");

            if (!File.Exists(modelPath))
                Assert.Ignore($"no model at {modelPath}");

            string wanted = Environment.GetEnvironmentVariable("OSUTEST_MAP");
            string map = Directory.EnumerateFiles(corpus, "*.osu", SearchOption.AllDirectories)
                                  .FirstOrDefault(f => wanted == null
                                                       || Path.GetFileName(f).Contains(wanted, StringComparison.OrdinalIgnoreCase));

            if (map == null)
                Assert.Ignore($"no .osu file in {corpus} matching {wanted}");

            string folder = Path.GetDirectoryName(map);
            string audio = folder == null ? null : audioFor(map, folder);

            if (audio == null)
                Assert.Ignore($"no audio beside {map}");

            double from = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_FROM"), NumberStyles.Float, CultureInfo.InvariantCulture, out double parsedFrom) ? parsedFrom : 60_000;
            double length = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_LENGTH"), NumberStyles.Float, CultureInfo.InvariantCulture, out double parsedLength) ? parsedLength : 20_000;
            string output = Environment.GetEnvironmentVariable("OSUTEST_OUT")
                            ?? Path.Combine(Path.GetTempPath(), $"{Path.GetFileNameWithoutExtension(map)}-{from:0}-{length:0}.wav");

            // The analysis is done at the model's own rate, because that is what decides where the tracker puts a beat
            // and the point here is to hear the tracker's answer. The window is written at a rate the ear can use:
            // 22050Hz stops at 11kHz, which is where cymbals and the attack of a snare live, and those are exactly the
            // sounds a beat is judged against.
            float[] samples = BassAudioDecoder.DecodeMono(audio, AnalysisAudio.SampleRate);
            (float[] activations, _) = BeatThisBeatTracker.Activations(samples, modelPath);

            // The analysis samples are not needed again, and the window can be tens of megabytes at 44.1kHz.
            samples = Array.Empty<float>();

            double[] tracked = BeatThisBeatTracker.RawPeaks(activations)
                                                  .Select(f => BeatThisBeatTracker.FrameToMilliseconds(f))
                                                  .ToArray();

            double[] grid = beatmapGrid(timingPoints(map), tracked.Length > 0 ? tracked[^1] : 0);

            int rate = int.TryParse(Environment.GetEnvironmentVariable("OSUTEST_RATE"), out int parsedRate) && parsedRate > 0
                ? parsedRate
                : 44_100;

            TestContext.Out.WriteLine($"decoding the window at {rate}Hz for listening, and analysing at {AnalysisAudio.SampleRate}Hz");

            float[] full = BassAudioDecoder.DecodeMono(audio, rate);

            int first = (int)Math.Max(0, Math.Round(from * rate / 1000.0));
            int count = (int)Math.Min(full.Length - first, Math.Round(length * rate / 1000.0));

            if (count <= 0)
                Assert.Ignore($"{from}ms is past the end of the track, which is {full.Length * 1000.0 / rate:0}ms");

            var window = new float[count];
            Array.Copy(full, first, window, 0, count);

            full = Array.Empty<float>();

            int mapClicks = mark(window, grid, from, rate, 0.22, rising: true);
            int modelClicks = mark(window, tracked, from, rate, 0.22, rising: false);

            writeWav(output, window, rate);

            TestContext.Out.WriteLine($"wrote {output}");
            TestContext.Out.WriteLine($"  {length / 1000:0}s from {from / 1000:0}s of {Path.GetFileName(audio)}");
            TestContext.Out.WriteLine($"  the map's beat is a click, {mapClicks} of them in the window");
            TestContext.Out.WriteLine($"  the tracker's beat is a tick, {modelClicks} of them in the window");
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("listen for a tick that arrives on its own - a beat only the tracker believes in -");
            TestContext.Out.WriteLine("and for a click that arrives on its own, which is a beat it missed. A tick that");
            TestContext.Out.WriteLine("sounds early or late against the music is the placement error this suite has been");
            TestContext.Out.WriteLine("measuring as a median distance.");
        }

        /// <summary>Puts a short click at each of a set of times on top of some audio, in place.</summary>
        private static int mark(float[] window, double[] times, double from, int rate, double gain, bool rising)
        {
            int marked = 0;

            foreach (double time in times)
            {
                double at = (time - from) * rate / 1000.0;
                int start = (int)Math.Round(at);

                if (start < 0 || start >= window.Length)
                    continue;

                // Twenty milliseconds is short enough to read as a click and long enough to have a pitch, and a
                // Hann-shaped body keeps it from splattering across the spectrum the way a bare step does.
                int span = (int)(0.020 * rate);

                for (int i = 0; i < span && start + i < window.Length; i++)
                {
                    double envelope = 0.5 * (1 - Math.Cos(2 * Math.PI * i / span));
                    double tone = Math.Sin(2 * Math.PI * 1800 * i / rate);
                    double value = gain * envelope * tone * (rising ? 1 : -1);

                    window[start + i] = (float)Math.Clamp(window[start + i] + value, -1, 1);
                }

                marked++;
            }

            return marked;
        }

        /// <summary>Writes mono float samples as a 16-bit PCM wav.</summary>
        /// <remarks>
        /// Hand-written because there is no wav writer in the engine and this is the only thing that needs one. The
        /// header is the forty-four bytes every player expects; nothing here is ever read back by this code.
        /// </remarks>
        private static void writeWav(string path, float[] samples, int rate)
        {
            using var file = File.Create(path);
            using var writer = new BinaryWriter(file);

            int dataBytes = samples.Length * 2;

            writer.Write("RIFF"u8.ToArray());
            writer.Write(36 + dataBytes);
            writer.Write("WAVE"u8.ToArray());

            writer.Write("fmt "u8.ToArray());
            writer.Write(16);
            writer.Write((short)1);
            writer.Write((short)1);
            writer.Write(rate);
            writer.Write(rate * 2);
            writer.Write((short)2);
            writer.Write((short)16);

            writer.Write("data"u8.ToArray());
            writer.Write(dataBytes);

            foreach (float sample in samples)
                writer.Write((short)Math.Clamp(sample * short.MaxValue, short.MinValue, short.MaxValue));
        }

        private static void report(string name, double[] beats, double[] grid)
        {
            double[] errors = beats.Select(b => Math.Abs(signedDistance(grid, b))).OrderBy(e => e).ToArray();
            double middle = medianInterval(beats);
            double median = errors.Length == 0 ? 0 : errors[errors.Length / 2];
            double p90 = errors.Length == 0 ? 0 : errors[Math.Min(errors.Length - 1, (int)(errors.Length * 0.9))];
            int covered = grid.Count(g => beats.Any(b => Math.Abs(b - g) <= 60));

            // The signed offsets, which is a different question from the distances above. Their middle is how far the
            // whole reading sits from the map, and it is invisible on screen: the nearest beat to a beat is the same
            // beat whether every reading is thirty milliseconds late or the map is. Their scatter about that middle is
            // what is left once the constant is taken out, and it is the part a player can actually see and hear.
            double[] signed = beats.Select(b => signedDistance(grid, b)).OrderBy(v => v).ToArray();
            double settled = signed.Length == 0 ? 0 : signed[signed.Length / 2];
            double low = signed.Length == 0 ? 0 : signed[Math.Min(signed.Length - 1, (int)(signed.Length * 0.25))];
            double high = signed.Length == 0 ? 0 : signed[Math.Min(signed.Length - 1, (int)(signed.Length * 0.75))];

            TestContext.Out.WriteLine($"  {name,-48} {beats.Length,6} {(middle > 0 ? $"{1000 / middle,6:0.0}" : "   n/a")} {median,7:0.0}ms {p90,7:0.0}ms {100.0 * covered / Math.Max(1, grid.Length),7:0.0}% {settled,8:+0;-0;0}ms {high - low,8:0}ms");
        }

        private static double medianInterval(double[] beats)
        {
            if (beats.Length < 3)
                return 0;

            double[] gaps = beats.Zip(beats.Skip(1), (a, b) => b - a).OrderBy(g => g).ToArray();

            return gaps[gaps.Length / 2];
        }

        private static double signedDistance(double[] grid, double time)
        {
            int index = Array.BinarySearch(grid, time);

            if (index >= 0)
                return 0;

            index = ~index;

            if (index == 0)
                return time - grid[0];

            if (index >= grid.Length)
                return time - grid[^1];

            double before = time - grid[index - 1];
            double after = time - grid[index];

            return Math.Abs(before) <= Math.Abs(after) ? before : after;
        }

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

        /// <summary>
        /// The tempo points a map declares, in time order.
        /// </summary>
        /// <remarks>
        /// Only positive beat lengths are tempo. A negative one is an inherited point, which carries volume and sample
        /// changes and says nothing about the tempo; reading those as tempo is what makes a naive pass over a map see
        /// hundreds of changes in a track that holds one tempo.
        ///
        /// Section headers matter for the same reason. An object line under <c>[HitObjects]</c> is
        /// <c>x,y,time,type,...</c> and parses as a timing point with a time and a beat length without complaining, so
        /// a parse that does not stop at the next section header turns every hit object into a tempo change. This is
        /// the same reading <see cref="BeatmapCorpusProbe"/> uses, and the two are deliberately separate copies of it:
        /// it is short, and a shared helper would have to be public on a library that has no reason to know what an
        /// .osu file is.
        /// </remarks>
        private static List<TimingPoint> timingPoints(string map)
        {
            var points = new List<TimingPoint>();

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

                if (length <= 0)
                    continue;

                points.Add(new TimingPoint(time, length));
            }

            return points.OrderBy(p => p.Time).ToList();
        }

        /// <summary>A map's own beats: each timing point's section stepped at its beat length, to the end.</summary>
        private static double[] beatmapGrid(List<TimingPoint> timing, double until)
        {
            var grid = new List<double>();

            for (int i = 0; i < timing.Count; i++)
            {
                // A section runs until the next timing point, and the last one runs to the end of the track.
                double end = i + 1 < timing.Count ? timing[i + 1].Time : Math.Max(until, timing[i].Time);

                for (double time = timing[i].Time; time < end && grid.Count < 500_000; time += timing[i].BeatLength)
                    grid.Add(time);
            }

            grid.Sort();

            return grid.ToArray();
        }

        private readonly struct TimingPoint
        {
            public TimingPoint(double time, double beatLength)
            {
                Time = time;
                BeatLength = beatLength;
            }

            public double Time { get; }

            public double BeatLength { get; }
        }
    }
}
