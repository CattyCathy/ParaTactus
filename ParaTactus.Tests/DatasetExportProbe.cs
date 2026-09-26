using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using NUnit.Framework;
using ParaTactus;
using ParaTactus.Decoding;

namespace ParaTactus.Tests
{
    /// <summary>
    /// Turns a corpus of osu! beatmapsets into audio and labels a detector can be trained on.
    /// </summary>
    /// <remarks>
    /// What is written is the decoded mono audio and a per-frame label, and deliberately not any features. Features are
    /// the part most likely to be changed - a log-mel, an onset envelope, both, at what resolution - and writing them
    /// here would mean re-running this exporter for every experiment. The audio is the part that costs, because
    /// decoding a corpus takes minutes, so it is written once and read many times.
    ///
    /// The labels come from a beatmap's uninherited timing points, which are the grid the mapper snapped the objects
    /// to and therefore the author's own statement of where the beats are. Inherited points are the slider and scroll
    /// speed and say nothing about the beat. Hit objects are not used for labels because a dense map places them on
    /// subdivisions: on this corpus the commonest gap between consecutive objects is 80ms on maps whose beat is 300.
    ///
    /// One thing the timing points cannot say is which octave of themselves is the beat. A map's grid is a snap
    /// resolution and a dense map snaps to a quarter of the beat, so the labels are built at the octave the model
    /// itself reads, decided per window by which octave's beats land on the activation. That is recorded per frame as
    /// well, because it is the thing a detector is being trained to get right and a corpus where it varies is the
    /// corpus worth having.
    ///
    /// Set <c>OSUTEST_CORPUS</c> to a folder of .osz files, and <c>OSUTEST_DATASET</c> for where to write.
    /// </remarks>
    [Explicit("Needs OSUTEST_CORPUS pointing at a folder of .osz files, and the model.")]
    public class DatasetExportProbe
    {
        private const string defaultModel = "beat-this-final0-int8";

        /// <summary>How much audio one example covers, in seconds.</summary>
        private const double window_seconds = 8;

        [OneTimeSetUp]
        public void SetUp() => AudioTestEnvironment.Initialise();

        [Test]
        public void ExportAudioAndLabelsForTraining()
        {
            string corpus = Environment.GetEnvironmentVariable("OSUTEST_CORPUS");
            string output = Environment.GetEnvironmentVariable("OSUTEST_DATASET") ?? @"D:\Linux\Proj\OsuTest\dataset";

            if (string.IsNullOrEmpty(corpus) || !Directory.Exists(corpus))
                Assert.Ignore("Set OSUTEST_CORPUS to a folder of .osz files.");

            string modelPath = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                               ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", $"{defaultModel}.onnx");

            if (!File.Exists(modelPath))
                Assert.Ignore($"no model at {modelPath}");

            var sets = Directory.EnumerateFiles(corpus, "*.osz", SearchOption.AllDirectories).OrderBy(f => f).ToList();

            // A shard, so several exports can run at once. The work here is almost entirely the model's inference - 363
            // of 370 seconds for one four-minute track, measured - and that is already using every core the runtime
            // will give it, so the only way to go faster is to run several tracks at a time. This process's share is
            // chosen by taking the sets at a stride rather than a contiguous block, so the shards get the same mix of
            // long and short tracks.
            string shard = Environment.GetEnvironmentVariable("OSUTEST_SHARD");

            if (!string.IsNullOrEmpty(shard) && shard.Contains('/'))
            {
                string[] parts = shard.Split('/');
                int index = int.Parse(parts[0], CultureInfo.InvariantCulture);
                int of = int.Parse(parts[1], CultureInfo.InvariantCulture);

                sets = sets.Where((_, i) => i % of == index).ToList();

                TestContext.Out.WriteLine($"shard {index} of {of}");
            }

            if (sets.Count == 0)
                Assert.Ignore($"no .osz file in {corpus}");

            Directory.CreateDirectory(output);

            string audioDirectory = Path.Combine(output, "audio");
            string labelDirectory = Path.Combine(output, "labels");

            Directory.CreateDirectory(audioDirectory);
            Directory.CreateDirectory(labelDirectory);

            TestContext.Out.WriteLine($"{sets.Count} beatmapsets in {corpus}");
            TestContext.Out.WriteLine($"writing to {output}");
            TestContext.Out.WriteLine("");

            var manifest = new StringBuilder();
            manifest.AppendLine("id\tset\taudio\tlabels\tframes\tbeats\tperiod\tlag\toctave\tusable");

            int exported = 0;
            int skipped = 0;
            var failures = new List<string>();

            foreach (string set in sets)
            {
                string id = Path.GetFileNameWithoutExtension(set).Split(' ')[0];
                string audioPath = Path.Combine(audioDirectory, id + ".f32");
                string labelPath = Path.Combine(labelDirectory, id + ".i8");

                if (File.Exists(audioPath) && File.Exists(labelPath))
                {
                    skipped++;
                    continue;
                }

                try
                {
                    Row row = export(set, id, audioPath, labelPath, modelPath);
                    manifest.AppendLine(row.ToLine(id, audioPath, labelPath));
                    exported++;

                    TestContext.Out.WriteLine($"  [{exported,4}] {Path.GetFileName(set),-52} {row.Frames,7} frames  {row.Beats,6} beats  period {row.Period,4:0}ms  {(row.Usable ? "usable" : "not usable")}");
                }
                catch (Exception error)
                {
                    failures.Add($"{Path.GetFileName(set)}: {error.Message}");

                    // Half-written pairs are removed, or a resumed run would treat a truncated label file as done.
                    foreach (string path in new[] { audioPath, labelPath })
                    {
                        if (File.Exists(path))
                            File.Delete(path);
                    }
                }
            }

            File.WriteAllText(Path.Combine(output, "manifest.tsv"), manifest.ToString());

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine($"exported {exported}, already present {skipped}, failed {failures.Count}");

            foreach (string failure in failures.Take(20))
                TestContext.Out.WriteLine($"  {failure}");

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine($"manifest: {Path.Combine(output, "manifest.tsv")}");
        }

        private readonly struct Row
        {
            public Row(int frames, int beats, double period, double lag, int octave, bool usable)
            {
                Frames = frames;
                Beats = beats;
                Period = period;
                Lag = lag;
                Octave = octave;
                Usable = usable;
            }

            public int Frames { get; }
            public int Beats { get; }
            public double Period { get; }
            public double Lag { get; }
            public int Octave { get; }
            public bool Usable { get; }

            public string ToLine(string id, string audioPath, string labelPath)
                => string.Join('\t', id, id, Path.GetFileName(audioPath), Path.GetFileName(labelPath),
                               Frames.ToString(CultureInfo.InvariantCulture),
                               Beats.ToString(CultureInfo.InvariantCulture),
                               Period.ToString("0", CultureInfo.InvariantCulture),
                               Lag.ToString("0", CultureInfo.InvariantCulture),
                               Octave.ToString(CultureInfo.InvariantCulture),
                               Usable ? "1" : "0");
        }

        private static Row export(string set, string id, string audioPath, string labelPath, string modelPath)
        {
            string scratch = Path.Combine(Path.GetTempPath(), "paratactus-set-" + id);
            var clock = System.Diagnostics.Stopwatch.StartNew();
            double unzip = 0, decode = 0, infer = 0, label = 0;

            try
            {
                if (Directory.Exists(scratch))
                    Directory.Delete(scratch, true);

                Directory.CreateDirectory(scratch);
                ZipFile.ExtractToDirectory(set, scratch);
                unzip = clock.Elapsed.TotalSeconds;

                string map = Directory.EnumerateFiles(scratch, "*.osu", SearchOption.AllDirectories).FirstOrDefault()
                             ?? throw new InvalidOperationException("no .osu in the set");

                string audio = audioFor(map, Path.GetDirectoryName(map))
                               ?? throw new InvalidOperationException("no audio in the set");

                List<(double Time, double BeatLength)> timing = uninherited(map);

                if (timing.Count == 0)
                    throw new InvalidOperationException("no uninherited timing points");

                float[] samples = BassAudioDecoder.DecodeMono(audio, AnalysisAudio.SampleRate);
                decode = clock.Elapsed.TotalSeconds;

                if (samples.Length < AnalysisAudio.SampleRate * 10)
                    throw new InvalidOperationException("under ten seconds of audio");

                // The model's own activation is read for the octave decision and for the lag, so it is run here rather
                // than left to the training side. It is the one part of the export that needs the model, and it is what
                // makes the labels unambiguous.
                //
                // Two threads, and that is a measurement rather than a guess. This model gets slower the more threads it
                // is given: on a four-minute track the same analysis takes 24.5 seconds on two threads, 55.5 on eight
                // and 113.5 on sixteen, so the operators' synchronisation costs more than the parallelism returns. The
                // playing path asks for every processor but two, which is right when something is waiting on it and
                // four to five times wrong here.
                int threads = int.TryParse(Environment.GetEnvironmentVariable("OSUTEST_THREADS"), out int wanted) && wanted > 0
                    ? wanted
                    : 2;

                using var tracker = new StreamingBeatTracker(modelPath, threads);
                tracker.Add(samples);
                tracker.Flush();

                float[] activation = tracker.Activation.ToArray();
                infer = clock.Elapsed.TotalSeconds;
                int frames = activation.Length;

                if (frames < 100)
                    throw new InvalidOperationException("too few frames");

                double until = frames * 1000.0 / BeatThisBeatTracker.FramesPerSecond;

                double period = dominantBeatLength(timing);

                // The lag between this map's grid and the model, measured once for the whole track. It is reported
                // rather than applied: a map's timing points are placed by hand against the audio and are routinely
                // tens of milliseconds from where the music is, so a shift here is a fact about the map as much as
                // about the model, and the training side is the place to decide what to do with it.
                int lag = bestLag(activation, gridFor(timing, until, 1));

                var labels = new sbyte[frames];
                var octaves = new sbyte[frames];

                int beats = 0;
                int usableWindows = 0;
                int windows = 0;

                // Per window, because a map's grid can change level part way through and because the decision needs
                // context: one window's worth of beats is what says whether they are the beat or half of it.
                int windowFrames = (int)Math.Round(window_seconds * BeatThisBeatTracker.FramesPerSecond);
                int stride = windowFrames / 2;

                for (int start = 0; start < frames; start += stride)
                {
                    int end = Math.Min(frames, start + windowFrames);
                    double from = start * 1000.0 / BeatThisBeatTracker.FramesPerSecond;
                    double to = end * 1000.0 / BeatThisBeatTracker.FramesPerSecond;

                    (int octave, double rise) = chooseOctave(activation, timing, from, to, lag);

                    windows++;

                    if (rise > 2)
                        usableWindows++;

                    // Only the usable windows contribute labels. A window where the model does not separate a beat from
                    // the space between two of them has nothing to teach: training on it would be training the network
                    // to reproduce the model's uncertainty, and the octave chosen there is a coin toss.
                    if (rise <= 2)
                        continue;

                    foreach (double beat in gridFor(timing, to, Math.Pow(2, octave)).Where(b => b >= from && b < to))
                    {
                        int frame = (int)Math.Round(beat * BeatThisBeatTracker.FramesPerSecond / 1000.0);

                        if (frame >= start && frame < end && labels[frame] == 0)
                        {
                            labels[frame] = 1;
                            octaves[frame] = (sbyte)octave;
                            beats++;
                        }
                    }
                }

                label = clock.Elapsed.TotalSeconds;

                WriteAudio(audioPath, samples);
                File.WriteAllBytes(labelPath, labels.Select(v => (byte)v).ToArray());
                File.WriteAllBytes(Path.ChangeExtension(labelPath, ".oct"), octaves.Select(v => (byte)v).ToArray());

                TestContext.Out.WriteLine($"        timing: unzip {unzip:0.0}s  decode {decode - unzip:0.0}s  infer {infer - decode:0.0}s  label {label - infer:0.0}s  write {clock.Elapsed.TotalSeconds - label:0.0}s");

                return new Row(frames, beats, period, lag * 1000.0 / BeatThisBeatTracker.FramesPerSecond, 0, usableWindows > 0);
            }
            finally
            {
                if (Directory.Exists(scratch))
                    Directory.Delete(scratch, true);
            }
        }

        /// <summary>The octave of a map's grid whose beats land most on the model's activation in a window.</summary>
        /// <remarks>
        /// A map's timing points are a snap resolution as much as a tempo, and a dense map snaps to a quarter of the
        /// beat, so the level a player taps can be an octave or two above what the map says. The model was trained on
        /// the level a person taps, so the octave it reads is the one the labels belong at, and asking it is more
        /// reliable than guessing from the beat length.
        ///
        /// The measure is the difference between the activation at the grid's beats and at the points halfway between
        /// them, both as a difference from the track's own mean. The difference rather than the level, because a grid
        /// an octave too fine puts beats where the music has none and that shows up as a small difference rather than
        /// as a small value.
        /// </remarks>
        private static (int Octave, double Rise) chooseOctave(float[] activation, List<(double Time, double BeatLength)> timing, double from, double to, int lag)
        {
            double mean = 0;

            foreach (float value in activation)
                mean += value;

            mean /= Math.Max(1, activation.Length);

            int best = 0;
            double bestRise = double.NegativeInfinity;

            for (int octave = -2; octave <= 2; octave++)
            {
                double[] grid = gridFor(timing, to, Math.Pow(2, octave));
                var at = new List<double>();
                var off = new List<double>();

                for (int i = 0; i < grid.Length; i++)
                {
                    if (grid[i] < from || grid[i] >= to)
                        continue;

                    at.Add(activationAt(activation, grid[i], lag) - mean);

                    if (i + 1 < grid.Length)
                        off.Add(activationAt(activation, (grid[i] + grid[i + 1]) / 2, lag) - mean);
                }

                if (at.Count < 4)
                    continue;

                at.Sort();
                off.Sort();

                double rise = at[at.Count / 2] - (off.Count == 0 ? 0 : off[off.Count / 2]);

                if (rise > bestRise)
                {
                    bestRise = rise;
                    best = octave;
                }
            }

            return (best, bestRise);
        }

        private static double activationAt(float[] activation, double milliseconds, int shift)
        {
            int frame = (int)Math.Round(milliseconds * BeatThisBeatTracker.FramesPerSecond / 1000.0) + shift;

            return frame < 0 || frame >= activation.Length ? 0 : activation[frame];
        }

        /// <summary>The shift at which a grid and the activation line up best, within a tenth of a second.</summary>
        private static int bestLag(float[] activation, double[] grid)
        {
            double mean = 0;

            foreach (float value in activation)
                mean += value;

            mean /= Math.Max(1, activation.Length);

            int best = 0;
            double bestTotal = double.NegativeInfinity;

            for (int lag = -8; lag <= 8; lag++)
            {
                double total = 0;

                foreach (double beat in grid)
                    total += activationAt(activation, beat, lag) - mean;

                if (total > bestTotal)
                {
                    bestTotal = total;
                    best = lag;
                }
            }

            return best;
        }

        /// <summary>The beat length a map's timing points spend most of their length at.</summary>
        private static double dominantBeatLength(List<(double Time, double BeatLength)> timing)
        {
            var weight = new Dictionary<double, double>();

            for (int i = 0; i < timing.Count; i++)
            {
                double end = i + 1 < timing.Count ? timing[i + 1].Time : timing[i].Time;
                double length = Math.Round(timing[i].BeatLength);

                weight[length] = weight.TryGetValue(length, out double seen) ? seen + (end - timing[i].Time) : end - timing[i].Time;
            }

            return weight.OrderByDescending(pair => pair.Value).First().Key;
        }

        private static double[] gridFor(List<(double Time, double BeatLength)> timing, double until, double scale)
        {
            var grid = new List<double>();

            for (int i = 0; i < timing.Count; i++)
            {
                double length = timing[i].BeatLength * scale;

                if (length <= 5)
                    continue;

                double end = i + 1 < timing.Count ? timing[i + 1].Time : Math.Max(until, timing[i].Time);

                for (double time = timing[i].Time; time < end && grid.Count < 5_000_000; time += length)
                    grid.Add(time);
            }

            grid.Sort();

            return grid.ToArray();
        }

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

                if (fields.Length >= 2
                    && double.TryParse(fields[0], NumberStyles.Float, CultureInfo.InvariantCulture, out double time)
                    && double.TryParse(fields[1], NumberStyles.Float, CultureInfo.InvariantCulture, out double length)
                    && length > 0)
                {
                    points.Add((time, length));
                }
            }

            points.Sort((a, b) => a.Item1.CompareTo(b.Item1));

            return points;
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

        /// <summary>Writes mono float samples as raw little-endian float32, which is what a numpy reader wants.</summary>
        private static void WriteAudio(string path, float[] samples)
        {
            using var file = File.Create(path);
            using var writer = new BinaryWriter(file);
            var buffer = new byte[samples.Length * 4];

            Buffer.BlockCopy(samples, 0, buffer, 0, buffer.Length);
            writer.Write(buffer);
        }
    }
}
