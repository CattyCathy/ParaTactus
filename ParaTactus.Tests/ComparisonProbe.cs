using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using NUnit.Framework;
using ParaTactus;
using ParaTactus.Decoding;

namespace ParaTactus.Tests
{
    /// <summary>
    /// Measures the existing beat pipeline and the trained detector on the same tracks, the same way.
    /// </summary>
    /// <remarks>
    /// The numbers reported for the two have been measured on different corpora until now: the peak picking on eight
    /// beatmaps collected early, the detector on the twelve-hundred-track corpus built for it. That is not a comparison,
    /// because the material differs and so did the harness - one measured against the analysis library's own reader and
    /// the other against a Python script. This runs both over the same audio, with the same reference and the same
    /// arithmetic, in one process, which is the only way the difference can be attributed to the readers.
    ///
    /// The existing pipeline is run exactly as the player runs it: PeakFrames over the activation assembled by the
    /// streaming tracker, through BeatTrainRegulariser, then BeatGrid.FromBeats. Not an approximation of it - the same
    /// calls, so a change to any of them shows up here.
    ///
    /// Both are judged against a beatmap's uninherited timing points. That reference is a snap grid rather than a
    /// statement of the musical beat, which is a caveat on every number in this project and is stated here rather than
    /// repeated: what "the right metrical level" means throughout is the level the map is written at.
    ///
    /// Set <c>OSUTEST_DATASET</c>, <c>OSUTEST_CORPUS</c>, <c>OSUTEST_DETECTOR</c> and optionally
    /// <c>OSUTEST_ACTIVATION</c> for the Beat This! model.
    /// </remarks>
    [Explicit("Needs OSUTEST_DATASET, OSUTEST_CORPUS and OSUTEST_DETECTOR.")]
    public class ComparisonProbe
    {
        [OneTimeSetUp]
        public void SetUp() => AudioTestEnvironment.Initialise();

        [Test]
        public void CompareBothReadersOnTheSameTracks()
        {
            string dataset = Environment.GetEnvironmentVariable("OSUTEST_DATASET");
            string corpus = Environment.GetEnvironmentVariable("OSUTEST_CORPUS");
            string detectorPath = Environment.GetEnvironmentVariable("OSUTEST_DETECTOR");

            if (string.IsNullOrEmpty(dataset) || !Directory.Exists(dataset))
                Assert.Ignore("Set OSUTEST_DATASET to the exported dataset.");

            if (string.IsNullOrEmpty(corpus) || !Directory.Exists(corpus))
                Assert.Ignore("Set OSUTEST_CORPUS to the .osz files.");

            if (string.IsNullOrEmpty(detectorPath) || !File.Exists(detectorPath))
                Assert.Ignore("Set OSUTEST_DETECTOR to the exported model.");

            string activationPath = Environment.GetEnvironmentVariable("OSUTEST_ACTIVATION")
                                    ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                                                    "Downloads", "beat-this-final0-int8.onnx");

            int limit = int.TryParse(Environment.GetEnvironmentVariable("OSUTEST_LIMIT"), out int wanted) ? wanted : 12;

            // The comparison is run twice - once with the four slowest maps and once with everything - because the
            // objective is specifically about tempo-changing and fast material, and a median over a corpus that is
            // mostly one tempo cannot answer a question about the rest of it.
            var archives = new Dictionary<string, string>();

            foreach (string file in Directory.EnumerateFiles(corpus, "*.osz", SearchOption.AllDirectories))
                archives[Path.GetFileName(file).Split(' ')[0]] = file;

            var rows = new List<Row>();

            using var detector = new LearnedBeatDetector(detectorPath);

            foreach (string line in File.ReadLines(Path.Combine(dataset, "manifest.tsv")).Skip(1))
            {
                if (rows.Count >= limit)
                    break;

                string[] fields = line.Split('\t');

                if (fields.Length < 7 || !archives.TryGetValue(fields[0], out string archive))
                    continue;

                string audioPath = Path.Combine(dataset, "audio", fields[2]);

                if (!File.Exists(audioPath))
                    continue;

                float[] samples = ReadAudio(audioPath);
                double seconds = samples.Length / (double)LogMel.SampleRate;
                double[] grid = MapBeats(archive, seconds * 1000);

                if (grid.Length < 16)
                    continue;

                double[] detected = detector.Beats(samples);

                // The existing pipeline, stage by stage, because the summary hides which stage does the damage. The
                // stages are: the peaks themselves, those beats spaced onto a metre, and the grid the player is handed -
                // which is the second of those put through a metrical level decision for the whole track.
                double[] tracked = ExistingPipeline(samples, activationPath);
                double[] regularised = tracked.Length == 0 ? Array.Empty<double>() : BeatTrainRegulariser.Regularise(tracked);
                BeatGrid playerGrid = regularised.Length == 0 ? null : BeatGrid.FromBeats(regularised);

                double[] gridBeats = playerGrid == null ? Array.Empty<double>() : playerGrid.Beats.ToArray();

                rows.Add(new Row(
                    fields[0],
                    double.TryParse(fields[6], NumberStyles.Float, CultureInfo.InvariantCulture, out double period) ? period : 0,
                    playerGrid?.MetricalShift ?? 0,
                    Measure(tracked, grid, seconds),
                    Measure(regularised, grid, seconds),
                    Measure(gridBeats, grid, seconds),
                    Measure(detected, grid, seconds)));
            }

            if (rows.Count == 0)
                Assert.Fail("nothing could be measured");

            Report(rows);

            // The maps whose own beat is fastest, which is the material the objective names and the material a median
            // over the corpus hides.
            var fastest = rows.OrderBy(r => r.MapPeriod).Take(4).ToList();

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("the four fastest beatmaps, on their own:");

            Report(fastest);
        }

        private static void Report(List<Row> rows)
        {
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("  map         period  shift |     peaks: n  rate |  regularised: n | player grid: n |   detector: n  rate  prec");

            foreach (Row row in rows.OrderBy(r => r.MapPeriod))
            {
                TestContext.Out.WriteLine($"  {row.Id,-10} {row.MapPeriod,5:0}ms {row.Shift,5} | "
                                          + $"{row.Raw.Count,12} {row.Raw.Rate,5:0.00} | "
                                          + $"{row.Regularised.Count,15} | {row.PlayerGrid.Count,14} | "
                                          + $"{row.Detected.Count,14} {row.Detected.Rate,5:0.00} {row.Detected.Precision,5:0}%");
            }

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("  reading                 rate med   on level   coverage   precision   error med");

            foreach ((string name, Func<Row, Result> pick) in new (string, Func<Row, Result>)[]
                     {
                         ("tracked, raw peaks", r => r.Raw),
                         ("tracked, regularised", r => r.Regularised),
                         ("tracked, player grid", r => r.PlayerGrid),
                         ("trained detector", r => r.Detected),
                     })
            {
                var rates = rows.Select(r => pick(r).Rate).ToArray();
                var coverage = rows.Select(r => pick(r).Coverage).ToArray();
                var precision = rows.Select(r => pick(r).Precision).ToArray();
                var median = rows.Select(r => pick(r).Median).ToArray();

                int onLevel = rates.Count(r => r > 0.75 && r < 1.33);

                TestContext.Out.WriteLine($"  {name,-22} {Median(rates),7:0.00} {onLevel,10}/{rates.Length} "
                                          + $"{Median(coverage),9:0}% {Median(precision),10:0}% {Median(median),10:0}ms");
            }
        }

        /// <summary>
        /// The beats the player uses today: peaks over the activation, spaced onto a metre, then levelled.
        /// </summary>
        private static double[] ExistingPipeline(float[] samples, string activationPath)
        {
            if (!File.Exists(activationPath))
                return Array.Empty<double>();

            using var tracker = new StreamingBeatTracker(activationPath, 2);
            tracker.Add(samples);
            tracker.Flush();

            return BeatThisBeatTracker.Peaks(tracker.Activation.ToArray()).ToArray();
        }

        private readonly struct Row
        {
            public Row(string id, double mapPeriod, int shift, Result raw, Result regularised, Result playerGrid, Result detected)
            {
                Id = id;
                MapPeriod = mapPeriod;
                Shift = shift;
                Raw = raw;
                Regularised = regularised;
                PlayerGrid = playerGrid;
                Detected = detected;
            }

            public string Id { get; }
            public double MapPeriod { get; }

            /// <summary>The octaves the grid was moved by for the whole track, which is the stage under suspicion.</summary>
            public int Shift { get; }

            /// <summary>The tracked beats before anything is done to them.</summary>
            public Result Raw { get; }

            /// <summary>The tracked beats spaced onto a metre, which is where the subdivisions are meant to go.</summary>
            public Result Regularised { get; }

            /// <summary>The grid the player is actually handed.</summary>
            public Result PlayerGrid { get; }

            public Result Detected { get; }
        }

        private readonly struct Result
        {
            public Result(int count, double rate, double coverage, double precision, double median)
            {
                Count = count;
                Rate = rate;
                Coverage = coverage;
                Precision = precision;
                Median = median;
            }

            public int Count { get; }
            public double Rate { get; }
            public double Coverage { get; }
            public double Precision { get; }
            public double Median { get; }
        }

        private static Result Measure(double[] beats, double[] grid, double seconds)
        {
            if (beats.Length == 0)
                return new Result(0, 0, 0, 0, 0);

            var distances = new List<double>();
            int precise = 0;

            foreach (double beat in beats)
            {
                double best = double.MaxValue;

                foreach (double reference in grid)
                {
                    double distance = Math.Abs(reference - beat);

                    if (distance < best)
                        best = distance;
                }

                distances.Add(best);

                if (best <= 60)
                    precise++;
            }

            int covered = 0;

            foreach (double reference in grid)
            {
                foreach (double beat in beats)
                {
                    if (Math.Abs(beat - reference) <= 60)
                    {
                        covered++;
                        break;
                    }
                }
            }

            distances.Sort();

            return new Result(
                beats.Length,
                (beats.Length / seconds) / (grid.Length / seconds),
                100.0 * covered / grid.Length,
                100.0 * precise / beats.Length,
                distances[distances.Count / 2]);
        }

        private static double Median(double[] values)
        {
            var sorted = values.OrderBy(v => v).ToArray();

            return sorted[sorted.Length / 2];
        }

        private static float[] ReadAudio(string path)
        {
            byte[] bytes = File.ReadAllBytes(path);
            var samples = new float[bytes.Length / 4];

            Buffer.BlockCopy(bytes, 0, samples, 0, samples.Length * 4);

            return samples;
        }

        private static double[] MapBeats(string archive, double until)
        {
            string scratch = Path.Combine(Path.GetTempPath(), "paratactus-compare-" + Guid.NewGuid().ToString("N"));

            try
            {
                Directory.CreateDirectory(scratch);
                ZipFile.ExtractToDirectory(archive, scratch);

                string map = Directory.EnumerateFiles(scratch, "*.osu", SearchOption.AllDirectories).First();
                var points = new List<(double Time, double Length)>();
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

                    if (double.TryParse(fields[0], NumberStyles.Float, CultureInfo.InvariantCulture, out double time)
                        && double.TryParse(fields[1], NumberStyles.Float, CultureInfo.InvariantCulture, out double length)
                        && length > 0)
                    {
                        points.Add((time, length));
                    }
                }

                points.Sort((a, b) => a.Time.CompareTo(b.Time));

                var beats = new List<double>();

                for (int i = 0; i < points.Count; i++)
                {
                    double end = i + 1 < points.Count ? points[i + 1].Time : Math.Max(until, points[i].Time);

                    for (double time = points[i].Time; time < end; time += points[i].Length)
                        beats.Add(time);
                }

                beats.Sort();

                return beats.ToArray();
            }
            finally
            {
                if (Directory.Exists(scratch))
                    Directory.Delete(scratch, true);
            }
        }
    }
}
