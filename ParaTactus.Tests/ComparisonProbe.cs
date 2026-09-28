using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using NUnit.Framework;
using ParaTactus.Decoding;
using ParaTactus.Detector;
using ParaTactus.Features;
using ParaTactus.Grid;
using ParaTactus.Tracking;

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

            // Which tracks to measure. The manifest is ordered by identifier, which has nothing to do with tempo, so a
            // first twelve taken in that order is an arbitrary sample and the "four fastest" of it is four arbitrary
            // tracks. OSUTEST_FASTEST takes the shortest label periods instead, which is the material the objective
            // names, and OSUTEST_OLDEST keeps the manifest order for a comparison with earlier runs.
            bool fastest = string.Equals(Environment.GetEnvironmentVariable("OSUTEST_FASTEST"), "true", StringComparison.OrdinalIgnoreCase);

            // The suppression shares to sweep, as a comma-separated list.
            string shareList = Environment.GetEnvironmentVariable("OSUTEST_SHARES") ?? "0.4,0.5,0.6,0.75";
            double[] shares = shareList.Split(',')
                                       .Select(part => double.TryParse(part.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double value) ? value : 0)
                                       .Where(value => value > 0)
                                       .ToArray();

            // The period smoothings to sweep, in seconds either side. Zero is the head used as it comes.
            string smoothingList = Environment.GetEnvironmentVariable("OSUTEST_SMOOTHING") ?? "0";
            double[] smoothings = smoothingList.Split(',')
                                               .Select(part => double.TryParse(part.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double value) ? value : -1)
                                               .Where(value => value >= 0)
                                               .ToArray();

            if (smoothings.Length == 0)
                smoothings = new double[] { 0 };

            // The comparison is run twice - once with the four slowest maps and once with everything - because the
            // objective is specifically about tempo-changing and fast material, and a median over a corpus that is
            // mostly one tempo cannot answer a question about the rest of it.
            var archives = new Dictionary<string, string>();

            foreach (string file in Directory.EnumerateFiles(corpus, "*.osz", SearchOption.AllDirectories))
                archives[Path.GetFileName(file).Split(' ')[0]] = file;

            var lines = File.ReadLines(Path.Combine(dataset, "manifest.tsv")).Skip(1)
                            .Select(line => line.Split('\t'))
                            .Where(fields => fields.Length >= 7 && archives.ContainsKey(fields[0])
                                             && File.Exists(Path.Combine(dataset, "audio", fields[2])))
                            .ToList();

            if (fastest)
            {
                lines = lines
                        .OrderBy(fields => double.TryParse(fields[6], NumberStyles.Float, CultureInfo.InvariantCulture, out double p) ? p : double.MaxValue)
                        .ToList();
            }

            var rows = new List<Row>();

            using var detector = new LearnedBeatDetector(detectorPath);

            foreach (string[] fields in lines)
            {
                if (rows.Count >= limit)
                    break;

                string archive = archives[fields[0]];
                string audioPath = Path.Combine(dataset, "audio", fields[2]);

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

                // The detector at each combination of suppression share and period smoothing under test, from one pass
                // over the audio. Both are swept in the same run as the comparison because the share in use was first
                // chosen against a comparison whose units were wrong, and a value tuned on a bad reference is worth
                // nothing. Smoothing is in the sweep because it is the candidate answer to the pulses being unsteady:
                // the period sets how far apart beats must be, so a period that wobbles frame to frame wobbles the
                // spacing, and taking the middle of a window is the cheapest way to stop that.
                var atShares = new Dictionary<(double Share, double Smoothing), Result>();

                foreach (double share in shares)
                {
                    foreach (double smoothing in smoothings)
                        atShares[(share, smoothing)] = Measure(detector.Beats(samples, share, smoothing), grid, seconds);
                }

                rows.Add(new Row(
                    fields[0],
                    double.TryParse(fields[6], NumberStyles.Float, CultureInfo.InvariantCulture, out double period) ? period : 0,
                    playerGrid?.MetricalShift ?? 0,
                    Measure(tracked, grid, seconds),
                    Measure(regularised, grid, seconds),
                    Measure(gridBeats, grid, seconds),
                    atShares));
            }

            if (rows.Count == 0)
                Assert.Fail("nothing could be measured");

            Report(rows, shares, smoothings);

            // The maps whose own beat is fastest, which is the material the objective names and the material a median
            // over the corpus hides.
            var quickest = rows.OrderBy(r => r.MapPeriod).Take(Math.Min(4, rows.Count)).ToList();

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine($"the {quickest.Count} fastest beatmaps of those measured, on their own:");

            Report(quickest, shares, smoothings);
        }

        private static void Report(List<Row> rows, double[] shares, double[] smoothings)
        {
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("  map         period  shift |    peaks: n  rate | regularised: n | player grid: n | "
                                      + string.Join(" | ", from s in shares from m in smoothings
                                                           select $"@{s:0.00}/{m:0.0}s: n  rate  stead"));

            foreach (Row row in rows.OrderBy(r => r.MapPeriod))
            {
                var detectorCells = from s in shares from m in smoothings
                                    select row.Detected.TryGetValue((s, m), out Result at)
                                        ? $"{at.Count,4} {at.Rate,5:0.00} {at.Steadiness,4:0}"
                                        : "  n/a";

                TestContext.Out.WriteLine($"  {row.Id,-10} {row.MapPeriod,5:0}ms {row.Shift,5} | "
                                          + $"{row.Raw.Count,12} {row.Raw.Rate,5:0.00} | "
                                          + $"{row.Regularised.Count,15} | {row.PlayerGrid.Count,14} | "
                                          + string.Join(" | ", detectorCells));
            }

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("  reading                 rate med   on level   coverage   precision   error med   unsteady");

            foreach ((string name, Func<Row, Result> pick) in new (string, Func<Row, Result>)[]
                     {
                         ("tracked, raw peaks", r => r.Raw),
                         ("tracked, regularised", r => r.Regularised),
                         ("tracked, player grid", r => r.PlayerGrid),
                     })
            {
                Summarise(rows, name, pick);
            }

            foreach (double share in shares)
            {
                foreach (double smoothing in smoothings)
                    Summarise(rows, $"detector @{share:0.00}/{smoothing:0.0}s", r => r.Detected[(share, smoothing)]);
            }
        }

        private static void Summarise(List<Row> rows, string name, Func<Row, Result> pick)
        {
            var rates = rows.Select(r => pick(r).Rate).ToArray();
            var coverage = rows.Select(r => pick(r).Coverage).ToArray();
            var precision = rows.Select(r => pick(r).Precision).ToArray();
            var median = rows.Select(r => pick(r).Median).ToArray();
            var steadiness = rows.Select(r => pick(r).Steadiness).ToArray();

            int onLevel = rates.Count(r => r > 0.75 && r < 1.33);

            TestContext.Out.WriteLine($"  {name,-22} {Median(rates),7:0.00} {onLevel,10}/{rates.Length} "
                                      + $"{Median(coverage),9:0}% {Median(precision),10:0}% {Median(median),10:0}ms "
                                      + $"{Median(steadiness),10:0}ms");
        }

        /// <summary>
        /// The beats the player uses today, produced the way the player produces them.
        /// </summary>
        /// <remarks>
        /// The tracker's own beat list rather than the peak picker's, and that is the whole point of doing it here
        /// rather than in a probe. <c>BeatThisBeatTracker.Peaks</c> answers in frames and
        /// <c>BeatTrainRegulariser.Regularise</c> takes milliseconds, so feeding one to the other is a factor of twenty
        /// - and a first version of this did exactly that, which made the grid look as though it kept a sixteenth of
        /// the beats and sent an afternoon into the metrical level decision. The tracker converts, because it has to in
        /// order to have a beat time at all, and this uses the converted list.
        /// </remarks>
        private static double[] ExistingPipeline(float[] samples, string activationPath)
        {
            if (!File.Exists(activationPath))
                return Array.Empty<double>();

            using var tracker = new StreamingBeatTracker(activationPath, 2);
            tracker.Add(samples);
            tracker.Flush();

            return tracker.Beats.ToArray();
        }

        private readonly struct Row
        {
            public Row(string id, double mapPeriod, int shift, Result raw, Result regularised, Result playerGrid,
                       Dictionary<(double Share, double Smoothing), Result> detected)
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

            /// <summary>The octaves the grid was moved by for the whole track.</summary>
            public int Shift { get; }

            /// <summary>The tracked beats before anything is done to them.</summary>
            public Result Raw { get; }

            /// <summary>The tracked beats spaced onto a metre, which is where the subdivisions are meant to go.</summary>
            public Result Regularised { get; }

            /// <summary>The grid the player is actually handed.</summary>
            public Result PlayerGrid { get; }

            /// <summary>The trained detector, at each suppression share and period smoothing under test.</summary>
            public Dictionary<(double Share, double Smoothing), Result> Detected { get; }
        }

        private readonly struct Result
        {
            public Result(int count, double rate, double coverage, double precision, double median, double steadiness)
            {
                Count = count;
                Rate = rate;
                Coverage = coverage;
                Precision = precision;
                Median = median;
                Steadiness = steadiness;
            }

            public int Count { get; }
            public double Rate { get; }
            public double Coverage { get; }
            public double Precision { get; }
            public double Median { get; }

            /// <summary>
            /// How much the gap from one beat to the next changes, as the middle change in milliseconds.
            /// </summary>
            /// <remarks>
            /// The number that says whether the pulses are steady, which is a different question from whether they are
            /// in the right place. A reading can be right about the tempo over a track and still look wrong if the gaps
            /// alternate, and the eye reads a changing gap as a rhythm of its own. It is the middle of the absolute
            /// change from one gap to the next rather than the spread of the gaps, because one change of tempo in a
            /// track would show up in the spread and is not what makes a pulse look unsteady.
            /// </remarks>
            public double Steadiness { get; }
        }

        private static Result Measure(double[] beats, double[] grid, double seconds)
        {
            if (beats.Length == 0)
                return new Result(0, 0, 0, 0, 0, 0);

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

            var changes = new List<double>();

            for (int i = 2; i < beats.Length; i++)
                changes.Add(Math.Abs((beats[i] - beats[i - 1]) - (beats[i - 1] - beats[i - 2])));

            changes.Sort();

            return new Result(
                beats.Length,
                (beats.Length / seconds) / (grid.Length / seconds),
                100.0 * covered / grid.Length,
                100.0 * precise / beats.Length,
                distances[distances.Count / 2],
                changes.Count == 0 ? 0 : changes[changes.Count / 2]);
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
