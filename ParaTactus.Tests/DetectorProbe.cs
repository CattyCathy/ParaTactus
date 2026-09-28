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

namespace ParaTactus.Tests
{
    /// <summary>
    /// Runs the trained detector from C# and measures its beats against a beatmap's own grid.
    /// </summary>
    /// <remarks>
    /// The same measurement the Python side makes, through the code the player would use, so that agreement between the
    /// two is evidence and not an assumption. Everything about the detector is checked here at once: the frontend, the
    /// session, the exponentiation of the period head, the suppression, and the conversion from frames to milliseconds.
    /// A mistake in any one of them moves these numbers, and the Python numbers they are compared with are the ones the
    /// model was selected on.
    ///
    /// Set <c>OSUTEST_DATASET</c> to the exported dataset, <c>OSUTEST_CORPUS</c> to the .osz files and
    /// <c>OSUTEST_DETECTOR</c> to the ONNX model.
    /// </remarks>
    [Explicit("Needs OSUTEST_DATASET, OSUTEST_CORPUS and OSUTEST_DETECTOR.")]
    public class DetectorProbe
    {
        [OneTimeSetUp]
        public void SetUp() => AudioTestEnvironment.Initialise();

        [Test]
        public void MeasureTheTrainedDetectorAgainstTheMaps()
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

            int limit = int.TryParse(Environment.GetEnvironmentVariable("OSUTEST_LIMIT"), out int wanted) ? wanted : 8;

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

                if (fields.Length < 4)
                    continue;

                string id = fields[0];

                if (!archives.TryGetValue(id, out string archive))
                    continue;

                string audioPath = Path.Combine(dataset, "audio", fields[2]);

                if (!File.Exists(audioPath))
                    continue;

                float[] samples = ReadAudio(audioPath);
                double seconds = samples.Length / (double)LogMel.SampleRate;

                double[] grid = MapBeats(archive, seconds * 1000);

                if (grid.Length < 16)
                    continue;

                double[] beats = detector.Beats(samples);

                rows.Add(Measure(id, beats, grid, seconds));
            }

            TestContext.Out.WriteLine($"measured {rows.Count} tracks");
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("  track           map   beats     rate   coverage  precision   median      p90");

            foreach (Row row in rows.OrderByDescending(r => r.Rate))
            {
                TestContext.Out.WriteLine($"  {row.Id,-12} {row.MapBeats,5} {row.Beats,7} {row.Rate,8:0.00} "
                                          + $"{row.Coverage,9:0}% {row.Precision,9:0}% {row.Median,7:0}ms {row.P90,7:0}ms");
            }

            // Both heads, on one track, printed as distributions. A beat curve that is mostly zero and a period head
            // that is anywhere but a few frames are different faults with different causes, and the summary above
            // cannot tell them apart - it can only say that almost nothing survived.
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("the two heads on the first track, as distributions:");

            string firstId = rows[0].Id;
            string firstAudio = Path.Combine(dataset, "audio", firstId + ".f32");

            if (File.Exists(firstAudio))
            {
                float[] samples = ReadAudio(firstAudio);
                (float[] beats, float[] periods) = detector.Activations(samples);

                (double beatMin, double beatMax, double beatMean) = Describe(beats);
                (double periodMin, double periodMax, double periodMean) = Describe(periods);

                TestContext.Out.WriteLine($"  beat confidence: min {beatMin:0.0000} max {beatMax:0.0000} mean {beatMean:0.0000}");
                TestContext.Out.WriteLine($"  period frames:   min {periodMin:0.00} max {periodMax:0.00} mean {periodMean:0.00}");
                TestContext.Out.WriteLine($"  frames: {beats.Length}");

                int above = 0;

                foreach (float value in beats)
                {
                    if (value >= 0.5f)
                        above++;
                }

                TestContext.Out.WriteLine($"  frames at or above 0.5: {above}");

                int[] suppressed = LearnedBeatDetector.Suppress(beats, periods);

                TestContext.Out.WriteLine($"  after suppression: {suppressed.Length} beats");

                if (suppressed.Length > 1)
                {
                    var gaps = new List<double>();

                    for (int i = 1; i < suppressed.Length; i++)
                        gaps.Add(suppressed[i] - suppressed[i - 1]);

                    gaps.Sort();

                    TestContext.Out.WriteLine($"  their gaps in frames: median {gaps[gaps.Count / 2]:0}, "
                                              + $"first {gaps[0]:0}, last {gaps[^1]:0}");
                }

                TestContext.Out.WriteLine($"  the Python side on this track reports hundreds of beats, so a beat head that "
                                          + "is mostly zero here and not there is a frontend problem, and a period head "
                                          + "that is not around 16 to 35 frames is an exponentiation problem.");
            }

            if (rows.Count == 0)
                Assert.Fail("nothing could be measured");

            double[] rates = rows.Select(r => r.Rate).ToArray();
            double[] coverages = rows.Select(r => r.Coverage).ToArray();
            double[] precisions = rows.Select(r => r.Precision).ToArray();
            double[] medians = rows.Select(r => r.Median).ToArray();

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine($"  rate      median {Median(rates):0.00}   "
                                      + $"within 0.75-1.33: {rates.Count(r => r > 0.75 && r < 1.33)}/{rates.Length}");
            TestContext.Out.WriteLine($"  coverage  median {Median(coverages):0}%");
            TestContext.Out.WriteLine($"  precision median {Median(precisions):0}%");
            TestContext.Out.WriteLine($"  error     median {Median(medians):0}ms");
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("  the Python side reports rate 1.04, coverage 85%, precision 81% and median 11ms on");
            TestContext.Out.WriteLine("  the same measurement, so a large disagreement here is a defect in this path.");
        }

        private readonly struct Row
        {
            public Row(string id, int mapBeats, int beats, double rate, double coverage, double precision,
                       double median, double p90)
            {
                Id = id;
                MapBeats = mapBeats;
                Beats = beats;
                Rate = rate;
                Coverage = coverage;
                Precision = precision;
                Median = median;
                P90 = p90;
            }

            public string Id { get; }
            public int MapBeats { get; }
            public int Beats { get; }
            public double Rate { get; }
            public double Coverage { get; }
            public double Precision { get; }
            public double Median { get; }
            public double P90 { get; }
        }

        private static Row Measure(string id, double[] beats, double[] grid, double seconds)
        {
            if (beats.Length == 0)
                return new Row(id, grid.Length, 0, 0, 0, 0, 0, 0);

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

            return new Row(id, grid.Length, beats.Length,
                (beats.Length / seconds) / (grid.Length / seconds),
                100.0 * covered / grid.Length,
                100.0 * precise / beats.Length,
                distances[distances.Count / 2],
                distances[Math.Min(distances.Count - 1, (int)(distances.Count * 0.9))]);
        }

        private static double Median(double[] values)
        {
            var sorted = values.OrderBy(v => v).ToArray();

            return sorted[sorted.Length / 2];
        }

        private static (double Minimum, double Maximum, double Mean) Describe(float[] values)
        {
            if (values.Length == 0)
                return (0, 0, 0);

            double minimum = double.MaxValue;
            double maximum = double.MinValue;
            double total = 0;

            foreach (float value in values)
            {
                minimum = Math.Min(minimum, value);
                maximum = Math.Max(maximum, value);
                total += value;
            }

            return (minimum, maximum, total / values.Length);
        }

        /// <summary>Reads the raw float32 audio the exporter wrote.</summary>
        private static float[] ReadAudio(string path)
        {
            byte[] bytes = File.ReadAllBytes(path);
            var samples = new float[bytes.Length / 4];

            Buffer.BlockCopy(bytes, 0, samples, 0, samples.Length * 4);

            return samples;
        }

        /// <summary>A map's beats from its uninherited timing points, in milliseconds.</summary>
        private static double[] MapBeats(string archive, double until)
        {
            string scratch = Path.Combine(Path.GetTempPath(), "paratactus-grid-" + Guid.NewGuid().ToString("N"));

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
