using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using NUnit.Framework;
using ParaTactus;

namespace ParaTactus.Tests
{
    /// <summary>
    /// Lays the detector's beats against a beatmap's own timing points, one beat at a time.
    /// </summary>
    /// <remarks>
    /// Every measurement of this detector so far has been an aggregate: a middle distance, a share covered, a rate. An
    /// aggregate cannot show a systematic offset, a whole-track shift, or a passage where the two readings disagree
    /// about the level, and all three are things a person hears. This prints the two sequences against each other so
    /// they can be looked at rather than summarised, and reports the offsets as a histogram so that a constant shift and
    /// a scattered one are told apart.
    ///
    /// It runs what the player runs: <see cref="LearnedBeatDetector.Beats(IReadOnlyList{float})"/> with the defaults,
    /// so a value changed in the player's path changes what this shows.
    ///
    /// Set <c>OSUTEST_AUDIO</c> to the track, <c>OSUTEST_MAP</c> to its .osz or .osu, and <c>OSUTEST_DETECTOR</c> to
    /// the model. <c>OSUTEST_FROM</c> and <c>OSUTEST_LENGTH</c> choose the window to print.
    /// </remarks>
    [Explicit("Needs OSUTEST_AUDIO, OSUTEST_MAP and OSUTEST_DETECTOR.")]
    public class AlignmentProbe
    {
        [OneTimeSetUp]
        public void SetUp() => AudioTestEnvironment.Initialise();

        [Test]
        public void ShowTheDetectorsBeatsAgainstTheMapsTimingPoints()
        {
            string audioPath = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");
            string mapPath = Environment.GetEnvironmentVariable("OSUTEST_MAP");
            string detectorPath = Environment.GetEnvironmentVariable("OSUTEST_DETECTOR");

            if (string.IsNullOrEmpty(audioPath) || !File.Exists(audioPath))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            if (string.IsNullOrEmpty(mapPath) || !File.Exists(mapPath))
                Assert.Ignore("Set OSUTEST_MAP to the .osz or .osu.");

            if (string.IsNullOrEmpty(detectorPath) || !File.Exists(detectorPath))
                Assert.Ignore("Set OSUTEST_DETECTOR to the model.");

            string scratch = Path.Combine(Path.GetTempPath(), "paratactus-align-" + Guid.NewGuid().ToString("N"));
            string map = mapPath;

            try
            {
                if (mapPath.EndsWith(".osz", StringComparison.OrdinalIgnoreCase))
                {
                    Directory.CreateDirectory(scratch);
                    ZipFile.ExtractToDirectory(mapPath, scratch);
                    map = Directory.EnumerateFiles(scratch, "*.osu", SearchOption.AllDirectories).First();
                }

                float[] samples = Read(audioPath);
                double seconds = samples.Length / (double)LogMel.SampleRate;
                double[] grid = TimingPoints(map, seconds * 1000);

                TestContext.Out.WriteLine($"track  {Path.GetFileName(audioPath)}  {seconds:0.0}s");
                TestContext.Out.WriteLine($"map    {Path.GetFileName(map)}");
                TestContext.Out.WriteLine($"timing points give {grid.Length} beats, first {grid.FirstOrDefault():0}ms, "
                                          + $"last {grid[^1]:0}ms, middle gap {MiddleGap(grid):0}ms");

                using var detector = new LearnedBeatDetector(detectorPath);

                var clock = System.Diagnostics.Stopwatch.StartNew();
                double[] beats = detector.Beats(samples);
                clock.Stop();

                TestContext.Out.WriteLine($"detector  {beats.Length} beats in {clock.Elapsed.TotalSeconds:0.0}s, "
                                          + $"first {beats.FirstOrDefault():0}ms, last {beats[^1]:0}ms, "
                                          + $"middle gap {MiddleGap(beats):0}ms");
                TestContext.Out.WriteLine("");

                if (grid.Length == 0 || beats.Length == 0)
                    Assert.Fail("one of the two readings is empty");

                // The offset of each detector beat from the nearest timing point. Signed, so a constant shift shows as a
                // histogram in one bin and a scattered reading as one spread across many.
                var offsets = new List<double>();

                foreach (double beat in beats)
                    offsets.Add(SignedDistance(grid, beat));

                var sorted = offsets.OrderBy(o => o).ToArray();

                TestContext.Out.WriteLine($"offset of each detector beat from the nearest timing point:");
                TestContext.Out.WriteLine($"  middle {sorted[sorted.Length / 2]:+0;-0;0}ms   "
                                          + $"p10 {sorted[sorted.Length / 10]:+0;-0;0}ms   "
                                          + $"p90 {sorted[sorted.Length * 9 / 10]:+0;-0;0}ms   "
                                          + $"half the readings within "
                                          + $"{(sorted[sorted.Length * 3 / 4] - sorted[sorted.Length / 4]) / 2:0}ms of the middle");

                var buckets = new SortedDictionary<int, int>();

                foreach (double offset in offsets)
                    buckets[(int)(Math.Round(offset / 20) * 20)] = buckets.TryGetValue((int)(Math.Round(offset / 20) * 20), out int seen) ? seen + 1 : 1;

                TestContext.Out.WriteLine("");
                TestContext.Out.WriteLine("  offset   beats");

                foreach (var pair in buckets.OrderBy(p => p.Key))
                    TestContext.Out.WriteLine($"  {pair.Key,+5}ms {new string('#', Math.Min(60, pair.Value / 4))} {pair.Value}");

                // And the two sequences side by side over a window, in time order, which is what says whether they agree
                // about which beat is which. One line per beat from either reading, with the other reading's nearest
                // beat beside it, so a beat only one of them has is a line with nothing next to it.
                double from = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_FROM"), NumberStyles.Float, CultureInfo.InvariantCulture, out double f) ? f : 60_000;
                double length = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_LENGTH"), NumberStyles.Float, CultureInfo.InvariantCulture, out double l) ? l : 12_000;

                TestContext.Out.WriteLine("");
                TestContext.Out.WriteLine($"both readings from {from / 1000:0}s to {(from + length) / 1000:0}s, in time order:");
                TestContext.Out.WriteLine("    source      time    nearest other     offset   own gap   other gap");

                double previousBeat = double.NaN;
                double previousGrid = double.NaN;

                foreach ((bool isBeat, double time) in Timeline(beats, grid, from, from + length))
                {
                    double other = isBeat ? Nearest(grid, time) : Nearest(beats, time);
                    double ownGap = isBeat ? time - previousBeat : time - previousGrid;
                    double otherGap = isBeat ? other - previousGrid : other - previousBeat;

                    TestContext.Out.WriteLine($"  {(isBeat ? "detector" : "map"),-10} {time,9:0}ms {other,14:0}ms "
                                              + $"{time - other,+10:0}ms {ownGap,8:0}ms {otherGap,10:0}ms"
                                              + (isBeat ? "" : "   <- the map has this beat and the detector does not"));

                    if (isBeat)
                        previousBeat = time;
                    else
                        previousGrid = time;
                }
            }
            finally
            {
                if (Directory.Exists(scratch))
                    Directory.Delete(scratch, true);
            }
        }

        /// <summary>
        /// Both readings merged into one sequence for a window, each entry saying which reading it came from.
        /// </summary>
        private static IEnumerable<(bool IsBeat, double Time)> Timeline(double[] beats, double[] grid, double from, double to)
        {
            int b = Array.FindIndex(beats, time => time >= from);
            int g = Array.FindIndex(grid, time => time >= from);

            if (b < 0)
                b = beats.Length;

            if (g < 0)
                g = grid.Length;

            while (true)
            {
                bool haveBeat = b < beats.Length && beats[b] < to;
                bool haveGrid = g < grid.Length && grid[g] < to;

                if (!haveBeat && !haveGrid)
                    break;

                // The earlier of the two goes first; a tie takes the detector's, because a beat both readings have
                // should be shown as the detector's with the map's beside it rather than as a beat the detector missed.
                if (haveBeat && (!haveGrid || beats[b] <= grid[g]))
                {
                    yield return (true, beats[b]);
                    b++;
                }
                else
                {
                    yield return (false, grid[g]);
                    g++;
                }
            }
        }

        private static double Nearest(double[] times, double at)
        {
            double best = double.MaxValue;

            foreach (double time in times)
            {
                if (Math.Abs(time - at) < Math.Abs(best))
                    best = time - at;
            }

            return at + best;
        }

        private static double SignedDistance(double[] grid, double time)
        {
            return Nearest(grid, time) - time;
        }

        private static double MiddleGap(double[] times)
        {
            if (times.Length < 2)
                return 0;

            var gaps = new List<double>();

            for (int i = 1; i < times.Length; i++)
                gaps.Add(times[i] - times[i - 1]);

            gaps.Sort();

            return gaps[gaps.Count / 2];
        }

        /// <summary>The beats a map's uninherited timing points imply.</summary>
        private static double[] TimingPoints(string map, double until)
        {
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

                if (fields.Length >= 2
                    && double.TryParse(fields[0], NumberStyles.Float, CultureInfo.InvariantCulture, out double time)
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

        private static float[] Read(string path)
        {
            if (path.EndsWith(".f32", StringComparison.OrdinalIgnoreCase))
            {
                byte[] bytes = File.ReadAllBytes(path);
                var samples = new float[bytes.Length / 4];

                Buffer.BlockCopy(bytes, 0, samples, 0, samples.Length * 4);

                return samples;
            }

            return ParaTactus.Decoding.BassAudioDecoder.DecodeMono(path, LogMel.SampleRate);
        }
    }
}
