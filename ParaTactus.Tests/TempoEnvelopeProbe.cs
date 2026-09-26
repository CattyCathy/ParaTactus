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
    /// Whether a pulse built from a smoothed tempo envelope is steadier than the beats it was built from, and still
    /// lands where the music does.
    /// </summary>
    /// <remarks>
    /// A different question from the one the other probes ask. The player made the beats visible, and what a person
    /// sees is not "is each beat within 30ms" but "does this look like it is moving with the music". The two are not
    /// the same measurement: a pulse that is perfectly even and right about the tempo reads as locked to the music
    /// even when it is forty milliseconds out, and a pulse that is exactly right on two beats in three and a fifth of
    /// a beat out on the rest reads as scattered - which is what the beats look like now.
    ///
    /// So this measures three things per section, against the beatmap's own grid:
    ///
    /// - how far the pulse is from the map's beats, which is the old measurement;
    /// - how even the pulse is, as the spread of its own intervals, which is what the eye reads as steadiness;
    /// - how many of the map's beats have a pulse near them, which is whether a beat goes by with nothing happening.
    ///
    /// The tempo envelope is fitted to the tracker's own beats and then a regular pulse is laid down from it. That is
    /// the method the onset arbiter's remarks say is dead, and the reason it is worth measuring anyway is that the
    /// arbiter used the audio's own onsets for the phase, which cannot work in dense music - it says so itself. This
    /// uses the tracker's phase, which is where the evidence is: the model's activation at the map's own beats in a
    /// ramp is 0.93 against a whole-track mean of -1.79, so the beats are there to be found even when the pulse wanders.
    /// </remarks>
    [TestFixture]
    [Explicit("Needs OSUTEST_AUDIO pointing at Designant. - the audio is not in the repository.")]
    public class TempoEnvelopeProbe
    {
        private const string model = "beat-this-final0-int8";

        [OneTimeSetUp]
        public void SetUp() => AudioTestEnvironment.Initialise();

        [Test]
        public void CompareTheTrackersBeatsWithAPulseBuiltFromASmoothedTempoEnvelope()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string modelPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", $"{model}.onnx");

            if (!File.Exists(modelPath))
                Assert.Ignore($"no model at {modelPath}");

            double[] grid = beatmapGrid(timingPoints());
            double[] tracked = BeatThisBeatTracker.BeatTimes(audio, modelPath, BassAudioDecoder.Default);

            TestContext.Out.WriteLine($"the model reports {tracked.Length} beats; the map's grid has {grid.Length}");

            // A window of half a second either way in beats, which at the tempi here is about fifteen beats: long
            // enough to average out where the tracker is wrong, short enough to follow a tempo that changes every few
            // seconds. The pass is repeated because a line fitted to beats that include the wrong ones is itself a
            // little wrong, and refitting to the beats it has already moved is what converges.
            var envelopes = new List<(string Name, double[] Beats)>();

            foreach (double window in new[] { 6.0, 10.0, 16.0, 24.0 })
                envelopes.Add(($"pulse, {window:0} beats either side", pulse(tracked, window)));

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("measuring each population over the whole track:");
            report("the model's own beats", tracked, grid);

            foreach ((string name, double[] beats) in envelopes)
                report(name, beats, grid);

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("and per section, because the failure is local:");

            foreach ((double from, double to) in new[]
                     {
                         (0.0, 82_000.0), (82_000.0, 90_000.0), (90_000.0, 110_000.0), (110_000.0, 140_000.0),
                         (140_000.0, 178_000.0),
                     })
            {
                double[] sectionGrid = grid.Where(g => g >= from && g < to).ToArray();

                TestContext.Out.WriteLine("");
                TestContext.Out.WriteLine($"  {from / 1000:0}s to {to / 1000:0}s, {sectionGrid.Length} beats in the map");

                report("    model", tracked.Where(b => b >= from && b < to).ToArray(), sectionGrid);

                foreach ((string name, double[] beats) in envelopes)
                    report($"    {name}", beats.Where(b => b >= from && b < to).ToArray(), sectionGrid);
            }
        }

        /// <summary>
        /// A regular pulse from the tempo envelope of a set of beats.
        /// </summary>
        /// <remarks>
        /// For each beat, a line is fitted to its neighbours - the tracker's own beats, not the pulse being built - and
        /// the pulse is placed at the point on that line nearest the beat. The line carries a period and a phase, so
        /// the pulse follows the tempo where it changes and stays even where it does not; the beat's own position is used
        /// only to choose which turn of the pulse it belongs to.
        ///
        /// Fitted to the beats rather than to the audio deliberately. The onset arbiter in this suite tried the audio
        /// and measured why it cannot work: in music dense enough to have an onset on every sixteenth, a comb at any
        /// phase finds onsets, so the best phase is the noise in the comparison rather than a fact about the track.
        /// </remarks>
        private static double[] pulse(double[] beats, double window)
        {
            if (beats.Length < 4)
                return beats;

            var fitted = new double[beats.Length];

            for (int i = 0; i < beats.Length; i++)
            {
                int low = Math.Max(0, i - (int)window);
                int high = Math.Min(beats.Length - 1, i + (int)window);

                double slope = 0;
                double intercept = 0;

                leastSquares(beats, low, high, out slope, out intercept);

                // Where the beat ought to be on that line: the line evaluated at the beat's own index.
                double expected = intercept + (slope * i);

                // A step of the pulse, so that a beat the tracker put a little late is placed a whole period where the
                // pulse says the period is. The index used is the beat's own, because the tracker reports one beat per
                // beat: it is the placement that is wrong, not the count, in the passages this is for.
                fitted[i] = expected;
            }

            return fitted;
        }

        /// <summary>A least-squares line through beats from one index to another, inclusive.</summary>
        private static void leastSquares(double[] beats, int low, int high, out double slope, out double intercept)
        {
            double sumX = 0;
            double sumY = 0;
            double sumXY = 0;
            double sumXX = 0;
            double n = high - low + 1;

            for (int i = low; i <= high; i++)
            {
                double x = i - low;
                double y = beats[i];

                sumX += x;
                sumY += y;
                sumXY += x * y;
                sumXX += x * x;
            }

            double denominator = (n * sumXX) - (sumX * sumX);

            if (Math.Abs(denominator) < 1e-9)
            {
                slope = 0;
                intercept = sumY / n;
                return;
            }

            slope = ((n * sumXY) - (sumX * sumY)) / denominator;
            intercept = (sumY - (slope * sumX)) / n;
        }

        /// <summary>How far a population is from the map, and how even it is in itself.</summary>
        private static void report(string name, double[] beats, double[] grid)
        {
            if (beats.Length < 3 || grid.Length == 0)
            {
                TestContext.Out.WriteLine($"  {name,-34} too few beats to say");
                return;
            }

            var errors = beats.Select(b => Math.Abs(signedDistance(grid, b))).OrderBy(e => e).ToArray();
            var gaps = new List<double>();

            for (int i = 1; i < beats.Length; i++)
                gaps.Add(beats[i] - beats[i - 1]);

            gaps.Sort();

            double medianGap = gaps[gaps.Count / 2];
            var deviations = gaps.Select(g => Math.Abs(g - medianGap)).OrderBy(d => d).ToArray();

            int covered = grid.Count(g => beats.Any(b => Math.Abs(b - g) <= 60));

            TestContext.Out.WriteLine($"  {name,-34} median {errors[errors.Length / 2],6:0.0}ms   "
                                      + $"spread of its own intervals {deviations[deviations.Length / 2],6:0.0}ms   "
                                      + $"map beats covered {covered * 100.0 / grid.Length,5:0.0}%");
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

        private static double[] beatmapGrid(List<(double Time, double BeatLength)> timing)
        {
            var grid = new List<double>();

            for (int i = 0; i < timing.Count; i++)
            {
                double until = i + 1 < timing.Count ? timing[i + 1].Time : timing[i].Time + 60_000;

                for (double time = timing[i].Time; time < until && grid.Count < 500_000; time += timing[i].BeatLength)
                    grid.Add(time);
            }

            grid.Sort();

            return grid.ToArray();
        }

        private static List<(double Time, double BeatLength)> timingPoints()
        {
            string file = Path.Combine(AppContext.BaseDirectory, "DesignantTimingPoints.csv");

            Assert.That(File.Exists(file), Is.True, $"{file} should be copied beside the test");

            var points = new List<(double, double)>();

            foreach (string line in File.ReadLines(file))
            {
                if (line.StartsWith("#") || !line.Contains(','))
                    continue;

                string[] fields = line.Split(',');

                if (fields.Length >= 2
                    && double.TryParse(fields[0], NumberStyles.Float, CultureInfo.InvariantCulture, out double time)
                    && double.TryParse(fields[1], NumberStyles.Float, CultureInfo.InvariantCulture, out double length))
                {
                    points.Add((time, length));
                }
            }

            return points.OrderBy(p => p.Item1).ToList();
        }
    }
}
