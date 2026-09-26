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
    /// How far the tracker's beats are from the beatmap's own grid, and whether a smoothing pass gets them closer.
    /// </summary>
    /// <remarks>
    /// This exists because the player made the problem visible and there is no honest way to work on it without a
    /// reference that is not the tracker. The reference here is the beatmap's own <c>[TimingPoints]</c>: the tempo the
    /// map's author set, which is the one thing about the tempo of a track that is not an estimate. For the first
    /// eighty seconds of this track that is a single tempo, so the map's own grid is a line and every beat's error is
    /// a number.
    ///
    /// The interesting question is the second one. The regulariser places every beat it keeps exactly where the tracker
    /// reported it, so a pass that moves beats onto a locally regular spacing would help if the tracker's error is
    /// jitter around a steady pulse and hurt if it is the tracker reading a real tempo change. Both happen, which is
    /// why this measures a held-out prediction: fit the pulse to beats before a point and ask whether it predicts the
    /// beat after it better than the tracker's own neighbouring gaps do. A pass that only fits the noise cannot win
    /// that bet, and a pass that finds the pulse wins it by however much the tracker is jittering.
    ///
    /// <c>[Explicit]</c> because the audio cannot be committed. Set <c>OSUTEST_AUDIO</c> to the track.
    /// </remarks>
    [TestFixture]
    [Explicit("Needs OSUTEST_AUDIO pointing at Designant. - the audio is not in the repository.")]
    public class BeatPlacementProbe
    {
        /// <summary>Where the beatmap's tempo changes stop being one tempo, in milliseconds.</summary>
        /// <remarks>
        /// The map declares 300ms from 409ms to 82008ms and then ramps. Everything after that is a moving target that
        /// a single line cannot describe, so the measurement is of the steady part - which is also where a person
        /// watching a player notices the beats not landing.
        /// </remarks>
        private const double steady_until_ms = 82000;

        private const string model = "beat-this-final0-int8";

        [OneTimeSetUp]
        public void SetUp() => AudioTestEnvironment.Initialise();

        [Test]
        public void ReportTheErrorAgainstTheBeatmapAndWhetherSmoothingWouldReduceIt()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string modelPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", $"{model}.onnx");

            if (!File.Exists(modelPath))
                Assert.Ignore($"no model at {modelPath}");

            List<TimingPoint> timing = timingPoints();
            List<double> beats = analyse(audio, modelPath);

            TestContext.Out.WriteLine($"timing points: {timing.Count}; first tempo {60000 / timing[0].BeatLength:0.#} BPM "
                                      + $"from {timing[0].Time:0}ms to {steady_until_ms:0}ms");
            TestContext.Out.WriteLine($"beats: {beats.Count} over {beats[^1] / 1000:0.#}s");

            // The reference grid: the map's own tempo from its own offset, out to where the tempo changes.
            var reference = new List<double>();

            for (double t = timing[0].Time; t <= steady_until_ms; t += timing[0].BeatLength)
                reference.Add(t);

            TestContext.Out.WriteLine($"reference beats in the steady section: {reference.Count}");

            List<double> measured = beats.Where(b => b >= timing[0].Time && b <= steady_until_ms).ToList();

            TestContext.Out.WriteLine($"{measured.Count} tracked beats in the same span; "
                                      + $"{measured.Count / (steady_until_ms / timing[0].BeatLength):0.00} of the map's beats");

            List<double> errors = measured.Select(b => nearest(reference, b)).Select(r => r.Error).ToList();
            List<double> absolute = errors.Select(Math.Abs).OrderBy(e => e).ToList();

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("tracked beats against the map's own grid, in the steady section:");
            TestContext.Out.WriteLine($"  median   {absolute[absolute.Count / 2],7:0.0}ms");
            TestContext.Out.WriteLine($"  90th     {absolute[(int)(absolute.Count * 0.9)],7:0.0}ms");
            TestContext.Out.WriteLine($"  worst    {absolute[^1],7:0.0}ms");
            TestContext.Out.WriteLine($"  within 30ms  {absolute.Count(a => a <= 30) * 100.0 / absolute.Count,5:0.0}%");
            TestContext.Out.WriteLine($"  within 60ms  {absolute.Count(a => a <= 60) * 100.0 / absolute.Count,5:0.0}%");
            TestContext.Out.WriteLine($"  beyond 60ms  {absolute.Count(a => a > 60) * 100.0 / absolute.Count,5:0.0}%");

            (double raw, double smoothed, int compared) = heldOutPrediction(measured);

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("predicting each beat from the ones before it, in the steady section:");
            TestContext.Out.WriteLine($"  the tracker's own gaps  {raw,7:0.0}ms median error");
            TestContext.Out.WriteLine($"  a fitted pulse          {smoothed,7:0.0}ms median error");
            TestContext.Out.WriteLine($"  over {compared} held-out beats");
            TestContext.Out.WriteLine(smoothed < raw
                ? "  -> the pulse is there to be found, and finding it would place beats better"
                : "  -> fitting a pulse does not predict the next beat better; the tracker is not jittering around one");

            // The measurement above says most beats are close and a few are far out, which is a tail rather than a
            // jitter. If that is right, the thing to try is not moving every beat onto a pulse but moving the beats
            // that are nowhere near one - so these are the same population measured after each of those passes.
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("the same tracked beats after a pass that only moves the ones far from the pulse:");

            foreach (double tolerance in new[] { 0.5, 0.35, 0.25, 0.15 })
            {
                List<double> snapped = snapOutliers(measured, tolerance);
                List<double> after = snapped.Select(b => nearest(reference, b)).Select(r => Math.Abs(r.Error)).OrderBy(e => e).ToList();

                TestContext.Out.WriteLine($"  beyond {tolerance * 100,3:0}% of a beat moved:  median {after[after.Count / 2],6:0.0}ms   "
                                          + $"90th {after[(int)(after.Count * 0.9)],6:0.0}ms   within 30ms {after.Count(a => a <= 30) * 100.0 / after.Count,5:0.0}%   "
                                          + $"beyond 60ms {after.Count(a => a > 60) * 100.0 / after.Count,5:0.0}%");
            }

            List<double> everything = snapOutliers(measured, 0);
            List<double> allAfter = everything.Select(b => nearest(reference, b)).Select(r => Math.Abs(r.Error)).OrderBy(e => e).ToList();

            TestContext.Out.WriteLine($"  every beat moved:              median {allAfter[allAfter.Count / 2],6:0.0}ms   "
                                      + $"90th {allAfter[(int)(allAfter.Count * 0.9)],6:0.0}ms   within 30ms {allAfter.Count(a => a <= 30) * 100.0 / allAfter.Count,5:0.0}%   "
                                      + $"beyond 60ms {allAfter.Count(a => a > 60) * 100.0 / allAfter.Count,5:0.0}%");
        }

        /// <summary>
        /// Moves only the beats that are nowhere near the pulse around them, and leaves the rest alone.
        /// </summary>
        /// <remarks>
        /// The pulse and its phase come from the beats either side, by least squares over a window - and a beat being
        /// moved is left out of the fit that moves it, so a misplaced beat cannot drag the pulse towards itself.
        ///
        /// The tolerance is a share of a beat: a beat less than that far from where the pulse says it should be is
        /// taken to be the tracker being precise, and a beat further than that is taken to be the tracker being wrong
        /// about where the beat is. At zero every beat moves, which is the smoothing pass the measurement says would
        /// throw away the two thirds of beats that are already within 30ms.
        /// </remarks>
        private static List<double> snapOutliers(List<double> beats, double tolerance)
        {
            const int window = 10;

            var result = new List<double>(beats);

            // The local beat period: the median gap over a wide window, which a single misplaced beat cannot move.
            for (int i = window; i < beats.Count - window; i++)
            {
                var gaps = new List<double>();

                for (int j = i - window; j < i + window; j++)
                    gaps.Add(beats[j + 1] - beats[j]);

                gaps.Sort();

                double period = gaps[gaps.Count / 2];

                if (period <= 0)
                    continue;

                // A line through the beats either side, excluding the one being placed.
                var neighbours = new List<double>();

                for (int j = i - window; j <= i + window; j++)
                {
                    if (j != i)
                        neighbours.Add(beats[j]);
                }

                double slope = 0;
                double intercept = 0;

                leastSquares(neighbours.ToArray(), out slope, out intercept);

                // The line is in beat numbers; this beat's number within its own window is i - (i - window) = window.
                double expected = intercept + (slope * window);

                if (Math.Abs(beats[i] - expected) > tolerance * period)
                    result[i] = expected;
            }

            return result;
        }

        /// <summary>
        /// How well a locally fitted pulse predicts a beat the fit has not seen, against the tracker's own gaps.
        /// </summary>
        /// <remarks>
        /// The fit is a line through the last few seconds of beats, which is a pulse and its phase. It is fitted to
        /// beats <em>before</em> the beat being predicted and asked about that beat alone, so a fit that has only
        /// learned the noise in its window fails, and the tracker's own neighbouring gaps are the thing to beat.
        /// </remarks>
        private static (double Raw, double Smoothed, int Compared) heldOutPrediction(List<double> beats)
        {
            const int window = 12;

            var raw = new List<double>();
            var smoothed = new List<double>();

            for (int i = window; i < beats.Count; i++)
            {
                double[] previous = beats.Skip(i - window).Take(window).ToArray();

                // A line through the window: time against beat number. Its slope is the pulse, its value at the next
                // beat number is where that beat should be.
                double slope = 0;
                double intercept = 0;

                leastSquares(previous, out slope, out intercept);

                double predicted = intercept + (slope * window);

                raw.Add(Math.Abs(previous[^1] + (previous[^1] - previous[^2]) - beats[i]));
                smoothed.Add(Math.Abs(predicted - beats[i]));
            }

            raw.Sort();
            smoothed.Sort();

            return (raw[raw.Count / 2], smoothed[smoothed.Count / 2], raw.Count);
        }

        /// <summary>A least-squares line through a window of beats, in milliseconds per beat.</summary>
        private static void leastSquares(double[] window, out double slope, out double intercept)
        {
            double sumX = 0;
            double sumY = 0;
            double sumXY = 0;
            double sumXX = 0;

            for (int i = 0; i < window.Length; i++)
            {
                sumX += i;
                sumY += window[i];
                sumXY += i * window[i];
                sumXX += i * i;
            }

            double n = window.Length;

            slope = ((n * sumXY) - (sumX * sumY)) / ((n * sumXX) - (sumX * sumX));
            intercept = (sumY - (slope * sumX)) / n;
        }

        /// <summary>The reference beat nearest a tracked one, and how far off it is.</summary>
        private static (double Beat, double Error) nearest(List<double> reference, double at)
        {
            int low = 0;
            int high = reference.Count - 1;
            int best = 0;

            while (low <= high)
            {
                int middle = (low + high) / 2;

                if (reference[middle] <= at)
                {
                    best = middle;
                    low = middle + 1;
                }
                else
                {
                    high = middle - 1;
                }
            }

            if (best + 1 < reference.Count && Math.Abs(reference[best + 1] - at) < Math.Abs(reference[best] - at))
                best++;

            return (reference[best], at - reference[best]);
        }

        private static List<double> analyse(string audio, string modelPath)
        {
            string cache = Path.Combine(Path.GetTempPath(), "paratactus-placement-probe");
            var provider = new BeatGridProvider(modelPath, cache, BassAudioDecoder.Default);

            var watch = System.Diagnostics.Stopwatch.StartNew();
            BeatGrid grid = provider.Get(audio);
            watch.Stop();

            TestContext.Out.WriteLine($"analysed in {watch.Elapsed.TotalSeconds:0.#}s "
                                      + $"({grid.Beats.Count} beats, {grid.BpmAt(0):0.#} BPM at the start)");

            return grid.Beats.ToList();
        }

        /// <summary>The beatmap's own tempo declarations, as committed beside this test.</summary>
        private static List<TimingPoint> timingPoints()
        {
            string file = Path.Combine(AppContext.BaseDirectory, "DesignantTimingPoints.csv");

            Assert.That(File.Exists(file), Is.True, $"{file} should be copied beside the test");

            var points = new List<TimingPoint>();

            foreach (string line in File.ReadLines(file))
            {
                if (line.StartsWith("#") || string.IsNullOrWhiteSpace(line))
                    continue;

                string[] fields = line.Split(',');

                if (fields.Length >= 2
                    && double.TryParse(fields[0], NumberStyles.Float, CultureInfo.InvariantCulture, out double time)
                    && double.TryParse(fields[1], NumberStyles.Float, CultureInfo.InvariantCulture, out double length))
                {
                    points.Add(new TimingPoint(time, length));
                }
            }

            return points;
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
