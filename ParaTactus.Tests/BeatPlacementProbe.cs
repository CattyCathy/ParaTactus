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

            // The reference grid, built the way the tempo agreement test builds it: each timing point's own tempo laid
            // out from that point until the next one.
            double[] grid = beatmapGrid(timing);

            List<double> raw = analyse(audio, modelPath, regularise: false);
            List<double> regularised = analyse(audio, modelPath, regularise: true);

            TestContext.Out.WriteLine($"timing points: {timing.Count}; the grid is {grid.Length} beats over "
                                      + $"{grid[^1] / 1000:0.#}s");
            TestContext.Out.WriteLine($"the model's own beats: {raw.Count}, after the regulariser: {regularised.Count}");

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("how far each tracked beat is from the beatmap's grid, and how many of the");
            TestContext.Out.WriteLine("beatmap's beats have a tracked beat near them - the second is what a listener sees");

            score("the model's own beats", raw, grid);
            score("after the regulariser", regularised, grid);

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("the same, per section, because the failure is local and the sections differ:");

            foreach ((double from, double to) in new[]
                     {
                         (0.0, 82_000.0), (82_000.0, 90_000.0), (90_000.0, 110_000.0), (110_000.0, 140_000.0),
                         (140_000.0, 178_000.0),
                     })
            {
                string name = describeSection(timing, from, to);
                double[] sectionGrid = within(grid, from, to);

                TestContext.Out.WriteLine($"");
                TestContext.Out.WriteLine($"  {from / 1000:0}s to {to / 1000:0}s - {name}");
                TestContext.Out.WriteLine($"      the map has {sectionGrid.Length} beats in this section");

                // The two populations are not subsets of each other: the regulariser invents beats where it believes
                // some are missing and drops beats it believes are subdivisions, so neither count says anything about
                // the other. Both are measured against the same section of the map.
                score("    model", slice(raw, from, to), sectionGrid);
                score("    regularised", slice(regularised, from, to), sectionGrid);
            }

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("the model's beats against the map's, over twelve seconds of a ramp where the");
            TestContext.Out.WriteLine("tempo climbs: the shape of the error decides what can be done about it");
            dump(raw, grid, 110_000, 122_000);

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("and what the model believes at each of the map's own beats in that ramp, against");
            TestContext.Out.WriteLine("what it believes at the beats it chose: if there is a peak where the music's beat");
            TestContext.Out.WriteLine("is, the beats are there to be picked and the picking is what is wrong");
            activation(modelPath, audio, grid, 110_000, 120_000);

            (double rawPrediction, double smoothed, int compared) = heldOutPrediction(slice(raw, 0, steady_until_ms));

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("predicting each beat from the ones before it, in the steady section:");
            TestContext.Out.WriteLine($"  the tracker's own gaps  {rawPrediction,7:0.0}ms median error");
            TestContext.Out.WriteLine($"  a fitted pulse          {smoothed,7:0.0}ms median error");
            TestContext.Out.WriteLine($"  over {compared} held-out beats");
        }

        /// <summary>
        /// Two sequences side by side over a window, with each tracked beat's distance from the nearest map beat.
        /// </summary>
        /// <remarks>
        /// A table like this is what says which of the two failures it is. If every tracked beat sits the same distance
        /// from the map's, the tracker has the tempo and the wrong phase; if the distance grows and shrinks again, it
        /// has the phase and the wrong tempo; and if the gaps between tracked beats are not the gaps between the map's,
        /// it is pulsing at some other level entirely.
        /// </remarks>
        private static void dump(List<double> beats, double[] grid, double from, double to)
        {
            foreach (double beat in beats.Where(b => b >= from && b < to))
            {
                double distance = signedDistance(grid, beat);

                TestContext.Out.WriteLine($"    {beat,9:0}ms   map {beat + distance,9:0}ms   off by {distance,7:+0;-0;0}ms");
            }
        }

        /// <summary>
        /// The model's own belief, frame by frame, at the map's beats and at the beats the model chose.
        /// </summary>
        /// <remarks>
        /// The distinction that decides everything about how to fix this. If the model is confident at the place the
        /// music's beat is, then the beats are visible to it and the peak picking is what is losing them - which is a
        /// problem with a known shape and a known answer. If it is confident somewhere else, no amount of picking
        /// recovers a beat the model does not believe in, and the answer is elsewhere.
        /// </remarks>
        private static void activation(string modelPath, string audio, double[] grid, double from, double to)
        {
            float[] samples = BassAudioDecoder.DecodeMono(audio, AnalysisAudio.SampleRate);
            (float[] beats, float[] downbeats) = BeatThisBeatTracker.Logits(BeatThisBeatTracker.LogMelSpectrogram(samples), modelPath);

            double atMap = 0;
            int mapCount = 0;

            foreach (double beat in grid.Where(b => b >= from && b < to))
            {
                atMap += value(beats, beat);
                mapCount++;
            }

            TestContext.Out.WriteLine($"    at the map's {mapCount} beats: mean beat activation {atMap / mapCount:0.000}");

            // The same span, and the model's own choice of beat in it, which is what the peak picking produced.
            var chosen = new List<double>();

            for (double time = from; time < to; time += 10)
            {
                double best = 0;

                for (int frame = -3; frame <= 3; frame++)
                {
                    // A beat is a local maximum of the activation; taking the best of a small neighbourhood is what
                    // the peak picking does, and it is the fair comparison to the activation at the map's beat.
                    best = Math.Max(best, beats[Math.Clamp(frameIndex(time, beats.Length) + frame, 0, beats.Length - 1)]);
                }

                chosen.Add(best);
            }

            TestContext.Out.WriteLine($"    the best activation anywhere in the same span: {chosen.Max():0.000}");

            // The whole-track picture, so that the ramp can be compared with a passage that is known to work.
            double mean = 0;

            foreach (float logit in beats)
                mean += logit;

            TestContext.Out.WriteLine($"    mean activation over the whole track: {mean / beats.Length:0.000}");
        }

        private static int frameIndex(double milliseconds, int length)
        {
            int frame = (int)Math.Round(milliseconds / BeatThisBeatTracker.FrameToMilliseconds(1));

            return Math.Clamp(frame, 0, length - 1);
        }

        private static double value(float[] logits, double milliseconds)
        {
            int frame = frameIndex(milliseconds, logits.Length);

            double best = 0;

            for (int offset = -2; offset <= 2; offset++)
                best = Math.Max(best, logits[Math.Clamp(frame + offset, 0, logits.Length - 1)]);

            return best;
        }

        /// <summary>
        /// How well a set of tracked beats matches the map's own grid, both ways round.
        /// </summary>
        /// <remarks>
        /// Both directions, because they are different failures. How far a tracked beat is from the grid is the
        /// precision of the pulses the player draws; how many of the map's beats have a tracked beat within 60ms is
        /// whether a beat of the music goes by with nothing happening, which is the one a listener notices.
        /// </remarks>
        private static void score(string name, List<double> beats, double[] grid)
        {
            if (beats.Count == 0)
            {
                TestContext.Out.WriteLine($"  {name}: nothing");
                return;
            }

            var errors = beats.Select(b => Math.Abs(signedDistance(grid, b))).OrderBy(e => e).ToArray();

            int covered = grid.Count(g => beats.Any(b => Math.Abs(b - g) <= 60));

            TestContext.Out.WriteLine($"  {name,-26} n={beats.Count,4}  median {errors[errors.Length / 2],6:0.0}ms  "
                                      + $"p90 {errors[(int)(errors.Length * 0.9)],6:0.0}ms  "
                                      + $"within 60ms {covered * 100.0 / grid.Length,5:0.0}% of the map's beats");
        }

        private static string describeSection(List<TimingPoint> timing, double from, double to)
        {
            var inSection = timing.Where(p => p.Time > from && p.Time < to).ToList();

            if (inSection.Count == 0)
                return $"{60000 / tempoAt(timing, (from + to) / 2):0} BPM held";

            return $"{inSection.Count} tempo change(s), {60000 / tempoAt(timing, from + 1):0} to "
                   + $"{60000 / tempoAt(timing, to - 1):0} BPM";
        }

        private static double tempoAt(List<TimingPoint> timing, double time)
        {
            double length = 0;

            foreach (TimingPoint point in timing)
            {
                if (point.Time <= time)
                    length = point.BeatLength;
                else
                    break;
            }

            return length;
        }

        /// <summary>The map's own beats, from its own timing points.</summary>
        private static double[] beatmapGrid(List<TimingPoint> timing)
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

        /// <summary>The beats of one section, from either kind of sequence.</summary>
        private static List<double> slice(IEnumerable<double> beats, double from, double to)
            => beats.Where(b => b >= from && b < to).ToList();

        private static double[] within(double[] beats, double from, double to)
            => beats.Where(b => b >= from && b < to).ToArray();

        /// <summary>How far a time is from the nearest line of a sorted grid, signed.</summary>
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

        /// <summary>The model's beats, optionally through the regulariser.</summary>
        private static List<double> analyse(string audio, string modelPath, bool regularise)
        {
            string cache = Path.Combine(Path.GetTempPath(), "paratactus-placement-probe");
            var provider = new BeatGridProvider(modelPath, cache, BassAudioDecoder.Default);

            var watch = System.Diagnostics.Stopwatch.StartNew();
            BeatGrid grid = provider.Get(audio);
            watch.Stop();

            double[] beats = regularise
                ? BeatTrainRegulariser.Regularise(BeatThisBeatTracker.BeatTimes(audio, modelPath, BassAudioDecoder.Default))
                : BeatThisBeatTracker.BeatTimes(audio, modelPath, BassAudioDecoder.Default);

            TestContext.Out.WriteLine($"analysed in {watch.Elapsed.TotalSeconds:0.#}s "
                                      + $"({grid.Beats.Count} beats in the grid, {grid.BpmAt(0):0.#} BPM at the start)");

            return beats.ToList();
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
