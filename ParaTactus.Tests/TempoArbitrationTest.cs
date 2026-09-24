using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NUnit.Framework;
using ParaTactus;
using ParaTactus.Decoding;

namespace ParaTactus.Tests
{
    /// <summary>
    /// Asks the audio, and the model's own activation, what a passage the model disagrees with the beatmap about is
    /// really doing.
    /// </summary>
    /// <remarks>
    /// The model can be confidently wrong, and when it is there is nothing in its own output that reveals it. Two
    /// things can, and they answer different questions. The onset envelope is a measurement of the sound, so it can say
    /// what period the onsets line up on. The activation says whether the model is certain there is no beat where the
    /// beatmap has one, which is the difference between a passage a tempo prior could repair and one it could not.
    ///
    /// The comb scan is deliberately naive - a grid of candidate periods, each with its best phase, scored by the mean
    /// envelope value at its own instants. It is not a tempo estimator and is not meant to become one. It is biased
    /// towards long periods in a short window, because a grid with four points scores the four highest onsets, so the
    /// per-period scores are only comparable against each other and the whole list is printed rather than a winner.
    /// </remarks>
    [TestFixture]
    public class TempoArbitrationTest
    {
        [OneTimeSetUp]
        public void SetUp()
        {
            AudioTestEnvironment.Initialise();
        }

        [Test]
        [Explicit("diagnostic: needs OSUTEST_AUDIO, the model file and OSUTEST_FROM/OSUTEST_TO")]
        public void TestOnsetTempoScan()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            double from = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_FROM"), out double f) ? f : 144400;
            double to = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_TO"), out double t) ? t : 150000;

            float[] samples = BassAudioDecoder.DecodeMono(audio, BassAudioDecoder.ANALYSIS_SAMPLE_RATE);
            double[] envelope = OnsetEnvelope.FromSamples(samples, out int frames);
            var (logits, _) = BeatThisBeatTracker.Logits(BeatThisBeatTracker.LogMelSpectrogram(samples), model);

            TestContext.Out.WriteLine($"[{from:0},{to:0}] ms, envelope {frames} frames, mean 1 over the whole track");
            TestContext.Out.WriteLine();

            TestContext.Out.WriteLine(" period   phase   score  count");

            for (double period = 150; period <= 500; period += 5)
            {
                double best = -1;
                double bestPhase = 0;
                int bestCount = 0;

                for (double phase = 0; phase < period; phase += 5)
                {
                    double sum = 0;
                    int count = 0;

                    for (double time = phase; time <= to; time += period)
                    {
                        if (time < from)
                            continue;

                        sum += strength(envelope, frames, time);
                        count++;
                    }

                    if (count < 4)
                        continue;

                    double score = sum / count;

                    if (score > best)
                    {
                        best = score;
                        bestPhase = phase;
                        bestCount = count;
                    }
                }

                if (best >= 0)
                    TestContext.Out.WriteLine($"{period,7:0} {bestPhase,7:0} {best,7:0.000} {bestCount,6}");
            }

            TestContext.Out.WriteLine();

            // The beatmap's grid at 200 BPM, whose phase is 409ms modulo the beat length.
            TestContext.Out.WriteLine("  beatmap grid   onset   logit   |   model beat   onset   logit   offset");
            TestContext.Out.WriteLine();

            var grid = new List<double>();

            for (double time = 409; time <= to; time += 300)
            {
                if (time >= from)
                    grid.Add(time);
            }

            var beats = BeatThisBeatTracker.BeatTimes(samples, model).Where(b => b >= from && b <= to).ToArray();

            for (int i = 0; i < Math.Max(grid.Count, beats.Length); i++)
            {
                string map = i < grid.Count
                    ? $"{grid[i] / 1000,14:0.000} {strength(envelope, frames, grid[i]),7:0.000} {logitAt(logits, grid[i]),7:0.000}"
                    : new string(' ', 30);

                string beat = i < beats.Length
                    ? $"   | {beats[i] / 1000,10:0.000} {strength(envelope, frames, beats[i]),7:0.000} {logitAt(logits, beats[i]),7:0.000} {signedOffset(beats[i], 409, 300),7:0}"
                    : string.Empty;

                TestContext.Out.WriteLine(map + beat);
            }

            TestContext.Out.WriteLine();

            double[] gridLogits = grid.Select(g => logitAt(logits, g)).ToArray();
            double[] beatLogits = beats.Select(b => logitAt(logits, b)).ToArray();

            TestContext.Out.WriteLine($"beatmap grid: {grid.Count} instants, mean onset {grid.Average(g => strength(envelope, frames, g)):0.000}, "
                                      + $"mean logit {gridLogits.Average():0.000}, above zero {gridLogits.Count(v => v > 0)}");
            TestContext.Out.WriteLine($"model beats:  {beats.Length} beats, mean onset {beats.Average(b => strength(envelope, frames, b)):0.000}, "
                                      + $"mean logit {beatLogits.Average():0.000}, median gap {medianGap(beats):0}ms");
        }

        /// <summary>
        /// Everywhere the envelope's own periodicity disagrees with the period the model's beats are actually at.
        /// </summary>
        /// <remarks>
        /// The comb scan in the other test can only arbitrate a passage somebody already suspects, and it is biased
        /// towards long periods. This asks the same question everywhere instead, with an estimator that does not have
        /// that bias: the normalised autocorrelation of the onset envelope over a sliding window, whose peaks are the
        /// envelope's own periodicities. Each lag is divided by the number of pairs it had, so a long lag is not
        /// penalised for overlapping less of the window.
        ///
        /// What it is for is deciding whether a correction is safe. A passage where the two disagree can be repaired
        /// by trusting the envelope; if they disagree all over the track then the envelope is not a second opinion, and
        /// anything built on it would be worse than the model it corrects.
        /// </remarks>
        [Test]
        [Explicit("diagnostic: needs OSUTEST_AUDIO, the model file and OSUTEST_FROM/OSUTEST_TO")]
        public void TestEnvelopePeriodVersusModel()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            double from = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_FROM"), out double f) ? f : 0;
            double to = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_TO"), out double t) ? t : 1e9;

            float[] samples = BassAudioDecoder.DecodeMono(audio, BassAudioDecoder.ANALYSIS_SAMPLE_RATE);
            double[] envelope = OnsetEnvelope.FromSamples(samples, out int frames);
            var (logits, _) = BeatThisBeatTracker.Logits(BeatThisBeatTracker.LogMelSpectrogram(samples), model);
            var beats = BeatThisBeatTracker.BeatTimes(samples, model);

            TestContext.Out.WriteLine($"{beats.Length} beats, envelope {frames} frames");
            TestContext.Out.WriteLine();
            TestContext.Out.WriteLine("    time  model |  peak 1        peak 2        peak 3");

            const int window = 200;
            const int step = 50;
            const int minimumLag = 8;
            const int maximumLag = 60;

            for (int centre = window / 2; centre + window / 2 < frames; centre += step)
            {
                double time = centre * 20.0;

                if (time < from || time > to)
                    continue;

                double[]? peaks = periodicity(envelope, centre, window, minimumLag, maximumLag);

                var local = new List<double>();

                foreach (double beat in beats)
                {
                    if (beat >= time - 500 && beat <= time + 500)
                        local.Add(beat);
                }

                local.Sort();

                double modelPeriod = medianGap(local.ToArray());

                string text = peaks == null
                    ? "  none"
                    : string.Join("  ", peaks.Select(p => $"{p,6:0}ms/{autocorrelation(envelope, centre, window, (int)Math.Round(p / 20)):0.00}"));

                // The phase the envelope itself puts a 300ms grid on, against the beatmap's own phase of 109ms, because
                // the model's beats cannot anchor it: in the passages this is for they sit on the half beat. The
                // model's own activation is scanned the same way, since it is the sharper of the two and the point of
                // the comparison is whether the envelope's phase is usable at all.
                int frame = centre;
                double bestPhase = 0;
                double bestScore = -1;
                double bestLogitPhase = 0;
                double bestLogit = double.MinValue;

                for (double phase = 0; phase < 300; phase += 5)
                {
                    double sum = 0;
                    double logitSum = 0;
                    int count = 0;

                    for (double at = frame * 20.0 - 1000 + phase; at <= frame * 20.0 + 1000; at += 300)
                    {
                        sum += strength(envelope, frames, at);
                        logitSum += logitAt(logits, at);
                        count++;
                    }

                    if (count == 0)
                        continue;

                    if (sum / count > bestScore)
                    {
                        bestScore = sum / count;
                        bestPhase = ((phase - 109) % 300 + 450) % 300 - 150;
                    }

                    if (logitSum / count > bestLogit)
                    {
                        bestLogit = logitSum / count;
                        bestLogitPhase = ((phase - 109) % 300 + 450) % 300 - 150;
                    }
                }

                TestContext.Out.WriteLine($"{time / 1000,7:0.0} {modelPeriod,6:0} |  {text}  | onset {bestPhase,5:0} {bestScore:0.00} | logit {bestLogitPhase,5:0} {bestLogit:0.00}");
            }
        }

        /// <summary>
        /// The strongest few lags the envelope repeats at inside a window, as periods in milliseconds.
        /// </summary>
        /// <remarks>
        /// Local maxima only, because the autocorrelation of a periodic signal is itself periodic: the neighbouring
        /// lags of a true peak are all high, so taking the largest value would report the same period repeatedly and
        /// hide the alternatives that decide between a period and a multiple of it.
        /// </remarks>
        private static double[]? periodicity(double[] envelope, int centre, int window, int minimumLag, int maximumLag)
        {
            var candidates = new List<(double Value, int Lag)>();

            for (int lag = minimumLag; lag <= maximumLag; lag++)
            {
                double value = autocorrelation(envelope, centre, window, lag);

                if (value <= 0)
                    continue;

                if (value < autocorrelation(envelope, centre, window, lag - 1) || value < autocorrelation(envelope, centre, window, lag + 1))
                    continue;

                candidates.Add((value, lag));
            }

            if (candidates.Count == 0)
                return null;

            return candidates.OrderByDescending(c => c.Value).Take(3).Select(c => c.Lag * 20.0).ToArray();
        }

        private static double autocorrelation(double[] envelope, int centre, int window, int lag)
        {
            int start = Math.Max(centre - window / 2, 0);
            int end = Math.Min(centre + window / 2, envelope.Length - 1 - lag);

            if (end <= start)
                return 0;

            double mean = 0;
            int count = end - start;

            for (int t = start; t < end; t++)
                mean += envelope[t];

            mean /= count;

            double covariance = 0;
            double variance = 0;

            for (int t = start; t < end; t++)
            {
                covariance += (envelope[t] - mean) * (envelope[t + lag] - mean);
                variance += (envelope[t] - mean) * (envelope[t] - mean);
            }

            return variance > 0 ? covariance / variance : 0;
        }

        /// <summary>
        /// How the model's beats in a window are distributed against the beatmap's own grid.
        /// </summary>
        /// <remarks>
        /// Whether a passage is unstable because the beats are scattered around the right positions or because they sit
        /// on a consistent wrong spacing decides what can be done about it. Scattered beats can be filtered by how far
        /// they are from the grid; a consistently short spacing cannot, because every one of those beats is nearer the
        /// neighbouring grid position than it is to being an error. A histogram tells the two apart and a median gap
        /// does not.
        /// </remarks>
        [Test]
        [Explicit("diagnostic: needs OSUTEST_AUDIO, the model file and OSUTEST_FROM/OSUTEST_TO")]
        public void TestBeatResidueHistogram()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            double from = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_FROM"), out double f) ? f : 36000;
            double to = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_TO"), out double t) ? t : 44000;

            float[] samples = BassAudioDecoder.DecodeMono(audio, BassAudioDecoder.ANALYSIS_SAMPLE_RATE);
            var beats = BeatThisBeatTracker.BeatTimes(samples, model).Where(b => b >= from && b <= to).ToArray();

            // The beatmap's grid in its 200 BPM section: one timing point at 409ms with a beat length of 300.
            var residues = beats.Select(b => ((b - 109) % 300 + 450) % 300 - 150).ToArray();

            TestContext.Out.WriteLine($"[{from:0},{to:0}] ms, {beats.Length} beats, median gap {medianGap(beats):0}ms");
            TestContext.Out.WriteLine();

            for (int bin = 0; bin < 12; bin++)
            {
                double low = -150 + bin * 25;
                int count = residues.Count(r => r >= low && r < low + 25);

                TestContext.Out.WriteLine($"{low,6:0} .. {low + 25,4:0}  {new string('#', count),-40} {count}");
            }

            TestContext.Out.WriteLine();
            TestContext.Out.WriteLine($"within 40ms of the grid: {residues.Count(r => Math.Abs(r) <= 40)} of {residues.Length}");
            TestContext.Out.WriteLine($"within 40ms of a half beat: {residues.Count(r => Math.Abs(Math.Abs(r) - 150) <= 40)} of {residues.Length}");
            TestContext.Out.WriteLine("gaps: " + string.Join(", ", beats.Skip(1).Select((b, i) => (b - beats[i]).ToString("0"))));

            // Whether the beats that are on the grid are also the ones the audio is loudest on, because that is the only
            // thing that could decide the phase when a majority of the model's beats are somewhere else.
            double[] envelope = OnsetEnvelope.FromSamples(samples, out int frames);

            TestContext.Out.WriteLine();
            TestContext.Out.WriteLine(" offset   onset   | offset   onset");

            var ordered = beats.Select((b, i) => (Offset: residues[i], Onset: OnsetEnvelope.Compare(new[] { b - 1, b, b + 1, b + 2 }, envelope, frames).AtBeats)).ToArray();

            for (int i = 0; i < ordered.Length; i += 2)
            {
                string left = $"{ordered[i].Offset,7:0} {ordered[i].Onset,7:0.00}";

                string right = i + 1 < ordered.Length ? $"   | {ordered[i + 1].Offset,6:0} {ordered[i + 1].Onset,7:0.00}" : string.Empty;

                TestContext.Out.WriteLine(left + right);
            }

            double onGrid = ordered.Where(o => Math.Abs(o.Offset) <= 60).Select(o => o.Onset).DefaultIfEmpty(0).Average();
            double offGrid = ordered.Where(o => Math.Abs(o.Offset) > 60).Select(o => o.Onset).DefaultIfEmpty(0).Average();

            TestContext.Out.WriteLine();
            TestContext.Out.WriteLine($"mean onset on the grid {onGrid:0.00}, off it {offGrid:0.00}");
        }

        /// <summary>
        /// What snapping the misplaced beats onto the passage's own phase does to the finished grid.
        /// </summary>
        /// <remarks>
        /// Measured against the same track with and without, because the only thing that justifies a stage that moves
        /// beats is that the grid it produces is more regular than the one the model and the existing filler produce
        /// without it. The beatmap's 200 BPM section is the case it exists for, so the window is usually 30-70s.
        /// </remarks>
        [Test]
        [Explicit("diagnostic: needs OSUTEST_AUDIO and the model file")]
        public void TestSnapEffect()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            double from = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_FROM"), out double f) ? f : 30000;
            double to = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_TO"), out double t) ? t : 70000;

            float[] samples = BassAudioDecoder.DecodeMono(audio, BassAudioDecoder.ANALYSIS_SAMPLE_RATE);
            double[] envelope = OnsetEnvelope.FromSamples(samples, out int frames);
            double[] beats = BeatThisBeatTracker.BeatTimes(samples, model);

            double[] snapped = OnsetPeriodArbiter.Snap(beats, envelope, frames);
            double[] loose = OnsetPeriodArbiter.Snap(beats, envelope, frames, 0.05, 0.5, 4);
            double[] plain = BeatTrainRegulariser.Regularise(beats);
            double[] withSnap = BeatTrainRegulariser.Regularise(snapped);
            double[] withLoose = BeatTrainRegulariser.Regularise(loose);

            int moved = 0;
            double movedBy = 0;

            for (int i = 0; i < beats.Length; i++)
            {
                if (Math.Abs(snapped[i] - beats[i]) > 0.5)
                {
                    moved++;
                    movedBy += Math.Abs(snapped[i] - beats[i]);
                }
            }

            TestContext.Out.WriteLine($"{beats.Length} tracked beats, {moved} snapped"
                                      + (moved > 0 ? $", by {movedBy / moved:0}ms on average, {movedBy:0}ms in total" : string.Empty));
            TestContext.Out.WriteLine();

            foreach (var (name, grid) in new[] { ("plain", plain), ("snapped", withSnap), ("loose", withLoose) })
            {
                var gaps = grid.Skip(1).Select((b, i) => b - grid[i]).Where(g => g > 0).ToArray();
                var inside = gaps.Where(g => grid.Length > 0 && g > 0).ToArray();
                var deviations = inside.Select(g => Math.Abs(g - 300)).OrderBy(v => v).ToArray();

                var window = new List<double>();

                for (int i = 1; i < grid.Length; i++)
                {
                    if (grid[i] >= from && grid[i] <= to)
                        window.Add(grid[i] - grid[i - 1]);
                }

                var within = window.Where(g => Math.Abs(g - 300) <= 30).Count();

                TestContext.Out.WriteLine($"{name,8}: {grid.Length} beats, "
                                          + $"mean |gap-300| {deviations.Average():0.0}ms, p90 {deviations[(int)(deviations.Length * 0.9)]:0}ms, "
                                          + $"within 30ms of 300 {within * 100.0 / Math.Max(window.Count, 1):0}%");
            }

            TestContext.Out.WriteLine();
            TestContext.Out.WriteLine("    time      plain            |    snapped");

            for (double time = from; time < to; time += 1000)
            {
                TestContext.Out.WriteLine($"{time / 1000,7:0.0} {describe(plain, time),-18} | {describe(withSnap, time)}");
            }
        }

        private static string describe(double[] grid, double time)
        {
            var gaps = new List<double>();

            for (int i = 1; i < grid.Length; i++)
            {
                if (grid[i] >= time && grid[i] < time + 1000)
                    gaps.Add(grid[i] - grid[i - 1]);
            }

            if (gaps.Count == 0)
                return "-";

            gaps.Sort();

            return $"({string.Join(" ", gaps.Select(g => g.ToString("0")))})";
        }

        private static double logitAt(float[] logits, double milliseconds)
        {
            int frame = (int)Math.Round(milliseconds / 20.0);

            if (frame < 0 || frame >= logits.Length)
                return 0;

            double best = logits[frame];

            if (frame > 0)
                best = Math.Max(best, logits[frame - 1]);

            if (frame + 1 < logits.Length)
                best = Math.Max(best, logits[frame + 1]);

            return best;
        }

        private static double signedOffset(double time, double phase, double period)
        {
            double position = (time - phase) / period;

            return (position - Math.Round(position)) * period;
        }

        private static double medianGap(double[] beats)
        {
            if (beats.Length < 2)
                return 0;

            var gaps = new List<double>();

            for (int i = 1; i < beats.Length; i++)
                gaps.Add(beats[i] - beats[i - 1]);

            gaps.Sort();

            return gaps[gaps.Count / 2];
        }

        private static double strength(double[] envelope, int frames, double milliseconds)
        {
            int frame = (int)Math.Round(milliseconds / 20.0);

            if (frame < 0 || frame >= frames)
                return 0;

            double best = envelope[frame];

            if (frame > 0)
                best = Math.Max(best, envelope[frame - 1]);

            if (frame + 1 < frames)
                best = Math.Max(best, envelope[frame + 1]);

            return best;
        }
    }
}
