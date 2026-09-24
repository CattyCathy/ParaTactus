using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ManagedBass;
using NUnit.Framework;
using ParaTactus;
using ParaTactus.Decoding;

namespace ParaTactus.Tests
{
    /// <summary>
    /// The reported tempo has to agree with the tempo the beatmap itself declares, all the way through the track.
    /// </summary>
    /// <remarks>
    /// This is the test the whole tempo problem should have been written against from the start, and the absence of
    /// something like it is why so much of the work before it went wrong. Every earlier check compared the analyser
    /// with either a synthetic fixture or with another part of the analyser, and the two are not the same thing: a
    /// fixture of a click track at a steady tempo cannot tell a system that only handles steady tempi from one that
    /// handles music, and a second copy of the same estimator shares every systematic bias the first one has, so
    /// agreeing with it proves nothing at all.
    ///
    /// The reference here is a beatmap's own <c>[TimingPoints]</c> list - the tempo the map's author set, which is the
    /// one thing about a track's tempo that is not an estimate. On the track this was written against it is a hard
    /// case by construction: 200 BPM for most of its length, with continuous ramps down to 90 and up to 400.
    ///
    /// It is <c>[Explicit]</c> because the audio cannot be committed, so it is not part of the standing suite. It is
    /// still the specification: until it passes, the tempo reporting is not fixed, whatever else is green.
    ///
    /// It currently fails at 2 of its 21 sampled points, down from 7 before the frontend was corrected: at 45s it reads
    /// 130 where the map says 200, and at 155s it reads 273 where the map says 200. Both are whole-beat errors - the
    /// model found or missed a beat rather than misplacing one by a frame - and both are in passages dense enough that
    /// the published postprocessor produces the same thing, so they are not something the frontend correction fixes.
    /// See the note on <c>tolerance</c> for why the other 19 are not errors at all.
    /// </remarks>
    [TestFixture]
    public class BeatmapTempoAgreementTest
    {
        [OneTimeSetUp]
        public void SetUp()
        {
            AudioTestEnvironment.Initialise();
        }

        /// <summary>
        /// How far the reported tempo may be from the beatmap's, as a fraction, at any sampled point.
        /// </summary>
        /// <remarks>
        /// This cannot be as tight as it looks like it should be, and the reason is the model's own time resolution
        /// rather than anyone's tolerance for error. The model reports whole frames at 50 frames per second, so its
        /// beats are quantised to 20ms while the beatmap's are not. At 200 BPM a beat is 300ms, and one frame either
        /// side of that is 280ms or 320ms - a tempo of 214 or 187.5, which is 0.10 octaves away from 200. A window
        /// median lands on one of those whenever the model's peaks jitter by a single frame, which they routinely do.
        ///
        /// The original 0.02 was therefore not measuring the model but the frame grid: it demanded agreement to within
        /// 0.4 of a frame. At 0.10 the test still fails on the things worth failing on - the remaining points are 0.45
        /// and 0.62 octaves out, which are whole beats found or missed rather than rounding.
        /// </remarks>
        private const double tolerance = 0.10;

        /// <summary>How often the track is sampled. A change shorter than this is not what this test is about.</summary>
        private const double sample_interval_ms = 5000;

        /// <summary>
        /// How far a sample has to be from the nearest timing point for its beatmap tempo to count as steady.
        /// </summary>
        /// <remarks>
        /// A model tempo is read here as the median interval over a five-second window, so within 2.5s of a timing
        /// point the window spans two different tempi and the comparison says nothing about either.
        /// </remarks>
        private const double stable_margin_ms = 3000;

        [Test]
        [Explicit("needs OSUTEST_AUDIO and the model file")]
        public void TestModelBeatsMatchBeatmap()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            var truth = loadTimingPoints();
            var beats = BeatThisBeatTracker.BeatTimes(audio, model, BassAudioDecoder.Default);

            TestContext.Out.WriteLine($"model produced {beats.Length} beats over {beats[^1] / 1000:0.#}s");

            // The tempo the model implies, at each point the beatmap declares one: the median of the intervals between
            // the beats around that time. Median rather than mean because a single missed or added beat should not move
            // the answer.
            //
            // Two things have to be true of a sample for it to mean anything. It has to sit inside a beatmap section
            // that holds one tempo well clear of its edges, because a median over a five-second window straddles
            // several timing points of a ramp and a disagreement there measures the window rather than the model. And
            // the comparison has to allow a whole octave, because the model and the map are entitled to describe the
            // same music at different metrical levels - on this track they do exactly that, at 18 of 35 points equal
            // and 16 at exactly half.
            var failures = new List<string>();
            var ratios = new List<double>();
            var levels = new Dictionary<int, int>();
            int sampled = 0;

            for (double time = sample_interval_ms; time < beats[^1] - 1000; time += sample_interval_ms)
            {
                double expected = tempoAt(truth, time);

                if (expected <= 0 || !InStableSection(truth, time, stable_margin_ms))
                    continue;

                var nearby = beats.Where(b => Math.Abs(b - time) <= 5000).OrderBy(b => b).ToArray();

                if (nearby.Length < 4)
                    continue;

                var intervals = new double[nearby.Length - 1];

                for (int i = 1; i < nearby.Length; i++)
                    intervals[i - 1] = nearby[i] - nearby[i - 1];

                Array.Sort(intervals);
                double implied = 60000 / intervals[intervals.Length / 2];
                double ratio = implied / expected;

                ratios.Add(ratio);
                sampled++;

                double octaves = Math.Log2(ratio);
                int level = (int)Math.Round(octaves);

                levels[level] = levels.TryGetValue(level, out int seen) ? seen + 1 : 1;

                if (Math.Abs(octaves - level) > tolerance)
                    failures.Add($"{time / 1000,7:0.0}s  beatmap {expected,7:0.#}  model {implied,7:0.#}  ({ratio:0.00}x, {octaves:0.00} octaves)");
            }

            TestContext.Out.WriteLine($"sampled {sampled} points in steady sections, {failures.Count} not within {tolerance:0.###} of a metrical level");

            if (ratios.Count > 0)
            {
                var sorted = ratios.OrderBy(r => r).ToArray();
                TestContext.Out.WriteLine($"ratio: min {sorted[0]:0.00} median {sorted[sorted.Length / 2]:0.00} max {sorted[^1]:0.00}");

                foreach (var pair in levels.OrderBy(p => p.Key))
                    TestContext.Out.WriteLine($"  model is {Math.Pow(2, pair.Key):0.###}x the beatmap level at {pair.Value} of {ratios.Count} points");
            }

            if (failures.Count > 0)
            {
                TestContext.Out.WriteLine();
                TestContext.Out.Write(string.Join(Environment.NewLine, failures.Take(15)));
            }

            Assert.That(failures, Is.Empty, $"{failures.Count} of {ratios.Count} points are not at a metrical level of the beatmap");
        }

        /// <summary>
        /// Prints the raw beats the model reports in a time range, as intervals and as a histogram of implied tempi.
        /// </summary>
        /// <remarks>
        /// A tempo derived from a median interval hides what actually happened: half tempo and "the model only found
        /// every other beat" look identical in the ratio, but only the second one is a bug in the frontend. The
        /// histogram separates them, because a genuinely halved reading still has a tight distribution.
        /// </remarks>
        [Test]
        [Explicit("diagnostic: needs OSUTEST_AUDIO, the model file and OSUTEST_FROM/OSUTEST_TO")]
        public void TestModelBeatDump()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            double from = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_FROM"), out double f) ? f : 0;
            double to = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_TO"), out double t) ? t : 10000;

            var beats = BeatThisBeatTracker.BeatTimes(audio, model, BassAudioDecoder.Default);
            var inside = beats.Where(b => b >= from && b <= to).ToArray();

            TestContext.Out.WriteLine($"{beats.Length} beats overall, {inside.Length} in [{from:0},{to:0}] ms");

            var intervals = new List<double>();

            for (int i = 1; i < inside.Length; i++)
                intervals.Add(inside[i] - inside[i - 1]);

            TestContext.Out.WriteLine("intervals: " + string.Join(", ", intervals.Select(v => v.ToString("0"))));

            foreach (var group in intervals.GroupBy(v => Math.Round(v / 25) * 25).OrderBy(g => g.Key))
                TestContext.Out.WriteLine($"  {group.Key,6:0} ms  ({60000 / group.Key,6:0.#} BPM)  x{group.Count()}");
        }

        /// <summary>
        /// The tempo the beatmap declares at a time, in BPM.
        /// </summary>
        private static double tempoAt(IReadOnlyList<(double Time, double BeatLength)> timingPoints, double time)
        {
            double beatLength = 0;

            foreach (var point in timingPoints)
            {
                if (point.Time <= time)
                    beatLength = point.BeatLength;
                else
                    break;
            }

            return beatLength > 0 ? 60000 / beatLength : 0;
        }

        /// <summary>
        /// Whether a time sits far enough inside one timing point's section that the beatmap holds a single tempo all
        /// the way through a window centred on it.
        /// </summary>
        private static bool InStableSection(IReadOnlyList<(double Time, double BeatLength)> timingPoints, double time, double margin)
        {
            int index = -1;

            for (int i = 0; i < timingPoints.Count; i++)
            {
                if (timingPoints[i].Time <= time)
                    index = i;
                else
                    break;
            }

            if (index < 0)
                return false;

            double end = index + 1 < timingPoints.Count ? timingPoints[index + 1].Time : double.MaxValue;

            return time - timingPoints[index].Time >= margin && end - time >= margin;
        }

        private static List<(double Time, double BeatLength)> loadTimingPoints()
        {
            string path = Path.Combine(AppContext.BaseDirectory, "DesignantTimingPoints.csv");

            Assert.That(File.Exists(path), Is.True, $"timing points not found at {path}");

            var points = new List<(double, double)>();

            foreach (string line in File.ReadAllLines(path))
            {
                if (line.StartsWith('#') || !line.Contains(','))
                    continue;

                var fields = line.Split(',');

                if (fields.Length == 2 && double.TryParse(fields[0], out double time) && double.TryParse(fields[1], out double beatLength))
                    points.Add((time, beatLength));
            }

            return points.OrderBy(p => p.Item1).ToList();
        }

        /// <summary>
        /// How strongly the model believes in the beats it did not report, at the points halfway between the ones it did.
        /// </summary>
        /// <remarks>
        /// This decides how the half-tempo problem gets fixed, and it is not guessable from the beat times. If the model
        /// has a real peak at the midpoint, the beats are there and the peak-picking is discarding them, and the fix is
        /// in the picking. If the midpoint is flat or negative, the model genuinely believes the slow pulse and no
        /// amount of better picking will help - the level has to be chosen against the audio instead.
        /// </remarks>
        [Test]
        [Explicit("diagnostic: needs OSUTEST_AUDIO and the model file")]
        public void TestActivationAtBeatMidpoints()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            float[] samples = BassAudioDecoder.DecodeMono(audio, BassAudioDecoder.ANALYSIS_SAMPLE_RATE);
            var (logits, _) = BeatThisBeatTracker.Logits(BeatThisBeatTracker.LogMelSpectrogram(samples), model);
            var beats = BeatThisBeatTracker.Peaks(logits);

            double from = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_FROM"), out double f) ? f : 1000;
            double to = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_TO"), out double t) ? t : 80000;

            int firstFrame = (int)(from * 50 / 1000);
            int lastFrame = (int)(to * 50 / 1000);
            var inside = beats.Where(b => b >= firstFrame && b <= lastFrame).ToArray();

            var atBeats = new List<double>();
            var atMidpoints = new List<double>();
            var nearMidpoints = new List<double>();
            int aboveZero = 0;

            for (int i = 1; i < inside.Length; i++)
            {
                atBeats.Add(logits[inside[i]]);

                int middle = (inside[i - 1] + inside[i]) / 2;
                atMidpoints.Add(logits[middle]);

                double best = double.MinValue;

                for (int j = Math.Max(0, middle - 3); j <= Math.Min(logits.Length - 1, middle + 3); j++)
                    best = Math.Max(best, logits[j]);

                nearMidpoints.Add(best);

                if (logits[middle] > 0)
                    aboveZero++;
            }

            string describe(List<double> values) => values.Count == 0
                ? "n/a"
                : $"mean {values.Average():0.000}  min {values.Min():0.000}  max {values.Max():0.000}";

            TestContext.Out.WriteLine($"reported beats in [{from / 1000:0}s, {to / 1000:0}s]: {inside.Length}");
            TestContext.Out.WriteLine($"activation at beats:      {describe(atBeats)}");
            TestContext.Out.WriteLine($"activation at midpoints:  {describe(atMidpoints)}");
            TestContext.Out.WriteLine($"best within 3 frames:     {describe(nearMidpoints)}");
            TestContext.Out.WriteLine($"midpoints above zero:     {aboveZero} of {atMidpoints.Count}"
                                      + $" ({(atMidpoints.Count > 0 ? aboveZero * 100.0 / atMidpoints.Count : 0):0}%)");
        }

        /// <summary>
        /// Compares the model's beats with the grid the beatmap's own timing points declare.
        /// </summary>
        /// <remarks>
        /// The tempo comparison says the two agree about the tempo; it cannot say whether the beats are in the right
        /// place. This measures the residual - how far each of the model's beats is from the nearest beat the map puts
        /// on its grid - and reports it every ten seconds, because the failure being chased is local: a section where
        /// the beats wander, not a track that is uniformly offset.
        ///
        /// This map declares twice the tempo the music is at, so the model's beats land on every other grid line. The
        /// residual is measured against the full grid, where a correct beat is never more than half a grid line away
        /// whatever multiplier the map used, so the comparison does not depend on knowing that.
        /// </remarks>
        [Test]
        [Explicit("needs OSUTEST_AUDIO and the model file")]
        public void TestModelBeatsAgainstBeatmapGrid()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            double[] beats = BeatTrainRegulariser.Regularise(BeatThisBeatTracker.BeatTimes(audio, model, BassAudioDecoder.Default));
            double[] grid = buildBeatmapGrid(loadTimingPoints());

            var residuals = new List<double>();

            foreach (double beat in beats)
                residuals.Add(distanceToGrid(grid, beat));

            var sorted = residuals.OrderBy(r => r).ToArray();

            TestContext.Out.WriteLine($"{beats.Length} model beats against {grid.Length} grid lines from the beatmap");
            TestContext.Out.WriteLine($"residual: median {sorted[sorted.Length / 2]:0.#}ms  "
                                      + $"p90 {sorted[(int)(sorted.Length * 0.9)]:0.#}ms  max {sorted[^1]:0.#}ms  "
                                      + $"over 60ms: {residuals.Count(r => r > 60)} of {residuals.Count}");

            // Restricted to the beats where the model's local period matches the tempo the map declares there, so the
            // comparison is between two grids at the same metrical level. Everywhere else the model is pulsing at some
            // multiple of the map's beat and a residual only measures that difference, not whether a beat is misplaced.
            // What is left is the number that matters: where the model claims the map's own tempo, is it in the right
            // place within the beat?
            var sameLevel = new List<double>();
            var sameLevelTimes = new List<double>();

            for (int i = 1; i < beats.Length; i++)
            {
                double declared = tempoAt(loadTimingPoints(), beats[i]);

                if (declared <= 0)
                    continue;

                // The local period from the beats either side, which is what the model is claiming at this point.
                int low = Math.Max(0, i - 4);
                int high = Math.Min(beats.Length - 1, i + 4);
                var nearby = new List<double>();

                for (int j = low; j < high; j++)
                    nearby.Add(beats[j + 1] - beats[j]);

                if (nearby.Count == 0)
                    continue;

                nearby.Sort();
                double period = nearby[nearby.Count / 2];
                double implied = period > 0 ? 60000 / period : 0;

                if (implied <= 0 || Math.Abs(Math.Log2(implied / declared)) > 0.3)
                    continue;

                sameLevel.Add(distanceToGrid(grid, beats[i]));
                sameLevelTimes.Add(beats[i]);
            }

            if (sameLevel.Count > 0)
            {
                var sameSorted = sameLevel.OrderBy(r => r).ToArray();

                TestContext.Out.WriteLine();
                TestContext.Out.WriteLine($"at the map's own level ({sameLevel.Count} of {beats.Length} beats): "
                                          + $"median {sameSorted[sameSorted.Length / 2]:0.#}ms  "
                                          + $"p90 {sameSorted[(int)(sameSorted.Length * 0.9)]:0.#}ms  "
                                          + $"over 60ms: {sameLevel.Count(r => r > 60)} of {sameLevel.Count} "
                                          + $"({100.0 * sameLevel.Count(r => r > 60) / sameLevel.Count:0}%)");

                var signed = new List<double>();

                for (int i = 0; i < sameLevel.Count; i++)
                {
                    int index = Array.BinarySearch(grid, sameLevelTimes[i]);
                    int at = Math.Clamp(index < 0 ? ~index : index, 0, grid.Length - 1);
                    double nearest = grid[at];

                    if (at > 0 && Math.Abs(grid[at - 1] - sameLevelTimes[i]) < Math.Abs(nearest - sameLevelTimes[i]))
                        nearest = grid[at - 1];

                    signed.Add(sameLevelTimes[i] - nearest);
                }

                signed.Sort();

                TestContext.Out.WriteLine($"signed offset at that level: p10 {signed[(int)(signed.Count * 0.1)]:0}  "
                                          + $"median {signed[signed.Count / 2]:0}  "
                                          + $"p90 {signed[(int)(signed.Count * 0.9)]:0}  ms");
            }

            TestContext.Out.WriteLine();
            TestContext.Out.WriteLine("   from      n  median     p90     max  over60");

            // The other direction, and the one that describes what is seen: of the beats the file puts on its grid, how
            // many does the UI actually pulse on? The UI pulses on exactly the grid's beats, so a miss here is a beat
            // the animation does not mark, and a hit is one it marks in the right place.
            //
            // Read with the level in mind: the model often reports half the map's tempo, in which case half the map's
            // lines have no pulse by construction and that is the multiplier being different, not a beat being missed.
            int hits = 0;
            int missing = 0;
            int cursor = 0;
            int considered = 0;

            foreach (double line in grid)
            {
                // The beatmap's grid runs on past the end of the audio by construction, and a line beyond the track has
                // no pulse to find. Counting those as missed beats made this read far worse than it is.
                if (line > beats[^1])
                    break;

                considered++;

                while (cursor + 1 < beats.Length && beats[cursor + 1] < line)
                    cursor++;

                double nearest = Math.Min(cursor < beats.Length ? Math.Abs(beats[cursor] - line) : double.MaxValue,
                                          cursor + 1 < beats.Length ? Math.Abs(beats[cursor + 1] - line) : double.MaxValue);

                if (nearest <= 60)
                    hits++;
                else if (nearest > 149)
                    missing++;
            }

            TestContext.Out.WriteLine();
            TestContext.Out.WriteLine($"of the beatmap's beats within the track, {hits} of {considered} "
                                      + $"({100.0 * hits / Math.Max(1, considered):0}%) have a UI pulse within 60ms, and "
                                      + $"{missing} ({100.0 * missing / Math.Max(1, considered):0}%) have none within 149ms");
            TestContext.Out.WriteLine();
            TestContext.Out.WriteLine("   from      n  median     p90     max  over60");

            for (double from = 0; from < beats[^1]; from += 10_000)
            {
                var bucket = new List<double>();

                for (int i = 0; i < beats.Length; i++)
                {
                    if (beats[i] >= from && beats[i] < from + 10_000)
                        bucket.Add(residuals[i]);
                }

                if (bucket.Count == 0)
                    continue;

                bucket.Sort();

                TestContext.Out.WriteLine($"{from / 1000,6:0}s {bucket.Count,6} "
                                          + $"{bucket[bucket.Count / 2],7:0.#} "
                                          + $"{bucket[(int)(bucket.Count * 0.9)],7:0.#} "
                                          + $"{bucket[^1],7:0.#} "
                                          + $"{bucket.Count(r => r > 60),7}");
            }
        }

        /// <summary>
        /// Measures what the pruning step does to the beats, against the beatmap's grid.
        /// </summary>
        /// <remarks>
        /// A beat in the wrong place can have been put there by the model or by the code that reads the model's
        /// activation, and the pruned output cannot tell the two apart. Reporting both against the map's grid
        /// attributes every wrong beat to one side or the other before anything is changed.
        /// </remarks>
        [Test]
        [Explicit("diagnostic: needs OSUTEST_AUDIO and the model file")]
        public void TestPeakPruningEffect()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            float[] samples = BassAudioDecoder.DecodeMono(audio, BassAudioDecoder.ANALYSIS_SAMPLE_RATE);
            var (logits, _) = BeatThisBeatTracker.Logits(BeatThisBeatTracker.LogMelSpectrogram(samples), model);

            var raw = BeatThisBeatTracker.RawPeaks(logits);
            var pruned = BeatThisBeatTracker.Peaks(logits);
            double[] grid = buildBeatmapGrid(loadTimingPoints());

            TestContext.Out.WriteLine($"raw peaks {raw.Count}, after pruning {pruned.Count}, removed {raw.Count - pruned.Count}");

            double[] timesOf(List<int> frames)
            {
                var times = new double[frames.Count];

                for (int i = 0; i < frames.Count; i++)
                    times[i] = BeatThisBeatTracker.FrameToMilliseconds(frames[i]);

                return times;
            }

            double[] rawTimes = timesOf(raw);
            double[] prunedTimes = timesOf(pruned);

            TestContext.Out.WriteLine();
            TestContext.Out.WriteLine("   from   rawN  raw>60   rawMd |  prN  pr>60   prMd | removed");

            double end = Math.Min(rawTimes[^1], prunedTimes[^1]);

            for (double from = 0; from < end; from += 10_000)
            {
                var rawBucket = rawTimes.Where(t => t >= from && t < from + 10_000).Select(t => distanceToGrid(grid, t)).OrderBy(r => r).ToArray();
                var prBucket = prunedTimes.Where(t => t >= from && t < from + 10_000).Select(t => distanceToGrid(grid, t)).OrderBy(r => r).ToArray();

                int rawN = rawBucket.Length;
                int prN = prBucket.Length;

                if (rawN == 0 && prN == 0)
                    continue;

                TestContext.Out.WriteLine($"{from / 1000,6:0}s {rawN,6} {rawBucket.Count(r => r > 60),6} "
                                          + $"{rawBucket[rawN / 2],6:0.#} | {prN,4} {prBucket.Count(r => r > 60),6} "
                                          + $"{prBucket[prN / 2],6:0.#} | {rawN - prN,7}");
            }

            // The beats the pruning threw away, with how far they sit from the map's grid: if the pruning is dropping
            // the wrong one of a pair, the ones it drops are the ones that were in the right place.
            var removed = new List<double>();

            foreach (int frame in raw)
            {
                if (!pruned.Contains(frame))
                    removed.Add(BeatThisBeatTracker.FrameToMilliseconds(frame));
            }

            if (removed.Count > 0)
            {
                var removedResiduals = removed.Select(t => distanceToGrid(grid, t)).OrderBy(r => r).ToArray();

                TestContext.Out.WriteLine();
                TestContext.Out.WriteLine($"removed beats: median residual {removedResiduals[removedResiduals.Length / 2]:0.#}ms, "
                                          + $"over 60ms {removedResiduals.Count(r => r > 60)} of {removedResiduals.Length}");
                TestContext.Out.WriteLine("first 40 removed: " + string.Join(", ", removed.Take(40).Select(t => (t / 1000).ToString("0.00"))));
            }
        }

        /// <summary>
        /// Prints every beat the model reports in a range with its activation, its gap to the previous beat and its
        /// distance from the beatmap's grid.
        /// </summary>
        /// <remarks>
        /// The summary statistics cannot separate a beat the model is sure about from one it barely crosses zero on, and
        /// the two have entirely different fixes. This prints the activation next to the placement so a spurious beat
        /// can be recognised by being weak rather than by being inferred to be spurious from its position.
        /// </remarks>
        [Test]
        [Explicit("diagnostic: needs OSUTEST_AUDIO, the model file and OSUTEST_FROM/OSUTEST_TO")]
        public void TestBeatPlacementDump()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            float[] samples = BassAudioDecoder.DecodeMono(audio, BassAudioDecoder.ANALYSIS_SAMPLE_RATE);
            var (logits, _) = BeatThisBeatTracker.Logits(BeatThisBeatTracker.LogMelSpectrogram(samples), model);
            var raw = BeatThisBeatTracker.RawPeaks(logits);
            double[] grid = buildBeatmapGrid(loadTimingPoints());

            double from = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_FROM"), out double f) ? f : 28000;
            double to = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_TO"), out double t) ? t : 47000;

            TestContext.Out.WriteLine($"    time   gap    logit   resid  |  (raw peaks, {model})");
            TestContext.Out.WriteLine();

            // Activation of correct and incorrect beats, so "wrong beats are weak" is measured and not assumed.
            var onGrid = new List<double>();
            var offGrid = new List<double>();

            double previous = double.NaN;

            for (int i = 0; i < raw.Count; i++)
            {
                double time = BeatThisBeatTracker.FrameToMilliseconds(raw[i]);
                double residual = distanceToGrid(grid, time);

                if (residual <= 60)
                    onGrid.Add(logits[raw[i]]);
                else
                    offGrid.Add(logits[raw[i]]);

                if (time < from || time > to)
                {
                    previous = time;
                    continue;
                }

                string gap = double.IsNaN(previous) ? "    -" : $"{time - previous,5:0}";

                TestContext.Out.WriteLine($"{time / 1000,8:0.00} {gap} {logits[raw[i]],8:0.000} {residual,7:0.#}");
                previous = time;
            }

            var sortedOn = onGrid.OrderBy(v => v).ToArray();
            var sortedOff = offGrid.OrderBy(v => v).ToArray();

            string describe(double[] values) => values.Length == 0
                ? "n/a"
                : $"n {values.Length,4}  min {values[0],7:0.000}  p25 {values[values.Length / 4],7:0.000}  "
                  + $"median {values[values.Length / 2],7:0.000}  p75 {values[values.Length * 3 / 4],7:0.000}  max {values[^1],7:0.000}";

            TestContext.Out.WriteLine();
            TestContext.Out.WriteLine($"activation of beats on the grid:   {describe(sortedOn)}");
            TestContext.Out.WriteLine($"activation of beats off the grid:  {describe(sortedOff)}");
        }

        /// <summary>
        /// Prints the tempo readout over time next to the intervals it is read from.
        /// </summary>
        /// <remarks>
        /// A tempo that flickers and a tempo that is simply wrong look the same in a summary and have nothing in common
        /// as faults, so this shows the readout and its inputs together. If the intervals are all one length and the
        /// readout still moves, the readout is at fault; if the intervals themselves alternate, the beats are.
        /// </remarks>
        [Test]
        [Explicit("diagnostic: needs OSUTEST_AUDIO and the model file")]
        public void TestLocalTempoStability()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            double from = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_FROM"), out double f) ? f : 0;
            double to = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_TO"), out double t) ? t : double.MaxValue;

            var grid = BeatGrid.FromBeats(BeatTrainRegulariser.Regularise(BeatThisBeatTracker.BeatTimes(audio, model, BassAudioDecoder.Default)));

            TestContext.Out.WriteLine($"{grid.Beats.Count} beats, metrical shift {grid.MetricalShift}, "
                                      + $"duration {grid.Duration / 1000:0.0}s");

            var tempi = new List<double>();

            for (double time = 0; time <= Math.Min(grid.Duration, to); time += 1000)
            {
                if (time < from)
                    continue;

                int centre = Math.Max(0, grid.IndexAt(time));
                int first = Math.Max(0, centre - 6);
                int last = Math.Min(grid.Beats.Count - 1, centre + 6);

                var gaps = new List<double>();

                for (int i = first; i < last; i++)
                    gaps.Add(grid.Beats[i + 1] - grid.Beats[i]);

                double bpm = grid.BpmAt(time);
                tempi.Add(bpm);

                gaps.Sort();

                // Alternating intervals show up as a gap between the smallest and the largest, so both ends are printed
                // rather than a mean that would hide exactly the thing being looked for.
                TestContext.Out.WriteLine($"{time / 1000,7:0.0}s  {bpm,6:0.0} BPM   "
                                          + $"gaps {gaps[0],4:0} .. {gaps[gaps.Count / 2],4:0} .. {gaps[^1],4:0}   "
                                          + $"({string.Join(" ", gaps.Select(g => g.ToString("0")))})");
            }

            // How much the readout moves from one second to the next, as a fraction. A readout that is stable in a
            // constant section barely moves; one that flips between a tempo and its double jumps by 100%.
            var jumps = new List<double>();

            for (int i = 1; i < tempi.Count; i++)
            {
                if (tempi[i] > 0 && tempi[i - 1] > 0)
                    jumps.Add(Math.Abs(tempi[i] / tempi[i - 1] - 1));
            }

            if (jumps.Count > 0)
            {
                var sorted = jumps.OrderBy(j => j).ToArray();

                TestContext.Out.WriteLine();
                TestContext.Out.WriteLine($"second-to-second change: median {sorted[sorted.Length / 2] * 100:0.0}%  "
                                          + $"p90 {sorted[(int)(sorted.Length * 0.9)] * 100:0.0}%  max {sorted[^1] * 100:0.0}%");
                TestContext.Out.WriteLine($"changes over 50%: {jumps.Count(j => j > 0.5)} of {jumps.Count}");

                // The jumps themselves, next to how long each reading lasted. A doubling that stands for twenty seconds
                // is a section of the track; one that lasts a second is a fragment, and telling those apart is the whole
                // question.
                TestContext.Out.WriteLine();
                TestContext.Out.WriteLine("jumps over 25%:");

                for (int i = 1; i < tempi.Count; i++)
                {
                    if (tempi[i] <= 0 || tempi[i - 1] <= 0)
                        continue;

                    double change = tempi[i] / tempi[i - 1] - 1;

                    if (Math.Abs(change) <= 0.25)
                        continue;

                    // How far the new reading persists before it moves by a quarter again.
                    int run = 1;

                    for (int j = i + 1; j < tempi.Count; j++)
                    {
                        if (tempi[j] <= 0 || tempi[j - 1] <= 0 || Math.Abs(tempi[j] / tempi[j - 1] - 1) > 0.25)
                            break;

                        run++;
                    }

                    int back = 0;

                    for (int j = i; j < tempi.Count && tempi[j] > 0; j++)
                    {
                        if (Math.Abs(tempi[j] - tempi[i]) > tempi[i] * 0.15)
                            break;

                        back++;
                    }

                    TestContext.Out.WriteLine($"  {i,5:0}s  {tempi[i - 1],6:0.0} -> {tempi[i],6:0.0} "
                                              + $"({change * 100,+5:0}%)  holds {back,4:0}s");
                }
            }
        }

        /// <summary>
        /// Prints the model's tempo next to the beatmap's through a section the map ramps continuously.
        /// </summary>
        /// <remarks>
        /// The agreement test deliberately skips anything within a few seconds of a timing point, because a window
        /// median across two tempi measures the window rather than the model. That rules out a ramp entirely - every
        /// point in one is near a timing point - so the hardest part of the track is also the part nothing checks.
        /// This looks at it anyway, comparing against the timing point in force at each instant.
        /// </remarks>
        [Test]
        [Explicit("diagnostic: needs OSUTEST_AUDIO and the model file")]
        public void TestRampAgreement()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            double from = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_FROM"), out double f) ? f : 100000;
            double to = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_TO"), out double t) ? t : 300000;

            var truth = loadTimingPoints();
            var grid = BeatGrid.FromBeats(BeatTrainRegulariser.Regularise(BeatThisBeatTracker.BeatTimes(audio, model, BassAudioDecoder.Default)));

            TestContext.Out.WriteLine($"{grid.Beats.Count} beats over {grid.Duration / 1000:0.0}s, "
                                      + $"metrical shift {grid.MetricalShift}");
            TestContext.Out.WriteLine();
            TestContext.Out.WriteLine("  time     map   model   ratio  octaves  beats/s");

            for (double time = from; time <= Math.Min(to, grid.Duration); time += 2000)
            {
                double expected = tempoAt(truth, time);

                if (expected <= 0)
                    continue;

                double actual = grid.BpmAt(time);
                double ratio = actual / expected;
                double octaves = Math.Log2(ratio);

                int first = grid.IndexAt(time);
                int count = 0;

                for (int i = Math.Max(0, first); i < grid.Beats.Count && grid.Beats[i] < time + 2000; i++)
                    count++;

                TestContext.Out.WriteLine($"{time / 1000,6:0}s {expected,7:0.0} {actual,7:0.0} "
                                          + $"{ratio,7:0.00} {octaves,+8:0.00} {count / 2.0,8:0.0}");
            }
        }

        /// <summary>
        /// Measures whether each beat of the grid lands on a real onset in the audio.
        /// </summary>
        /// <remarks>
        /// This is the measurement the goal actually calls for, and it is not the same as tempo agreement. What the
        /// visuals need is for every beat they pulse on to coincide with a real beat in the music; whether the tempo is
        /// reported at the music's own rate or at half of it does not matter, because pulsing on every other beat still
        /// hits every beat it pulses on. A tempo can be perfectly stable and still be useless if its beats sit between
        /// the music's.
        ///
        /// So the reference here is neither the beatmap nor the model but the audio: an onset envelope built straight
        /// from the samples, smoothed by one frame, and sampled at each beat. A grid whose beats are real sits well
        /// above the envelope's own mean; a grid of the right tempo in the wrong phase, or one whose beats are invented,
        /// sits at or below it. Because the envelope is a flux, "at" means "an onset happened here", which is exactly
        /// the property the animation depends on.
        /// </remarks>
        [Test]
        [Explicit("diagnostic: needs OSUTEST_AUDIO and the model file")]
        public void TestBeatHitRate()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            float[] samples = BassAudioDecoder.DecodeMono(audio, BassAudioDecoder.ANALYSIS_SAMPLE_RATE);

            double[] flux = onsetEnvelope(samples, out int frames);

            // Each frame is compared against its own neighbourhood, so a beat counts as hit whether the onset lands on
            // the frame or a frame either side of it - the model's beats are quantised to 20ms and the envelope is not.
            var smoothed = new double[frames];

            for (int t = 0; t < frames; t++)
            {
                double best = 0;

                for (int j = Math.Max(0, t - 1); j <= Math.Min(frames - 1, t + 1); j++)
                    best = Math.Max(best, flux[j]);

                smoothed[t] = best;
            }

            double baseline = smoothed.Average();

            TestContext.Out.WriteLine($"onset envelope: {frames} frames, baseline {baseline:0.000}");
            TestContext.Out.WriteLine();

            // The model's grid, and for comparison the beatmap's own grid over the same audio.
            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            var ours = BeatGrid.FromBeats(BeatTrainRegulariser.Regularise(BeatThisBeatTracker.BeatTimes(audio, model, BassAudioDecoder.Default)));
            double[] reference = buildBeatmapGrid(loadTimingPoints());

            TestContext.Out.WriteLine($"model grid: {ours.Beats.Count} beats, shift {ours.MetricalShift}");
            TestContext.Out.WriteLine();

            TestContext.Out.WriteLine("   from      n   model hit  above base  |      n   map hit  above base");

            // The beatmap's grid runs on past the end of the audio by construction, so the comparison stops where the
            // audio does rather than reporting frames that were clamped to the last one.
            double duration = Math.Min(samples.Length * 1000.0 / BassAudioDecoder.ANALYSIS_SAMPLE_RATE,
                                       Math.Max(ours.Duration, reference[^1]));

            for (double from = 0; from < duration; from += 10_000)
            {
                var oursBucket = ours.Beats.Where(b => b >= from && b < from + 10_000).ToArray();
                var mapBucket = reference.Where(b => b >= from && b < from + 10_000).ToArray();

                if (oursBucket.Length == 0 && mapBucket.Length == 0)
                    continue;

                double oursHit = oursBucket.Length == 0
                    ? 0
                    : oursBucket.Average(b => smoothed[Math.Clamp((int)(b * 50 / 1000), 0, frames - 1)]);

                double mapHit = mapBucket.Length == 0
                    ? 0
                    : mapBucket.Average(b => smoothed[Math.Clamp((int)(b * 50 / 1000), 0, frames - 1)]);

                double above(IEnumerable<double> beats) => beats.Any()
                    ? beats.Count(b => smoothed[Math.Clamp((int)(b * 50 / 1000), 0, frames - 1)] > baseline) * 100.0 / beats.Count()
                    : 0;

                TestContext.Out.WriteLine($"{from / 1000,6:0}s {oursBucket.Length,6} "
                                          + $"{oursHit / baseline,10:0.00} {above(oursBucket),11:0}%  | "
                                          + $"{mapBucket.Length,6} {mapHit / baseline,8:0.00} {above(mapBucket),11:0}%");
            }

            double mean_ratio(IReadOnlyList<double> beats) => beats.Count == 0
                ? 0
                : beats.Average(b => smoothed[Math.Clamp((int)(b * 50 / 1000), 0, frames - 1)]) / baseline;

            TestContext.Out.WriteLine();
            TestContext.Out.WriteLine($"overall: model {mean_ratio(ours.Beats):0.00}x baseline, "
                                      + $"beatmap {mean_ratio(reference):0.00}x");
        }

        /// <summary>
        /// A half-wave rectified spectral flux at the model's own frame rate, normalised to a mean of one.
        /// </summary>
        private static double[] onsetEnvelope(float[] samples, out int frames)
        {
            const int hop = 441;
            const int nfft = 1024;

            frames = 1 + samples.Length / hop;

            var flux = new double[frames];
            var frame = new float[nfft];
            var magnitudes = new float[Fft.BIN_COUNT];
            var previous = new float[Fft.BIN_COUNT];

            for (int t = 0; t < frames; t++)
            {
                int start = t * hop - nfft / 2;

                for (int i = 0; i < nfft; i++)
                {
                    int index = start + i;
                    frame[i] = index >= 0 && index < samples.Length ? samples[index] : 0;
                }

                Fft.AnalyseMagnitudes(frame, magnitudes);

                double sum = 0;

                for (int b = 0; b < Fft.BIN_COUNT; b++)
                {
                    double delta = magnitudes[b] - previous[b];

                    if (delta > 0)
                        sum += delta;

                    previous[b] = magnitudes[b];
                }

                flux[t] = sum;
            }

            double mean = flux.Average();

            if (mean > 0)
            {
                for (int t = 0; t < frames; t++)
                    flux[t] /= mean;
            }

            return flux;
        }

        /// <summary>
        /// Reports, section by section, whether the audio has onsets between the beats the tracker reports.
        /// </summary>
        /// <remarks>
        /// Detection only, so it changes nothing. A section where the middle of a beat carries almost as much onset as
        /// the beat itself is one where the tracker has halved the tactus - a failure it cannot report on itself, since
        /// its activation halfway between its own beats is strongly negative either way.
        /// </remarks>
        [Test]
        [Explicit("diagnostic: needs OSUTEST_AUDIO and the model file")]
        public void TestHalvingDetection()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            float[] samples = BassAudioDecoder.DecodeMono(audio, BassAudioDecoder.ANALYSIS_SAMPLE_RATE);
            double[] envelope = OnsetEnvelope.FromSamples(samples, out int frames);
            double[] beats = BeatTrainRegulariser.Regularise(BeatThisBeatTracker.BeatTimes(audio, model, BassAudioDecoder.Default));

            TestContext.Out.WriteLine($"{beats.Length} beats, {frames} envelope frames");
            TestContext.Out.WriteLine();
            TestContext.Out.WriteLine("   from     n    beats      mid    ratio");

            for (double from = 0; from < beats[beats.Length - 1]; from += 5000)
            {
                var inside = beats.Where(b => b >= from && b < from + 5000).ToArray();

                if (inside.Length < 4)
                    continue;

                var (atBeats, atMidpoints, count) = OnsetEnvelope.Compare(inside, envelope, frames);
                double ratio = atBeats > 0 ? atMidpoints / atBeats : 0;

                TestContext.Out.WriteLine($"{from / 1000,6:0}s {count,5} {atBeats,8:0.00} {atMidpoints,8:0.00} {ratio,8:0.00}"
                                          + (OnsetEnvelope.LooksHalved(atBeats, atMidpoints) ? "   <-- halved" : ""));
            }
        }
        /// <summary>
        /// Prints the period the level unification treats as the track's own.
        /// </summary>
        /// <remarks>
        /// One number, and the whole of the behaviour that depends on it. It did not fire on a Stage 5 passage that is
        /// at twice the density of everything around it, and the only way to tell whether the statistic picked the
        /// wrong level or the ratio test is too strict is to look at what it picked.
        /// </remarks>
        [Test]
        [Explicit("diagnostic: needs OSUTEST_AUDIO and the model file")]
        public void TestDominantPeriod()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            double[] tracked = BeatThisBeatTracker.BeatTimes(audio, model, BassAudioDecoder.Default);
            double dominant = BeatTrainRegulariser.Dominant(tracked);

            TestContext.Out.WriteLine($"{tracked.Length} tracked beats over {tracked[tracked.Length - 1] / 1000:0.#}s");
            TestContext.Out.WriteLine($"dominant period {dominant:0}ms  ({60000 / Math.Max(1, dominant):0.#} BPM)");

            // The same statistic without the duration weighting, for comparison: if the two disagree, the weighting is
            // what is choosing the wrong level.
            var counts = new Dictionary<int, int>();

            for (int i = 1; i < tracked.Length; i++)
            {
                double gap = tracked[i] - tracked[i - 1];

                if (gap <= 0)
                    continue;

                int bucket = (int)Math.Round(gap / 20) * 20;
                counts[bucket] = counts.TryGetValue(bucket, out int seen) ? seen + 1 : 1;
            }

            TestContext.Out.WriteLine();
            TestContext.Out.WriteLine("most common gaps (by beat count):");

            foreach (var pair in counts.OrderByDescending(p => p.Value).Take(8))
                TestContext.Out.WriteLine($"  {pair.Key,5}ms  {60000.0 / pair.Key,7:0.#} BPM  x{pair.Value}");

            // The local estimate against the dominant one, through a passage that should have been pulled back to the
            // track's own level and was not.
            double from = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_FROM"), out double f) ? f : 285000;
            double to = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_TO"), out double t) ? t : 295000;

            TestContext.Out.WriteLine();
            TestContext.Out.WriteLine($"local period through [{from / 1000:0}s, {to / 1000:0}s], "
                                      + $"ratio to the dominant {dominant:0}ms:");

            for (int i = 1; i < tracked.Length; i++)
            {
                if (tracked[i] < from || tracked[i] > to)
                    continue;

                double local = BeatTrainRegulariser.LocalPeriodAt(tracked, i);
                double ratio = dominant > 0 ? local / dominant : 0;

                TestContext.Out.WriteLine($"  {tracked[i] / 1000,7:0.00}s  gap {tracked[i] - tracked[i - 1],5:0}ms  "
                                          + $"local {local,5:0}ms  ratio {ratio,5:0.00}"
                                          + (ratio <= 0.59 ? "   <- should be doubled" : ratio >= 1.7 ? "   <- should be halved" : ""));
            }
        }

        /// <summary>Every beat the beatmap's timing points put on its grid, in milliseconds.</summary>
        private static double[] buildBeatmapGrid(List<(double Time, double BeatLength)> timingPoints)
        {
            var grid = new List<double>();

            for (int i = 0; i < timingPoints.Count; i++)
            {
                double until = i + 1 < timingPoints.Count ? timingPoints[i + 1].Time : timingPoints[i].Time + 60_000;
                double length = timingPoints[i].BeatLength;

                if (length <= 0)
                    continue;

                for (double time = timingPoints[i].Time; time < until && grid.Count < 500_000; time += length)
                    grid.Add(time);
            }

            grid.Sort();

            return grid.ToArray();
        }

        /// <summary>How far a time is from the nearest line of a sorted grid.</summary>
        private static double distanceToGrid(double[] grid, double time)
        {
            int index = Array.BinarySearch(grid, time);

            if (index < 0)
                index = ~index;

            double best = double.MaxValue;

            for (int i = Math.Max(0, index - 2); i <= Math.Min(grid.Length - 1, index + 2); i++)
                best = Math.Min(best, Math.Abs(grid[i] - time));

            return best == double.MaxValue ? 0 : best;
        }

    }
}
