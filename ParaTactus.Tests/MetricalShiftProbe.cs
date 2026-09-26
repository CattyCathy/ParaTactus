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
    /// Shows why the metrical level decision moves a grid by four octaves on tracks that need no move at all.
    /// </summary>
    /// <remarks>
    /// On five of twelve beatmaps the grid the player is handed holds four per cent of the beats the beats it is built
    /// from: five hundred and one beats become thirty-two, three hundred and thirty become twenty-one. The shift
    /// reported is four, which keeps every sixteenth beat, and the decision that produces it requires the intervals to
    /// look like a tempo sixteen times too fast to be in the comfortable range. So the intervals really are that short -
    /// a median of thirty-three milliseconds, against a frame of twenty - and the beats being levelled are therefore not
    /// evenly spaced musical beats.
    ///
    /// This prints what the decision reads, and two things about the beats themselves that the summary cannot show:
    /// where in the track the tracker's beats fall, and how much of the track the model was confident about.
    ///
    /// A caution for anyone reading the output: <c>FramePeaks</c> answers in frame indices and <c>Peaks</c> answers in
    /// milliseconds, and at fifty frames a second the two differ by a factor of twenty. Comparing one against the other
    /// produces a contradiction that is entirely an artefact of the units, which cost an afternoon here.
    ///
    /// Set <c>OSUTEST_DATASET</c> and optionally <c>OSUTEST_ACTIVATION</c>.
    /// </remarks>
    [Explicit("Needs OSUTEST_DATASET and the Beat This! model.")]
    public class MetricalShiftProbe
    {
        [OneTimeSetUp]
        public void SetUp() => AudioTestEnvironment.Initialise();

        [Test]
        public void ShowWhatTheShiftDecisionSees()
        {
            string dataset = Environment.GetEnvironmentVariable("OSUTEST_DATASET");

            if (string.IsNullOrEmpty(dataset) || !Directory.Exists(dataset))
                Assert.Ignore("Set OSUTEST_DATASET to the exported dataset.");

            string activationPath = Environment.GetEnvironmentVariable("OSUTEST_ACTIVATION")
                                    ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                                                    "Downloads", "beat-this-final0-int8.onnx");

            if (!File.Exists(activationPath))
                Assert.Ignore($"no model at {activationPath}");

            int limit = int.TryParse(Environment.GetEnvironmentVariable("OSUTEST_LIMIT"), out int wanted) ? wanted : 4;
            int seen = 0;

            using var tracker = new StreamingBeatTracker(activationPath, 2);

            foreach (string line in File.ReadLines(Path.Combine(dataset, "manifest.tsv")).Skip(1))
            {
                if (seen >= limit)
                    break;

                string[] fields = line.Split('\t');

                if (fields.Length < 4)
                    continue;

                string audioPath = Path.Combine(dataset, "audio", fields[2]);

                if (!File.Exists(audioPath))
                    continue;

                seen++;

                float[] samples = ReadAudio(audioPath);

                using var each = new StreamingBeatTracker(activationPath, 2);
                each.Add(samples);
                each.Flush();

                double[] peaks = BeatThisBeatTracker.Peaks(each.Activation.ToArray()).ToArray();
                double[] regularised = BeatTrainRegulariser.Regularise(peaks);

                var intervals = new List<double>();

                for (int i = 1; i < regularised.Length; i++)
                {
                    double gap = regularised[i] - regularised[i - 1];

                    if (gap > 0)
                        intervals.Add(gap);
                }

                TestContext.Out.WriteLine($"  peaks: {peaks.Length} from {peaks.FirstOrDefault():0}-{peaks.LastOrDefault():0}ms");
                TestContext.Out.WriteLine($"  the first ten peaks: {string.Join(", ", peaks.Take(10).Select(v => v.ToString("0.0")))}");
                TestContext.Out.WriteLine($"  the last ten peaks:  {string.Join(", ", peaks.Skip(Math.Max(0, peaks.Length - 10)).Select(v => v.ToString("0.0")))}");

                double shortest = double.MaxValue;
                double longest = 0;

                for (int i = 1; i < peaks.Length; i++)
                {
                    double gap = peaks[i] - peaks[i - 1];

                    shortest = Math.Min(shortest, gap);
                    longest = Math.Max(longest, gap);
                }

                TestContext.Out.WriteLine($"  gaps between consecutive peaks: shortest {shortest:0.00}ms, longest {longest:0}ms");

                TestContext.Out.WriteLine($"track {fields[0]}: {regularised.Length} beats, {intervals.Count} intervals");

                float[] activation = each.Activation.ToArray();
                double activationSeconds = activation.Length / 50.0;

                TestContext.Out.WriteLine($"  audio {samples.Length / 22050.0:0.0}s, activation {activation.Length} frames "
                                          + $"= {activationSeconds:0.0}s, peaks from {peaks.Length} of them");
                TestContext.Out.WriteLine($"  the last peak is at {peaks[^1]:0}ms, which is "
                                          + $"{100.0 * peaks[^1] / (samples.Length / 22.05):0.0}% of the track");

                // How much of the activation the peaks reach, which is the question: an activation that covers the track
                // and peaks that stop a few seconds in is a peak-picking problem, and an activation that stops early is
                // a tracking problem.
                int beyond = activation.Count(v => v > 0);

                TestContext.Out.WriteLine($"  activation frames above zero: {beyond} of {activation.Length}");

                // Where in the track the positive frames are, which is the one thing left that could explain peaks in
                // the first five per cent of a track whose activation covers all of it. A model that is confident early
                // and doubtful later is a different fault from a peak picker that stops, and the summary cannot say
                // which this is.
                var perTenth = new System.Text.StringBuilder();

                for (int tenth = 0; tenth < 10; tenth++)
                {
                    int from = tenth * activation.Length / 10;
                    int to = (tenth + 1) * activation.Length / 10;
                    int positive = 0;
                    double best = double.MinValue;

                    for (int frame = from; frame < to; frame++)
                    {
                        if (activation[frame] > 0)
                            positive++;

                        best = Math.Max(best, activation[frame]);
                    }

                    perTenth.Append($"{tenth}:{positive}/{best:0.0} ");
                }

                TestContext.Out.WriteLine($"  positives per tenth of the track (count/best value): {perTenth}");

                // And what the padding value would be, so a frame the model never produced can be told from one it
                // judged and disliked.
                int padded = activation.Count(v => v <= -999);

                TestContext.Out.WriteLine($"  frames holding the not-a-peak padding: {padded}");

                // The two candidate explanations for peaks in the first five per cent of a track, told apart directly:
                // the frames the peak picker accepts, before anything is merged or refined, and the intervals they are
                // judged against. If the frames themselves reach the end of the track then the loss is downstream of the
                // picker, and if they do not then the activation the picker was given is not the activation printed
                // above.
                var frames = BeatThisBeatTracker.FramePeaks(activation);

                TestContext.Out.WriteLine($"  frames the picker accepts: {frames.Count}, "
                                          + $"from {frames.FirstOrDefault()} to {frames.LastOrDefault()} of {activation.Length}");

                // The same tenths again, but counting the peaks rather than the positive frames. The two lines have to
                // agree about where the track is busy: if a tenth holds twenty positive frames and no peaks then the
                // picker is refusing frames the activation is offering, and if it holds twenty and twenty peaks then
                // the positive frames printed above are not the ones the picker was given.
                var peaksPerTenth = new System.Text.StringBuilder();

                for (int tenth = 0; tenth < 10; tenth++)
                {
                    int from = tenth * activation.Length / 10;
                    int to = (tenth + 1) * activation.Length / 10;
                    int count = frames.Count(f => f >= from && f < to);

                    peaksPerTenth.Append($"{tenth}:{count} ");
                }

                TestContext.Out.WriteLine($"  peaks per tenth of the track: {peaksPerTenth}");

                var widths = BeatThisBeatTracker.MergedWidths(activation);
                var merged = BeatThisBeatTracker.MergedPeaks(activation);

                TestContext.Out.WriteLine($"  after merging: {merged.Count} beats, "
                                          + $"from {merged.FirstOrDefault():0.0} to {merged.LastOrDefault():0.0} frames");

                TestContext.Out.WriteLine($"  merged widths: {string.Join(", ", widths.Take(20))}"
                                          + (widths.Count > 20 ? $" ... ({widths.Count} groups)" : ""));

                // The activation sampled across the track, because a length that covers the track and a last peak five
                // per cent of the way in is a contradiction that only the values themselves can resolve.
                var samplesText = new System.Text.StringBuilder();

                for (int frame = 0; frame < activation.Length; frame += Math.Max(1, activation.Length / 16))
                    samplesText.Append($"{frame}:{activation[frame]:0.0} ");

                TestContext.Out.WriteLine($"  activation across the track: {samplesText}");

                int earliestGap = -1;

                for (int frame = 1; frame < activation.Length; frame++)
                {
                    if (activation[frame] != activation[frame - 1])
                    {
                        earliestGap = frame;
                        break;
                    }
                }

                TestContext.Out.WriteLine($"  the first frame that differs from its predecessor: {earliestGap}");
                TestContext.Out.WriteLine($"  the tracker reports it reached {each.BeatsThrough:0}ms of the "
                                          + $"{samples.Length / 22.05:0}ms track");

                if (peaks.Length > 0)
                {
                    int lastFrame = (int)Math.Round(peaks[^1] * 50 / 1000.0);

                    TestContext.Out.WriteLine($"  the last peak is frame {lastFrame} of {activation.Length}");
                }

                if (intervals.Count == 0)
                    continue;

                intervals.Sort();

                TestContext.Out.WriteLine($"  first beats: {string.Join(", ", regularised.Take(12).Select(v => v.ToString("0.00")))}");
                TestContext.Out.WriteLine($"  last beats:  {string.Join(", ", regularised.Skip(Math.Max(0, regularised.Length - 6)).Select(v => v.ToString("0.00")))}");

                int duplicates = 0;

                for (int i = 1; i < regularised.Length; i++)
                {
                    if (regularised[i] - regularised[i - 1] < 1)
                        duplicates++;
                }

                double span = regularised.Length > 1 ? regularised[^1] - regularised[0] : 0;

                TestContext.Out.WriteLine($"  span {span:0}ms over {regularised.Length} beats, so the mean gap is "
                                          + $"{span / Math.Max(1, regularised.Length - 1):0}ms");
                TestContext.Out.WriteLine($"  gaps under a millisecond: {duplicates}");

                TestContext.Out.WriteLine($"  intervals: min {intervals[0]:0.00}ms  p10 {intervals[intervals.Count / 10]:0}ms  "
                                          + $"median {intervals[intervals.Count / 2]:0}ms  max {intervals[^1]:0}ms");

                int below80 = intervals.Count(i => 60000 / i < 80);
                int inRange = intervals.Count(i => 60000 / i >= 80 && 60000 / i < 220);
                int above220 = intervals.Count(i => 60000 / i >= 220);

                TestContext.Out.WriteLine($"  by tempo: below 80 BPM {below80}, in 80-220 {inRange}, above 220 {above220}");

                // The very short ones, which are what a four-octave shift would need to exist.
                var tiny = intervals.Where(i => i < 20).ToList();

                TestContext.Out.WriteLine($"  intervals under 20ms: {tiny.Count}"
                                          + (tiny.Count > 0 ? $"  ({string.Join(", ", tiny.Take(6).Select(v => v.ToString("0.00")))})" : ""));

                int shift = MetricalLevel.ChooseShift(regularised);

                TestContext.Out.WriteLine($"  the shift chosen: {shift}");

                for (int candidate = 0; candidate <= 4; candidate++)
                {
                    int inside = intervals.Count(i => 60000 / i / Math.Pow(2, candidate) >= 80
                                                      && 60000 / i / Math.Pow(2, candidate) < 220);

                    TestContext.Out.WriteLine($"    at shift {candidate}: {inside * 100.0 / intervals.Count,5:0.0}% of intervals in range");
                }

                TestContext.Out.WriteLine("");
            }
        }

        private static float[] ReadAudio(string path)
        {
            byte[] bytes = File.ReadAllBytes(path);
            var samples = new float[bytes.Length / 4];

            Buffer.BlockCopy(bytes, 0, samples, 0, samples.Length * 4);

            return samples;
        }
    }
}
