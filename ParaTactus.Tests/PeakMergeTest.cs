using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using ParaTactus.Tracking;

namespace ParaTactus.Tests
{
    /// <summary>
    /// The peak merge groups adjacent frames the way the published postprocessor does, and then sharpens the result.
    /// </summary>
    /// <remarks>
    /// The grouping is the reference's <c>deduplicate_peaks(width=1)</c> in <c>beat_this/model/postprocessor.py</c>:
    /// adjacent peak frames no more than one frame apart become one beat at their mean, and that mean is a real number
    /// - it is converted to a time only at the end. This port accumulated it in an <c>int</c>, so integer division
    /// dropped the fraction at every step: a plateau of frames 10, 11 and 12 came out at 10 rather than 10.5, which is
    /// a systematic 10ms shift on every merged beat. At this frame rate a frame is 20ms.
    ///
    /// What is deliberately not the reference's is where the beat is within the group. The model reports one value per
    /// 20ms frame, so a beat read off it is on a frame; the reference leaves it there and this reads the peak's real
    /// position from a parabola through it and its neighbours. That is a deviation and it is here on purpose: measured
    /// against a beatmap's own grid, a frame is the seventh of a beat at 200 BPM and the quantisation is a real part of
    /// the distance. The grouping is still compared with the reference, because the grouping is what the reference
    /// settled and there is no argument with it; the refinement is compared with arithmetic.
    /// </remarks>
    [TestFixture]
    public class PeakMergeTest
    {
        /// <summary><c>deduplicate_peaks</c> from the published postprocessor, transcribed.</summary>
        private static double[] referenceDeduplicate(IEnumerable<int> peaks, int width = 1)
        {
            var result = new List<double>();
            using var iterator = peaks.GetEnumerator();

            if (!iterator.MoveNext())
                return result.ToArray();

            double position = iterator.Current;
            int count = 1;

            while (iterator.MoveNext())
            {
                int next = iterator.Current;

                if (next - position <= width)
                {
                    count++;
                    position += (next - position) / count;
                }
                else
                {
                    result.Add(position);
                    position = next;
                    count = 1;
                }
            }

            result.Add(position);

            return result.ToArray();
        }

        private static float[] activation(params (int Frame, float Value)[] peaks)
        {
            var logits = new float[64];
            Array.Fill(logits, -1f);

            foreach (var (frame, value) in peaks)
                logits[frame] = value;

            return logits;
        }

        [Test]
        public void TheGroupsAreTheOnesTheReferenceForms()
        {
            // A three-frame plateau above zero, an isolated peak, then a two-frame plateau.
            float[] logits = activation((10, 2f), (11, 2f), (12, 2f), (20, 3f), (30, 1.5f), (31, 1.5f));
            double[] reference = referenceDeduplicate(new[] { 10, 11, 12, 20, 30, 31 });
            double[] actual = BeatThisBeatTracker.RawPeaks(logits).ToArray();

            // The groups are the reference's, and each is where the reference puts it to within the half frame a
            // refinement is allowed to move it. Note the plateau: the reference splits three adjacent frames into two
            // beats, because once its running mean has moved the third frame is more than one frame from it. Both
            // readings put a beat either side of the plateau's middle, so the split is not what is being argued here;
            // its exact positions are, and the reference's are within the half frame this allows.
            Assert.That(actual.Length, Is.EqualTo(reference.Length), "the reference forms four groups from these peaks");

            for (int i = 0; i < actual.Length; i++)
            {
                Assert.That(actual[i], Is.EqualTo(reference[i]).Within(0.5),
                    $"group {i} moved further than the half frame a refinement is allowed");
            }
        }

        [Test]
        public void TheBeatIsReadBetweenTheFrames()
        {
            // The value after the peak is larger than the value before it, so the peak is later than its frame. The
            // vertex of the parabola through (4, 1), (5, 2), (6, 1.5) is at 5 + 0.5*(1-1.5)/(1 - 4 + 1.5).
            float[] leaningRight = activation((4, 1f), (5, 2f), (6, 1.5f));
            double later = BeatThisBeatTracker.RawPeaks(leaningRight).Single();

            Assert.That(later, Is.GreaterThan(5), "the peak leans to the right, so the beat is after its frame");
            Assert.That(later, Is.EqualTo(5 + (0.5 * (1 - 1.5) / (1 - 4 + 1.5))).Within(1e-9));

            float[] leaningLeft = activation((4, 1.5f), (5, 2f), (6, 1f));
            double earlier = BeatThisBeatTracker.RawPeaks(leaningLeft).Single();

            Assert.That(earlier, Is.LessThan(5));
            Assert.That(earlier, Is.EqualTo(5 + (0.5 * (1.5 - 1) / (1.5 - 4 + 1))).Within(1e-9));
        }

        [Test]
        public void AMergedPeakKeepsItsFraction()
        {
            double[] peaks = BeatThisBeatTracker.RawPeaks(activation((5, 2f), (6, 2f))).ToArray();

            // Integer division would have answered 5, which is where the old code was wrong.
            Assert.That(peaks, Is.EqualTo(new[] { 5.5 }).Within(1e-9));
        }

        [Test]
        public void ALonePeakIsUnmoved()
        {
            double[] peaks = BeatThisBeatTracker.RawPeaks(activation((7, 2f))).ToArray();

            Assert.That(peaks, Is.EqualTo(new[] { 7.0 }).Within(1e-9));
        }

        [Test]
        public void SilenceHasNoPeaks()
        {
            var logits = new float[64];
            Array.Fill(logits, -1f);

            Assert.That(BeatThisBeatTracker.RawPeaks(logits), Is.Empty);
        }

        [Test]
        public void ReadingTheActivationOfAMergedPeakUsesTheFrameItSitsOn()
        {
            float[] logits = activation((10, 2f), (11, 4f), (12, 2f));

            Assert.That(BeatThisBeatTracker.FrameOf(11.0, logits.Length), Is.EqualTo(11));
            Assert.That(logits[BeatThisBeatTracker.FrameOf(BeatThisBeatTracker.RawPeaks(logits)[0], logits.Length)],
                Is.GreaterThan(0f));
        }
    }
}
