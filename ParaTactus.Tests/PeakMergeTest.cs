using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using ParaTactus;

namespace ParaTactus.Tests
{
    /// <summary>
    /// The peak merge has to agree with the published postprocessor, floating-point mean included.
    /// </summary>
    /// <remarks>
    /// The reference is <c>deduplicate_peaks(width=1)</c> in <c>beat_this/model/postprocessor.py</c>: groups of adjacent
    /// peak frames no more than one frame apart become the mean of the group, and that mean is a real number - it is
    /// converted to a time only at the end. This port accumulated it in an <c>int</c>, so integer division dropped the
    /// fraction at every step: a plateau of frames 10, 11 and 12 came out at 10 rather than 10.5, which is a systematic
    /// 10ms shift on every merged beat. At this frame rate a frame is 20ms, over 3% of a 600ms beat.
    ///
    /// The test carries the reference's algorithm, transcribed, and compares it with what the tracker produces for the
    /// same peaks.
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
        public void TheMergeAgreesWithTheReference()
        {
            // A three-frame plateau above zero, an isolated peak, then a two-frame plateau.
            float[] logits = activation((10, 2f), (11, 2f), (12, 2f), (20, 3f), (30, 1.5f), (31, 1.5f));
            double[] expected = referenceDeduplicate(new[] { 10, 11, 12, 20, 30, 31 });
            double[] actual = BeatThisBeatTracker.RawPeaks(logits).ToArray();

            Assert.That(actual, Is.EqualTo(expected).Within(1e-9));

            // Stated separately, because these are the numbers the rule means. Note the third frame of the plateau: once
            // the running mean has moved back to 10.5, frame 12 is more than one frame away and starts a group of its
            // own, so a three-frame plateau yields two peaks. That is the reference's behaviour as well - the mean is
            // what the next comparison is against - and it is pinned here so that a future tidy-up has to argue with it.
            Assert.That(actual, Is.EqualTo(new[] { 10.5, 12.0, 20.0, 30.5 }).Within(1e-9));
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
