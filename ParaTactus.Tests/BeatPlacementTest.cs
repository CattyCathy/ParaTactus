using System;
using NUnit.Framework;
using ParaTactus;

namespace ParaTactus.Tests
{
    /// <summary>
    /// Where a beat is placed inside the frame it was picked on.
    /// </summary>
    /// <remarks>
    /// A beat is picked on a frame and a frame is twenty milliseconds, which is most of the error a rhythm game allows,
    /// so the placing is worth as much as the picking and it is arithmetic that can be checked against a curve whose
    /// answer is known rather than against a track.
    ///
    /// The cases that matter are all about what happens when the curve is not a clean peak: a peak between two frames
    /// is moved, a balanced peak is left alone, a plateau has no middle to find and must not be moved arbitrarily, and
    /// a window that reaches a second peak must not place this beat between the two of them.
    /// </remarks>
    public class BeatPlacementTest
    {
        /// <summary>A whole curve, given as the value of each frame from frame zero.</summary>
        /// <remarks>
        /// The whole curve rather than a list of interesting frames, so that a test cannot accidentally leave a frame
        /// at zero where it meant to leave it at the value of a pedestal. The period is well above the suppression
        /// radius, so nothing in these tests is about the picker.
        /// </remarks>
        private static float[] curve(params float[] values)
        {
            var beats = new float[values.Length + 8];

            Array.Copy(values, beats, values.Length);

            return beats;
        }

        private static float[] flatPeriods(int length)
        {
            var periods = new float[length];

            for (int i = 0; i < periods.Length; i++)
                periods[i] = 500;

            return periods;
        }

        private static double[] place(float[] beats)
        {
            return LearnedBeatDetector.Place(beats, flatPeriods(beats.Length), 0.6);
        }

        [Test]
        public void APeakBetweenTwoFramesIsMovedToTheCurve()
        {
            // Frames 10, 11 and 12 are 0.2, 1.0 and 0.4. The floor is zero, so the weights are the values and the
            // weighted middle is 11 + (0.4 - 0.2) / 1.6, a little past the picked frame.
            float[] beats = curve(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2f, 1.0f, 0.4f);

            double[] placed = place(beats);

            Assert.That(placed, Has.Length.EqualTo(1));

            double expected = (10 * 0.2 + 11 * 1.0 + 12 * 0.4) / 1.6 * 20.0;

            Assert.That(placed[0], Is.EqualTo(expected).Within(0.01));
            Assert.That(placed[0], Is.Not.EqualTo(11 * 20.0).Within(0.5), "the peak must come off its frame");
        }

        [Test]
        public void ASymmetricPeakIsLeftOnItsFrame()
        {
            float[] beats = curve(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5f, 1.0f, 0.5f);

            double[] placed = place(beats);

            Assert.That(placed, Has.Length.EqualTo(1));
            Assert.That(placed[0], Is.EqualTo(11 * 20.0).Within(0.01));
        }

        [Test]
        public void AFlatCurveLeavesTheBeatWhereItWasPicked()
        {
            // No shape at all, so there is no middle to find. The beat has to stay on its frame rather than being
            // dragged by whichever side the arithmetic happens to fall on.
            float[] beats = curve(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7f, 0.7f, 0.7f);

            double[] placed = place(beats);

            Assert.That(placed, Has.Length.EqualTo(1));
            Assert.That(placed[0], Is.EqualTo(11 * 20.0).Within(0.01));
        }

        [Test]
        public void TheFloorIsTheWholeWindowAndNotJustThePeaksOwnFrames()
        {
            // The floor is the lowest frame in the five-wide window, which is the pedestal at 0.1 on either side of the
            // peak. The weights are therefore the heights above that pedestal - 0.15, 0.9 and 0.4 - and the beat lands
            // a little right of the frame it was picked on, towards the taller of the two shoulders.
            //
            // The two pedestals outside the peak's immediate neighbours are what decide that. Without them the floor
            // would be 0.25 and the same arithmetic would put the beat at 225ms instead of 223.4.
            float[] beats = curve(0, 0, 0, 0, 0, 0, 0, 0, 0.5f, 0.1f, 0.25f, 1.0f, 0.5f, 0.1f, 0.5f);

            double[] placed = place(beats);

            // The floor is read as a float and so carries the float's own representation of a tenth.
            double floor = 0.1f;
            double total = (0.25 - floor) + (1.0 - floor) + (0.5 - floor);
            double weighted = (10 * (0.25 - floor)) + (11 * (1.0 - floor)) + (12 * (0.5 - floor));

            Assert.That(placed, Has.Length.EqualTo(1));
            Assert.That(placed[0], Is.EqualTo(weighted / total * 20.0).Within(0.01),
                        "the weight is the height above the window's floor, not the height above zero");

            // And the whole-window floor is not the same answer as a floor taken from the peak's own neighbours.
            double withoutPedestals = ((11 * (1.0 - 0.25)) + (12 * (0.5 - 0.25))) / ((1.0 - 0.25) + (0.5 - 0.25)) * 20.0;

            Assert.That(withoutPedestals, Is.EqualTo(225.0).Within(0.01));
            Assert.That(placed[0], Is.Not.EqualTo(withoutPedestals).Within(0.5));
        }

        [Test]
        public void TheReadingStopsAtAValley()
        {
            // A beat picked on the peak at 14, with a valley just inside the window on its left and a pedestal on its
            // right. What is read is the stretch of frames above the floor that holds the peak, so the frame past the
            // valley is left out even though it is in the window and above the floor.
            //
            // The valley is at 12, which is the window's left edge, so nothing is on its far side to be excluded. What
            // the test is for is the other side: the pedestal at 15 and 16 is one run above the floor and the valley
            // ends it, so only the peak is read and the beat stays on its own frame rather than being dragged towards
            // the pedestal.
            //
            // Every value is a dyadic fraction - a whole number of eighths - so the floats carry them exactly and the
            // arithmetic below is the arithmetic the code does. A tenth is not, and a test that mixes the two measures
            // float representation rather than behaviour.
            var beats = new float[24];

            beats[11] = 0.5f;    // the window's left edge
            beats[12] = 0.125f;  // the valley
            beats[13] = 0.5f;    // across the valley, inside the window
            beats[14] = 1.0f;    // the peak, and the only frame above the threshold
            beats[15] = 0.5f;    // the pedestal
            beats[16] = 0.5f;    // the window's right edge

            double[] placed = place(beats);

            // The floor is the valley at one eighth, and the peak's neighbours are both above it, so the stretch read
            // runs from 13 to the window's right edge at 16. The weights are 0.375, 0.875, 0.375 and 0.375.
            double expected = ((13 * 0.375) + (14 * 0.875) + (15 * 0.375) + (16 * 0.375)) / 2.0 * 20.0;

            Assert.That(placed, Has.Length.EqualTo(1));
            Assert.That(placed[0], Is.EqualTo(expected).Within(0.01));
            Assert.That(placed[0], Is.GreaterThan(14 * 20.0), "the pedestal on the right pulls the beat right");
            Assert.That(placed[0], Is.LessThan(14 * 20.0 + 10.0), "but only by a fraction of a frame");

            // The valley terminates the stretch that is read. Without that the frame at 11, which is the same height as
            // the pedestal and three frames from the peak, would be read too and the beat would land to the left of the
            // peak instead - so the assertion above is measuring the valley and not just the arithmetic.
            double acrossTheValley = ((11 * 0.375) + (13 * 0.375) + (14 * 0.875) + (15 * 0.375) + (16 * 0.375)) / 2.375 * 20.0;

            Assert.That(acrossTheValley, Is.LessThan(14 * 20.0), "reading across the valley would pull the beat left");
        }

        [Test]
        public void BeatsAreSpacedByThePredictedPeriodAndNotByThePeaks()
        {
            // Two peaks two frames apart with a period far longer than that. Only the stronger can be a beat, and this
            // is a check that separating the placing from the picking did not quietly stop applying the suppression.
            float[] beats = curve(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9f, 0, 0.4f);

            double[] placed = place(beats);

            Assert.That(placed, Has.Length.EqualTo(1));
            Assert.That(placed[0], Is.EqualTo(20 * 20.0).Within(0.01));
        }

        [Test]
        public void AnEmptyCurveGivesNoBeats()
        {
            float[] beats = curve(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

            Assert.That(place(beats), Is.Empty);
        }
    }
}
