using System.Collections.Generic;
using NUnit.Framework;
using ParaTactus;

namespace ParaTactus.Tests
{
    /// <summary>
    /// The beat train has to come out evenly spaced, without losing real tempo changes or carpeting a break in the
    /// music with pulses that are not there.
    /// </summary>
    /// <remarks>
    /// Written against synthetic trains because the property being asserted is about the shape of the output, not about
    /// any particular track, and a fixture makes the arithmetic checkable by hand. The behaviour on a real track is
    /// measured separately, against a beatmap's own timing points.
    /// </remarks>
    [TestFixture]
    public class BeatTrainRegulariserTest
    {
        private static double[] train(double start, double period, int count)
        {
            var beats = new double[count];

            for (int i = 0; i < count; i++)
                beats[i] = start + i * period;

            return beats;
        }

        private static void assertEvenlySpaced(IReadOnlyList<double> beats, double period, double tolerance)
        {
            for (int i = 1; i < beats.Count; i++)
            {
                Assert.That(beats[i] - beats[i - 1], Is.EqualTo(period).Within(tolerance),
                    $"gap {i} of {beats.Count - 1} is not the period");
            }
        }

        [Test]
        public void TestASteadyTrainIsLeftAlone()
        {
            double[] beats = train(0, 300, 40);
            double[] result = BeatTrainRegulariser.Regularise(beats);

            Assert.That(result.Length, Is.EqualTo(beats.Length));
            assertEvenlySpaced(result, 300, 0.001);
        }

        [Test]
        public void TestAnInsertedBeatIsRemoved()
        {
            // A steady 300ms train with one extra beat fired halfway between two of them.
            var beats = new List<double>(train(0, 300, 40));
            beats.Insert(21, 6150);
            beats.Sort();

            double[] result = BeatTrainRegulariser.Regularise(beats);

            Assert.That(result.Length, Is.EqualTo(40), "the inserted beat has to go");
            assertEvenlySpaced(result, 300, 40);
        }

        [Test]
        public void TestARunOfInsertedBeatsIsRemoved()
        {
            // The model firing on every subdivision for a stretch, which is what it does in dense passages.
            var beats = new List<double>();

            for (int i = 0; i < 20; i++)
                beats.Add(i * 300);

            for (int i = 0; i < 8; i++)
                beats.Add(6000 + i * 150);

            for (int i = 0; i < 20; i++)
                beats.Add(7200 + i * 300);

            beats.Sort();

            double[] result = BeatTrainRegulariser.Regularise(beats);
            double[] expected = train(0, 300, 44);

            Assert.That(result.Length, Is.EqualTo(expected.Length),
                "a stretch at double density has to come back to one beat per period");
            assertEvenlySpaced(result, 300, 40);
        }

        [Test]
        public void TestAMissedBeatIsFilled()
        {
            // Every fourth beat missing, as in a passage where the tracker loses the pulse.
            var beats = new List<double>();

            for (int i = 0; i < 40; i++)
            {
                if (i % 4 != 3)
                    beats.Add(i * 300);
            }

            double[] result = BeatTrainRegulariser.Regularise(beats);

            // 0 to 11400 at one beat per 300ms is 39 beats. The fixture skips index 39 as well as every fourth before
            // it, so the train ends at 38 rather than 39 and 39 is the count that a regular spacing produces.
            Assert.That(result.Length, Is.EqualTo(39), "the missing beats have to come back");
            assertEvenlySpaced(result, 300, 40);
        }

        [Test]
        public void TestABreakInTheMusicIsNotFilled()
        {
            // Two passages of the same track separated by several seconds of nothing, which is not a missed beat.
            var beats = new List<double>(train(0, 300, 20));

            for (int i = 0; i < 20; i++)
                beats.Add(10_000 + i * 300);

            double[] result = BeatTrainRegulariser.Regularise(beats);

            Assert.That(result.Length, Is.EqualTo(40), "a break is not a run of missing beats");

            // And the break itself survives: no pulse was inserted inside it.
            foreach (double beat in result)
                Assert.That(beat < 5900 || beat > 9900, $"a pulse was inserted into the break at {beat}");
        }

        [Test]
        public void TestAGenuineTempoChangeSurvives()
        {
            // Eight seconds at 300ms followed by eight at 200ms: a real change of the kind the visuals must follow.
            var beats = new List<double>();

            for (int i = 0; i < 27; i++)
                beats.Add(i * 300);

            for (int i = 1; i <= 40; i++)
                beats.Add(8100 + i * 200);

            double[] result = BeatTrainRegulariser.Regularise(beats);

            Assert.That(result.Length, Is.GreaterThanOrEqualTo(beats.Count - 2),
                "a real tempo change must not have its beats thrown away as insertions");

            // Counted in each half, the spacing is that half's period.
            int slow = 0;

            for (int i = 1; i < result.Length && result[i] < 8000; i++)
            {
                Assert.That(result[i] - result[i - 1], Is.EqualTo(300).Within(40));
                slow++;
            }

            Assert.That(slow, Is.GreaterThan(20), "the slow half must keep its beats");
        }

        [Test]
        public void TestTooFewBeatsIsPassedThrough()
        {
            double[] beats = { 0, 300, 700 };
            double[] result = BeatTrainRegulariser.Regularise(beats);

            Assert.That(result, Is.EqualTo(beats));
        }
    }
}
