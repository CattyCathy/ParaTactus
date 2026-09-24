using System.Collections.Generic;
using NUnit.Framework;
using ParaTactus;

namespace ParaTactus.Tests
{
    /// <summary>
    /// The metrical level policy: keep the model's tactus unless the whole track is outside tapping range.
    /// </summary>
    [TestFixture]
    public class MetricalLevelTest
    {
        private static double[] Steady(double bpm, int count)
        {
            var beats = new double[count];
            double period = 60000 / bpm;

            for (int i = 0; i < count; i++)
                beats[i] = i * period;

            return beats;
        }

        [Test]
        public void AComfortableParaTactusIsKept()
        {
            // The case this exists for: the model reports a clean 100 BPM where the map declares 200, and the tactus
            // is what a listener taps, so it stands.
            Assert.That(MetricalLevel.ChooseShift(Steady(100, 120)), Is.Zero);
            Assert.That(MetricalLevel.ChooseShift(Steady(140, 120)), Is.Zero);
            Assert.That(MetricalLevel.ChooseShift(Steady(179, 120)), Is.Zero);

            // 200 BPM is fast but tappable, and halving it is exactly what made the visuals run at half the speed of
            // the music, so it is left alone.
            Assert.That(MetricalLevel.ChooseShift(Steady(200, 120)), Is.Zero);
            Assert.That(MetricalLevel.ChooseShift(Steady(219, 120)), Is.Zero);
        }

        [Test]
        public void ATrackThatIsEntirelyTooFastIsHalved()
        {
            int shift = MetricalLevel.ChooseShift(Steady(250, 120));

            Assert.That(shift, Is.EqualTo(1));

            double[] normalised = MetricalLevel.Apply(Steady(250, 120), shift);

            Assert.That(normalised.Length, Is.EqualTo(60));
            Assert.That(normalised[1] - normalised[0], Is.EqualTo(60000 / 125.0).Within(1e-6));
        }

        [Test]
        public void ATrackThatIsEntirelyTooSlowIsDoubled()
        {
            int shift = MetricalLevel.ChooseShift(Steady(50, 60));

            Assert.That(shift, Is.EqualTo(-1));

            double[] normalised = MetricalLevel.Apply(Steady(50, 60), shift);

            // Doubling fills the gaps rather than inventing beats past the end, so one fewer than twice as many.
            Assert.That(normalised.Length, Is.EqualTo(119));
            Assert.That(normalised[1] - normalised[0], Is.EqualTo(60000 / 100.0).Within(1e-6));
        }

        [Test]
        public void AMixedTempoTrackKeepsItsParaTactus()
        {
            // A ramp from 150 to 400 BPM has a median near 240, so folding by the median alone would halve the slow
            // end that was already right. No single side holds a majority, so nothing moves.
            var beats = new List<double> { 0 };
            double position = 0;

            for (int i = 0; i < 200; i++)
            {
                double bpm = 150 + 250.0 * i / 200;
                position += 60000 / bpm;
                beats.Add(position);
            }

            Assert.That(MetricalLevel.ChooseShift(beats), Is.Zero);
        }

        [Test]
        public void TheSmallestMoveWins()
        {
            // 250 is one halving away from the range, 500 is two, and neither should overshoot to below 80.
            Assert.That(MetricalLevel.ShiftFor(250), Is.EqualTo(1));
            Assert.That(MetricalLevel.ShiftFor(500), Is.EqualTo(2));
            Assert.That(MetricalLevel.ShiftFor(75), Is.EqualTo(-1));
            Assert.That(MetricalLevel.ShiftFor(120), Is.Zero);

            // Anything inside the range is left where it is, including the fast end that used to be halved.
            Assert.That(MetricalLevel.ShiftFor(200), Is.Zero);
            Assert.That(MetricalLevel.ShiftFor(80), Is.Zero);
        }

        [Test]
        public void AZeroShiftChangesNothing()
        {
            double[] beats = Steady(120, 10);
            double[] normalised = MetricalLevel.Normalise(beats, out int shift);

            Assert.That(shift, Is.Zero);
            Assert.That(normalised, Is.EqualTo(beats));
        }

        [Test]
        public void AnEmptyGridIsHandled()
        {
            Assert.That(MetricalLevel.ChooseShift(new double[0]), Is.Zero);
            Assert.That(MetricalLevel.Apply(new double[0], 2), Is.Empty);
            Assert.That(MetricalLevel.Apply(new double[0], -2), Is.Empty);
        }
    }
}
