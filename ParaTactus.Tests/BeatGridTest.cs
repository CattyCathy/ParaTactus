using System;
using NUnit.Framework;
using ParaTactus;

namespace ParaTactus.Tests
{
    /// <summary>
    /// The beat grid a player drives an animation from: phase inside a beat, and tempo over a window.
    /// </summary>
    [TestFixture]
    public class BeatGridTest
    {
        private static double[] Steady(double bpm, int count, double offset = 0)
        {
            var beats = new double[count];
            double period = 60000 / bpm;

            for (int i = 0; i < count; i++)
                beats[i] = offset + i * period;

            return beats;
        }

        [Test]
        public void PhaseRunsFromZeroAtABeatToJustUnderOneAtTheNext()
        {
            var grid = BeatGrid.FromBeats(Steady(120, 20));

            Assert.That(grid.PhaseAt(0), Is.EqualTo(0).Within(1e-9));
            Assert.That(grid.PhaseAt(250), Is.EqualTo(0.5).Within(1e-9));
            Assert.That(grid.PhaseAt(499), Is.EqualTo(0.998).Within(1e-3));

            // Halfway between the first two beats belongs to the first, not the second.
            Assert.That(grid.IndexAt(250), Is.Zero);
            Assert.That(grid.IndexAt(500), Is.EqualTo(1));
        }

        [Test]
        public void PhaseIsZeroBeforeTheFirstBeat()
        {
            var grid = BeatGrid.FromBeats(Steady(120, 10, 1000));

            Assert.That(grid.IndexAt(500), Is.EqualTo(-1));
            Assert.That(grid.PhaseAt(500), Is.Zero);
        }

        [Test]
        public void TempoIsReadOverAWindowRatherThanOneInterval()
        {
            // A steady grid with a single interval doubled, as one dropped beat would look. Read from that interval
            // alone the tempo would read 60; read over the window around it the median is unmoved.
            var beats = new System.Collections.Generic.List<double>();

            for (int i = 0; i < 40; i++)
            {
                // Everything after the gap is shifted by one beat, so the gap stays a single anomaly rather than
                // becoming the tempo of the rest of the track.
                double shift = i >= 20 ? 500 : 0;
                beats.Add(i * 500.0 + shift);
            }

            var grid = BeatGrid.FromBeats(beats);

            Assert.That(beats[20] - beats[19], Is.EqualTo(1000), "the fixture should have exactly one doubled gap");
            Assert.That(beats[21] - beats[20], Is.EqualTo(500), "and the rest should be steady");
            Assert.That(grid.BpmAt(beats[19]), Is.EqualTo(120).Within(1));
            Assert.That(grid.BpmAt(beats[10]), Is.EqualTo(120).Within(1));
        }

        [Test]
        public void TempoFollowsATempoChange()
        {
            // Two steady halves an octave apart. The window is short enough that the tempo on either side is the tempo
            // of that side, which is the whole point of reporting a curve rather than one number.
            var beats = new System.Collections.Generic.List<double>();
            double position = 0;

            for (int i = 0; i < 40; i++)
            {
                beats.Add(position);
                position += i < 20 ? 500 : 250;
            }

            var grid = BeatGrid.FromBeats(beats);

            Assert.That(grid.BpmAt(beats[5]), Is.EqualTo(120).Within(1));
            Assert.That(grid.BpmAt(beats[35]), Is.EqualTo(240).Within(1));
        }

        [Test]
        public void AnEmptyTrackHasNoTempoAndNoPhase()
        {
            var grid = BeatGrid.FromBeats(new double[0]);

            Assert.That(grid.IsEmpty, Is.True);
            Assert.That(grid.BpmAt(0), Is.Zero);
            Assert.That(grid.PhaseAt(0), Is.Zero);
            Assert.That(grid.Duration, Is.Zero);
        }

        [Test]
        public void AMetricalShiftIsReportedAndApplied()
        {
            // A whole track at 240 BPM is outside tapping range, so it is halved to 120 and says so.
            var grid = BeatGrid.FromBeats(Steady(240, 64));

            Assert.That(grid.MetricalShift, Is.EqualTo(1));
            Assert.That(grid.Beats.Count, Is.EqualTo(32));
            Assert.That(grid.BpmAt(grid.Beats[10]), Is.EqualTo(120).Within(1));
        }

        [Test]
        public void AComfortableTrackKeepsItsParaTactus()
        {
            // The Designant case: a clean 100 BPM pulse the map calls 200 stays at 100.
            var grid = BeatGrid.FromBeats(Steady(100, 64));

            Assert.That(grid.MetricalShift, Is.Zero);
            Assert.That(grid.Beats.Count, Is.EqualTo(64));
            Assert.That(grid.BpmAt(grid.Beats[30]), Is.EqualTo(100).Within(1));
        }
    }
}
