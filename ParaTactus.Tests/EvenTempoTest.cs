using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using ParaTactus;

namespace ParaTactus.Tests
{
    /// <summary>
    /// The even-tempo layout a display can draw: each passage of one tempo as an exactly even pulse.
    /// </summary>
    /// <remarks>
    /// Separate from the tracked beats on purpose and tested on its own, because it is not an improvement on them and
    /// is not meant to be. It is the same passages with the extra beats taken out and the missing ones put back, so it
    /// is even where they are not and further from the music where they are right - which is a thing to look at and
    /// decide about, so what is tested here is that it does what it claims rather than that it is better.
    /// </remarks>
    public class EvenTempoTest
    {
        /// <summary>Beats at a fixed period, with the listed gaps replaced before they are turned into times.</summary>
        private static double[] beatsWith(double period, int count, params (int Gap, double Length)[] exceptions)
        {
            var gaps = new double[count];

            for (int i = 0; i < count; i++)
                gaps[i] = period;

            foreach ((int gap, double length) in exceptions)
                gaps[gap] = length;

            var times = new List<double> { 0 };

            for (int i = 0; i < count; i++)
                times.Add(times[^1] + gaps[i]);

            return times.ToArray();
        }

        private static double[] gaps(IReadOnlyList<double> beats)
        {
            var out_ = new double[beats.Count - 1];

            for (int i = 0; i < out_.Length; i++)
                out_[i] = beats[i + 1] - beats[i];

            return out_;
        }

        /// <summary>The share of gaps within a quarter of the local period, which is what evenness means here.</summary>
        private static double uniformly(IReadOnlyList<double> beats, int width = 8)
        {
            var g = gaps(beats);
            var local = new double[g.Length];

            for (int i = 0; i < g.Length; i++)
            {
                int low = Math.Max(0, i - width);
                int high = Math.Min(g.Length, i + width);
                var window = new List<double>();

                for (int j = low; j < high; j++)
                    window.Add(g[j]);

                window.Sort();
                local[i] = window[window.Count / 2];
            }

            int near = 0;

            for (int i = 0; i < g.Length; i++)
            {
                if (local[i] > 0 && Math.Abs((g[i] / local[i]) - 1) <= 0.25)
                    near++;
            }

            return 100.0 * near / g.Length;
        }

        [Test]
        public void ASteadyTrackIsLeftAsAnEvenPulse()
        {
            var grid = BeatGrid.FromBeats(beatsWith(417, 200));
            IReadOnlyList<double> even = grid.EvenBeats;

            Assert.That(even, Is.Not.Null);
            Assert.That(uniformly(even), Is.EqualTo(100).Within(0.01));

            foreach (double gap in gaps(even))
                Assert.That(gap, Is.EqualTo(417).Within(0.01), "every gap should be the passage's own period");
        }

        [Test]
        public void AnExtraBeatIsTakenOut()
        {
            // One gap split in two, which is what the tracker does every twenty or thirty beats.
            var grid = BeatGrid.FromBeats(beatsWith(417, 240, (100, 200.0), (101, 217.0)));
            IReadOnlyList<double> even = grid.EvenBeats;

            Assert.That(even, Is.Not.Null);
            Assert.That(even.Count, Is.EqualTo(239), "the inserted beat should be gone");
            Assert.That(uniformly(even), Is.EqualTo(100).Within(0.01));
        }

        [Test]
        public void AMissedBeatIsPutBack()
        {
            // One gap of two periods, which is what the tracker does every fifty beats.
            var grid = BeatGrid.FromBeats(beatsWith(417, 240, (100, 834.0)));
            IReadOnlyList<double> even = grid.EvenBeats;

            Assert.That(even, Is.Not.Null);
            Assert.That(even.Count, Is.EqualTo(241), "the missed beat should be back");
            Assert.That(uniformly(even), Is.EqualTo(100).Within(0.01));
        }

        [Test]
        public void AChangeOfTempoLaysOutEachPassageAtItsOwnPeriod()
        {
            var times = new List<double>();
            double at = 0;

            for (int i = 0; i < 200; i++)
            {
                times.Add(at);
                at += 400;
            }

            for (int i = 0; i < 200; i++)
            {
                times.Add(at);
                at += 600;
            }

            IReadOnlyList<double> even = BeatGrid.FromBeats(times.ToArray()).EvenBeats;

            Assert.That(even, Is.Not.Null);
            Assert.That(uniformly(even), Is.EqualTo(100).Within(0.01));

            // Both tempos should still be there, at roughly a third of the beats each way.
            var g = gaps(even);

            Assert.That(g.Count(x => Math.Abs(x - 400) < 20), Is.GreaterThan(100), "the first passage should hold 400ms");
            Assert.That(g.Count(x => Math.Abs(x - 600) < 20), Is.GreaterThan(100), "the second should hold 600ms");
        }

        [Test]
        public void AGridWithNoPassageHasNoEvenLayout()
        {
            // Four beats is below what a passage can be measured from, so there is nothing to lay out and the caller
            // has to fall back to the tracked beats rather than drawing nothing.
            Assert.That(BeatGrid.FromBeats(beatsWith(417, 4)).EvenBeats, Is.Null);
            Assert.That(BeatGrid.FromBeats(Array.Empty<double>()).EvenBeats, Is.Null);
        }

        [Test]
        public void WithEvenTempoLeavesTheTrackedBeatsAlone()
        {
            double[] tracked = beatsWith(417, 240, (100, 200.0), (101, 217.0));
            var grid = BeatGrid.FromBeats(tracked);
            BeatGrid even = grid.WithEvenTempo();

            Assert.That(grid.Beats.Count, Is.EqualTo(tracked.Length), "the tracked beats must not be disturbed");
            Assert.That(even.Beats.Count, Is.EqualTo(239));

            for (int i = 0; i < tracked.Length; i++)
                Assert.That(grid.Beats[i], Is.EqualTo(tracked[i]).Within(1e-9));
        }
    }
}
