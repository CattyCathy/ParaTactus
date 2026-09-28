using System;
using System.Collections.Generic;
using NUnit.Framework;
using ParaTactus.Grid;

namespace ParaTactus.Tests
{
    /// <summary>
    /// The tempo sections a beat list implies.
    /// </summary>
    /// <remarks>
    /// Checked on beats whose answer is known, because the corpus cannot settle these: the corpus says whether the
    /// reading is useful, and these say whether it does what it claims. The two that matter are a change that should
    /// split and a fault that should not - a passage at one tempo with a beat missed in it is the case that broke the
    /// first version of this, and it broke it invisibly, because a section that splits still has the right tempo on
    /// both sides of the split.
    /// </remarks>
    public class TempoSectionTest
    {
        /// <summary>Beats at a fixed period, with the listed gaps replaced by the values given.</summary>
        private static List<double> beats(double period, int count, params (int Gap, double Length)[] exceptions)
        {
            var gaps = new double[count];

            for (int i = 0; i < count; i++)
                gaps[i] = period;

            foreach ((int gap, double length) in exceptions)
                gaps[gap] = length;

            var times = new List<double> { 0 };

            for (int i = 0; i < count; i++)
                times.Add(times[^1] + gaps[i]);

            return times;
        }

        [Test]
        public void ASteadyTrackIsOneSection()
        {
            TempoSection[] sections = TempoSections.Analyse(beats(300, 400));

            Assert.That(sections, Has.Length.EqualTo(1));
            Assert.That(sections[0].Period, Is.EqualTo(300).Within(0.01));
            Assert.That(sections[0].BeatsPerMinute, Is.EqualTo(200).Within(0.01));
        }

        [Test]
        public void AChangeOfTempoIsTwoSections()
        {
            // Two hundred beats at 300ms then two hundred at 400ms. The boundary is looked for by comparing the tempo
            // either side of it, so it lands near the change rather than at it - what is asked here is that the two
            // tempos are right and that the boundary is within a beat of the join.
            var times = new List<double>();
            double at = 0;

            for (int i = 0; i < 200; i++)
            {
                times.Add(at);
                at += 300;
            }

            for (int i = 0; i < 200; i++)
            {
                times.Add(at);
                at += 400;
            }

            TempoSection[] sections = TempoSections.Analyse(times);

            Assert.That(sections, Has.Length.EqualTo(2), "a doubling of the beat is one change and not several");
            Assert.That(sections[0].Period, Is.EqualTo(300).Within(1));
            Assert.That(sections[1].Period, Is.EqualTo(400).Within(1));
            Assert.That(sections[0].End, Is.EqualTo(sections[1].Start).Within(0.01));

            double join = 200 * 300.0;

            // A boundary cannot be found until the tempo on both sides of it has been measured, so it trails the change
            // by about half a window of gaps - which at this window and tempo is a couple of seconds. What matters is
            // that it is that and not an arbitrary distance, and that it never leads the change, which would be a
            // boundary found from music that had not happened yet.
            double window = TempoSections.Window * 300.0;

            Assert.That(sections[1].Start - join, Is.GreaterThan(-1),
                        "a boundary is not placed before the change it was found from");
            Assert.That(sections[1].Start - join, Is.LessThanOrEqualTo(window),
                        "the boundary should trail the join by about a window of gaps, not by more");
        }

        [Test]
        public void AMissedBeatIsNotAChangeOfTempo()
        {
            // One gap of double length in the middle of a steady passage. This is what a detector produces when it
            // loses a beat, and it is not a tempo change however much it looks like one to a rule that reads the gap
            // gap on its own.
            TempoSection[] sections = TempoSections.Analyse(beats(300, 400, (200, 600.0)));

            Assert.That(sections, Has.Length.EqualTo(1), "a missed beat is not a change of tempo");
            Assert.That(sections[0].Period, Is.EqualTo(300).Within(1));
        }

        [Test]
        public void SeveralMissedBeatsInARowAreNotAChangeOfTempo()
        {
            // Three missed beats together, which is a longer fault than the neighbourhood used to find one can absorb.
            TempoSection[] sections = TempoSections.Analyse(beats(300, 400, (200, 900.0), (201, 900.0), (202, 900.0)));

            Assert.That(sections, Has.Length.EqualTo(1), "a run of missed beats is still not a change of tempo");
            Assert.That(sections[0].Period, Is.EqualTo(300).Within(1));
        }

        [Test]
        public void ABadlyPlacedBeatDoesNotMoveTheTempo()
        {
            // A beat placed a third of a period late, which is a placement error rather than a missed beat. The period
            // is a middle over the section and a middle is what makes one bad beat unable to move it.
            TempoSection[] sections = TempoSections.Analyse(beats(300, 400, (200, 400.0)));

            Assert.That(sections, Has.Length.EqualTo(1));
            Assert.That(sections[0].Period, Is.EqualTo(300).Within(2),
                        "one badly placed beat should not be able to move the tempo");
        }

        [Test]
        public void AShortPassageIsJoinedToItsNeighbour()
        {
            // Four beats at another tempo, which is shorter than a section may be. Reported as its own tempo it would
            // be a tempo change the player sees for a third of a second, so it is joined.
            var times = new List<double>();
            double at = 0;

            for (int i = 0; i < 200; i++)
            {
                times.Add(at);
                at += 300;
            }

            for (int i = 0; i < 4; i++)
            {
                times.Add(at);
                at += 400;
            }

            for (int i = 0; i < 200; i++)
            {
                times.Add(at);
                at += 300;
            }

            TempoSection[] sections = TempoSections.Analyse(times);

            Assert.That(sections.Length, Is.LessThanOrEqualTo(2),
                        "a passage too short to be a section is not reported as one");
        }

        [Test]
        public void TooFewBeatsGiveNothing()
        {
            Assert.That(TempoSections.Analyse(new List<double>()), Is.Empty);
            Assert.That(TempoSections.Analyse(new List<double> { 0, 300 }), Is.Empty);
        }

        [Test]
        public void TheSectionsCoverTheBeatsWithoutGapsOrOverlap()
        {
            TempoSection[] sections = TempoSections.Analyse(beats(300, 200, (60, 600.0), (140, 450.0)));

            Assert.That(sections.Length, Is.GreaterThan(0));

            for (int i = 1; i < sections.Length; i++)
            {
                Assert.That(sections[i].Start, Is.EqualTo(sections[i - 1].End).Within(0.01),
                            "one section has to begin where the one before it ended");
                Assert.That(sections[i].Start, Is.GreaterThan(sections[i - 1].Start));
            }
        }
    }
}
