using System;
using System.Collections.Generic;
using NUnit.Framework;
using ParaTactus;

namespace ParaTactus.Tests
{
    /// <summary>
    /// Deciding whether the tracker has halved the tactus, using the audio rather than the tracker.
    /// </summary>
    /// <remarks>
    /// Synthetic envelopes, because the question is arithmetic: given onsets at known places, does the comparison
    /// notice when the beats only land on every other one. The behaviour on a real track is measured separately and
    /// reported, not asserted, because what it should say for a given track is the thing being found out.
    /// </remarks>
    [TestFixture]
    public class OnsetEnvelopeTest
    {
        private const int fps = 50;

        /// <summary>An envelope that is one at the given frames and a small floor everywhere else.</summary>
        private static double[] envelopeAt(int frames, params int[] onsets)
        {
            var envelope = new double[frames];

            for (int i = 0; i < frames; i++)
                envelope[i] = 0.05;

            foreach (int onset in onsets)
                envelope[onset] = 1;

            return envelope;
        }

        private static List<double> beatsEvery(int count, double period, double start = 0)
        {
            var beats = new List<double>();

            for (int i = 0; i < count; i++)
                beats.Add(start + i * period);

            return beats;
        }

        [Test]
        public void TestOnsetsOnEveryBeatAreNotHalved()
        {
            // Onsets every 300ms and beats every 300ms: the tracker is at the music's own rate.
            const int frames = 400;
            var onsets = new List<int>();

            for (int frame = 0; frame < frames; frame += 15)
                onsets.Add(frame);

            double[] envelope = envelopeAt(frames, onsets.ToArray());
            var (atBeats, atMidpoints, count) = OnsetEnvelope.Compare(beatsEvery(20, 300), envelope, frames);

            TestContext.Out.WriteLine($"at beats {atBeats:0.###}, at midpoints {atMidpoints:0.###}, n {count}");

            Assert.That(count, Is.EqualTo(19));
            Assert.That(OnsetEnvelope.LooksHalved(atBeats, atMidpoints), Is.False,
                "music with a beat every 300ms is not halved by a tracker reporting every 300ms");
        }

        [Test]
        public void TestOnsetsOnEveryOtherBeatAreHalved()
        {
            // Onsets every 300ms but the tracker only reporting every 600ms: the tactus has been halved.
            const int frames = 400;
            var onsets = new List<int>();

            for (int frame = 0; frame < frames; frame += 15)
                onsets.Add(frame);

            double[] envelope = envelopeAt(frames, onsets.ToArray());
            var (atBeats, atMidpoints, _) = OnsetEnvelope.Compare(beatsEvery(20, 600), envelope, frames);

            TestContext.Out.WriteLine($"at beats {atBeats:0.###}, at midpoints {atMidpoints:0.###}");

            Assert.That(OnsetEnvelope.LooksHalved(atBeats, atMidpoints), Is.True,
                "a tracker reporting every 600ms against onsets every 300ms has halved the tactus");
        }

        [Test]
        public void TestMusicWithNoOnsetsInBetweenIsNotHalved()
        {
            // The tracker reporting every 600ms and the music genuinely having a beat only every 600ms.
            const int frames = 400;
            var onsets = new List<int>();

            for (int frame = 0; frame < frames; frame += 30)
                onsets.Add(frame);

            double[] envelope = envelopeAt(frames, onsets.ToArray());
            var (atBeats, atMidpoints, _) = OnsetEnvelope.Compare(beatsEvery(20, 600), envelope, frames);

            TestContext.Out.WriteLine($"at beats {atBeats:0.###}, at midpoints {atMidpoints:0.###}");

            Assert.That(OnsetEnvelope.LooksHalved(atBeats, atMidpoints), Is.False,
                "a real 100 BPM passage must not be mistaken for a halved 200 BPM one");
        }

        [Test]
        public void TestTooFewBeatsIsReportedAsSuch()
        {
            double[] envelope = envelopeAt(100, 10, 25, 40);
            var (_, _, count) = OnsetEnvelope.Compare(new List<double> { 200, 400, 600 }, envelope, 100);

            Assert.That(count, Is.Zero);
            Assert.That(OnsetEnvelope.LooksHalved(0, 0), Is.False);
        }
    }
}
