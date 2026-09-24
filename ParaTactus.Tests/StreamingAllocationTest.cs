using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NUnit.Framework;
using ParaTactus;

namespace ParaTactus.Tests
{
    /// <summary>
    /// Feeding audio in must not cost more as the track grows.
    /// </summary>
    /// <remarks>
    /// The streaming tracker holds every sample it is given, which is right - a chunk is analysed with the audio before
    /// it - but it used to hand the whole list to the frontend as an array on every chunk: <c>samples.ToArray()</c>
    /// inside the chunk loop, so a copy of the entire history once per chunk and quadratic work over a track. Nothing
    /// about the output changes, which is why it needs a test of its own.
    /// </remarks>
    [TestFixture]
    public class StreamingAllocationTest
    {
        /// <summary>0.2s per call: ten frames, far under the model's own chunk of 1500.</summary>
        private const int chunkSamples = AnalysisAudio.SampleRate / 5;

        /// <summary>
        /// One model chunk's stride, which is how much audio buys exactly one analysis.
        /// </summary>
        /// <remarks>
        /// Adding a stride at a time makes every call do the same work, so the cost per call can be compared between
        /// the start of a track and its end. With smaller chunks the analyses land on a few calls out of many and the
        /// measurement is mostly noise.
        /// </remarks>
        private static readonly int strideSamples =
            (BeatThisBeatTracker.ModelChunkFrames - 2 * BeatThisBeatTracker.ModelBorderFrames) * 441;

        private const int calls = 10;

        private static long medianOf(List<long> values)
        {
            long[] sorted = values.OrderBy(v => v).ToArray();

            return sorted.Length % 2 == 0 ? (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2 : sorted[sorted.Length / 2];
        }

        private static string modelPath()
        {
            return Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                   ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");
        }

        [Test]
        public void TheFrontendReadsTheSameFramesFromAListAsFromAnArray()
        {
            var samples = new float[AnalysisAudio.SampleRate];

            for (int i = 0; i < samples.Length; i++)
                samples[i] = (float)Math.Sin(2 * Math.PI * 220 * i / AnalysisAudio.SampleRate);

            float[,] fromArray = BeatThisBeatTracker.LogMelSpectrogram(samples);
            float[,] fromList = BeatThisBeatTracker.LogMelSpectrogram(
                new List<float>(samples), 0, BeatThisBeatTracker.FrameCount(samples.Length));

            Assert.That(fromList, Is.EqualTo(fromArray));
        }

        [Test]
        public void AddingAudioDoesNotCopyTheHistory()
        {
            string model = modelPath();

            if (!File.Exists(model))
                Assert.Ignore("Set OSUTEST_MODEL to the ONNX file.");

            using var tracker = new StreamingBeatTracker(model);

            var chunk = new float[strideSamples];
            var allocations = new List<long>();

            for (int call = 0; call < calls; call++)
            {
                for (int i = 0; i < chunk.Length; i++)
                    chunk[i] = (float)Math.Sin(2 * Math.PI * 120 * (call * (long)chunk.Length + i) / AnalysisAudio.SampleRate);

                long before = GC.GetAllocatedBytesForCurrentThread();
                tracker.Add(chunk);
                allocations.Add(GC.GetAllocatedBytesForCurrentThread() - before);
            }

            // Medians rather than peaks, because two things dominate a single call and neither is what is under test:
            // the model session and its arenas, which are allocated once, and the list's capacity doubling, which
            // happens a handful of times. What is under test is what every call costs, and with the history copied per
            // chunk that cost grows with the length of the track.
            long firstHalf = medianOf(allocations.Take(calls / 2).ToList());
            long secondHalf = medianOf(allocations.Skip(calls / 2).ToList());
            long historyGrowth = calls / 2L * strideSamples * sizeof(float);

            Assert.That(tracker.SampleCount, Is.EqualTo((long)calls * strideSamples), "every sample added should be remembered");

            Assert.That(
                secondHalf,
                Is.LessThan(firstHalf + historyGrowth / 2),
                $"the second half of the track costs a median of {secondHalf} bytes a call against {firstHalf} for the "
                + $"first half, and the {calls / 2} extra chunks are {historyGrowth} bytes of history - a cost that grew "
                + "by the history is a copy of it");
        }
    }
}
