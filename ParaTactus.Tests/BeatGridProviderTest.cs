using System;
using System.Diagnostics;
using System.IO;
using NUnit.Framework;
using ParaTactus;
using ParaTactus.Decoding;

namespace ParaTactus.Tests
{
    /// <summary>
    /// The provider end to end: analysing a real track once, and serving it from the cache afterwards.
    /// </summary>
    [TestFixture]
    public class BeatGridProviderTest
    {
        private string directory;

        [OneTimeSetUp]
        public void SetUp()
        {
            AudioTestEnvironment.Initialise();
        }

        [SetUp]
        public void CreateCacheDirectory()
        {
            directory = Path.Combine(Path.GetTempPath(), "paratactus-provider-" + Guid.NewGuid().ToString("n").Substring(0, 8));
        }

        [TearDown]
        public void RemoveCacheDirectory()
        {
            try
            {
                if (System.IO.Directory.Exists(directory))
                    System.IO.Directory.Delete(directory, true);
            }
            catch (IOException)
            {
            }
        }

        [Test]
        public void TestACachedGridIsServedWithoutAnalysing()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            var provider = new BeatGridProvider(model, directory, BassAudioDecoder.Default);

            Assert.That(provider.IsCached(audio), Is.False, "a fresh cache directory should hold nothing");

            // A grid put in the cache by hand, with a beat count and a metrical level the real track cannot produce.
            // If the provider analyses instead of reading, the answer will be hundreds of beats at level 0 and this is
            // unambiguous; timing alone would only say it was quick.
            string key = BeatGridCache.KeyFor(audio, model);
            provider.Cache.Store(key, BeatGrid.FromNormalisedBeats(new[] { 1000.0, 2000.0, 3000.0, 4000.0 }, 3));

            Assert.That(provider.IsCached(audio), Is.True);

            var clock = Stopwatch.StartNew();
            BeatGrid served = provider.Get(audio);
            double seconds = clock.Elapsed.TotalSeconds;

            TestContext.Out.WriteLine($"served {served.Beats.Count} beats at level {served.MetricalShift} in {seconds:0.000}s");

            Assert.That(served.Beats.Count, Is.EqualTo(4), "the provider analysed instead of using the cached grid");
            Assert.That(served.MetricalShift, Is.EqualTo(3));
            Assert.That(served.Beats[2], Is.EqualTo(3000).Within(1e-6));
        }

        [Test]
        public void TestAnAnalysisIsStoredAndMatchesTheStreamedGrid()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            var provider = new BeatGridProvider(model, directory, BassAudioDecoder.Default);
            BeatGrid grid = provider.Get(audio);

            TestContext.Out.WriteLine($"analysed {grid.Beats.Count} beats, metrical shift {grid.MetricalShift}");

            Assert.That(grid.IsEmpty, Is.False, "the analysis produced no beats at all");
            Assert.That(provider.IsCached(audio), Is.True, "the analysis was not stored");

            // The path a player takes when the track is not in the cache: fed a third of a second at a time, as audio
            // arrives.
            float[] samples = BassAudioDecoder.DecodeMono(audio, BassAudioDecoder.ANALYSIS_SAMPLE_RATE);
            var tracker = provider.BeginStreaming();

            const int piece = 22050 / 3;

            for (int start = 0; start < samples.Length; start += piece)
            {
                int length = Math.Min(piece, samples.Length - start);
                var buffer = new float[length];

                Array.Copy(samples, start, buffer, 0, length);
                tracker.Add(buffer);
            }

            tracker.Flush();

            TestContext.Out.WriteLine($"cached grid: {grid.Beats.Count} beats, streamed: {tracker.Beats.Count} beats");

            // Regularised the way the provider regularises, so this compares like with like: the provider stores what
            // the regulariser produced, and raw streamed beats are a different pipeline - which is what this test
            // compared while it was [Explicit] and never ran to the end.
            BeatGrid streamed = BeatGrid.FromBeats(BeatTrainRegulariser.Regularise(tracker.Beats));

            // The level is chosen from the track, so both should land on the same octave.
            Assert.That(streamed.MetricalShift, Is.EqualTo(grid.MetricalShift));
            Assert.That(streamed.Beats.Count, Is.EqualTo(grid.Beats.Count).Within(2));
        }
    }
}
