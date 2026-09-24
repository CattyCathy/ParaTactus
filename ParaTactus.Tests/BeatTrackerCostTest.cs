using System;
using System.Diagnostics;
using System.IO;
using NUnit.Framework;
using ParaTactus;
using ParaTactus.Decoding;

namespace ParaTactus.Tests
{
    /// <summary>
    /// What the model path costs, stage by stage.
    /// </summary>
    /// <remarks>
    /// A player cannot spend longer analysing a track than it takes to play it, and the first measurement of this path
    /// was 48 seconds of Release build for a 176 second track - about a quarter of realtime, which would be unusable.
    /// Where that time goes decides what to do about it: a slow frontend is a tight loop to optimise, whereas slow
    /// inference means the model itself is the wrong size for the job and no amount of tidying the analyse code helps.
    /// </remarks>
    [TestFixture]
    public class BeatTrackerCostTest
    {
        [OneTimeSetUp]
        public void SetUp()
        {
            AudioTestEnvironment.Initialise();
        }

        [Test]
        [Explicit("performance; needs OSUTEST_AUDIO and the model file")]
        public void TestStageCosts()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            var clock = Stopwatch.StartNew();

            float[] samples = BassAudioDecoder.DecodeMono(audio, 22050);
            double decode = clock.Elapsed.TotalSeconds;

            clock.Restart();
            float[,] spectrogram = BeatThisBeatTracker.LogMelSpectrogram(samples);
            double frontend = clock.Elapsed.TotalSeconds;

            clock.Restart();
            BeatThisBeatTracker.Predict(spectrogram, model);
            double inference = clock.Elapsed.TotalSeconds;

            double duration = samples.Length / 22050.0;
            double total = decode + frontend + inference;

            TestContext.Out.WriteLine($"track:     {duration:0.0}s of audio, {spectrogram.GetLength(0)} frames");
            TestContext.Out.WriteLine($"decode:    {decode,7:0.00}s");
            TestContext.Out.WriteLine($"frontend:  {frontend,7:0.00}s  ({frontend / duration * 100:0.0}% of realtime)");
            TestContext.Out.WriteLine($"inference: {inference,7:0.00}s  ({inference / duration * 100:0.0}% of realtime)");
            TestContext.Out.WriteLine($"total:     {total,7:0.00}s  ({total / duration:0.00}x realtime)");

            // Loading the session is a one-off cost, but it is paid per analysis unless the session is kept, so it is
            // measured separately from the inference itself.
            clock.Restart();
            using (var session = new Microsoft.ML.OnnxRuntime.InferenceSession(model))
            {
                double load = clock.Elapsed.TotalSeconds;
                TestContext.Out.WriteLine($"session load: {load:0.00}s");
            }
        }

        /// <summary>
        /// Whether inference is actually using the machine's cores.
        /// </summary>
        /// <remarks>
        /// Inference is the whole cost of this path, so before treating the model as too slow for a player this rules
        /// out the cheapest explanation: a session running single-threaded by default would make the same model many
        /// times faster for free, and that would be a configuration mistake rather than a property of the model.
        /// </remarks>
        [Test]
        [Explicit("performance; needs OSUTEST_AUDIO and the model file")]
        public void TestInferenceThreading()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            float[] samples = BassAudioDecoder.DecodeMono(audio, 22050);
            float[,] spectrogram = BeatThisBeatTracker.LogMelSpectrogram(samples);

            const int frames = 1500;
            var input = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(new[] { 1, frames, 128 });

            for (int t = 0; t < frames; t++)
            {
                for (int m = 0; m < 128; m++)
                    input[0, t, m] = spectrogram[t, m];
            }

            TestContext.Out.WriteLine($"logical processors: {Environment.ProcessorCount}");

            foreach (int threads in new[] { 0, Environment.ProcessorCount })
            {
                var options = new Microsoft.ML.OnnxRuntime.SessionOptions();

                if (threads > 0)
                    options.IntraOpNumThreads = threads;

                using var session = new Microsoft.ML.OnnxRuntime.InferenceSession(model, options);
                var inputs = new System.Collections.Generic.List<Microsoft.ML.OnnxRuntime.NamedOnnxValue>
                {
                    Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor("spectrogram", input),
                };

                // One untimed run, so the measurement is of steady state rather than of first-call setup.
                using (session.Run(inputs))
                {
                }

                var clock = Stopwatch.StartNew();
                using (session.Run(inputs))
                {
                }

                double seconds = clock.Elapsed.TotalSeconds;
                string label = threads == 0 ? "default" : $"{threads} threads";

                TestContext.Out.WriteLine($"30s chunk, {label,-12}: {seconds:0.00}s  ({seconds / 30 * 100:0.0}% of realtime)");
            }
        }
    }
}
