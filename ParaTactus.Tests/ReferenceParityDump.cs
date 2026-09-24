using System;
using System.IO;
using NUnit.Framework;
using ParaTactus;
using ParaTactus.Decoding;

namespace ParaTactus.Tests
{
    /// <summary>
    /// Dumps the decoder output, the frontend output and the model output so the published Python implementation can
    /// be run on exactly the same samples and the two compared stage by stage.
    /// </summary>
    /// <remarks>
    /// Comparing beats end to end cannot say which of the three stages introduced a difference, and each of them has a
    /// different fix. Dumping the intermediate values and feeding the same PCM to the reference removes the decoder as
    /// a variable and leaves the frontend and the chunking to be checked one at a time.
    /// </remarks>
    [TestFixture]
    public class ReferenceParityDump
    {
        [OneTimeSetUp]
        public void SetUp()
        {
            AudioTestEnvironment.Initialise();
        }

        [Test]
        [Explicit("diagnostic: needs OSUTEST_AUDIO, the model file and OSUTEST_DUMP_DIR")]
        public void TestDumpForReferenceParity()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            string directory = Environment.GetEnvironmentVariable("OSUTEST_DUMP_DIR");

            if (string.IsNullOrEmpty(directory))
                Assert.Ignore("Set OSUTEST_DUMP_DIR to where the dumps should go.");

            Directory.CreateDirectory(directory);

            float[] samples = BassAudioDecoder.DecodeMono(audio, BassAudioDecoder.ANALYSIS_SAMPLE_RATE);

            var spectrogram = BeatThisBeatTracker.LogMelSpectrogram(samples);
            var (logits, _) = BeatThisBeatTracker.Logits(spectrogram, model);

            writeFloats(Path.Combine(directory, "pcm.f32"), samples);

            writeFloats(Path.Combine(directory, "spect.f32"), spectrogram);

            writeFloats(Path.Combine(directory, "logits.f32"), logits);

            var filterbank = BeatThisBeatTracker.SlaneyMelFilterbank();
            var flattened = new float[filterbank.GetLength(0) * filterbank.GetLength(1)];

            for (int m = 0; m < filterbank.GetLength(0); m++)
            {
                for (int bin = 0; bin < filterbank.GetLength(1); bin++)
                    flattened[m * filterbank.GetLength(1) + bin] = (float)filterbank[m, bin];
            }

            writeFloats(Path.Combine(directory, "filterbank.f32"), flattened);

            TestContext.Out.WriteLine($"samples {samples.Length}, frames {spectrogram.GetLength(0)}, "
                                      + $"bins {spectrogram.GetLength(1)}");
            TestContext.Out.WriteLine($"wrote pcm.f32, spect.f32 and logits.f32 to {directory}");
        }

        private static void writeFloats(string path, float[] values)
        {
            using var stream = File.Create(path);
            using var writer = new BinaryWriter(stream);

            foreach (float value in values)
                writer.Write(value);
        }

        private static void writeFloats(string path, float[,] values)
        {
            using var stream = File.Create(path);
            using var writer = new BinaryWriter(stream);

            int rows = values.GetLength(0);
            int columns = values.GetLength(1);

            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < columns; c++)
                    writer.Write(values[r, c]);
            }
        }
    }
}
