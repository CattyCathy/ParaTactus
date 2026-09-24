using System;
using System.IO;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using NUnit.Framework;
using ParaTactus;

namespace ParaTactus.Tests
{
    /// <summary>
    /// Reports what a beat-tracking ONNX model expects and produces.
    /// </summary>
    /// <remarks>
    /// Nothing about running such a model can be written before this is known, and it is not guessable: the input is
    /// some form of spectrogram, and the mel frontend that produced it during training has to be reproduced exactly -
    /// sample rate, window, mel band count, frequency range, hop, normalisation. A frontend that is subtly wrong does
    /// not fail, it just makes the model quietly worse, which is the worst possible failure mode for a component whose
    /// whole job is to be more accurate than the hand-written alternative.
    /// </remarks>
    [TestFixture]
    public class BeatThisModelProbe
    {
        [Test]
        [Explicit("introspection; needs the model file")]
        public void DescribeModel()
        {
            string path = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                          ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(path), Is.True, $"model not found at {path}");

            var report = new StringBuilder();
            var file = new FileInfo(path);
            report.AppendLine($"model: {path}");
            report.AppendLine($"size:  {file.Length / 1024.0 / 1024.0:0.##} MB");

            using var session = new InferenceSession(path);

            foreach (var input in session.InputMetadata)
                report.AppendLine($"  IN  {input.Key}: {input.Value.ElementType} [{string.Join(", ", input.Value.Dimensions)}]");

            foreach (var output in session.OutputMetadata)
                report.AppendLine($"  OUT {output.Key}: {output.Value.ElementType} [{string.Join(", ", output.Value.Dimensions)}]");

            TestContext.Out.WriteLine(report.ToString());

            string written = Path.Combine(Path.GetTempPath(), "paratactus-model-probe.txt");
            File.WriteAllText(written, report.ToString());
            TestContext.Out.WriteLine($"written to {written}");
        }
    }
}
