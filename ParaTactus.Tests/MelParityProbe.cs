using System;
using System.IO;
using NUnit.Framework;
using ParaTactus;
using ParaTactus.Decoding;

namespace ParaTactus.Tests
{
    /// <summary>
    /// Writes the frontend's own output so a Python implementation can be checked against it.
    /// </summary>
    /// <remarks>
    /// A trained detector is trained on features computed in Python and run in C#, and if the two frontends differ by
    /// a little the model is being asked to read something it was not trained on. The differences are not hypothetical:
    /// the published frontend uses magnitude rather than power, Slaney-scale mel filters that are not area normalised,
    /// a frame-length STFT normalisation, and a logarithm of a thousand times the result - and torchaudio's defaults
    /// disagree with all four. A comparison of numbers is the only way to know the Python side matches, because both
    /// sides look correct on their own.
    ///
    /// Writes the spectrogram as raw little-endian float32 with the frame count and the bin count in a header, which
    /// is what a numpy reader wants.
    ///
    /// Set <c>OSUTEST_AUDIO</c> to an audio file and <c>OSUTEST_MEL_OUT</c> for where to write.
    /// </remarks>
    [Explicit("Needs OSUTEST_AUDIO and OSUTEST_MEL_OUT.")]
    public class MelParityProbe
    {
        [OneTimeSetUp]
        public void SetUp() => AudioTestEnvironment.Initialise();

        [Test]
        public void TimeTheInferenceAtSeveralThreadCounts()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to an audio file.");

            string modelPath = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                               ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0-int8.onnx");

            float[] samples = BassAudioDecoder.DecodeMono(audio, AnalysisAudio.SampleRate);
            double seconds = samples.Length / 22050.0;

            TestContext.Out.WriteLine($"audio: {seconds:0.0}s at {AnalysisAudio.SampleRate}Hz");

            foreach (int threads in new[] { 2, 4, 8, 12, 16 })
            {
                var clock = System.Diagnostics.Stopwatch.StartNew();

                // The whole frontend and the whole model, run the way the exporter runs them, so the number is the one
                // the export actually pays.
                using (var tracker = new StreamingBeatTracker(modelPath, threads))
                {
                    tracker.Add(samples);
                    tracker.Flush();
                }

                clock.Stop();
                double elapsed = clock.Elapsed.TotalSeconds;

                TestContext.Out.WriteLine($"  {threads,3} threads: {elapsed,7:0.0}s   {seconds / elapsed,5:0.00}x realtime");
            }
        }

        [Test]
        public void ThePublicFrontendAgreesWithTheTracker()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to an audio file.");

            float[] samples = BassAudioDecoder.DecodeMono(audio, AnalysisAudio.SampleRate);

            float[,] reference = BeatThisBeatTracker.LogMelSpectrogram(samples);
            float[,] mine = LogMel.Spectrogram(samples);

            TestContext.Out.WriteLine($"reference {reference.GetLength(0)} x {reference.GetLength(1)}");
            TestContext.Out.WriteLine($"public    {mine.GetLength(0)} x {mine.GetLength(1)}");

            Assert.That(mine.GetLength(0), Is.EqualTo(reference.GetLength(0)), "frame counts differ");
            Assert.That(mine.GetLength(1), Is.EqualTo(reference.GetLength(1)), "bin counts differ");

            double worst = 0;
            double total = 0;
            int worstFrame = 0;
            int worstBin = 0;

            for (int t = 0; t < reference.GetLength(0); t++)
            {
                for (int m = 0; m < reference.GetLength(1); m++)
                {
                    double difference = Math.Abs(mine[t, m] - reference[t, m]);

                    total += difference;

                    if (difference > worst)
                    {
                        worst = difference;
                        worstFrame = t;
                        worstBin = m;
                    }
                }
            }

            int cells = reference.GetLength(0) * reference.GetLength(1);

            TestContext.Out.WriteLine($"worst difference {worst:e3} at frame {worstFrame}, bin {worstBin}");
            TestContext.Out.WriteLine($"mean difference  {total / cells:e3}");

            // The detector is trained on Python's version of this frontend and run against this one, so the two have to
            // agree to within float noise. A tolerance that a real difference could hide under is not a test.
            Assert.That(worst, Is.LessThan(1e-4), "the public frontend and the tracker's disagree");
        }

        [Test]
        public void WriteTheFrontendsOutput()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");
            string output = Environment.GetEnvironmentVariable("OSUTEST_MEL_OUT") ?? Path.Combine(Path.GetTempPath(), "mel-reference.bin");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to an audio file.");

            float[] samples = BassAudioDecoder.DecodeMono(audio, AnalysisAudio.SampleRate);
            float[,] spectrogram = BeatThisBeatTracker.LogMelSpectrogram(samples);

            int frames = spectrogram.GetLength(0);
            int bins = spectrogram.GetLength(1);

            TestContext.Out.WriteLine($"audio: {samples.Length} samples, {samples.Length / 22050.0:0.0}s");
            TestContext.Out.WriteLine($"spectrogram: {frames} frames x {bins} bins");

            using var file = File.Create(output);
            using var writer = new BinaryWriter(file);

            writer.Write(frames);
            writer.Write(bins);

            var flat = new float[frames * bins];

            for (int t = 0; t < frames; t++)
            {
                for (int m = 0; m < bins; m++)
                    flat[(t * bins) + m] = spectrogram[t, m];
            }

            var bytes = new byte[flat.Length * 4];
            Buffer.BlockCopy(flat, 0, bytes, 0, bytes.Length);
            writer.Write(bytes);

            // The raw samples too, so the Python side reads exactly the audio this was computed from rather than
            // decoding the same file again and hoping two decoders agree.
            var sampleBytes = new byte[samples.Length * 4];
            Buffer.BlockCopy(samples, 0, sampleBytes, 0, sampleBytes.Length);
            writer.Write(sampleBytes);

            writer.Write(samples.Length);

            TestContext.Out.WriteLine($"wrote {output} ({new FileInfo(output).Length / 1024 / 1024}MB)");
        }
    }
}
