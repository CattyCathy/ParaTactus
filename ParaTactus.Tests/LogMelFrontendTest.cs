using System;
using System.IO;
using NUnit.Framework;
using ParaTactus;

namespace ParaTactus.Tests
{
    /// <summary>
    /// The log-mel frontend has to be the one the model was trained with, to the last detail.
    /// </summary>
    /// <remarks>
    /// The fixture is torchaudio's own output for a fixed signal, produced by
    /// <c>tools/ref/make_logmel_fixture.py</c>. It is committed rather than generated because torchaudio is not a
    /// dependency of this project and should not become one, and because a fixture that is regenerated at test time
    /// would agree with whatever the generator does rather than with the published frontend.
    ///
    /// This exists because the frontend was wrong twice in ways nothing caught. The first was a Hann window applied on
    /// top of a windowing FFT. The second was worse: the mel filters were scaled to unit area, which torchaudio only
    /// does for <c>norm="slaney"</c> and not for the <c>mel_scale="slaney"</c> the reference actually asks for, and the
    /// STFT was missing the <c>1/sqrt(n_fft)</c> that <c>normalized="frame_length"</c> applies. The second one tilted
    /// the spectrogram by up to 25x across the bands. Both left the model working well enough on steady music to look
    /// fine, and only showed up as unusable beats on dense music, so neither was visible without a reference to compare
    /// against. This is that reference.
    /// </remarks>
    [TestFixture]
    public class LogMelFrontendTest
    {
        /// <summary>
        /// How far a band may be from torchaudio's value.
        /// </summary>
        /// <remarks>
        /// This is not a tight bound on float32 and does not try to be. The reference computes its STFT in float32
        /// while this frontend accumulates in double and narrows once, so on the quietest high bands the leakage from
        /// the strong low ones lands about 3e-4 apart. Both mistakes this test exists to catch are three orders of
        /// magnitude larger than that: a missing 1/sqrt(n_fft) shifts every band by log(32) = 3.47, and area-normalised
        /// filters shift the top bands by more than 2.
        /// </remarks>
        private const double tolerance = 1e-3;

        private const int binCount = 128;

        private static string fixturePath()
        {
            return Path.Combine(AppContext.BaseDirectory, "LogMelFixture.bin");
        }

        private static (int Frames, int Bins, int SampleCount, float[] LogMel, float[] Signal) load()
        {
            string path = fixturePath();

            Assert.That(File.Exists(path), Is.True, $"frontend fixture not found at {path}");

            byte[] bytes = File.ReadAllBytes(path);
            int frames = BitConverter.ToInt32(bytes, 0);
            int bins = BitConverter.ToInt32(bytes, 4);
            int samples = BitConverter.ToInt32(bytes, 8);

            var logMel = new float[frames * bins];
            Buffer.BlockCopy(bytes, 12, logMel, 0, frames * bins * sizeof(float));

            var signal = new float[samples];
            Buffer.BlockCopy(bytes, 12 + frames * bins * sizeof(float), signal, 0, samples * sizeof(float));

            return (frames, bins, samples, logMel, signal);
        }

        [Test]
        public void TestFrontendMatchesThePublishedOne()
        {
            var (frames, bins, samples, expected, signal) = load();

            Assert.That(bins, Is.EqualTo(binCount));
            Assert.That(samples, Is.EqualTo(frames * 441 + 512),
                "the fixture's signal must be long enough that the last wanted frame does not reflect off the end");

            var spectrogram = BeatThisBeatTracker.LogMelSpectrogram(signal, 0, frames);

            double worst = 0;
            int worstFrame = 0;
            int worstBin = 0;

            for (int t = 0; t < frames; t++)
            {
                for (int m = 0; m < bins; m++)
                {
                    double difference = Math.Abs(spectrogram[t, m] - expected[t * bins + m]);

                    if (difference > worst)
                    {
                        worst = difference;
                        worstFrame = t;
                        worstBin = m;
                    }
                }
            }

            // The bands are reported separately because the two known bugs have different signatures: a missing
            // 1/sqrt(n_fft) is a constant offset in log space and shows up everywhere, while area-normalising the
            // filters tilts the spectrogram and is worst at the top.
            TestContext.Out.WriteLine($"worst difference {worst:0.######} at frame {worstFrame}, band {worstBin}");
            TestContext.Out.WriteLine($"  ours {spectrogram[worstFrame, worstBin]:0.######}, "
                                      + $"torchaudio {expected[worstFrame * bins + worstBin]:0.######}");
            TestContext.Out.WriteLine($"  band 0   ours {spectrogram[0, 0]:0.####}  torchaudio {expected[0]:0.####}");
            TestContext.Out.WriteLine($"  band 127 ours {spectrogram[0, 127]:0.####}  torchaudio {expected[127]:0.####}");

            Assert.That(worst, Is.LessThan(tolerance),
                $"the log-mel frontend differs from torchaudio's by {worst:0.######} at frame {worstFrame}, band {worstBin}");
        }
    }
}
