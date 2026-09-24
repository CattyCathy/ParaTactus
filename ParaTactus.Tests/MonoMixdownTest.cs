using System;
using NUnit.Framework;
using ParaTactus;

namespace ParaTactus.Tests
{
    /// <summary>
    /// The resampler has to remove what is above the target Nyquist before it decimates.
    /// </summary>
    /// <remarks>
    /// Linear interpolation passes it instead: decimating a 16kHz tone from 44.1kHz to 22.05kHz with one lands the tone
    /// at 6.05kHz at nearly full amplitude, inside the mel bands the model reads. Every signal below is a whole second
    /// of a tone whose frequency completes a whole number of periods, so the measurements are not reading leakage.
    /// </remarks>
    [TestFixture]
    public class MonoMixdownTest
    {
        private const int sourceRate = 44100;
        private const int targetRate = AnalysisAudio.SampleRate;

        private static float[] tone(double frequency, int sampleRate, double seconds = 1)
        {
            int length = (int)(sampleRate * seconds);
            var samples = new float[length];

            for (int i = 0; i < length; i++)
                samples[i] = (float)Math.Sin(2 * Math.PI * frequency * i / sampleRate);

            return samples;
        }

        private static float[] convert(float[] samples, int rate)
        {
            return MonoMixdown.ToMono(new RawPcm(samples, 1, rate), targetRate);
        }

        private static double rms(float[] samples)
        {
            double sum = 0;

            foreach (float sample in samples)
                sum += (double)sample * sample;

            return samples.Length == 0 ? 0 : Math.Sqrt(sum / samples.Length);
        }

        /// <summary>Amplitude of one frequency in a signal, by Goertzel.</summary>
        private static double amplitudeAt(float[] samples, double frequency)
        {
            double omega = 2 * Math.PI * frequency / targetRate;
            double coefficient = 2 * Math.Cos(omega);
            double previous = 0;
            double previousPrevious = 0;

            foreach (float sample in samples)
            {
                double current = sample + coefficient * previous - previousPrevious;
                previousPrevious = previous;
                previous = current;
            }

            double real = previous - previousPrevious * Math.Cos(omega);
            double imaginary = previousPrevious * Math.Sin(omega);

            return samples.Length == 0 ? 0 : 2 * Math.Sqrt(real * real + imaginary * imaginary) / samples.Length;
        }

        [Test]
        public void AToneAboveTheTargetNyquistIsFilteredOutRatherThanFoldedDown()
        {
            float[] converted = convert(tone(16000, sourceRate), sourceRate);

            Assert.That(converted.Length, Is.EqualTo(targetRate), "one second in is one second out");

            // 16kHz against a 22.05kHz rate folds to |16000 - 22050|.
            double folded = amplitudeAt(converted, Math.Abs(16000 - targetRate));
            double relative = 20 * Math.Log10(rms(converted) / (1 / Math.Sqrt(2)));

            Assert.That(folded, Is.LessThan(0.01), $"the folded tone came out at amplitude {folded:0.0000}");
            Assert.That(relative, Is.LessThanOrEqualTo(-40), $"the output is only {relative:0.0}dB below the input");
        }

        [TestCase(440, 0.3)]
        [TestCase(8000, 1.0)]
        public void AToneInThePassbandKeepsItsAmplitude(double frequency, double toleranceDecibels)
        {
            float[] converted = convert(tone(frequency, sourceRate), sourceRate);
            double decibels = 20 * Math.Log10(amplitudeAt(converted, frequency));

            Assert.That(Math.Abs(decibels), Is.LessThan(toleranceDecibels), $"{frequency}Hz came through at {decibels:0.00}dB");
        }

        [TestCase(44100)]
        [TestCase(48000)]
        [TestCase(32000)]
        public void TheOutputLengthFollowsTheRateRatio(int rate)
        {
            int sourceLength = rate * 2;
            float[] converted = convert(new float[sourceLength], rate);

            Assert.That(converted.Length, Is.EqualTo((int)(sourceLength * (long)targetRate / rate)));
        }

        [Test]
        public void AShortInputIsConvertedRatherThanRejected()
        {
            Assert.DoesNotThrow(() => convert(new[] { 0.1f, -0.2f, 0.3f, -0.4f, 0.5f }, sourceRate));
        }

        [Test]
        public void ChannelsAreMixedBeforeTheRateIsChanged()
        {
            float[] left = tone(1000, sourceRate, 0.5);
            var interleaved = new float[left.Length * 2];

            for (int i = 0; i < left.Length; i++)
            {
                interleaved[2 * i] = left[i];
                interleaved[2 * i + 1] = -left[i];
            }

            float[] converted = MonoMixdown.ToMono(new RawPcm(interleaved, 2, sourceRate), targetRate);

            // Out-of-phase channels cancel, which is only true if the mixdown happened while they were still aligned.
            Assert.That(rms(converted), Is.LessThan(1e-6));
        }

        [Test]
        public void AMatchingRateIsPassedThroughUnchanged()
        {
            float[] samples = tone(1000, targetRate, 0.1);

            Assert.That(MonoMixdown.ToMono(new RawPcm(samples, 1, targetRate), targetRate), Is.EqualTo(samples));
        }
    }
}
