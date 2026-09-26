using System;
using System.Numerics;
using NUnit.Framework;
using ParaTactus;

namespace ParaTactus.Tests
{
    /// <summary>
    /// Prints the frontend's intermediate values for a tone, so a mismatch can be located rather than guessed at.
    /// </summary>
    [TestFixture]
    public class LogMelSpectrumTest
    {
        [Test]
        public void ShowWhereAToneLands()
        {
            const int rate = 22050;
            const double frequency = 1000;
            int samples = rate * 2;
            var signal = new float[samples];

            for (int i = 0; i < samples; i++)
                signal[i] = (float)(0.5 * Math.Sin(2 * Math.PI * frequency * i / rate));

            // A reference transform of the same window, from the framework, so the comparison is against something
            // known rather than against my own arithmetic twice.
            int start = (100 * LogMel.Hop) - (LogMel.FftSize / 2);
            var window = new double[LogMel.FftSize];
            var real = new double[LogMel.FftSize];
            var imaginary = new double[LogMel.FftSize];

            for (int i = 0; i < LogMel.FftSize; i++)
            {
                window[i] = 0.5 * (1 - Math.Cos(2 * Math.PI * i / LogMel.FftSize));
                real[i] = Reflect(signal, start + i) * window[i];
            }

            Complex[] spectrum = DirectTransform(real);

            TestContext.Out.WriteLine($"1000Hz belongs in bin {(int)Math.Round(frequency * LogMel.FftSize / rate)}");

            // The loudest bins by the framework's transform.
            var order = new int[spectrum.Length];

            for (int i = 0; i < order.Length; i++)
                order[i] = i;

            Array.Sort(order, (a, b) => spectrum[b].Magnitude.CompareTo(spectrum[a].Magnitude));

            TestContext.Out.WriteLine("the framework's transform, loudest bins:");

            for (int i = 0; i < 6; i++)
            {
                int bin = order[i];
                TestContext.Out.WriteLine($"  bin {bin,4} ({bin * (double)rate / LogMel.FftSize,8:0}Hz): {spectrum[bin].Magnitude:0.0000}");
            }

            // And the filterbank's own centre frequencies, so the band a tone belongs in can be read off.
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("the mel bands nearest 1000Hz:");

            float[,] mine = LogMel.Spectrogram(signal);
            int peak = 0;

            for (int m = 0; m < mine.GetLength(1); m++)
            {
                if (mine[100, m] > mine[100, peak])
                    peak = m;
            }

            TestContext.Out.WriteLine($"  the frontend says the loudest band is {peak}");
            TestContext.Out.WriteLine($"  band {peak}: {mine[100, peak]:0.000}");
            TestContext.Out.WriteLine($"  band {peak - 1}: {mine[100, peak - 1]:0.000}");
            TestContext.Out.WriteLine($"  band {peak + 1}: {mine[100, peak + 1]:0.000}");
        }

        /// <summary>
        /// Where each band's energy sits in frequency, read back out of the frontend itself.
        /// </summary>
        /// <remarks>
        /// A tone at a known frequency is the only way to ask a filterbank what it is tuned to: the band whose output
        /// rises is the band that contains the tone, and doing that at a few frequencies gives the mapping the
        /// implementation actually has rather than the one it was meant to have. Comparing against the other
        /// implementation's band centres then says which of the two is wrong instead of only that they differ.
        /// </remarks>
        [Test]
        public void ShowBandCentreFrequencies()
        {
            const int rate = 22050;

            TestContext.Out.WriteLine("a tone at each of these frequencies, and the band that answers loudest:");

            foreach (int frequency in new[] { 100, 250, 500, 1000, 2000, 4000, 8000 })
            {
                int samples = rate;
                var signal = new float[samples];

                for (int i = 0; i < samples; i++)
                    signal[i] = (float)(0.5 * Math.Sin(2 * Math.PI * frequency * i / rate));

                float[,] spectrogram = LogMel.Spectrogram(signal);

                int peak = 0;

                for (int m = 0; m < spectrogram.GetLength(1); m++)
                {
                    if (spectrogram[50, m] > spectrogram[50, peak])
                        peak = m;
                }

                TestContext.Out.WriteLine($"  {frequency,6}Hz -> band {peak,3}   ({spectrogram[50, peak]:0.000}), "
                                          + $"neighbours {spectrogram[50, Math.Max(0, peak - 1)]:0.000} "
                                          + $"{spectrogram[50, Math.Min(127, peak + 1)]:0.000}");
            }

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("the reference implementation puts 1000Hz in band 37, 4000Hz near band 72 and "
                                      + "8000Hz near band 100, so a mapping that disagrees is a wrong filterbank.");
        }

        /// <summary>A sample, reflected at both edges, which is what a centred window needs at the ends of a track.</summary>
        private static double Reflect(float[] signal, int index)
        {
            int count = signal.Length;

            while (index < 0 || index >= count)
            {
                if (index < 0)
                    index = -index;
                else
                    index = (2 * count) - 2 - index;
            }

            return signal[index];
        }

        /// <summary>
        /// A transform by the definition, so the fast one has something independent to be checked against. A thousand
        /// points is a million operations, which is nothing here, and the definition cannot be subtly wrong the way a
        /// butterfly can.
        /// </summary>
        private static Complex[] DirectTransform(double[] input)
        {
            int n = input.Length;
            var result = new Complex[n];

            for (int k = 0; k < n; k++)
            {
                double totalReal = 0;
                double totalImaginary = 0;

                for (int t = 0; t < n; t++)
                {
                    double angle = -2 * Math.PI * k * t / n;

                    totalReal += input[t] * Math.Cos(angle);
                    totalImaginary += input[t] * Math.Sin(angle);
                }

                result[k] = new Complex(totalReal, totalImaginary);
            }

            return result;
        }
    }
}
