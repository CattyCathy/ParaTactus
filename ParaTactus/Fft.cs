using System;
using ParaTactus;

namespace ParaTactus
{
    /// <summary>
    /// Minimal in-place radix-2 Cooley-Tukey FFT.
    /// </summary>
    /// <remarks>
    /// Written by hand rather than pulled in as a dependency because the beat tracker only ever
    /// needs forward transforms of a single fixed size, applied thousands of times.
    ///
    /// An instance holds its tables and its scratch buffers, so it must not be used from two threads at once.
    /// <see cref="AnalyseMagnitudes"/> is the way to call this from anywhere: it keeps one instance per thread, which
    /// makes it safe from the draw thread and the analysis thread at the same time and keeps it allocation-free.
    ///
    /// Note on why this exists at all, kept as background rather than as a dependency: <c>osu.Framework</c> exposes
    /// <c>ChannelAmplitudes.FrequencyAmplitudes</c>, but that is a 256-bin magnitude spectrum spanning 0-20kHz
    /// (~78Hz per bin) that is only refreshed a few dozen times per second with no cross-frame consistency guarantee.
    /// Onset detection needs per-frame spectral difference at a stable frame rate, so the analysis decodes PCM and runs
    /// its own transform instead.
    /// </remarks>
    public sealed class Fft
    {
        /// <summary>Transform size used for spectrum display and analysis.</summary>
        public const int SIZE = 1024;

        /// <summary>Number of magnitude bins produced by <see cref="SIZE"/>.</summary>
        public const int BIN_COUNT = SIZE / 2;

        // One transform per thread, because the scratch buffers inside it are instance state. A single shared instance
        // was wrong, and wrong in the case that matters: the draw thread computes a magnitude spectrum for the
        // visualiser while the analysis thread runs this same entry point for the model's frontend, and the two
        // overwrote each other's windowed frame.
        [ThreadStatic]
        private static Fft perThread;

        /// <summary>
        /// A magnitude spectrum for callers that only need one, such as a visualiser.
        /// </summary>
        /// <remarks>
        /// Safe to call from any thread: each thread gets its own transform, so the tables are built once per thread
        /// and no caller can disturb another's scratch. Nothing is allocated after the first call on a thread.
        /// </remarks>
        public static void AnalyseMagnitudes(ReadOnlySpan<float> samples, Span<float> magnitudes)
        {
            var fft = perThread ??= new Fft(SIZE);
            var windowed = fft.windowedScratch;

            int count = Math.Min(samples.Length, fft.size);

            for (int i = 0; i < count; i++)
                windowed[i] = samples[i] * fft.hannWindow[i];

            for (int i = count; i < fft.size; i++)
                windowed[i] = 0;

            var spectrum = fft.spectrumScratch;
            fft.ComputeMagnitudes(windowed, spectrum);

            for (int i = 0; i < Math.Min(spectrum.Length, magnitudes.Length); i++)
                magnitudes[i] = (float)spectrum[i];
        }

        private readonly double[] windowedScratch;
        private readonly double[] realScratch;
        private readonly double[] imagScratch;
        private readonly double[] spectrumScratch;
        private readonly double[] hannWindow;
        private readonly int size;
        private readonly int levels;
        private readonly double[] cosTable;
        private readonly double[] sinTable;
        private readonly int[] reverseTable;

        public Fft(int size)
        {
            if (size < 2 || (size & (size - 1)) != 0)
                throw new ArgumentException("FFT size must be a power of two greater than 1.", nameof(size));

            this.size = size;
            levels = (int)Math.Round(Math.Log(size, 2));

            cosTable = new double[size / 2];
            sinTable = new double[size / 2];

            for (int i = 0; i < size / 2; i++)
            {
                cosTable[i] = Math.Cos(2 * Math.PI * i / size);
                sinTable[i] = Math.Sin(2 * Math.PI * i / size);
            }

            reverseTable = new int[size];

            for (int i = 0; i < size; i++)
                reverseTable[i] = reverseBits(i, levels);

            // Scratch for every entry point, because the callers that matter run this once per frame - 50 times a
            // second for the analysis, once a draw for a visualiser - and per-call arrays are the whole cost of a
            // frame this size. Instance state, which is why the static entry point hands each thread its own instance.
            windowedScratch = new double[size];
            realScratch = new double[size];
            imagScratch = new double[size];
            spectrumScratch = new double[size / 2];
            hannWindow = new double[size];

            for (int i = 0; i < size; i++)
                hannWindow[i] = 0.5 * (1 - Math.Cos(2 * Math.PI * i / size));
        }

        /// <summary>
        /// The transform size this instance was constructed for.
        /// </summary>
        public int Size => size;

        /// <summary>
        /// The number of magnitude bins produced by <see cref="ComputeMagnitudes"/>.
        /// </summary>
        public int BinCount => size / 2;

        /// <summary>
        /// Transforms <paramref name="windowedInput"/> (length <see cref="Size"/>) and writes its
        /// magnitude spectrum into <paramref name="magnitudes"/> (length <see cref="BinCount"/>).
        /// </summary>
        public void ComputeMagnitudes(ReadOnlySpan<double> windowedInput, Span<double> magnitudes)
        {
            if (windowedInput.Length != size)
                throw new ArgumentException($"Input must have length {size}.", nameof(windowedInput));

            if (magnitudes.Length < BinCount)
                throw new ArgumentException($"Output must have length at least {BinCount}.", nameof(magnitudes));

            var real = realScratch;
            var imag = imagScratch;

            // The permutation below writes every element of real, so it carries nothing over, but imag is only ever
            // read by the butterflies and has to start at zero. Reused scratch is why this line exists at all.
            Array.Clear(imag);

            for (int i = 0; i < size; i++)
                real[reverseTable[i]] = windowedInput[i];

            for (int levelSize = 2; levelSize <= size; levelSize <<= 1)
            {
                int halfSize = levelSize >> 1;
                int tableStep = size / levelSize;

                for (int i = 0; i < size; i += levelSize)
                {
                    for (int j = i, k = 0; j < i + halfSize; j++, k += tableStep)
                    {
                        int l = j + halfSize;

                        double cos = cosTable[k];
                        double sin = sinTable[k];

                        double tre = real[l] * cos + imag[l] * sin;
                        double tim = -real[l] * sin + imag[l] * cos;

                        real[l] = real[j] - tre;
                        imag[l] = imag[j] - tim;
                        real[j] += tre;
                        imag[j] += tim;
                    }
                }
            }

            for (int i = 0; i < BinCount; i++)
                magnitudes[i] = Math.Sqrt(real[i] * real[i] + imag[i] * imag[i]);
        }

        private static int reverseBits(int value, int bitCount)
        {
            int result = 0;

            for (int i = 0; i < bitCount; i++)
            {
                result = (result << 1) | (value & 1);
                value >>= 1;
            }

            return result;
        }
    }
}
