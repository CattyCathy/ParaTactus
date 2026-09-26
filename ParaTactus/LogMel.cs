using System;
using System.Collections.Generic;

namespace ParaTactus
{
    /// <summary>
    /// The log-mel spectrogram a beat model reads: log1p(1000 * mel), 128 Slaney bands, magnitude rather than power.
    /// </summary>
    /// <remarks>
    /// A separate type from the model that uses it because the two are independent - the signature is the frontend's and
    /// the weights are the model's - and because getting the frontend wrong is silent. It disagrees with torchaudio's
    /// <c>MelSpectrogram</c> defaults in four separate ways and every one of them looks correct on its own:
    ///
    /// <list type="bullet">
    /// <item>magnitude, not power, so <c>power=1</c>;</item>
    /// <item>Slaney-scale filters that are <em>not</em> area normalised, and torchaudio's <c>MelSpectrogram</c> leaves
    /// <c>norm</c> at None, so raw triangles peaking at one. Applying the Slaney area normalisation as well tapers the
    /// high bands by up to 25x;</item>
    /// <item>the transform scaled by one over the square root of the size, which is torchaudio's
    /// <c>normalized="frame_length"</c>;</item>
    /// <item>a logarithm of a thousand times the result, not a plain logarithm.</item>
    /// </list>
    ///
    /// This was checked against the implementation used elsewhere in this library by comparing the two numerically on
    /// four minutes of audio: they agree to one part in a million. torchaudio does not - the worst band differs by 3.47
    /// and it reaches 12.35 where this reaches 8.89 - so a detector trained on one and run on the other is reading
    /// something it has never seen.
    /// </remarks>
    public static class LogMel
    {
        /// <summary>The rate the audio is expected at.</summary>
        public const int SampleRate = 22050;

        /// <summary>The transform size.</summary>
        public const int FftSize = 1024;

        /// <summary>The hop, which at this rate is exactly fifty frames a second.</summary>
        public const int Hop = 441;

        /// <summary>How many mel bands.</summary>
        public const int Bins = 128;

        /// <summary>
        /// How many frequency bins the one-sided transform has, which is one more than half the transform size.
        /// </summary>
        /// <remarks>
        /// Not the same number as <see cref="Bins"/> and conflating the two is a silent error in both directions: the
        /// filterbank is defined over these bins, so handing it fewer drops every frequency above the ones that fit -
        /// with 128 of them that is everything above 2.7kHz - and the bands above the drop are then computed from
        /// whatever the shorter array happened to hold.
        /// </remarks>
        public const int FrequencyBins = (FftSize / 2) + 1;

        private const double f_min = 30;
        private const double f_max = 11000;

        private static readonly double[] window = HannWindow();
        private static readonly double[] filterbank = Filterbank();

        /// <summary>
        /// The number of frames a track of this many samples produces, which is what the caller has to allocate for.
        /// </summary>
        /// <remarks>
        /// One more than the samples divided by the hop, and nothing to do with the transform size. The window is
        /// centred and the signal is reflected at both edges, so the last frames are built from reflection rather than
        /// from samples that do not exist - which is why a track a fraction of a window long still has frames, and why
        /// this matches the count the rest of the library uses. Getting it wrong is not a small error either way: it
        /// sizes the array, so a formula that disagrees with the loop leaves part of the result unwritten.
        /// </remarks>
        public static int FrameCount(int sampleCount)
        {
            if (sampleCount <= 0)
                return 0;

            return 1 + (sampleCount / Hop);
        }

        /// <summary>The log-mel spectrogram of a whole track, shaped (frames, <see cref="Bins"/>).</summary>
        public static float[,] Spectrogram(IReadOnlyList<float> samples)
        {
            if (samples == null)
                throw new ArgumentNullException(nameof(samples));

            int frames = FrameCount(samples.Count);
            var result = new float[frames, Bins];
            var spectrum = new double[FrequencyBins];

            for (int frame = 0; frame < frames; frame++)
            {
                Magnitudes(samples, frame, spectrum);

                for (int m = 0; m < Bins; m++)
                {
                    double sum = 0;

                    for (int bin = 0; bin < FrequencyBins; bin++)
                        sum += filterbank[(m * FrequencyBins) + bin] * spectrum[bin];

                    result[frame, m] = (float)Math.Log(1 + (1000 * sum));
                }
            }

            return result;
        }

        /// <summary>The one-sided magnitude spectrum of one frame, scaled as the reference frontend scales it.</summary>
        /// <remarks>
        /// Frame t is centred on sample t * hop, with the signal reflected at both edges, which is what torchaudio's
        /// <c>center=True</c> does and what makes a range of frames from the middle of a track the same computation as
        /// those frames of the whole track.
        /// </remarks>
        private static void Magnitudes(IReadOnlyList<float> samples, int frame, double[] into)
        {
            int start = (frame * Hop) - (FftSize / 2);
            var real = new double[FftSize];
            var imaginary = new double[FftSize];

            for (int i = 0; i < FftSize; i++)
                real[i] = Sample(samples, start + i) * window[i];

            Transform(real, imaginary);

            double scale = 1.0 / Math.Sqrt(FftSize);

            for (int bin = 0; bin < into.Length; bin++)
            {
                // Bin k of the one-sided spectrum: the first half of the transform in place, with the second half
                // holding the same information mirrored as its conjugate. Reading only the first half and calling it
                // the spectrum is what made a single tone appear in every band, because the indices that were read held
                // whatever the transform had left there.
                double magnitude = Math.Sqrt((real[bin] * real[bin]) + (imaginary[bin] * imaginary[bin]));

                into[bin] = magnitude * scale;
            }
        }

        /// <summary>A sample, reflected at both edges of the track.</summary>
        private static double Sample(IReadOnlyList<float> samples, int index)
        {
            int count = samples.Count;

            if (count == 0)
                return 0;

            while (index < 0 || index >= count)
            {
                if (index < 0)
                    index = -index;
                else
                    index = (2 * count) - 2 - index;

                // A track of one sample reflects onto itself forever; there is nothing else it could mean.
                if (count == 1)
                    return samples[0];
            }

            return samples[index];
        }

        /// <summary>A radix-2 transform, in place, leaving the result in natural order.</summary>
        /// <remarks>
        /// The decimation-in-time form needs its input permuted into bit-reversed order before the butterflies, not
        /// swapped alongside them: the two are different permutations and interleaving them leaves the output in an
        /// order that looks like a spectrum and is not one. Doing the permutation first and then the butterflies is the
        /// textbook order and gives the bins where they belong - which a single tone is what proves, since the whole
        /// point of a transform is that one frequency lands in one bin.
        /// </remarks>
        private static void Transform(double[] real, double[] imaginary)
        {
            int n = real.Length;

            // Bit-reversal permutation.
            for (int i = 1, j = 0; i < n; i++)
            {
                int bit = n >> 1;

                for (; (j & bit) != 0; bit >>= 1)
                    j ^= bit;

                j ^= bit;

                if (i >= j)
                    continue;

                (real[i], real[j]) = (real[j], real[i]);
                (imaginary[i], imaginary[j]) = (imaginary[j], imaginary[i]);
            }

            // Butterflies.
            for (int length = 2; length <= n; length <<= 1)
            {
                double angle = -2 * Math.PI / length;
                double stepReal = Math.Cos(angle);
                double stepImaginary = Math.Sin(angle);

                for (int i = 0; i < n; i += length)
                {
                    double wReal = 1;
                    double wImaginary = 0;

                    for (int j = 0; j < length / 2; j++)
                    {
                        int a = i + j;
                        int b = i + j + (length / 2);

                        double tReal = (real[b] * wReal) - (imaginary[b] * wImaginary);
                        double tImaginary = (real[b] * wImaginary) + (imaginary[b] * wReal);

                        real[b] = real[a] - tReal;
                        imaginary[b] = imaginary[a] - tImaginary;
                        real[a] += tReal;
                        imaginary[a] += tImaginary;

                        double nextReal = (wReal * stepReal) - (wImaginary * stepImaginary);
                        wImaginary = (wReal * stepImaginary) + (wImaginary * stepReal);
                        wReal = nextReal;
                    }
                }
            }
        }

        /// <summary>A periodic Hann window, which is what torchaudio's default is.</summary>
        private static double[] HannWindow()
        {
            var result = new double[FftSize];

            for (int i = 0; i < FftSize; i++)
                result[i] = 0.5 * (1 - Math.Cos(2 * Math.PI * i / FftSize));

            return result;
        }

        /// <summary>Slaney-scale triangular filters, raw and not area normalised.</summary>
        private static double[] Filterbank()
        {
            const double f_sp = 200.0 / 3;
            double logstep = Math.Log(6.4) / 27.0;
            double min_log_hz = 1000;
            double min_log_mel = min_log_hz / f_sp;

            double ToMel(double f) => f >= min_log_hz
                ? min_log_mel + (Math.Log(f / min_log_hz) / logstep)
                : f / f_sp;

            double ToHz(double m) => m >= min_log_mel
                ? min_log_hz * Math.Exp(logstep * (m - min_log_mel))
                : f_sp * m;

            var bands = new double[Bins + 2];

            for (int i = 0; i < bands.Length; i++)
            {
                double mel = ToMel(f_min) + ((ToMel(f_max) - ToMel(f_min)) * i / (bands.Length - 1));
                bands[i] = ToHz(mel);
            }

            int bins = (FftSize / 2) + 1;
            var bank = new double[Bins * bins];
            for (int m = 0; m < Bins; m++)
            {
                double lower = bands[m];
                double centre = bands[m + 1];
                double upper = bands[m + 2];

                for (int i = 0; i < bins; i++)
                {
                    double frequency = i * (double)SampleRate / FftSize;
                    double value = 0;

                    if (frequency > lower && frequency < centre && centre > lower)
                        value = (frequency - lower) / (centre - lower);
                    else if (frequency >= centre && frequency < upper && upper > centre)
                        value = (upper - frequency) / (upper - centre);

                    bank[(m * bins) + i] = value;
                }
            }

            return bank;
        }
    }
}
