using System;

namespace ParaTactus
{
    /// <summary>
    /// Channel mixdown and sample-rate conversion, which every decoder needs and none of them owns.
    /// </summary>
    /// <remarks>
    /// Decoding is the decoder's business; getting from whatever it produced to mono at the analysis rate is not, and
    /// doing it here means two decoders cannot disagree about it. The order matters and is the reason this is shared
    /// rather than copied: mix down first and then resample. Averaging channels after decimation would be wrong, and
    /// this way the resampler only ever sees complete frames.
    /// </remarks>
    public static class MonoMixdown
    {
        /// <summary>
        /// Where the anti-aliasing filter stops passing, as a fraction of the target rate.
        /// </summary>
        /// <remarks>
        /// As close to the target Nyquist as a filter can practicably be, because the model's mel filterbank reaches
        /// 11kHz on a 22.05kHz rate: almost the whole target band is signal to it, and the only content worth removing
        /// is what folds onto that band from above. 0.45 was tried first and is measurably too low - on the reference
        /// track it cut into the top of the model's band and moved the tracker onto half tempo through the 140-180s
        /// section, where it had held the 200 BPM the beatmap declares before the filter existed. With nothing above
        /// 11kHz to spare, the trade has to favour the band the model actually uses.
        /// </remarks>
        private const double cutoff_fraction = 0.495;

        /// <summary>Length of the anti-aliasing filter, in taps at the source rate.</summary>
        /// <remarks>
        /// Long enough that the transition band is narrow enough to fit between the top of the model's band and the
        /// target Nyquist, which is a gap of a few hundred hertz. 64 taps spread the transition across the top mel
        /// band; 256 narrows it to roughly a quarter of that, for four times the arithmetic - about a second of work
        /// per track against the roughly 45 seconds the analysis takes.
        /// </remarks>
        private const int filter_taps = 256;

        /// <summary>Kaiser shape parameter: a wider main lobe and a lower stopband than the usual 5 or 6.</summary>
        private const double kaiser_beta = 8.0;

        /// <summary>Phases of the filter that are precomputed; positions between them are interpolated.</summary>
        private const int phase_count = 128;

        /// <summary>
        /// The raw decode mixed to one channel and converted to the requested rate.
        /// </summary>
        public static float[] ToMono(RawPcm raw, int sampleRate)
        {
            int channels = Math.Max(1, raw.Channels);
            int frames = raw.Samples.Length / channels;

            if (frames == 0)
                return Array.Empty<float>();

            var mono = new float[frames];

            if (channels == 1)
                Array.Copy(raw.Samples, mono, frames);
            else
            {
                for (int frame = 0; frame < frames; frame++)
                {
                    float sum = 0;

                    for (int channel = 0; channel < channels; channel++)
                        sum += raw.Samples[frame * channels + channel];

                    mono[frame] = sum / channels;
                }
            }

            return raw.SampleRate == sampleRate ? mono : resample(mono, raw.SampleRate, sampleRate);
        }

        /// <summary>
        /// Converts a mono signal to another rate, filtering before it decimates.
        /// </summary>
        /// <remarks>
        /// This used to be plain linear interpolation, justified by a comment saying analysis only needs the onset
        /// envelope and so does not need a filter. The second half of that is false: the model's input is a log-mel
        /// spectrogram reaching 11kHz, so whatever is above the target Nyquist and is not removed folds back down into
        /// the bands the model reads. A 44.1kHz source reaches here unfiltered - the BASS decoder deliberately decodes
        /// at the native rate - carrying content the model was never trained on.
        ///
        /// So: a windowed-sinc low-pass immediately below the target Nyquist, evaluated at the fractional position of
        /// each output sample by interpolating between two precomputed phases. Downsampling only; a source at or below
        /// the target rate cannot alias, and is interpolated linearly, which is cheaper and adequate.
        /// </remarks>
        private static float[] resample(float[] samples, int sourceRate, int targetRate)
        {
            int targetLength = (int)(samples.Length * (long)targetRate / sourceRate);
            var result = new float[targetLength];

            if (targetLength == 0 || samples.Length < 2)
                return result;

            double ratio = sourceRate / (double)targetRate;
            double cutoff = cutoff_fraction * targetRate / sourceRate;

            if (sourceRate <= targetRate || cutoff >= 0.5)
            {
                for (int i = 0; i < targetLength; i++)
                    result[i] = interpolate(samples, i * ratio);

                return result;
            }

            double[][] phases = buildPhases(cutoff);
            int half = filter_taps / 2;

            for (int i = 0; i < targetLength; i++)
            {
                double position = i * ratio;
                int centre = (int)position;
                double scaled = (position - centre) * phase_count;
                int phase = (int)scaled;
                double blend = scaled - phase;

                double[] lower = phases[phase];
                double[] upper = phases[phase + 1];

                double sum = 0;

                for (int tap = 0; tap < filter_taps; tap++)
                {
                    // The ends are clamped rather than treated as silence, so that a track does not fade in over the
                    // first millisecond of the filter's length.
                    int index = centre - half + 1 + tap;

                    if ((uint)index >= (uint)samples.Length)
                        index = index < 0 ? 0 : samples.Length - 1;

                    sum += (lower[tap] + (upper[tap] - lower[tap]) * blend) * samples[index];
                }

                result[i] = (float)sum;
            }

            return result;
        }

        /// <summary>Linear interpolation, for the case where nothing can fold down.</summary>
        private static float interpolate(float[] samples, double position)
        {
            int index = (int)position;

            if (index >= samples.Length - 1)
                return samples[^1];

            double fraction = position - index;

            return (float)(samples[index] + (samples[index + 1] - samples[index]) * fraction);
        }

        /// <summary>
        /// The filter's impulse response at each precomputed phase, normalised to unity gain.
        /// </summary>
        /// <remarks>
        /// Normalised per phase rather than once for the design: interpolating two phases that each sum to one cannot
        /// introduce a ripple in the passband, which a single overall normalisation would allow.
        /// </remarks>
        private static double[][] buildPhases(double cutoff)
        {
            var phases = new double[phase_count + 1][];
            int half = filter_taps / 2;
            double normalisation = besselI0(kaiser_beta);

            for (int phase = 0; phase <= phase_count; phase++)
            {
                double fraction = phase / (double)phase_count;
                var taps = new double[filter_taps];
                double total = 0;

                for (int tap = 0; tap < filter_taps; tap++)
                {
                    double offset = tap - (half - 1) - fraction;
                    double weight = 2 * cutoff * sinc(2 * cutoff * offset) * kaiser(offset / half, normalisation);

                    taps[tap] = weight;
                    total += weight;
                }

                for (int tap = 0; tap < filter_taps; tap++)
                    taps[tap] /= total;

                phases[phase] = taps;
            }

            return phases;
        }

        private static double sinc(double value)
        {
            if (Math.Abs(value) < 1e-12)
                return 1;

            double angle = Math.PI * value;

            return Math.Sin(angle) / angle;
        }

        private static double kaiser(double position, double normalisation)
        {
            double magnitude = Math.Abs(position);

            return magnitude >= 1 ? 0 : besselI0(kaiser_beta * Math.Sqrt(1 - magnitude * magnitude)) / normalisation;
        }

        /// <summary>Modified Bessel function of the first kind, order zero, by its series.</summary>
        private static double besselI0(double value)
        {
            double half = value / 2;
            double sum = 1;
            double term = 1;

            for (int k = 1; k < 32; k++)
            {
                term *= (half / k) * (half / k);
                sum += term;

                if (term < 1e-15 * sum)
                    break;
            }

            return sum;
        }
    }
}
