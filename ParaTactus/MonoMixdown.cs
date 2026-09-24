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
        /// Linear-interpolating decimator. Analysis only needs the onset envelope, which is far below the
        /// Nyquist limit of the target rate, so the aliasing of a linear resampler is not worth a filter.
        /// </summary>
        private static float[] resample(float[] samples, int sourceRate, int targetRate)
        {
            int targetLength = (int)(samples.Length * (long)targetRate / sourceRate);
            var result = new float[targetLength];
            double ratio = sourceRate / (double)targetRate;

            for (int i = 0; i < targetLength; i++)
            {
                double position = i * ratio;
                int index = (int)position;

                if (index >= samples.Length - 1)
                {
                    result[i] = samples[^1];
                    continue;
                }

                double fraction = position - index;
                result[i] = (float)(samples[index] + (samples[index + 1] - samples[index]) * fraction);
            }

            return result;
        }
    }
}
