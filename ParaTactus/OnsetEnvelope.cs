using System;
using System.Collections.Generic;
using ParaTactus;

namespace ParaTactus
{
    /// <summary>
    /// How much onset there is at a set of times, and whether there is any halfway between them.
    /// </summary>
    /// <remarks>
    /// This exists because of the one failure the model cannot report on itself. On Designant's 47-76s the tracker
    /// settles onto a clean, confident 600ms pulse where the music has a beat every 300ms - the tactus, halved. Nothing
    /// in the model's output reveals it: its activation at its own beats is strongly positive (+1.79 on average) and
    /// halfway between them strongly negative (-2.48), which is exactly what a genuine 100 BPM passage looks like. The
    /// model is not undecided, it is wrong and certain.
    ///
    /// The audio does not have that problem. Whether a beat falls between two of the tracker's beats is a fact about
    /// the sound, and it can be measured without asking the model anything. So the two methods divide the work: the
    /// model says where the beats are, because its phase is sharp and reliable, and the onset envelope says how many
    /// there are, because the model is blind to that.
    ///
    /// Deliberately only a comparison of two sets of instants rather than a tempo estimator. The analyser this replaced
    /// searched for the best lag across the whole envelope and needed candidates, a prior, a phase-locked score, a
    /// continuity term and a segmenter to choose between its own answers; measured against a beatmap it was wrong at 32
    /// of 35 points. One question with one answer does not need any of that.
    /// </remarks>
    public static class OnsetEnvelope
    {
        /// <summary>
        /// A half-wave rectified spectral flux at 50 frames per second, normalised to a mean of one.
        /// </summary>
        public static double[] FromSamples(float[] samples, out int frames)
        {
            int hop = AnalysisAudio.SampleRate / BeatThisBeatTracker.FramesPerSecond;
            const int nfft = 1024;

            frames = samples.Length >= nfft / 2 ? 1 + samples.Length / hop : 0;

            var flux = new double[frames];
            var frame = new float[nfft];
            var magnitudes = new float[Fft.BIN_COUNT];
            var previous = new float[Fft.BIN_COUNT];

            for (int t = 0; t < frames; t++)
            {
                int start = t * hop - nfft / 2;

                for (int i = 0; i < nfft; i++)
                {
                    int index = start + i;
                    frame[i] = index >= 0 && index < samples.Length ? samples[index] : 0;
                }

                Fft.AnalyseMagnitudes(frame, magnitudes);

                double sum = 0;

                for (int b = 0; b < Fft.BIN_COUNT; b++)
                {
                    double delta = magnitudes[b] - previous[b];

                    if (delta > 0)
                        sum += delta;

                    previous[b] = magnitudes[b];
                }

                flux[t] = sum;
            }

            double mean = 0;

            foreach (double value in flux)
                mean += value;

            mean = flux.Length > 0 ? mean / flux.Length : 0;

            if (mean > 0)
            {
                for (int t = 0; t < frames; t++)
                    flux[t] /= mean;
            }

            return flux;
        }

        /// <summary>
        /// The mean onset at a set of beats and at the points halfway between them.
        /// </summary>
        public static (double AtBeats, double AtMidpoints, int Count) Compare(
            IReadOnlyList<double> beats, double[] envelope, int frames)
        {
            if (beats.Count < 4 || frames <= 2 || envelope == null || envelope.Length < frames)
                return (0, 0, 0);

            double atBeats = 0;
            double atMidpoints = 0;
            int count = 0;

            for (int i = 1; i < beats.Count; i++)
            {
                atBeats += strength(envelope, frames, frameAt(beats[i]));
                atMidpoints += strength(envelope, frames, frameAt((beats[i - 1] + beats[i]) / 2));
                count++;
            }

            return count == 0 ? (0, 0, 0) : (atBeats / count, atMidpoints / count, count);
        }

        /// <summary>
        /// Whether the beats are at half the density the audio is actually playing at.
        /// </summary>
        /// <remarks>
        /// The threshold is a fraction rather than a comparison of the two outright, because the midpoint of a beat
        /// always carries some onset in real music - off-beat hats, a snare on the backbeat - so a passage at the right
        /// density does not score zero there. What separates the two cases is how close the midpoint comes to the beat:
        /// a genuine subdivision reaches half or more, whereas the middle of a beat the music does not have sits far
        /// below it.
        /// </remarks>
        public static bool LooksHalved(double atBeats, double atMidpoints, double fraction = 0.6)
        {
            return atBeats > 0 && atMidpoints >= atBeats * fraction;
        }

        private static int frameAt(double milliseconds)
        {
            return (int)Math.Round(milliseconds * BeatThisBeatTracker.FramesPerSecond / 1000);
        }

        /// <summary>
        /// The strongest envelope value within a frame of a time.
        /// </summary>
        /// <remarks>
        /// One frame either way, because the tracker's beats are quantised to 20ms while the envelope is a continuous
        /// measurement: a beat half a frame out is not a beat in the wrong place.
        /// </remarks>
        private static double strength(double[] envelope, int frames, int frame)
        {
            double best = 0;

            for (int i = Math.Max(0, frame - 1); i <= Math.Min(frames - 1, frame + 1); i++)
                best = Math.Max(best, envelope[i]);

            return best;
        }
    }
}
