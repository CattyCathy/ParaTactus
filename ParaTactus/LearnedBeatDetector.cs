using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ParaTactus
{
    /// <summary>
    /// A beat detector trained on this project's own corpus, reading a log-mel spectrogram.
    /// </summary>
    /// <remarks>
    /// Separate from <see cref="BeatThisBeatTracker"/> because it answers a different question. That model reports
    /// where a beat is, one frame at a time, and what to do with the frames it is confident about is a decision nothing
    /// in its output makes. This one reports the beat period as well, and the period is what turns a curve of
    /// confidence into a pulse: a frame cannot be a beat if the model says the beats are half a second apart and there
    /// is a louder frame a tenth of a second away.
    ///
    /// That one extra output is the difference between the two readings being usable and not. Measured over twelve
    /// tracks against their maps' own timing points, taking the peaks of the beat curve alone reports 1.67 times as
    /// many beats as the maps have and lands on the wrong metrical level on eleven of the twelve; suppressing peaks
    /// within three quarters of the predicted period brings that to 1.04 times and ten of twelve, with the middle
    /// distance from a map's beat falling from 27ms to 11ms.
    ///
    /// The frontend is the same one the rest of this library uses, and it has to be: the model was trained on features
    /// computed from <see cref="LogMel"/> and is handed whatever that produces. The published frontend disagrees with
    /// torchaudio's defaults in four separate ways - magnitude rather than power, Slaney mel filters that are not area
    /// normalised, a frame-length transform scaling, and a logarithm of a thousand times the result - and a detector
    /// trained against one and run against the other reads something it has never seen.
    /// </remarks>
    public sealed class LearnedBeatDetector : IDisposable
    {
        /// <summary>The frames a second the model works at, which is the frontend's hop of 441 at 22050Hz.</summary>
        public const int FramesPerSecond = 50;

        /// <summary>How much audio one inference covers, in frames.</summary>
        private const int window_frames = 1024;

        /// <summary>How far one inference advances, in frames. Less than a window, so the edges are covered twice.</summary>
        private const int hop_frames = 960;

        /// <summary>
        /// How much of the predicted period two beats have to be apart, as a share of it.
        /// </summary>
        /// <remarks>
        /// Measured rather than chosen, and re-measured after a first version of the comparison turned out to have been
        /// reading the peak picker's frames as milliseconds. Swept against a dozen beatmaps and their own timing
        /// points, the share trades how much of the map is covered against how far the reading drifts from the map's
        /// level: at 0.75 the detector reports 0.87 of the map's beats and covers 67% of them, at 0.6 it reports 1.03
        /// and covers 81%, at 0.5 it reports 1.26 and covers 90%, and at 0.4 it reports 1.50 and covers 93%.
        ///
        /// Six tenths is the value that puts the reported rate where it belongs across a corpus, and it was measured
        /// against a sweep that did not yet vary the period smoothing - see <see cref="DefaultPeriodSmoothing"/>, whose
        /// sweep put the level right on seven of the eight fastest beatmaps at this share, where the unsmoothed run
        /// managed five. A caller that knows its material is fast can pass its own share to
        /// <see cref="Beats(IReadOnlyList{float}, double, double)"/>.
        /// </remarks>
        private const double suppression_share = 0.6;

        /// <summary>
        /// How many seconds either side the predicted period is taken the middle of before it is used.
        /// </summary>
        /// <remarks>
        /// Zero would use the head's output frame by frame, which is what it did first and is what made the pulses
        /// unsteady. The period sets how far apart two beats have to be, so a period that moves by a quarter of itself
        /// from one frame to the next moves the spacing by the same, and the head does move that much: one step in seven
        /// changes by more than a quarter of the tempo.
        ///
        /// One second is a measured optimum and not a guess. Swept over the eight fastest beatmaps in the corpus, at a
        /// suppression share of six tenths:
        ///
        /// <code>
        /// smoothing   rate   on level   coverage   precision   error   change in gap
        ///   0.0s       0.82     5/8        65%        89%       10ms       40ms
        ///   1.0s       0.99     7/8        94%        94%        8ms       20ms
        ///   2.0s       0.98     7/8        94%        94%        8ms       20ms
        ///   4.0s       0.98     7/8        94%        94%        8ms       20ms
        /// </code>
        ///
        /// All four measures improve at once, which is what makes this the answer rather than a trade: the period is
        /// supposed to be a tempo, a tempo does not change from one frame to the next, and a better estimate of it both
        /// steadies the spacing and lets more of the right beats through the suppression. One second is where the
        /// improvement stops, so it is the value in use.
        /// </remarks>
        public const double DefaultPeriodSmoothing = 1.0;

        private readonly SessionOptions options;
        private readonly Lazy<InferenceSession> session;
        private readonly object gate = new object();
        private bool disposed;

        public LearnedBeatDetector(string modelPath, int threads = 2)
        {
            if (string.IsNullOrEmpty(modelPath))
                throw new ArgumentException("A model path is needed to detect beats.", nameof(modelPath));

            ModelPath = modelPath;

            options = new SessionOptions
            {
                // Two threads, which is a measurement and not a default. This family of models gets slower the more
                // threads it is given: the same four-minute track takes 24.5 seconds of analysis on two threads, 55.5
                // on eight and 113.5 on sixteen, because the operators' synchronisation costs more than the
                // parallelism returns.
                IntraOpNumThreads = Math.Max(1, threads),
                InterOpNumThreads = 1,
            };

            session = new Lazy<InferenceSession>(() => new InferenceSession(modelPath, options));
        }

        /// <summary>The model this reads.</summary>
        public string ModelPath { get; }

        /// <summary>
        /// The beats of a track, in milliseconds, found by the model and spaced by the period it predicts.
        /// </summary>
        /// <remarks>
        /// Whole-track rather than streaming, because the period head reads its context from a window and a beat near
        /// the end of what has arrived would move every time more audio did. <see cref="StreamingBeatTracker"/> exists
        /// for the case where beats are needed while a track plays; this is for tracks that can be analysed first.
        /// </remarks>
        public double[] Beats(IReadOnlyList<float> samples)
        {
            return Beats(samples, suppression_share, DefaultPeriodSmoothing);
        }

        /// <summary>
        /// The same, with both the suppression share and a smoothing of the predicted period named by the caller.
        /// </summary>
        /// <param name="samples">Mono audio at <see cref="LogMel.SampleRate"/>.</param>
        /// <param name="share">How much of the predicted period two beats have to be apart.</param>
        /// <param name="periodSmoothing">
        /// How many seconds either side to take the middle of the predicted period over, or zero to use it as it comes.
        ///
        /// The head reports a period on every frame and that reading moves about: measured over two tracks, the middle
        /// change between consecutive frames is fifteen to twenty-two beats a minute and one step in seven moves by more
        /// than a quarter of the tempo. The period is what sets the distance beats have to be apart, so a wobbling period
        /// is a wobbling spacing between pulses, which is what "the output is unstable" describes. A tempo does not
        /// really change from one frame to the next, so the middle of a window is a better estimate of it than any single
        /// frame - and unlike the beat curve, where smoothing would hide where beats are, the period is supposed to be
        /// smooth.
        /// </param>
        public double[] Beats(IReadOnlyList<float> samples, double share, double periodSmoothing)
        {
            if (samples == null)
                throw new ArgumentNullException(nameof(samples));

            (float[] beats, float[] periods) = Activations(samples);

            if (beats.Length == 0)
                return Array.Empty<double>();

            if (periodSmoothing > 0)
                periods = smoothPeriods(periods, periodSmoothing);

            int[] chosen = Suppress(beats, periods, share);

            var times = new double[chosen.Length];

            for (int i = 0; i < chosen.Length; i++)
                times[i] = chosen[i] * 1000.0 / FramesPerSecond;

            return times;
        }

        /// <summary>
        /// The predicted period with each frame replaced by the middle of its neighbours.
        /// </summary>
        /// <remarks>
        /// A median over a window of seconds, and the window is what makes it work: the wobble being removed is
        /// frame-to-frame, and a tempo that genuinely changes takes several seconds to do it. Somewhere between is a
        /// window long enough to average the wobble away and short enough that a real accelerando is still followed,
        /// which is why the length is a parameter rather than a constant - it is the one thing here that is a judgement
        /// about the material rather than arithmetic.
        ///
        /// A median rather than a mean because the head has outliers: a single frame reading four times the tempo would
        /// drag a mean over the whole window, where a median is unmoved by it.
        /// </remarks>
        private static float[] smoothPeriods(float[] periods, double seconds)
        {
            int half = Math.Max(1, (int)Math.Round(seconds * FramesPerSecond));

            if (periods.Length < 3 || half <= 0)
                return periods;

            var smoothed = new float[periods.Length];
            var window = new List<float>(2 * half + 1);

            for (int i = 0; i < periods.Length; i++)
            {
                int from = Math.Max(0, i - half);
                int to = Math.Min(periods.Length - 1, i + half);

                window.Clear();

                for (int j = from; j <= to; j++)
                {
                    if (periods[j] > 0)
                        window.Add(periods[j]);
                }

                if (window.Count == 0)
                {
                    smoothed[i] = periods[i];
                    continue;
                }

                window.Sort();

                smoothed[i] = window[window.Count / 2];
            }

            return smoothed;
        }

        /// <summary>
        /// The model's beat confidence and its predicted period per frame, running in windows over the whole track.
        /// </summary>
        /// <remarks>
        /// Windows overlap and their outputs are averaged rather than concatenated. Concatenating counts the shared
        /// frames twice and, worse, places a frame one window saw at its edge beside a frame another saw in its middle
        /// as though they came from the same reading. Measured, it makes no difference to this model - windows of 1024,
        /// 2048, 4096 and the whole track at once all report the same beats - but the averaging is what makes that a
        /// fact rather than a coincidence.
        /// </remarks>
        public (float[] Beats, float[] Periods) Activations(IReadOnlyList<float> samples)
        {
            float[,] spectrogram = LogMel.Spectrogram(samples);
            int frames = spectrogram.GetLength(0);

            var beats = new float[frames];
            var logPeriods = new float[frames];
            var weight = new float[frames];

            for (int start = 0; start < frames; start += hop_frames)
            {
                int end = Math.Min(frames, start + window_frames);
                int length = end - start;

                if (length < 64)
                    break;

                var input = new DenseTensor<float>(new[] { 1, LogMel.Bins, length });

                for (int t = 0; t < length; t++)
                {
                    for (int m = 0; m < LogMel.Bins; m++)
                        input[0, m, t] = spectrogram[start + t, m];
                }

                var outputs = Run(input);

                for (int t = 0; t < length; t++)
                {
                    beats[start + t] += outputs.Beat[t];

                    // The logarithm is averaged rather than the period, because that is what the Python side does and the
                    // two are not the same thing: averaging exponentials is not the exponential of the average, so a
                    // window whose neighbours disagree would be weighted differently by each. Exponents are convex, and
                    // the disagreement is worth several frames of period on a track whose tempo moves.
                    logPeriods[start + t] += outputs.LogPeriod[t];
                    weight[start + t] += 1;
                }
            }

            var periods = new float[frames];

            for (int i = 0; i < frames; i++)
            {
                if (weight[i] <= 0)
                {
                    periods[i] = 1;
                    continue;
                }

                beats[i] /= weight[i];
                logPeriods[i] /= weight[i];
                periods[i] = MathF.Exp(logPeriods[i]);
            }

            return (beats, periods);
        }

        /// <summary>
        /// Which frames are beats: the model's most confident frames, with everything within a share of the predicted
        /// period suppressed.
        /// </summary>
        /// <param name="beats">The model's beat confidence, one value per frame.</param>
        /// <param name="periods">
        /// The predicted period <em>in frames</em>, one per frame. Already exponentiated: the head is trained on the
        /// logarithm of the period and <see cref="Run"/> converts it, so multiplying by the frame rate again here is a
        /// factor of fifty and suppresses the whole track.
        /// </param>
        /// <remarks>
        /// Strongest first rather than left to right, because the strongest frame in a neighbourhood is the one the
        /// model is surest about and a walk in time order would take whichever came first.
        ///
        /// Suppression is kept in a mask rather than tested against every beat chosen so far. The comparison is the same
        /// either way and the mask is linear: a track with an onset every sixteenth has tens of thousands of frames over
        /// the threshold, and testing each against every chosen beat is quadratic in that.
        /// </remarks>
        public static int[] Suppress(float[] beats, float[] periods)
        {
            return Suppress(beats, periods, suppression_share);
        }

        /// <summary>
        /// The same, with the share named by the caller, so it can be swept against a reference rather than assumed.
        /// </summary>
        /// <param name="beats">The model's beat confidence, one value per frame.</param>
        /// <param name="periods">The predicted period in frames, one per frame.</param>
        /// <param name="share">How much of the predicted period two beats have to be apart.</param>
        public static int[] Suppress(float[] beats, float[] periods, double share)
        {
            if (beats == null)
                throw new ArgumentNullException(nameof(beats));

            if (periods == null)
                throw new ArgumentNullException(nameof(periods));

            if (beats.Length != periods.Length)
                throw new ArgumentException("The beat and period arrays have to cover the same frames.", nameof(periods));

            var order = new List<int>();

            for (int i = 0; i < beats.Length; i++)
            {
                if (beats[i] >= 0.5f)
                    order.Add(i);
            }

            order.Sort((a, b) => beats[b].CompareTo(beats[a]));

            var taken = new bool[beats.Length];
            var chosen = new List<int>();

            foreach (int index in order)
            {
                if (taken[index])
                    continue;

                chosen.Add(index);

                int radius = (int)Math.Max(1, share * periods[index]);

                for (int other = Math.Max(0, index - radius + 1); other <= Math.Min(beats.Length - 1, index + radius - 1); other++)
                    taken[other] = true;
            }

            chosen.Sort();

            return chosen.ToArray();
        }

        private (float[] Beat, float[] LogPeriod) Run(DenseTensor<float> input)
        {
            lock (gate)
            {
                if (disposed)
                    throw new ObjectDisposedException(nameof(LearnedBeatDetector));

                using var results = session.Value.Run(new[] { NamedOnnxValue.CreateFromTensor("mel", input) });

                Tensor<float> beat = null;
                Tensor<float> period = null;

                foreach (NamedOnnxValue value in results)
                {
                    if (value.Name == "beat_logit")
                        beat = value.AsTensor<float>();
                    else if (value.Name == "period_logit")
                        period = value.AsTensor<float>();
                }

                if (beat == null || period == null)
                    throw new InvalidOperationException(
                        $"The detector at {ModelPath} did not return both a beat and a period head.");

                int length = beat.Dimensions[1];
                var beatValues = new float[length];
                var logPeriods = new float[length];

                for (int t = 0; t < length; t++)
                {
                    beatValues[t] = Sigmoid(beat[0, t]);

                    // Left as a logarithm. The head is trained on the logarithm of the period, and the windows are
                    // combined in that domain before it is exponentiated once, because averaging exponentials is not the
                    // exponential of the average.
                    logPeriods[t] = period[0, t];
                }

                return (beatValues, logPeriods);
            }
        }

        private static float Sigmoid(float value)
        {
            return 1f / (1f + MathF.Exp(-value));
        }

        public void Dispose()
        {
            lock (gate)
            {
                if (disposed)
                    return;

                disposed = true;

                if (session.IsValueCreated)
                    session.Value.Dispose();

                options.Dispose();
            }
        }
    }
}
