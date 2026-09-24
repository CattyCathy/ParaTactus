using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using ParaTactus;

namespace ParaTactus
{
    /// <summary>
    /// Beat tracking with the Beat This! model: the frontend and inference the model was trained with, run through
    /// ONNX Runtime.
    /// </summary>
    /// <remarks>
    /// This lives in the game assembly rather than beside the tests because it is the shipping beat source, not a
    /// prototype. The hand-tuned estimator it replaces decides a tempo three different ways for the same candidate set
    /// (<c>FindBestLag</c>, <c>BuildCandidates</c> and <c>TrackTempoSequence</c>) and needed a prior, a phase-locked
    /// score, a continuity term and a segmenter to choose between them; measured against a beatmap's own timing points
    /// it got 32 of 35 sampled points wrong. This gets every steady section of the same track to a metrical level of
    /// the truth, and 26 of 26 consecutive intervals exactly right on a constant-tempo track.
    ///
    /// Everything here is transcribed from the published implementation rather than guessed at, because the frontend
    /// has to reproduce the one the model was trained with. From <c>beat_this/preprocessing.py</c>:
    ///
    /// <code>
    /// LogMelSpect(sample_rate=22050, n_fft=1024, hop_length=441, f_min=30, f_max=11000,
    ///             n_mels=128, mel_scale="slaney", normalized="frame_length", power=1)
    /// forward: log1p(1000 * mel_spectrogram(x).T)
    /// </code>
    ///
    /// so: resample to 22.05kHz, magnitude spectrum (power=1, not power=2), 128 Slaney mel bands, and the logarithm of
    /// a thousand times that. The hop of 441 at 22050Hz is exactly 50 frames per second, which is the frame rate the
    /// model's outputs are in and the reason none of the timings below need converting.
    ///
    /// One detail is worth spelling out because getting it wrong is silent. torchaudio's <c>mel_scale="slaney"</c>
    /// chooses the frequency scale only; whether the filters are also scaled to unit area is a separate <c>norm</c>
    /// argument, and <c>MelSpectrogram</c> leaves <c>norm</c> at <c>None</c>. The published frontend therefore uses raw
    /// triangular filters peaking at one. Applying the Slaney area normalisation as well tapers the high bands by up
    /// to 25x and tilts the spectrogram, which the model tolerates on steady music and falls apart on dense music.
    /// <c>normalized="frame_length"</c> is separate again and does apply: it divides the STFT by <c>sqrt(n_fft)</c>.
    ///
    /// From <c>beat_this/inference.py</c>, long inputs are cut into 1500-frame chunks (30 seconds) with 6 frames of
    /// overlap at each edge, because the model was not trained on its own input edges.
    ///
    /// From <c>beat_this/model/postprocessor.py</c>, the "minimal" postprocessing the paper is named for: a peak is a
    /// frame at least as large as its neighbours within 3 frames either way, kept if its logit is above zero, with
    /// adjacent peaks merged.
    /// </remarks>
    public static class BeatThisBeatTracker
    {
        private const int sample_rate = 22050;
        private const int n_fft = 1024;
        private const int hop_length = 441;
        private const double f_min = 30;
        private const double f_max = 11000;
        private const int n_mels = 128;
        private const int chunk_frames = 1500;
        private const int border_frames = 6;
        private const int peak_radius = 3;
        private const int fps = 50;

        /// <summary>How much shorter than the local beat period a gap has to be before it counts as an insertion.</summary>
        /// <remarks>
        /// Half was too loose and it is the direct cause of the most visible failure this had. On Designant the beatmap
        /// holds 200 BPM from 145s to 175s, and the model's inter-beat gaps there scatter between 140ms and 600ms while
        /// their median stays near 300 - so every tempo measurement reads correctly and the beats are still unusable,
        /// because the visuals pulse on an irregular train. Half of 300ms is 150, which removes the 140s and keeps every
        /// 160, 180, 220 and 260, which are the model firing on subdivisions of the beat.
        ///
        /// Seven tenths removes those while staying safe on genuinely fast music: the period is estimated locally, so in
        /// a real 400 BPM passage the period is 150ms and this only removes gaps under 105.
        /// </remarks>
        private const double prune_ratio = 0.7;

        /// <summary>How many beats either side of a gap set the local beat period.</summary>
        private const int prune_window = 8;

        /// <summary>Model beat times in milliseconds, taken from a file and decoded by the caller's decoder.</summary>
        /// <remarks>
        /// A decoder is asked for rather than chosen, because the only one this project has ever used is proprietary
        /// and the analysis should not be. See <see cref="IAudioDecoder"/>. Callers that already hold samples, or that
        /// decode with something else, want the overload below instead.
        /// </remarks>
        public static double[] BeatTimes(string audioPath, string modelPath, IAudioDecoder decoder)
        {
            if (decoder == null)
                throw new ArgumentNullException(nameof(decoder));

            return BeatTimes(decoder.DecodeMono(audioPath, AnalysisAudio.SampleRate), modelPath);
        }

        /// <summary>Model beat times in milliseconds, for mono audio at <see cref="AnalysisAudio.SampleRate"/>.</summary>
        /// <remarks>
        /// Also accepts samples directly, which is what callers that have already decoded a track - or that are feeding
        /// audio as it plays - should use.
        /// </remarks>
        public static double[] BeatTimes(float[] samples, string modelPath)
        {
            var spectrogram = LogMelSpectrogram(samples);
            var (beats, _) = Predict(spectrogram, modelPath);

            return beats.Select(frame => frame * 1000.0 / fps).ToArray();
        }

        /// <summary>
        /// Log-mel spectrogram of a waveform, shaped (frames, 128).
        /// </summary>
        internal static float[,] LogMelSpectrogram(float[] samples)
        {
            return LogMelSpectrogram(samples, 0, FrameCount(samples.Length));
        }

        /// <summary>
        /// One over the square root of the transform size: the scale torchaudio's <c>normalized="frame_length"</c>
        /// applies to the STFT.
        /// </summary>
        private static readonly double frame_length_scale = 1.0 / Math.Sqrt(n_fft);

        /// <summary>
        /// Log-mel spectrogram of frames <paramref name="firstFrame"/> onwards, shaped (frameCount, 128).
        /// </summary>
        /// <remarks>
        /// The frame grid is global - frame t is always centred on sample t * hop - so a range of frames from the
        /// middle of a track is the same computation as those frames of the whole track. That is what lets a track be
        /// analysed in pieces without the pieces disagreeing with the whole, which a frontend that re-centred its
        /// window on each piece could not do.
        /// </remarks>
        internal static float[,] LogMelSpectrogram(float[] samples, int firstFrame, int frameCount)
        {
            var frame = new float[n_fft];
            var magnitudes = new float[Fft.BIN_COUNT];
            var filterbank = SlaneyMelFilterbank();
            var result = new float[frameCount, n_mels];

            for (int t = 0; t < frameCount; t++)
            {
                int start = (firstFrame + t) * hop_length - n_fft / 2;

                for (int i = 0; i < n_fft; i++)
                    frame[i] = SampleAt(samples, start + i);

                // AnalyseMagnitudes applies the Hann window itself, so pass the raw frame: windowing here as well
                // would give the model a Hann-squared window and a mel spectrogram it was never trained on.
                Fft.AnalyseMagnitudes(frame, magnitudes);

                for (int m = 0; m < n_mels; m++)
                {
                    double sum = 0;

                    for (int bin = 0; bin < Fft.BIN_COUNT; bin++)
                        sum += filterbank[m, bin] * magnitudes[bin];

                    // The frame_length scale is linear, so applying it here rather than to every magnitude is the
                    // same number with fewer multiplies.
                    result[t, m] = (float)Math.Log(1.0 + 1000.0 * frame_length_scale * sum);
                }
            }

            return result;
        }

        /// <summary>How many frames a whole waveform produces.</summary>
        internal static int FrameCount(int sampleCount)
        {
            return 1 + sampleCount / hop_length;
        }

        /// <summary>
        /// How many frames can be analysed from a waveform this long without using samples that have not arrived yet.
        /// </summary>
        /// <remarks>
        /// The last frames of a complete track are produced by reflecting the tail, which for a track still being fed
        /// would mean reflecting onto audio that does not exist yet. Frames are therefore only counted once their whole
        /// window lies inside the samples that are actually present, and the reflected tail is added at the end.
        /// </remarks>
        public static int CompleteFrameCount(int sampleCount)
        {
            int pad = n_fft / 2;

            if (sampleCount < n_fft - pad)
                return 0;

            return (sampleCount - n_fft + pad) / hop_length + 1;
        }

        /// <summary>The time of a frame, in milliseconds.</summary>
        public static double FrameToMilliseconds(double frame)
        {
            return frame * 1000.0 / fps;
        }

        /// <summary>The model's frame rate. The hop of 441 at 22050Hz is exactly 50 frames per second.</summary>
        public const int FramesPerSecond = fps;

        /// <summary>The number of frames the model was trained on at once.</summary>
        public const int ModelChunkFrames = chunk_frames;

        /// <summary>How many frames at each end of a chunk the model is not trusted on.</summary>
        public const int ModelBorderFrames = border_frames;

        /// <summary>
        /// The sample at an index, reflected at the edges, which is how torchaudio's <c>center=True</c> padding works.
        /// </summary>
        /// <remarks>
        /// Reflection does not repeat the edge sample, so index -1 reads sample 1 and not sample 0. Reading the edge
        /// twice instead is the "symmetric" mode and would shift every frame slightly.
        /// </remarks>
        private static float SampleAt(float[] samples, int index)
        {
            if (samples.Length == 0)
                return 0;

            if (samples.Length == 1)
                return samples[0];

            while (index < 0 || index >= samples.Length)
                index = index < 0 ? -index : 2 * samples.Length - 2 - index;

            return samples[index];
        }

        /// <summary>
        /// Triangular mel filters on the Slaney scale, un-normalised.
        /// </summary>
        /// <remarks>
        /// The frequency scale is the Slaney one - linear below 1kHz and logarithmic above - which is what
        /// <c>mel_scale="slaney"</c> selects, and it is not the same as the HTK scale that most implementations
        /// default to. The filters are built over the bins the FFT produces; the Nyquist bin torchaudio would add is
        /// above every filter's right edge at <c>f_max</c>, so leaving it out changes nothing.
        ///
        /// The weights peak at one and are <b>not</b> area-normalised. That is the part that is easy to get wrong:
        /// <c>mel_scale="slaney"</c> is the frequency scale, while area normalisation is a separate <c>norm</c>
        /// argument, and <c>torchaudio.transforms.MelSpectrogram</c> defaults <c>norm</c> to <c>None</c> even when the
        /// scale is Slaney. <c>LogMelSpect</c> never passes <c>norm</c>, so the published frontend uses raw triangles.
        /// Scaling each filter to unit area instead - which is what <c>norm="slaney"</c> would do, and what this code
        /// originally did - tapers the high bands by up to 25x and tilts the whole spectrogram.
        /// </remarks>
        internal static double[,] SlaneyMelFilterbank()
        {
            var edges = new double[n_mels + 2];
            double low = hzToMel(f_min);
            double high = hzToMel(f_max);

            for (int i = 0; i < edges.Length; i++)
                edges[i] = melToHz(low + (high - low) * i / (n_mels + 1));

            var filterbank = new double[n_mels, Fft.BIN_COUNT];

            for (int m = 1; m <= n_mels; m++)
            {
                double left = edges[m - 1];
                double centre = edges[m];
                double right = edges[m + 1];

                for (int bin = 0; bin < Fft.BIN_COUNT; bin++)
                {
                    double frequency = bin * (double)sample_rate / n_fft;
                    double weight = 0;

                    if (frequency >= left && frequency <= centre)
                        weight = (frequency - left) / Math.Max(1e-9, centre - left);
                    else if (frequency > centre && frequency <= right)
                        weight = (right - frequency) / Math.Max(1e-9, right - centre);

                    filterbank[m - 1, bin] = weight;
                }
            }

            return filterbank;
        }

        private static double hzToMel(double frequency)
            => frequency < 1000 ? 3.0 * frequency / 200.0 : 15.0 + 27.0 * Math.Log(frequency / 1000.0) / Math.Log(6.4);

        private static double melToHz(double mel)
            => mel < 15 ? 200.0 * mel / 3.0 : 1000.0 * Math.Exp((mel - 15.0) * Math.Log(6.4) / 27.0);

        /// <summary>Runs the model over the whole spectrogram and returns the frames it reports as beats.</summary>
        internal static (List<double> Beats, List<double> Downbeats) Predict(float[,] spectrogram, string modelPath)
        {
            var (beat, downbeat) = Logits(spectrogram, modelPath);

            return (Peaks(beat), Peaks(downbeat));
        }

        /// <summary>
        /// The model's raw per-frame beat and downbeat activation.
        /// </summary>
        /// <remarks>
        /// Separate from <see cref="Predict"/> because whether a beat exists and how strongly the model believes in it
        /// are different questions, and the second is the only way to tell a genuinely slow track from one the model
        /// has halved: both produce the same beat times, so the times alone cannot decide between them.
        /// </remarks>
        /// <summary>
        /// Opens the model, for callers that want to run several chunks through one session.
        /// </summary>
        /// <remarks>
        /// Loading the model costs seconds, so anything that runs more than one chunk - the streaming tracker, which
        /// runs one per 30 seconds of audio - has to keep the session rather than build one per call.
        ///
        /// The thread count is deliberately below what the machine has. ONNX Runtime takes every logical processor by
        /// default and holds them for the better part of a minute, and this runs while a track is being played, so the
        /// audio thread ends up queued behind it and the playback stutters. Two spare processors cost some analysis
        /// time that nothing is waiting on and give the audio callback somewhere to run.
        /// </remarks>
        internal static InferenceSession OpenSession(string modelPath)
        {
            var options = new SessionOptions
            {
                IntraOpNumThreads = Math.Max(1, Environment.ProcessorCount - 2),
                InterOpNumThreads = 1,
            };

            return new InferenceSession(modelPath, options);
        }

        /// <summary>
        /// Runs the model over one prepared chunk, returning the beat and downbeat activation for every frame of it.
        /// </summary>
        /// <remarks>
        /// This is deliberately not <see cref="Predict"/>: it does no chunking of its own. A caller that has already
        /// decided what one chunk is - the streaming tracker, which has to match the whole-file chunk schedule frame
        /// for frame - must not have its input split up again underneath it.
        /// </remarks>
        internal static (float[] Beats, float[] Downbeats) RunChunk(InferenceSession session, Tensor<float> input)
        {
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("spectrogram", input) };

            using var outputs = session.Run(inputs);
            var beat = outputs.First(o => o.Name == "beat").AsTensor<float>();
            var downbeat = outputs.First(o => o.Name == "downbeat").AsTensor<float>();

            int length = input.Dimensions[1];
            var beats = new float[length];
            var downbeats = new float[length];

            for (int t = 0; t < length; t++)
            {
                beats[t] = beat[0, t];
                downbeats[t] = downbeat[0, t];
            }

            return (beats, downbeats);
        }

        internal static (float[] Beats, float[] Downbeats) Logits(float[,] spectrogram, string modelPath)
        {
            int total = spectrogram.GetLength(0);
            var beatLogits = new float[total];
            var downbeatLogits = new float[total];

            // Frames no chunk claims are left at -1000, which no peak test can pass, rather than at zero, which a
            // correct-but-quiet frame would also read as.
            for (int i = 0; i < total; i++)
            {
                beatLogits[i] = -1000;
                downbeatLogits[i] = -1000;
            }

            using var session = OpenSession(modelPath);

            // Chunks are taken in reverse so the earlier of two overlapping chunks wins, matching the reference's
            // "keep_first" aggregation.
            var starts = ChunkStarts(total);

            for (int c = starts.Count - 1; c >= 0; c--)
            {
                int start = starts[c];
                int from = Math.Max(start, 0);
                int to = Math.Min(start + chunk_frames, total);

                // The model was not trained on its own input edges, so the first chunk is padded with zeros at the
                // head and the last one at the tail and those frames are then thrown away below.
                int left = Math.Max(0, -start);
                int right = Math.Max(0, Math.Min(border_frames, start + chunk_frames - total));
                int length = left + (to - from) + right;

                if (length <= 2 * border_frames)
                    continue;

                var input = new DenseTensor<float>(new[] { 1, length, n_mels });

                for (int t = 0; t < to - from; t++)
                {
                    for (int m = 0; m < n_mels; m++)
                        input[0, left + t, m] = spectrogram[from + t, m];
                }

                var (beat, downbeat) = RunChunk(session, input);

                // The chunk is trusted only between its own borders; the frames outside that belong to a neighbour.
                int owned = start + border_frames;

                for (int t = 0; t < length - 2 * border_frames; t++)
                {
                    int index = owned + t;

                    if (index < 0 || index >= total)
                        continue;

                    beatLogits[index] = beat[border_frames + t];
                    downbeatLogits[index] = downbeat[border_frames + t];
                }
            }

            return (beatLogits, downbeatLogits);
        }

        /// <summary>
        /// Where the model's chunks start, in frames.
        /// </summary>
        /// <remarks>
        /// Transcribed from <c>split_piece</c> in the reference. Two details matter and neither is obvious. The first
        /// chunk starts at a <em>negative</em> frame, because the border it is not trusted on is filled with zeros
        /// rather than taken from the audio, so the first chunk's real content begins exactly at frame zero. And the
        /// last start is moved left to <c>total - (chunk - border)</c> so that the final chunk ends exactly on the last
        /// frame instead of running off the end and leaving a short tail the model would be asked to judge unedited.
        ///
        /// Starting at zero instead, as this did, shifts every chunk boundary by the border width and leaves the tail
        /// unprocessed; it changes what the model sees by enough to move beats near every boundary.
        /// </remarks>
        private static List<int> ChunkStarts(int total)
        {
            int stride = chunk_frames - 2 * border_frames;
            var starts = new List<int>();

            for (int start = -border_frames; start < total - border_frames; start += stride)
                starts.Add(start);

            if (starts.Count > 0 && total > stride)
                starts[starts.Count - 1] = total - (chunk_frames - border_frames);

            return starts;
        }

        /// <summary>
        /// Runs the model over the whole spectrogram and returns the frames it reports as beats.
        /// </summary>
        internal static List<double> Peaks(float[] logits)
        {
            return Prune(RawPeaks(logits), logits);
        }

        /// <summary>
        /// The peaks before <see cref="Prune"/> has seen them, in frames.
        /// </summary>
        /// <remarks>
        /// Exposed so the pruning can be measured rather than assumed: whether a wrongly placed beat was put there by
        /// the model or by the pruning is not visible in the pruned output, and the two have different fixes.
        /// </remarks>
        internal static List<double> RawPeaks(float[] logits)
        {
            var peaks = new List<int>();

            for (int i = 0; i < logits.Length; i++)
            {
                if (logits[i] <= 0)
                    continue;

                bool isPeak = true;

                for (int j = Math.Max(0, i - peak_radius); j <= Math.Min(logits.Length - 1, i + peak_radius); j++)
                {
                    if (logits[j] > logits[i])
                    {
                        isPeak = false;
                        break;
                    }
                }

                if (isPeak)
                    peaks.Add(i);
            }

            // Adjacent frames within one of each other describe one beat; the average of the group is its position.
            // The average is a real number rather than a frame, which is what the reference keeps it as
            // (deduplicate_peaks in the published postprocessor). Rounding it to a frame here would move every merged
            // plateau by up to one frame, which is 20ms at this frame rate and over 3% of a 600ms beat. The mean used
            // to be accumulated in an int, so integer division truncated it at every step of the group.
            var merged = new List<double>();

            if (peaks.Count == 0)
                return merged;

            double position = peaks[0];
            int count = 1;

            for (int i = 1; i < peaks.Count; i++)
            {
                if (peaks[i] - position <= 1)
                {
                    count++;
                    position += (peaks[i] - position) / count;
                }
                else
                {
                    merged.Add(position);
                    position = peaks[i];
                    count = 1;
                }
            }

            merged.Add(position);

            return merged;
        }

        /// <summary>
        /// The frame a peak's fractional position is nearest to, clamped into the activation it indexes.
        /// </summary>
        /// <remarks>
        /// A peak is the mean of a group of adjacent frames, so it is not a whole frame. Anything that wants the
        /// activation a peak was found in wants the frame it sits on, and that is what this answers. Internal so the
        /// diagnostics in the test suite read peaks exactly the way the tracker does rather than approximating it.
        /// </remarks>
        internal static int FrameOf(double position, int length)
        {
            int index = (int)Math.Round(position, MidpointRounding.AwayFromZero);

            return index < 0 ? 0 : index >= length ? length - 1 : index;
        }

        /// <summary>
        /// Drops beats that sit far closer to a neighbour than the local beat period, keeping the stronger of the two.
        /// </summary>
        /// <remarks>
        /// The model occasionally fires between two beats as well as on them. Those extras are what drag a median
        /// interval off the tempo the model is actually reporting - on the track this was measured against they turned
        /// a clean 600ms pulse into a 460ms one in places, which reads as a tempo a third too fast.
        ///
        /// The threshold is deliberately conservative. A gap can only be removed when it is under half the local beat
        /// period, because the alternative mistake is far worse: a genuine acceleration and a spurious insertion look
        /// the same in one gap, and a threshold loose enough to remove the second would delete real beats from the
        /// 400 BPM section, where the beat period itself is 150ms.
        /// </remarks>
        private static List<double> Prune(List<double> frames, float[] logits)
        {
            if (frames.Count < 3)
                return frames;

            var keep = new bool[frames.Count];

            for (int i = 0; i < keep.Length; i++)
                keep[i] = true;

            // Removing a beat can expose the next one as too close in turn, so this runs to a fixed point.
            for (int pass = 0; pass < 8; pass++)
            {
                bool changed = false;

                for (int i = 0; i + 1 < frames.Count; i++)
                {
                    if (!keep[i] || !keep[i + 1])
                        continue;

                    double period = LocalPeriod(frames, keep, i);

                    if (period <= 0)
                        continue;

                    if ((frames[i + 1] - frames[i]) / period >= prune_ratio)
                        continue;

                    // The weaker of the pair is the insertion, so it is the one that goes. The position is a mean within
                    // the group that produced it, so rounding it lands on one of that group's own frames and the
                    // activation read is the activation at the beat.
                    int left = FrameOf(frames[i], logits.Length);
                    int right = FrameOf(frames[i + 1], logits.Length);

                    keep[logits[left] >= logits[right] ? i + 1 : i] = false;
                    changed = true;
                }

                if (!changed)
                    break;
            }

            var result = new List<double>();

            for (int i = 0; i < frames.Count; i++)
            {
                if (keep[i])
                    result.Add(frames[i]);
            }

            return result;
        }

        /// <summary>How many beats either side of a gap are candidates for the period it is judged against.</summary>
        /// <remarks>
        /// Much wider than <see cref="prune_window"/> on purpose, and the difference matters more than it looks. The
        /// period has to be estimated from a neighbourhood in which the real beats are the majority, because every
        /// estimator of a dominant value fails when the thing it is estimating from is mostly the thing it is trying to
        /// find. A dense burst of inserted beats is exactly that case: over a window of sixteen gaps the inserts are
        /// most of the sample, the mode lands on the inserted spacing, and the pruning then confirms it and keeps them.
        /// That is what made a passage of Designant read 428 BPM where the surrounding music is at 200, and widening
        /// this from the sixteen gaps that were enough for the median to thirty-two is what fixed it.
        ///
        /// A span of time rather than a count of gaps was tried here and measured worse, not better: on Stage 5 the
        /// burst that reads 500 BPM then held for nine seconds instead of six. The reason is that a burst long enough to
        /// be the majority of a wide window is also long enough to be the majority of a wider one, so widening buys
        /// nothing while making the estimate slower to follow a real tempo change.
        /// </remarks>
        private const int period_window = 32;

        /// <summary>
        /// The beat period around a beat, in frames, taken as the dominant gap in the neighbourhood rather than as the
        /// median of the gaps.
        /// </summary>
        /// <remarks>
        /// The median is the obvious choice and it is the wrong one here, because the thing being measured is the thing
        /// being removed. An inserted beat turns one gap into two shorter ones, and the model tends to fire between
        /// beats in runs, so in a passage with a third of its gaps spurious the median lands on a spurious gap and the
        /// pruning then concludes that the spurious spacing is the real one and keeps it. That is not a corner case: it
        /// is what made the reported tempo read 300 where the music is at 200.
        ///
        /// The dominant gap - the value with the most neighbours within a quarter of itself - stays on the period the
        /// model is actually marking even when a minority of its gaps are wrong, which is the same reason a mode is
        /// used for this in tempo estimation generally.
        /// </remarks>
        private static double LocalPeriod(List<double> frames, bool[] keep, int index)
        {
            var gaps = new List<double>();

            for (int i = Math.Max(0, index - period_window); i < Math.Min(frames.Count - 1, index + period_window); i++)
            {
                if (keep[i] && keep[i + 1])
                    gaps.Add(frames[i + 1] - frames[i]);
            }

            if (gaps.Count == 0)
                return 0;

            if (gaps.Count < 3)
            {
                gaps.Sort();
                return gaps[gaps.Count / 2];
            }

            double bestCentre = 0;
            int bestCount = 0;

            foreach (double candidate in gaps)
            {
                var cluster = new List<double>();

                foreach (double gap in gaps)
                {
                    if (gap >= candidate * 0.75 && gap <= candidate * 1.25)
                        cluster.Add(gap);
                }

                // Ties go to the longer gap: a doubled interval is a real beat that was missed, a halved one is a beat
                // that was inserted, and keeping the longer period is the choice that removes rather than preserves.
                if (cluster.Count > bestCount || (cluster.Count == bestCount && candidate > bestCentre))
                {
                    bestCount = cluster.Count;
                    cluster.Sort();
                    bestCentre = cluster[cluster.Count / 2];
                }
            }

            return bestCentre;
        }
    }
}
