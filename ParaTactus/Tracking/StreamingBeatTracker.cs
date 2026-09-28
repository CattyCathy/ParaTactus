using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using ParaTactus.Detector;

namespace ParaTactus.Tracking
{
    /// <summary>
    /// Finds beats in a track as its audio arrives, instead of analysing the whole file at once.
    /// </summary>
    /// <remarks>
    /// The model path costs about a third of the track's own length, all of it inference - 65 seconds for a 178 second
    /// track on a 24-core machine. Blocking a player on that is not acceptable, but the same measurement says inference
    /// runs at roughly a quarter of realtime, so a tracker fed audio as it plays stays ahead of the playhead after one
    /// chunk of head start. This is that tracker.
    ///
    /// The pieces are the same computation as the whole, not an approximation of it. The mel frame grid is global -
    /// frame t is always centred on sample t * hop - so analysing frames 1500..3000 of a track in isolation gives the
    /// same numbers as analysing them as part of the whole file. A frontend that re-centred its window on each piece
    /// could not make that claim, and the drift it would introduce is exactly what this work exists to remove.
    ///
    /// Each piece is analysed with a little of the audio before it, because the model sees only its own chunk and is
    /// least reliable at the chunk's edges. Beats from the repeat are then discarded, so no beat is reported twice.
    /// </remarks>
    public sealed class StreamingBeatTracker : IDisposable
    {
        /// <summary>
        /// How many frames at the end of the assembled activation are left undecided until more audio arrives.
        /// </summary>
        /// <remarks>
        /// The peak picking reaches a bounded distance - it looks a fixed number of gaps either side of a gap and then
        /// settles - and this is deliberately several times that bound rather than exactly it. It costs 1.3 seconds of
        /// extra delay on beats that are being found thirty seconds ahead of the playhead anyway.
        /// </remarks>
        private const int settle_frames = 64;

        private readonly string modelPath;
        private readonly CancellationToken cancellation;
        private readonly List<float> samples = new List<float>();
        private readonly List<double> beats = new List<double>();

        /// <summary>The assembled beat activation, one entry per frame from the start of the track.</summary>
        private readonly List<float> activation = new List<float>();

        /// <summary>The assembled downbeat activation, the same length and the same frames.</summary>
        private readonly List<float> downbeat = new List<float>();

        private InferenceSession session;
        private int nextFrame;
        private int emittedThrough;
        private bool flushed;

        private readonly int threads;

        public StreamingBeatTracker(string modelPath, CancellationToken cancellation = default)
            : this(modelPath, 0, cancellation)
        {
        }

        /// <summary>
        /// A tracker that runs the model on a chosen number of threads.
        /// </summary>
        /// <param name="modelPath">The model to run.</param>
        /// <param name="threads">
        /// How many threads the model may use, or zero to leave the machine two spare processors. The default is for a
        /// track being played, where the audio callback must not end up queued behind the analysis; passing a count is
        /// for analysing a corpus, where nothing is playing and the spare processors go to waste.
        /// </param>
        /// <param name="cancellation">Cancels a long analysis.</param>
        public StreamingBeatTracker(string modelPath, int threads, CancellationToken cancellation = default)
        {
            if (string.IsNullOrEmpty(modelPath))
                throw new ArgumentException("A model path is needed to track beats.", nameof(modelPath));

            this.modelPath = modelPath;
            this.threads = threads;
            this.cancellation = cancellation;
        }

        /// <summary>The beats found so far, in milliseconds from the start of the track.</summary>
        public IReadOnlyList<double> Beats => beats;

        /// <summary>
        /// The model's own belief about a beat, one value per frame from the start of the track.
        /// </summary>
        /// <remarks>
        /// The raw output rather than anything derived from it, and exposed because everything this class does with it
        /// - picking peaks, searching for a tempo, spacing the result onto a metre - loses the part of it that says how
        /// sure the model was. Where a beat is and whether there is one are different questions, and a reader that
        /// wants the second cannot get it back out of a list of positions.
        ///
        /// Only meaningful once the whole track has been flushed, for the same reason <see cref="SearchedBeats"/> is.
        /// </remarks>
        public IReadOnlyList<float> Activation => activation;

        /// <summary>
        /// The model's own belief about a downbeat, one value per frame from the start of the track.
        /// </summary>
        /// <remarks>
        /// Kept beside <see cref="Activation"/> rather than thrown away, and thrown away is what happened to it: the
        /// chunk that produced it run took the beat head and dropped this one. What it adds is the metre - where the bar
        /// starts - which is the kind of context that a decision about the metrical level needs, and it costs one array
        /// to keep.
        ///
        /// It is a separate head and not a stronger beat, so nothing here should be read as the beat head scaled: on
        /// this model's material the two disagree about a third of the time, and the disagreement is the information.
        /// </remarks>
        public IReadOnlyList<float> Downbeats => downbeat;

        /// <summary>
        /// The beats chosen by searching for the steadiest tempo, over the same activation.
        /// </summary>
        /// <remarks>
        /// A second reading of the same evidence, not a replacement for <see cref="Beats"/>. Peak picking decides each
        /// beat on its own merits with a local rule about gaps, so nothing in it can say "the tempo was 300ms a moment
        /// ago and it cannot be 500ms now" - and on a track whose tempo moves that is the whole failure. The search
        /// carries the interval as part of its state and pays for changing it, which is what stops a phase coming
        /// adrift while the tempo is very nearly right.
        ///
        /// Only meaningful once the whole track has been flushed, because the search reads the whole activation: a path
        /// chosen from part of a track is a different path, and the beats near the end of what has arrived would move
        /// every time more audio did.
        /// </remarks>
        public IReadOnlyList<double> SearchedBeats
            => BeatSequenceSearch.Frames(activation.ToArray(), null)
                                 .Select(frame => BeatThisBeatTracker.FrameToMilliseconds(frame))
                                 .ToArray();

        /// <summary>How much audio has been fed in, in samples.</summary>
        public int SampleCount => samples.Count;

        /// <summary>How far the beats found so far reach, in milliseconds.</summary>
        public double BeatsThrough => nextFrame * 1000.0 / BeatThisBeatTracker.FramesPerSecond;

        /// <summary>
        /// Adds more mono audio at 22050Hz. Beats for the newly analysable audio are found, which costs about a
        /// quarter of the duration added.
        /// </summary>
        /// <remarks>
        /// The model runs on the calling thread and blocks it for that cost. That is deliberate - which thread pays for
        /// an analysis is the caller's decision, and the caller is the one that knows whether it has a thread to spare -
        /// but it is the thing to know before wiring this to a UI: adding a second of audio stalls the calling thread
        /// for about 250ms. <see cref="BeatGridProvider"/> is the wrapper that puts it on a background thread.
        ///
        /// The audio added is kept for the lifetime of the tracker, because a chunk is analysed with the samples before
        /// it as well as its own. That is about 88KB per second of track.
        /// </remarks>
        public void Add(ReadOnlySpan<float> chunk)
        {
            if (flushed)
                throw new InvalidOperationException("The track has already been flushed; start a new tracker for a new track.");

            samples.AddRange(chunk);

            Drain(false);
        }

        /// <summary>
        /// Finishes the track, analysing every frame that is left including the reflected tail. Call this once the last
        /// of the audio has been added, from the same thread that has been adding it.
        /// </summary>
        public void Flush()
        {
            if (flushed)
                return;

            Drain(true);
            flushed = true;
            Dispose();
        }

        /// <summary>
        /// Releases the model. Called by <see cref="Flush"/> as well, because once the track is finished the session
        /// has nothing left to do and holding it keeps the model's weights resident.
        /// </summary>
        public void Dispose()
        {
            session?.Dispose();
            session = null;
        }

        private void Drain(bool final)
        {
            cancellation.ThrowIfCancellationRequested();

            int available = final
                ? BeatThisBeatTracker.FrameCount(samples.Count)
                : BeatThisBeatTracker.CompleteFrameCount(samples.Count);

            const int chunk = BeatThisBeatTracker.ModelChunkFrames;
            const int border = BeatThisBeatTracker.ModelBorderFrames;
            const int stride = chunk - 2 * border;

            // The regular chunks, each owning `stride` frames. One of these is never the chunk the reference moves:
            // needing the frames to run it at all means the audio already reaches past the next start, so a later
            // chunk exists and this one is not the last.
            while (true)
            {
                // Checked here as well as at the top, because one model chunk is the bulk of the work and a cancelled
                // analysis should stop before paying for the next one. Superseded analyses used to run to completion,
                // and several of them at once - each holding most of the machine's threads - is what made the player
                // stutter while a track was merely being chosen.
                cancellation.ThrowIfCancellationRequested();

                int needed = nextFrame == 0 ? chunk - border : nextFrame + chunk - border;

                if (available < needed)
                    break;

                if (nextFrame == 0)
                {
                    // The first chunk is not trusted on its leading border, so that border is filled with zeros rather
                    // than audio and the chunk's real content begins exactly at frame zero.
                    Analyse(0, chunk - border, border, 0, 0, stride);
                }
                else
                {
                    Analyse(nextFrame - border, chunk, 0, 0, nextFrame, nextFrame + stride);
                }

                nextFrame += stride;
            }

            if (!final)
            {
                Emit(false);
                return;
            }

            // The last chunk is the one the reference moves: its start is pulled left so the chunk ends on the last
            // frame instead of running past it and leaving a short tail for the model to judge unedited.
            int lastStart = available - (chunk - border);
            int first = Math.Max(nextFrame, lastStart + border);

            if (lastStart < 0)
            {
                // Too short to hold even one chunk. The reference has no well-defined behaviour here either, so the
                // whole of it is treated as a single chunk with the same zero borders.
                if (nextFrame < available)
                    Analyse(0, available, border, Math.Max(0, Math.Min(border, chunk - border - available)), nextFrame, available);
            }
            else if (first < available)
            {
                Analyse(lastStart, chunk - border, 0, border, first, available);
            }

            nextFrame = available;
            Emit(true);
        }

        /// <summary>
        /// Picks the beats out of the activation assembled so far.
        /// </summary>
        /// <remarks>
        /// The peak picking runs over the whole assembled activation rather than over each chunk, because it is not a
        /// per-frame operation: a peak's fate depends on the peaks near it, so a chunk that picks its own peaks makes a
        /// different decision about a beat on its boundary than the whole-file analysis does. Running it over
        /// everything is what makes the streamed beats the same beats, not merely similar ones.
        ///
        /// Until the track ends, the last <see cref="settle_frames"/> frames are left alone: a beat there could still
        /// be changed by peaks belonging to audio that has not arrived, so it is decided again on the next call rather
        /// than emitted now.
        /// </remarks>
        private void Emit(bool final)
        {
            int count = activation.Count;
            int limit = final ? count : count - settle_frames;

            if (limit <= emittedThrough)
                return;

            var peaks = BeatThisBeatTracker.Peaks(activation.ToArray());

            foreach (double peak in peaks)
            {
                if (peak < emittedThrough || peak >= limit)
                    continue;

                beats.Add(BeatThisBeatTracker.FrameToMilliseconds(peak));
            }

            emittedThrough = limit;
        }

        /// <summary>
        /// Runs one chunk and stores the activation it owns into the assembled track activation.
        /// </summary>
        /// <remarks>
        /// The model input is <paramref name="leftPad"/> zero frames, then frames
        /// <c>[firstFrame, firstFrame + frameCount)</c> of the track, then <paramref name="rightPad"/> more zeros -
        /// the same construction the whole-file path uses, so the two see identical input.
        /// </remarks>
        private void Analyse(int firstFrame, int frameCount, int leftPad, int rightPad, int firstOwned, int endOwned)
        {
            // The tracker's own list, not a copy of it: this is inside the per-chunk loop, and copying the whole
            // history per chunk is what made a long track cost more than the sum of its chunks.
            float[,] spectrogram = BeatThisBeatTracker.LogMelSpectrogram(samples, firstFrame, frameCount);

            int bins = spectrogram.GetLength(1);
            var input = new DenseTensor<float>(new[] { 1, leftPad + frameCount + rightPad, bins });

            for (int t = 0; t < frameCount; t++)
            {
                for (int m = 0; m < bins; m++)
                    input[0, leftPad + t, m] = spectrogram[t, m];
            }

            // The model is run once on this chunk and this chunk only. Going through the whole-file entry point would
            // split the input up again on the whole-file schedule, which is not what a chunk that has already been cut
            // to the reference's own chunk size should be put through.
            session ??= threads > 0
                ? BeatThisBeatTracker.OpenSession(modelPath, threads)
                : BeatThisBeatTracker.OpenSession(modelPath);

            var (chunkActivation, chunkDownbeat) = BeatThisBeatTracker.RunChunk(session, input);

            // Chunks own contiguous runs of frames, so this normally just appends; a gap is filled with the same
            // not-a-peak value the whole-file path uses for frames no chunk claimed.
            while (activation.Count < firstOwned)
            {
                activation.Add(-1000);
                downbeat.Add(-1000);
            }

            for (int frame = firstOwned; frame < endOwned; frame++)
            {
                // The chunk's own index for a track frame is the frame minus the chunk's first frame, shifted by
                // whatever zeros were put in front of it.
                int index = frame - firstFrame + leftPad;
                float value = index >= 0 && index < chunkActivation.Length ? chunkActivation[index] : -1000;
                float bar = index >= 0 && index < chunkDownbeat.Length ? chunkDownbeat[index] : -1000;

                if (frame < activation.Count)
                {
                    activation[frame] = value;
                    downbeat[frame] = bar;
                }
                else
                {
                    activation.Add(value);
                    downbeat.Add(bar);
                }
            }
        }
    }
}
