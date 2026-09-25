using System;
using System.IO;
using System.Runtime.ExceptionServices;
using System.Threading;
using ParaTactus;

namespace ParaTactus
{
    /// <summary>
    /// The way a player gets a track's beats: from the cache when it has them, otherwise from the model.
    /// </summary>
    /// <remarks>
    /// Analysis costs about a quarter of the track's own length, so this is not something to call on a frame. It exists
    /// to give the rest of the player one call that is usually instant and never wrong, and to keep the decision of
    /// what "already analysed" means in one place.
    ///
    /// For a track being played rather than looked up, <see cref="BeginStreaming"/> is the other half: it hands back a
    /// tracker that can be fed audio as it plays and stays ahead of the playhead, which is what makes the cost
    /// acceptable for audio that is not in the cache yet.
    /// </remarks>
    public sealed class BeatGridProvider
    {
        private readonly string modelPath;
        private readonly IAudioDecoder decoder;

        /// <summary>
        /// A provider that can analyse a track from its path, given something that can decode it.
        /// </summary>
        /// <remarks>
        /// The decoder is optional because two of the three things this class does never need one: a grid already in
        /// the cache is returned without touching the audio, and the streaming tracker is fed samples by its caller.
        /// Only a cache miss on a path needs a decode, and only then is its absence an error.
        /// </remarks>
        public BeatGridProvider(string modelPath, string cacheDirectory, IAudioDecoder decoder = null)
            : this(modelPath, new BeatGridCache(cacheDirectory), decoder)
        {
        }

        public BeatGridProvider(string modelPath, BeatGridCache cache, IAudioDecoder decoder = null)
        {
            if (string.IsNullOrEmpty(modelPath))
                throw new ArgumentException("A model path is needed to analyse tracks.", nameof(modelPath));

            if (cache == null)
                throw new ArgumentNullException(nameof(cache));

            this.modelPath = modelPath;
            this.decoder = decoder;
            Cache = cache;
        }

        /// <summary>The model the analysis runs.</summary>
        public string ModelPath => modelPath;

        /// <summary>Where analysed grids are kept.</summary>
        public BeatGridCache Cache { get; }

        /// <summary>
        /// Whether a track's beats are already known, without analysing anything.
        /// </summary>
        public bool IsCached(string audioPath)
        {
            return Cache.TryLoad(BeatGridCache.KeyFor(audioPath, modelPath), out _);
        }

        /// <summary>
        /// A track's beats, analysed if they are not already known. This blocks for about a quarter of the track's
        /// length on a miss and returns immediately on a hit.
        /// </summary>
        /// <remarks>
        /// The analysis runs on its own thread, so the caller waits only on the result and not on any of the work, and
        /// that thread runs at normal priority. It used to be lowered to below normal so that the audio callback would
        /// always be ahead of it, and that does not work: with ONNX Runtime's thread pool created from a below-normal
        /// thread the inference never finishes. Measured on the reference track, the same sequence on a below-normal
        /// thread spun at full CPU for minutes where the identical code at normal priority returned in 19 seconds.
        /// What actually protects playback is keeping processors out of the model's own pool, two of them, which
        /// <see cref="BeatThisBeatTracker.OpenSession"/> does without starving the inference it is protecting playback
        /// from.
        /// </remarks>
        public BeatGrid Get(string audioPath, CancellationToken cancellation = default)
        {
            if (string.IsNullOrEmpty(audioPath))
                throw new ArgumentException("An audio path is needed.", nameof(audioPath));

            if (!File.Exists(audioPath))
                throw new FileNotFoundException($"No track at {audioPath}.", audioPath);

            string key = BeatGridCache.KeyFor(audioPath, modelPath);

            if (Cache.TryLoad(key, out BeatGrid cached))
                return cached;

            BeatGrid grid = null;
            ExceptionDispatchInfo failure = null;

            var thread = new Thread(() =>
            {
                try
                {
                    // Disposed here rather than relying on Flush, because Flush is what released the model and it is
                    // only reached when the analysis finishes. A cancelled analysis threw straight out of Add, so its
                    // session - tens of megabytes and a thread pool of its own - was never released. Choosing a song and
                    // then choosing another leaked one per click, and the player got slower the more it was used.
                    using var tracker = BeginStreaming(cancellation);

                    if (decoder == null)
                    {
                        throw new InvalidOperationException(
                            "Analysing a track from its path needs a decoder, and none was given. Reference the "
                            + "ParaTactus.Bass package and pass BassAudioDecoder.Default, or decode the audio yourself and "
                            + "feed it to a StreamingBeatTracker.");
                    }

                    tracker.Add(decoder.DecodeMono(audioPath, AnalysisAudio.SampleRate));
                    tracker.Flush();

                    // The tracked beats are put onto a locally regular spacing before they become a grid. The model's
                    // tempo is right and its individual beats are not: on one track's 200 BPM section the gaps it
                    // reports run from 200ms to 620ms around a 300ms beat, firing on subdivisions in places and
                    // missing beats in others. A display driven straight from that pulses unevenly through music a
                    // listener hears as steady, which is the one thing the visuals must not do.
                    grid = BeatGrid.FromBeats(BeatTrainRegulariser.Regularise(tracker.Beats));
                }
                catch (Exception error)
                {
                    failure = ExceptionDispatchInfo.Capture(error);
                }
            })
            {
                IsBackground = true,
                Name = "beat analysis",
            };

            thread.Start();
            thread.Join();

            // Rethrown on the calling thread so a failure to analyse is still a failure to analyse, not a silent empty
            // grid that would be indistinguishable from a track the model simply found no beats in.
            failure?.Throw();

            // A grid with no beats is a failed analysis, not an answer, and caching it would make the failure permanent.
            if (!grid.IsEmpty)
                Cache.Store(key, grid);

            return grid;
        }

        /// <summary>
        /// A tracker to feed a track's audio to as it plays, for tracks that are not in the cache yet.
        /// </summary>
        public StreamingBeatTracker BeginStreaming(CancellationToken cancellation = default)
        {
            return new StreamingBeatTracker(modelPath, cancellation);
        }
    }
}
