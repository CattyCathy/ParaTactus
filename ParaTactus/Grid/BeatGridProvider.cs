using System;
using System.IO;
using System.Runtime.ExceptionServices;
using System.Threading;
using ParaTactus.Audio;
using ParaTactus.Detector;
using ParaTactus.Tracking;

namespace ParaTactus.Grid
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

        /// <summary>
        /// Whether to choose the beats by searching for a steady tempo rather than by picking peaks.
        /// </summary>
        /// <remarks>
        /// Off by default, out of honesty about which is better rather than about which is newer. Measured against the
        /// beatmap's own grid on the reference track, the search covers more of the map's beats - 47% of them in a
        /// passage accelerating from 150 to 400 BPM against the peak picking's 25%, and 87% in the last passage against
        /// 56% - and finds a lower metrical level in the fastest passages, which is a judgement about what a pulse
        /// should look like that a measurement cannot make for itself.
        /// </remarks>
        public bool SearchForBeats { get; set; }

        /// <summary>
        /// A path to a trained <see cref="LearnedBeatDetector"/> model, to read the beats with that instead.
        /// </summary>
        /// <remarks>
        /// Empty by default, and empty means the beats come from Beat This! and the post-processing as they always have.
        /// The detector is a different instrument rather than a better one for every case: it was trained on this
        /// project's own corpus of beatmaps and reads a log-mel spectrogram directly, and it reports the beat period
        /// alongside the beats, which is what lets it hold a metrical level.
        ///
        /// Measured on twelve tracks against their maps' timing points, it places beats a median 12ms from the map's
        /// own against 101ms for the peak picking over the activation, and lands on the map's metrical level on seven of
        /// the twelve. It is also slower and needs a second model file, which is why switching it on is a decision and
        /// not a default.
        /// </remarks>
        public string DetectorPath { get; set; }

        /// <summary>Where analysed grids are kept.</summary>
        public BeatGridCache Cache { get; }

        /// <summary>
        /// Which reading of the audio this provider is set to use, as a name for the cache key.
        /// </summary>
        /// <remarks>
        /// Every setting that changes what the beats are has to appear here, or two settings share one cached answer
        /// and the second one to be tried is served the first one's. The detector's path is in it as well as its name,
        /// because two different detectors are two different readings of the same track.
        /// </remarks>
        private string reading => !string.IsNullOrEmpty(DetectorPath)
            ? "detector:" + DetectorPath
            : SearchForBeats ? "search" : "peaks";

        /// <summary>
        /// Whether a track's beats are already known, without analysing anything.
        /// </summary>
        public bool IsCached(string audioPath)
        {
            return Cache.TryLoad(BeatGridCache.KeyFor(audioPath, modelPath, reading), out _);
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
        /// <see cref="BeatThisBeatTracker.OpenSession(string)"/> does without starving the inference it is protecting
        /// playback from.
        /// </remarks>
        public BeatGrid Get(string audioPath, CancellationToken cancellation = default)
        {
            if (string.IsNullOrEmpty(audioPath))
                throw new ArgumentException("An audio path is needed.", nameof(audioPath));

            if (!File.Exists(audioPath))
                throw new FileNotFoundException($"No track at {audioPath}.", audioPath);

            string key = BeatGridCache.KeyFor(audioPath, modelPath, reading);

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

                    float[] samples = decoder.DecodeMono(audioPath, AnalysisAudio.SampleRate);

                    if (!string.IsNullOrEmpty(DetectorPath))
                    {
                        // The detector reads the audio itself rather than anything Beat This! produced, so the tracker
                        // is not run at all in this branch: two models over the same track would be twice the cost for
                        // one answer, and the answer would be the detector's either way.
                        //
                        // FromNormalisedBeats rather than FromBeats, and that is the point of the detector. FromBeats
                        // decides a metrical level for the whole track and moves every beat onto it; the detector has
                        // already decided its level from the audio with a period head trained for it, and re-deciding
                        // here would undo the thing it was built for.
                        using var detector = new LearnedBeatDetector(DetectorPath);

                        grid = BeatGrid.FromNormalisedBeats(detector.Beats(samples), 0);
                    }
                    else
                    {
                        tracker.Add(samples);
                        tracker.Flush();

                        // The tracked beats are put onto a locally regular spacing before they become a grid. The model's
                        // tempo is right and its individual beats are not: on one track's 200 BPM section the gaps it
                        // reports run from 200ms to 620ms around a 300ms beat, firing on subdivisions in places and
                        // missing beats in others. A display driven straight from that pulses unevenly through music a
                        // listener hears as steady, which is the one thing the visuals must not do.
                        //
                        // The search is an alternative to that, not a second stage after it: it chooses the beats from the
                        // activation with the tempo in its state, and re-spacing what it chose would throw away the
                        // steadiness it just paid for.
                        grid = SearchForBeats
                            ? BeatGrid.FromBeats(tracker.SearchedBeats)
                            : BeatGrid.FromBeats(BeatTrainRegulariser.Regularise(tracker.Beats));
                    }
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
