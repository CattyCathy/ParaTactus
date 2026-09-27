using System;
using System.Collections.Generic;
using ParaTactus;

namespace ParaTactus
{
    /// <summary>
    /// A track's beat times, at the metrical level the track calls for, with the tempo and phase a player needs to
    /// drive an animation from them.
    /// </summary>
    /// <remarks>
    /// This is deliberately thin. The hand-tuned analyser needed a tempo curve, a segmenter and a phase profile
    /// because it was fitting a grid to an onset envelope it could not trust; a beat tracker that reports beat times
    /// directly already carries the phase, and the tempo between two beats is the interval between them. Anything more
    /// here would be a second estimator disagreeing with the first.
    ///
    /// Tempo is read over a window rather than from one interval, because a single interval is quantised to the
    /// analysis frame - 20ms at the model's 50 frames per second, which is over 3% of a 600ms beat - and one dropped
    /// or added beat would otherwise show as a tempo jump.
    /// </remarks>
    public sealed class BeatGrid
    {
        /// <summary>
        /// How many beats either side of a point set the tempo reported there.
        /// </summary>
        /// <remarks>
        /// A count of beats rather than a span of time, which was measured against the alternative and came out better:
        /// a time window is the more principled thing on paper, because a count shrinks as the music gets faster and
        /// so averages over less exactly where there are the most beats to average, but on these tracks it made the
        /// second-to-second movement of the readout slightly worse on Designant (p90 6.7% against 5.6%) for no gain in
        /// the number of places the readout more than doubles. The reason is probably that the beats are quantised to
        /// 20ms, so a longer window is mostly averaging quantisation rather than noise.
        /// </remarks>
        public const int DefaultTempoWindow = 6;

        /// <summary>
        /// The shortest span of track a tempo is read over, whatever the beat count works out to.
        /// </summary>
        /// <remarks>
        /// A window counted in beats shrinks as the music gets faster, and the failure that causes is specific: where
        /// the tracker reports a short burst of beats at two or three times the surrounding rate, the window fits
        /// entirely inside the burst and the burst is reported as the tempo. On Stage 5 a passage reading 500 BPM for
        /// six seconds had a beat every 120ms, so six beats either side covered 1.2 seconds of it and nothing else.
        ///
        /// Two seconds is the shortest span over which a tempo means anything at these tempi - roughly one bar - and it
        /// is small enough to leave ordinary readings alone. At 200 BPM six beats already spans 1.8 seconds, so this
        /// changes almost nothing there; at 500 it is the difference between reading the burst and reading the music.
        /// </remarks>
        public const double MinimumTempoWindowMs = 2000;

        /// <summary>How many times the local window the fallback window spans.</summary>
        private const int wide_factor = 8;

        private readonly double[] beats;

        private BeatGrid(double[] beats, int metricalShift)
        {
            this.beats = beats;
            MetricalShift = metricalShift;
        }

        /// <summary>The beats, in milliseconds.</summary>
        public IReadOnlyList<double> Beats => beats;

        /// <summary>
        /// The same beats with every passage of one tempo laid out as an exactly even pulse, or null when the track has
        /// no passage long enough to lay out.
        /// </summary>
        /// <remarks>
        /// Computed on demand and kept beside <see cref="Beats"/> rather than replacing it, because the two answer
        /// different questions and only one of them can be right. The tracked beats say where the music is - measured on
        /// this corpus they land a middle three and a half milliseconds from a map's own beats - and their spacing is
        /// uneven, because the tracker reports an extra beat every twenty or thirty and loses one every fifty. An even
        /// pulse is what a person watching expects to see, and it is measurably further from the map's beats: laid over
        /// these it moves the middle distance from 4.8ms to between 56 and 88 and takes the coverage from 93% to between
        /// 32% and 59%.
        ///
        /// So this is for looking at rather than for counting on. A caller that draws beats can draw these; a caller
        /// that needs to know where the music is has to use the tracked ones.
        ///
        /// The period of a passage is the middle of its own gaps rather than a line fitted through them, because a line
        /// is dragged by exactly the extra beats being corrected - on one track a passage whose gaps have a middle of
        /// 409.8ms fits a line of 386.5. The phase is the passage's own first beat, so an error in the period can only
        /// accumulate within one passage and is bounded by its length rather than by the track's.
        /// </remarks>
        public IReadOnlyList<double> EvenBeats
        {
            get
            {
                if (even == null)
                    even = evenTempo();

                return even;
            }
        }

        private double[] even;

        /// <summary>
        /// How many beats a track needs before a passage can be laid out at all.
        /// </summary>
        /// <remarks>
        /// Eight, so that a passage has a few gaps to be measured from and the result is not one gap's width presented
        /// as a tempo. Under this the tracked beats are drawn as they are.
        /// </remarks>
        private const int even_minimum_beats = 8;

        private double[] evenTempo()
        {
            // A passage is measured from the gaps inside it, and one of eight beats is a gap or two either side - not
            // enough to say what its tempo is, and enough to say it wrongly. Below that the tracked beats are all there
            // is to draw, which is why this reports nothing rather than something even and unjustified.
            if (beats.Length < even_minimum_beats)
                return null;

            TempoSection[] sections = TempoSections.Analyse(beats);

            if (sections.Length == 0)
                return null;

            var out_ = new List<double>(beats.Length);
            double last = double.NegativeInfinity;

            foreach (TempoSection section in sections)
            {
                if (section.Period <= 0)
                    continue;

                // The passage's own first beat, which is the phase it is laid out from.
                double anchor = double.NaN;

                for (int i = 0; i < beats.Length; i++)
                {
                    if (beats[i] >= section.Start - 0.5)
                    {
                        anchor = beats[i];
                        break;
                    }
                }

                if (double.IsNaN(anchor))
                    continue;

                double at = anchor;

                while (at < section.End)
                {
                    // Half a period is the guard against two passages both claiming the beat they share, and the same
                    // either side of the track's first beat because the opening passage begins at it.
                    if (at > last + (section.Period * 0.5) && at >= beats[0] - (section.Period * 0.5))
                    {
                        out_.Add(at);
                        last = at;
                    }

                    at += section.Period;
                }
            }

            return out_.Count >= 4 ? out_.ToArray() : null;
        }

        /// <summary>
        /// The same beats laid out evenly, as a grid, or this grid when there is no passage to lay out.
        /// </summary>
        public BeatGrid WithEvenTempo()
        {
            IReadOnlyList<double> laid = EvenBeats;

            return laid == null ? this : new BeatGrid(copyOf(laid), MetricalShift);
        }

        private static double[] copyOf(IReadOnlyList<double> values)
        {
            var copy = new double[values.Count];

            for (int i = 0; i < copy.Length; i++)
                copy[i] = values[i];

            return copy;
        }

        /// <summary>
        /// How many octaves the beats were moved from what the tracker reported: positive means beats were dropped to
        /// halve the tempo, negative means beats were inserted to double it, zero means the tracker's own tactus stood.
        /// </summary>
        public int MetricalShift { get; }

        /// <summary>Whether any beat was found at all.</summary>
        public bool IsEmpty => beats.Length == 0;

        /// <summary>The time of the last beat, in milliseconds.</summary>
        public double Duration => beats.Length > 0 ? beats[beats.Length - 1] : 0;

        /// <summary>
        /// Builds a grid from tracked beats, moved to the metrical level the track as a whole calls for.
        /// </summary>
        public static BeatGrid FromBeats(
            IReadOnlyList<double> beatTimes,
            double minimumBpm = MetricalLevel.DefaultMinimumBpm,
            double maximumBpm = MetricalLevel.DefaultMaximumBpm,
            double majority = MetricalLevel.DefaultMajority)
        {
            double[] normalised = MetricalLevel.Normalise(beatTimes, out int shift, minimumBpm, maximumBpm, majority);

            return new BeatGrid(normalised, shift);
        }

        /// <summary>
        /// Builds a grid from beats that are already at the level they should be, as when reloading a cached grid.
        /// </summary>
        /// <remarks>
        /// Separate from <see cref="FromBeats"/> because the metrical level is a property of the track, not of the
        /// lookup: deciding it again on reload could land on a different octave from the one that was stored, and the
        /// grid would shift under the player between the first play and the second.
        /// </remarks>
        public static BeatGrid FromNormalisedBeats(IReadOnlyList<double> beatTimes, int metricalShift)
        {
            var copy = new double[beatTimes.Count];

            for (int i = 0; i < beatTimes.Count; i++)
                copy[i] = beatTimes[i];

            return new BeatGrid(copy, metricalShift);
        }

        /// <summary>
        /// How long the beat at an index lasts: the time to the next beat, or for the last beat the length of the one
        /// before it. Zero when there are not two beats to measure between.
        /// </summary>
        /// <remarks>
        /// The last beat reaches forward to a beat that does not exist, so it borrows the previous interval rather than
        /// reporting zero. A zero here would be clamped to the fastest beat the visuals allow and make the final beat
        /// of every track strobe.
        /// </remarks>
        public double BeatLengthAt(int index)
        {
            if (index < 0 || index >= beats.Length)
                return 0;

            if (index + 1 < beats.Length)
                return beats[index + 1] - beats[index];

            return beats.Length > 1 ? beats[beats.Length - 1] - beats[beats.Length - 2] : 0;
        }

        /// <summary>
        /// The same grid moved in time, for correcting output latency.
        /// </summary>
        /// <remarks>
        /// The metrical level is carried over rather than reconsidered: this is the same track's beats, moved, and
        /// re-deciding the level on a shifted grid would let the octave change when the user nudges the offset.
        /// </remarks>
        public BeatGrid WithOffset(double milliseconds)
        {
            var shifted = new double[beats.Length];

            for (int i = 0; i < beats.Length; i++)
                shifted[i] = beats[i] + milliseconds;

            return new BeatGrid(shifted, MetricalShift);
        }

        /// <summary>
        /// The tempo around a time, in BPM: the median interval across the beats either side of it, or zero when there
        /// are not two beats to measure between.
        /// </summary>
        public double BpmAt(double timeMs, int window = DefaultTempoWindow)
        {
            if (beats.Length < 2)
                return 0;

            // Before the first beat there is no interval to stand on, so the opening tempo is reported instead.
            int centre = Math.Max(0, IndexAt(timeMs));
            int from = Math.Max(0, centre - window);
            int to = Math.Min(beats.Length - 1, centre + window);

            // Widened, never narrowed, until it covers at least MinimumTempoWindowMs. This only ever matters where the
            // beats are close together, so a track at an ordinary tempo reads exactly as it did before.
            while (beats[to] - beats[from] < MinimumTempoWindowMs && (from > 0 || to < beats.Length - 1))
            {
                if (from > 0)
                    from--;

                if (to < beats.Length - 1)
                    to++;
            }

            var intervals = new List<double>();

            for (int i = from; i < to; i++)
                intervals.Add(beats[i + 1] - beats[i]);

            if (intervals.Count == 0)
                return 0;

            // Whether the intervals are moving one way, which is what a tempo change in progress looks like. Read before
            // the sort below, because it is about their order in time and sorting throws that away.
            int rising = 0;
            int falling = 0;

            for (int i = 1; i < intervals.Count; i++)
            {
                if (intervals[i] > intervals[i - 1])
                    rising++;
                else if (intervals[i] < intervals[i - 1])
                    falling++;
            }

            bool changing = intervals.Count > 3
                            && Math.Max(rising, falling) * 10 >= (intervals.Count - 1) * 7;

            intervals.Sort();

            // The true median, not the upper of the two middle values. The beats are quantised to the model's frame -
            // 20ms, which is over 4% of a 450ms beat - so the intervals straddle the real period rather than sitting on
            // it, and taking the upper middle reports the beat consistently long: on the track this was measured
            // against, 460ms where the period is 451.5, which reads as 130.4 BPM for music that is at 132.9.
            int middle = intervals.Count / 2;

            double period = intervals.Count % 2 == 1
                ? intervals[middle]
                : 0.5 * (intervals[middle - 1] + intervals[middle]);

            if (period <= 0)
                return 0;

            // A reading is only trusted when the beats it was taken from actually agree on a tempo. Where they do not -
            // which is what a burst of inserted beats, a passage of missed ones, and a stretch where the tracker cannot
            // make up its mind between two pulses all look like - the median of a scattered window is not a tempo the
            // music plays, and answering from it reports noise as a tempo.
            //
            // This is what separates a real tempo change from a spurious burst, and duration cannot: an octave change
            // lasting five seconds and a burst lasting six are indistinguishable by length alone. A real section holds
            // one rate throughout, so its beats agree; a burst does not, so its beats do not.
            //
            // A change in progress is the third case and it has to be let through. Its intervals disagree too, so an
            // agreement test alone rejects it and falls through to the wide window below, which holds the tempo the
            // track had before the change for as long as the change lasts - the whole of a slow accelerando, which is
            // what made the reported tempo take so long to follow one. Order is what tells the two apart: noise
            // scatters in both directions, a change moves one way.
            int agreeing = 0;

            foreach (double interval in intervals)
            {
                if (interval >= period * 0.85 && interval <= period * 1.15)
                    agreeing++;
            }

            if (changing)
                return 60000 / period;

            if (agreeing * 3 >= intervals.Count * 2)
                return 60000 / period;

            // Fall back to a much wider window, which a short excursion cannot move but a real change does. This is the
            // cheap half of tempo continuity and deliberately only the cheap half: it rejects a reading that disagrees
            // with its surroundings, it does not supply the beats a passage is missing, and it does not touch the beat
            // train at all.
            var wide = new List<double>();
            int wideFrom = Math.Max(0, centre - window * wide_factor);
            int wideTo = Math.Min(beats.Length - 1, centre + window * wide_factor);

            for (int i = wideFrom; i < wideTo; i++)
                wide.Add(beats[i + 1] - beats[i]);

            if (wide.Count <= intervals.Count)
                return 60000 / period;

            wide.Sort();

            double widePeriod = wide[wide.Count / 2];

            return widePeriod > 0 ? 60000 / widePeriod : 60000 / period;
        }

        /// <summary>
        /// The index of the last beat at or before a time, or -1 before the first beat.
        /// </summary>
        public int IndexAt(double timeMs)
        {
            int low = 0;
            int high = beats.Length - 1;
            int found = -1;

            while (low <= high)
            {
                int middle = (low + high) / 2;

                if (beats[middle] <= timeMs)
                {
                    found = middle;
                    low = middle + 1;
                }
                else
                {
                    high = middle - 1;
                }
            }

            return found;
        }

        /// <summary>
        /// How far through the current beat a time is, from 0 at the beat to just under 1 at the next one, which is
        /// what an animation keyed to the beat advances on.
        /// </summary>
        public double PhaseAt(double timeMs)
        {
            int index = IndexAt(timeMs);

            if (index < 0 || index + 1 >= beats.Length)
                return 0;

            double span = beats[index + 1] - beats[index];

            return span > 0 ? (timeMs - beats[index]) / span : 0;
        }
    }
}
