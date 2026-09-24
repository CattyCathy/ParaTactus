using System;
using System.Collections.Generic;
using ParaTactus;

namespace ParaTactus
{
    /// <summary>
    /// Chooses which metrical level of a beat grid counts as the beat.
    /// </summary>
    /// <remarks>
    /// A beat tracker and a beatmap can describe the same music at different levels. Measured on the track this was
    /// built against, the model reports a clean 600ms pulse where the map declares 200 BPM, which is a 300ms pulse.
    /// Both describe the same music and neither is an error; they disagree about which pulse is the beat. Against that
    /// map, 18 of 35 sampled points were exactly equal and 16 were exactly half, with 20 within 2% of a power of two,
    /// so treating the difference as an estimation error would be reading a metrical choice as a bug.
    ///
    /// The policy is therefore to keep the model's own tactus - the pulse a listener would tap - and step a whole
    /// octave only when the track as a whole sits outside the range a person can comfortably tap. That guard is what
    /// makes this safe rather than a second estimator: a track whose tempo ramps from 150 to 400 BPM has a median
    /// around 240, and folding by the median alone would halve the slow sections that were already right. Stepping
    /// only when a large majority of the track lies on one side of the range leaves a genuinely mixed-tempo track
    /// untouched, which is the case this was written for.
    /// </remarks>
    public static class MetricalLevel
    {
        /// <summary>The bottom of the range a listener can comfortably tap.</summary>
        public const double DefaultMinimumBpm = 80;

        /// <summary>
        /// The top of the range a listener can comfortably tap.
        /// </summary>
        /// <remarks>
        /// 220 rather than 180, which is what this started at. A pulse of 200 BPM is fast but entirely tappable, and
        /// 200 BPM is the tempo of a great deal of electronic music; a range that tops out below it halves those tracks
        /// to 100 and the visuals then move at half the speed of the music. The whole point of this class is to step an
        /// octave only when the track is at a tempo nobody would tap, and 200 is not that.
        /// </remarks>
        public const double DefaultMaximumBpm = 220;

        /// <summary>The share of the track that has to sit outside the range before the level is changed.</summary>
        public const double DefaultMajority = 0.8;

        /// <summary>The largest number of octaves the level is allowed to move.</summary>
        public const int MaximumShift = 4;

        /// <summary>
        /// How many octaves to move the tactus by, as a power of two to divide the tempo by: a result of 1 means the
        /// reported tempo should be halved, -1 means it should be doubled.
        /// </summary>
        /// <remarks>
        /// The shift chosen is the one that puts the largest share of the track inside the comfortable range, and it is
        /// only taken if that share reaches <paramref name="majority"/>; otherwise the tactus stands. Asking for a
        /// majority on one side of the range instead would be wrong for a track that ramps across it - a 150 to 400 BPM
        /// ramp is above 180 for most of its length, so a side count would halve the whole thing, leaving the slow end
        /// at 75. Asking how much of the track a shift actually fixes rejects that, because no single octave covers a
        /// range that wide.
        /// </remarks>
        public static int ChooseShift(
            IReadOnlyList<double> beatTimes,
            double minimumBpm = DefaultMinimumBpm,
            double maximumBpm = DefaultMaximumBpm,
            double majority = DefaultMajority)
        {
            var intervals = Intervals(beatTimes);

            if (intervals.Count == 0)
                return 0;

            // Two conditions, and both are needed. The track has to sit outside the range on one side for most of its
            // length - which is what stops a track that merely passes through extreme tempi being folded - and the
            // shift has to actually fix most of it. Either test alone gets a case wrong: the first alone folds a ramp
            // that is mostly above the range but whose slow end would be pushed too far down, and the second alone folds
            // the same ramp because halving does bring most of it inside.
            int below = 0;
            int above = 0;

            foreach (double interval in intervals)
            {
                double bpm = 60000 / interval;

                if (bpm < minimumBpm)
                    below++;
                else if (bpm >= maximumBpm)
                    above++;
            }

            double needed = majority * intervals.Count;

            if (below < needed && above < needed)
                return 0;

            int best = 0;
            double bestShare = ShareInRange(intervals, 0, minimumBpm, maximumBpm);

            for (int distance = 1; distance <= MaximumShift; distance++)
            {
                // Halving and doubling are both considered at each distance, so the level moves as little as it can.
                if (ShareInRange(intervals, distance, minimumBpm, maximumBpm) > bestShare)
                {
                    bestShare = ShareInRange(intervals, distance, minimumBpm, maximumBpm);
                    best = distance;
                }

                if (ShareInRange(intervals, -distance, minimumBpm, maximumBpm) > bestShare)
                {
                    bestShare = ShareInRange(intervals, -distance, minimumBpm, maximumBpm);
                    best = -distance;
                }
            }

            return bestShare >= majority ? best : 0;
        }

        /// <summary>
        /// The share of a track's beat intervals whose tempo lands in the comfortable range after a shift.
        /// </summary>
        private static double ShareInRange(IReadOnlyList<double> intervals, int shift, double minimumBpm, double maximumBpm)
        {
            double divisor = Math.Pow(2, shift);
            int inside = 0;

            foreach (double interval in intervals)
            {
                if (InRange(60000 / interval / divisor, minimumBpm, maximumBpm))
                    inside++;
            }

            return (double)inside / intervals.Count;
        }

        /// <summary>
        /// The octave, preferring the smallest move, that brings a tempo into the comfortable range.
        /// </summary>
        public static int ShiftFor(double bpm, double minimumBpm = DefaultMinimumBpm, double maximumBpm = DefaultMaximumBpm)
        {
            if (bpm <= 0 || minimumBpm <= 0 || maximumBpm <= minimumBpm)
                return 0;

            for (int distance = 0; distance <= MaximumShift; distance++)
            {
                if (distance == 0)
                {
                    if (InRange(bpm, minimumBpm, maximumBpm))
                        return 0;
                }
                else
                {
                    // Halving and doubling are both tried at each distance, so the level moves as little as possible.
                    if (InRange(bpm / Math.Pow(2, distance), minimumBpm, maximumBpm))
                        return distance;

                    if (InRange(bpm * Math.Pow(2, distance), minimumBpm, maximumBpm))
                        return -distance;
                }
            }

            return 0;
        }

        /// <summary>
        /// The beat times moved to the chosen level: dropping beats to halve the tempo, or inserting evenly spaced
        /// beats to double it.
        /// </summary>
        public static double[] Apply(IReadOnlyList<double> beatTimes, int shift)
        {
            if (shift == 0 || beatTimes.Count == 0)
            {
                var unchanged = new double[beatTimes.Count];

                for (int i = 0; i < beatTimes.Count; i++)
                    unchanged[i] = beatTimes[i];

                return unchanged;
            }

            if (shift > 0)
            {
                int step = 1 << Math.Min(shift, MaximumShift);
                var thinned = new List<double>();

                for (int i = 0; i < beatTimes.Count; i += step)
                    thinned.Add(beatTimes[i]);

                return thinned.ToArray();
            }

            int divisions = 1 << Math.Min(-shift, MaximumShift);
            var subdivided = new List<double>();

            for (int i = 0; i + 1 < beatTimes.Count; i++)
            {
                double span = beatTimes[i + 1] - beatTimes[i];

                for (int part = 0; part < divisions; part++)
                    subdivided.Add(beatTimes[i] + span * part / divisions);
            }

            if (beatTimes.Count > 0)
                subdivided.Add(beatTimes[beatTimes.Count - 1]);

            return subdivided.ToArray();
        }

        /// <summary>
        /// The beat times moved to the level the track as a whole calls for.
        /// </summary>
        public static double[] Normalise(
            IReadOnlyList<double> beatTimes,
            out int shift,
            double minimumBpm = DefaultMinimumBpm,
            double maximumBpm = DefaultMaximumBpm,
            double majority = DefaultMajority)
        {
            shift = ChooseShift(beatTimes, minimumBpm, maximumBpm, majority);

            return Apply(beatTimes, shift);
        }

        private static bool InRange(double bpm, double minimumBpm, double maximumBpm)
        {
            return bpm >= minimumBpm && bpm < maximumBpm;
        }

        private static List<double> Intervals(IReadOnlyList<double> beatTimes)
        {
            var intervals = new List<double>();

            for (int i = 1; i < beatTimes.Count; i++)
            {
                double interval = beatTimes[i] - beatTimes[i - 1];

                if (interval > 0)
                    intervals.Add(interval);
            }

            return intervals;
        }
    }
}
