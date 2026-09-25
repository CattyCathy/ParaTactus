using System;
using System.Collections.Generic;
using ParaTactus;

namespace ParaTactus
{
    /// <summary>
    /// Removes the beats the tracker reported on subdivisions and fills in the ones it missed, so the visuals pulse on
    /// the metre rather than on every detection.
    /// </summary>
    /// <remarks>
    /// What this does not do is re-space the beats it keeps: every one of them is placed exactly where the tracker
    /// reported it. A passage the tracker tracks unevenly therefore comes out uneven, and a passage it reports slightly
    /// late comes out slightly late. That is the measured shape of the reference track's steady 200 BPM section, where
    /// the gaps scatter over 275-375ms around a 300ms beat: all of them fall inside the band this walks with
    /// (<see cref="minimum_ratio"/> to <see cref="maximum_ratio"/>), so nothing there is corrected, and the grid sits a
    /// median 69ms from the beatmap's own grid with 18 of its 28 beats more than 60ms out.
    ///
    /// A second pass that measured each steady run against its own lattice and translated the run onto it was written
    /// and measured, and it made the track worse: overall median 29ms to 35.1ms, 200 to 223 beats more than 60ms out,
    /// and several ten-second windows regressing by more than the target window improved. The reason is structural
    /// rather than a matter of tuning - a lattice fitted to the tracker's own beats carries the same offset it would be
    /// correcting - so it is recorded here rather than tried again.
    ///
    /// The model reports a tempo that is right and individual beats that are not. On Designant's 200 BPM section the
    /// gaps it produces run from 200ms to 620ms around a 300ms beat - it fires on subdivisions in some places and
    /// misses beats in others - and a display driven straight from those pulses unevenly through music a listener hears
    /// as steady. Removing the short ones is not enough on its own, because the long ones are missing beats and nothing
    /// that only deletes can put them back.
    ///
    /// Missing beats are filled, and this is not invention. A beat is a position in a metre, not an audible event: a
    /// passage of a track can have no onset at all for several bars and still have a beat on every one of them. The
    /// model's activation being negative there says there is nothing to hear, not that there is nothing to count, and
    /// the visuals are supposed to follow the metre. Filling is therefore what keeps the animation on the beat through
    /// the quiet parts of a track instead of stuttering across them.
    ///
    /// What is not filled is a genuine break. Beyond a few periods a gap stops looking like a missed beat and starts
    /// looking like the music having stopped, so it is left alone rather than carpeted with pulses the track does not
    /// have.
    ///
    /// The period comes from a wide neighbourhood rather than from the two beats either side of a gap, because a
    /// measurement taken from the thing being corrected is not a measurement: a run of inserted beats is most of the
    /// sample in a narrow window, and the estimate then agrees with them.
    /// </remarks>
    public static class BeatTrainRegulariser
    {
        /// <summary>Above this multiple of the track's own period a passage is treated as half speed.</summary>
        /// <remarks>
        /// Not far above one, because the ambiguity being resolved is exactly a factor of two and the estimate of the
        /// track's own period is itself approximate. On Designant the dominant comes out at 340ms while the passage to
        /// be corrected sits at 600 - a factor of 1.76 - so a threshold of 1.7 put it on the boundary and it was
        /// corrected in some places and left alone in others, which is worse than either.
        /// </remarks>
        private const double halving_ratio = 1.5;

        /// <summary>Below this multiple of the track's own period a passage is treated as double speed.</summary>
        /// <remarks>
        /// Deliberately left strict while the other direction was loosened, and the asymmetry is not an accident. A
        /// passage 1.5 times faster than the rest of the track is a real tempo change - the test fixture has one, from
        /// 300ms to 200ms - and widening this too made the beats of that change be thrown away as insertions. The
        /// halved direction has no such case: a passage slower than the track by that little is far more likely to be
        /// the tracker losing the pulse, which is what that threshold is for.
        /// </remarks>
        private const double doubling_ratio = 0.59;

        /// <summary>How many beats either side set the period a gap is judged against.</summary>
        private const int period_window = 16;

        /// <summary>Below this fraction of the period a gap is an inserted beat.</summary>
        private const double minimum_ratio = 0.85;

        /// <summary>Above this fraction of the period a gap is missing beats.</summary>
        private const double maximum_ratio = 1.45;

        /// <summary>Above this fraction a gap is a break in the music and is left as it is.</summary>
        /// <remarks>
        /// Raised from 2.6, which was too cautious by a factor of about two and left audible holes. On Designant's 200
        /// BPM section the tracker drops a beat in three places - gaps of 880, 1060 and 1360ms against a beat of 300 -
        /// and every one of them was over the old limit, so the passage played with a hole where the music has a beat.
        /// Those are 2.9, 3.5 and 4.5 beats, which at this limit are filled.
        ///
        /// The gap this is here to protect is much larger: the same track has a genuine break of 2020ms, which is 6.7
        /// beats, and it stays protected. Filling is the right answer in between because a beat is a position in a
        /// metre, not an audible event: a few bars with nothing to hear still have a beat on every one of them, and the
        /// visuals are supposed to follow the metre.
        /// </remarks>
        private const double maximum_fill_ratio = 5.0;

        /// <summary>
        /// The beats moved onto a locally regular spacing, in milliseconds.
        /// </summary>
        public static double[] Regularise(IReadOnlyList<double> beats)
        {
            var result = new double[beats.Count];

            for (int i = 0; i < beats.Count; i++)
                result[i] = beats[i];

            if (beats.Count < 4)
                return result;

            var period = new double[beats.Count];

            for (int i = 0; i < beats.Count; i++)
                period[i] = periodAt(beats, i);

            double dominant = dominantPeriod(beats, period);

            var regularised = new List<double> { beats[0] };
            double anchor = beats[0];

            for (int i = 1; i < beats.Count; i++)
            {
                double p = period[i];

                if (p <= 0)
                {
                    regularised.Add(beats[i]);
                    anchor = beats[i];
                    continue;
                }

                // The target is the local period moved onto the level the track spends most of its time at. This is the
                // part that catches a halved tactus, and it cannot be done from the local period alone: the model's own
                // answer is what is being corrected, so a target taken from it agrees with the mistake.
                //
                // Compared as a factor of two against the dominant period rather than by bucketing the logarithm. The
                // bucketed version treated 300ms and 200ms as an octave apart because their logarithms fall either side
                // of an integer, so a real tempo change between them was "corrected" and half its beats were thrown
                // away.
                if (dominant > 0)
                {
                    // Repeated, not applied once. A passage can be more than one factor of two away from the track's
                    // level: Stage 5's 288s runs at a local period of 100ms against a dominant of 400ms, and shifting
                    // it a single octave lands on 200ms, which is still not the track's tempo and reads as 300 BPM
                    // against everything around it at 150. The loop stops as soon as the period is at the track's own
                    // level, and the guard is only there so a degenerate dominant cannot spin.
                    for (int shift = 0; shift < 8; shift++)
                    {
                        double factor = p / dominant;

                        if (factor >= halving_ratio)
                            p /= 2;
                        else if (factor <= doubling_ratio)
                            p *= 2;
                        else
                            break;
                    }
                }

                double gap = beats[i] - anchor;
                double ratio = gap / p;

                // Too close to be a beat of this metre, so it is a subdivision the tracker mistook for one.
                if (ratio < minimum_ratio)
                    continue;

                // Too far, but not so far that the music has stopped: the beats in between are put back at the spacing
                // the rest of the passage is using, spread evenly rather than bunched at one end.
                if (ratio > maximum_ratio && ratio <= maximum_fill_ratio)
                {
                    int missing = (int)Math.Round(ratio) - 1;

                    for (int k = 1; k <= missing; k++)
                        regularised.Add(anchor + k * (gap / (missing + 1)));
                }

                regularised.Add(beats[i]);
                anchor = beats[i];
            }

            return regularised.ToArray();
        }

        /// <summary>
        /// The beat period the track spends most of its time at, weighted by how long it spends there.
        /// </summary>
        /// <remarks>
        /// Weighted by duration rather than by beat count, because a passage at double density has twice as many beats
        /// for the same amount of music and would otherwise outvote the rest of the track simply by being fast.
        ///
        /// A factor of two away from this is the ambiguity being resolved and nothing else is touched: a passage at the
        /// same level as the rest of the track keeps whatever tempo it has, including a real change short of an octave.
        /// </remarks>
        /// <summary>
        /// The period the track spends most of its time at, for inspection.
        /// </summary>
        /// <remarks>
        /// Exposed because the level unification is only as good as this number, and it did not fire on a passage that
        /// is plainly at twice the density of everything around it. Whether the statistic picked the wrong level or the
        /// ratio test is too strict cannot be told apart without seeing it, and guessing has been the expensive way to
        /// find out.
        /// </remarks>
        internal static double Dominant(IReadOnlyList<double> beats)
        {
            if (beats.Count < 4)
                return 0;

            var period = new double[beats.Count];

            for (int i = 0; i < beats.Count; i++)
                period[i] = periodAt(beats, i);

            return dominantPeriod(beats, period);
        }

        /// <summary>
        /// The local period the walk uses at a beat, for inspection.
        /// </summary>
        /// <remarks>
        /// The level unification is applied by comparing this against the track's dominant period, so a passage that
        /// should have been pulled back to the track's own level and was not can only be explained by one of the two:
        /// either this estimate already looks like the surrounding tempo, or it does not and the walk failed to act on
        /// it. Those need different fixes and there is no way to tell which from the output alone.
        /// </remarks>
        internal static double LocalPeriodAt(IReadOnlyList<double> beats, int index) => periodAt(beats, index);

        private static double dominantPeriod(IReadOnlyList<double> beats, double[] period)
        {
            double best = 0;
            double bestWeight = 0;

            for (int i = 0; i + 1 < beats.Count; i++)
            {
                double candidate = period[i];

                if (candidate <= 0)
                    continue;

                double weight = 0;

                for (int j = 0; j + 1 < beats.Count; j++)
                {
                    if (period[j] >= candidate * 0.75 && period[j] <= candidate * 1.25)
                        weight += beats[j + 1] - beats[j];
                }

                if (weight > bestWeight)
                {
                    bestWeight = weight;
                    best = candidate;
                }
            }

            return best;
        }
        private static int dominantOctave(IReadOnlyList<double> beats, double[] period, int[] octave)
        {
            var weight = new Dictionary<int, double>();

            for (int i = 0; i + 1 < beats.Count; i++)
            {
                if (octave[i] == int.MinValue)
                    continue;

                double span = beats[i + 1] - beats[i];

                weight[octave[i]] = weight.TryGetValue(octave[i], out double seen) ? seen + span : span;
            }

            int best = int.MinValue;
            double bestWeight = 0;

            foreach (var pair in weight)
            {
                if (pair.Value > bestWeight)
                {
                    bestWeight = pair.Value;
                    best = pair.Key;
                }
            }

            return best;
        }

        /// <summary>
        /// The beat period around a beat, as the dominant gap nearby rather than as the median of them.
        /// </summary>
        /// <remarks>
        /// The median is corrupted by exactly the thing being corrected: a run of inserted beats turns one gap into two
        /// shorter ones, so in a passage where a third of the gaps are wrong the median lands on a wrong one and the
        /// correction then treats the wrong spacing as the truth. The dominant gap stays on the period the passage is
        /// actually in.
        /// </remarks>
        private static double periodAt(IReadOnlyList<double> beats, int index)
        {
            int from = Math.Max(0, index - period_window);
            int to = Math.Min(beats.Count - 1, index + period_window);
            int count = to - from;

            if (count < 3)
                return 0;

            var gaps = new double[count];

            for (int i = 0; i < count; i++)
                gaps[i] = beats[from + i + 1] - beats[from + i];

            Array.Sort(gaps);

            double bestCentre = 0;
            int bestCount = 0;

            foreach (double candidate in gaps)
            {
                int agreeing = 0;

                foreach (double gap in gaps)
                {
                    if (gap >= candidate * 0.75 && gap <= candidate * 1.25)
                        agreeing++;
                }

                // Ties go to the shorter gap. This was the other way round and it halved the tempo: in a passage
                // mixing 300ms and 600ms gaps the longer cluster won, every real 300ms beat then fell below the
                // threshold and was thrown away as an insertion, and the train came out at half speed. The shorter
                // spacing is the one to trust because the other direction is recoverable - a missed beat is a gap of
                // two periods, and filling puts it back - whereas deleting real beats cannot be undone.
                if (agreeing > bestCount || (agreeing == bestCount && candidate < bestCentre))
                {
                    bestCount = agreeing;
                    bestCentre = candidate;
                }
            }

            return bestCentre;
        }
    }
}