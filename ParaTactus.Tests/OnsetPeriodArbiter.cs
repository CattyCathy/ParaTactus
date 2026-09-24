using System;
using System.Collections.Generic;
using ParaTactus;

namespace ParaTactus.Tests
{
    /// <summary>
    /// An experiment that was measured and did not work, kept because what it rules out is not obvious and is worth
    /// not rediscovering.
    /// </summary>
    /// <remarks>
    /// The idea is to use the onset envelope as a second opinion on where the beats are, because the model can be
    /// confidently wrong and nothing in its own output reveals it. It has two halves, and one is dead while the other
    /// is safe and nearly useless.
    ///
    /// The phase is the dead half. Scanning a 300ms comb over the envelope's own best phase gives a different answer
    /// every second - over Designant's 30-44s it runs -114, +96, +26, -114, +126, +26, -114, +141, +26, -74, +86, +1,
    /// -94, +106, +26 - even where the envelope's periodicity at 300ms is strong (a normalised autocorrelation of 0.59
    /// to 0.73). The model's own activation scanned the same way is no better (+96, +141, +41, -59, +126, +6, -94,
    /// +126, -114, ...). In music dense enough to have an onset near every sixteenth, a comb at any phase finds onsets,
    /// so the argmax is the noise in the comparison rather than a fact about the track. An earlier attempt to re-place
    /// a passage from the audio alone moved a whole grid onto the off-beats by a measured 152ms, and this is why.
    ///
    /// So the phase comes from the model - the beats keep the phase most of them already agree on and only the ones
    /// that disagree are moved, by at most a third of a period - and the envelope is asked only whether the sound
    /// really repeats at the track's dominant period, a question it can answer yes or no to and which it never answers
    /// with a level of its own. In that form it is safe. On Designant it moves 43 of 420 beats by an average of 12ms,
    /// costs one beat at 54s that the filler then treats as an insertion, and does something real in 48-56s, where the
    /// model's beats sit on the beatmap's grid to within about 20ms: with the anchors pulled exactly onto it, the gaps
    /// the filler divides are exact multiples of 300ms and the passage comes out at 300/300/300 instead of
    /// 290/300/300/310.
    ///
    /// What it does not do is reach the two passages that are audibly wrong, and that is the reason it is not wired
    /// into the player. At 36-44s the model reports sixteen beats of which nine are on the half beat, so there is no
    /// majority phase to snap to and the audio cannot supply one either. At 57-63s there are only ten beats over seven
    /// seconds and never a long enough run to act on; loosening the run and the support far enough to try makes things
    /// worse rather than better - at 0.5 support and four beats 57-58s is still left at 247/247/300/353 and 69s has its
    /// density halved to 600ms, and the window's mean deviation from 300ms rises from 38.0 to 44.9ms.
    ///
    /// One implementation note, because it invalidated a round of measurements and looks like a working gate: the
    /// autocorrelation window is a second either way. The first version correlated over sixteen <em>frames</em> - 320ms
    /// - so at a lag of 300ms it had almost no overlap and produced noise, which the strict gate never cleared and the
    /// loose one acted on, halving a passage that had been correct.
    /// </remarks>
    public static class OnsetPeriodArbiter
    {
        /// <summary>How well the envelope has to repeat at a period before it is believed.</summary>
        private const double default_confidence = 0.05;

        /// <summary>What fraction of a passage's beats has to agree on a phase before it is used.</summary>
        private const double default_support = 0.55;

        /// <summary>How many beats in a row have to be trusted before a passage is aligned at all.</summary>
        private const int default_run = 6;

        /// <summary>How far the envelope's period may be from the model's before it is not about the same level.</summary>
        private const double level_low = 0.75;
        private const double level_high = 1.35;

        /// <summary>How far a beat may be moved, as a fraction of the period.</summary>
        private const double maximum_move = 0.35;

        /// <summary>How many frames either side the envelope is correlated over.</summary>
        /// <remarks>
        /// A second each way. The window has to be long enough to hold several periods of the longest lag asked about,
        /// and the first version of this got that wrong in a way worth recording: the window was sixteen frames, which
        /// is 320ms, so at a lag of 300ms the autocorrelation had sixteen overlapping samples and almost none of the
        /// signal - the confidence it produced was noise, the strict gate never fired at all, and the loose one fired
        /// on the noise and halved a passage that had been correct.
        /// </remarks>
        private const int autocorrelation_half = 50;

        /// <summary>How many phase bins a period is divided into when the beats' own phase is found.</summary>
        private const int bins = 12;

        private const int minimum_lag = 10;
        private const int maximum_lag = 50;

        /// <summary>The beats with the misplaced ones moved onto the phase the rest of the passage is on.</summary>
        public static double[] Snap(
            IReadOnlyList<double> beats,
            double[] envelope,
            int frames,
            double minimumConfidence = default_confidence,
            double minimumSupport = default_support,
            int minimumRun = default_run)
        {
            var result = new double[beats.Count];

            for (int i = 0; i < beats.Count; i++)
                result[i] = beats[i];

            if (beats.Count < minimumRun || envelope == null || frames < 200)
                return result;

            // The level the envelope is asked about is the track's dominant period and not the passage's own, which is
            // the whole difference between this working and halving a correct passage. A passage where the model has
            // lost half the tactus has a local period of twice the truth, so asking the envelope to confirm the local
            // period confirms the mistake - and then the beats are held on the doubled grid and the filler that would
            // have put the missing ones back has nothing left to fill. On Designant that cost 51-55s, which had been
            // right, and halved its density to 600ms.
            double dominant = BeatTrainRegulariser.Dominant(beats);

            if (dominant <= 0)
                return result;

            var period = new double[beats.Count];
            var trusted = new bool[beats.Count];

            for (int i = 0; i < beats.Count; i++)
            {
                var (candidate, confidence) = periodAt(envelope, frames, beats[i], dominant);

                period[i] = candidate;
                trusted[i] = candidate > 0 && confidence >= minimumConfidence
                             && candidate / dominant >= level_low && candidate / dominant <= level_high;
            }

            int index = 0;

            while (index < beats.Count)
            {
                if (!trusted[index])
                {
                    index++;
                    continue;
                }

                int start = index;

                while (index < beats.Count && trusted[index])
                    index++;

                if (index - start >= minimumRun)
                    align(result, beats, period, start, index, minimumSupport);
            }

            return result;
        }

        /// <summary>
        /// Moves the beats of one passage onto the phase most of them already share.
        /// </summary>
        private static void align(double[] result, IReadOnlyList<double> beats, double[] period, int start, int end, double minimumSupport)
        {
            double q = median(period, start, end);

            if (q <= 0)
                return;

            var residues = new double[end - start];

            for (int i = start; i < end; i++)
                residues[i - start] = wrap(beats[i], q);

            double width = q / bins;
            var count = new int[bins];

            for (int i = 0; i < residues.Length; i++)
                count[bin(residues[i], q, width)]++;

            int modal = 0;

            for (int b = 1; b < bins; b++)
            {
                if (count[b] > count[modal])
                    modal = b;
            }

            if (count[modal] < minimumSupport * residues.Length)
                return;

            // Refined to the mean of the beats in and beside the winning bin rather than left on the bin's centre, so
            // the phase is not quantised to a twelfth of a period when the beats agree far more closely than that.
            double centre = -q / 2 + (modal + 0.5) * width;
            double sum = 0;
            int members = 0;

            foreach (double residue in residues)
            {
                double delta = wrap(residue - centre, q);

                if (Math.Abs(delta) > width)
                    continue;

                sum += delta;
                members++;
            }

            double phase = members > 0 ? wrap(centre + sum / members, q) : centre;

            for (int i = start; i < end; i++)
            {
                double delta = wrap(beats[i] - phase, q);

                if (Math.Abs(delta) <= maximum_move * q)
                    result[i] = beats[i] - delta;
            }
        }

        /// <summary>
        /// The period the envelope repeats at near a time, and how well it does.
        /// </summary>
        /// <remarks>
        /// Among the lags the autocorrelation actually peaks at, the one nearest the model's own period is taken
        /// rather than the highest. The highest is a harmonic most of the time - the envelope of a 300ms passage
        /// correlates better at 600ms and 1200ms because a bar repeats more exactly than a beat - and choosing it
        /// would put the answer a factor of two or four away from the question being asked.
        /// </remarks>
        public static (double Period, double Confidence) PeriodAt(double[] envelope, int frames, double time, double local)
            => periodAt(envelope, frames, time, local);

        private static (double Period, double Confidence) periodAt(double[] envelope, int frames, double time, double local)
        {
            int centre = (int)Math.Round(time / 20.0);
            double best = 0;
            double chosen = 0;
            double chosenValue = 0;

            var candidates = new List<(double Period, double Value)>();

            for (int lag = minimum_lag; lag <= maximum_lag; lag++)
            {
                double value = autocorrelation(envelope, centre, lag);

                if (value <= 0)
                    continue;

                if (value < autocorrelation(envelope, centre, lag - 1) || value < autocorrelation(envelope, centre, lag + 1))
                    continue;

                candidates.Add((lag * 20.0, value));
                best = Math.Max(best, value);
            }

            if (candidates.Count == 0 || best <= 0)
                return (0, 0);

            foreach (var candidate in candidates)
            {
                if (candidate.Value < best * 0.75)
                    continue;

                if (chosen == 0 || (local > 0 && Math.Abs(candidate.Period - local) < Math.Abs(chosen - local)))
                {
                    chosen = candidate.Period;
                    chosenValue = candidate.Value;
                }
            }

            return chosen > 0 ? (chosen, chosenValue) : (0, 0);
        }

        /// <summary>The normalised autocorrelation of the envelope at a lag, over a window around a frame.</summary>
        private static double autocorrelation(double[] envelope, int centre, int lag)
        {
            int start = Math.Max(centre - autocorrelation_half, 0);
            int end = Math.Min(centre + autocorrelation_half, envelope.Length - 1 - lag);

            if (end - start < 4 * lag)
                return 0;

            double mean = 0;

            for (int t = start; t < end; t++)
                mean += envelope[t];

            mean /= end - start;

            double covariance = 0;
            double variance = 0;

            for (int t = start; t < end; t++)
            {
                covariance += (envelope[t] - mean) * (envelope[t + lag] - mean);
                variance += (envelope[t] - mean) * (envelope[t] - mean);
            }

            return variance > 0 ? covariance / variance : 0;
        }

        private static double median(double[] values, int start, int end)
        {
            var copy = new double[end - start];

            for (int i = start; i < end; i++)
                copy[i - start] = values[i];

            Array.Sort(copy);

            return copy[copy.Length / 2];
        }

        /// <summary>A time or a residue as a signed offset within half a period either side of zero.</summary>
        private static double wrap(double value, double period) => (value % period + 1.5 * period) % period - 0.5 * period;

        private static int bin(double residue, double period, double width)
        {
            int index = (int)Math.Floor((residue + period / 2) / width);

            return Math.Clamp(index, 0, bins - 1);
        }
    }
}
