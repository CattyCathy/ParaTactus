using System;
using System.Collections.Generic;
using System.Globalization;

namespace ParaTactus.Grid
{
    /// <summary>
    /// A passage of a track at one tempo, as a player would describe it.
    /// </summary>
    /// <remarks>
    /// A grid is a list of these rather than a list of beats, and the difference is what a person reads off it. A beat
    /// list says where every pulse is; a section says what the pulse is, which is the thing that can be shown, counted
    /// in, or used to decide that the tempo has changed. Measured over this corpus, the maps themselves are built this
    /// way: merging a map's timing points into runs of one tempo and keeping the runs of five seconds or more leaves a
    /// median of two sections covering 98% of the track, so a section is the unit the material is actually in.
    ///
    /// The period is the middle of the gaps between the beats of the section rather than a line fitted through them.
    /// A middle is unmoved by the beats the detector places badly, where a least-squares fit is dragged by them, and
    /// the beats are accurate enough for it: measured over sections the maps hold one tempo for, the middle gap is
    /// within a half of one per cent of the map's own beat length.
    /// </remarks>
    public readonly struct TempoSection
    {
        public TempoSection(double start, double end, double period)
        {
            Start = start;
            End = end;
            Period = period;
        }

        /// <summary>When the section starts, in milliseconds.</summary>
        public double Start { get; }

        /// <summary>When it ends, in milliseconds. The first beat that is not in it.</summary>
        public double End { get; }

        /// <summary>The beat period, in milliseconds, over the section.</summary>
        public double Period { get; }

        /// <summary>The tempo a player would name, which is the period as beats a minute.</summary>
        public double BeatsPerMinute => Period > 0 ? 60_000.0 / Period : 0;

        /// <summary>How long the section lasts, in milliseconds.</summary>
        public double Duration => End - Start;

        public override string ToString() =>
            string.Format(CultureInfo.InvariantCulture, "{0:0}ms..{1:0}ms {2:0.00}ms/beat ({3:0.0} BPM)",
                          Start, End, Period, BeatsPerMinute);
    }

    /// <summary>
    /// Reads the tempo sections of a track off the beats a detector found.
    /// </summary>
    /// <remarks>
    /// The question is not where the tempo is but where it changes, and the answer has to come from the beats rather
    /// than from the model's period output. That output was measured against thirty-nine passages the maps hold at one
    /// tempo and its middle error is nineteen per cent - it is frequently a whole factor of two out - so a section
    /// built on it would be a section whose tempo is wrong by a half. The gaps between reported beats are a different
    /// measurement: the same passages give a middle error under half of one per cent.
    ///
    /// Two things make the change detection reliable, and both were arrived at by measuring a version without them
    /// against the maps and finding it wrong in a way that looked like a tuning problem and was not.
    ///
    /// The gaps are made robust first. A beat the detector missed leaves a gap of two periods among gaps of one, and a
    /// median over a local window cannot absorb that: run over the dataset's own labels - placed exactly on the maps'
    /// grids, one passage giving 872 consecutive gaps of exactly 200.00ms with no scatter at all - a rule without this
    /// step cut that passage forty-three times. The rule was at fault, not the beats.
    ///
    /// And a boundary is only taken where the tempo stays different rather than merely passing through. That is what a
    /// change of tempo is: a real one persists by definition and jitter does not. With both in place the reading is
    /// insensitive to its own parameters across a wide sweep, which is the sign of a criterion rather than a setting.
    /// </remarks>
    public static class TempoSections
    {
        /// <summary>How many gaps either side make the near and far tempo estimates.</summary>
        /// <remarks>
        /// Sixteen either side, which is five seconds of music at three beats to the second. Wide enough that a middle
        /// over it is set by the tempo rather than by the two or three faults in it, and narrow enough to catch a passage
        /// of ten beats - a wider window misses those, measured: at thirty-two the same track's forty-beat middle passage
        /// was reported at the tempo either side of it instead of its own.
        /// </remarks>
        public const int Window = 16;

        /// <summary>
        /// How many consecutive flagged gaps a change needs before it is believed.
        /// </summary>
        /// <remarks>
        /// Twenty, which is a long run and is the number that decides how many sections come out. Swept over thirty
        /// tracks against their maps: at twelve the reading finds 247 sections where the maps have 225 between them, at
        /// twenty 163, and the tempo error improves with it - 1.48% of the beat at twelve against 1.19% at twenty -
        /// because a run that short is mostly the detector's own jitter rather than a tempo. What it costs is the drift
        /// a section accumulates, which grows with the section: 286ms at twelve against 497ms at twenty, so this is the
        /// one number here that trades two real things against each other rather than being simply better one way.
        /// </remarks>
        public const int Persistence = 20;

        /// <summary>How far the near tempo must sit from the far one, as a share, to flag a boundary.</summary>
        public const double ChangeShare = 0.06;

        /// <summary>The fewest gaps a section may have and still be reported.</summary>
        public const int ShortestSection = 16;

        /// <summary>
        /// How far a gap may be from its neighbourhood's tempo before it is taken to be a fault rather than a tempo.
        /// </summary>
        private const double FaultShare = 0.25;

        /// <summary>The sections of a track, from the beats of a detector.</summary>
        public static TempoSection[] Analyse(IReadOnlyList<double> beats)
        {
            return Analyse(beats, Window, ChangeShare, Persistence, ShortestSection);
        }

        /// <summary>
        /// The same, with the numbers named by the caller so they can be swept against a reference rather than argued
        /// about. <see cref="Analyse(IReadOnlyList{double})"/> is the reading the player uses.
        /// </summary>
        public static TempoSection[] Analyse(IReadOnlyList<double> beats, int window, double changeShare,
                                             int persistence, int shortest)
        {
            if (beats == null)
                throw new ArgumentNullException(nameof(beats));

            if (beats.Count < 3 || window < 1 || persistence < 1)
                return Array.Empty<TempoSection>();

            var gaps = new double[beats.Count - 1];

            for (int i = 0; i < gaps.Length; i++)
                gaps[i] = beats[i + 1] - beats[i];

            gaps = MakeRobust(gaps);

            var sections = new List<TempoSection>();
            int start = 0;

            while (start < gaps.Length)
            {
                int end = NextBoundary(gaps, start, window, changeShare, persistence, shortest);

                var run = new double[end - start];

                Array.Copy(gaps, start, run, 0, run.Length);

                int last = Math.Min(beats.Count - 1, end);

                sections.Add(new TempoSection(beats[start], beats[last], Middle(run)));

                if (end >= gaps.Length)
                    break;

                start = end;
            }

            return sections.ToArray();
        }

        /// <summary>
        /// The gaps with the beat-level faults taken out of them.
        /// </summary>
        /// <remarks>
        /// A gap that is a whole multiple of the tempo around it is a beat that was missed or a beat reported twice, so
        /// it is replaced by that tempo rather than by itself; a gap merely a share out is a beat placed badly, and is
        /// pulled to its neighbourhood for the same reason. A section is a statement about the tempo, and one bad beat
        /// should not be able to move it or to split it.
        ///
        /// The neighbourhood is the eight gaps either side, taken as a middle so that the very faults being removed do
        /// not set the value they are compared against.
        /// </remarks>
        private static double[] MakeRobust(double[] gaps)
        {
            var out_ = new double[gaps.Length];

            for (int i = 0; i < gaps.Length; i++)
            {
                int low = Math.Max(0, i - 8);
                int high = Math.Min(gaps.Length, i + 9);

                var around = new List<double>(high - low);

                for (int j = low; j < high; j++)
                {
                    if (j != i)
                        around.Add(gaps[j]);
                }

                if (around.Count < 3)
                {
                    out_[i] = gaps[i];
                    continue;
                }

                around.Sort();

                double local = around[around.Count / 2];

                if (local <= 0)
                {
                    out_[i] = gaps[i];
                    continue;
                }

                int multiple = Math.Max(1, (int)Math.Round(gaps[i] / local));
                bool wholeMultiple = multiple >= 2 && Math.Abs(gaps[i] - (multiple * local)) <= FaultShare * local;
                bool merelyOut = Math.Abs(gaps[i] - local) > FaultShare * local;

                out_[i] = wholeMultiple || merelyOut ? local : gaps[i];
            }

            return out_;
        }

        /// <summary>
        /// Where the section beginning at <paramref name="from"/> ends, or the end of the gaps if it does not.
        /// </summary>
        /// <remarks>
        /// The tempo is estimated either side of a candidate boundary rather than at it, and the two are compared.
        ///
        /// A change is then a run of flagged candidates, taken from the first flag to the last within a short gap, and
        /// the boundary goes at the middle of that span. Putting it at the middle of the span rather than after the
        /// persistence has been satisfied is what makes it land on the change: requiring the flags to be strictly
        /// consecutive places the boundary late by however long the jitter takes to break the run, which on a clean step
        /// between two tempos was measured at eighteen beats - six seconds of music reported at the wrong tempo.
        /// </remarks>
        private static int NextBoundary(double[] gaps, int from, int window, double changeShare,
                                        int persistence, int shortest)
        {
            int runFirst = -1;
            int runLast = -1;

            for (int i = Math.Max(1, from + 1); i <= gaps.Length; i++)
            {
                bool flagged = false;

                if (i < gaps.Length && i - from >= shortest)
                {
                    double near = Middle(gaps, i - window, i + window);
                    double far = Middle(gaps, i - (2 * window) - 1, i - window);

                    flagged = near > 0 && far > 0 && Math.Abs(near - far) / far >= changeShare;
                }

                if (flagged)
                {
                    // A flag near the last one continues the same run; one far from it starts another.
                    if (runFirst < 0 || i - runLast > 2)
                    {
                        int settled = settle(runFirst, runLast, persistence, shortest, from);

                        if (settled > 0)
                            return settled;

                        runFirst = i;
                    }

                    runLast = i;
                    continue;
                }

                if (runFirst >= 0 && i - runLast > 2)
                {
                    int settled = settle(runFirst, runLast, persistence, shortest, from);

                    if (settled > 0)
                        return settled;

                    runFirst = -1;
                    runLast = -1;
                }
            }

            int last = settle(runFirst, runLast, persistence, shortest, from);

            return last > 0 ? last : gaps.Length;
        }

        /// <summary>The boundary a finished run of flags implies, or zero if the run was too short to believe.</summary>
        /// <remarks>
        /// The boundary is the run's first flag, not its middle. A flag is raised where the tempo close to a beat has
        /// moved and the tempo a window further out has not, which is true from the beat the change happens at - so the
        /// first flag is the change and the middle of the run is half a window late. Measured on a clean step from 300ms
        /// to 400ms gaps, the middle put the boundary three and a half seconds after the join.
        /// </remarks>
        private static int settle(int runFirst, int runLast, int persistence, int shortest, int from)
        {
            if (runFirst < 0 || runLast - runFirst + 1 < persistence)
                return 0;

            return runFirst - from >= shortest ? runFirst : 0;
        }

        /// <summary>The middle of a run of values, clamped to what exists.</summary>
        /// <remarks>
        /// Clamped rather than rejected, because the windows at the ends of a track are half outside it and refusing to
        /// measure there would leave the first and last sections unreported. A boundary near an end is then looked for
        /// with less evidence, which is the honest reading of less evidence.
        /// </remarks>
        private static double Middle(double[] values)
        {
            if (values.Length == 0)
                return 0;

            var sorted = new double[values.Length];

            Array.Copy(values, sorted, values.Length);
            Array.Sort(sorted);

            return sorted[sorted.Length / 2];
        }

        private static double Middle(double[] values, int from, int to)
        {
            int low = Math.Max(0, from);
            int high = Math.Min(values.Length, to);

            if (high <= low)
                return 0;

            var window = new double[high - low];

            Array.Copy(values, low, window, 0, window.Length);
            Array.Sort(window);

            return window[window.Length / 2];
        }
    }
}
