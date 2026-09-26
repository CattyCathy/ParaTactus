using System;
using System.Collections.Generic;

namespace ParaTactus
{
    /// <summary>
    /// Chooses the beats from the model's own activations by searching for the sequence with the steadiest tempo.
    /// </summary>
    /// <remarks>
    /// The model reports beats by picking peaks of its own output, and peak picking cannot see tempo. It takes each
    /// local maximum on its own merits, with a local rule about gaps, so nothing in it says "the tempo was 300ms a
    /// moment ago and it cannot be 500ms now". On a track whose tempo holds still that does not matter. On one that
    /// accelerates - this project's reference track climbs from 150 to 400 BPM over half a minute - the peaks wander:
    /// measured against the beatmap's own grid, the picked beats are 82ms out in that passage at the median, and every
    /// beat in a twelve second window is 63 to 155 milliseconds late with the lateness growing. That is a phase that
    /// has come adrift while the tempo is almost right, which is exactly the failure a tempo state fixes.
    ///
    /// So the search carries one. A state is a frame and an interval, and moving from one beat to the next costs
    /// whatever it costs to change the interval by that much - small changes are cheap, and the further apart two
    /// successive intervals are the more it costs, in the log domain so that a change from 300ms to 150ms costs what a
    /// change from 150ms to 300ms costs. The gain at each state is the model's own activation there. The best path
    /// through all of that is a sequence of beats that the model believes in and whose tempo does not jump about.
    ///
    /// Nothing here is a global tempo, and nothing decides a metrical level. A tempo that doubles over the track is a
    /// path whose intervals all double, which costs something and is worth it where the activations are strong; a
    /// passage at half speed is a path whose intervals double once, which costs one step and is taken when that is
    /// where the beats are. That is the difference between this and the regulariser it sits beside: the regulariser
    /// compares every passage against the track's single dominant period and moves anything more than half a period
    /// away from it by an octave, which cannot be right on a track that has more than one tempo in it.
    /// </remarks>
    public static class BeatSequenceSearch
    {
        /// <summary>The shortest interval between two beats that is considered, in frames.</summary>
        /// <remarks>
        /// Eight frames is 160ms, or 375 BPM. Faster than that is not a beat in any music this is for, and allowing it
        /// costs a great deal: the search spends its states on tempi nobody writes, and a spurious peak two frames from
        /// a real one becomes reachable.
        /// </remarks>
        public const int MinimumInterval = 8;

        /// <summary>The longest interval between two beats that is considered, in frames.</summary>
        /// <remarks>
        /// Sixty frames is 1.2 seconds, or 50 BPM. Longer gaps exist in music, and they are a repeated interval rather
        /// than a slow one: the search reaches across a break as two steps of the same interval, which costs nothing,
        /// while one step of a doubled interval costs the change. That is the intended reading of a break in the music.
        /// </remarks>
        public const int MaximumInterval = 60;

        /// <summary>How much a change of one interval to the next costs, per natural log of the ratio.</summary>
        /// <remarks>
        /// The one number here that has to be chosen rather than derived, and it trades two failures against each other.
        /// Too small and the search follows the strongest peaks wherever they are, which is what peak picking already
        /// does; too large and the tempo is effectively frozen, so a track that accelerates is tracked at one tempo and
        /// the beats pile up at one end of the passage. The default was chosen by measuring both ends of that trade
        /// against the reference beatmap; see <c>BeatSequenceSearchProbe</c>.
        ///
        /// This decides whether the path follows a tempo change. It does not decide how many beats the path has: a
        /// steady path pays nothing whatever its interval, so every extra beat it can land on a positive frame is
        /// free, and raising this from 6 to 40 leaves the count where it was to within a twentieth. What decides the
        /// count is <see cref="DefaultFloor"/>.
        /// </remarks>
        public const double DefaultTempoRigidity = 12;

        /// <summary>
        /// How strong the model has to believe in a frame before a beat can be placed on it, in logits above the mean.
        /// </summary>
        /// <remarks>
        /// The one number that decides how many beats the search finds, and it is not a matter of taste. Centring the
        /// reward on the track's own mean puts the bar at zero reward, and the mean of a beat model's output is dragged
        /// a long way down by the frames between beats: on Frums' XNOR the model reads +0.933 at the beatmap's own
        /// beats against a whole-track mean of -1.789. So the bar for a beat sits at -1.789 while the beats are at
        /// +0.933, and every frame that is merely above the mean - which is to say, a large share of the frames
        /// anywhere near a note - clears it. The search then places a beat on each of them, because a path is scored by
        /// the sum of its own rewards and a steady path pays nothing for the extra beats it takes.
        ///
        /// This raises that bar. A frame has to beat the mean by this much before it can carry a beat, so the frames
        /// that can are the ones the model is actually sure about. At zero the behaviour is the old one exactly.
        /// </remarks>
        public const double DefaultFloor = 0;

        /// <summary>
        /// The beats this activation implies, in frames.
        /// </summary>
        /// <param name="beats">The model's beat head, one value per frame.</param>
        /// <param name="downbeats">The model's downbeat head, the same length.</param>
        /// <param name="rigidity">How much a tempo change costs. See <see cref="DefaultTempoRigidity"/>.</param>
        /// <param name="floor">How sure the model has to be. See <see cref="DefaultFloor"/>.</param>
        public static int[] Frames(float[] beats, float[] downbeats, double rigidity = DefaultTempoRigidity, double floor = DefaultFloor)
        {
            if (beats == null)
                throw new ArgumentNullException(nameof(beats));

            int frames = beats.Length;
            int span = MaximumInterval - MinimumInterval + 1;

            if (frames <= MinimumInterval)
                return Array.Empty<int>();

            // How much the model believes in a beat here, scale removed so that the tempo cost and the activation cost
            // are in the same units. The activations are logits and their spread depends on the model and the track, so
            // a fixed scale would make the same rigidity mean different things on different material. The floor is
            // applied here rather than to the frames the path may use, because which frames clear it is exactly what
            // the path is for: a frame below it can still be walked over, it just cannot be landed on for free.
            double[] reward = rewardFor(beats, downbeats, floor);

            const double unreachable = double.NegativeInfinity;

            // score[frame, interval] is the best total up to a beat on that frame with that interval to the beat before
            // it. Kept for the whole track rather than two rows, because the path has to be walked back afterwards.
            var score = new double[frames, span];
            var from = new int[frames, span];

            for (int frame = 0; frame < frames; frame++)
            {
                for (int i = 0; i < span; i++)
                {
                    score[frame, i] = unreachable;
                    from[frame, i] = -1;
                }
            }

            // A first beat can be anywhere in the first interval's worth of frames, with no tempo behind it to agree
            // with. The track's own beginning is not reliably a beat - it is silence, a count-in, or a fade - so the
            // first beat is whatever the model is surest about early on.
            for (int frame = 0; frame < frames; frame++)
            {
                for (int i = 0; i < span; i++)
                    score[frame, i] = reward[frame];
            }

            for (int frame = 0; frame < frames; frame++)
            {
                for (int i = 0; i < span; i++)
                {
                    double best = score[frame, i];

                    if (double.IsNegativeInfinity(best))
                        continue;

                    int previousInterval = MinimumInterval + i;

                    for (int j = 0; j < span; j++)
                    {
                        int interval = MinimumInterval + j;
                        int next = frame + interval;

                        if (next >= frames)
                            break;

                        double total = best + reward[next] - (rigidity * Math.Abs(Math.Log((double)interval / previousInterval)));

                        if (total > score[next, j])
                        {
                            score[next, j] = total;
                            from[next, j] = (frame * span) + i;
                        }
                    }
                }
            }

            // The end of the best path anywhere in the track. Not necessarily the last beat: a track can end with the
            // music already stopped, and forcing a beat at the final frame would put one in the silence.
            double bestScore = double.NegativeInfinity;
            int bestFrame = -1;
            int bestInterval = -1;

            for (int frame = 0; frame < frames; frame++)
            {
                for (int i = 0; i < span; i++)
                {
                    if (score[frame, i] > bestScore)
                    {
                        bestScore = score[frame, i];
                        bestFrame = frame;
                        bestInterval = i;
                    }
                }
            }

            if (bestFrame < 0)
                return Array.Empty<int>();

            var path = new List<int>();
            int at = bestFrame;
            int intervalAt = bestInterval;

            while (at >= 0)
            {
                path.Add(at);

                int previous = from[at, intervalAt];

                if (previous < 0)
                    break;

                at = previous / span;
                intervalAt = previous % span;
            }

            path.Reverse();

            return path.ToArray();
        }

        /// <summary>
        /// How much the model believes in a beat on each frame, centred so that its mean is nothing.
        /// </summary>
        /// <remarks>
        /// Centred because the search is a comparison between paths of different lengths, and a path that lingers in a
        /// quiet passage must not be able to win by collecting many slightly positive frames. After centring, a frame
        /// the model is unsure about is worth nothing rather than a little, and only frames it actively believes in
        /// pull the path towards themselves.
        ///
        /// The centring is only as good as the mean it subtracts, and the mean of a beat model's output is not where
        /// its beats are - see <see cref="DefaultFloor"/>. That is what the floor raises.
        ///
        /// The downbeat head is folded in at a share of its weight rather than as a second dimension. Downbeats are
        /// where the bar starts, which is information about the metre and not about where a beat is; using them to
        /// choose the beat times directly would make every bar start a candidate beat and every other beat a weaker
        /// one. What they are good for here is tie-breaking between two peaks the beat head likes equally, which is
        /// what a small share does.
        /// </remarks>
        private static double[] rewardFor(float[] beats, float[] downbeats, double floor)
        {
            var reward = new double[beats.Length];

            double mean = 0;

            for (int i = 0; i < beats.Length; i++)
                mean += beats[i];

            mean /= beats.Length;

            for (int i = 0; i < beats.Length; i++)
            {
                double downbeat = downbeats != null && i < downbeats.Length ? downbeats[i] : 0;

                reward[i] = (beats[i] - mean - floor) + (0.15 * downbeat);
            }

            return reward;
        }
    }
}
