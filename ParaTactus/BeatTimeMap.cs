using System;
using System.Collections.Generic;
using ParaTactus;

namespace ParaTactus
{
    /// <summary>
    /// A bpm/keyframe map describing how tempo evolves over the course of a track.
    /// </summary>
    /// <remarks>
    /// This is deliberately modelled on osu!'s <c>ControlPointInfo</c>: instead of a single global BPM,
    /// tempo is stored as a list of (time, bpm) keyframes and queried by time. That is what makes
    /// tracks whose tempo is not constant representable at all - the answer to "what is the BPM"
    /// is always "at time t, it is x".
    ///
    /// All times are in milliseconds, which is the convention of the player this grew out of.
    /// </remarks>
    public class BeatTimeMap
    {
        /// <summary>
        /// Fallback tempo when no keyframe covers a queried time (mirrors <c>TimingControlPoint.DEFAULT_BEAT_LENGTH</c>).
        /// </summary>
        public const double DEFAULT_BEAT_LENGTH = 1000;

        private readonly record struct Keyframe(double Time, double BeatLength);

        private readonly List<Keyframe> keyframes = new List<Keyframe>();

        /// <summary>
        /// Number of tempo keyframes in this map.
        /// </summary>
        public int KeyframeCount => keyframes.Count;

        /// <summary>
        /// The time of the first keyframe, or 0 when empty.
        /// </summary>
        public double StartTime => keyframes.Count == 0 ? 0 : keyframes[0].Time;

        /// <summary>
        /// The time of the last keyframe, or 0 when empty.
        /// </summary>
        public double EndTime => keyframes.Count == 0 ? 0 : keyframes[^1].Time;

        /// <summary>
        /// Adds a tempo keyframe. Times must be non-decreasing; duplicate times overwrite.
        /// </summary>
        public void Add(double time, double bpm)
        {
            if (!(bpm > 0) || double.IsNaN(bpm) || double.IsInfinity(bpm))
                throw new ArgumentOutOfRangeException(nameof(bpm), bpm, "BPM must be a finite positive number.");

            double beatLength = 60000 / bpm;

            if (keyframes.Count > 0)
            {
                double lastTime = keyframes[^1].Time;

                if (time < lastTime)
                    throw new ArgumentException($"Keyframes must be added in non-decreasing time order ({time} < {lastTime}).", nameof(time));

                if (time == lastTime)
                {
                    keyframes[^1] = new Keyframe(time, beatLength);
                    return;
                }
            }

            keyframes.Add(new Keyframe(time, beatLength));
        }

        /// <summary>
        /// Removes every keyframe, returning the map to an empty state.
        /// </summary>
        public void Clear() => keyframes.Clear();

        /// <summary>
        /// The beat length (milliseconds per beat) in effect at the given time.
        /// </summary>
        public double BeatLengthAt(double time)
        {
            if (keyframes.Count == 0)
                return DEFAULT_BEAT_LENGTH;

            int index = binarySearch(time);

            if (index < 0)
            {
                // Before the first keyframe: extend the first one backwards, as osu! does.
                index = 0;
            }

            return keyframes[index].BeatLength;
        }

        /// <summary>
        /// The tempo (beats per minute) in effect at the given time.
        /// </summary>
        public double BpmAt(double time) => 60000 / BeatLengthAt(time);

        /// <summary>
        /// The lowest tempo represented anywhere in this map.
        /// </summary>
        public double BpmMinimum
        {
            get
            {
                if (keyframes.Count == 0)
                    return 60000 / DEFAULT_BEAT_LENGTH;

                double maxBeatLength = double.MinValue;

                foreach (var k in keyframes)
                    maxBeatLength = Math.Max(maxBeatLength, k.BeatLength);

                return 60000 / maxBeatLength;
            }
        }

        /// <summary>
        /// The highest tempo represented anywhere in this map.
        /// </summary>
        public double BpmMaximum
        {
            get
            {
                if (keyframes.Count == 0)
                    return 60000 / DEFAULT_BEAT_LENGTH;

                double minBeatLength = double.MaxValue;

                foreach (var k in keyframes)
                    minBeatLength = Math.Min(minBeatLength, k.BeatLength);

                return 60000 / minBeatLength;
            }
        }

        /// <summary>
        /// Enumerates the keyframes as (time, bpm) pairs.
        /// </summary>
        public IEnumerable<(double Time, double Bpm)> Enumerate()
        {
            foreach (var k in keyframes)
                yield return (k.Time, 60000 / k.BeatLength);
        }

        private int binarySearch(double time)
        {
            int lo = 0;
            int hi = keyframes.Count - 1;

            while (lo <= hi)
            {
                int mid = lo + ((hi - lo) >> 1);
                double midTime = keyframes[mid].Time;

                if (midTime == time)
                    return mid;

                if (midTime < time)
                    lo = mid + 1;
                else
                    hi = mid - 1;
            }

            // hi is the last keyframe at or before `time`.
            return hi;
        }

        public override string ToString()
            => keyframes.Count == 0
                ? "BeatTimeMap(empty)"
                : $"BeatTimeMap({keyframes.Count} keyframes, {BpmMinimum:0.#}-{BpmMaximum:0.#} BPM)";
    }
}
