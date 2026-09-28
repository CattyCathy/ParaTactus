using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using NUnit.Framework;
using ParaTactus.Detector;
using ParaTactus.Features;
using ParaTactus.Grid;

namespace ParaTactus.Tests
{
    /// <summary>
    /// Checks the tempo sections a detector's beats imply against the sections a map actually declares.
    /// </summary>
    /// <remarks>
    /// The sections are a second reading of the same beats, so they fail in a way the beats do not: a section is one
    /// number for a whole passage, and a number a tenth of a per cent out is a tenth of a second out by the end of a
    /// hundred seconds. Three things are therefore reported and not one - how many sections came out against how many
    /// the map has, how much of the map's own pulse they account for, and how far each tempo is from the tempo the map
    /// holds over the same passage.
    ///
    /// The two numbers the reading rests on are swept, because they trade against each other and the trade is the whole
    /// question: a window short enough to catch a tempo change of a few beats is also short enough to fire on the
    /// detector's own jitter, and a share low enough to catch a small change also fires on it.
    ///
    /// The map's side is built the way the detector's is, from its uninherited timing points merged into runs of one
    /// beat length, so the counts are comparable. A map whose points are one tempo with hundreds of small edits written
    /// over them collapses to a few sections here, which is the point of the merge.
    ///
    /// Set <c>OSUTEST_AUDIO</c>, <c>OSUTEST_MAP</c> and <c>OSUTEST_DETECTOR</c>.
    /// </remarks>
    [Explicit("Needs OSUTEST_AUDIO, OSUTEST_MAP and OSUTEST_DETECTOR.")]
    public class TempoSectionProbe
    {
        [OneTimeSetUp]
        public void SetUp() => AudioTestEnvironment.Initialise();

        [Test]
        public void CompareTheSectionsTheBeatsImplyWithTheMaps()
        {
            string audioPath = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");
            string mapPath = Environment.GetEnvironmentVariable("OSUTEST_MAP");
            string detectorPath = Environment.GetEnvironmentVariable("OSUTEST_DETECTOR");

            if (string.IsNullOrEmpty(audioPath) || !File.Exists(audioPath))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            if (string.IsNullOrEmpty(mapPath) || !File.Exists(mapPath))
                Assert.Ignore("Set OSUTEST_MAP to the .osz or .osu.");

            if (string.IsNullOrEmpty(detectorPath) || !File.Exists(detectorPath))
                Assert.Ignore("Set OSUTEST_DETECTOR to the model.");

            string scratch = Path.Combine(Path.GetTempPath(), "paratactus-sections-" + Guid.NewGuid().ToString("N"));
            string map = mapPath;

            try
            {
                if (mapPath.EndsWith(".osz", StringComparison.OrdinalIgnoreCase))
                {
                    Directory.CreateDirectory(scratch);
                    ZipFile.ExtractToDirectory(mapPath, scratch);
                    map = Directory.EnumerateFiles(scratch, "*.osu", SearchOption.AllDirectories).First();
                }

                float[] samples = Read(audioPath);
                double until = samples.Length / (double)LogMel.SampleRate * 1000.0;

                using var detector = new LearnedBeatDetector(detectorPath);
                double[] beats = detector.Beats(samples);

                TempoSection[] truth = FromTimingPoints(map, until);

                TestContext.Out.WriteLine($"track  {Path.GetFileName(audioPath)}  {until / 1000:0.0}s");
                TestContext.Out.WriteLine($"beats  {beats.Length} from the detector");
                TestContext.Out.WriteLine($"map    {TimingPointCount(map)} uninherited points, merging into {truth.Length} sections");
                TestContext.Out.WriteLine("");
                TestContext.Out.WriteLine("  window  share  persist | sections  coverage   on tempo   octave   middle error   longest tempos");
                TestContext.Out.WriteLine("                        |            at half %   at two %");

                // The window and the share decide how small a change is seen; the persistence decides how long it has to
                // last. Swept together because they trade: a shorter window sees a change sooner and also sees jitter.
                foreach (int window in new[] { 8, 16, 24, 32, 48 })
                {
                    foreach (double share in new[] { 0.04, 0.06, 0.08, 0.10 })
                    {
                        Report(window, share, TempoSections.Persistence,
                               TempoSections.Analyse(beats, window, share, TempoSections.Persistence, TempoSections.ShortestSection),
                               truth);
                    }
                }

                TestContext.Out.WriteLine("");

                foreach (int persist in new[] { 1, 2, 3, 6, 12, 24 })
                {
                    Report(TempoSections.Window, TempoSections.ChangeShare, persist,
                           TempoSections.Analyse(beats, TempoSections.Window, TempoSections.ChangeShare, persist, TempoSections.ShortestSection),
                           truth);
                }

                // And the reading that would ship, in full, so that the sections themselves can be looked at.
                TestContext.Out.WriteLine("");
                TestContext.Out.WriteLine($"the sections at window {TempoSections.Window} and share {TempoSections.ChangeShare:0.00}:");
                TestContext.Out.WriteLine("   from       to        BPM      map BPM    error    map beats inside");

                double[] mapBeats = Beats(map);

                foreach (TempoSection section in TempoSections.Analyse(beats).OrderByDescending(s => s.Duration).Take(20))
                {
                    TempoSection? over = Over(section, truth);
                    double error = over.HasValue && over.Value.Period > 0
                        ? (section.Period - over.Value.Period) / over.Value.Period
                        : double.NaN;

                    int inside = mapBeats.Count(b => b >= section.Start && b < section.End);

                    TestContext.Out.WriteLine($"  {section.Start / 1000,7:0.0}s {section.End / 1000,7:0.0}s "
                                              + $"{section.BeatsPerMinute,8:0.00}  "
                                              + (over.HasValue ? $"{over.Value.BeatsPerMinute,8:0.00}" : "       -")
                                              + $"  {error * 100,7:+0.00;-0.00;0.00}%  {inside,8}"
                                              + (over.HasValue && IsOctaveOut(section, over.Value) ? "   <- the map's octave away" : ""));
                }
            }
            finally
            {
                if (Directory.Exists(scratch))
                    Directory.Delete(scratch, true);
            }
        }

        private static void Report(int window, double changeShare, int persistence, TempoSection[] found, TempoSection[] truth)
        {
            int covered = truth.Count(s => found.Any(c => c.Start < s.End && c.End > s.Start));
            double coverage = truth.Length == 0 ? 0 : 100.0 * covered / truth.Length;

            int half = found.Count(s => OnTempo(s, truth, 0.005));
            int two = found.Count(s => OnTempo(s, truth, 0.02));
            int octave = found.Count(s =>
            {
                TempoSection? over = Over(s, truth);

                return over.HasValue && IsOctaveOut(s, over.Value);
            });

            var errors = found
                .Select(s => (Section: s, Over: Over(s, truth)))
                .Where(p => p.Over.HasValue && !IsOctaveOut(p.Section, p.Over.Value) && p.Over.Value.Period > 0)
                .Select(p => Math.Abs(p.Section.Period - p.Over.Value.Period) / p.Over.Value.Period)
                .OrderBy(e => e)
                .ToArray();

            double middle = errors.Length == 0 ? double.NaN : errors[errors.Length / 2] * 100;

            // The sections that are on the map's tempo, longest first, so that the reading can be judged on the
            // passages that matter rather than on the short ones a boundary left behind.
            string longest = string.Join(" ", found
                .OrderByDescending(s => s.Duration)
                .Take(4)
                .Select(s => $"{s.BeatsPerMinute:0.0}"));

            TestContext.Out.WriteLine($"  {window,6}  {changeShare,5:0.00}  {persistence,7} | "
                                      + $"{found.Length,8}  {coverage,7:0}%  {half,7}  {two,6}  {octave,6}  "
                                      + $"{middle,11:0.000}%   {longest}");
        }

        private static TempoSection? Over(TempoSection section, TempoSection[] truth)
        {
            double middle = (section.Start + section.End) / 2;

            foreach (TempoSection candidate in truth)
            {
                if (middle >= candidate.Start && middle < candidate.End)
                    return candidate;
            }

            return null;
        }

        private static bool IsOctaveOut(TempoSection section, TempoSection over)
        {
            if (over.Period <= 0 || section.Period <= 0)
                return false;

            return Math.Abs(Math.Abs(Math.Log2(section.Period / over.Period)) - 1) < 0.15;
        }

        private static bool OnTempo(TempoSection section, TempoSection[] truth, double share)
        {
            TempoSection? over = Over(section, truth);

            if (over == null || over.Value.Period <= 0)
                return false;

            return Math.Abs(section.Period - over.Value.Period) / over.Value.Period <= share;
        }

        /// <summary>How many of the map's own beats fall inside a span, which is what a section claims to cover.</summary>
        private static int CountBeats(string map, double from, double to)
        {
            int count = 0;

            foreach (double beat in Beats(map))
            {
                if (beat >= from && beat < to)
                    count++;
            }

            return count;
        }

        private static double[] Beats(string map)
        {
            var points = new List<(double Time, double Length)>();
            bool inTiming = false;

            foreach (string raw in File.ReadLines(map))
            {
                string line = raw.Trim();

                if (line.StartsWith("[", StringComparison.Ordinal))
                {
                    inTiming = line.Equals("[TimingPoints]", StringComparison.OrdinalIgnoreCase);
                    continue;
                }

                if (!inTiming || line.Length == 0)
                    continue;

                string[] fields = line.Split(',');

                if (fields.Length >= 2
                    && double.TryParse(fields[0], NumberStyles.Float, CultureInfo.InvariantCulture, out double time)
                    && double.TryParse(fields[1], NumberStyles.Float, CultureInfo.InvariantCulture, out double length)
                    && length > 0)
                {
                    points.Add((time, length));
                }
            }

            points.Sort((a, b) => a.Time.CompareTo(b.Time));

            var beats = new List<double>();
            double last = points.Count > 0 ? points[^1].Time : 0;

            for (int i = 0; i < points.Count; i++)
            {
                double end = i + 1 < points.Count ? points[i + 1].Time : last;

                for (double time = points[i].Time; time < end; time += points[i].Length)
                    beats.Add(time);
            }

            return beats.OrderBy(b => b).ToArray();
        }

        /// <summary>A map's own sections, from its timing points merged into runs of one beat length.</summary>
        private static TempoSection[] FromTimingPoints(string map, double until)
        {
            var points = new List<(double Time, double Length)>();
            bool inTiming = false;

            foreach (string raw in File.ReadLines(map))
            {
                string line = raw.Trim();

                if (line.StartsWith("[", StringComparison.Ordinal))
                {
                    inTiming = line.Equals("[TimingPoints]", StringComparison.OrdinalIgnoreCase);
                    continue;
                }

                if (!inTiming || line.Length == 0)
                    continue;

                string[] fields = line.Split(',');

                if (fields.Length >= 2
                    && double.TryParse(fields[0], NumberStyles.Float, CultureInfo.InvariantCulture, out double time)
                    && double.TryParse(fields[1], NumberStyles.Float, CultureInfo.InvariantCulture, out double length)
                    && length > 0)
                {
                    points.Add((time, length));
                }
            }

            points.Sort((a, b) => a.Time.CompareTo(b.Time));

            var merged = new List<(double Start, double End, double Length)>();

            for (int i = 0; i < points.Count; i++)
            {
                double end = i + 1 < points.Count ? points[i + 1].Time : Math.Max(until, points[i].Time);

                // A thousandth, so that a mapper nudging a point to move the scroll speed does not read as a change of
                // tempo. Anything wider starts merging passages a player would hear as different.
                if (merged.Count > 0 && Math.Abs(points[i].Length - merged[^1].Length) <= 0.001 * merged[^1].Length)
                    merged[^1] = (merged[^1].Start, end, merged[^1].Length);
                else
                    merged.Add((points[i].Time, end, points[i].Length));
            }

            return merged.Select(m => new TempoSection(m.Start, m.End, m.Length)).ToArray();
        }

        private static int TimingPointCount(string map)
        {
            int count = 0;
            bool inTiming = false;

            foreach (string raw in File.ReadLines(map))
            {
                string line = raw.Trim();

                if (line.StartsWith("[", StringComparison.Ordinal))
                {
                    inTiming = line.Equals("[TimingPoints]", StringComparison.OrdinalIgnoreCase);
                    continue;
                }

                if (!inTiming || line.Length == 0)
                    continue;

                string[] fields = line.Split(',');

                if (fields.Length >= 2
                    && double.TryParse(fields[1], NumberStyles.Float, CultureInfo.InvariantCulture, out double length)
                    && length > 0)
                {
                    count++;
                }
            }

            return count;
        }

        private static float[] Read(string path)
        {
            if (path.EndsWith(".f32", StringComparison.OrdinalIgnoreCase))
            {
                byte[] bytes = File.ReadAllBytes(path);
                var samples = new float[bytes.Length / 4];

                Buffer.BlockCopy(bytes, 0, samples, 0, samples.Length * 4);

                return samples;
            }

            return ParaTactus.Decoding.BassAudioDecoder.DecodeMono(path, LogMel.SampleRate);
        }
    }
}
