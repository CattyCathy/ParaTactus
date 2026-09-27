using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using NUnit.Framework;
using ParaTactus;

namespace ParaTactus.Tests
{
    /// <summary>
    /// The tempo sections over a corpus of tracks, against the sections the maps themselves declare.
    /// </summary>
    /// <remarks>
    /// One track says whether the reading can work; several say whether it works. Three things are counted per track and
    /// summarised, because a section reading can be wrong in three separate ways and a single number hides which:
    ///
    ///   the tempos      how far a reported section's beat is from the beat the map holds over the same passage, which
    ///                   is what makes a section readable as a BPM;
    ///   the boundaries  how far a section's start is from the map's nearest beat, which is what makes a tempo change
    ///                   appear where it happened rather than near it;
    ///   the drift       the error in the section's period carried over its own length, which is what a section costs
    ///                   and what a beat list does not: a section is one number for a whole passage.
    ///
    /// Sections whose tempo is an octave from the map's are counted separately and left out of the tempo error, because
    /// they are a different fault - the detector read the level of the music rather than the tempo of it - and averaging
    /// them in would report a reading that is a few per cent out as one that is a half out.
    ///
    /// Set <c>OSUTEST_CORPUS</c> to a folder of .osz files and <c>OSUTEST_DATASET</c> to where the decoded audio is.
    /// <c>OSUTEST_LIMIT</c> caps how many tracks are read.
    /// </remarks>
    [Explicit("Needs OSUTEST_CORPUS, OSUTEST_DATASET and OSUTEST_DETECTOR.")]
    public class TempoSectionCorpusProbe
    {
        [OneTimeSetUp]
        public void SetUp() => AudioTestEnvironment.Initialise();

        [Test]
        public void CompareTheSectionsOverACorpus()
        {
            string corpus = Environment.GetEnvironmentVariable("OSUTEST_CORPUS");
            string dataset = Environment.GetEnvironmentVariable("OSUTEST_DATASET") ?? @"D:\Linux\Proj\OsuTest\dataset";
            string detectorPath = Environment.GetEnvironmentVariable("OSUTEST_DETECTOR")
                                  ?? @"D:\Linux\Proj\OsuTest\model\detector.onnx";

            if (string.IsNullOrEmpty(corpus) || !Directory.Exists(corpus))
                Assert.Ignore("Set OSUTEST_CORPUS to a folder of .osz files.");

            if (!File.Exists(detectorPath))
                Assert.Ignore($"no detector at {detectorPath}");

            int limit = int.TryParse(Environment.GetEnvironmentVariable("OSUTEST_LIMIT"), out int wanted) ? wanted : 40;
            int window = int.TryParse(Environment.GetEnvironmentVariable("OSUTEST_WINDOW"), out int w) ? w : TempoSections.Window;
            int persistence = int.TryParse(Environment.GetEnvironmentVariable("OSUTEST_PERSIST"), out int p) ? p : TempoSections.Persistence;
            double share = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_SHARE"),
                                           NumberStyles.Float, CultureInfo.InvariantCulture, out double s)
                ? s
                : TempoSections.ChangeShare;

            var sets = Directory.EnumerateFiles(corpus, "*.osz", SearchOption.AllDirectories)
                                .Select(path => (Path: path, Id: Path.GetFileNameWithoutExtension(path).Split(' ')[0]))
                                .Where(pair => File.Exists(Path.Combine(dataset, "audio", pair.Id + ".f32")))
                                .OrderBy(pair => pair.Id)
                                .Take(limit)
                                .ToList();

            TestContext.Out.WriteLine($"{sets.Count} tracks with decoded audio, "
                                      + $"window {window} share {share:0.00} persistence {persistence}");
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("  track       map sec   found   on tempo   boundary     octave   drift    the map's tempos");

            var tempos = new List<double>();
            var boundaries = new List<double>();
            var drifts = new List<double>();
            var counts = new List<(int Map, int Found)>();
            int octaves = 0;
            int sections = 0;

            using var detector = new LearnedBeatDetector(detectorPath);

            foreach ((string path, string id) in sets)
            {
                float[] samples = Read(Path.Combine(dataset, "audio", id + ".f32"));
                double until = samples.Length / (double)LogMel.SampleRate * 1000.0;

                double[] beats = detector.Beats(samples);
                TempoSection[] found = TempoSections.Analyse(beats, window, share, persistence, TempoSections.ShortestSection);
                TempoSection[] truth = MapSections(path, until);
                double[] mapBeats = MapBeats(path);

                if (found.Length == 0 || truth.Length == 0)
                    continue;

                var errors = new List<double>();
                var places = new List<double>();
                var wander = new List<double>();
                int here = 0;

                foreach (TempoSection section in found)
                {
                    sections++;

                    TempoSection? over = Over(section, truth);

                    if (over.HasValue && over.Value.Period > 0)
                    {
                        if (IsOctave(section, over.Value))
                        {
                            octaves++;
                        }
                        else
                        {
                            errors.Add(Math.Abs(section.Period - over.Value.Period) / over.Value.Period);

                            // The drift a section costs, from the error in its period carried over its own length. It is
                            // the statement a beat list does not have to make: a section is one number for a whole
                            // passage, so an error in that number accumulates for as long as the passage lasts.
                            wander.Add(Math.Abs(section.Period - over.Value.Period) / over.Value.Period * section.Duration);
                        }
                    }

                    // Where the section begins against the map's own beats, which is where a tempo change would be
                    // shown. Only counted when the map has beats near it at all.
                    if (mapBeats.Length > 0)
                    {
                        double nearest = mapBeats.Min(b => Math.Abs(b - section.Start));

                        if (nearest < 2000)
                            places.Add(nearest);
                    }

                    if (over.HasValue)
                        here++;
                }

                if (errors.Count > 0)
                {
                    tempos.AddRange(errors);
                    drifts.AddRange(wander);
                }

                boundaries.AddRange(places);
                counts.Add((truth.Length, found.Length));

                TestContext.Out.WriteLine($"  {id,9} {truth.Length,8} {found.Length,7} {errors.Count,10} "
                                          + $"{(places.Count > 0 ? places.Average() : 0),9:0}ms "
                                          + $"{(errors.Count > 0 ? errors.Count(e => e > 0.002) : 0),8} "
                                          + $"{(wander.Count > 0 ? wander.Average() : 0),8:0}ms   "
                                          + string.Join(" ", truth.OrderByDescending(s => s.Duration).Take(4)
                                                               .Select(s => $"{s.BeatsPerMinute:0}/{s.Duration / 1000:0}s")));
            }

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine($"over {counts.Count} tracks and {sections} sections:");

            if (tempos.Count > 0)
            {
                tempos.Sort();
                drifts.Sort();

                TestContext.Out.WriteLine($"  sections on the map's tempo, within a half of one per cent: "
                                          + $"{100.0 * tempos.Count(t => t <= 0.005) / tempos.Count:0}%");
                TestContext.Out.WriteLine($"  sections on the map's tempo, within two per cent:          "
                                          + $"{100.0 * tempos.Count(t => t <= 0.02) / tempos.Count:0}%");
                TestContext.Out.WriteLine($"  middle tempo error over every section on the map's level:  "
                                          + $"{100.0 * tempos[tempos.Count / 2]:0.000}%");
                TestContext.Out.WriteLine($"  middle drift a section costs over its own length:          {drifts[drifts.Count / 2]:0}ms");
            }

            TestContext.Out.WriteLine($"  sections an octave from the map's:                         {100.0 * octaves / Math.Max(1, sections):0}%");

            if (boundaries.Count > 0)
            {
                boundaries.Sort();

                TestContext.Out.WriteLine($"  middle distance from a section's start to a map beat:       {boundaries[boundaries.Count / 2]:0}ms");
                TestContext.Out.WriteLine($"  boundaries within a frame of a map beat:                    "
                                          + $"{100.0 * boundaries.Count(b => b <= 20) / boundaries.Count:0}%");
            }

            double mapMean = counts.Average(c => (double)c.Map);
            double foundMean = counts.Average(c => (double)c.Found);
            TestContext.Out.WriteLine($"  sections per track: map {mapMean:0.0}, found {foundMean:0.0}");
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

        private static bool IsOctave(TempoSection section, TempoSection over)
        {
            if (over.Period <= 0 || section.Period <= 0)
                return false;

            return Math.Abs(Math.Abs(Math.Log2(section.Period / over.Period)) - 1) < 0.15;
        }

        /// <summary>A map's own sections, from its timing points merged into runs of one beat length.</summary>
        private static TempoSection[] MapSections(string path, double until)
        {
            var points = TimingPoints(path);
            var merged = new List<(double Start, double End, double Length)>();

            for (int i = 0; i < points.Count; i++)
            {
                double end = i + 1 < points.Count ? points[i + 1].Time : Math.Max(until, points[i].Time);

                if (merged.Count > 0 && Math.Abs(points[i].Length - merged[^1].Length) <= 0.001 * merged[^1].Length)
                    merged[^1] = (merged[^1].Start, end, merged[^1].Length);
                else
                    merged.Add((points[i].Time, end, points[i].Length));
            }

            return merged.Select(m => new TempoSection(m.Start, m.End, m.Length)).ToArray();
        }

        private static double[] MapBeats(string path)
        {
            var points = TimingPoints(path);
            var beats = new List<double>();

            for (int i = 0; i < points.Count; i++)
            {
                double end = i + 1 < points.Count ? points[i + 1].Time : points[i].Time;

                for (double time = points[i].Time; time < end; time += points[i].Length)
                    beats.Add(time);
            }

            return beats.OrderBy(b => b).ToArray();
        }

        private static List<(double Time, double Length)> TimingPoints(string path)
        {
            string text;

            if (path.EndsWith(".osz", StringComparison.OrdinalIgnoreCase))
            {
                using var archive = ZipFile.OpenRead(path);
                var entry = archive.Entries.First(e => e.Name.EndsWith(".osu", StringComparison.OrdinalIgnoreCase));

                using var reader = new StreamReader(entry.Open());

                text = reader.ReadToEnd();
            }
            else
            {
                text = File.ReadAllText(path);
            }

            var points = new List<(double Time, double Length)>();
            bool inTiming = false;

            foreach (string raw in text.Split('\n'))
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

            return points;
        }

        private static float[] Read(string path)
        {
            byte[] bytes = File.ReadAllBytes(path);
            var samples = new float[bytes.Length / 4];

            Buffer.BlockCopy(bytes, 0, samples, 0, samples.Length * 4);

            return samples;
        }
    }
}
