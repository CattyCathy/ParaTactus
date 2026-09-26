using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using NUnit.Framework;
using ParaTactus;
using ParaTactus.Decoding;

namespace ParaTactus.Tests
{
    /// <summary>
    /// Measures both readings of the beats over every beatmap in a folder, so that a change can be judged on more than
    /// one track.
    /// </summary>
    /// <remarks>
    /// The reference track this work has been measured against is one track, and it has been adjusted against many
    /// times. That is how a change is made that is right for Designant and wrong for everything else - the library's own
    /// documentation says as much about a single track being a corpus. This runs the same comparison over a folder
    /// instead: the model's peak picking against the search, per beatmap, with the beatmaps' own timing points as the
    /// reference, and the beatmaps sorted by how much their tempo moves so the hard ones are the ones to read first.
    ///
    /// A beatmap is a `.osu` file with its audio beside it, which is what the osu! client's editor writes when it
    /// exports one. Nothing here downloads anything: point <c>OSUTEST_CORPUS</c> at a folder of them.
    ///
    /// <c>[Explicit]</c>: it needs the audio and the model, and it takes about a minute a track on the first run. The
    /// second run is quick, because the analysis is cached beside the beat grids the player uses.
    /// </remarks>
    [TestFixture]
    [Explicit("Needs OSUTEST_CORPUS pointing at a folder of .osu files with their audio, and the model.")]
    public class BeatmapCorpusProbe
    {
        private const string defaultModel = "beat-this-final0-int8";

        [OneTimeSetUp]
        public void SetUp() => AudioTestEnvironment.Initialise();

        [Test]
        public void CompareBothReadingsOverEveryBeatmapInTheFolder()
        {
            string corpus = Environment.GetEnvironmentVariable("OSUTEST_CORPUS");

            if (string.IsNullOrEmpty(corpus) || !Directory.Exists(corpus))
                Assert.Ignore("Set OSUTEST_CORPUS to a folder holding .osu files and their audio.");

            string modelPath = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                               ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", $"{defaultModel}.onnx");

            if (!File.Exists(modelPath))
                Assert.Ignore($"no model at {modelPath}");

            List<Entry> entries = discover(corpus);

            if (entries.Count == 0)
                Assert.Ignore($"no .osu file with audio beside it in {corpus}");

            TestContext.Out.WriteLine($"{entries.Count} beatmap(s) in {corpus}");

            var results = new List<Result>();

            foreach (Entry entry in entries)
            {
                try
                {
                    results.Add(measure(entry, modelPath));
                }
                catch (Exception error)
                {
                    TestContext.Out.WriteLine($"  {entry.Name}: could not be measured - {error.Message}");
                }
            }

            if (results.Count == 0)
                Assert.Fail("nothing could be measured");

            // Hardest first. A track whose tempo never moves does not need the search and will not show whether it
            // helps; the ones worth reading are the ones where the tempo moves.
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("sorted by how much the tempo moves, most first:");
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine($"  {"beatmap",-34} {"tempo",22} {"peaks",22} {"search",22}");

            foreach (Result result in results.OrderByDescending(r => r.TempoSpread))
                TestContext.Out.WriteLine($"  {Trim(result.Name),-34} {result.TempoCell(),22} {result.PeaksCell(),22} {result.SearchCell(),22}");

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("the same numbers, as the median distance from each beatmap's own beats (ms), and the");
            TestContext.Out.WriteLine("share of those beats with a chosen beat within 60ms:");

            foreach (Result result in results.OrderByDescending(r => r.TempoSpread))
            {
                TestContext.Out.WriteLine($"");
                TestContext.Out.WriteLine($"  {result.Name}");
                TestContext.Out.WriteLine($"    tempo: {result.TempoCell()}");
                TestContext.Out.WriteLine($"    peaks:  median {result.PeaksMedian,6:0.0}ms  p90 {result.PeaksP90,6:0.0}ms  covered {result.PeaksCoverage,5:0.0}%  ({result.PeaksCount} beats)");
                TestContext.Out.WriteLine($"    search: median {result.SearchMedian,6:0.0}ms  p90 {result.SearchP90,6:0.0}ms  covered {result.SearchCoverage,5:0.0}%  ({result.SearchCount} beats)");
            }

            int better = results.Count(r => r.SearchCoverage > r.PeaksCoverage + 1);
            int worse = results.Count(r => r.SearchCoverage < r.PeaksCoverage - 1);

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine($"covered more of the map's beats with the search: {better} of {results.Count}; "
                                      + $"with the peaks: {worse}; "
                                      + $"about the same: {results.Count - better - worse}");
        }

        /// <summary>One beatmap and the audio beside it.</summary>
        private readonly struct Entry
        {
            public Entry(string name, string map, string audio)
            {
                Name = name;
                Map = map;
                Audio = audio;
            }

            public string Name { get; }

            public string Map { get; }

            public string Audio { get; }
        }

        private static List<Entry> discover(string corpus)
        {
            var entries = new List<Entry>();

            foreach (string map in Directory.EnumerateFiles(corpus, "*.osu", SearchOption.AllDirectories))
            {
                string folder = Path.GetDirectoryName(map);

                if (folder == null)
                    continue;

                string audio = audioFor(map, folder);

                if (audio == null)
                    continue;

                entries.Add(new Entry(Path.GetFileNameWithoutExtension(map), map, audio));
            }

            return entries;
        }

        /// <summary>
        /// The audio a beatmap names, or the only audio file beside it.
        /// </summary>
        /// <remarks>
        /// The name in the map is the authority and is tried first. The fallback is for maps whose audio has been
        /// renamed since, which is common in collections that have been passed around: if there is exactly one audio
        /// file in the folder it is the one, and if there are several, guessing would measure the wrong track.
        /// </remarks>
        private static string audioFor(string map, string folder)
        {
            string named = null;

            foreach (string line in File.ReadLines(map))
            {
                if (line.StartsWith("AudioFilename:", StringComparison.OrdinalIgnoreCase))
                {
                    named = line.Substring("AudioFilename:".Length).Trim();
                    break;
                }
            }

            if (!string.IsNullOrEmpty(named))
            {
                string candidate = Path.Combine(folder, named);

                if (File.Exists(candidate))
                    return candidate;
            }

            string[] extensions = { ".mp3", ".ogg", ".wav", ".flac", ".m4a" };

            var found = Directory.EnumerateFiles(folder)
                                 .Where(f => extensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
                                 .ToList();

            return found.Count == 1 ? found[0] : null;
        }

        private static Result measure(Entry entry, string modelPath)
        {
            List<TimingPoint> timing = timingPoints(entry.Map);

            Assert.That(timing.Count, Is.GreaterThan(1), $"{entry.Name} declares no tempo");

            double[] grid = beatmapGrid(timing);

            float[] samples = BassAudioDecoder.DecodeMono(entry.Audio, AnalysisAudio.SampleRate);
            (float[] activations, _) = BeatThisBeatTracker.Activations(samples, modelPath);

            double[] peaks = BeatThisBeatTracker.BeatTimes(samples, modelPath);
            double[] search = BeatSequenceSearch.Frames(activations, null)
                                                 .Select(frame => BeatThisBeatTracker.FrameToMilliseconds(frame))
                                                 .ToArray();

            double low = 60000 / timing.Max(p => p.BeatLength);
            double high = 60000 / timing.Min(p => p.BeatLength);

            return new Result
            {
                Name = entry.Name,
                TempoLow = low,
                TempoHigh = high,
                TempoSpread = high / Math.Max(1, low),
                Changes = timing.Count,
                PeaksCount = peaks.Length,
                SearchCount = search.Length,
                PeaksMedian = median(peaks, grid),
                PeaksP90 = percentile(peaks, grid, 0.9),
                PeaksCoverage = coverage(peaks, grid),
                SearchMedian = median(search, grid),
                SearchP90 = percentile(search, grid, 0.9),
                SearchCoverage = coverage(search, grid),
            };
        }

        private sealed class Result
        {
            public string Name;
            public double TempoLow;
            public double TempoHigh;
            public double TempoSpread;
            public int Changes;
            public int PeaksCount;
            public int SearchCount;
            public double PeaksMedian;
            public double PeaksP90;
            public double PeaksCoverage;
            public double SearchMedian;
            public double SearchP90;
            public double SearchCoverage;

            public string TempoCell() => Changes <= 1
                ? $"{TempoLow:0} BPM held"
                : $"{TempoLow:0}-{TempoHigh:0} BPM, {Changes} points";

            public string PeaksCell() => $"{PeaksMedian,5:0}ms {PeaksCoverage,5:0}% {PeaksCount,5} beats";

            public string SearchCell() => $"{SearchMedian,5:0}ms {SearchCoverage,5:0}% {SearchCount,5} beats";
        }

        private static string Trim(string name) => name.Length <= 34 ? name : name.Substring(0, 33) + "…";

        private static double median(double[] beats, double[] grid)
        {
            double[] sorted = errors(beats, grid);

            return sorted.Length == 0 ? 0 : sorted[sorted.Length / 2];
        }

        private static double percentile(double[] beats, double[] grid, double at)
        {
            double[] values = errors(beats, grid);

            return values.Length == 0 ? 0 : values[Math.Min(values.Length - 1, (int)(values.Length * at))];
        }

        private static double[] errors(double[] beats, double[] grid)
            => beats.Select(b => Math.Abs(signedDistance(grid, b))).OrderBy(e => e).ToArray();

        private static double coverage(double[] beats, double[] grid)
        {
            if (grid.Length == 0 || beats.Length == 0)
                return 0;

            int covered = grid.Count(g => beats.Any(b => Math.Abs(b - g) <= 60));

            return covered * 100.0 / grid.Length;
        }

        private static double signedDistance(double[] grid, double time)
        {
            int index = Array.BinarySearch(grid, time);

            if (index >= 0)
                return 0;

            index = ~index;

            if (index == 0)
                return time - grid[0];

            if (index >= grid.Length)
                return time - grid[^1];

            double before = time - grid[index - 1];
            double after = time - grid[index];

            return Math.Abs(before) <= Math.Abs(after) ? before : after;
        }

        private static double[] beatmapGrid(List<TimingPoint> timing)
        {
            var grid = new List<double>();

            for (int i = 0; i < timing.Count; i++)
            {
                double until = i + 1 < timing.Count ? timing[i + 1].Time : timing[i].Time + 60_000;

                for (double time = timing[i].Time; time < until && grid.Count < 500_000; time += timing[i].BeatLength)
                    grid.Add(time);
            }

            grid.Sort();

            return grid.ToArray();
        }

        private static List<TimingPoint> timingPoints(string map)
        {
            var points = new List<TimingPoint>();

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

                if (fields.Length < 2)
                    continue;

                if (!double.TryParse(fields[0], NumberStyles.Float, CultureInfo.InvariantCulture, out double time)
                    || !double.TryParse(fields[1], NumberStyles.Float, CultureInfo.InvariantCulture, out double length))
                {
                    continue;
                }

                // A negative beat length is an inherited point: it changes the volume or the samples and says nothing
                // about the tempo. Those are the ones that make a naive reading of a map think it has hundreds of
                // tempo changes.
                if (length <= 0)
                    continue;

                points.Add(new TimingPoint(time, length));
            }

            return points.OrderBy(p => p.Time).ToList();
        }

        private readonly struct TimingPoint
        {
            public TimingPoint(double time, double beatLength)
            {
                Time = time;
                BeatLength = beatLength;
            }

            public double Time { get; }

            public double BeatLength { get; }
        }
    }
}
