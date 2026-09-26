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
    /// Whether choosing the beats by searching for the steadiest tempo beats picking the peaks.
    /// </summary>
    /// <remarks>
    /// The measurement that decides whether <see cref="BeatSequenceSearch"/> goes into the pipeline, and the one that
    /// chooses its rigidity. Everything is measured against the beatmap's own grid, per section, because the failures
    /// are local: the first eighty seconds are steady at 200 BPM and everything after is a tempo that moves, down to 90
    /// and then up to 400 BPM in steps.
    ///
    /// The compare is against the peak picking the model ships with, on the same decode, so the difference is the
    /// search and nothing else.
    ///
    /// <c>[Explicit]</c> because the audio cannot be committed. Set <c>OSUTEST_AUDIO</c> to the track.
    /// </remarks>
    [TestFixture]
    [Explicit("Needs OSUTEST_AUDIO pointing at Designant. - the audio is not in the repository.")]
    public class BeatSequenceSearchProbe
    {
        private const string model = "beat-this-final0-int8";

        private static readonly (double From, double To, string Name)[] sections =
        {
            (0.0, 82_000.0, "steady 200 BPM"),
            (82_000.0, 90_000.0, "down to 95"),
            (90_000.0, 110_000.0, "up to 150"),
            (110_000.0, 140_000.0, "up to 400"),
            (140_000.0, 178_000.0, "the last stretch"),
        };

        [OneTimeSetUp]
        public void SetUp() => AudioTestEnvironment.Initialise();

        [Test]
        public void ComparePeakPickingWithASearchOverTempo()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string modelPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", $"{model}.onnx");

            if (!File.Exists(modelPath))
                Assert.Ignore($"no model at {modelPath}");

            double[] grid = beatmapGrid(timingPoints());

            // Decoded and run once: everything below is arithmetic on the same activations, which is what makes four
            // rigidities affordable to compare.
            float[] samples = BassAudioDecoder.DecodeMono(audio, AnalysisAudio.SampleRate);

            var watch = System.Diagnostics.Stopwatch.StartNew();
            (float[] beatActivations, float[] downbeatActivations) = BeatThisBeatTracker.Activations(samples, modelPath);
            watch.Stop();

            TestContext.Out.WriteLine($"{beatActivations.Length} frames ({beatActivations.Length * 20.0 / 1000:0.#}s), "
                                      + $"model ran in {watch.Elapsed.TotalSeconds:0.#}s");

            double[] picked = BeatThisBeatTracker.BeatTimes(samples, modelPath);

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("whole track:");
            report("peaks, as the model picks them", picked, grid);

            var searched = new List<(double Rigidity, double[] Beats)>();

            foreach (double rigidity in new[] { 3.0, 12.0, 24.0 })
            {
                var search = System.Diagnostics.Stopwatch.StartNew();
                int[] frames = BeatSequenceSearch.Frames(beatActivations, downbeatActivations, rigidity);
                search.Stop();

                double[] beats = frames.Select(frame => BeatThisBeatTracker.FrameToMilliseconds(frame)).ToArray();

                searched.Add((rigidity, beats));

                TestContext.Out.WriteLine($"  search, rigidity {rigidity,5:0.#}      {beats.Length,5} beats  "
                                          + $"({search.ElapsedMilliseconds}ms)");
                report("", beats, grid);
            }

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("per section, as the median distance from the map's own beats:");

            TestContext.Out.WriteLine($"  {"",-26} {string.Join("", sections.Select(s => $"{s.From / 1000,7:0}s"))}");

            reportSections("peaks, as picked", picked, grid);

            foreach ((double rigidity, double[] beats) in searched)
                reportSections($"search, rigidity {rigidity:0.#}", beats, grid);

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("and the count of the map's beats covered within 60ms, per section:");

            TestContext.Out.WriteLine($"  {"",-26} {string.Join("", sections.Select(s => $"{s.From / 1000,7:0}s"))}");

            coverSections("peaks, as picked", picked, grid);

            foreach ((double rigidity, double[] beats) in searched)
                coverSections($"search, rigidity {rigidity:0.#}", beats, grid);

            // The metrical level, which decides whether the pulse reads as the music's beat or as twice it. A search
            // that covers more of the map's beats while finding twice as many is still wrong for a person watching: a
            // pulse at 400 BPM against music at 200 is not harmony, it is a strobe.
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("the median interval between the beats it found, in ms, per section:");
            TestContext.Out.WriteLine($"  {"",-26} {string.Join("", sections.Select(s => $"{s.From / 1000,7:0}s"))}");

            intervalSections("peaks, as picked", picked);

            foreach ((double rigidity, double[] beats) in searched)
                intervalSections($"search, rigidity {rigidity:0.#}", beats);

            TestContext.Out.WriteLine($"  {"the beatmap, for comparison",-26} {string.Join("", sections.Select(s => $"{s.From / 1000,7:0}s"))}");
            intervalSections("the beatmap", grid);
        }

        private static void intervalSections(string name, double[] beats)
        {
            var cells = new List<string>();

            foreach ((double from, double to, string _) in sections)
            {
                double[] inSection = beats.Where(b => b >= from && b < to).ToArray();

                if (inSection.Length < 3)
                {
                    cells.Add($"{0,7:0}");
                    continue;
                }

                var gaps = new List<double>();

                for (int i = 1; i < inSection.Length; i++)
                    gaps.Add(inSection[i] - inSection[i - 1]);

                gaps.Sort();

                cells.Add($"{gaps[gaps.Count / 2],7:0}");
            }

            TestContext.Out.WriteLine($"  {name,-26} {string.Join("", cells)}");
        }

        private static void report(string name, double[] beats, double[] grid)
        {
            if (beats.Length < 3)
            {
                TestContext.Out.WriteLine($"  {name,-26} too few beats");
                return;
            }

            var errors = beats.Select(b => Math.Abs(signedDistance(grid, b))).OrderBy(e => e).ToArray();

            TestContext.Out.WriteLine($"  {name,-26} n={beats.Length,5}  median {errors[errors.Length / 2],6:0.0}ms  "
                                      + $"p90 {errors[(int)(errors.Length * 0.9)],6:0.0}ms  worst {errors[^1],7:0.0}ms");
        }

        private static void reportSections(string name, double[] beats, double[] grid)
        {
            var cells = new List<string>();

            foreach ((double from, double to, string _) in sections)
            {
                double[] inSection = beats.Where(b => b >= from && b < to).ToArray();
                double[] sectionGrid = grid.Where(g => g >= from && g < to).ToArray();

                if (inSection.Length < 3 || sectionGrid.Length == 0)
                {
                    cells.Add($"{0,7:0}");
                    continue;
                }

                var errors = inSection.Select(b => Math.Abs(signedDistance(sectionGrid, b))).OrderBy(e => e).ToArray();

                cells.Add($"{errors[errors.Length / 2],7:0}");
            }

            TestContext.Out.WriteLine($"  {name,-26} {string.Join("", cells)}");
        }

        private static void coverSections(string name, double[] beats, double[] grid)
        {
            var cells = new List<string>();

            foreach ((double from, double to, string _) in sections)
            {
                double[] inSection = beats.Where(b => b >= from && b < to).ToArray();
                double[] sectionGrid = grid.Where(g => g >= from && g < to).ToArray();

                if (sectionGrid.Length == 0)
                {
                    cells.Add($"{0,6:0}%");
                    continue;
                }

                int covered = sectionGrid.Count(g => inSection.Any(b => Math.Abs(b - g) <= 60));

                cells.Add($"{covered * 100.0 / sectionGrid.Length,6:0}%");
            }

            TestContext.Out.WriteLine($"  {name,-26} {string.Join("", cells)}");
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

        private static double[] beatmapGrid(List<(double Time, double BeatLength)> timing)
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

        private static List<(double Time, double BeatLength)> timingPoints()
        {
            string file = Path.Combine(AppContext.BaseDirectory, "DesignantTimingPoints.csv");

            Assert.That(File.Exists(file), Is.True, $"{file} should be copied beside the test");

            var points = new List<(double, double)>();

            foreach (string line in File.ReadLines(file))
            {
                if (line.StartsWith("#") || !line.Contains(','))
                    continue;

                string[] fields = line.Split(',');

                if (fields.Length >= 2
                    && double.TryParse(fields[0], NumberStyles.Float, CultureInfo.InvariantCulture, out double time)
                    && double.TryParse(fields[1], NumberStyles.Float, CultureInfo.InvariantCulture, out double length))
                {
                    points.Add((time, length));
                }
            }

            return points.OrderBy(p => p.Item1).ToList();
        }
    }
}
