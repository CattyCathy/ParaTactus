using System;
using System.IO;
using NUnit.Framework;
using ParaTactus;
using ParaTactus.Decoding;

namespace ParaTactus.Tests
{
    /// <summary>
    /// Prints the beat grid a player would drive an animation from, for any track.
    /// </summary>
    /// <remarks>
    /// The other tests in this area check the analysis against a known answer; this one is for looking at a track
    /// nobody has an answer for yet. It uses the same provider and the same cache the player does, so the first run on
    /// a track costs a full analysis and every run after it is instant - which is also the quickest way to see the
    /// caching working rather than take it on trust.
    /// </remarks>
    [TestFixture]
    public class BeatGridProbe
    {
        [OneTimeSetUp]
        public void SetUp()
        {
            AudioTestEnvironment.Initialise();
        }

        [Test]
        [Explicit("needs OSUTEST_AUDIO and the model file")]
        public void TestTrackBeatGrid()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            string cache = Path.Combine(Path.GetTempPath(), "osutest-beatgrid-cache");
            var provider = new BeatGridProvider(model, cache, BassAudioDecoder.Default);

            bool cached = provider.IsCached(audio);

            var clock = System.Diagnostics.Stopwatch.StartNew();
            BeatGrid grid = provider.Get(audio);
            double seconds = clock.Elapsed.TotalSeconds;

            TestContext.Out.WriteLine($"{Path.GetFileName(audio)}");
            TestContext.Out.WriteLine($"  model:    {Path.GetFileName(model)} ({new FileInfo(model).Length / 1024.0 / 1024.0:0.#} MB)");
            TestContext.Out.WriteLine($"  {(cached ? "from cache" : "analysed")} in {seconds:0.000}s");
            TestContext.Out.WriteLine($"  beats:    {grid.Beats.Count} over {grid.Duration / 1000:0.0}s");
            TestContext.Out.WriteLine($"  octaves:  {grid.MetricalShift:+#;-#;0} from what the model reported");

            if (grid.IsEmpty)
                return;

            TestContext.Out.WriteLine();
            TestContext.Out.WriteLine("  tempo through the track:");

            for (int i = 0; i <= 10; i++)
            {
                double at = grid.Duration * i / 10.0;
                double bpm = grid.BpmAt(at);

                TestContext.Out.WriteLine($"    {at / 1000,7:0.0}s  {bpm,6:0.0} BPM  phase {grid.PhaseAt(at):0.00}");
            }

            TestContext.Out.WriteLine();

            var head = new System.Collections.Generic.List<string>();

            for (int i = 0; i < Math.Min(8, grid.Beats.Count); i++)
                head.Add($"{grid.Beats[i]:0}");

            TestContext.Out.WriteLine($"  first beats (ms): {string.Join(", ", head)}");
        }
    }
}
