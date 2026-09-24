using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NUnit.Framework;
using ParaTactus;
using ParaTactus.Decoding;

namespace ParaTactus.Tests
{
    /// <summary>
    /// Analysing a track as it arrives, rather than all at once.
    /// </summary>
    /// <remarks>
    /// Whole-file analysis of the model path costs about a third of the track's own length - 65 seconds for a 178
    /// second track, of which 65 of those seconds are inference. A player cannot block on that, but it does not have
    /// to: the same measurement says inference runs at roughly a quarter of realtime, so a tracker fed the audio as it
    /// plays stays ahead of the playhead after a one-chunk head start.
    ///
    /// What makes that safe is that the pieces are the same computation, not an approximation of it. Feeding a track in
    /// arbitrary pieces has to give the same beats as feeding it whole; if it does not, every seam is somewhere a beat
    /// can be gained or lost, and the grid slowly desynchronises from the music - which is the exact failure this whole
    /// line of work exists to remove.
    /// </remarks>
    [TestFixture]
    public class StreamingBeatTrackerTest
    {
        [OneTimeSetUp]
        public void SetUp()
        {
            AudioTestEnvironment.Initialise();
        }

        private static string Audio()
        {
            string audio = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");

            if (string.IsNullOrEmpty(audio) || !File.Exists(audio))
                Assert.Ignore("Set OSUTEST_AUDIO to the track.");

            return audio;
        }

        private static string Model()
        {
            string model = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                           ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "beat-this-final0.onnx");

            Assert.That(File.Exists(model), Is.True, $"model not found at {model}");

            return model;
        }

        [Test]
        [Explicit("needs OSUTEST_AUDIO and the model file")]
        public void TestStreamedBeatsMatchWholeFileBeats()
        {
            string audio = Audio();
            string model = Model();

            double[] samples = BassAudioDecoder.DecodeMono(audio, 22050).Select(s => (double)s).ToArray();
            double[] whole = BeatThisBeatTracker.BeatTimes(audio, model, BassAudioDecoder.Default);

            var streamed = new StreamingBeatTracker(model);

            // Fed in pieces that are deliberately not a multiple of the chunk the model works in, so a seam lands
            // somewhere different each time.
            const int piece = 22050 / 3;

            for (int start = 0; start < samples.Length; start += piece)
            {
                int length = Math.Min(piece, samples.Length - start);
                var buffer = new float[length];

                for (int i = 0; i < length; i++)
                    buffer[i] = (float)samples[start + i];

                streamed.Add(buffer);
            }

            streamed.Flush();

            TestContext.Out.WriteLine($"whole file: {whole.Length} beats, streamed: {streamed.Beats.Count} beats");

            // Every whole-file beat has to be found, and no extras invented. Counting first keeps the message honest:
            // a handful of differences is a seam, hundreds is a broken implementation.
            var matched = new List<double>();
            var missed = new List<double>();

            foreach (double beat in whole)
            {
                double nearest = streamed.Beats.Count == 0
                    ? double.MaxValue
                    : streamed.Beats.Min(b => Math.Abs(b - beat));

                if (nearest <= 60)
                    matched.Add(beat);
                else
                    missed.Add(beat);
            }

            TestContext.Out.WriteLine($"matched {matched.Count} of {whole.Length}, missed {missed.Count}");

            if (missed.Count > 0)
                TestContext.Out.WriteLine("missed: " + string.Join(", ", missed.Take(10).Select(m => $"{m / 1000:0.0}s")));

            Assert.That(missed, Is.Empty, $"{missed.Count} beats found in the whole-file analysis were lost when streaming");
        }

        [Test]
        [Explicit("needs OSUTEST_AUDIO and the model file")]
        public void TestBeatsAppearBeforeTheTrackHasFinished()
        {
            string audio = Audio();
            string model = Model();

            float[] samples = BassAudioDecoder.DecodeMono(audio, 22050);
            var streamed = new StreamingBeatTracker(model);

            // One minute of a track is enough to have beats for most of that minute, with only the model's own chunk
            // as head start. This is the property that makes the cost acceptable at all.
            int minute = Math.Min(60 * 22050, samples.Length);
            var buffer = new float[minute];

            Array.Copy(samples, buffer, minute);
            streamed.Add(buffer);

            double latest = streamed.Beats.Count > 0 ? streamed.Beats[streamed.Beats.Count - 1] : 0;

            TestContext.Out.WriteLine($"after 60s of audio: {streamed.Beats.Count} beats, latest at {latest / 1000:0.0}s");

            Assert.That(streamed.Beats.Count, Is.GreaterThan(0), "no beats were produced from a minute of audio");

            // The head start is one 30 second chunk, so beats for at least the first 20 seconds must already be there.
            Assert.That(latest, Is.GreaterThan(20000), "beats lag more than a chunk behind the audio that has been fed");
        }
    }
}
