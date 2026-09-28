using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using NUnit.Framework;
using ParaTactus.Detector;
using ParaTactus.Features;

namespace ParaTactus.Tests
{
    /// <summary>
    /// Writes the trained detector's tempo reading for a track, for a player to draw.
    /// </summary>
    /// <remarks>
    /// The detector reports a beat period on every frame, and that curve is the thing worth looking at when asking
    /// whether it is reading the right metrical level: a number cannot show where a track doubles and a curve can. The
    /// player cannot compute it while playing - the period head reads its context from a window, so beats near the end
    /// of what has arrived move every time more audio does - so it is computed once here and stored.
    ///
    /// What is written is one line per sample of the curve: the time in milliseconds, the tempo in BPM, and the model's
    /// beat confidence. The tempo is written rather than the period because a curve of periods against time is
    /// hyperbolic and the question being asked is about octaves, which a logarithmic axis of tempo turns into equal
    /// steps.
    ///
    /// Set <c>OSUTEST_AUDIO</c> to a raw float32 file or an audio file, <c>OSUTEST_CURVE_OUT</c> for where to write,
    /// and <c>OSUTEST_DETECTOR</c> for the model.
    /// </remarks>
    [Explicit("Needs OSUTEST_AUDIO, OSUTEST_CURVE_OUT and OSUTEST_DETECTOR.")]
    public class TempoCurveProbe
    {
        /// <summary>How many frames of the model's output make one line, so the file stays small enough to read.</summary>
        private const int frames_per_line = 5;

        [OneTimeSetUp]
        public void SetUp() => AudioTestEnvironment.Initialise();

        [Test]
        public void WriteTheDetectorsTempoCurve()
        {
            string audioPath = Environment.GetEnvironmentVariable("OSUTEST_AUDIO");
            string output = Environment.GetEnvironmentVariable("OSUTEST_CURVE_OUT");
            string detectorPath = Environment.GetEnvironmentVariable("OSUTEST_DETECTOR");

            if (string.IsNullOrEmpty(audioPath) || !File.Exists(audioPath))
                Assert.Ignore("Set OSUTEST_AUDIO to a raw float32 file or an audio file.");

            if (string.IsNullOrEmpty(output))
                Assert.Ignore("Set OSUTEST_CURVE_OUT for where to write.");

            if (string.IsNullOrEmpty(detectorPath) || !File.Exists(detectorPath))
                Assert.Ignore("Set OSUTEST_DETECTOR to the exported model.");

            float[] samples = Load(audioPath);
            double seconds = samples.Length / (double)LogMel.SampleRate;

            TestContext.Out.WriteLine($"audio {seconds:0.0}s from {Path.GetFileName(audioPath)}");

            using var detector = new LearnedBeatDetector(detectorPath);

            var clock = System.Diagnostics.Stopwatch.StartNew();
            (float[] beats, float[] periods) = detector.Activations(samples);
            clock.Stop();

            TestContext.Out.WriteLine($"analysis {clock.Elapsed.TotalSeconds:0.0}s for {beats.Length} frames "
                                      + $"({seconds / Math.Max(0.001, clock.Elapsed.TotalSeconds):0.0}x realtime)");

            if (beats.Length == 0)
                Assert.Fail("the detector produced no frames");

            // The tempo the curve implies, which is what a reader wants rather than the period: at fifty frames a second
            // a period of twenty frames is a beat every four hundred milliseconds, which is a hundred and fifty a minute.
            var curve = new List<(double Time, double Bpm, double Confidence)>();
            double total = 0;
            int counted = 0;

            for (int start = 0; start < beats.Length; start += frames_per_line)
            {
                int end = Math.Min(beats.Length, start + frames_per_line);
                double periodSum = 0;
                double confidenceSum = 0;

                for (int frame = start; frame < end; frame++)
                {
                    periodSum += periods[frame];
                    confidenceSum += beats[frame];
                }

                double period = periodSum / (end - start);
                double bpm = period > 0 ? 60000.0 / (period * 1000.0 / LearnedBeatDetector.FramesPerSecond) : 0;

                if (bpm > 0)
                {
                    total += bpm;
                    counted++;
                }

                curve.Add(((start + (end - start) / 2.0) * 1000.0 / LearnedBeatDetector.FramesPerSecond,
                           bpm,
                           confidenceSum / (end - start)));
            }

            using (var file = File.CreateText(output))
            {
                file.WriteLine("# time_ms\tbpm\tconfidence");

                foreach ((double time, double bpm, double confidence) in curve)
                {
                    file.WriteLine($"{time:0.0}\t{bpm:0.00}\t{confidence:0.0000}");
                }
            }

            var tempos = curve.Where(point => point.Bpm > 0).Select(point => point.Bpm).OrderBy(v => v).ToArray();

            TestContext.Out.WriteLine($"wrote {curve.Count} samples to {output}");
            TestContext.Out.WriteLine($"  tempo: min {tempos.FirstOrDefault():0}  p10 {tempos[tempos.Length / 10]:0}  "
                                      + $"median {tempos[tempos.Length / 2]:0}  p90 {tempos[tempos.Length * 9 / 10]:0}  "
                                      + $"max {tempos.LastOrDefault():0} BPM");
            TestContext.Out.WriteLine($"  confidence: mean {curve.Average(point => point.Confidence):0.000}");
            TestContext.Out.WriteLine($"  file: {new FileInfo(output).Length / 1024}KB");
        }

        /// <summary>Raw float32 if the file looks like it, otherwise decode it.</summary>
        private static float[] Load(string path)
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
