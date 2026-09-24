using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading.Tasks;
using NUnit.Framework;
using ParaTactus;

namespace ParaTactus.Tests
{
    /// <summary>
    /// The static entry point has to be usable from several threads at once.
    /// </summary>
    /// <remarks>
    /// It keeps one transform per thread, and it used to keep one across all of them: the draw thread running this for
    /// a visualiser while the analysis thread ran it for the model's frontend overwrote the same windowed frame, which
    /// corrupts the model's input instead of failing. A single call proves nothing about that, so every concurrent
    /// result is compared with what the same call produces on its own.
    /// </remarks>
    [TestFixture]
    public class FftThreadSafetyTest
    {
        private const int signalCount = 8;
        private const int rounds = 40;

        private static float[] signal(int seed)
        {
            var samples = new float[Fft.SIZE];
            var random = new Random(seed);

            for (int i = 0; i < samples.Length; i++)
                samples[i] = (float)(random.NextDouble() * 2 - 1);

            return samples;
        }

        [Test]
        public void ConcurrentCallsAgreeWithSerialOnes()
        {
            var inputs = Enumerable.Range(0, signalCount).Select(signal).ToArray();
            var expected = new float[signalCount][];

            for (int s = 0; s < signalCount; s++)
            {
                expected[s] = new float[Fft.BIN_COUNT];
                Fft.AnalyseMagnitudes(inputs[s], expected[s]);
            }

            var failures = new ConcurrentBag<string>();

            Parallel.For(0, signalCount * rounds, index =>
            {
                int s = index % signalCount;
                var actual = new float[Fft.BIN_COUNT];

                Fft.AnalyseMagnitudes(inputs[s], actual);

                for (int bin = 0; bin < Fft.BIN_COUNT; bin++)
                {
                    if (actual[bin] != expected[s][bin])
                    {
                        failures.Add($"signal {s} bin {bin}: {actual[bin]} against {expected[s][bin]}");
                        return;
                    }
                }
            });

            Assert.That(failures, Is.Empty, string.Join("; ", failures.Take(3)));
        }

        [Test]
        public void TheSameCallTwiceGivesTheSameAnswer()
        {
            var samples = signal(17);
            var first = new float[Fft.BIN_COUNT];
            var second = new float[Fft.BIN_COUNT];

            Fft.AnalyseMagnitudes(samples, first);
            Fft.AnalyseMagnitudes(samples, second);

            Assert.That(second, Is.EqualTo(first), "the scratch buffers must not carry state between calls");
        }

        [Test]
        public void AShortInputIsZeroPaddedRatherThanRejected()
        {
            var samples = signal(23).Take(16).ToArray();
            var magnitudes = new float[Fft.BIN_COUNT];

            Assert.DoesNotThrow(() => Fft.AnalyseMagnitudes(samples, magnitudes));
            Assert.That(magnitudes, Has.Some.Not.EqualTo(0), "a short non-silent input should still produce a spectrum");
        }
    }
}
