using System;
using System.IO;
using NUnit.Framework;
using ParaTactus;

namespace ParaTactus.Tests
{
    /// <summary>
    /// The grid cache: what it stores, what counts as the same track, and what a damaged file does.
    /// </summary>
    /// <remarks>
    /// No model and no audio here on purpose. Everything that decides whether a cached grid is used is file identity
    /// and parsing, and all of it can be checked without paying for an analysis - which matters, because the cases
    /// worth being sure about are the ones that are rare: a track that was edited, a model that was replaced, and a
    /// file that was truncated by a crash mid-write.
    /// </remarks>
    [TestFixture]
    public class BeatGridCacheTest
    {
        private string directory;
        private string audio;
        private string model;

        [SetUp]
        public void SetUp()
        {
            directory = Path.Combine(Path.GetTempPath(), "osutest-cache-" + Guid.NewGuid().ToString("n").Substring(0, 8));
            System.IO.Directory.CreateDirectory(directory);

            audio = Path.Combine(directory, "track.mp3");
            model = Path.Combine(directory, "model.onnx");

            File.WriteAllText(audio, "not really audio");
            File.WriteAllText(model, "not really a model");
        }

        [TearDown]
        public void TearDown()
        {
            try
            {
                System.IO.Directory.Delete(directory, true);
            }
            catch (IOException)
            {
            }
        }

        private static BeatGrid Grid(params double[] beats)
        {
            return BeatGrid.FromNormalisedBeats(beats, 1);
        }

        [Test]
        public void AGridSurvivesARoundTrip()
        {
            var cache = new BeatGridCache(directory);
            string key = BeatGridCache.KeyFor(audio, model);

            cache.Store(key, Grid(0, 461.5, 923, 1384.5));

            Assert.That(cache.TryLoad(key, out BeatGrid loaded), Is.True);
            Assert.That(loaded.Beats.Count, Is.EqualTo(4));
            Assert.That(loaded.Beats[1], Is.EqualTo(461.5).Within(1e-6));
            Assert.That(loaded.Beats[3], Is.EqualTo(1384.5).Within(1e-6));

            // The level is part of the answer: recomputing it on load could land on the other octave.
            Assert.That(loaded.MetricalShift, Is.EqualTo(1));
        }

        [Test]
        public void AMissingGridIsAMissRatherThanAnError()
        {
            var cache = new BeatGridCache(directory);

            Assert.That(cache.TryLoad("nothing-here", out BeatGrid grid), Is.False);
            Assert.That(grid, Is.Null);
        }

        [Test]
        public void EditingTheTrackMissesTheCache()
        {
            string before = BeatGridCache.KeyFor(audio, model);

            File.WriteAllText(audio, "different audio, different length");

            Assert.That(BeatGridCache.KeyFor(audio, model), Is.Not.EqualTo(before),
                "a re-encoded or edited track must not be served beats that describe the old one");
        }

        [Test]
        public void ReplacingTheModelMissesTheCache()
        {
            string before = BeatGridCache.KeyFor(audio, model);

            File.WriteAllText(model, "a different model entirely, of another size");

            Assert.That(BeatGridCache.KeyFor(audio, model), Is.Not.EqualTo(before),
                "beats from a previous model are not this model's answer");
        }

        [Test]
        public void DifferentTracksDoNotShareAGrid()
        {
            string other = Path.Combine(directory, "another.mp3");
            File.WriteAllText(other, "another track");

            Assert.That(BeatGridCache.KeyFor(audio, model), Is.Not.EqualTo(BeatGridCache.KeyFor(other, model)));
        }

        [Test]
        public void AdamagedFileIsAMissRatherThanAnException()
        {
            var cache = new BeatGridCache(directory);
            string key = BeatGridCache.KeyFor(audio, model);

            cache.Store(key, Grid(0, 500, 1000));

            // A crash part way through a write, and a file from a format that no longer exists.
            foreach (string damage in new[] { "osutest-beatgrid 1", "osutest-beatgrid 1\n1", "something else\n1\n0,1,2", "osutest-beatgrid 1\nnot-a-number\n0" })
            {
                File.WriteAllText(Path.Combine(directory, key + ".beats"), damage);

                Assert.That(cache.TryLoad(key, out BeatGrid grid), Is.False, $"accepted a damaged file: {damage.Replace("\n", "\\n")}");
                Assert.That(grid, Is.Null);
            }
        }

        [Test]
        public void AnEmptyGridRoundTrips()
        {
            // Distinct from a miss: a track with no beats is still an answer that was stored.
            var cache = new BeatGridCache(directory);
            string key = BeatGridCache.KeyFor(audio, model);

            cache.Store(key, BeatGrid.FromNormalisedBeats(new double[0], 0));

            Assert.That(cache.TryLoad(key, out BeatGrid loaded), Is.True);
            Assert.That(loaded.IsEmpty, Is.True);
        }
    }
}
