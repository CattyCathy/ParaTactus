using System;
using System.Linq;
using NUnit.Framework;
using ParaTactus;
using ParaTactus.Decoding;

namespace ParaTactus.Tests
{
    /// <summary>
    /// What the analysis is allowed to depend on, checked against the compiled assembly's own reference list, which is
    /// the one description of its dependencies that cannot drift away from the truth.
    /// </summary>
    /// <remarks>
    /// The analysis used to decode audio itself, through BASS, which is proprietary - free for non-commercial use and
    /// licensed per product otherwise - and it arrived with osu.Framework, FFmpeg, ImageSharp under a non-open-source
    /// split licence, a tablet driver stack, and tens of megabytes of native binaries for platforms this library has
    /// nothing to say about. Someone can put all of that back with one using statement, and it would compile and pass
    /// every other test here, which is exactly why this one exists.
    /// </remarks>
    [TestFixture]
    public class AnalysisLibraryBoundaryTest
    {
        /// <summary>
        /// Assemblies whose terms would spread to whoever uses the analysis, which must therefore not be referenced.
        /// </summary>
        private static readonly string[] proprietary = { "ManagedBass", "osu.Framework", "Bass" };

        [Test]
        public void TestTheAnalysisCarriesNothingProprietary()
        {
            var referenced = references(typeof(BeatGrid).Assembly);

            TestContext.Out.WriteLine("the analysis project references:");
            foreach (string name in referenced)
                TestContext.Out.WriteLine($"  {name}");

            var offenders = referenced
                            .Where(r => proprietary.Any(p => r.Contains(p, StringComparison.OrdinalIgnoreCase)))
                            .ToArray();

            Assert.That(offenders, Is.Empty,
                "the analysis can reach a library whose terms are not MIT, so it cannot be published as one: "
                + string.Join(", ", offenders));
        }

        [Test]
        public void TestDecodingIsOptionalAndSeparate()
        {
            var library = typeof(BeatGrid).Assembly;

            // The seam is part of the analysis; the implementation is not, and the point is that the analysis compiles
            // and works without ever loading the assembly that holds it.
            Assert.That(typeof(IAudioDecoder).Assembly, Is.EqualTo(library), "the decode seam belongs with the analysis");
            Assert.That(typeof(MonoMixdown).Assembly, Is.EqualTo(library), "mixdown is arithmetic and belongs with the analysis");
            Assert.That(typeof(BassAudioDecoder).Assembly, Is.Not.EqualTo(library), "the BASS decoder must not be inside the analysis");

            Assert.That(references(typeof(BassAudioDecoder).Assembly), Does.Contain("ParaTactus"),
                "the adapter is a consumer of the analysis like any other, not the other way round");
        }

        [Test]
        public void TestAllOfTheAnalysisLivesTogether()
        {
            var library = typeof(BeatGrid).Assembly;

            Assert.That(typeof(BeatGridProvider).Assembly, Is.EqualTo(library), "the provider belongs with the analysis");
            Assert.That(typeof(BeatThisBeatTracker).Assembly, Is.EqualTo(library));
            Assert.That(typeof(BeatTrainRegulariser).Assembly, Is.EqualTo(library));
            Assert.That(typeof(OnsetEnvelope).Assembly, Is.EqualTo(library));
            Assert.That(typeof(BeatGridCache).Assembly, Is.EqualTo(library));
        }

        [Test]
        public void TestBeatsCanBeTakenFromAFileOrFromSamples()
        {
            // Both forms are offered deliberately. A caller with a file should not have to decode it to ask for its
            // beats, and a caller feeding audio as it plays has no file to give. The file form asks for a decoder
            // rather than choosing one, which is what keeps the analysis free of BASS.
            var overloads = typeof(BeatThisBeatTracker)
                            .GetMethods()
                            .Where(m => m.IsPublic && m.Name == nameof(BeatThisBeatTracker.BeatTimes))
                            .ToArray();

            var fromPath = overloads.SingleOrDefault(m => m.GetParameters()[0].ParameterType == typeof(string));
            var fromSamples = overloads.SingleOrDefault(m => m.GetParameters()[0].ParameterType == typeof(float[]));

            Assert.That(fromPath, Is.Not.Null, "a path should be accepted");
            Assert.That(fromSamples, Is.Not.Null, "samples should be accepted");
            Assert.That(fromPath!.GetParameters().Select(p => p.ParameterType), Does.Contain(typeof(IAudioDecoder)),
                "the path form should take a decoder rather than choosing one");
        }

        private static string[] references(System.Reflection.Assembly assembly)
        {
            return assembly.GetReferencedAssemblies()
                           .Select(a => a.Name ?? string.Empty)
                           .OrderBy(n => n)
                           .ToArray();
        }
    }
}
