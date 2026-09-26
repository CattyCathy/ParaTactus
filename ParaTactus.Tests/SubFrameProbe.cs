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
    /// Measures one beatmap in one inference pass, for the questions that need an answer inside a minute.
    /// </summary>
    /// <remarks>
    /// <see cref="BeatmapCorpusProbe"/> answers how the readings compare across a folder and takes an hour to do it,
    /// because it decodes and runs the model twice per beatmap. A change to the peak pass is a question about one
    /// number - how far the beats it places sit from the map's own - and asking the whole folder for it is the wrong
    /// instrument. This decodes once, runs the model once, and reads the same activation through every peak pass that
    /// is worth comparing:
    ///
    ///   frames      the model's peak frames, no merge and no refinement, which is where the 20ms quantisation is;
    ///   merged      the frames merged into beats at their mean, which is the published postprocessor;
    ///   refined     the merged beats with a single-frame peak read between its frames, which is what ships.
    ///
    /// The point of having all three is that the two changes are separate claims and a single before-and-after cannot
    /// tell which of them earned the improvement. `merged` against `frames` is the fixing of an integer division that
    /// dropped the fraction at every step of a merged group; `refined` against `merged` is the parabola.
    ///
    /// Set <c>OSUTEST_CORPUS</c> to a folder of .osu files with their audio, and optionally <c>OSUTEST_MAP</c> to the
    /// part of one file name, which defaults to the first beatmap found. <c>OSUTEST_MODEL</c> overrides the model.
    /// </remarks>
    [Explicit("Needs OSUTEST_CORPUS pointing at a folder of .osu files with their audio, and the model.")]
    public class SubFrameProbe
    {
        private const string defaultModel = "beat-this-final0-int8";

        [OneTimeSetUp]
        public void SetUp() => AudioTestEnvironment.Initialise();

        [Test]
        public void CompareThePeakPassesOnOneBeatmap()
        {
            string corpus = Environment.GetEnvironmentVariable("OSUTEST_CORPUS");

            if (string.IsNullOrEmpty(corpus) || !Directory.Exists(corpus))
                Assert.Ignore("Set OSUTEST_CORPUS to a folder holding .osu files and their audio.");

            string modelPath = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                               ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", $"{defaultModel}.onnx");

            if (!File.Exists(modelPath))
                Assert.Ignore($"no model at {modelPath}");

            string wanted = Environment.GetEnvironmentVariable("OSUTEST_MAP");
            string map = Directory.EnumerateFiles(corpus, "*.osu", SearchOption.AllDirectories)
                                  .FirstOrDefault(f => wanted == null
                                                       || Path.GetFileName(f).Contains(wanted, StringComparison.OrdinalIgnoreCase));

            if (map == null)
                Assert.Ignore($"no .osu file in {corpus} matching {wanted}");

            string folder = Path.GetDirectoryName(map);
            string audio = folder == null ? null : audioFor(map, folder);

            if (audio == null)
                Assert.Ignore($"no audio beside {map}");

            List<TimingPoint> timing = timingPoints(map);
            Assert.That(timing.Count, Is.GreaterThanOrEqualTo(1), "the map declares no tempo");

            TestContext.Out.WriteLine($"beatmap: {Path.GetFileName(map)}");
            TestContext.Out.WriteLine($"audio:   {Path.GetFileName(audio)}");
            TestContext.Out.WriteLine($"tempo:   {(timing.Count <= 1 ? $"{60000 / timing[0].BeatLength:0} BPM held" : $"{60000 / timing.Max(p => p.BeatLength):0}-{60000 / timing.Min(p => p.BeatLength):0} BPM, {timing.Count} points")}");

            float[] samples = BassAudioDecoder.DecodeMono(audio, AnalysisAudio.SampleRate);
            (float[] activations, _) = BeatThisBeatTracker.Activations(samples, modelPath);

            double[] frames = BeatThisBeatTracker.FramePeaks(activations)
                                                 .Select(f => BeatThisBeatTracker.FrameToMilliseconds(f))
                                                 .ToArray();

            double[] merged = BeatThisBeatTracker.MergedPeaks(activations)
                                                 .Select(f => BeatThisBeatTracker.FrameToMilliseconds(f))
                                                 .ToArray();

            double[] refined = BeatThisBeatTracker.RawPeaks(activations)
                                                  .Select(f => BeatThisBeatTracker.FrameToMilliseconds(f))
                                                  .ToArray();

            double[] search = BeatSequenceSearch.Frames(activations, null)
                                                .Select(f => BeatThisBeatTracker.FrameToMilliseconds(f))
                                                .ToArray();

            double until = new[] { frames, merged, refined, search }.Max(set => set.Length > 0 ? set[^1] : 0);
            double[] grid = beatmapGrid(timing, until);

            TestContext.Out.WriteLine($"the map has {grid.Length} beats up to {until:0}ms");
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("  reading                                              count   rate   median      p90  covered   settled  scatter");
            report("the model's peak frames, as frames", frames, grid);
            report("the frames merged into beats at their mean", merged, grid);
            report("merged, then a lone peak read between frames", refined, grid);
            report("the tempo search over the activation", search, grid);

            // How many peaks came from more than one frame. If the model reports a beat as one sharp frame every time,
            // the merge step is doing nothing at all and the two readings above are the same array twice, which is
            // worth knowing rather than inferring from two identical columns.
            var widths = new SortedDictionary<int, int>();

            foreach (int width in BeatThisBeatTracker.MergedWidths(activations))
                widths[width] = widths.TryGetValue(width, out int seen) ? seen + 1 : 1;

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("how many frames each merged peak was made of:");
            TestContext.Out.WriteLine("  " + string.Join("  ", widths.Select(pair => $"{pair.Key} frame{(pair.Key == 1 ? "" : "s")}: {pair.Value}")));
            // The interval a beat map's own beats sit at, which is the scale everything above has to be read against. A
            // reading that puts beats at half this is not more precise, it is at the wrong metrical level.
            double mapInterval = grid.Length > 1 ? grid[1] - grid[0] : 0;

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine($"the map's own beat is {mapInterval:0}ms, so a pulse twice as often would be {mapInterval / 2:0}ms");

            // Which of the two failures it is, which decides what can be done about it. A signed offset that sits at
            // roughly one value all track long is a phase error, and a phase error is a constant that can be measured
            // and removed. An offset that walks in one direction is a tempo error, and a tempo error cannot be removed
            // by moving the beats - only by finding the tempo the beats are actually at. A offset that jumps around is
            // the peaks landing on the wrong events, which is neither.
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("the signed offset of the closest beat in each minute, in the model's own reading:");
            TestContext.Out.WriteLine("  minute        beats   median offset      spread");

            double[] reading = refined;

            for (int minute = 0; minute * 60_000 <= until; minute++)
            {
                double from = minute * 60_000;
                double to = from + 60_000;
                double[] signed = reading.Where(b => b >= from && b < to)
                                         .Select(b => signedDistance(grid, b))
                                         .OrderBy(v => v)
                                         .ToArray();

                if (signed.Length < 4)
                    continue;

                double middle = signed[signed.Length / 2];
                double spread = signed[Math.Min(signed.Length - 1, (int)(signed.Length * 0.75))]
                                - signed[Math.Min(signed.Length - 1, (int)(signed.Length * 0.25))];

                TestContext.Out.WriteLine($"  {minute,3}:00  {signed.Length,10}  {middle,12:+0;-0;0}ms  {spread,9:0}ms");
            }

            // The search finds more beats than the peaks on every track measured so far - 2315 against 1539 here - and
            // it is not the tempo rigidity that decides that. A steady path pays nothing whatever its interval, so
            // every extra beat it can land on a positive frame is free, and rigidity only prices a change of tempo.
            // Sweeping it from 6 to 40 moves the count by a twentieth and the scatter not at all.
            //
            // What decides it is the bar a frame has to clear to carry a beat. The reward is centred on the track's own
            // mean, and the mean of a beat model's output sits far below its beats, so the bar is met by a large share
            // of the frames anywhere near a note. The floor raises it. Read down for the floor at which the pulses
            // stop coming twice as often as the map's beats without the distance getting worse; if there is no such
            // row, the peaks are where the beats are and the search is not the instrument for them.
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("the search at a range of floors, at the default rigidity:");
            TestContext.Out.WriteLine("  reading                                              count   rate   median      p90  covered   settled  scatter");

            foreach (double floor in new[] { 0.0, 0.5, 1.0, 1.5, 2.0, 3.0 })
            {
                double[] atFloor = BeatSequenceSearch.Frames(activations, null, BeatSequenceSearch.DefaultTempoRigidity, floor)
                                                     .Select(f => BeatThisBeatTracker.FrameToMilliseconds(f))
                                                     .ToArray();

                report($"the search with a floor of {floor:0.0}", atFloor, grid);
            }

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("and the same floors with the rigidity raised, in case the two interact:");
            TestContext.Out.WriteLine("  reading                                              count   rate   median      p90  covered   settled  scatter");

            foreach (double floor in new[] { 1.0, 1.5, 2.0 })
            {
                double[] both = BeatSequenceSearch.Frames(activations, null, 32, floor)
                                                  .Select(f => BeatThisBeatTracker.FrameToMilliseconds(f))
                                                  .ToArray();

                report($"rigidity 32, floor {floor:0.0}", both, grid);
            }
        }

        /// <summary>
        /// What the regulariser's global octave fold does to each part of a track whose tempo moves.
        /// </summary>
        /// <remarks>
        /// Peak picking and the search both place beats, and neither decides how many there should be: a beat model
        /// reports a beat wherever the music has an event, so on a track with a section at half the density of the rest
        /// it reports the events of both at whatever density they are. The regulariser is what turns that into a metre,
        /// and it does it by comparing every passage against one number - the track's dominant period - and halving or
        /// doubling the local period until it is within half again of that number.
        ///
        /// One number for a whole track is a strong claim. A track whose tempo genuinely moves by more than a factor of
        /// 1.5 - a section at 100 BPM against a track that mostly sits at 180, say - has passages that are legitimately
        /// far from the dominant, and every one of them is folded onto it. The intervals then come out regular and the
        /// density in those passages is wrong, which is what a listener hears as the pulses holding a steady spacing
        /// while they stop agreeing with the music.
        ///
        /// This reads the fold directly rather than through the output: the dominant the walk used, the local period at
        /// every beat, and how many factors of two each was moved by. Read the histogram for whether the fold fires at
        /// all, and the per-section table for whether the passages it fires on are the ones the beatmap says are at a
        /// different tempo.
        /// </remarks>
        [Test]
        public void ShowWhatTheRegularisersOctaveFoldDoesToEachSection()
        {
            string corpus = Environment.GetEnvironmentVariable("OSUTEST_CORPUS");

            if (string.IsNullOrEmpty(corpus) || !Directory.Exists(corpus))
                Assert.Ignore("Set OSUTEST_CORPUS to a folder holding .osu files and their audio.");

            string modelPath = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                               ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", $"{defaultModel}.onnx");

            if (!File.Exists(modelPath))
                Assert.Ignore($"no model at {modelPath}");

            string wanted = Environment.GetEnvironmentVariable("OSUTEST_MAP");
            string map = Directory.EnumerateFiles(corpus, "*.osu", SearchOption.AllDirectories)
                                  .FirstOrDefault(f => wanted == null
                                                       || Path.GetFileName(f).Contains(wanted, StringComparison.OrdinalIgnoreCase));

            if (map == null)
                Assert.Ignore($"no .osu file in {corpus} matching {wanted}");

            string folder = Path.GetDirectoryName(map);
            string audio = folder == null ? null : audioFor(map, folder);

            if (audio == null)
                Assert.Ignore($"no audio beside {map}");

            List<TimingPoint> timing = timingPoints(map);

            TestContext.Out.WriteLine($"beatmap: {Path.GetFileName(map)}");
            TestContext.Out.WriteLine($"  its own tempo: {(timing.Count <= 1 ? $"{60000 / timing[0].BeatLength:0} BPM held" : $"{60000 / timing.Max(p => p.BeatLength):0}-{60000 / timing.Min(p => p.BeatLength):0} BPM over {timing.Count} points")}");

            float[] samples = BassAudioDecoder.DecodeMono(audio, AnalysisAudio.SampleRate);
            (float[] activations, _) = BeatThisBeatTracker.Activations(samples, modelPath);

            double[] picked = BeatThisBeatTracker.Peaks(activations)
                                                 .Select(f => BeatThisBeatTracker.FrameToMilliseconds(f))
                                                 .ToArray();

            double dominant = BeatTrainRegulariser.Dominant(picked);

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine($"the tracker's {picked.Length} picked beats give a dominant period of {dominant:0}ms, or {60000 / dominant:0} BPM");
            TestContext.Out.WriteLine($"  so the fold leaves alone every passage between {dominant * 0.59:0}ms and {dominant * 1.5:0}ms");
            TestContext.Out.WriteLine($"  which is {60000 / (dominant * 1.5):0} to {60000 / (dominant * 0.59):0} BPM");

            // The fold itself, replayed on the tracker's own beats so the count of shifts can be reported. This is the
            // same arithmetic the regulariser applies, read for what it did rather than for what came out.
            var shifts = new SortedDictionary<int, int>();

            for (int i = 0; i < picked.Length; i++)
            {
                double p = BeatTrainRegulariser.LocalPeriodAt(picked, i);

                if (p <= 0)
                    continue;

                int moved = 0;

                if (dominant > 0)
                {
                    for (int shift = 0; shift < 8; shift++)
                    {
                        double factor = p / dominant;

                        if (factor >= 1.5)
                        {
                            p /= 2;
                            moved--;
                        }
                        else if (factor <= 0.59)
                        {
                            p *= 2;
                            moved++;
                        }
                        else
                        {
                            break;
                        }
                    }
                }

                shifts[moved] = shifts.TryGetValue(moved, out int seen) ? seen + 1 : 1;
            }

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("how many factors of two the fold moved each beat's local period by:");
            TestContext.Out.WriteLine("  (a passage faster than the dominant has its period divided, which moves it up a level)");

            foreach (var pair in shifts)
            {
                string label = pair.Key == 0
                    ? "left alone"
                    : pair.Key < 0
                        ? $"period / {1 << -pair.Key}"
                        : $"period * {1 << pair.Key}";

                TestContext.Out.WriteLine($"  {label,-20} {pair.Value,6} beats");
            }

            // The same question from the beatmap's side: which of its own tempo sections sit outside the band, and so
            // are the ones the fold is entitled to move. If those are the passages that sound wrong, the fold is the
            // thing to change; if the wrong passages are inside the band, it is not.
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("the beatmap's own tempo sections, and whether the fold may move them:");
            TestContext.Out.WriteLine("   at      BPM    period   inside the band   the tracker's local period there");

            foreach (TimingPoint point in timing.Take(24))
            {
                bool inside = point.BeatLength >= dominant * 0.59 && point.BeatLength <= dominant * 1.5;
                double local = nearestLocalPeriod(picked, point.Time);

                TestContext.Out.WriteLine($"  {point.Time / 1000,5:0}s {60000 / point.BeatLength,6:0} {point.BeatLength,8:0}ms   {(inside ? "yes" : "no "),-16} {local,8:0}ms");
            }

            if (timing.Count > 24)
                TestContext.Out.WriteLine($"  ... and {timing.Count - 24} more sections");
        }

        /// <summary>
        /// The grid the player is actually given, at each of the settings it can be given one.
        /// </summary>
        /// <remarks>
        /// Every other reading in this probe is an intermediate: the model's peaks, the merged beats, the search. The
        /// player never sees any of those. What it draws comes out of <see cref="BeatGrid.FromBeats"/>, which passes the
        /// beats through <see cref="MetricalLevel.Normalise"/> on the way, and that is a third whole-track octave
        /// decision sitting on top of the two already in the beat train. Measuring the intermediate readings and not
        /// this one is how a grid can come out at the wrong level while every number in the probe looks reasonable.
        ///
        /// The metrical shift is reported because it is the decision itself rather than its result: a positive shift is
        /// the grid having been thinned by that many octaves and a negative one its having been subdivided, and either
        /// applies to the whole track at once.
        /// </remarks>
        [Test]
        public void ShowTheGridThePlayerActuallyDraws()
        {
            string corpus = Environment.GetEnvironmentVariable("OSUTEST_CORPUS");

            if (string.IsNullOrEmpty(corpus) || !Directory.Exists(corpus))
                Assert.Ignore("Set OSUTEST_CORPUS to a folder holding .osu files and their audio.");

            string modelPath = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                               ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", $"{defaultModel}.onnx");

            if (!File.Exists(modelPath))
                Assert.Ignore($"no model at {modelPath}");

            string wanted = Environment.GetEnvironmentVariable("OSUTEST_MAP");
            string map = Directory.EnumerateFiles(corpus, "*.osu", SearchOption.AllDirectories)
                                  .FirstOrDefault(f => wanted == null
                                                       || Path.GetFileName(f).Contains(wanted, StringComparison.OrdinalIgnoreCase));

            if (map == null)
                Assert.Ignore($"no .osu file in {corpus} matching {wanted}");

            string folder = Path.GetDirectoryName(map);
            string audio = folder == null ? null : audioFor(map, folder);

            if (audio == null)
                Assert.Ignore($"no audio beside {map}");

            List<TimingPoint> timing = timingPoints(map);

            TestContext.Out.WriteLine($"beatmap: {Path.GetFileName(map)}");
            TestContext.Out.WriteLine($"  its own tempo: {(timing.Count <= 1 ? $"{60000 / timing[0].BeatLength:0} BPM held" : $"{60000 / timing.Max(p => p.BeatLength):0}-{60000 / timing.Min(p => p.BeatLength):0} BPM over {timing.Count} points")}");

            float[] samples = BassAudioDecoder.DecodeMono(audio, AnalysisAudio.SampleRate);
            (float[] activations, _) = BeatThisBeatTracker.Activations(samples, modelPath);

            double[] picked = BeatThisBeatTracker.Peaks(activations)
                                                 .Select(f => BeatThisBeatTracker.FrameToMilliseconds(f))
                                                 .ToArray();

            double[] searched = BeatSequenceSearch.Frames(activations, null)
                                                  .Select(f => BeatThisBeatTracker.FrameToMilliseconds(f))
                                                  .ToArray();

            double[] regularised = BeatTrainRegulariser.Regularise(picked);

            double until = new[] { picked, searched, regularised }.Max(set => set.Length > 0 ? set[^1] : 0);
            double[] mapGrid = beatmapGrid(timing, until);

            // What the beatmap's own beats run at, as the rate the grid ought to be aiming for. The median gap rather
            // than the mean, because a map's tempo points include its slow sections and the mean is dragged by them.
            double mapInterval = medianInterval(mapGrid);

            TestContext.Out.WriteLine($"  the map's own beats are {1000 / mapInterval:0.00} a second, one every {mapInterval:0}ms");

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("the same beats at each stage, and the grid each stage hands the player:");
            TestContext.Out.WriteLine("  stage                                                count   rate    median gap");

            stage("the model's peaks, pruned", picked);
            stage("the peaks onto a regular spacing", regularised);
            stage("the tempo search", searched);

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("what the player gets, at each setting it can have:");
            TestContext.Out.WriteLine("  grid                                                 count   rate   metrical shift");

            grid_("peaks, regularised, then the metrical level", BeatGrid.FromBeats(regularised));
            grid_("the search, then the metrical level", BeatGrid.FromBeats(searched));

            // The search's own beat count is a function of the floor and nothing else - see the sweep above - so the
            // floor is the one dial that can put it on the same level as the peaks. Reported as a grid rather than as a
            // reading because a floor that matches the peaks in count can still be folded differently by the metrical
            // level, and what matters is where it lands after that.
            foreach (double floor in new[] { 1.0, 1.5, 2.0 })
            {
                double[] atFloor = BeatSequenceSearch.Frames(activations, null, BeatSequenceSearch.DefaultTempoRigidity, floor)
                                                     .Select(f => BeatThisBeatTracker.FrameToMilliseconds(f))
                                                     .ToArray();

                grid_($"the search at floor {floor:0.0}, then the metrical level", BeatGrid.FromBeats(atFloor));
            }

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("and the same two grids without the metrical level being decided at all, for contrast:");
            TestContext.Out.WriteLine("  grid                                                 count   rate   metrical shift");

            grid_("peaks, regularised, no metrical level", BeatGrid.FromNormalisedBeats(regularised, 0));
            grid_("the search, no metrical level", BeatGrid.FromNormalisedBeats(searched, 0));

            // Which octave the level decision preferred, and how strongly. A majority barely over the threshold is a
            // track whose tempo genuinely moves and where one level for all of it is a poor description; one near one is
            // a track at a single level and the decision is easy.
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("the metrical level decision on each reading:");
            TestContext.Out.WriteLine("  reading                                  shift        share inside 80-220 BPM");

            foreach ((string name, double[] beats) in new[] { ("peaks, regularised", regularised), ("the search", searched) })
            {
                var intervals = new List<double>();

                for (int i = 1; i < beats.Length; i++)
                {
                    if (beats[i] - beats[i - 1] > 0)
                        intervals.Add(beats[i] - beats[i - 1]);
                }

                int shift = MetricalLevel.ChooseShift(beats);

                TestContext.Out.WriteLine($"  {name,-40} {shift,5}   {100 * MetricalLevel.ShareInRange(intervals, shift, MetricalLevel.DefaultMinimumBpm, MetricalLevel.DefaultMaximumBpm),22:0.0}%");

                // Every shift rather than the one that won. The decision is a majority vote, so a reading that is in
                // the wrong octave but not overwhelmingly so loses the vote and is left where it is - and the numbers
                // that say whether that happened are the shares at the shifts that were not taken.
                for (int candidate = -2; candidate <= 2; candidate++)
                {
                    double share = 100 * MetricalLevel.ShareInRange(intervals, candidate, MetricalLevel.DefaultMinimumBpm, MetricalLevel.DefaultMaximumBpm);

                    TestContext.Out.WriteLine($"      at shift {candidate,2}: {share,5:0.0}% of intervals in the band{(share >= 100 * MetricalLevel.DefaultMajority ? "  <- would win a vote" : "")}");
                }
            }

            // What the wrong octave costs, in the terms a listener hears. The grid is right when a beat of it is near a
            // beat of the map; at twice the map's rate every other beat of the grid has no map beat to be near, and at
            // half the rate half the map's beats go by with nothing. Both are reported, because "twice as many" and
            // "half as many" are different faults and the fix is different.
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("each grid against the map's own beats:");
            TestContext.Out.WriteLine("  reading                                              count   rate   median      p90  covered   settled  scatter");

            foreach ((string name, BeatGrid grid) in new[]
                     {
                         ("peaks, regularised, then the metrical level", BeatGrid.FromBeats(regularised)),
                         ("the search, then the metrical level", BeatGrid.FromBeats(searched)),
                     })
            {
                report(name, grid.Beats.ToArray(), mapGrid);
            }
        }

        /// <summary>
        /// Why the regulariser drops beats instead of filling the ones the tracker missed.
        /// </summary>
        /// <remarks>
        /// The regulariser exists to do two opposite things: delete a beat the tracker put on a subdivision, and put
        /// back a beat the tracker missed. Which one it does is decided entirely by the local period it measures, and
        /// that period is measured from the tracker's own beats - so a passage where the tracker has lost every other
        /// beat is a passage where most of the gaps are two periods long, the local estimate agrees with them, and
        /// every remaining beat looks correctly spaced. Nothing is deleted and nothing is filled, and the passage plays
        /// at half the density it should.
        ///
        /// This reads the two numbers that decide it: the local period the walk settles on, and the track's own
        /// dominant period that a passage is compared against for the halving test. If the local period in a lost
        /// passage has already come out at twice the tempo the rest of the track is at, the fill cannot fire, because
        /// the gap it would fill is exactly one local period wide.
        /// </remarks>
        [Test]
        public void ShowWhyTheRegulariserDropsBeatsInsteadOfFillingThem()
        {
            string corpus = Environment.GetEnvironmentVariable("OSUTEST_CORPUS");

            if (string.IsNullOrEmpty(corpus) || !Directory.Exists(corpus))
                Assert.Ignore("Set OSUTEST_CORPUS to a folder holding .osu files and their audio.");

            string modelPath = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                               ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", $"{defaultModel}.onnx");

            if (!File.Exists(modelPath))
                Assert.Ignore($"no model at {modelPath}");

            string wanted = Environment.GetEnvironmentVariable("OSUTEST_MAP");
            string map = Directory.EnumerateFiles(corpus, "*.osu", SearchOption.AllDirectories)
                                  .FirstOrDefault(f => wanted == null
                                                       || Path.GetFileName(f).Contains(wanted, StringComparison.OrdinalIgnoreCase));

            if (map == null)
                Assert.Ignore($"no .osu file in {corpus} matching {wanted}");

            string folder = Path.GetDirectoryName(map);
            string audio = folder == null ? null : audioFor(map, folder);

            if (audio == null)
                Assert.Ignore($"no audio beside {map}");

            List<TimingPoint> timing = timingPoints(map);

            TestContext.Out.WriteLine($"beatmap: {Path.GetFileName(map)}");

            float[] samples = BassAudioDecoder.DecodeMono(audio, AnalysisAudio.SampleRate);
            (float[] activations, _) = BeatThisBeatTracker.Activations(samples, modelPath);

            double[] picked = BeatThisBeatTracker.Peaks(activations)
                                                 .Select(f => BeatThisBeatTracker.FrameToMilliseconds(f))
                                                 .ToArray();

            double[] regularised = BeatTrainRegulariser.Regularise(picked);
            double dominant = BeatTrainRegulariser.Dominant(picked);

            TestContext.Out.WriteLine($"  the tracker picked {picked.Length} beats; the regulariser's dominant period is {dominant:0}ms, or {60000 / dominant:0} BPM");
            TestContext.Out.WriteLine($"  the regulariser kept {regularised.Length} of them ({100.0 * regularised.Length / Math.Max(1, picked.Length):0}%)");
            TestContext.Out.WriteLine($"  the beatmap's own beats are one every {medianInterval(beatmapGrid(timing, picked.Length > 0 ? picked[^1] : 0)):0}ms");

            // The gaps the tracker actually produced, which is what the local period is measured from. A spike at twice
            // the map's interval with almost nothing at the map's own interval is the signature of a passage lost by
            // halves: the beats are not ragged, they are every other one.
            var buckets = new SortedDictionary<int, int>();

            for (int i = 1; i < picked.Length; i++)
            {
                int bucket = (int)(Math.Round((picked[i] - picked[i - 1]) / 20) * 20);

                buckets[bucket] = buckets.TryGetValue(bucket, out int seen) ? seen + 1 : 1;
            }

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("the gaps between the tracker's picked beats, in 20ms buckets, most common first:");
            TestContext.Out.WriteLine("     gap    count   BPM at that gap");

            foreach (var pair in buckets.OrderByDescending(pair => pair.Value).Take(12))
                TestContext.Out.WriteLine($"  {pair.Key,6}ms {pair.Value,6}   {60000.0 / Math.Max(1, pair.Key),10:0}");

            // The same question the regulariser asks, at a handful of places spread through the track: the gap to the
            // previous beat, the local period it is judged against, and which of the three branches that lands in.
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("the decision at each tenth of the track:");
            TestContext.Out.WriteLine("      at     gap   local period   gap/period   what the walk does");

            for (int tenth = 1; tenth <= 10; tenth++)
            {
                int index = Math.Min(picked.Length - 1, picked.Length * tenth / 10);

                if (index < 1)
                    continue;

                double gap = picked[index] - picked[index - 1];
                double period = BeatTrainRegulariser.LocalPeriodAt(picked, index);
                double ratio = period > 0 ? gap / period : 0;

                string verdict = period <= 0
                    ? "no period measured"
                    : ratio < 0.85
                        ? "deletes the beat as a subdivision"
                        : ratio > 1.45 && ratio <= 5.0
                            ? $"fills {Math.Round(ratio) - 1} missing beat(s)"
                            : ratio > 5.0
                                ? "treats it as a break in the music"
                                : "keeps it as it is";

                TestContext.Out.WriteLine($"  {picked[index] / 1000,5:0}s {gap,7:0}ms {period,12:0}ms {ratio,12:0.00}   {verdict}");
            }

            // The evidence that is not the beats themselves. Whether a long gap is a missed beat or a real change of
            // tempo is invisible in the beat positions - the gap is the same shape either way - and visible in the
            // model's own activation, which either believes in a beat in the middle of it or does not. What is needed
            // is not an opinion about that but a measurement of how separable the two cases are on real material, and
            // in particular of whether a genuine tempo change looks like the halves of the fast passages.
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("what the model believes in the middle of each gap, which is what tells a missed beat");
            TestContext.Out.WriteLine("from a genuine one:");
            TestContext.Out.WriteLine("   gap class    gaps   midpoint above zero   midpoint is a local peak   midpoint - track mean   best within 100ms");

            // The gap classes are read off the map's own beat, so they mean something: a gap of one map beat is the
            // track's own spacing and needs no beat inside it, and a gap of two is exactly where one is missing.
            double mapInterval = medianInterval(beatmapGrid(timing, picked.Length > 0 ? picked[^1] : 0));

            // Before any of the distances is believed, the reference itself has to be checked. Every number here is a
            // distance from this grid, and a grid built wrongly makes the tracker look wrong by exactly as much. What
            // is checked is not whether the parse succeeded - that is a different question and the parser has its own
            // notes - but whether the grid the parse produced is a plausible description of the track: its first beats,
            // and how long its own spacings last, because a map that spends most of its length at one beat length and
            // is being read as another is a map this cannot see.
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine($"the reference grid, which every distance below is measured against ({timing.Count} tempo points):");
            TestContext.Out.WriteLine("   at        BPM    beat length");

            foreach (TimingPoint point in timing.Take(10))
                TestContext.Out.WriteLine($"  {point.Time / 1000,6:0.0}s {60000 / point.BeatLength,7:0.0} {point.BeatLength,13:0}ms");

            if (timing.Count > 10)
                TestContext.Out.WriteLine($"  ... and {timing.Count - 10} more points, the last at {timing[^1].Time / 1000:0.0}s");

            // How much of the track's duration each beat length accounts for, which is what the grid actually is rather
            // than what its points say. A map whose points are mostly one value but whose length is mostly another is a
            // map where the tempo points and the music disagree, and then the grid is not a reference at all.
            var durations = new SortedDictionary<int, double>();

            for (int i = 0; i < timing.Count; i++)
            {
                double end = i + 1 < timing.Count ? timing[i + 1].Time : picked.Length > 0 ? picked[^1] : timing[i].Time;
                int bucket = (int)(Math.Round(timing[i].BeatLength / 20) * 20);

                durations[bucket] = durations.TryGetValue(bucket, out double seen) ? seen + (end - timing[i].Time) : end - timing[i].Time;
            }

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("how much of the track each beat length is declared over:");

            foreach (var pair in durations.OrderByDescending(pair => pair.Value).Take(6))
                TestContext.Out.WriteLine($"  {pair.Key,6}ms ({60000.0 / pair.Key,5:0} BPM): {pair.Value / 1000,7:0.0}s  {100 * pair.Value / Math.Max(1, picked.Length > 0 ? picked[^1] : 1),5:0.0}%");


            var byClass = new SortedDictionary<int, List<int>>();

            for (int i = 1; i < picked.Length; i++)
            {
                int classes = (int)Math.Round((picked[i] - picked[i - 1]) / mapInterval);

                if (classes < 1 || classes > 6)
                    continue;

                if (!byClass.TryGetValue(classes, out List<int> frames))
                {
                    frames = new List<int>();
                    byClass[classes] = frames;
                }

                frames.Add(midpointFrame(activations, picked[i - 1], picked[i]));
            }

            foreach (var pair in byClass)
            {
                int above = 0;
                int peaked = 0;
                double sum = 0;
                double bestNearby = double.NegativeInfinity;
                int bestOffset = 0;

                foreach (int frame in pair.Value)
                {
                    if (frame < 0 || frame >= activations.Length)
                        continue;

                    if (activations[frame] > 0)
                        above++;

                    if (isPeakAt(activations, frame))
                        peaked++;

                    sum += activations[frame];

                    // The strongest activation within a sixth of a second of the midpoint, and how far away it was. The
                    // midpoint is where a missing beat would be if the beats either side of it were where they should
                    // be, and they are not: the tracker places a beat 25 to 40ms early on average, so the beat that is
                    // missing from a gap is not exactly halfway. A search at the exact midpoint alone therefore
                    // measures the offset as much as it measures the beat, and reports a beat the model is sure about
                    // as absent.
                    for (int offset = -5; offset <= 5; offset++)
                    {
                        int at = frame + offset;

                        if (at < 0 || at >= activations.Length)
                            continue;

                        if (activations[at] > bestNearby)
                        {
                            bestNearby = activations[at];
                            bestOffset = offset;
                        }
                    }
                }

                double average = pair.Value.Count == 0 ? 0 : sum / pair.Value.Count;

                TestContext.Out.WriteLine($"  {pair.Key,9}   {pair.Value.Count,6}   {100.0 * above / Math.Max(1, pair.Value.Count),17:0}%   {100.0 * peaked / Math.Max(1, pair.Value.Count),22:0}%   {average,20:+0.00;-0.00;0.00}   {bestNearby,14:+0.00;-0.00;0.00} at {bestOffset * 20,+4}ms");
            }

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("for contrast, what the model believes at the beats it did detect:");
            TestContext.Out.WriteLine("   reading                          count   above zero   is a local peak   value - track mean");

            double mean = 0;

            foreach (float value in activations)
                mean += value;

            mean /= Math.Max(1, activations.Length);

            int atBeatAbove = 0;
            int atBeatPeak = 0;
            double atBeatSum = 0;
            int atBeatCount = 0;

            foreach (double beat in picked)
            {
                int frame = BeatThisBeatTracker.FrameOf(beat, activations.Length);

                if (frame < 0 || frame >= activations.Length)
                    continue;

                atBeatCount++;

                if (activations[frame] > 0)
                    atBeatAbove++;

                if (isPeakAt(activations, frame))
                    atBeatPeak++;

                atBeatSum += activations[frame];
            }

            TestContext.Out.WriteLine($"  the tracker's beats        {atBeatCount,8}   {100.0 * atBeatAbove / Math.Max(1, atBeatCount),9:0}%   {100.0 * atBeatPeak / Math.Max(1, atBeatCount),15:0}%   {atBeatSum / Math.Max(1, atBeatCount) - mean,19:+0.00;-0.00;0.00}");
            TestContext.Out.WriteLine($"  the whole track mean is {mean:+0.00;-0.00;0.00}, which is the bar a beat has to clear");

            // Whether the doubled gaps come in passages or one at a time, which is the difference between the music
            // changing tempo and the picking losing a beat. A real change is a run of one spacing; a lost beat is a
            // gap of two in the middle of gaps of one. Written as a string of digits, one per gap, because that is the
            // shape of it and a histogram cannot show a shape.
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine($"the gaps in time order as a multiple of the map's own beat ({mapInterval:0}ms), from a tenth in:");
            TestContext.Out.WriteLine("  (a run of 2s is a passage at half density; 1 and 2 alternating is a beat being missed)");

            var run = new System.Text.StringBuilder();
            int runStart = picked.Length / 10;

            for (int i = runStart; i < Math.Min(picked.Length, runStart + 160); i++)
            {
                int classes = (int)Math.Round((picked[i] - picked[i - 1]) / mapInterval);

                run.Append(classes is >= 1 and <= 9 ? (char)('0' + classes) : '?');
            }

            TestContext.Out.WriteLine($"  {run}");

            var runs = new SortedDictionary<int, int>();

            for (int i = 1; i < picked.Length; i++)
            {
                int classes = (int)Math.Round((picked[i] - picked[i - 1]) / mapInterval);

                if (classes is < 1 or > 9)
                    classes = 0;

                runs[classes] = runs.TryGetValue(classes, out int seen) ? seen + 1 : 1;
            }

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("how many gaps of each size, as a multiple of the map's own beat:");

            foreach (var pair in runs.OrderBy(pair => pair.Key))
            {
                string what = pair.Key switch
                {
                    0 => "not a whole number of map beats",
                    1 => "the map's own spacing",
                    2 => "a gap of two, so a beat is missing",
                    _ => $"{pair.Key} map beats, so several are missing",
                };

                TestContext.Out.WriteLine($"  {pair.Key,3}: {pair.Value,6} gaps   {what}");
            }

            // The measurement the fill's threshold rests on: in a gap of two, is the model's strongest belief near the
            // middle closer to what it reads on a beat or to what it reads between beats? A gap of one is the control,
            // because nothing is missing there and its middle must read as the space between beats.
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("the strongest activation near each gap's midpoint, as a share of the model's strength on the");
            TestContext.Out.WriteLine("beats it did report:");

            double level = beatLevel(activations, picked);

            TestContext.Out.WriteLine($"  (that strength is {level:+0.00;-0.00;0.00}, and the whole track's mean is {mean:+0.00;-0.00;0.00})");

            foreach (int classes in new[] { 1, 2, 3 })
            {
                var shares = new List<double>();

                for (int i = 1; i < picked.Length; i++)
                {
                    if ((int)Math.Round((picked[i] - picked[i - 1]) / mapInterval) != classes)
                        continue;

                    if (strongestNear(activations, (picked[i - 1] + picked[i]) / 2, out double strength))
                        shares.Add(strength / level);
                }

                if (shares.Count < 4)
                    continue;

                shares.Sort();

                TestContext.Out.WriteLine($"  gaps of {classes}: {shares.Count,5} of them, share at the 25th/50th/75th percentile "
                                          + $"{shares[shares.Count / 4],5:0.00} / {shares[shares.Count / 2],5:0.00} / {shares[shares.Count * 3 / 4],5:0.00}");
            }
        }

        /// <summary>The frame halfway between two beats.</summary>
        private static int midpointFrame(float[] activations, double from, double to)
        {
            double middle = (from + to) / 2;
            double frame = middle * BeatThisBeatTracker.FramesPerSecond / 1000.0;

            return BeatThisBeatTracker.FrameOf(frame, activations.Length);
        }

        /// <summary>Whether the activation at a frame is as high as everything within the peak radius of it.</summary>
        private static bool isPeakAt(float[] activations, int frame)
        {
            if (frame < 0 || frame >= activations.Length || activations[frame] <= 0)
                return false;

            const int radius = 3;

            for (int j = Math.Max(0, frame - radius); j <= Math.Min(activations.Length - 1, frame + radius); j++)
            {
                if (activations[j] > activations[frame])
                    return false;
            }

            return true;
        }

        /// <summary>The model's median activation on the beats a tracker reported.</summary>
        private static double beatLevel(float[] activations, double[] beats)
        {
            var values = new List<double>(beats.Length);

            foreach (double beat in beats)
            {
                int frame = BeatThisBeatTracker.FrameOf(beat * BeatThisBeatTracker.FramesPerSecond / 1000.0, activations.Length);

                if (frame >= 0 && frame < activations.Length)
                    values.Add(activations[frame]);
            }

            if (values.Count == 0)
                return 0;

            values.Sort();

            return values[values.Count / 2];
        }

        /// <summary>The strongest activation near a time, within a sixth of a second.</summary>
        private static bool strongestNear(float[] activations, double milliseconds, out double strength)
        {
            int centre = BeatThisBeatTracker.FrameOf(milliseconds * BeatThisBeatTracker.FramesPerSecond / 1000.0, activations.Length);
            const int radius = 6;

            strength = double.NegativeInfinity;

            for (int offset = -radius; offset <= radius; offset++)
            {
                int at = centre + offset;

                if (at < 0 || at >= activations.Length)
                    continue;

                if (activations[at] > strength)
                    strength = activations[at];
            }

            return !double.IsNegativeInfinity(strength);
        }

        /// <summary>One stage of the beat train, as a count, a rate and a middle gap.</summary>
        private static void stage(string name, double[] beats)
        {
            double middle = medianInterval(beats);

            TestContext.Out.WriteLine($"  {name,-52} {beats.Length,6} {(middle > 0 ? $"{1000 / middle,6:0.00}" : "   n/a")} {middle,11:0}ms");
        }

        /// <summary>One grid, as a count, a rate and the octave it was moved by.</summary>
        private static void grid_(string name, BeatGrid grid)
        {
            double middle = medianInterval(grid.Beats.ToArray());

            TestContext.Out.WriteLine($"  {name,-52} {grid.Beats.Count,6} {(middle > 0 ? $"{1000 / middle,6:0.00}" : "   n/a")} {grid.MetricalShift,16}");
        }
        private static double nearestLocalPeriod(double[] beats, double time)
        {
            if (beats.Length == 0)
                return 0;

            int best = 0;

            for (int i = 1; i < beats.Length; i++)
            {
                if (Math.Abs(beats[i] - time) < Math.Abs(beats[best] - time))
                    best = i;
            }

            return BeatTrainRegulariser.LocalPeriodAt(beats, best);
        }

        /// <summary>
        /// Writes a window of a track with the map's beat and the tracker's beat marked by different clicks.
        /// </summary>
        /// <remarks>
        /// The point of this is to hear the answer instead of deriving it, and to hear it against the audio the
        /// tracker actually saw. Every number in this suite is a distance from the beatmap's own grid, which assumes
        /// the map is right about where the music's beats are - true of a well-made map and not something this code
        /// can check. If the tracker's clicks and the map's clicks land on top of each other and both land on the
        /// music, the reading is right; if the model's clicks sound early against the music, the error is in the
        /// tracker, and the map's clicks are what say so.
        ///
        /// The two are separable by ear: the map's beat is a click with a rising edge and the tracker's is a tick
        /// with the phase flipped, so they are opposite in a waveform view as well as different in timbre. A pair that
        /// arrives together is a beat both readings agree on; a tick on its own is a beat only the tracker believes
        /// in, and a click on its own is one it missed.
        ///
        /// Set <c>OSUTEST_CORPUS</c>, optionally <c>OSUTEST_MAP</c>, and:
        ///   <c>OSUTEST_FROM</c>     milliseconds into the track to start, default 60000
        ///   <c>OSUTEST_LENGTH</c>   how much to write, default 20000
        ///   <c>OSUTEST_OUT</c>      where to write the .wav, default beside the corpus
        ///   <c>OSUTEST_RATE</c>     the rate to write at, default 44100
        /// </remarks>
        [Test]
        public void WriteAWindowWithBothPulsesAudible()
        {
            string corpus = Environment.GetEnvironmentVariable("OSUTEST_CORPUS");

            if (string.IsNullOrEmpty(corpus) || !Directory.Exists(corpus))
                Assert.Ignore("Set OSUTEST_CORPUS to a folder holding .osu files and their audio.");

            string modelPath = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                               ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", $"{defaultModel}.onnx");

            if (!File.Exists(modelPath))
                Assert.Ignore($"no model at {modelPath}");

            string wanted = Environment.GetEnvironmentVariable("OSUTEST_MAP");
            string map = Directory.EnumerateFiles(corpus, "*.osu", SearchOption.AllDirectories)
                                  .FirstOrDefault(f => wanted == null
                                                       || Path.GetFileName(f).Contains(wanted, StringComparison.OrdinalIgnoreCase));

            if (map == null)
                Assert.Ignore($"no .osu file in {corpus} matching {wanted}");

            string folder = Path.GetDirectoryName(map);
            string audio = folder == null ? null : audioFor(map, folder);

            if (audio == null)
                Assert.Ignore($"no audio beside {map}");

            double from = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_FROM"), NumberStyles.Float, CultureInfo.InvariantCulture, out double parsedFrom) ? parsedFrom : 60_000;
            double length = double.TryParse(Environment.GetEnvironmentVariable("OSUTEST_LENGTH"), NumberStyles.Float, CultureInfo.InvariantCulture, out double parsedLength) ? parsedLength : 20_000;
            string output = Environment.GetEnvironmentVariable("OSUTEST_OUT")
                            ?? Path.Combine(Path.GetTempPath(), $"{Path.GetFileNameWithoutExtension(map)}-{from:0}-{length:0}.wav");

            // The analysis is done at the model's own rate, because that is what decides where the tracker puts a beat
            // and the point here is to hear the tracker's answer. The window is written at a rate the ear can use:
            // 22050Hz stops at 11kHz, which is where cymbals and the attack of a snare live, and those are exactly the
            // sounds a beat is judged against.
            float[] samples = BassAudioDecoder.DecodeMono(audio, AnalysisAudio.SampleRate);
            (float[] activations, _) = BeatThisBeatTracker.Activations(samples, modelPath);

            // The analysis samples are not needed again, and the window can be tens of megabytes at 44.1kHz.
            samples = Array.Empty<float>();

            double[] tracked = BeatThisBeatTracker.RawPeaks(activations)
                                                  .Select(f => BeatThisBeatTracker.FrameToMilliseconds(f))
                                                  .ToArray();

            double[] grid = beatmapGrid(timingPoints(map), tracked.Length > 0 ? tracked[^1] : 0);

            int rate = int.TryParse(Environment.GetEnvironmentVariable("OSUTEST_RATE"), out int parsedRate) && parsedRate > 0
                ? parsedRate
                : 44_100;

            TestContext.Out.WriteLine($"decoding the window at {rate}Hz for listening, and analysing at {AnalysisAudio.SampleRate}Hz");

            float[] full = BassAudioDecoder.DecodeMono(audio, rate);

            int first = (int)Math.Max(0, Math.Round(from * rate / 1000.0));
            int count = (int)Math.Min(full.Length - first, Math.Round(length * rate / 1000.0));

            if (count <= 0)
                Assert.Ignore($"{from}ms is past the end of the track, which is {full.Length * 1000.0 / rate:0}ms");

            var window = new float[count];
            Array.Copy(full, first, window, 0, count);

            full = Array.Empty<float>();

            int mapClicks = mark(window, grid, from, rate, 0.22, rising: true);
            int modelClicks = mark(window, tracked, from, rate, 0.22, rising: false);

            writeWav(output, window, rate);

            TestContext.Out.WriteLine($"wrote {output}");
            TestContext.Out.WriteLine($"  {length / 1000:0}s from {from / 1000:0}s of {Path.GetFileName(audio)}");
            TestContext.Out.WriteLine($"  the map's beat is a click, {mapClicks} of them in the window");
            TestContext.Out.WriteLine($"  the tracker's beat is a tick, {modelClicks} of them in the window");
            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("listen for a tick that arrives on its own - a beat only the tracker believes in -");
            TestContext.Out.WriteLine("and for a click that arrives on its own, which is a beat it missed. A tick that");
            TestContext.Out.WriteLine("sounds early or late against the music is the placement error this suite has been");
            TestContext.Out.WriteLine("measuring as a median distance.");
        }

        /// <summary>Puts a short click at each of a set of times on top of some audio, in place.</summary>
        private static int mark(float[] window, double[] times, double from, int rate, double gain, bool rising)
        {
            int marked = 0;

            foreach (double time in times)
            {
                double at = (time - from) * rate / 1000.0;
                int start = (int)Math.Round(at);

                if (start < 0 || start >= window.Length)
                    continue;

                // Twenty milliseconds is short enough to read as a click and long enough to have a pitch, and a
                // Hann-shaped body keeps it from splattering across the spectrum the way a bare step does.
                int span = (int)(0.020 * rate);

                for (int i = 0; i < span && start + i < window.Length; i++)
                {
                    double envelope = 0.5 * (1 - Math.Cos(2 * Math.PI * i / span));
                    double tone = Math.Sin(2 * Math.PI * 1800 * i / rate);
                    double value = gain * envelope * tone * (rising ? 1 : -1);

                    window[start + i] = (float)Math.Clamp(window[start + i] + value, -1, 1);
                }

                marked++;
            }

            return marked;
        }

        /// <summary>Writes mono float samples as a 16-bit PCM wav.</summary>
        /// <remarks>
        /// Hand-written because there is no wav writer in the engine and this is the only thing that needs one. The
        /// header is the forty-four bytes every player expects; nothing here is ever read back by this code.
        /// </remarks>
        private static void writeWav(string path, float[] samples, int rate)
        {
            using var file = File.Create(path);
            using var writer = new BinaryWriter(file);

            int dataBytes = samples.Length * 2;

            writer.Write("RIFF"u8.ToArray());
            writer.Write(36 + dataBytes);
            writer.Write("WAVE"u8.ToArray());

            writer.Write("fmt "u8.ToArray());
            writer.Write(16);
            writer.Write((short)1);
            writer.Write((short)1);
            writer.Write(rate);
            writer.Write(rate * 2);
            writer.Write((short)2);
            writer.Write((short)16);

            writer.Write("data"u8.ToArray());
            writer.Write(dataBytes);

            foreach (float sample in samples)
                writer.Write((short)Math.Clamp(sample * short.MaxValue, short.MinValue, short.MaxValue));
        }

        /// <summary>
        /// The lag at which the model's activation lines up with the audio's own onsets.
        /// </summary>
        /// <remarks>
        /// The reading in this probe that does not trust the beatmap at all. Everything else here is a distance from a
        /// map's own timing points, which assumes the map is right both about where the music's beats are and about
        /// where the audio file starts - and a map can be wrong about the second by tens of milliseconds without anyone
        /// noticing, because the offset is stored in the map and the map is what a player's hit timing is judged
        /// against. A tracker that places beats thirty milliseconds early against a map that places them thirty
        /// milliseconds late is right about the music and would still read as thirty milliseconds out here.
        ///
        /// The spectral flux is computed on the same grid as the model - the same rate, the same hop, the same
        /// half-window centring - so a frame means the same instant in both, and the lag that best lines them up is
        /// the model's own displacement from the audio it was given.
        ///
        /// Read it as: a lag of zero means the model is where the sound is; a lag of minus one frame means the model
        /// reports each beat one frame - twenty milliseconds - before the sound that caused it, which is a frontend
        /// delay and not a disagreement about the beat.
        /// </remarks>
        [Test]
        public void MeasureTheModelsLagAgainstTheAudioItself()
        {
            string corpus = Environment.GetEnvironmentVariable("OSUTEST_CORPUS");

            if (string.IsNullOrEmpty(corpus) || !Directory.Exists(corpus))
                Assert.Ignore("Set OSUTEST_CORPUS to a folder holding .osu files and their audio.");

            string modelPath = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                               ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", $"{defaultModel}.onnx");

            if (!File.Exists(modelPath))
                Assert.Ignore($"no model at {modelPath}");

            string wanted = Environment.GetEnvironmentVariable("OSUTEST_MAP");
            string map = Directory.EnumerateFiles(corpus, "*.osu", SearchOption.AllDirectories)
                                  .FirstOrDefault(f => wanted == null
                                                       || Path.GetFileName(f).Contains(wanted, StringComparison.OrdinalIgnoreCase));

            if (map == null)
                Assert.Ignore($"no .osu file in {corpus} matching {wanted}");

            string folder = Path.GetDirectoryName(map);
            string audio = folder == null ? null : audioFor(map, folder);

            if (audio == null)
                Assert.Ignore($"no audio beside {map}");

            TestContext.Out.WriteLine($"beatmap: {Path.GetFileName(map)}");

            float[] samples = BassAudioDecoder.DecodeMono(audio, AnalysisAudio.SampleRate);
            (float[] activations, _) = BeatThisBeatTracker.Activations(samples, modelPath);

            double[] flux = OnsetEnvelope.FromSamples(samples, out int frameCount);

            TestContext.Out.WriteLine($"  {activations.Length} activation frames, {flux.Length} onset frames");

            // Both are treated as zero-mean and unit-variance so a lag is a correlation rather than whichever series
            // happens to have the larger numbers in it. The onset envelope is normalised to a mean of one and a beat
            // model's output is logits; neither is on the other's scale.
            // Both series repeat at the beat rate, so their correlation does too, and searching far enough in either
            // direction finds a second peak at one beat's displacement that is as tall as the first. A search wide
            // enough to reach it reports whichever of the two is a fraction larger, which is a phase the music does not
            // have - the first version of this searched twenty-five frames either way, found minus twenty-three, and
            // reported the model as half a second early. The window is therefore kept inside half a beat of the
            // fastest tempo in the corpus, which is the widest a displacement can be without being a different beat.
            double period = 60000 / 375;
            int limit = Math.Max(2, (int)Math.Floor(period / 2 / (1000.0 / BeatThisBeatTracker.FramesPerSecond)));

            TestContext.Out.WriteLine($"searching {limit} frames either way, which is half a beat at 375 BPM");

            double best = double.NegativeInfinity;
            int bestLag = 0;
            var scores = new List<(int Lag, double Score)>();

            for (int lag = -limit; lag <= limit; lag++)
            {
                double score = correlate(activations, flux, lag);

                scores.Add((lag, score));

                if (score > best)
                {
                    best = score;
                    bestLag = lag;
                }
            }

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine("correlation of the model's beat head with the audio's onset strength:");
            TestContext.Out.WriteLine("  lag      correlation");

            foreach ((int lag, double score) in scores)
                TestContext.Out.WriteLine($"  {lag,3}  {score,14:+0.0000;-0.0000; 0.0000}");

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine($"the best lag is {bestLag} frames, or {bestLag * 1000.0 / BeatThisBeatTracker.FramesPerSecond:+0;-0;0}ms");
            TestContext.Out.WriteLine(bestLag == 0
                ? "the model is where the sound is."
                : bestLag < 0
                    ? "the model reports each beat before the sound that caused it, which is a delay in the frontend."
                    : "the model reports each beat after the sound that caused it.");
        }

        /// <summary>Zero-mean unit-variance correlation of two series at a lag, over the frames they share.</summary>
        private static double correlate(float[] beats, double[] flux, int lag)
        {
            int first = Math.Max(0, -lag);
            int last = Math.Min(beats.Length, flux.Length - lag);

            if (last - first < 100)
                return double.NegativeInfinity;

            double sumA = 0;
            double sumB = 0;

            for (int i = first; i < last; i++)
            {
                sumA += beats[i];
                sumB += flux[i + lag];
            }

            double meanA = sumA / (last - first);
            double meanB = sumB / (last - first);

            double covariance = 0;
            double varianceA = 0;
            double varianceB = 0;

            for (int i = first; i < last; i++)
            {
                double a = beats[i] - meanA;
                double b = flux[i + lag] - meanB;

                covariance += a * b;
                varianceA += a * a;
                varianceB += b * b;
            }

            double denominator = Math.Sqrt(varianceA * varianceB);

            return denominator < 1e-9 ? 0 : covariance / denominator;
        }

        private static void report(string name, double[] beats, double[] grid)
        {
            double[] errors = beats.Select(b => Math.Abs(signedDistance(grid, b))).OrderBy(e => e).ToArray();
            double middle = medianInterval(beats);
            double median = errors.Length == 0 ? 0 : errors[errors.Length / 2];
            double p90 = errors.Length == 0 ? 0 : errors[Math.Min(errors.Length - 1, (int)(errors.Length * 0.9))];
            int covered = grid.Count(g => beats.Any(b => Math.Abs(b - g) <= 60));

            // The signed offsets, which is a different question from the distances above. Their middle is how far the
            // whole reading sits from the map, and it is invisible on screen: the nearest beat to a beat is the same
            // beat whether every reading is thirty milliseconds late or the map is. Their scatter about that middle is
            // what is left once the constant is taken out, and it is the part a player can actually see and hear.
            double[] signed = beats.Select(b => signedDistance(grid, b)).OrderBy(v => v).ToArray();
            double settled = signed.Length == 0 ? 0 : signed[signed.Length / 2];
            double low = signed.Length == 0 ? 0 : signed[Math.Min(signed.Length - 1, (int)(signed.Length * 0.25))];
            double high = signed.Length == 0 ? 0 : signed[Math.Min(signed.Length - 1, (int)(signed.Length * 0.75))];

            TestContext.Out.WriteLine($"  {name,-48} {beats.Length,6} {(middle > 0 ? $"{1000 / middle,6:0.0}" : "   n/a")} {median,7:0.0}ms {p90,7:0.0}ms {100.0 * covered / Math.Max(1, grid.Length),7:0.0}% {settled,8:+0;-0;0}ms {high - low,8:0}ms");
        }

        private static double medianInterval(double[] beats)
        {
            if (beats.Length < 3)
                return 0;

            double[] gaps = beats.Zip(beats.Skip(1), (a, b) => b - a).OrderBy(g => g).ToArray();

            return gaps[gaps.Length / 2];
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

        private static string audioFor(string map, string folder)
        {
            foreach (string line in File.ReadLines(map))
            {
                if (!line.StartsWith("AudioFilename", StringComparison.Ordinal))
                    continue;

                int colon = line.IndexOf(':');

                if (colon < 0)
                    continue;

                string name = line[(colon + 1)..].Trim();

                if (name.Length == 0)
                    continue;

                string candidate = Path.Combine(folder, name);

                if (File.Exists(candidate))
                    return candidate;

                string byName = Directory.EnumerateFiles(folder, "*", SearchOption.TopDirectoryOnly)
                                          .FirstOrDefault(f => string.Equals(Path.GetFileName(f), Path.GetFileName(name), StringComparison.OrdinalIgnoreCase));

                if (byName != null)
                    return byName;
            }

            return null;
        }

        /// <summary>
        /// The tempo points a map declares, in time order.
        /// </summary>
        /// <remarks>
        /// Only positive beat lengths are tempo. A negative one is an inherited point, which carries volume and sample
        /// changes and says nothing about the tempo; reading those as tempo is what makes a naive pass over a map see
        /// hundreds of changes in a track that holds one tempo.
        ///
        /// Section headers matter for the same reason. An object line under <c>[HitObjects]</c> is
        /// <c>x,y,time,type,...</c> and parses as a timing point with a time and a beat length without complaining, so
        /// a parse that does not stop at the next section header turns every hit object into a tempo change. This is
        /// the same reading <see cref="BeatmapCorpusProbe"/> uses, and the two are deliberately separate copies of it:
        /// it is short, and a shared helper would have to be public on a library that has no reason to know what an
        /// .osu file is.
        /// </remarks>
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

                if (length <= 0)
                    continue;

                points.Add(new TimingPoint(time, length));
            }

            return points.OrderBy(p => p.Time).ToList();
        }

        /// <summary>A map's own beats: each timing point's section stepped at its beat length, to the end.</summary>
        private static double[] beatmapGrid(List<TimingPoint> timing, double until)
        {
            var grid = new List<double>();

            for (int i = 0; i < timing.Count; i++)
            {
                // A section runs until the next timing point, and the last one runs to the end of the track.
                double end = i + 1 < timing.Count ? timing[i + 1].Time : Math.Max(until, timing[i].Time);

                for (double time = timing[i].Time; time < end && grid.Count < 500_000; time += timing[i].BeatLength)
                    grid.Add(time);
            }

            grid.Sort();

            return grid.ToArray();
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
