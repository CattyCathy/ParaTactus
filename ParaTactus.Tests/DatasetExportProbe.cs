using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using NUnit.Framework;
using ParaTactus;
using ParaTactus.Decoding;

namespace ParaTactus.Tests
{
    /// <summary>
    /// Turns a corpus of osu! beatmapsets into audio and labels a detector can be trained on.
    /// </summary>
    /// <remarks>
    /// What is written is the decoded mono audio and a per-frame label, and deliberately not any features. Features are
    /// the part most likely to be changed - a log-mel, an onset envelope, both, at what resolution - and writing them
    /// here would mean re-running this exporter for every experiment. The audio is the part that costs, because
    /// decoding a corpus takes minutes, so it is written once and read many times.
    ///
    /// The labels come from a beatmap's uninherited timing points, which are the grid the mapper snapped the objects
    /// to and therefore the author's own statement of where the beats are. Inherited points are the slider and scroll
    /// speed and say nothing about the beat. Hit objects are not used for labels because a dense map places them on
    /// subdivisions: on this corpus the commonest gap between consecutive objects is 80ms on maps whose beat is 300.
    ///
    /// One thing the timing points cannot say is which octave of themselves is the beat. A map's grid is a snap
    /// resolution and a dense map snaps to a quarter of the beat, so the labels are built at the octave the model
    /// itself reads, decided once per track by which octave's beats the activation separates from the points between
    /// them. That is recorded per frame as well, because it is the thing a detector is being trained to get right and a
    /// corpus where it varies is the corpus worth having.
    ///
    /// Every grid point inside the audio is labelled, without exception: a label file with holes in it teaches a
    /// detector to skip beats, and a skipped beat is a pulse the player cannot follow. The downbeats are written as
    /// their own class, which the exporter used to leave unused - the first beat of every measure, from each timing
    /// point's own phase, so that a reading which wants to know where the bar is has something to read.
    ///
    /// Set <c>OSUTEST_CORPUS</c> to a folder of .osz files, and <c>OSUTEST_DATASET</c> for where to write.
    /// </remarks>
    [Explicit("Needs OSUTEST_CORPUS pointing at a folder of .osz files, and the model.")]
    public class DatasetExportProbe
    {
        private const string defaultModel = "beat-this-final0-int8";

        [OneTimeSetUp]
        public void SetUp() => AudioTestEnvironment.Initialise();

        [Test]
        public void ExportAudioAndLabelsForTraining()
        {
            string corpus = Environment.GetEnvironmentVariable("OSUTEST_CORPUS");
            string output = Environment.GetEnvironmentVariable("OSUTEST_DATASET") ?? @"D:\Linux\Proj\OsuTest\dataset";

            if (string.IsNullOrEmpty(corpus) || !Directory.Exists(corpus))
                Assert.Ignore("Set OSUTEST_CORPUS to a folder of .osz files.");

            string modelPath = Environment.GetEnvironmentVariable("OSUTEST_MODEL")
                               ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", $"{defaultModel}.onnx");

            if (!File.Exists(modelPath))
                Assert.Ignore($"no model at {modelPath}");

            var sets = Directory.EnumerateFiles(corpus, "*.osz", SearchOption.AllDirectories).OrderBy(f => f).ToList();

            // A single set, so that one track's labels can be looked at without paying for the corpus. The id is what
            // the manifest and every other tool keys on, which is the part of the file name before the first space.
            string only = Environment.GetEnvironmentVariable("OSUTEST_ONLY");

            if (!string.IsNullOrEmpty(only))
                sets = sets.Where(f => Path.GetFileNameWithoutExtension(f).Split(' ')[0] == only).ToList();

            // A shard, so several exports can run at once. The work here is almost entirely the model's inference - 363
            // of 370 seconds for one four-minute track, measured - and that is already using every core the runtime
            // will give it, so the only way to go faster is to run several tracks at a time. This process's share is
            // chosen by taking the sets at a stride rather than a contiguous block, so the shards get the same mix of
            // long and short tracks.
            string shard = Environment.GetEnvironmentVariable("OSUTEST_SHARD");

            if (!string.IsNullOrEmpty(shard) && shard.Contains('/'))
            {
                string[] parts = shard.Split('/');
                int index = int.Parse(parts[0], CultureInfo.InvariantCulture);
                int of = int.Parse(parts[1], CultureInfo.InvariantCulture);

                sets = sets.Where((_, i) => i % of == index).ToList();

                TestContext.Out.WriteLine($"shard {index} of {of}");
            }

            if (sets.Count == 0)
                Assert.Ignore($"no .osz file in {corpus}");

            Directory.CreateDirectory(output);

            string audioDirectory = Path.Combine(output, "audio");
            string labelDirectory = Path.Combine(output, "labels");

            Directory.CreateDirectory(audioDirectory);
            Directory.CreateDirectory(labelDirectory);

            TestContext.Out.WriteLine($"{sets.Count} beatmapsets in {corpus}");
            TestContext.Out.WriteLine($"writing to {output}");
            TestContext.Out.WriteLine("");

            var manifest = new StringBuilder();
            manifest.AppendLine("id\tset\taudio\tlabels\tframes\tbeats\tperiod\tlag\toctave\tusable\tseparations");

            int exported = 0;
            int skipped = 0;
            var failures = new List<string>();

            foreach (string set in sets)
            {
                string id = Path.GetFileNameWithoutExtension(set).Split(' ')[0];
                string audioPath = Path.Combine(audioDirectory, id + ".f32");
                string labelPath = Path.Combine(labelDirectory, id + ".i8");

                // A single set that was asked for by name is re-exported even when its files exist, because the reason
                // to ask for one is to look at what a change did to it.
                if (string.IsNullOrEmpty(only) && File.Exists(audioPath) && File.Exists(labelPath))
                {
                    skipped++;
                    continue;
                }

                try
                {
                    Row row = export(set, id, audioPath, labelPath, modelPath);
                    manifest.AppendLine(row.ToLine(id, audioPath, labelPath) + "\t" + row.Separations);
                    exported++;

                    TestContext.Out.WriteLine($"  [{exported,4}] {Path.GetFileName(set),-52} {row.Frames,7} frames  {row.Beats,6} beats  period {row.Period,4:0}ms  {(row.Usable ? "usable" : "not usable")}");
                }
                catch (Exception error)
                {
                    failures.Add($"{Path.GetFileName(set)}: {error.Message}");

                    // Half-written pairs are removed, or a resumed run would treat a truncated label file as done.
                    foreach (string path in new[] { audioPath, labelPath })
                    {
                        if (File.Exists(path))
                            File.Delete(path);
                    }
                }
            }

            File.WriteAllText(Path.Combine(output, "manifest.tsv"), manifest.ToString());

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine($"exported {exported}, already present {skipped}, failed {failures.Count}");

            foreach (string failure in failures.Take(20))
                TestContext.Out.WriteLine($"  {failure}");

            TestContext.Out.WriteLine("");
            TestContext.Out.WriteLine($"manifest: {Path.Combine(output, "manifest.tsv")}");
        }

        private readonly struct Row
        {
            public Row(int frames, int beats, double period, double lag, int octave, bool usable, string separations)
            {
                Frames = frames;
                Beats = beats;
                Period = period;
                Lag = lag;
                Octave = octave;
                Usable = usable;
                Separations = separations;
            }

            public int Frames { get; }
            public int Beats { get; }
            public double Period { get; }
            public double Lag { get; }
            public int Octave { get; }
            public bool Usable { get; }

            /// <summary>What the model scored each octave at, so that a decision can be judged without a second run.</summary>
            public string Separations { get; }

            public string ToLine(string id, string audioPath, string labelPath)
                => string.Join('\t', id, id, Path.GetFileName(audioPath), Path.GetFileName(labelPath),
                               Frames.ToString(CultureInfo.InvariantCulture),
                               Beats.ToString(CultureInfo.InvariantCulture),
                               Period.ToString("0", CultureInfo.InvariantCulture),
                               Lag.ToString("0", CultureInfo.InvariantCulture),
                               Octave.ToString(CultureInfo.InvariantCulture),
                               Usable ? "1" : "0");
        }

        private static Row export(string set, string id, string audioPath, string labelPath, string modelPath)
        {
            string scratch = Path.Combine(Path.GetTempPath(), "paratactus-set-" + id);
            var clock = System.Diagnostics.Stopwatch.StartNew();
            double unzip = 0, decode = 0, infer = 0, label = 0;

            try
            {
                if (Directory.Exists(scratch))
                    Directory.Delete(scratch, true);

                Directory.CreateDirectory(scratch);
                ZipFile.ExtractToDirectory(set, scratch);
                unzip = clock.Elapsed.TotalSeconds;

                string map = Directory.EnumerateFiles(scratch, "*.osu", SearchOption.AllDirectories).FirstOrDefault()
                             ?? throw new InvalidOperationException("no .osu in the set");

                string audio = audioFor(map, Path.GetDirectoryName(map))
                               ?? throw new InvalidOperationException("no audio in the set");

                List<(double Time, double BeatLength)> timing = uninherited(map);

                if (timing.Count == 0)
                    throw new InvalidOperationException("no uninherited timing points");

                float[] samples = BassAudioDecoder.DecodeMono(audio, AnalysisAudio.SampleRate);
                decode = clock.Elapsed.TotalSeconds;

                if (samples.Length < AnalysisAudio.SampleRate * 10)
                    throw new InvalidOperationException("under ten seconds of audio");

                // The model's own activation is read for the octave decision and for the lag, so it is run here rather
                // than left to the training side. It is the one part of the export that needs the model, and it is what
                // makes the labels unambiguous.
                //
                // Two threads, and that is a measurement rather than a guess. This model gets slower the more threads it
                // is given: on a four-minute track the same analysis takes 24.5 seconds on two threads, 55.5 on eight
                // and 113.5 on sixteen, so the operators' synchronisation costs more than the parallelism returns. The
                // playing path asks for every processor but two, which is right when something is waiting on it and
                // four to five times wrong here.
                int threads = int.TryParse(Environment.GetEnvironmentVariable("OSUTEST_THREADS"), out int wanted) && wanted > 0
                    ? wanted
                    : 2;

                using var tracker = new StreamingBeatTracker(modelPath, threads);
                tracker.Add(samples);
                tracker.Flush();

                float[] activation = tracker.Activation.ToArray();
                infer = clock.Elapsed.TotalSeconds;
                int frames = activation.Length;

                if (frames < 100)
                    throw new InvalidOperationException("too few frames");

                double until = frames * 1000.0 / BeatThisBeatTracker.FramesPerSecond;

                double period = dominantBeatLength(timing);

                // The lag between this map's grid and the model, measured once for the whole track. It is reported
                // rather than applied: a map's timing points are placed by hand against the audio and are routinely
                // tens of milliseconds from where the music is, so a shift here is a fact about the map as much as
                // about the model, and the training side is the place to decide what to do with it.
                int lag = bestLag(activation, gridFor(timing, until, 1));

                var labels = new sbyte[frames];
                var octaves = new sbyte[frames];

                // The octave is chosen once for the whole track and every grid point inside the audio is labelled.
                //
                // It used to be chosen per window with the windows that the model could not separate a beat from the
                // space between two of them skipped entirely, and that was wrong in a way that only showed on the
                // material this exists for. Measured on one variable-tempo track, nine of its windows lost every one of
                // their labels and seven of them lost a hundred per cent, so twenty-one seconds in a row had no beats
                // at all; the labels came out at 69% of the map's beats over the track and the detector, faithfully
                // trained on them, reproduced 68%. A detector that skips beats is worse than useless to a rhythm game,
                // because the player is following its pulse and it is the pulse that has to be steady - so coverage of
                // the map's own grid is the one property the labels cannot be allowed to trade away.
                //
                // What the gate was protecting against is real and is kept: the octave is still decided by which one
                // the model separates best, which is measured below and reported. It is simply decided once, where a
                // track's grid is overwhelmingly at one level - measured across three tracks, the per-window choice
                // never once changed - and the decision is then not allowed to cost a beat.
                (int octave, double separation, string scores) = chooseOctave(activation, timing, lag, until);

                // A measure is four beats, so the downbeat is every fourth labelled beat. The phase is the timing
                // point's own, which is where a mapper puts the measure's first beat, and it restarts at each point
                // because a mapper who has moved the grid has usually moved it to a measure.
                int beatsToABar = beatsPerBar(timing);
                int beatInBar = 0;
                int beats = 0;
                int downbeats = 0;

                double scale = scaleFor(octave);

                for (int i = 0; i < timing.Count; i++)
                {
                    double length = timing[i].BeatLength * scale;

                    if (length <= 5)
                        continue;

                    double end = i + 1 < timing.Count ? timing[i + 1].Time : Math.Max(until, timing[i].Time);

                    for (double time = timing[i].Time; time < end && time <= until; time += length)
                    {
                        // Rounded to the nearest frame rather than truncated, so that the label sits where the grid
                        // does rather than up to a frame early. At fifty frames a second that is the difference
                        // between a twenty millisecond error and none.
                        int frame = (int)Math.Round(time * BeatThisBeatTracker.FramesPerSecond / 1000.0);

                        if (frame < 0 || frame >= frames || labels[frame] != 0)
                            continue;

                        labels[frame] = beatInBar == 0 ? (sbyte)2 : (sbyte)1;

                        if (beatInBar == 0)
                            downbeats++;

                        octaves[frame] = (sbyte)octave;
                        beats++;
                        beatInBar = (beatInBar + 1) % beatsToABar;
                    }
                }

                label = clock.Elapsed.TotalSeconds;

                WriteAudio(audioPath, samples);
                File.WriteAllBytes(labelPath, labels.Select(v => (byte)v).ToArray());
                File.WriteAllBytes(Path.ChangeExtension(labelPath, ".oct"), octaves.Select(v => (byte)v).ToArray());

                TestContext.Out.WriteLine($"        timing: unzip {unzip:0.0}s  decode {decode - unzip:0.0}s  infer {infer - decode:0.0}s  label {label - infer:0.0}s  write {clock.Elapsed.TotalSeconds - label:0.0}s");
                TestContext.Out.WriteLine($"        octave {octave} (separation {separation:0.00}) from {scores}, {beats} beats of which {downbeats} downbeats over {until / 1000:0}s");

                return new Row(frames, beats, period, lag * 1000.0 / BeatThisBeatTracker.FramesPerSecond, octave, separation > 0, scores);
            }
            finally
            {
                if (Directory.Exists(scratch))
                    Directory.Delete(scratch, true);
            }
        }

        /// <summary>
        /// How much better than the map's own octave another octave has to score before the labels move to it.
        /// </summary>
        /// <remarks>
        /// The model can be confidently wrong about the level, and on this corpus it is. Between the two exports, the
        /// per-track decision moved fourteen of a hundred and thirteen tracks an octave - thirteen of them down, one
        /// up - and on the one track that has been checked against its own hit objects the move was wrong and cost the
        /// detector a third of its beats there: it reports 0.60 of a tempo-variable map's beats where the model trained
        /// on the unchanged level reports 1.01.
        ///
        /// So the map's own grid is the default and the model only overrides it decisively. The margin is on the ratio
        /// rather than on the difference, because the scores are not on a common scale - they run from about 2 to about
        /// 18 across the corpus - and because the interval is required to straddle one, a track where both levels score
        /// near zero never moves.
        /// </remarks>
        private const double octave_override_ratio = 1.25;

        /// <summary>The octave of a map's grid the model's confidence agrees with, chosen once for the whole track.</summary>
        /// <remarks>
        /// A map's timing points are a snap resolution as much as a tempo, and a dense map snaps to a quarter of the
        /// beat, so the level a player taps can be an octave or two above what the map says. The model was trained on
        /// the level a person taps, so the octave it reads is the one the labels belong at, and asking it is more
        /// reliable than guessing from the beat length.
        ///
        /// Once for the whole track rather than per window. A window is a tenth of a second of a track's grid and a
        /// decision made from one is a decision made from very little: measured across three tracks the per-window
        /// choice never changed, so the extra resolution was never buying anything, and what it cost was that the
        /// windows the decision was unsure about were dropped along with every label in them. One decision for a track
        /// that holds one level is both steadier and free of that.
        ///
        /// The measure is how much of the grid the model reports a beat on, against how much of it falls between two
        /// of its beats, both as a difference from the track's own mean. The difference rather than the level, because a
        /// grid an octave too fine puts beats where the music has none and that shows up as a small difference rather
        /// than as a small value.
        /// </remarks>
        private static (int Octave, double Separation, string Scores) chooseOctave(float[] activation, List<(double Time, double BeatLength)> timing, int lag, double until)
        {
            double mean = 0;

            foreach (float value in activation)
                mean += value;

            mean /= Math.Max(1, activation.Length);

            int best = 0;
            double bestSeparation = double.NegativeInfinity;
            double ownSeparation = double.NaN;
            var scores = new System.Text.StringBuilder();

            for (int octave = -2; octave <= 2; octave++)
            {
                double[] grid = gridFor(timing, until, scaleFor(octave));
                var at = new List<double>();
                var off = new List<double>();

                for (int i = 0; i < grid.Length; i++)
                    at.Add(activationAt(activation, grid[i], lag) - mean);

                // The points halfway between the grid's own, which is where a grid an octave too fine puts its own
                // beats: it is the same set of instants seen from the other side, so an octave whose midpoints are as
                // confident as its beats is an octave with beats where the music has none.
                for (int i = 0; i + 1 < grid.Length; i++)
                    off.Add(activationAt(activation, (grid[i] + grid[i + 1]) / 2, lag) - mean);

                if (at.Count < 8)
                    continue;

                at.Sort();
                off.Sort();

                double separation = at[at.Count / 2] - (off.Count == 0 ? 0 : off[off.Count / 2]);

                // Reported for every octave rather than only the winner, because whether the winner won by a lot or by
                // a hair is the whole question when a decision is being judged.
                scores.Append($" {octave}:{separation:0.000}");

                if (octave == 0)
                    ownSeparation = separation;

                if (separation > bestSeparation)
                {
                    bestSeparation = separation;
                    best = octave;
                }
            }

            // The model's preference only wins if it is decisive. A track whose grid the model separates equally well
            // at two levels keeps the one its mapper wrote, which is the level the objects were placed against.
            //
            // And it never wins at all when the caller has said the map is the reference. That is a choice about what
            // the labels are for rather than about which is more musical, and the two genuinely differ: on this corpus
            // the hit objects of one track sit a quarter of the way through the map's declared beat, so the declared
            // tempo is a snap grid and the notes are four times denser than it - and on another they sit two and a
            // third times the declared beat. A detector is being built to agree with the file, so the file decides.
            bool mapIsReference = Environment.GetEnvironmentVariable("OSUTEST_MAP_LEVEL") == "1";

            if (best != 0 && !double.IsNaN(ownSeparation))
            {
                bool decisive = !mapIsReference
                                && bestSeparation > 0
                                && bestSeparation > ownSeparation * octave_override_ratio
                                && bestSeparation > ownSeparation + 1e-9;

                if (!decisive)
                {
                    scores.Append(mapIsReference
                        ? $"  (forced to 0, the map's level: {best} scored {bestSeparation:0.000} against {ownSeparation:0.000})"
                        : $"  (kept 0 over {best}: {bestSeparation:0.000} against {ownSeparation:0.000})");

                    best = 0;
                    bestSeparation = ownSeparation;
                }
            }

            return (best, bestSeparation, scores.ToString().Trim());
        }

        /// <summary>The multiplier that puts a map's grid at an octave of itself.</summary>
        private static double scaleFor(int octave) => Math.Pow(2, octave);

        /// <summary>
        /// How many of a map's beats make a measure, from the beat length's own whole part.
        /// </summary>
        /// <remarks>
        /// A beat of exactly a second is four beats to a bar and is the only case the format states, since a beat
        /// length is written in milliseconds and a mapper who writes 1000 is writing 60 beats a minute four to the bar.
        /// Every other length is the mapper's own convenience, so four is assumed and the whole part is only used when
        /// it is plausible: a third of a second is a third of a bar, which rounds to nothing, and a bar of no beats
        /// would put every beat on a downbeat.
        /// </remarks>
        private static int beatsPerBar(List<(double Time, double BeatLength)> timing)
        {
            double length = dominantBeatLength(timing);
            int beats = (int)Math.Round(4.0 * length / 1000.0);

            return beats is >= 2 and <= 16 ? beats : 4;
        }

        private static double activationAt(float[] activation, double milliseconds, int shift)
        {
            int frame = (int)Math.Round(milliseconds * BeatThisBeatTracker.FramesPerSecond / 1000.0) + shift;

            return frame < 0 || frame >= activation.Length ? 0 : activation[frame];
        }

        /// <summary>The shift at which a grid and the activation line up best, within a tenth of a second.</summary>
        private static int bestLag(float[] activation, double[] grid)
        {
            double mean = 0;

            foreach (float value in activation)
                mean += value;

            mean /= Math.Max(1, activation.Length);

            int best = 0;
            double bestTotal = double.NegativeInfinity;

            for (int lag = -8; lag <= 8; lag++)
            {
                double total = 0;

                foreach (double beat in grid)
                    total += activationAt(activation, beat, lag) - mean;

                if (total > bestTotal)
                {
                    bestTotal = total;
                    best = lag;
                }
            }

            return best;
        }

        /// <summary>The beat length a map's timing points spend most of their length at.</summary>
        private static double dominantBeatLength(List<(double Time, double BeatLength)> timing)
        {
            var weight = new Dictionary<double, double>();

            for (int i = 0; i < timing.Count; i++)
            {
                double end = i + 1 < timing.Count ? timing[i + 1].Time : timing[i].Time;
                double length = Math.Round(timing[i].BeatLength);

                weight[length] = weight.TryGetValue(length, out double seen) ? seen + (end - timing[i].Time) : end - timing[i].Time;
            }

            return weight.OrderByDescending(pair => pair.Value).First().Key;
        }

        private static double[] gridFor(List<(double Time, double BeatLength)> timing, double until, double scale)
        {
            var grid = new List<double>();

            for (int i = 0; i < timing.Count; i++)
            {
                double length = timing[i].BeatLength * scale;

                if (length <= 5)
                    continue;

                double end = i + 1 < timing.Count ? timing[i + 1].Time : Math.Max(until, timing[i].Time);

                for (double time = timing[i].Time; time < end && grid.Count < 5_000_000; time += length)
                    grid.Add(time);
            }

            grid.Sort();

            return grid.ToArray();
        }

        private static List<(double Time, double BeatLength)> uninherited(string map)
        {
            var points = new List<(double, double)>();
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

                if (fields.Length >= 2
                    && double.TryParse(fields[0], NumberStyles.Float, CultureInfo.InvariantCulture, out double time)
                    && double.TryParse(fields[1], NumberStyles.Float, CultureInfo.InvariantCulture, out double length)
                    && length > 0)
                {
                    points.Add((time, length));
                }
            }

            points.Sort((a, b) => a.Item1.CompareTo(b.Item1));

            return points;
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

        /// <summary>Writes mono float samples as raw little-endian float32, which is what a numpy reader wants.</summary>
        private static void WriteAudio(string path, float[] samples)
        {
            using var file = File.Create(path);
            using var writer = new BinaryWriter(file);
            var buffer = new byte[samples.Length * 4];

            Buffer.BlockCopy(samples, 0, buffer, 0, buffer.Length);
            writer.Write(buffer);
        }
    }
}
