using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using ParaTactus;

namespace ParaTactus
{
    /// <summary>
    /// Stores analysed beat grids on disk so a track is only paid for once.
    /// </summary>
    /// <remarks>
    /// Analysis costs about a quarter of the track's own length even with the quantised model, so re-doing it on every
    /// play is not an option and neither is holding a library's worth of grids in memory. What is cached is the answer,
    /// not the audio: one line of beat times per track.
    ///
    /// Anything unreadable is treated as a miss rather than an error. A cache that can stop a track from playing is
    /// worse than no cache, and the format will change again before this is finished.
    /// </remarks>
    public sealed class BeatGridCache
    {
        private const string header = "osutest-beatgrid 1";

        /// <summary>
        /// Identifies the analysis that produced a grid, so that changing it invalidates what is already stored.
        /// </summary>
        /// <remarks>
        /// This exists because its absence cost hours. The key used to be derived from the audio and the model alone,
        /// which cannot distinguish a grid produced by one version of the analysis from a grid produced by another. So
        /// every correction to the frontend, the chunk schedule, the pruning and the phase refinement was invisible in
        /// the player for any track that had already been analysed once: the code was fixed, the tests went green, and
        /// the application kept serving the same stale beats from disk. The tracks played earliest stayed wrong, which
        /// is exactly the pattern being reported and could not be reproduced from the tests, because the tests call the
        /// analysis directly and never touch the cache.
        ///
        /// It is the build identity of this assembly rather than a constant someone has to remember to bump. A change
        /// to the beats a track produces needs a rebuild, so the identity changes exactly when the cached answer stops
        /// being valid, and a rebuild that changes nothing leaves it alone and keeps the cache. The alternative was
        /// tried in spirit and failed: any hand-maintained version is only correct as long as whoever changes the
        /// analysis also remembers it exists.
        /// </remarks>
        private static readonly string analysis_version =
            typeof(BeatGridCache).Assembly.ManifestModule.ModuleVersionId.ToString("N");

        private readonly string directory;

        public BeatGridCache(string directory)
        {
            if (string.IsNullOrEmpty(directory))
                throw new ArgumentException("A cache directory is needed.", nameof(directory));

            this.directory = directory;
        }

        /// <summary>The directory the grids are stored in.</summary>
        public string Directory => directory;

        /// <summary>
        /// Identifies a track, the model that analysed it and the analysis that did the work, so that changing any of
        /// the three misses the cache rather than serving beats that no longer describe the audio.
        /// </summary>
        public static string KeyFor(string audioPath, string modelPath)
        {
            var identity = new StringBuilder();

            identity.Append(analysis_version).Append('\n');

            foreach (string path in new[] { audioPath, modelPath })
            {
                var file = new FileInfo(path);

                identity.Append(file.FullName).Append('|')
                        .Append(file.Exists ? file.Length : -1).Append('|')
                        .Append(file.Exists ? file.LastWriteTimeUtc.Ticks : 0).Append('\n');
            }

            byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(identity.ToString()));
            var text = new StringBuilder(32);

            for (int i = 0; i < 16; i++)
                text.Append(hash[i].ToString("x2", CultureInfo.InvariantCulture));

            return text.ToString();
        }

        /// <summary>Reads a grid, or reports that there is not a usable one.</summary>
        public bool TryLoad(string key, out BeatGrid grid)
        {
            grid = null;

            string path = PathFor(key);

            if (!File.Exists(path))
                return false;

            try
            {
                string[] lines = File.ReadAllLines(path);

                if (lines.Length < 3 || lines[0].Trim() != header)
                    return false;

                if (!int.TryParse(lines[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int shift))
                    return false;

                var beats = new List<double>();

                if (lines[2].Length > 0)
                {
                    foreach (string field in lines[2].Split(','))
                    {
                        if (!double.TryParse(field, NumberStyles.Float, CultureInfo.InvariantCulture, out double beat))
                            return false;

                        beats.Add(beat);
                    }
                }

                grid = BeatGrid.FromNormalisedBeats(beats, shift);

                return true;
            }
            catch (IOException)
            {
                return false;
            }
            catch (UnauthorizedAccessException)
            {
                return false;
            }
        }

        /// <summary>Writes a grid, replacing any previous one for the same key.</summary>
        public void Store(string key, BeatGrid grid)
        {
            System.IO.Directory.CreateDirectory(directory);

            var text = new StringBuilder();
            text.Append(header).Append('\n');
            text.Append(grid.MetricalShift.ToString(CultureInfo.InvariantCulture)).Append('\n');

            for (int i = 0; i < grid.Beats.Count; i++)
            {
                if (i > 0)
                    text.Append(',');

                text.Append(grid.Beats[i].ToString("0.###", CultureInfo.InvariantCulture));
            }

            text.Append('\n');

            File.WriteAllText(PathFor(key), text.ToString());
        }

        private string PathFor(string key)
        {
            return Path.Combine(directory, key + ".beats");
        }
    }
}
