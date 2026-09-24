using System;
using System.IO;
using ManagedBass;
using NUnit.Framework;
using ParaTactus;

namespace ParaTactus.Tests
{
    /// <summary>
    /// Makes BASS usable from a test process, which has no game host to start it.
    /// </summary>
    /// <remarks>
    /// The decoder is the only part of BASS any of these tests wants, so the natives are made loadable from the
    /// package cache and a no-sound device is opened. Without this every test that decodes audio fails with
    /// <c>Errors.Init</c>, which reads like a missing codec rather than an unstarted audio system.
    /// </remarks>
    internal static class AudioTestEnvironment
    {
        public static void Initialise()
        {
            string natives = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".nuget", "packages", "ppy.osu.framework.nativelibs");

            if (Directory.Exists(natives))
            {
                foreach (string version in Directory.GetDirectories(natives))
                {
                    string source = Path.Combine(version, "runtimes", "win-x64", "native");

                    if (!Directory.Exists(source))
                        continue;

                    foreach (string file in new[] { "bass.dll", "bass_fx.dll" })
                    {
                        string from = Path.Combine(source, file);
                        string to = Path.Combine(AppContext.BaseDirectory, file);

                        if (File.Exists(from) && !File.Exists(to))
                        {
                            try
                            {
                                File.Copy(from, to);
                            }
                            catch (IOException)
                            {
                            }
                        }
                    }

                    break;
                }
            }

            if (!Bass.Init(Bass.NoSoundDevice, 44100, DeviceInitFlags.Default, IntPtr.Zero) && Bass.LastError != Errors.Already)
                Assert.Ignore($"BASS could not be initialised: {Bass.LastError}");
        }
    }
}
