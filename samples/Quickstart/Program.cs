using System;
using System.IO;
using ParaTactus;
using ParaTactus.Decoding;

// The smallest useful thing this library does: analyse a track and print its beats with the tempo reported at each one.
//
//     dotnet run --project samples/Quickstart -- beat-this-final0.onnx track.mp3
//
// A UI does the same thing and then animates against the grid: the times in grid.Beats are where a pulse belongs, and
// grid.BpmAt(time) is the tempo to report. The two are deliberately separate, because the reported tempo is smoothed
// over a window and the beat instants are not 鈥?animating from the smoothed value is what makes visuals stutter.

if (args.Length < 2)
{
    Console.WriteLine("usage: Quickstart <model.onnx> <audio file>");
    Console.WriteLine();
    Console.WriteLine("The model is not distributed with this library; see docs/model.md.");

    return 1;
}

string modelPath = args[0];
string audioPath = args[1];

if (!File.Exists(modelPath))
{
    Console.WriteLine($"no model at {modelPath}");
    return 1;
}

if (!File.Exists(audioPath))
{
    Console.WriteLine($"no audio at {audioPath}");
    return 1;
}

// The provider analyses on a miss and keeps the result on disk. Analysis costs roughly a quarter of the track's length
// the first time and nothing afterwards. Passing a decoder is what lets it read from a path; without one it can still
// serve the cache and run the streaming tracker.
string cacheDirectory = Path.Combine(Path.GetTempPath(), "paratactus-quickstart");

// BASS has to be started before it can decode anything, and nothing in this library does that for the caller: an
// application built on osu.Framework gets it from the framework, and a standalone consumer does not. The no-sound
// device is enough to decode without touching audio hardware. (The adapter should be doing this itself; it is written
// out here so the requirement is visible rather than discovered as Errors.Init.)
if (!ManagedBass.Bass.Init(ManagedBass.Bass.NoSoundDevice, 44100, ManagedBass.DeviceInitFlags.Default, IntPtr.Zero)
    && ManagedBass.Bass.LastError != ManagedBass.Errors.Already)
{
    Console.WriteLine($"BASS could not be started: {ManagedBass.Bass.LastError}");
    return 1;
}

var provider = new BeatGridProvider(modelPath, cacheDirectory, BassAudioDecoder.Default);

Console.WriteLine($"analysing {Path.GetFileName(audioPath)}...");

BeatGrid grid = provider.Get(audioPath);

Console.WriteLine($"{grid.Beats.Count} beats over {grid.Duration / 1000:0.#}s");
Console.WriteLine();
Console.WriteLine("    time     tempo   gap to next");

for (int i = 0; i < grid.Beats.Count; i++)
{
    double beat = grid.Beats[i];
    double gap = i + 1 < grid.Beats.Count ? grid.Beats[i + 1] - beat : 0;

    Console.WriteLine($"{beat / 1000,8:0.000}s {grid.BpmAt(beat),6:0.#}    {gap,5:0}ms");
}

return 0;
