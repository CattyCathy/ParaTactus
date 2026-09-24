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

// BASS is started by the decoder itself on first use, with the no-sound device so that decoding does not need
// audio hardware. Nothing has to be initialised here.

if (Environment.GetEnvironmentVariable("PARATACTUS_DECODE_ONLY") == "1")
{
    Console.WriteLine("decoding only...");
    var watch = System.Diagnostics.Stopwatch.StartNew();
    float[] decoded = BassAudioDecoder.Default.DecodeMono(audioPath, AnalysisAudio.SampleRate);
    Console.WriteLine("decoded " + decoded.Length + " samples in " + watch.ElapsedMilliseconds + "ms");
    return 0;
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
