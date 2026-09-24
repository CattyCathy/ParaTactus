// Compares two models on one decode of one track: do the fp32 original and the int8 build produce the same beats?
//
//     ModelCompare <audio> <fp32.onnx> <int8.onnx>

using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using ParaTactus;
using ParaTactus.Decoding;

// Auto-flushed, so a run that hangs still shows how far it got.
Console.SetOut(new StreamWriter(Console.OpenStandardOutput()) { AutoFlush = true });

if (args.Length < 3)
{
    Console.WriteLine("usage: ModelCompare <audio> <fp32.onnx> <int8.onnx>");

    return 1;
}

string audioPath = args[0];
string fp32Model = args[1];
string int8Model = args[2];

var watch = Stopwatch.StartNew();
float[] samples = BassAudioDecoder.Default.DecodeMono(audioPath, AnalysisAudio.SampleRate);

Console.WriteLine($"decoded {samples.Length} samples ({samples.Length / (double)AnalysisAudio.SampleRate:0.#}s) in {watch.ElapsedMilliseconds}ms");
Console.WriteLine();

double[] beatsOf(string label, string model)
{
    watch.Restart();
    double[] beats = BeatThisBeatTracker.BeatTimes(samples, model);

    // A beat at the very end is as long as the audio, which is not a useful duration for a comparison of grids; the
    // count and the last time are what matter here.
    Console.WriteLine($"{label}: {beats.Length} beats, first {beats[0]:0}ms, last {beats[^1]:0}ms, {watch.ElapsedMilliseconds}ms");

    return beats;
}

double median(double[] values)
{
    double[] sorted = values.OrderBy(v => v).ToArray();

    return sorted.Length == 0 ? 0 : sorted[sorted.Length / 2];
}

void compare(string stage, double[] left, double[] right, double tolerance)
{
    Console.WriteLine();
    Console.WriteLine($"--- {stage} ---");
    Console.WriteLine($"counts: {left.Length} vs {right.Length}");

    if (left.Length == right.Length)
    {
        double[] deltas = left.Zip(right, (a, b) => Math.Abs(a - b)).ToArray();

        Console.WriteLine($"position by index: max {deltas.Max():0.#}ms, median {median(deltas):0.#}ms, mean {deltas.Average():0.#}ms, >{tolerance:0}ms: {deltas.Count(d => d > tolerance)}");
    }

    // Nearest match, so that a beat inserted or dropped early does not shift every later index and report the whole
    // rest of the track as different.
    double[] missing = left.Where(a => right.Min(b => Math.Abs(a - b)) > tolerance).ToArray();
    double[] extra = right.Where(b => left.Min(a => Math.Abs(a - b)) > tolerance).ToArray();

    Console.WriteLine($"only in the first  (no partner within {tolerance:0}ms): {missing.Length}");
    Console.WriteLine($"only in the second (no partner within {tolerance:0}ms): {extra.Length}");

    foreach (double time in missing.Take(6))
        Console.WriteLine($"    first only {time:0}ms -> nearest {right.Min(b => Math.Abs(time - b)):0}ms away");

    foreach (double time in extra.Take(6))
        Console.WriteLine($"    second only {time:0}ms -> nearest {left.Min(a => Math.Abs(a - time)):0}ms away");
}

double[] rawFp32 = beatsOf("fp32 raw", fp32Model);
double[] rawInt8 = beatsOf("int8 raw", int8Model);

compare("raw model output", rawFp32, rawInt8, BeatThisBeatTracker.FrameToMilliseconds(1));

// The production pipeline: the app animates against the regularised grid, not the raw output.
double[] regularFp32 = BeatTrainRegulariser.Regularise(rawFp32);
double[] regularInt8 = BeatTrainRegulariser.Regularise(rawInt8);

Console.WriteLine();
Console.WriteLine($"regularised: fp32 {regularFp32.Length} beats, int8 {regularInt8.Length} beats");

compare("regularised grid", regularFp32, regularInt8, BeatThisBeatTracker.FrameToMilliseconds(1));

// What the UI actually reads: the tempo reported at each beat.
BeatGrid gridFp32 = BeatGrid.FromBeats(regularFp32);
BeatGrid gridInt8 = BeatGrid.FromBeats(regularInt8);

double[] tempoDeltas = regularFp32
    .Where(t => t <= regularInt8[^1])
    .Select(t => Math.Abs(gridFp32.BpmAt(t) - gridInt8.BpmAt(t)))
    .ToArray();

Console.WriteLine();
Console.WriteLine($"BpmAt over the beats the two grids share: max {tempoDeltas.Max():0.###} BPM, median {median(tempoDeltas):0.###}, >0.5 BPM: {tempoDeltas.Count(d => d > 0.5)} of {tempoDeltas.Length}");

return 0;
