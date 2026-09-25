# ParaTactus.NET

Beat tracking for audio, in C#. It answers one question — *where are the beats in this track, including where the
tempo changes* — and hands back a grid a UI can animate against.

It is a port of [Beat This!](https://github.com/CPJKU/beat_this) (Foscarin, Schlüter and Widmer, ISMIR 2024): the
log-mel frontend, the chunked transformer inference and the peak-picking postprocessor are implemented here from the
published reference rather than called through it.

| Package | Project | What it is |
| --- | --- | --- |
| `ParaTactus.NET` | [`ParaTactus/`](ParaTactus) | the analysis. Three MIT dependencies and nothing else. |
| `ParaTactus.Bass` | [`ParaTactus.Bass/`](ParaTactus.Bass) | optional. Decodes a path for you, with BASS, which is **proprietary** — free for non-commercial use only. |
| — | [`ParaTactus.Tests/`](ParaTactus.Tests) | the suite. It is the specification: it pins the frontend against torchaudio, holds the model against the published implementation, and measures the postprocessing against a beatmap's declared tempo. |

## Getting started

**1. The packages are not on nuget.org yet.** Build them from this repository:

```powershell
dotnet pack ParaTactus      -c Release -o artifacts
dotnet pack ParaTactus.Bass -c Release -o artifacts
```

Point a local feed at `artifacts/`, or add the projects to your solution directly. Then:

```powershell
dotnet add package ParaTactus.NET
dotnet add package ParaTactus.Bass   # only if you want a path decoded for you
```

**2. You need the model.** It is not distributed here. The two builds this project was measured with are published as
release assets in [`beat-this-onnx`](https://github.com/CattyCathy/beat-this-onnx) — weights and licence only, no code —
and the same checksums, the licence and the citation are in
[`ParaTactus/MODEL-LICENCE.txt`](ParaTactus/MODEL-LICENCE.txt). `final0` from Beat This! is MIT licensed — copyright
(c) 2024 Institute of Computational Perception, JKU Linz, Austria.

The network takes a **log-mel spectrogram**, not audio: this library computes the mel itself, so the ONNX file only has
to contain the model. `BeatThisBeatTracker` documents the exact frontend it reproduces (22050 Hz mono, `n_fft` 1024,
hop 441, 128 Slaney-scale mel bands with un-normalised triangular filters, `log1p(1000 * mel)`).

[`tools/export_onnx.py`](tools/export_onnx.py) turns the checkpoint into that file, and can write the dynamically
quantised int8 build alongside it — 78.3 MB becomes 20.9 MB. Measured on one track, that build is not the same beats as
the float one — 557 regularised beats against 551 — and the two are too close to rank from one track: the median residual
to the beatmap's grid is 29 ms for both, with the float build a shade ahead on the tail (p90 137.7 ms against 141 ms).
[`docs/model.md`](docs/model.md) has the numbers and what they do not say, and
[`samples/ModelCompare`](samples/ModelCompare) re-runs the comparison on a track of your own:

```powershell
python tools/export_onnx.py --checkpoint final0 --out beat-this-final0.onnx --int8-out beat-this-final0-int8.onnx
```

> **The script has not been run end to end here**, because the checkpoint is not in this repository and the machine it
> was written on could not reach one. It is written against the reference's own export path and reviewed rather than
> executed. [`docs/model.md`](docs/model.md) records the I/O contract it has to match, and `--verify` compares two
> exported files against each other on the same input.

**3. First call.** The provider analyses a track, caches the result on disk, and returns a grid:

```csharp
using ParaTactus;
using ParaTactus.Decoding;   // the BASS decoder, from the optional package

var provider = new BeatGridProvider(modelPath, cacheDirectory, BassAudioDecoder.Default);

BeatGrid grid = provider.Get(audioPath);          // about a quarter of the track's length, once
double bpm = grid.BpmAt(timeInMilliseconds);      // the tempo at a time, following real tempo changes
IReadOnlyList<double> beats = grid.Beats;         // the beat instants, in milliseconds
```

> **What has been verified where.** The samples path below runs in a plain process, and it is what the suite measures.
> `BeatGridProvider.Get` works from a bare console host too, now that the defect described under Troubleshooting is
> fixed, and the test that covers it runs in a plain test host rather than behind an application.

**4. Or bring your own samples.** Nothing but the model is needed if you decode elsewhere — no BASS, no framework:

```csharp
float[] samples = myDecoder(audioPath, AnalysisAudio.SampleRate);   // mono, 22050 Hz

double[] beats   = BeatThisBeatTracker.BeatTimes(samples, modelPath);  // model output, unregularised
double[] regular = BeatTrainRegulariser.Regularise(beats);             // put on a locally regular spacing
BeatGrid grid    = BeatGrid.FromBeats(regular);
```

**5. Or feed it as it plays.** `StreamingBeatTracker` accepts chunks and stays ahead of the playhead, so a track can be
analysed while it is being listened to rather than before. `Add` and `Flush` run the model **on the calling thread** and
block it for roughly a quarter of the audio added, which is deliberate: which thread pays for an analysis is the
caller's decision. Feed it from a background thread — `BeatGridProvider` is the wrapper that does — because adding a
second of audio from a UI thread stalls that thread for about 250ms.

## What is where

| Type | What it does |
| --- | --- |
| `BeatGridProvider` | analyses on a miss, serves the cache on a hit, and hands back a `BeatGrid` |
| `BeatGrid` | `Beats`, `BpmAt(time)`, `Duration`, `PhaseAt(time)` — what a UI animates against |
| `BeatThisBeatTracker` | the model: log-mel frontend, chunked inference, peak picking |
| `BeatTrainRegulariser` | puts the tracked beats on a locally regular spacing |
| `StreamingBeatTracker` | the same analysis, fed in chunks |
| `BeatGridCache` | where analysed grids are kept; the key includes this assembly's build identity |
| `OnsetEnvelope`, `Fft`, `MetricalLevel`, `BeatTimeMap` | the stages underneath, usable on their own |
| `IAudioDecoder`, `MonoMixdown` | the seam a decoder plugs into, and the mixdown every decoder needs |

## Limits of the model

These are boundaries of the tool rather than defects, each measured rather than assumed: the tactus tops out around
215 BPM; fast tempo ramps are low-passed; the phase of a beat is not recoverable from the audio in dense music and
nothing here moves a beat the model did not propose; and short halved or doubled fragments survive in passages that are
dense and individually weak. [`ParaTactus/README.md`](ParaTactus/README.md) has the measurements behind each.

Two deliberate differences from the publication are stated here rather than only in the code. The peak picking runs an
extra pruning pass (`BeatThisBeatTracker.Prune`) that the reference's minimal postprocessor does not have, dropping a
beat that sits far closer to a neighbour than the local beat period; that is why the beats reported here are not the
beats `beat_this` reports, and `RawPeaks` exists so the two can be told apart when something is wrong. The resampler is
the other: it low-passes before it decimates, because the model's mel bands reach 11kHz and an unfiltered decimation
folds content from above the target Nyquist back into them.

## Troubleshooting

- **Beats look stale.** Delete the cache directory. The key is a hash of each file's path, size and last-write time plus
  this assembly's build identity, so replacing a track or a model, or rebuilding the analysis, produces a miss — but a
  stale grid is still the first thing to rule out, and deleting costs one re-analysis per track.
- **`Analysing a track from its path needs a decoder`.** `BeatGridProvider` was constructed without one. Either add the
  package and pass `BassAudioDecoder.Default`, or decode the audio yourself and use the samples path.
- **BASS is not free for commercial use.** It is free for non-commercial use and licensed per product otherwise, and the
  NuGet package carrying its native binaries declares MIT in a field that cannot cover them. See
  [`ParaTactus.Bass/THIRD-PARTY-NOTICES.md`](ParaTactus.Bass/THIRD-PARTY-NOTICES.md). The analysis package has no such
  dependency: if you supply samples, you never touch BASS.
- **A fresh clone of a consumer needs the packages in a local feed.** Until they are published, that is the trade for
  keeping the analysis decoupled from a private decoder.
- **`Get` used to block in a bare console host.** It set below-normal priority on its own analysis thread, and ONNX
  Runtime's thread pool, created from a thread like that, never finished the inference: measured on the reference track,
  the same sequence decoded, analysed and flushed in 22 seconds at normal priority and spun at full CPU for minutes
  below it. The fix is to leave that thread at normal priority - what protects playback is the two processors held back
  from the model's own thread pool, which is what the code comment claimed was doing the work all along. Lower that
  priority again and this is the symptom to expect.

## Licence

MIT, copyright (c) 2026 CattyCathy — see [`LICENSE`](LICENSE). `ParaTactus.Bass` is MIT as source but depends on the
proprietary BASS library; `ParaTactus/THIRD-PARTY-NOTICES.md` covers everything the analysis itself needs, which is
three MIT packages and nothing else.

## History

This repository starts at the split. The code, and the whole history of how it got here — the frontend bugs that
nothing local could detect, the postprocessing that was measured and reverted, the limits that were found one at a time
— is in the application repository it grew out of.
