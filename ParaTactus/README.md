# ParaTactus.NET

Beat tracking for audio, in C#. It answers one question — *where are the beats in this track, including when the tempo
changes* — and hands back a grid that a UI can animate against.

Two packages, and the split is deliberate:

| Package | What it is | Dependencies |
| --- | --- | --- |
| **`ParaTactus.NET`** | the analysis | three MIT packages, nothing else |
| **`ParaTactus.Bass`** | optional: decodes a path for you | BASS, which is **proprietary** — see below |

The assembly and namespace of both are `ParaTactus`, so the package identifier is the only place the `.NET` suffix appears
and code reads `ParaTactus.BeatGrid`. (The `ppy.osu.Framework` package this project grew up on does the same thing: the
package has a prefix, the assembly does not.)

It is a port of [Beat This!](https://github.com/CPJKU/beat_this) (Foscarin, Schlüter and Widmer, ISMIR 2024) to .NET:
the log-mel frontend, the chunked transformer inference, and the peak-picking postprocessor are implemented here from
the published reference rather than called through it. `tools/ref/` in the application repository holds the scripts that
were used to hold this implementation against the original, including the fixture that pins the frontend.

## What it is for

Driving animation from music. The intended use is a player that pulses on the beat: the visual requirement is that
every pulse lands on a real beat, not that the reported tempo is the music's own. A grid that reports half the true
tempo and pulses on every other beat is acceptable; a grid that drifts off the beat is not.

## Requirements

- .NET 10
- A Beat This! checkpoint exported to ONNX. **The model is not distributed with this library.** Obtain the official
  `final0` checkpoint from [CPJKU/beat_this](https://github.com/CPJKU/beat_this) and export it with
  [`tools/export_onnx.py`](../tools/export_onnx.py), which writes the int8 build as well. `../docs/model.md` carries the
  file contract, and what the quantised build costs against the float one: on the single track measured it is not the
  same beats — 557 regularised beats against 551 — and the two are close enough that which one leads moves with the
  pipeline rather than with the model. The SHA-256 of both builds is
  in `MODEL-LICENCE.txt`. The checkpoint is MIT licensed, copyright (c) 2024 Institute of Computational Perception, JKU
  Linz, Austria — that notice also applies to any quantised copy. The frontend the export has to match is documented in
  `BeatThisBeatTracker`: 22050 Hz mono, `n_fft` 1024, hop 441, 128 mel bands on the Slaney frequency scale with
  **un-normalised** triangular filters, `log1p(1000 * mel)`.
- Audio, as mono `float[]` at `AnalysisAudio.SampleRate`. If you would rather have a path decoded for you, take
  `ParaTactus.Bass`; if you already have samples, or a decoder of your own, you need nothing but this package.

## Using it

The usual entry point is the provider, which caches analysed grids on disk and analyses on a miss. It needs a decoder
only for the case where a grid has to be produced from a path:

```csharp
using ParaTactus;
using ParaTactus.Decoding;

var provider = new BeatGridProvider(modelPath, cacheDirectory, BassAudioDecoder.Default);
BeatGrid grid = provider.Get(audioPath);          // blocks for roughly a quarter of the track's length on a miss

double bpm = grid.BpmAt(timeInMilliseconds);      // the tempo reported at a time, following real tempo changes
var beats = grid.Beats;                           // the beat instants, in milliseconds
```

For audio that is already in memory the analysis takes samples and touches nothing else — no decoder, no framework, no
file:

```csharp
float[] samples = myOwnDecoder(path, AnalysisAudio.SampleRate);               // anything that produces mono PCM
double[] beats = BeatThisBeatTracker.BeatTimes(samples, modelPath);           // model output, one level, unregularised
double[] regular = BeatTrainRegulariser.Regularise(beats);                    // put on a locally regular spacing
BeatGrid grid = BeatGrid.FromBeats(regular);
```

Given a path, the same thing with the BASS decoder is one call:

```csharp
double[] beats = BeatThisBeatTracker.BeatTimes(audioPath, modelPath, BassAudioDecoder.Default);
```

For a track that is still playing, `StreamingBeatTracker` accepts audio in chunks and stays ahead of the playhead:

```csharp
using var tracker = new StreamingBeatTracker(modelPath);
tracker.Add(chunkOfSamples);
if (tracker.TryTake(...)) { /* beats are available */ }
```

Both of those are exercised inside an application host and by the test suite, which is a plain host of its own. The defect
that once made `BeatGridProvider.Get` block from a bare console program is fixed: it was below-normal priority on the
analysis thread, which stops ONNX Runtime's pool from completing, and `README.md` at the root of the repository records
the measurement that identified it.

To decode with something other than BASS, implement `IAudioDecoder`; `MonoMixdown` does the channel mixdown and
resampling that every decoder needs, so an implementation is usually a decode call and one line.

The stages are separable on purpose and are each useful alone: `OnsetEnvelope` (a spectral-flux envelope at 50 fps),
`Fft`, `MetricalLevel` (metrical relations between beat sequences), `BeatTimeMap` and `BeatGridCache`.

## Known limits of the model

These are boundaries of the tool rather than defects, and each was measured rather than assumed. They are recorded here
so they are not re-investigated as bugs.

- **The tactus tops out around 215 BPM.** The publication's own alternative postprocessor caps its DBN at `max_bpm=215`,
  and above that the model settles onto a slower pulse instead of failing visibly — it looks confident and is wrong.
  On a track whose beatmap ramps to 400 BPM the model holds a flat ~167 BPM through the top of the ramp.
- **Fast tempo ramps are low-passed.** Where a track brakes to 100 BPM and returns to 200 within 3.5 seconds, the model
  holds one intermediate interval (~360ms) for about five seconds after the music has settled at 300ms, at full
  activation confidence. No postprocessing in this library recovers this, and it was measured rather than assumed: the
  onset envelope's own best period in that passage is the correct 300ms, and the model's activation at the correct beat
  positions stays positive, but the envelope cannot supply a *phase* (see below), so there is nothing to correct against.
- **Phase is not recoverable from the audio in dense music.** A 300ms comb swept over the onset envelope picks a phase
  that moves by up to 150ms from one second to the next even where the envelope's periodicity at 300ms is strong, and
  the model's own activation behaves the same way. In music with an onset near every sixteenth, a comb at any phase
  finds onsets. An earlier attempt to re-place a grid from the audio alone moved it onto the off-beats by a measured
  152ms, which is why nothing in this library moves a beat to a position the model did not propose.
- **Occasional short halved or doubled fragments** remain in passages that are dense and individually weak, where no
  neighbourhood-based period estimate can call them spurious.

`BeatGrid.BpmAt` reports tempo with a window wide enough not to jump on a single mis-tracked beat, and the reported
tempo is deliberately separate from the interval the animation steps at, so a correction to one does not stutter the
other.

## Caching

Analysed grids are cached on disk. The cache key is a hash of each file's full path, size and last-write time — not its
contents — together with the build identity of this assembly. Replacing a track or a model, or rebuilding the analysis,
therefore produces a miss rather than stale beats; a file edited in place and left the same size with the same timestamp
would not, which is the one case it does not cover. If beats ever look stale, delete the cache directory — it costs one
re-analysis per track and rules out the entire class of problem.

## Licence

MIT, copyright (c) 2026 CattyCathy. See `LICENSE`. Three MIT dependencies, no native payloads, no copyleft — the terms of
the packages themselves are all there is.

`ParaTactus.Bass` is also MIT as source, but it depends on BASS, which is proprietary: free for non-commercial use, licensed
per product otherwise, and whose terms require a licensed product to be an end-user product rather than a component of
another product. Its `THIRD-PARTY-NOTICES.md` has the details, including the fact that the NuGet package carrying the
native BASS binaries declares MIT and is wrong to.

The model is a separate work under its own MIT notice — copyright (c) 2024 Institute of Computational Perception, JKU
Linz, Austria — reproduced in `MODEL-LICENCE.txt` together with the citation and the checksums.
