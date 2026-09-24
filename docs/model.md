# The model

This library does not contain a network. It contains the analysis around one, and it needs a file to run.

The network is **Beat This!** by Francesco Foscarin, Jan Schlüter and Gerhard Widmer, published at ISMIR 2024. The
checkpoint is MIT licensed, and the notice is specific about the copyright holder, which matters because a converted or
quantised copy is a derivative of it:

```
MIT License

Copyright (c) 2024 Institute of Computational Perception, JKU Linz, Austria
```

Reproduce that notice with any redistribution, including a quantised build, and cite the paper:

> Foscarin, F., Schlüter, J., and Widmer, G. (2024). *Beat This! Accurate Beat Tracking Without DBN Postprocessing*.
> Proceedings of the 25th International Society for Music Information Retrieval Conference (ISMIR).
> https://arxiv.org/abs/2407.21658

The licence text, the citation and the SHA-256 of both builds this project was measured with are in
[`ParaTactus/MODEL-LICENCE.txt`](../ParaTactus/MODEL-LICENCE.txt). Checkpoints come from
[CPJKU/beat_this](https://github.com/CPJKU/beat_this); `final0` is the one every measurement here used.

## What the file has to look like

Read off the artefact rather than assumed, because a mismatch shows up as wrong beats rather than as an error:

| | |
| --- | --- |
| input | `spectrogram`, `[1, frames, 128]`, float32, frame axis dynamic |
| outputs | `beat`, `downbeat`, `[batch, frames]`, float32 |
| opset | 17 |
| recorded producer | `pytorch 2.8.0`, `ir_version 8` |

The input is a **log-mel spectrogram, not audio**. This library computes the mel itself, which is why the export
contains only the model. The frontend it reproduces is documented on `BeatThisBeatTracker`: 22050 Hz mono, `n_fft`
1024, hop 441, 128 Slaney-scale mel bands with **un-normalised** triangular filters, `log1p(1000 * mel)`.

## Producing one

[`tools/export_onnx.py`](../tools/export_onnx.py) exports the checkpoint and, optionally, quantises the result:

```bash
pip install torch "https://github.com/CPJKU/beat_this/archive/main.zip" onnx onnxruntime
python tools/export_onnx.py --checkpoint final0 --out beat-this-final0.onnx --int8-out beat-this-final0-int8.onnx
```

It takes the shape, the names and the opset from the table above. The reference calls the model as
`model(chunk.unsqueeze(0))` with a `(time, bins)` chunk, so the traced module simply takes `(1, T, 128)` and returns the
two heads.

Two things to know about it:

- **It has not been run end to end in the repository it ships in.** The checkpoint was not available there, and fetching
  it needs the same network the export needs. The first thing to do with a new file is `--verify` it against a known
  good one:

  ```bash
  python tools/export_onnx.py --verify beat-this-final0.onnx /path/to/known-good.onnx
  ```

  That runs both on the same pseudo-random spectrogram and prints the largest difference per output. Anything beyond
  float noise means the export does not match, and `LogMelFrontendTest` plus the parity scripts in the application
  repository are the next place to look.
- **The int8 build's original settings are not recoverable.** The file recorded only that its producer called itself
  `onnx.quantize`, with opset 17 and the same input and output names. The script uses ONNX Runtime's dynamic
  quantisation, so the recipe is written down and repeatable rather than inferred from a binary.

## What the quantised build actually costs

Measured rather than assumed, because the choice between the two files is a real one and the obvious claim — that a
quantised model gives the same beats — is false. Both columns are the same decode of one track, Designant, 178.4s:

| | `final0` fp32, 78.3 MB | dynamically quantised, 20.9 MB |
| --- | --- | --- |
| raw model beats | 390 | 406 |
| beats after `BeatTrainRegulariser` | 551 | 557 |
| of those, with no partner within 20ms | 61 | 67 |
| residual to the beatmap's grid, median | 29 ms | 29 ms |
| residual, p90 | 137.7 ms | 141 ms |
| residual, max | 397 ms | 372 ms |
| beats more than 60ms from the grid | 192 of 551 (34.8%) | 202 of 557 (36.3%) |
| tempo level at 21 steady points | 3 not at a metrical level | 4 not at a metrical level |

The two are **not interchangeable beat for beat**: they disagree about how many beats the track has, and about where a
good fraction of them fall. They are also **too close to rank from one track**. The median residual is identical, the
float build is a shade ahead on the tail here, and the quantised one was a shade ahead on the same measurement before
the resampler gained its anti-aliasing filter and the peak merge stopped truncating - the ordering moves with the
pipeline, not with the model, so a preference between these two files is a size decision rather than a quality one.

The numbers above were measured after those two fixes, both of which move beats; the figures this table carried before
them are still in the history of this file if a comparison across versions is wanted.

The residuals are `BeatmapTempoAgreementTest.TestModelBeatsAgainstBeatmapGrid` and the tempo levels are
`TestModelBeatsMatchBeatmap`; each was run once per model with `OSUTEST_AUDIO` set and `OSUTEST_MODEL` pointed at that
model. The beat counts and the 20ms column come from [`samples/ModelCompare`](../samples/ModelCompare), which runs both
files over one decode of one track and reports where they part:

```bash
dotnet run --project samples/ModelCompare -- track.mp3 beat-this-final0.onnx beat-this-final0-int8.onnx
```

A single track is not a corpus: these numbers separate the two files, they do not rank them in general.

## Why the model is not in this repository

It is 78 MB, and 21 MB quantised. A package that size does not belong in git history or in a NuGet package, so the
library asks for a path instead and the README says where to get one. The two builds measured here are published as
release assets in [`beat-this-onnx`](https://github.com/CattyCathy/beat-this-onnx) — weights and licence, no code — with
their checksums and this licence text beside them. Where the file comes from is nevertheless your choice: it is a path
parameter, so `tools/export_onnx.py` against your own checkpoint is equally supported.
