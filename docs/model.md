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

## Why the model is not in this repository

It is 78 MB, and 21 MB quantised. A package that size does not belong in git history or in a NuGet package, so the
library asks for a path instead and the README says where to get one. Publish the file yourself as a release asset or on
a model hub if you want others to use it, with this licence text beside it.
