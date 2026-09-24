# Third-party notices

The code in this repository is MIT licensed (see `LICENSE`). It does not copy source from anywhere: the frontend, the
postprocessing and the grid are original implementations of a published algorithm.

## Dependencies

Three packages, all MIT, and that is the whole of it:

| Package | Version | Licence |
| --- | --- | --- |
| `Microsoft.ML.OnnxRuntime` | 1.30.0 | MIT |
| `Microsoft.ML.OnnxRuntime.Managed` | 1.30.0 | MIT |
| `System.Numerics.Tensors` | 9.0.0 | MIT |

No proprietary dependency, no copyleft dependency, no native payloads, and nothing to reproduce beyond this file. That
is worth stating plainly because it was not always true: decoding audio used to happen inside this assembly, through
BASS, which is proprietary — free for non-commercial use and licensed per product otherwise, and whose terms require a
licensed product to be an end-user product rather than a component of another product. It arrived with `osu.Framework`,
which brought FFmpeg, ImageSharp under a non-open-source split licence, a tablet driver stack, and 61 MB of native
binaries across the platforms it supports.

The decode now lives in the separate `ParaTactus.Bass` package, behind the `IAudioDecoder` interface defined here. A
consumer of this package never touches it, and a caller who already has samples needs nothing at all.

If you want a track decoded from its path, take `ParaTactus.Bass` — its `THIRD-PARTY-NOTICES.md` carries the terms you are
then accepting.

## The model is not distributed here

The network is not in this repository. It is Beat This! by Francesco Foscarin, Jan Schlüter and Gerhard Widmer, and it
is MIT licensed with a specific copyright holder that has to be reproduced:

```
MIT License

Copyright (c) 2024 Institute of Computational Perception, JKU Linz, Austria
```

- Paper: *Beat This! Accurate Beat Tracking Without DBN Postprocessing*, ISMIR 2024 — https://arxiv.org/abs/2407.21658
- Code and checkpoints: https://github.com/CPJKU/beat_this
- The `final0` checkpoint used for every measurement in this project, and its SHA-256, are recorded in
  `MODEL-LICENCE.txt`, which also carries the licence verbatim so it can travel with a redistributed copy of the model.

A quantised build — the int8 file this project measures with — is a derivative of that checkpoint and carries the same
notice. If you publish the model as a release asset or on a model hub, ship `MODEL-LICENCE.txt` beside it and cite the
paper.
