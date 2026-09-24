#!/usr/bin/env python3
"""Export the Beat This! `final0` checkpoint to the ONNX file this library loads.

Everything here is taken from the artefact it has to reproduce rather than from memory, because a mismatch would not be
visible until beats came out wrong:

* the graph's input is named `spectrogram`, shape `[1, frames, 128]`, float32, with a dynamic frame axis. The library
  computes the log-mel itself, so the network takes the spectrogram and not audio.
* the graph's outputs are `beat` and `downbeat`, shape `[batch, frames]`, float32.
* the recorded file is `ir_version 8`, producer `pytorch 2.8.0`, opset **17**.
* the reference calls the model as `model(chunk.unsqueeze(0))` where a chunk is `(time, bins)`
  (`beat_this/inference.py`), so the model is called with `(1, T, 128)` and needs no wrapper beyond unpacking its
  output into the two named tensors.

Not verified end to end in this environment: the checkpoint is not present here, and downloading it needs the network
the export itself needs. `--verify` is the step that closes that gap, and it should be run before trusting a new file.

    python tools/export_onnx.py --checkpoint final0 --out beat-this-final0.onnx --int8-out beat-this-final0-int8.onnx
    python tools/export_onnx.py --checkpoint final0 --out /tmp/x.onnx --verify beat-this-final0.onnx
"""

import argparse
import sys


class Exportable:
    """Deferred so that importing this module does not require torch."""

    @staticmethod
    def wrap(model):
        import torch

        class Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, spectrogram):
                output = self.inner(spectrogram)

                # The reference returns either a dict keyed by head or a tuple; the exported graph must return the two
                # tensors in the order recorded above.
                if isinstance(output, dict):
                    return output["beat"], output["downbeat"]

                if isinstance(output, (tuple, list)):
                    return output[0], output[1]

                raise SystemExit(f"unexpected model output: {type(output)}")

        wrapper = Wrapper(model)
        wrapper.eval()

        return wrapper


def load(checkpoint):
    import torch

    try:
        from beat_this.inference import load_model
    except ImportError:
        sys.exit(
            "the reference package is not importable. Install it, or point PYTHONPATH at its source:\n"
            "  pip install https://github.com/CPJKU/beat_this/archive/main.zip"
        )

    print(f"loading {checkpoint}")

    return load_model(checkpoint, device="cpu", dbn=False).eval() if callable(load_model) else None


def export(checkpoint, out, frames):
    import torch

    model = Exportable.wrap(load(checkpoint))
    dummy = torch.zeros(1, frames, 128, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy,),
        out,
        input_names=["spectrogram"],
        output_names=["beat", "downbeat"],
        dynamic_axes={
            "spectrogram": {0: "batch", 1: "frames"},
            "beat": {0: "batch", 1: "frames"},
            "downbeat": {0: "batch", 1: "frames"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    print(f"wrote {out}")

    with torch.no_grad():
        beat, downbeat = model(dummy)

    print(f"  {frames} frames of silence -> beat {beat.shape}, downbeat {downbeat.shape}")


def quantise(source, out):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # Dynamic quantisation: weights to int8, activations quantised at run time. The original int8 build records only
    # that it was made by a tool calling itself `onnx.quantize` and keeps opset 17 and the same input and output names,
    # so its exact settings are not recoverable from the file; this is the recipe that replaces them with something
    # documented.
    quantize_dynamic(source, out, weight_type=QuantType.QInt8)

    print(f"wrote {out}")


def verify(left, right):
    import numpy as np
    import onnxruntime as ort

    sessions = [ort.InferenceSession(path, providers=["CPUExecutionProvider"]) for path in (left, right)]
    spectrogram = np.random.default_rng(0).normal(size=(1, 750, 128)).astype(np.float32)

    outputs = [session.run(None, {"spectrogram": spectrogram}) for session in sessions]

    for index, name in enumerate(("beat", "downbeat")):
        difference = float(np.max(np.abs(outputs[0][index] - outputs[1][index])))
        print(f"{name}: max absolute difference {difference:.6f}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="final0", help="checkpoint name or path")
    parser.add_argument("--out", help="where to write the fp32 ONNX file")
    parser.add_argument("--int8-out", help="where to write the int8 ONNX file, built from --out")
    parser.add_argument("--frames", type=int, default=1500, help="frames in the dummy input used for tracing")
    parser.add_argument("--verify", nargs=2, metavar=("A", "B"), help="compare two ONNX files on the same input")
    arguments = parser.parse_args()

    if arguments.verify:
        sys.exit(verify(*arguments.verify))

    if not arguments.out:
        sys.exit("--out is required unless --verify is used")

    export(arguments.checkpoint, arguments.out, arguments.frames)

    if arguments.int8_out:
        quantise(arguments.out, arguments.int8_out)
