# ParaTactus.Bass

Audio decoding for [ParaTactus.NET](../ParaTactus), with BASS, so that a track can be analysed from its path.

The package is `ParaTactus.Bass`; the namespace is `ParaTactus.Decoding`. A namespace ending in the same word as a type shadows it
for everything declared inside it, so `ParaTactus.Bass` present in scope would make `Bass.Init` resolve to the namespace
rather than to `ManagedBass.Bass`. Only the namespace avoids that.

The analysis is defined on samples. If you already have them, you do not need this package, and you do not want it: it
pulls in a proprietary audio library, which `ParaTactus.NET` alone does not.

```csharp
using ParaTactus;
using ParaTactus.Decoding;

var provider = new BeatGridProvider(modelPath, cacheDirectory, BassAudioDecoder.Default);
BeatGrid grid = provider.Get(audioPath);

// or, one file at a time
double[] beats = BeatThisBeatTracker.BeatTimes(audioPath, modelPath, BassAudioDecoder.Default);
```

`BassAudioDecoder.Default` is the decoder; `BassAudioDecoder.DecodeMono` and `DecodeRaw` are static if you want the
samples rather than a grid. The decoder starts BASS itself on first use, with the no-sound device, so nothing has to be
initialised before it and decoding does not need audio hardware. The native BASS libraries have to be beside the
assembly that uses them, which is why a framework-dependent build with no runtime identifier will not run: pin a RID, or
deploy the natives yourself.

## Before you use this commercially

BASS is free for non-commercial use and licensed per product otherwise, and its terms require a licensed product to be
an end-user product rather than a component of another product. Read `THIRD-PARTY-NOTICES.md` in this directory: the
NuGet metadata on the package that carries the native BASS binaries declares MIT, which is the licence its publisher
distributes under and does not cover the binaries themselves.

If that is a problem, supply your own `IAudioDecoder` instead and reference only `ParaTactus.NET`.
