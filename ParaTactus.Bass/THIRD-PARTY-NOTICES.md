# Third-party notices — ParaTactus.Bass

This project's own source is MIT, like the analysis it decodes for. **What it depends on is not**, and that is the entire
reason it lives in a separate package: a product that includes this package is no longer free of proprietary terms,
while one that includes only `ParaTactus.NET` still is.

## BASS

`ppy.ManagedBass`, `ppy.ManagedBass.Fx`, `ppy.ManagedBass.Mix` and `ppy.ManagedBass.Wasapi` are MIT — they are bindings
only. The library they bind to is proprietary:

> BASS is free for non-commercial use. If you are a non-commercial entity (eg. an individual) and you are not making any
> money from your product (through sales, advertising, etc) then you can use BASS in it for free. Otherwise, one of the
> following licences will be required. […] **Your products must be end-user products, eg. not components used by other
> products.**
>
> — https://www.un4seen.com/bass.html (shareware EUR 125 / single commercial EUR 950 / unlimited EUR 3,450)

The native libraries arrive in `ppy.osu.Framework.NativeLibs`, whose NuGet metadata declares MIT. That declaration is
about the package ppy distributes, and it cannot reach the third-party binaries inside it: a licence can only be granted
by whoever holds the rights, and what governs BASS is un4seen's terms, not a field in someone else's `.nuspec`. The
package's maintainers have said as much — that it is intended for osu!'s own consumption, and that anyone consuming it
has to do their own due diligence. This file is that diligence, done once and written down. Treat the BASS binaries as
proprietary whatever the NuGet page says.

A commercial product that redistributes BASS needs its own licence from un4seen. Replacing this package with a decoder
of your own removes the requirement entirely; see `IAudioDecoder` in `ParaTactus.NET`.

## FFmpeg

The same native package also carries `avcodec`, `avformat`, `avutil` and `swscale` (FFmpeg 4.x) for every platform it
supports, under the same incorrect MIT declaration. FFmpeg built without `--enable-gpl` is LGPL-2.1-or-later, which
requires that the recipient can replace the library: ship the licence text, keep the binaries separable rather than
statically merged, and point at the corresponding source.

## Everything else

`ppy.osu.Framework` itself is MIT and is referenced only for `osu.Framework.Audio.Callbacks`, the file callbacks used to
decode a stream that has no path. Its full transitive closure — including `SixLabors.ImageSharp` under the Six Labors
Split License and `OpenTabletDriver` under LGPL-3.0-or-later — is listed in the analysis package's
`THIRD-PARTY-NOTICES.md`.

## The model

Not distributed here. Beat This! by Francesco Foscarin, Jan Schlüter and Gerhard Widmer, MIT, copyright (c) 2024
Institute of Computational Perception, JKU Linz, Austria. The licence text, the citation and the checksums of the
checkpoint are in the analysis package's `MODEL-LICENCE.txt`.
