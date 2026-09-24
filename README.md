# ParaTactus.NET

Beat tracking for audio, in C#: the beat instants of a track, including where the tempo changes, as a grid a UI can
animate against.

Two packages:

| Package | Project | What it is |
| --- | --- | --- |
| `ParaTactus.NET` | `ParaTactus/` | the analysis. Three MIT dependencies and nothing else. |
| `ParaTactus.Bass` | `ParaTactus.Bass/` | optional. Decodes a path for you, with BASS, which is **proprietary** — free for non-commercial use only. |

`ParaTactus.Tests/` holds the suite. It is the specification rather than an afterthought: it pins the log-mel frontend
against torchaudio's own output, holds the model against the published implementation, and measures every stage of the
postprocessing against a beatmap's declared tempo.

Start with [`ParaTactus/README.md`](ParaTactus/README.md) — the requirements (a model file is needed, and is not distributed
here), the usage, and the limits of the model that were measured rather than assumed.

## Licence

MIT, copyright (c) 2026 CattyCathy — see [`LICENSE`](LICENSE). `ParaTactus.Bass` is MIT as source but depends on the
proprietary BASS library; its [`THIRD-PARTY-NOTICES.md`](ParaTactus.Bass/THIRD-PARTY-NOTICES.md) has those terms, and
`ParaTactus/THIRD-PARTY-NOTICES.md` covers the rest.

## History

This repository starts at the split. The code, and the whole history of how it got here — the frontend bugs that
nothing local could detect, the postprocessing that was measured and reverted, the limits that were found one at a time
— is in the application repository it grew out of.
