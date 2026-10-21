# Changelog

Newest first. This recipe serves one DGX Spark. The image is `tensorfold-ornith:zig-d1018b3`, built by
`scripts/prepare.sh` from TensorFold `d1018b31` plus `patches/`.

## Unreleased

- Performance (2026-10-21): prefill 360 -> ~2,450 tok/s at 4k (~1,230 at 128k); decode 28 -> 35 tok/s without drafts,
  53 (prose) / 83 (code) with DFlash2 on one stream, up to 111 tok/s total at x4 (sparkDash, thinking off). New `patches/0008`-`0027`; DFlash2 and no-drafts outputs stay identical (21/21 prompts).
  `WEIGHTS_GIB` is 69. Fixed `--kv-cache` flag in `start.sh` and the image disk preflight in `prepare.sh`.

- DFlash2 drafts are the default (`DRAFTS=1`, `DRAFTER=dflash2`): `prepare.sh` downloads and verifies
  `incoai/Ornith-1.5-122B-DFlash2` @ `a5ed7396`, `start.sh` passes `--drafter <snapshot>`. `DRAFTS=0` disables them.
  `DRAFTER=mtp` is the experimental MTP drafter (exact, 0.87x). New `patches/0007` (DFlash2).
- `VISION=1` is now the default (patch 0006, tested with images); the memory check counts the 2 GiB vision workspace.
- Memory check and disk check count the drafter (+1 GiB device, +2 GB disk).
- Measured on a GB10 (2026-10-20): decode 28 tok/s without drafts, 59 (code) and 31-41 (prose) with DFlash2; prefill
  ~360 tok/s short, ~227 at 29k; model 60.25 GiB, load 56-76 s. See the README for caveats.
- First version of the recipe: `start.sh`, `stop.sh` and `scripts/` for Ornith-1.5-122B
  (`Mia-AiLab/Ornith-1.5-122B-EXL3-EAQv1-4-bpw` @ `9c5c913b`), adapted from the Qwen3.8 Flash Next single-Spark recipe.
- Defaults: 262,144-token window, 4 requests at once, fp8 KV, thinking on, DFlash2 drafts, image/video input, port 8888.
