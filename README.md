<h1 align="center">Ornith-1.5-122B EXL3 on one DGX Spark (TensorFold)</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>


> **Status: working, with caveats.** This recipe needs a **patched TensorFold**: upstream's Qwen3.5-MoE CUDA engine does
> not read EXL3 yet. The files in [`patches/`](patches/) add it, and `./start.sh` serves the model on a GB10 with the
> measurements below. Not everything is tested; see [Caveats](#caveats).

Serve **Ornith-1.5-122B** (Qwen3.5-MoE architecture) from a single NVIDIA DGX Spark (GB10, 128 GB) through an
OpenAI-compatible API, as a 4-bpw EXL3 quantization. It runs [TensorFold](https://github.com/ashhart/TensorFold)'s Zig
engine, `tensorfold-native` (pinned commit `d1018b31`, plus `patches/`), in NVIDIA's PyTorch container.

- Checkpoint: [`Mia-AiLab/Ornith-1.5-122B-EXL3-EAQv1-4-bpw`](https://huggingface.co/Mia-AiLab/Ornith-1.5-122B-EXL3-EAQv1-4-bpw)
  at revision `9c5c913b3de1eef901520d6fedaf450044ea0423` (9 safetensors shards, 62 GiB on disk; 60.25 GiB on the device
  in the Zig engine; `WEIGHTS_GIB=64` is a conservative figure for the memory check).
- Draft model (on by default): [`incoai/Ornith-1.5-122B-DFlash2`](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2) at
  revision `a5ed7396f5f9e5b1abf7ff50b7899a3186961432` (Apache-2.0, 1.86 GB, ~0.4 GB more on the device). `DRAFTS=0` serves without it.
- Quantization, from the quant card (not measured here): 4.049 bpw trellis average, KL divergence 0.0135 and top-1
  agreement 96.7% against the BF16 model.
- API model id: `Ornith-1.5-122B`
- Context: **262,144 tokens** a request (the native window). 4 requests at once by default (`PARALLEL`). A full window's
  fp8 cache is ~3.05 GiB (bf16 ~6.0 GiB), plus ~160 MiB of GatedDeltaNet state a stream.
- One command: `./start.sh` sets everything up on the first run and starts the server; `./stop.sh` stops it

## Performance

Measured on one DGX Spark (GB10) with [sparkDash](https://github.com/MiaAI-Lab/sparkDash), thinking off, DFlash2 drafts
(`DRAFTS=1`), fp8 KV. `×N` is N requests decoding at once; `agg` is the total across them, `/str` is per request.

**Decode** (tok/s)

| Prompt kind | Streams | agg | per stream | TTFT |
| --- | ---: | ---: | ---: | ---: |
| Prose | ×1 | 52.6 | 52.6 | 164 ms |
| Prose | ×2 | 68.0 | 35.5 | 265 ms |
| Prose | ×4 | 90.4 | 24.2 | 441 ms |
| Code | ×1 | 83.3 | 83.3 | 224 ms |
| Code | ×2 | 103.4 | 53.4 | 305 ms |
| Code | ×4 | 111.0 | 32.1 | 513 ms |

Without drafts (`DRAFTS=0`) a single stream decodes at about 35 tok/s. DFlash2 output is exact: greedy output matched the
no-draft run on 21 of 21 prompts (and two concurrent streams). MTP (`DRAFTER=mtp`) is experimental and slower than no drafts.

**Prefill** (tok/s)

| Prompt tokens | tok/s | TTFT |
| ---: | ---: | ---: |
| 4k (4,136) | 2,450 | 1.90 s |
| 8k (8,230) | 2,279 | 3.82 s |
| 16k (16,421) | 2,158 | 7.82 s |
| 32k (32,807) | 1,955 | 16.99 s |
| 64k (65,576) | 1,644 | 40.10 s |
| 128k (131,111) | 1,230 | 106.85 s |

A cached prefix resumes in about 0.6 s (17k prompt; resume points are every 4,096 tokens).

**Load and memory** (measured on a GB10, 2026-10-20): model plus scratch ~69 GiB on the device (with DFlash2), load 80-95 s.

**Quality:** not measured here. From the quant card: 4.049 bpw trellis average, KL divergence 0.0135 and top-1 agreement
96.7% against the BF16 model.

**Context:** needle-in-a-haystack passed at 32k tokens, and prefill ran to 131k tokens (sparkDash). The native window is
262,144 tokens and is what the server is configured for; answer quality beyond 32k has not been tested.

## Requirements

- A DGX Spark (or another GB10 system with 128 GB unified memory) with nothing else large on the GPU. The default
  window needs `MemAvailable` of about 92 GiB at start (weights ~64, drafter 1, one full fp8 window ~4, one GDN state ~1,
  scratch slack 8, vision workspace 2, floor 10, margin 2). `start.sh` stops before launching below that (`MEM_NEED_GIB`).
- Docker with the NVIDIA container runtime, and your user in the `docker` group.
- ~112 GB free disk on a fresh machine: 62 GiB for the checkpoint and 2 GB for the drafter under `~/.cache/huggingface`, and the
  image under Docker's root. `scripts/prepare.sh` checks both.
- Optional: the `hf` CLI on the host and a Hugging Face token in `~/.cache/huggingface/token` or `HF_TOKEN`.

## Quick start

```bash
git clone https://github.com/MiaAI-Lab/Ornith-1.5-122B-TensorFold.git
cd Ornith-1.5-122B-TensorFold
./start.sh
```

The first run builds the engine (TensorFold at the pinned commit, plus `patches/`) and downloads the checkpoint if it is
not already in the cache. Later starts load the weights. `start.sh` shows each step, runs a smoke test, prints
`Ornith-1.5-122B is now LIVE! on port 8888`, and returns you to the shell.

```bash
curl -s http://<spark-address>:8888/v1/models

curl -s http://<spark-address>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Ornith-1.5-122B",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 1000
}'
```

```bash
./start.sh restart
./stop.sh
docker logs -f ornith-122b-tf
curl -s http://<spark-address>:8888/health
```

## What `start.sh` and `scripts/prepare.sh` do

1. **Setup:** `scripts/prepare.sh` when the image or the checkpoint is not the one these settings name.
2. **Checks:** the arguments (tensorfold-native's own parser, in a throwaway container), the port, and free memory.
   A typo leaves a running server alone. `./start.sh restart` stops only after those checks pass.
3. **Launch:** `tensorfold-native serve` with the settings from `scripts/config.sh`.
4. **Loading:** the log, and a stop if `MemAvailable` falls below `MEM_FLOOR_GIB` (10 GiB).
5. **Smoke test**, then the LIVE message.

`FOREGROUND=1 ./start.sh` stays attached and exits with the container's exit code. Extra arguments go to
`tensorfold-native serve` after the defaults (`./start.sh restart --context 131072`). `DRY_RUN=1 ./start.sh` prints
the docker command and changes nothing.

**`scripts/prepare.sh`:**

1. Preflight: Docker, disk.
2. The image `tensorfold-ornith:zig-d1018b3`: TensorFold at `TF_REF` with every `patches/*.patch` applied
   (`git apply`), then `zig build -Doptimize=fast fatbins install native` with Zig 0.17.0, on
   `nvcr.io/nvidia/pytorch:26.07-py3`. It tries a matching prebuilt image from GitHub Container Registry first
   (`PULL=0` builds locally).
3. The checkpoint, at the pinned revision, and (`DRAFTS=1`) the DFlash2 draft snapshot at its pinned revision.
4. A check that `config.json` has `model_type` `qwen3_5_moe` and `quantization_config.quant_method` `exl3`, and that all
   9 shards named in `model.safetensors.index.json` are present; for the drafter, that `config.json` has `dflash_config`
   and `model.safetensors` is present.

```bash
scripts/prepare.sh
scripts/prepare.sh --rebuild
PREPARE=1 ./start.sh restart
```

## KV and memory

KV is FP8 by default (`KV_DTYPE=fp8`). It is lossy, so a free-running reply can differ from the bf16 one; its
agreement with a bf16 cache is not measured on this model yet. `KV_DTYPE=bf16` is the exact cache.

Ornith has 12 full-attention layers (2 kv heads, head_dim 256) and 36 GatedDeltaNet layers. Only the 12 attention
layers have a per-token cache:

| | bytes a token | one 262,144-token sequence |
| --- | ---: | ---: |
| bf16 | 24,576 | 6.00 GiB |
| fp8 | 12,480 (provisional) | 3.05 GiB |

fp8 is e4m3 codes for k and v plus one 4-byte scale per row: 12 x 2 x (256 + 4) x 2. Each stream also holds a fixed
GatedDeltaNet state: 36 layers x 64 heads x 128 x 128 x 4 B = 151 MB, plus the conv tails (`GDN_MIB=160`).

The engine's sequence budget is `MemAvailable` after the weights load, minus `TENSORFOLD_MEMORY_RESERVE_GIB` (10 GiB).
These are computed figures; the pool the Zig engine reports at a given context beyond 36k has not been checked here. `prepare.sh` and
`start.sh` refuse to run when `MemAvailable` is under 10 GiB (`MEM_FLOOR_GIB`). Keep at least 10 GiB free under load. On
this machine, exhausting unified memory freezes it.

## Configuration

Every setting lives in [`scripts/config.sh`](scripts/config.sh). Override it from the environment
(`PARALLEL=4 ./start.sh`) or from `.env` next to `start.sh` (`KEY=value`; the environment wins).

| Variable | Default | Meaning |
| --- | --- | --- |
| `PARALLEL` | `4` | requests decoded together. One that would not fit (prompt plus `max_tokens`) is refused |
| `CONTEXT` | `262144` | prompt + reply window (the native window; no YaRN) |
| `KV_DTYPE` | `fp8` | lossy; `bf16` is exact |
| `THINKING` | `1` | think before answering; `0` passes `--no-thinking` |
| `DRAFTS` | `1` | speculative decoding with the DFlash2 drafter (`--drafter <snapshot>`); `0` passes `--no-drafts` |
| `DRAFTER` | `dflash2` | `mtp` is experimental: the checkpoint's own MTP layer drafts (exact, but slower than no drafts) |
| `DRAFT_ID` / `DRAFT_REVISION` | `incoai/Ornith-1.5-122B-DFlash2` / `a5ed7396…` | the DFlash2 draft model and its pinned commit |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `0.6` / `0.95` / `20` | default sampling, from the model's `generation_config.json`; a request's own values win |
| `MAX_TOKENS` | `32768` | the reply budget when a request sets none |
| `VISION` / `VISION_URLS` | `1` / `0` | image input (video is not supported by this engine) (`--vision`, 2 GiB workspace); `VISION=0` is text only; `VISION_URLS=1` also accepts public `https://` URLs |
| `SERVED_NAME` | `Ornith-1.5-122B` | the model id in `/v1/models` |
| `PORT` / `HOST` | `8888` / `0.0.0.0` | where the API listens |
| `WEIGHTS_GIB` | `64` (`65` with DFlash2) | memory the weights and drafter take, for the memory check (measured: 60.25 GiB model, ~0.4 GB drafter) |
| `GDN_MIB` | `160` | GatedDeltaNet state a stream, for the memory check |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | `10` | kept free when the engine sizes the pool. `prepare.sh` and `start.sh` refuse a Spark below this |
| `MODEL_ID` / `MODEL_REVISION` | the Mia-AiLab checkpoint / `9c5c913b…` | the checkpoint and the commit this recipe serves |

### Thinking and sampling

By default the model thinks first. Per request, `temperature`, `top_p`, `top_k` and `seed` override the defaults
(`temperature: 0` is greedy). `"chat_template_kwargs": {"enable_thinking": false}` answers without thinking. A small
`max_tokens` can return empty `content` because the model was still thinking.

### API notes

The server is TensorFold's OpenAI-compatible server (`/v1/chat/completions`, `/v1/completions`, `/v1/models`,
`/health`, `/stats`, `/metrics` and more). Tested on this recipe: chat completions with greedy decoding, three image inputs, one tool call, and the thinking/answer split. Other endpoints and features are not verified.

## What the patches are

`patches/*.patch` add EXL3 support for the Qwen3.5-MoE CUDA engine to TensorFold `d1018b31`, applied in order with
`git apply` in the checkout root (all 27 apply cleanly on a clean checkout):

- `0001` bf16 embedding and dense-layer kernels for the Qwen3.5 path.
- `0002` scratch reserve for the EXL3 dense kernels.
- `0003` geometry check accepts EXL3 checkpoints.
- `0004` the EXL3 loader and engine, including the routed (MoE) experts.
- `0005` MTP drafter on EXL3 (opt-in with `DRAFTER=mtp`; exact, slower than no drafts).
- `0006` vision tower: accept the Hugging Face tensor names.
- `0007` DFlash2 drafter for the EXL3 engine (exact; needs `--drafter <dflash2 dir>`, which `start.sh` passes).

- `0008`-`0010` prompt (prefill) performance: fused mixed-width expert kernels, tensor-core attention, chunked GDN scan.
- `0011`-`0013` fixes for the larger row count (vision position stride, MTP absorb, narrow verify windows).
- `0014`-`0019` faster router and dense prompt kernels, 8,192-row prompt passes with mid-pass states, `TF_LOGITS_HASH` debug switch.
- `0020`-`0027` decode performance: cp.async expert ring, coalesced dense loads, `gd` decode kernels, GDN state ping-pong, batched decode attention, side-stream shared expert, PDL, cheaper verify windows.

`patches-dflash/` keeps the original DFlash2 patch file (`0201`), identical to `0007`.

### MTP finding

The MTP drafter works and is exact, but it is slower than no drafts (0.87x). EXL3 MoE decoding is memory-bandwidth
bound, and the extra rows of a verify step route to extra experts, so each step reads more expert weights than a single
token does; the accepted tokens do not pay for that. DFlash2 drafts a block in one pass and wins on code and prose. MTP
stays opt-in for experiments.

## Caveats

- Tested: greedy exactness against the no-draft run (8 of 8 prompts), vision with 3 images, one tool call, the thinking/answer split.
- Not tested: sampling with drafts (temperature above 0), multi-stream decoding (`PARALLEL` above 1 under load), context
  quality above 32k, tensor parallel on two Sparks (TP2).
- Memory is tight on 128 GB unified memory: keep other GPU work off the machine.
- Performance numbers are from one GB10 on 2026-10-21. `--parallel 2` aggregate decode is only ~20 tok/s.

## Repository layout

```
start.sh      set up (first run) and start the server
stop.sh       stop it
scripts/      prepare.sh, config.sh, publish-image.sh, banner.sh
patches/      EXL3, MTP, vision names and DFlash2 for the Qwen3.5-MoE engine, applied onto TensorFold at TF_REF
patches-dflash/ the original DFlash2 patch (same as patches/0007)
.github/      issue and pull request templates, GitHub Sponsors
CREDITS.md    who and what this builds on
CHANGELOG.md  upgrade notes
NOTICE        third-party notices
```

## License

MIT, see [`LICENSE`](LICENSE). The model weights, downloaded from Hugging Face and not part of this repository, are
under the license on the checkpoint's model card.

**Third-party software in the image.** The image `scripts/prepare.sh` builds is based on NVIDIA's PyTorch container
`nvcr.io/nvidia/pytorch:26.07-py3`, redistributed as a value-added runtime image. The NVIDIA software in it is
governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and the [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/),
which the container prints at every start. By pulling or running the image you accept them. The MIT license above
covers this repository's scripts only.

## Credits

- **Model and quantization:** [Mia's AI Lab](https://huggingface.co/Mia-AiLab) (Ornith-1.5-122B and its EXL3 EAQv1 4-bpw checkpoint).
- **EXL3 format:** [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp.
- **Draft model:** [DFlash2 drafter for Ornith-1.5-122B](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2) by incoai (Apache-2.0); DFlash by [z-lab](https://github.com/z-lab).
- **Reference recipe:** the upstream Qwen3.8-Flash-Next single-Spark recipe this one is adapted from.
- **Engine:** [TensorFold](https://github.com/ashhart/TensorFold) and its Zig engine (`tensorfold-native`) by Ash Hart
  ([ashhart](https://github.com/ashhart)) and the TensorFold contributors.
- **Patches:** everything in [`patches/`](patches/) (EXL3 for the Qwen3.5-MoE engine, the DFlash2 and MTP drafters, the prefill and
  decode kernels) is MiaAI-Lab's own work.

Full list: [`CREDITS.md`](CREDITS.md).
