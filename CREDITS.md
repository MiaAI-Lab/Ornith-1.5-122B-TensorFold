# Credits

This repository is a thin layer of scripts and patches. Almost everything that makes it work was built by others. Its own
work is licensed under the MIT License ([`LICENSE`](LICENSE)). The image `scripts/prepare.sh` builds also carries
TensorFold's license files from the pinned checkout.

## Model and quantization

- **[Ornith-1.5-122B](https://huggingface.co/Mia-AiLab/Ornith-1.5-122B-EXL3-EAQv1-4-bpw)**: the model (Qwen3.5-MoE
  architecture) and its 4-bpw EXL3 EAQv1 quantization, the checkpoint this recipe serves, by
  **[Mia's AI Lab](https://huggingface.co/Mia-AiLab)**. Used under the license on its model card.
- **[ExLlamaV3](https://github.com/turboderp-org/exllamav3)** by **turboderp** (MIT): the EXL3 format and the quantizer
  the checkpoint is made with.

## Draft model

- **[Ornith-1.5-122B-DFlash2](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2)** by **incoai** (Apache-2.0): the
  DFlash2 draft model that makes decoding up to ~2x faster. DFlash is by [z-lab](https://github.com/z-lab).

## Inference engine

- **[TensorFold](https://github.com/ashhart/TensorFold)** and its Zig engine (`tensorfold-native`) by Ash Hart
  ([ashhart](https://github.com/ashhart)) and the TensorFold contributors (Apache 2.0 from v0.6.0; code written before
  v0.6.0 keeps its MIT notice): the engine that serves the model, its CUDA runtime and kernels and the
  OpenAI-compatible server. `scripts/prepare.sh` builds it from TensorFold's source at the commit pinned in
  `scripts/config.sh`. The patches in `patches/` are
  **MiaAI-Lab's own work**: EXL3 loading and serving for
  the Qwen3.5-MoE CUDA engine, the DFlash2 and MTP drafters for it, the prefill and decode kernels and the fixes around
  them. They modify TensorFold's source and are applied to it at build time.
- **The Zig CUDA serving path and the CUDA family registry** were authored by Jürgen Schmied
  ([jschmied](https://github.com/jschmied)) in [TensorFold PR #443](https://github.com/ashhart/TensorFold/pull/443).
- **[Zig](https://ziglang.org)** by the Zig Software Foundation and the Zig contributors (MIT): the language and
  compiler TensorFold's Zig engine is written in and built with (0.17.0).

## Recipe

- The scripts (`start.sh`, `stop.sh`, `scripts/`) are MiaAI-Lab's, adapted from its
  [Qwen3.8-Flash-Next single-Spark TensorFold recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold).
- FP8 KV (`KV_DTYPE=fp8`) follows MiaAI-Lab's GLM recipe patch `0038-glm-kv-fp8` in
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold).

## Runtime stack

- **[NVIDIA PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)**
  (`nvcr.io/nvidia/pytorch:26.07-py3`), the base of the image, with NVIDIA's CUDA (nvcc builds the engine's kernels).
  Governed by the NVIDIA Software License Agreement and the Product-Specific Terms for NVIDIA AI Products.
- **[Hugging Face Hub](https://huggingface.co/)**: model hosting, the `hf` CLI and `huggingface_hub` (Apache 2.0), and
  the [safetensors](https://github.com/huggingface/safetensors) format (Apache 2.0) the checkpoint ships in.
- **[Docker](https://www.docker.com/)** and the
  **[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)** (Apache 2.0).

## Hardware

- **[NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)** (GB10 Grace Blackwell,
  128 GB unified memory).

If you believe something here is missing or credited wrongly, please open an issue.
