# Shared settings for start.sh, stop.sh and scripts/*.sh. A setting's value comes from the first of these that sets it:
#   1. the environment: `PORT=9000 ./start.sh`, `PULL=0 scripts/prepare.sh`
#   2. ./.env (KEY=value lines, read, never run)
# Settings marked PROVISIONAL follow TensorFold's Zig engine (tensorfold-native) plus this repo's patches and change as they land.
if [[ -f .env ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < .env
fi
unset _line _key _value

# Checkpoint: Mia's AI Lab EXL3 EAQv1 4-bpw quant of Ornith-1.5-122B (Qwen3.5-MoE architecture, 9 safetensors shards,
# 62 GiB on disk), pinned to one revision. The engine reads the EXL3 format from the checkpoint.
# WEIGHTS_GIB is what the Zig engine takes at start, scratch for 8192-row prompt passes included: 69.0 GiB measured (69.5 with
# the DFlash2 drafter), so 69 plus 1 with a drafter.
MODEL_ID="${MODEL_ID:-Mia-AiLab/Ornith-1.5-122B-EXL3-EAQv1-4-bpw}"
_pin=1; [[ "$MODEL_ID" == "Mia-AiLab/Ornith-1.5-122B-EXL3-EAQv1-4-bpw" ]] || _pin=0   # another MODEL_ID takes no revision pin
(( ! _pin )) || : "${MODEL_REVISION:=9c5c913b3de1eef901520d6fedaf450044ea0423}"
MODEL_REVISION="${MODEL_REVISION:-}"
unset _pin
CKPT_GIB=62
# Draft model for speculative decoding (DRAFTS=1, the default): incoai's DFlash2 drafter for Ornith-1.5-122B (Apache-2.0,
# 1.86 GB), pinned to one revision. DRAFTER=dflash2 (default) uses it; DRAFTER=mtp is EXPERIMENTAL: the checkpoint's own MTP
# layer drafts instead (exact, but measured SLOWER than no drafts, 0.87x). DRAFTS=0 serves without drafts.
DRAFT_ID="${DRAFT_ID:-incoai/Ornith-1.5-122B-DFlash2}"
DRAFT_REVISION="${DRAFT_REVISION:-a5ed7396f5f9e5b1abf7ff50b7899a3186961432}"
DRAFT_GIB=2
DRAFTS="${DRAFTS:-1}"
DRAFTER="${DRAFTER:-dflash2}"
# True when the DFlash2 draft snapshot is needed: drafts on and the dflash2 drafter chosen.
need_draft() { [[ "$DRAFTS" == 1 && "$DRAFTER" == dflash2 ]]; }
# Measured: the model takes 60.25 GiB on the device; 64 stays as a conservative figure. The DFlash2 drafter adds ~0.4 GB
# device memory (counted as 1 GiB here, whole numbers).
WEIGHTS_GIB="${WEIGHTS_GIB:-$(( 69 + ( $(need_draft && echo 1 || echo 0) ) ))}"
QUANT_LABEL="EXL3 EAQv1, 4.049 bpw trellis average"

# The engine: TensorFold's Zig engine, built from TF_REPO at TF_REF with patches/*.patch applied (git apply in the
# checkout's root), with Zig ZIG_VERSION. The patches add EXL3 for the Qwen3.5-MoE CUDA engine; upstream TensorFold does
# not read EXL3 exports of this model yet. Only the Zig build goes into the image.
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_REF="${TF_REF:-d1018b318fb51981c4a18f12e40269a4d3ee9713}"
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
TF_TAG="zig-${TF_REF:0:7}"
IMAGE="${IMAGE:-tensorfold-ornith:${TF_TAG}}"
# The engine's own (native) kernels, built by `zig build ... fatbins install native`; there is no captured kernel set.
KERNELS_PATH=native
image_hash() {
  ( cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
    cat patches/*.patch 2>/dev/null
    sed -n "/<<'DOCKERFILE'/,/^DOCKERFILE\$/p" scripts/prepare.sh
    echo "$TF_REPO $TF_REF zig-$ZIG_VERSION $ZIG_SHA256 $BASE_IMAGE tp-1"
  ) | sha256sum | cut -c1-12
}
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/miaai-lab/ornith-1.5-122b-tensorfold}"
IMAGE_TAG="${IMAGE_TAG:-}"
IMAGE_DIGEST="${IMAGE_DIGEST:-}"
prebuilt_image() {
  local tag="${TF_TAG}-$(image_hash)"
  if [[ "$tag" == "$IMAGE_TAG" && -n "$IMAGE_DIGEST" ]]; then echo "$GHCR_IMAGE@$IMAGE_DIGEST"; else echo "$GHCR_IMAGE:$tag"; fi
}
CONTAINER_NAME="${CONTAINER_NAME:-ornith-122b-tf}"

SERVED_NAME="${SERVED_NAME:-Ornith-1.5-122B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
# Native window 262,144 tokens (no YaRN here). One full fp8 window is ~3.1 GiB (bf16 ~6 GiB) beside ~64 GiB of weights.
CONTEXT="${CONTEXT:-262144}"
# Requests decoded together. The engine admits each one inside its own budget (MemAvailable after load, less
# TENSORFOLD_MEMORY_RESERVE_GIB) and refuses one that would not fit. The pool is shared: PARALLEL is a cap, not a
# promise that every stream can sit at a full window at once. Each stream also holds a fixed GatedDeltaNet state (GDN_MIB).
PARALLEL="${PARALLEL:-4}"
# KV cache (--kv-cache). fp8 is the default: about 1.9x the pool of bf16, and lossy. bf16 is exact: KV_DTYPE=bf16.
KV_DTYPE="${KV_DTYPE:-fp8}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
THINKING="${THINKING:-1}"
# Drafts: DRAFTS (above) is 1 by default and means DFlash2 drafts (DRAFTER=dflash2); DRAFTS=0 passes --no-drafts.
# Sampling defaults from the model's generation_config.json (a request's own values win).
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
# Image input (video is rejected by this engine): VISION=1 is the default (tested with images). It passes --vision and keeps a workspace out of the KV
# budget. VISION=0 serves text only. VISION_URLS=1 also accepts public https:// URLs.
VISION="${VISION:-1}"
VISION_URLS="${VISION_URLS:-0}"
export TENSORFOLD_MAX_IMAGES="${TENSORFOLD_MAX_IMAGES:-50}"
MAX_VIDEOS="${MAX_VIDEOS:-4}"
export TENSORFOLD_IMAGE_TOKENS="${TENSORFOLD_IMAGE_TOKENS:-16384}"
export TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"
export TENSORFOLD_VISION_WORKSPACE_MIB="${TENSORFOLD_VISION_WORKSPACE_MIB:-2048}"
# The engine keeps this much MemAvailable free when it sizes the KV pool. prepare.sh and start.sh refuse a Spark
# that has less than this free.
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-10}"
export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"

# Memory at start. Weights WEIGHTS_GIB. One full window's cache is kv_gib below, plus one stream's GDN state. Prefill
# scratch is outside the engine's cache budget; 8 GiB here is slack for a prompt chunk, not a measured peak.
# MEM_FLOOR_GIB is the line start.sh stops the server at while it loads, and the line prepare.sh refuses to start
# below. MEM_CHECK=0 turns both into a warning.
MEM_NEED_GIB="${MEM_NEED_GIB:-}"
MEM_FLOOR_GIB="${MEM_FLOOR_GIB:-$TENSORFOLD_MEMORY_RESERVE_GIB}"
# GatedDeltaNet state a stream: 36 layers x 64 heads x 128 x 128 x 4 B = 151 MB (144 MiB) plus the conv tails.
GDN_MIB="${GDN_MIB:-160}"
# kv_gib <tokens>: one sequence's full-attention cache. 12 full-attention layers x 2 kv heads x head_dim 256.
# bf16: 12 x 2 x 256 x 2 (k+v) x 2 B = 24,576 bytes a token.
# fp8 (e4m3 codes + one 4-byte scale per row, k and v): 12 x 2 x (256 + 4) x 2 = 12,480 bytes a token. PROVISIONAL until measured.
kv_gib() {
  local ctx=${1:-$CONTEXT} bytes=24576
  [[ "$ctx" =~ ^[0-9]+$ ]] || ctx=$CONTEXT
  [[ "$KV_DTYPE" == fp8 ]] && bytes=12480
  echo $(( (bytes * ctx + 1073741823) / 1073741824 ))
}
gdn_gib() { echo $(( (${1:-1} * GDN_MIB + 1023) / 1024 )); }
# mem_need_gib <context>: MemAvailable a start needs: weights, one full window, one GDN state, scratch slack, the vision
# workspace when VISION=1, floor, 2 GiB.
mem_need_gib() {
  local ctx=${1:-$CONTEXT} vision=0
  [[ "$VISION" == 1 ]] && vision=2
  echo $(( WEIGHTS_GIB + $(kv_gib "$ctx") + $(gdn_gib 1) + 8 + vision + MEM_FLOOR_GIB + 2 ))
}

HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
KERNEL_CACHE="${KERNEL_CACHE:-$HOME/.cache/tensorfold-ornith}"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/ornith-122b-tf}"
LOG_DIR="${LOG_DIR:-$HOME/.cache/tensorfold-ornith/logs}"
LOG_KEEP="${LOG_KEEP:-10}"
MIN_FREE_GB="${MIN_FREE_GB:-$((CKPT_GIB + 18 + ( $(need_draft && echo "$DRAFT_GIB" || echo 0) )))}"
IMAGE_FREE_GB="${IMAGE_FREE_GB:-40}"

_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

[[ -z "${HF_TOKEN:-}" ]] || export HF_TOKEN
model_cache_dir() { echo "$HF_CACHE/hub/models--${MODEL_ID//\//--}"; }
snapshot_rev() {
  if [[ -n "$MODEL_REVISION" ]]; then echo "$MODEL_REVISION"
  else cat "$(model_cache_dir)/refs/main" 2>/dev/null; fi
}

draft_cache_dir() { echo "$HF_CACHE/hub/models--${DRAFT_ID//\//--}"; }
draft_snapshot_rev() { echo "${DRAFT_REVISION:-$(cat "$(draft_cache_dir)/refs/main" 2>/dev/null)}"; }

PREPARED_MARKER="$STATE_DIR/prepared"
prepared_state() {
  local hash label rev
  hash=$(image_hash)
  label=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo missing)
  rev=$(snapshot_rev)
  echo "model=$MODEL_ID@$rev image=$label patches=$hash draft=$(need_draft && echo "$DRAFT_ID@$(draft_snapshot_rev)" || echo none)"
}
