#!/usr/bin/env bash
# Launch one ClarifyRL GRPO training job on HF Jobs.
#
# Usage:
#   HF_TOKEN=hf_xxx ./scripts/launch_hf_job.sh \
#       --model Qwen/Qwen3-0.6B \
#       --flavor a10g-small \
#       --steps 300
#
# Or as positional shortcuts:
#   HF_TOKEN=hf_xxx ./scripts/launch_hf_job.sh Qwen/Qwen3-0.6B a10g-small 300
#
# Smoke test (5 steps, NUM_GEN=2, no Hub push, ~$0.50, ~10min):
#   HF_TOKEN=hf_xxx SMOKE=1 ./scripts/launch_hf_job.sh Qwen/Qwen3-0.6B a10g-small
#
# Required env:
#   HF_TOKEN          write token of the account that owns this job (must be Pro/Team)
#
# Optional env:
#   ENV_BASE_URL      env Space URL (default: https://agarwalanu3103-clarify-rl.hf.space)
#   OUTPUT_DIR        repo name to push trained model to
#   TRACKIO_SPACE_ID  trackio space (default = OUTPUT_DIR)
#   TIMEOUT           HF Jobs timeout (default: 8h, 30m for SMOKE)
#   IMAGE             docker image override (default: HF Jobs' uv-enabled python3.12-bookworm)
#   SEED              training seed (default 42; use 84 for backup runs)
#   NUM_GENERATIONS   GRPO group size (default: auto-tuned by GPU)
#   GRAD_ACCUM_STEPS  default 8
#   LEARNING_RATE     default 1e-6
#   SAVE_STEPS        default 25
#   RESUME_FROM_CKPT  resume from a specific checkpoint dir
#   SMOKE             "1" → run 5 steps with no push (preflight)
#   DRY_RUN           "1" → print the command instead of executing
#
# Available flavours (verified from `hf jobs hardware` 2026-04-25):
#   cpu-basic, cpu-upgrade, cpu-performance, cpu-xl
#   t4-small, t4-medium       — 16 GB VRAM     ($0.40 / $0.60 per hour)
#   l4x1, l4x4                — 24 GB / 96 GB  ($0.80 / $3.80 per hour)
#   a10g-small, a10g-large    — 24 GB          ($1.00 / $1.50 per hour)
#   a10g-largex2, a10g-largex4 — 48 GB / 96 GB ($3.00 / $5.00 per hour)
#   l40sx1, l40sx4, l40sx8    — 48 / 192 / 384 GB ($1.80 / $8.30 / $23.50)
#   a100-large                — 1x A100 (80 GB)   ($2.50/hr)
#   a100x4, a100x8            — 4 / 8 x A100 (320/640 GB) ($10 / $20 per hour)
#   h200, h200x2..x8          — 1x H200 (141 GB)  ($5.00/hr) ← fastest single GPU
# NOTE: 'h100-large' does NOT exist — use 'h200' for biggest single-GPU jobs.

set -euo pipefail

# ---------- arg parsing ------------------------------------------------------
MODEL="Qwen/Qwen3-0.6B"
FLAVOR="a10g-small"
MAX_STEPS="300"

# positional shortcut: ./launch_hf_job.sh MODEL FLAVOR STEPS
if [ "$#" -ge 1 ] && [ "${1:0:2}" != "--" ]; then
    MODEL="${1}"
    [ "$#" -ge 2 ] && FLAVOR="${2}"
    [ "$#" -ge 3 ] && MAX_STEPS="${3}"
else
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --model)   MODEL="$2"; shift 2;;
            --flavor)  FLAVOR="$2"; shift 2;;
            --steps)   MAX_STEPS="$2"; shift 2;;
            --output)  OUTPUT_DIR="$2"; shift 2;;
            --image)   IMAGE="$2"; shift 2;;
            --timeout) TIMEOUT="$2"; shift 2;;
            --smoke)   SMOKE=1; shift 1;;
            -h|--help)
                grep '^#' "$0" | sed 's/^# \{0,1\}//'
                exit 0;;
            *)
                echo "Unknown arg: $1" >&2
                exit 1;;
        esac
    done
fi

# ---------- defaults --------------------------------------------------------
: "${HF_TOKEN:?HF_TOKEN is required (write token for the account running this job)}"
: "${ENV_BASE_URL:=https://agarwalanu3103-clarify-rl.hf.space}"
# Default to HF Jobs' built-in uv-enabled image. Custom images MUST have uv
# pre-installed or the job exits 127. Pass IMAGE=... to override.
: "${IMAGE:=}"
: "${SMOKE:=0}"
: "${SEED:=42}"

# Smoke runs: short timeout, no push, no resume
if [ "$SMOKE" = "1" ]; then
    : "${TIMEOUT:=30m}"
    MAX_STEPS="5"
fi
: "${TIMEOUT:=8h}"

MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]' | tr '.' '-')
DEFAULT_OUTPUT="clarify-rl-grpo-${MODEL_SHORT}"
if [ "$SMOKE" = "1" ]; then
    DEFAULT_OUTPUT="${DEFAULT_OUTPUT}-smoke"
fi
: "${OUTPUT_DIR:=$DEFAULT_OUTPUT}"
: "${TRACKIO_SPACE_ID:=$OUTPUT_DIR}"

# ---------- validate flavor against known list -------------------------------
KNOWN_FLAVORS=(
    cpu-basic cpu-upgrade cpu-performance cpu-xl
    t4-small t4-medium
    l4x1 l4x4
    a10g-small a10g-large a10g-largex2 a10g-largex4
    l40sx1 l40sx4 l40sx8
    a100-large a100x4 a100x8
    h200 h200x2 h200x4 h200x8
)
flavor_ok=0
for f in "${KNOWN_FLAVORS[@]}"; do
    if [ "$f" = "$FLAVOR" ]; then
        flavor_ok=1
        break
    fi
done
if [ "$flavor_ok" = "0" ]; then
    echo "WARNING: '$FLAVOR' is not in the known list; proceeding anyway." >&2
    echo "  Known flavors: ${KNOWN_FLAVORS[*]}" >&2
fi

# ---------- show plan -------------------------------------------------------
cat <<EOF
=========================================================================
ClarifyRL GRPO HF Jobs launcher
=========================================================================
  Model:              $MODEL
  Flavor:             $FLAVOR
  Max steps:          $MAX_STEPS
  Image:              $IMAGE
  Timeout:            $TIMEOUT
  Seed:               $SEED
  Output repo:        $OUTPUT_DIR
  Trackio space:      $TRACKIO_SPACE_ID
  Env base URL:       $ENV_BASE_URL
  Smoke test:         $SMOKE
=========================================================================
EOF

# ---------- locate the training script (must exist locally) -----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/training/train_grpo.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: training script not found at $TRAIN_SCRIPT" >&2
    exit 1
fi

# ---------- assemble the hf jobs uv run command -----------------------------
# We pass HF_TOKEN as a SECRET (encrypted server-side) and everything else
# as plain env vars. UV will install the deps listed via --with at job start.

CMD=(
    hf jobs uv run
    --flavor "$FLAVOR"
    --timeout "$TIMEOUT"
    --secrets "HF_TOKEN=$HF_TOKEN"
    --token "$HF_TOKEN"
    -e "MODEL_NAME=$MODEL"
    -e "MAX_STEPS=$MAX_STEPS"
    -e "OUTPUT_DIR=$OUTPUT_DIR"
    -e "TRACKIO_SPACE_ID=$TRACKIO_SPACE_ID"
    -e "ENV_BASE_URL=$ENV_BASE_URL"
    -e "SEED=$SEED"
    -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    -e "TRL_EXPERIMENTAL_SILENCE=1"
)

# Forward optional knobs only if explicitly set (lets the script auto-tune)
[ -n "${NUM_GENERATIONS:-}" ] && CMD+=(-e "NUM_GENERATIONS=$NUM_GENERATIONS")
[ -n "${GRAD_ACCUM_STEPS:-}" ] && CMD+=(-e "GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS")
[ -n "${LEARNING_RATE:-}" ] && CMD+=(-e "LEARNING_RATE=$LEARNING_RATE")
[ -n "${BETA:-}" ] && CMD+=(-e "BETA=$BETA")
[ -n "${SAVE_STEPS:-}" ] && CMD+=(-e "SAVE_STEPS=$SAVE_STEPS")
[ -n "${MAX_COMPLETION_LEN:-}" ] && CMD+=(-e "MAX_COMPLETION_LEN=$MAX_COMPLETION_LEN")
[ -n "${VLLM_GPU_MEM_UTIL:-}" ] && CMD+=(-e "VLLM_GPU_MEM_UTIL=$VLLM_GPU_MEM_UTIL")
[ -n "${RESUME_FROM_CKPT:-}" ] && CMD+=(-e "RESUME_FROM_CKPT=$RESUME_FROM_CKPT")

if [ "$SMOKE" = "1" ]; then
    CMD+=(-e "SMOKE_TEST=1")
fi

# Detach by default so the laptop doesn't have to keep the launcher process
# alive for the entire training run. Set DETACH=0 to stream logs.
: "${DETACH:=1}"
if [ "$DETACH" = "1" ]; then
    CMD+=(-d)
fi

CMD+=(
    # IMPORTANT — package compat notes:
    #
    # vllm + trl + transformers form a 3-way dependency knot. UV's strict
    # resolver insists on a valid graph or refuses to install. Two known-bad
    # combos that the smoke runs surfaced:
    #
    #   1. vllm 0.17.x + trl 0.27 + transformers @ main  →  installs cleanly
    #      but at runtime trl's vllm_client unconditionally imports
    #      `vllm_ascend.distributed.device_communicators.pyhccl`. That
    #      sub-module ships under vllm 0.17's plugin namespace stub but
    #      isn't actually present on x86/CUDA → ModuleNotFoundError, exit 1.
    #   2. vllm <=0.15.x + trl main + transformers @ main  →  resolver
    #      conflict: those vllm versions pin transformers<5 while trl main
    #      requires transformers>=4.56.2 (so transformers @ main, which is
    #      5.x, is incompatible with vllm).
    #   3. Adding `vllm-ascend` to fix combo 1's runtime crash fails because
    #      vllm-ascend pins torch-npu==2.5.1 which only has wheels for
    #      cp39/cp310/cp311 — uv's default Python is cp312. Resolver dies.
    #
    # Working combo: vllm latest (no pin, → 0.17.x) + trl latest +
    # transformers @ main, with the `vllm_ascend` detection neutralised by
    # a monkey-patch in train_grpo.py (overrides `importlib.util.find_spec`
    # to return None for `vllm_ascend`, so TRL's `is_vllm_ascend_available()`
    # reports False and the unconditional `from vllm_ascend...` import in
    # `trl/generation/vllm_client.py` is never reached).
    # Pin trl to >= 1.0 so we get `chat_template_kwargs` and other
    # Qwen3-friendly knobs. Without a pin, uv's resolver silently picks
    # an older trl (0.21.x) when transformers @ main + vllm[vllm extra]
    # constraints overlap, and `chat_template_kwargs` blows up at runtime.
    --with "vllm"
    --with "trl[vllm]>=1.0"
    --with "trackio"
    --with "websockets"
    --with "datasets"
    --with "jmespath"
    --with "bitsandbytes"
    --with "huggingface_hub"
    # TRL <1.0 eagerly imports mergekit (via trl/mergekit_utils.py top-level
    # `from mergekit.config import …`). TRL 1.0+ removed mergekit_utils.py
    # entirely, so the trl>=1.0 pin above lets us SKIP installing mergekit —
    # which is essential because all mergekit versions pin pydantic <=2.10.6
    # while vllm 0.10.2+ pins pydantic >=2.11.7 (3-way unsatisfiable graph).
    #
    # peft is still eagerly imported by callbacks.py for PeftModel, so it
    # stays in.
    #
    # `llm_blender` is NOT installed because it imports `TRANSFORMERS_CACHE`
    # from `transformers.utils.hub`, removed in transformers 5.x. We stub it
    # via a fake sys.modules entry in train_grpo.py instead.
    --with "peft"
    --with "git+https://github.com/huggingface/transformers.git@main"
    "$TRAIN_SCRIPT"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1 — would run:"
    printf '  %q\n' "${CMD[@]}"
    exit 0
fi

# ---------- preflight: make sure `hf` CLI exists ---------------------------
# Prefer the venv's patched hf (which injects truststore for macOS / corporate
# proxy SSL handling). Fall back to whichever hf is on PATH.
VENV_HF="$SCRIPT_DIR/.venv/bin/hf"
if [ -x "$VENV_HF" ]; then
    HF_BIN="$VENV_HF"
elif command -v hf >/dev/null 2>&1; then
    HF_BIN="$(command -v hf)"
else
    echo "ERROR: 'hf' CLI not found. Install with: pip install -U 'huggingface_hub[cli]'" >&2
    exit 1
fi
# Replace the leading 'hf jobs uv run' tokens with the resolved binary.
CMD[0]="$HF_BIN"

# ---------- run -------------------------------------------------------------
echo "Launching with: $HF_BIN"
echo
"${CMD[@]}"
