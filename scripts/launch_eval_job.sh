#!/usr/bin/env bash
# Launch a vLLM-powered eval as a HF Job for a trained ClarifyRL checkpoint.
#
# Usage:
#   HF_TOKEN=hf_xxx ./scripts/launch_eval_job.sh \
#       --model agarwalanu3103/clarify-rl-grpo-qwen3-0-6b \
#       --flavor a10g-small \
#       --limit 50
#
# Or as positional shortcuts:
#   HF_TOKEN=hf_xxx ./scripts/launch_eval_job.sh agarwalanu3103/clarify-rl-grpo-qwen3-0-6b a10g-small 50
#
# This works around the fact that HF Inference Router does not auto-warm
# fine-tuned community uploads — vllm must be hosted ourselves. We use the
# cheapest GPU that fits the model: a10g-small (24 GB) for ≤4B, a10g-large
# for 7-8B.
#
# Environment:
#   HF_TOKEN          (required) write token of the account hosting the eval.
#   ENV_BASE_URL      env Space URL (default: agarwalanu3103-clarify-rl).
#   PUSH_TO_REPO      override push target (default = MODEL).
#   EVAL_LABEL        suffix for output filename (default n${LIMIT}).
#   GPU_MEM_UTIL      vLLM GPU mem util (default 0.85).
#   TIMEOUT           HF Jobs timeout (default 1h).
#   IMAGE             docker image override.
#
# Example multi-checkpoint sweep:
#   for m in clarify-rl-grpo-qwen3-0-6b clarify-rl-grpo-qwen3-1-7b; do
#     HF_TOKEN=$HF_TOKEN ./scripts/launch_eval_job.sh agarwalanu3103/$m a10g-small 50
#   done

set -euo pipefail

MODEL=""
FLAVOR="a10g-small"
LIMIT="50"

if [ "$#" -ge 1 ] && [ "${1:0:2}" != "--" ]; then
    MODEL="${1}"
    [ "$#" -ge 2 ] && FLAVOR="${2}"
    [ "$#" -ge 3 ] && LIMIT="${3}"
else
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --model)   MODEL="$2"; shift 2;;
            --flavor)  FLAVOR="$2"; shift 2;;
            --limit)   LIMIT="$2"; shift 2;;
            --image)   IMAGE="$2"; shift 2;;
            --timeout) TIMEOUT="$2"; shift 2;;
            -h|--help)
                grep '^#' "$0" | sed 's/^# \{0,1\}//'
                exit 0;;
            *)
                echo "Unknown arg: $1" >&2
                exit 1;;
        esac
    done
fi

: "${MODEL:?MODEL is required (e.g. agarwalanu3103/clarify-rl-grpo-qwen3-0-6b)}"
: "${HF_TOKEN:?HF_TOKEN is required}"
: "${ENV_BASE_URL:=https://agarwalanu3103-clarify-rl.hf.space}"
: "${PUSH_TO_REPO:=$MODEL}"
: "${EVAL_LABEL:=n${LIMIT}}"
: "${GPU_MEM_UTIL:=0.85}"
: "${TIMEOUT:=1h}"
: "${IMAGE:=}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/scripts/eval_with_vllm.py"
RUN_EVAL="$SCRIPT_DIR/scripts/run_eval.py"
INFERENCE_PY="$SCRIPT_DIR/inference.py"
SCENARIOS="$SCRIPT_DIR/scenarios/eval_held_out.json"

for f in "$EVAL_SCRIPT" "$RUN_EVAL" "$INFERENCE_PY" "$SCENARIOS"; do
    [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done

cat <<EOF
=========================================================================
ClarifyRL vLLM eval HF Jobs launcher
=========================================================================
  Model:              $MODEL
  Flavor:             $FLAVOR
  Limit:              $LIMIT
  Push target:        $PUSH_TO_REPO
  Eval label:         $EVAL_LABEL
  Env base URL:       $ENV_BASE_URL
  GPU mem util:       $GPU_MEM_UTIL
  Timeout:            $TIMEOUT
  Image:              ${IMAGE:-<HF Jobs default uv-python>}
=========================================================================
EOF

CMD=(
    hf jobs uv run
    --flavor "$FLAVOR"
    --timeout "$TIMEOUT"
    --secrets "HF_TOKEN=$HF_TOKEN"
    --token "$HF_TOKEN"
    -e "MODEL_NAME=$MODEL"
    -e "ENV_BASE_URL=$ENV_BASE_URL"
    -e "PUSH_TO_REPO=$PUSH_TO_REPO"
    -e "LIMIT=$LIMIT"
    -e "EVAL_LABEL=$EVAL_LABEL"
    -e "GPU_MEM_UTIL=$GPU_MEM_UTIL"
    -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    -e "VLLM_USE_V1=1"
)

if [ -n "$IMAGE" ]; then
    CMD+=(--image "$IMAGE")
fi

: "${DETACH:=1}"
if [ "$DETACH" = "1" ]; then
    CMD+=(-d)
fi

# vLLM + openai (HTTP client used by run_eval.py via inference.py) +
# websockets (env Space connection) + huggingface_hub (Hub upload).
# We DO NOT pull `trl` here — eval is purely inference + HTTP.
CMD+=(
    --with "vllm"
    --with "openai>=1.40.0"
    --with "websockets>=12.0"
    --with "jmespath"
    --with "huggingface_hub"
    --with "truststore"
    "$EVAL_SCRIPT"
)

# Prefer the venv hf binary so SSL truststore patch applies.
VENV_HF="$SCRIPT_DIR/.venv/bin/hf"
if [ -x "$VENV_HF" ]; then
    HF_BIN="$VENV_HF"
elif command -v hf >/dev/null 2>&1; then
    HF_BIN="$(command -v hf)"
else
    echo "ERROR: 'hf' CLI not found." >&2
    exit 1
fi
CMD[0]="$HF_BIN"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1 — would run:"
    printf '  %q\n' "${CMD[@]}"
    exit 0
fi

echo "Launching with: $HF_BIN"
echo
"${CMD[@]}"
