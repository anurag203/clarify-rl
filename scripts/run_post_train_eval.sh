#!/usr/bin/env bash
# Run all evals after the production training jobs finish.
#
# This script orchestrates the post-training eval sweep:
#   1. Policy baseline (deterministic, no LLM)
#   2. Base model eval per size (Qwen3-0.6B / 1.7B / 4B, untrained)
#   3. Trained model eval per size (3 trained checkpoints)
#
# Outputs go to outputs/eval_<model>_<base|trained>.json — exactly the
# layout consumed by scripts/make_plots.py.
#
# Required env (one HF write-token with read access to the trained model repos):
#   HF_TOKEN          token for downloading the trained models from HF Hub
#
# Optional env:
#   ENV_BASE_URL      default https://agarwalanu3103-clarify-rl.hf.space
#   API_BASE_URL      default https://router.huggingface.co/v1 (HF Inference Router)
#   LIMIT             max scenarios to evaluate (default 100, set to 300 for full)
#   TIMEOUT_S         per-scenario timeout (default 60)
#   SKIP_POLICY       "1" to skip the policy baseline (already have it)
#   SKIP_BASE         "1" to skip base-model evals
#   SKIP_TRAINED      "1" to skip trained-model evals
#
# Usage:
#   HF_TOKEN=hf_xxx ./scripts/run_post_train_eval.sh
#
# Trained model repo names (these are the OUTPUT_DIRs from launch_all.sh):
#   <username>/clarify-rl-grpo-qwen3-0-6b
#   <username>/clarify-rl-grpo-qwen3-1-7b
#   <username>/clarify-rl-grpo-qwen3-4b
#
# Set MODEL_0_6B / MODEL_1_7B / MODEL_4B env vars if your usernames differ.

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN required (read access to trained model repos)}"
: "${ENV_BASE_URL:=https://agarwalanu3103-clarify-rl.hf.space}"
: "${API_BASE_URL:=https://router.huggingface.co/v1}"
: "${LIMIT:=100}"
: "${TIMEOUT_S:=60}"
: "${SKIP_POLICY:=0}"
: "${SKIP_BASE:=0}"
: "${SKIP_TRAINED:=0}"

# Defaults assume agarwalanu3103 owns the 0.6B run.
: "${MODEL_0_6B:=agarwalanu3103/clarify-rl-grpo-qwen3-0-6b}"
: "${MODEL_1_7B:=agarwalanu3103/clarify-rl-grpo-qwen3-1-7b}"
: "${MODEL_4B:=agarwalanu3103/clarify-rl-grpo-qwen3-4b}"

OUT_DIR="outputs"
mkdir -p "$OUT_DIR"

cat <<EOF
=========================================================================
ClarifyRL post-training eval sweep
=========================================================================
  Env Space:     $ENV_BASE_URL
  API Base URL:  $API_BASE_URL
  Limit:         $LIMIT scenarios
  Timeout:       ${TIMEOUT_S}s per scenario
  Trained 0.6B:  $MODEL_0_6B
  Trained 1.7B:  $MODEL_1_7B
  Trained 4B:    $MODEL_4B
  Output dir:    $OUT_DIR
=========================================================================
EOF

run_eval() {
    local mode="$1"
    local out_path="$2"
    local model="${3:-}"

    if [ -f "$out_path" ]; then
        echo "[SKIP] $out_path already exists (delete to re-run)"
        return 0
    fi

    echo
    echo "▶ Eval: mode=$mode out=$out_path model=${model:-N/A}"
    if [ "$mode" = "policy" ]; then
        ENV_BASE_URL="$ENV_BASE_URL" \
        python3 scripts/run_eval.py \
            --mode policy \
            --out "$out_path" \
            --limit "$LIMIT" \
            --timeout "$TIMEOUT_S"
    else
        MODEL_NAME="$model" \
        API_BASE_URL="$API_BASE_URL" \
        HF_TOKEN="$HF_TOKEN" \
        ENV_BASE_URL="$ENV_BASE_URL" \
        python3 scripts/run_eval.py \
            --mode api \
            --out "$out_path" \
            --limit "$LIMIT" \
            --timeout "$TIMEOUT_S"
    fi
}

# 1. Policy baseline
if [ "$SKIP_POLICY" != "1" ]; then
    run_eval policy "$OUT_DIR/eval_policy.json"
fi

# 2. Base-model evals (untrained Qwen3 family)
if [ "$SKIP_BASE" != "1" ]; then
    run_eval api "$OUT_DIR/eval_qwen3-0.6b_base.json" "Qwen/Qwen3-0.6B"
    run_eval api "$OUT_DIR/eval_qwen3-1.7b_base.json" "Qwen/Qwen3-1.7B"
    run_eval api "$OUT_DIR/eval_qwen3-4b_base.json"   "Qwen/Qwen3-4B"
fi

# 3. Trained-model evals
if [ "$SKIP_TRAINED" != "1" ]; then
    run_eval api "$OUT_DIR/eval_qwen3-0.6b_trained.json" "$MODEL_0_6B"
    run_eval api "$OUT_DIR/eval_qwen3-1.7b_trained.json" "$MODEL_1_7B"
    run_eval api "$OUT_DIR/eval_qwen3-4b_trained.json"   "$MODEL_4B"
fi

echo
echo "====================================================================="
echo "All evals done. Now run scripts/make_plots.py to generate PNGs."
echo "====================================================================="
ls -la "$OUT_DIR"
