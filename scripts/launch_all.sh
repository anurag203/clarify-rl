#!/usr/bin/env bash
# Fire ALL production runs in parallel across multiple HF accounts.
#
# Each account fires one job. The plan defaults to 3 simultaneous runs
# (matching the 3 Qwen3 sizes), with optional 4th and 5th insurance runs.
#
# Required env vars (one HF_TOKEN per account):
#   HF_TOKEN_1   token for account 1 (drives Qwen3-0.6B)
#   HF_TOKEN_2   token for account 2 (drives Qwen3-1.7B)
#   HF_TOKEN_3   token for account 3 (drives Qwen3-4B)
#   HF_TOKEN_4   (optional) token for account 4 — drives insurance run if INSURANCE=1
#
# Optional env:
#   ENV_BASE_URL          default: https://agarwalanu3103-clarify-rl.hf.space
#   INSURANCE             "1" → also launch a backup Qwen3-1.7B run (different seed)
#   DRY_RUN               "1" → print all commands but do not launch anything
#
# Usage:
#   HF_TOKEN_1=hf_a HF_TOKEN_2=hf_b HF_TOKEN_3=hf_c ./scripts/launch_all.sh
#
# Recommended budget for the default plan (without insurance): ~$70
# With INSURANCE=1: ~$95
# Either way well within the $120 cap.

set -euo pipefail

: "${HF_TOKEN_1:?HF_TOKEN_1 required (account 1 → Qwen3-0.6B)}"
: "${HF_TOKEN_2:?HF_TOKEN_2 required (account 2 → Qwen3-1.7B)}"
: "${HF_TOKEN_3:?HF_TOKEN_3 required (account 3 → Qwen3-4B)}"
: "${ENV_BASE_URL:=https://agarwalanu3103-clarify-rl.hf.space}"
: "${INSURANCE:=0}"
: "${DRY_RUN:=0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$SCRIPT_DIR/launch_hf_job.sh"

cat <<EOF
=========================================================================
ClarifyRL multi-account parallel launch
=========================================================================
  Env Space:     $ENV_BASE_URL
  Insurance:     $INSURANCE
  Dry run:       $DRY_RUN
=========================================================================
EOF

# ----------------------------------------------------------------------
# Plan (revised after `hf jobs hardware` 2026-04-25):
#   Account 1: Qwen3-0.6B  / a10g-large  / 500 steps / num_gen=4 / ~$7.50  (5h * $1.50)
#   Account 2: Qwen3-1.7B  / a100-large  / 400 steps / num_gen=8 / ~$12.50 (5h * $2.50)
#   Account 3: Qwen3-4B    / h200        / 250 steps / num_gen=8 / ~$25    (5h * $5.00)
#   Account 4: Qwen3-1.7B  / a100-large  / 400 steps / num_gen=8 / seed=84 / ~$12.50 (insurance)
#   Total without insurance: ~$45     With insurance: ~$57.50
#   Well within the $120 cap → leaves headroom for retries / longer runs / second pass.
# ----------------------------------------------------------------------

run() {
    local label="$1"; shift
    echo
    echo "──────────────────────────────────────────────────────────────────"
    echo "  Launching: $label"
    echo "──────────────────────────────────────────────────────────────────"
    if [ "$DRY_RUN" = "1" ]; then
        DRY_RUN=1 "$@"
    else
        "$@"
    fi
}

# Account 1 → 0.6B
HF_TOKEN="$HF_TOKEN_1" \
ENV_BASE_URL="$ENV_BASE_URL" \
SEED=42 \
run "Account 1: Qwen3-0.6B / a10g-large / 500 steps" \
    "$LAUNCHER" Qwen/Qwen3-0.6B a10g-large 500 &
PID1=$!

# Account 2 → 1.7B
HF_TOKEN="$HF_TOKEN_2" \
ENV_BASE_URL="$ENV_BASE_URL" \
SEED=42 \
run "Account 2: Qwen3-1.7B / a100-large / 400 steps" \
    "$LAUNCHER" Qwen/Qwen3-1.7B a100-large 400 &
PID2=$!

# Account 3 → 4B
HF_TOKEN="$HF_TOKEN_3" \
ENV_BASE_URL="$ENV_BASE_URL" \
SEED=42 \
run "Account 3: Qwen3-4B / h200 / 250 steps" \
    "$LAUNCHER" Qwen/Qwen3-4B h200 250 &
PID3=$!

PIDS=("$PID1" "$PID2" "$PID3")
LABELS=("0.6B" "1.7B" "4B")

# Optional 4th insurance run
if [ "$INSURANCE" = "1" ]; then
    : "${HF_TOKEN_4:?HF_TOKEN_4 required when INSURANCE=1}"
    HF_TOKEN="$HF_TOKEN_4" \
    ENV_BASE_URL="$ENV_BASE_URL" \
    SEED=84 \
    OUTPUT_DIR="clarify-rl-grpo-qwen3-1-7b-seed84" \
    run "Account 4: Qwen3-1.7B / a100-large / 400 steps / seed=84 (insurance)" \
        "$LAUNCHER" Qwen/Qwen3-1.7B a100-large 400 &
    PIDS+=("$!")
    LABELS+=("1.7B-seed84")
fi

# Wait for all launchers to exit. Each launcher submits the job and returns
# fairly fast — the actual training happens server-side on HF.
echo
echo "Waiting for all launches to complete (this only waits for *submission*,"
echo "not for the training itself — that runs server-side on HF Jobs)..."
echo

declare -i FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[OK]   ${LABELS[$i]} submitted"
    else
        echo "[FAIL] ${LABELS[$i]} submission exited non-zero"
        FAILED=$((FAILED + 1))
    fi
done

echo
echo "====================================================================="
if [ "$FAILED" = "0" ]; then
    echo "All ${#PIDS[@]} jobs submitted. Track them at:"
    echo "  https://huggingface.co/jobs   (per account)"
    echo "  https://huggingface.co/spaces (trackio dashboards)"
else
    echo "$FAILED of ${#PIDS[@]} submissions failed — check output above."
    exit 1
fi
echo "====================================================================="
