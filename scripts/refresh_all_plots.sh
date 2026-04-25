#!/usr/bin/env bash
# Download all eval JSONs + log_history files from the Hub, then regenerate
# the 6 submission plots with the same-base before/after comparison.
#
# Usage:
#   HF_TOKEN=hf_...   (Acct 1: agarwalanu3103) used by default
#   HF_TOKEN_KANAN=hf_...  (Acct 2: Kanan2005, owns Run 3 repo) — optional
#   HF_TOKEN_MNIT=hf_...   (Acct 3: 2022uec1542, owns Run 4 repo) — optional
#   ./scripts/refresh_all_plots.sh
#
# Repos covered:
#   Run 1: agarwalanu3103/clarify-rl-grpo-qwen3-0-6b   — 0.6B GRPO (beta=0,    LR=1e-6)
#   Run 2: agarwalanu3103/clarify-rl-grpo-qwen3-1-7b   — 1.7B GRPO (beta=0,    LR=1e-6)
#   Run 3: Kanan2005/clarify-rl-grpo-qwen3-4b          — 4B   GRPO (beta=0,    LR=1e-6)
#   Run 4: 2022uec1542/clarify-rl-grpo-qwen3-1-7b      — 1.7B GRPO (beta=0.2,  LR=5e-7) — KL-anchored

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN required (Acct 1: agarwalanu3103)}"

OUT=outputs/run_artifacts
mkdir -p "$OUT" plots

source .venv/bin/activate

# Forward all three tokens into the python pull so each repo can be reached
# with the right credentials. Kanan/MNIT are optional — we skip those repos
# silently if not provided.
HF_TOKEN_KANAN="${HF_TOKEN_KANAN:-}" \
HF_TOKEN_MNIT="${HF_TOKEN_MNIT:-}" \
HF_TOKEN_AGARWAL="${HF_TOKEN}" \
python <<'PY'
import os
import json
import truststore; truststore.inject_into_ssl()
from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path

out = Path("outputs/run_artifacts")
out.mkdir(parents=True, exist_ok=True)

# Map: short label → (repo_id, env-var name holding the token for that repo)
REPOS = {
    "0.6B":         ("agarwalanu3103/clarify-rl-grpo-qwen3-0-6b", "HF_TOKEN_AGARWAL"),
    "1.7B":         ("agarwalanu3103/clarify-rl-grpo-qwen3-1-7b", "HF_TOKEN_AGARWAL"),
    "4B":           ("Kanan2005/clarify-rl-grpo-qwen3-4b",        "HF_TOKEN_KANAN"),
    "1.7B-KL":      ("2022uec1542/clarify-rl-grpo-qwen3-1-7b",    "HF_TOKEN_MNIT"),
}

for size, (repo, token_var) in REPOS.items():
    token = os.environ.get(token_var) or ""
    if not token:
        print(f"[skip] {size} {repo}: {token_var} not set")
        continue
    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo)
    except Exception as exc:
        print(f"[skip] {repo}: {exc}")
        continue

    for f in files:
        if f.startswith("evals/") or f in ("log_history.json", "training_summary.json"):
            try:
                local = hf_hub_download(
                    repo_id=repo,
                    filename=f,
                    token=token,
                    local_dir=str(out / size),
                )
                print(f"[ok]  {size}/{f}")
            except Exception as exc:
                print(f"[err] {size}/{f}: {exc}")
PY

echo
echo "Files now under outputs/run_artifacts:"
find outputs/run_artifacts -name '*.json' | sort

# Build the eval-flag list for whatever was actually downloaded.
EVAL_FLAGS=()
add_if() {
    if [ -f "$1" ]; then
        EVAL_FLAGS+=(--eval "$2=$1")
        echo "  + ${2}: ${1}"
    fi
}

echo
echo "Building eval list:"
add_if outputs/eval_policy_v4.json                                                  "policy (deterministic)"
add_if outputs/run_artifacts/v4/evals/eval_qwen3-0.6b_n50_v4.json                   "0.6B base"
add_if outputs/run_artifacts/v4/evals/eval_clarify-rl-grpo-qwen3-0-6b_n50_v4.json   "0.6B GRPO (Run 1)"
add_if outputs/run_artifacts/v4/evals/eval_qwen3-1.7b_n50_v4.json                   "1.7B base"
add_if outputs/run_artifacts/1.7B/evals/eval_clarify-rl-grpo-qwen3-1-7b_n50.json    "1.7B GRPO no-KL (Run 2)"
add_if outputs/run_artifacts/4B-base/evals/eval_qwen3-4b_qwen3-4b-base_n50_v4.json  "4B base"
# After Run 4 finishes its first eval, refresh_all_plots.sh will pick up the new
# JSON automatically; until then it's silently skipped.
for f in outputs/run_artifacts/1.7B-KL/evals/eval_*_n50_v4.json; do
    [ -f "$f" ] && add_if "$f" "1.7B GRPO +KL (Run 4)"
done
for f in outputs/run_artifacts/4B/evals/eval_*_n50_v4.json; do
    [ -f "$f" ] && add_if "$f" "4B GRPO (Run 3)"
done
add_if outputs/eval_qwen3-4b-instruct_n50_v4.json                                    "4B-instruct"

LOG_FLAGS=()
[ -f outputs/run1_artifacts/log_history.json ] && LOG_FLAGS+=(--log-history "0.6B GRPO (Run 1, beta=0)=outputs/run1_artifacts/log_history.json")
[ -f outputs/run_artifacts/1.7B/log_history.json ] && LOG_FLAGS+=(--log-history "1.7B GRPO (Run 2, beta=0)=outputs/run_artifacts/1.7B/log_history.json")
[ -f outputs/run2_artifacts/log_history_partial.json ] && [ ! -f outputs/run_artifacts/1.7B/log_history.json ] && LOG_FLAGS+=(--log-history "1.7B GRPO (Run 2, in progress)=outputs/run2_artifacts/log_history_partial.json")
[ -f outputs/run_artifacts/4B/log_history.json ] && LOG_FLAGS+=(--log-history "4B GRPO (Run 3, beta=0)=outputs/run_artifacts/4B/log_history.json")
[ -f outputs/run_artifacts/1.7B-KL/log_history.json ] && LOG_FLAGS+=(--log-history "1.7B GRPO (Run 4, beta=0.2)=outputs/run_artifacts/1.7B-KL/log_history.json")
# Until each run's final log_history.json is pushed to its repo, fall back to
# the partial JSON that monitor_training scrapes from live job logs. This way
# the reward curves stay current even mid-training.
[ -f outputs/run4_artifacts/log_history_partial.json ] && [ ! -f outputs/run_artifacts/1.7B-KL/log_history.json ] && LOG_FLAGS+=(--log-history "1.7B GRPO (Run 4, beta=0.2 in-progress)=outputs/run4_artifacts/log_history_partial.json")
[ -f outputs/run3_artifacts/log_history_partial.json ] && [ ! -f outputs/run_artifacts/4B/log_history.json ] && LOG_FLAGS+=(--log-history "4B GRPO (Run 3, beta=0 in-progress)=outputs/run3_artifacts/log_history_partial.json")

echo
echo "Running make_plots.py:"
python scripts/make_plots.py "${LOG_FLAGS[@]}" "${EVAL_FLAGS[@]}" --out-dir plots

# Also build the hackathon-narrative plots: per-family same-base delta and
# the 4-run summary table. These need richer logic than make_plots.py supports,
# so they live in compare_runs.py.
echo
echo "Running compare_runs.py:"
python scripts/compare_runs.py --out-dir plots
