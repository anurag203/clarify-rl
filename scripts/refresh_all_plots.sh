#!/usr/bin/env bash
# Download all eval JSONs + log_history files from the Hub, then regenerate
# the 5 submission plots with the same-base before/after comparison.
#
# Usage:
#   HF_TOKEN=hf_... ./scripts/refresh_all_plots.sh
#
# Assumes evals were pushed to the model repos with these labels:
#   agarwalanu3103/clarify-rl-grpo-qwen3-0-6b/evals/eval_*_qwen3-0-6b-BASE_n50_v2.json
#   agarwalanu3103/clarify-rl-grpo-qwen3-0-6b/evals/eval_*_qwen3-1-7b-BASE_n50_v2.json
#   agarwalanu3103/clarify-rl-grpo-qwen3-0-6b/evals/eval_*_n50_parserfix.json
#   agarwalanu3103/clarify-rl-grpo-qwen3-1-7b/evals/eval_*_n50_v2.json     (after run 2)

set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN required}"

OUT=outputs/run_artifacts
mkdir -p "$OUT" plots

source .venv/bin/activate

python <<'PY'
import os
import json
import truststore; truststore.inject_into_ssl()
from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path

api = HfApi()
out = Path("outputs/run_artifacts")
out.mkdir(parents=True, exist_ok=True)

REPOS = {
    "0.6B": "agarwalanu3103/clarify-rl-grpo-qwen3-0-6b",
    "1.7B": "agarwalanu3103/clarify-rl-grpo-qwen3-1-7b",
}

for size, repo in REPOS.items():
    try:
        files = api.list_repo_files(repo, token=os.environ["HF_TOKEN"])
    except Exception as exc:
        print(f"[skip] {repo}: {exc}")
        continue

    for f in files:
        if f.startswith("evals/") or f in ("log_history.json", "training_summary.json"):
            try:
                local = hf_hub_download(
                    repo_id=repo,
                    filename=f,
                    token=os.environ["HF_TOKEN"],
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
add_if outputs/eval_policy_v4.json                                              "policy"
add_if outputs/run_artifacts/v4/evals/eval_qwen3-0.6b_n50_v4.json               "0.6B base"
add_if outputs/run_artifacts/v4/evals/eval_qwen3-1.7b_n50_v4.json               "1.7B base"
add_if outputs/run_artifacts/v4/evals/eval_clarify-rl-grpo-qwen3-0-6b_n50_v4.json "0.6B trained"
add_if outputs/run_artifacts/1.7B/evals/eval_clarify-rl-grpo-qwen3-1-7b_n50.json "1.7B trained"
add_if outputs/eval_qwen3-4b-instruct_n50_v4.json                               "4B-instruct (ceiling)"

LOG_FLAGS=()
[ -f outputs/run1_artifacts/log_history.json ] && LOG_FLAGS+=(--log-history "0.6B GRPO (Run 1, 300 steps)=outputs/run1_artifacts/log_history.json")
[ -f outputs/run_artifacts/1.7B/log_history.json ] && LOG_FLAGS+=(--log-history "1.7B GRPO (Run 2, 400 steps)=outputs/run_artifacts/1.7B/log_history.json")
[ -f outputs/run2_artifacts/log_history_partial.json ] && [ ! -f outputs/run_artifacts/1.7B/log_history.json ] && LOG_FLAGS+=(--log-history "1.7B GRPO (Run 2, in progress)=outputs/run2_artifacts/log_history_partial.json")

echo
echo "Running make_plots.py:"
python scripts/make_plots.py "${LOG_FLAGS[@]}" "${EVAL_FLAGS[@]}" --out-dir plots
