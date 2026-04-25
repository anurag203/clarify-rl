#!/usr/bin/env python
"""Watch HF Jobs training jobs and fire vLLM evals automatically when each
finishes — with the right token + flavor per run.

Designed to run interactively (the assistant pokes it periodically) or as a
detached watchdog. We avoid mid-training evals because TRL's GRPO pushes the
top-level weights every save_steps, so a download mid-push could see partial
state. Once status=COMPLETED, the final weights are stable and the eval is
deterministic.

Usage:
    HF_TOKEN_AGARWAL=hf_xxx HF_TOKEN_KANAN=hf_xxx HF_TOKEN_MNIT=hf_xxx \\
        python scripts/watch_and_eval.py [--once] [--interval 120]

State file (``outputs/auto_eval_state.json``) records what evals have been
launched so a restart of this script doesn't double-fire.

Why this lives separately from monitor_training.py: this script is a *driver*
(reads job status → triggers shell action), whereas monitor_training.py is a
*reporter* (reads logs → prints metrics). Mixing the two makes failure modes
hard to debug.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Lazy import so --help works even if HF SDK isn't installed locally.

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "outputs" / "auto_eval_state.json"
LAUNCH_EVAL = ROOT / "scripts" / "launch_eval_job.sh"


@dataclass
class TrainingRun:
    run_id: str
    job_id: str
    repo: str
    token_env: str            # which env-var holds the right HF write token
    eval_flavor: str          # GPU flavor for the eval job
    eval_limit: int = 50
    eval_label: str = "n50_v4"


# Edit this when launching new runs. Keep it sourced from outputs/runs.json
# manually so we don't accidentally fire evals against stale entries.
RUNS: list[TrainingRun] = [
    TrainingRun(
        run_id="run3",
        job_id="69ed1f5ad70108f37acdeed9",
        repo="Kanan2005/clarify-rl-grpo-qwen3-4b",
        token_env="HF_TOKEN_KANAN",
        eval_flavor="a100-large",       # 4B fp16 ≈ 8 GB weights, fits easily
        eval_limit=50,
        eval_label="n50_v4",
    ),
    TrainingRun(
        run_id="run4",
        job_id="69ed1a3fd70108f37acdee5e",
        repo="2022uec1542/clarify-rl-grpo-qwen3-1-7b",
        token_env="HF_TOKEN_MNIT",
        eval_flavor="a10g-small",
        eval_limit=50,
        eval_label="n50_v4",
    ),
]

TERMINAL_STAGES = {"COMPLETED", "FAILED", "CANCELLED", "ERROR", "TIMED_OUT"}


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _check_run(run: TrainingRun, state: dict, interval: int) -> str:
    """Returns the latest stage observed; updates state in place."""
    import truststore
    truststore.inject_into_ssl()
    from huggingface_hub import HfApi  # noqa: WPS433

    token = os.environ.get(run.token_env)
    if not token:
        print(f"[{run.run_id}] {run.token_env} not set — cannot watch")
        return "UNKNOWN"

    api = HfApi(token=token)
    try:
        job = api.inspect_job(job_id=run.job_id)
    except Exception as exc:
        print(f"[{run.run_id}] inspect_job failed: {exc}")
        return "UNKNOWN"

    stage = job.status.stage if job.status else "UNKNOWN"
    msg = (job.status.message if job.status else "") or ""
    rec = state.setdefault(run.run_id, {})
    rec["last_stage"] = stage
    rec["last_message"] = msg

    print(f"[{run.run_id}] stage={stage}  flavor={job.flavor}  msg={msg[:80]}")

    if stage in TERMINAL_STAGES and rec.get("eval_launched") != True:
        # Double-confirm there's a model.safetensors before firing the eval
        try:
            files = api.list_repo_files(run.repo)
        except Exception as exc:
            print(f"[{run.run_id}] list_repo_files failed: {exc}")
            return stage
        if "model.safetensors" not in files and not any(f.endswith(".safetensors") for f in files):
            print(f"[{run.run_id}] training ended in {stage} but no weights pushed — skipping eval")
            rec["eval_launched"] = "skipped_no_weights"
            return stage

        if stage != "COMPLETED":
            # FAILED but weights exist — still useful to eval the final checkpoint
            print(f"[{run.run_id}] stage={stage} but weights present → evaluating partial run")

        cmd = [
            "bash", str(LAUNCH_EVAL),
            run.repo, run.eval_flavor, str(run.eval_limit),
        ]
        env = os.environ.copy()
        env["HF_TOKEN"] = token
        env["EVAL_LABEL"] = run.eval_label
        env["DETACH"] = "1"

        print(f"[{run.run_id}] launching eval: {' '.join(cmd)}")
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        rec["eval_launched"] = True
        rec["eval_launched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rec["eval_stdout_tail"] = proc.stdout[-2000:]
        rec["eval_stderr_tail"] = proc.stderr[-2000:]
        rec["eval_returncode"] = proc.returncode
        print(proc.stdout[-1000:])
        if proc.returncode != 0:
            print(f"[{run.run_id}] EVAL LAUNCH FAILED rc={proc.returncode}\n{proc.stderr[-1500:]}")

    return stage


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--once", action="store_true", help="Single pass and exit")
    p.add_argument("--interval", type=int, default=180, help="Seconds between polls (default 180)")
    p.add_argument("--max-iterations", type=int, default=240,
                   help="Safety cap on poll iterations (default 240 ≈ 12 hours @ 180s)")
    args = p.parse_args()

    if not LAUNCH_EVAL.is_file():
        print(f"ERROR: {LAUNCH_EVAL} not found", file=sys.stderr)
        sys.exit(1)
    if shutil.which("bash") is None:
        print("ERROR: bash not on PATH", file=sys.stderr)
        sys.exit(1)

    iter_count = 0
    while True:
        iter_count += 1
        state = _load_state()
        all_done = True
        for run in RUNS:
            stage = _check_run(run, state, args.interval)
            rec = state.get(run.run_id, {})
            if stage not in TERMINAL_STAGES or rec.get("eval_launched") is True:
                # Still working: not terminal OR eval already kicked off (don't loop forever)
                pass
            if stage not in TERMINAL_STAGES:
                all_done = False
        _save_state(state)
        if args.once or all_done or iter_count >= args.max_iterations:
            break
        print(f"[wait] sleeping {args.interval}s (iter {iter_count}/{args.max_iterations})")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
