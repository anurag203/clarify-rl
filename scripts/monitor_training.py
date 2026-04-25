"""Monitor an HF Jobs training run and persist parsed metrics to outputs/job_logs/.

Usage:
    HF_TOKEN=hf_... python scripts/monitor_training.py <job_id> [duration_seconds]

Writes:
    outputs/job_logs/<job_id>_metrics.jsonl  (one parsed step per line)
    outputs/job_logs/<job_id>_summary.json   (aggregate summary)
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
import threading
from pathlib import Path

import truststore

truststore.inject_into_ssl()
from huggingface_hub import HfApi  # noqa: E402

JOB_ID = sys.argv[1] if len(sys.argv) > 1 else "69ece976d2c8bd8662bcdf48"
DURATION_S = int(sys.argv[2]) if len(sys.argv) > 2 else 60
NAMESPACE = os.environ.get("HF_NAMESPACE", "agarwalanu3103")

api = HfApi(token=os.environ["HF_TOKEN"])
out_dir = Path("outputs/job_logs")
out_dir.mkdir(parents=True, exist_ok=True)
metrics_path = out_dir / f"{JOB_ID}_metrics.jsonl"
summary_path = out_dir / f"{JOB_ID}_summary.json"

raw_log_path = out_dir / f"{JOB_ID}_raw.log"

DICT_RE = re.compile(r"^\s*\{.*'loss':.*'reward':.*\}\s*$")
captured: list[str] = []
done = threading.Event()

def reader():
    try:
        with raw_log_path.open("a") as raw:
            for log in api.fetch_job_logs(job_id=JOB_ID, namespace=NAMESPACE, follow=True):
                raw.write(str(log) + "\n")
                if done.is_set():
                    break
                captured.append(str(log))
    except Exception as exc:
        captured.append(f"### Error: {exc}")
    finally:
        done.set()

t = threading.Thread(target=reader, daemon=True)
t.start()
t.join(timeout=DURATION_S)
done.set()

steps: list[dict] = []
seen_lines: set[str] = set()
if metrics_path.exists():
    for ln in metrics_path.read_text().splitlines():
        if ln.strip():
            seen_lines.add(ln)

with metrics_path.open("a") as fh:
    for line in captured:
        if not DICT_RE.match(line):
            continue
        try:
            d = ast.literal_eval(line.strip())
        except Exception:
            continue
        out: dict[str, float | str] = {}
        for k, v in d.items():
            if isinstance(v, str):
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
            else:
                out[k] = v
        ser = json.dumps(out, sort_keys=True)
        if ser in seen_lines:
            continue
        fh.write(ser + "\n")
        seen_lines.add(ser)
        steps.append(out)

summary: dict[str, object] = {
    "job_id": JOB_ID,
    "captured_lines": len(captured),
    "new_steps": len(steps),
    "epochs_seen": [s.get("epoch") for s in steps[-5:]],
    "rewards_seen": [s.get("reward") for s in steps[-5:]],
    "losses_seen": [s.get("loss") for s in steps[-5:]],
}
if steps:
    summary.update({
        "latest": steps[-1],
        "median_step_time": sorted([float(s.get("step_time", 0)) for s in steps])[len(steps) // 2],
    })
summary_path.write_text(json.dumps(summary, indent=2, default=str))
print(f"Captured {len(captured)} log lines, {len(steps)} new step records.")
print(json.dumps(summary, indent=2, default=str))
