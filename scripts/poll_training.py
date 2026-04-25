"""Robust training monitor: polls a job until it terminates, retrying log fetch.

Usage:
    HF_TOKEN=hf_... python scripts/poll_training.py <job_id> [poll_interval_s]

Writes the same files as monitor_training.py:
    outputs/job_logs/<job_id>_metrics.jsonl
    outputs/job_logs/<job_id>_summary.json
    outputs/job_logs/<job_id>_raw.log
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import truststore

truststore.inject_into_ssl()
from huggingface_hub import HfApi  # noqa: E402

JOB_ID = sys.argv[1]
POLL_S = int(sys.argv[2]) if len(sys.argv) > 2 else 90
NAMESPACE = os.environ.get("HF_NAMESPACE", "agarwalanu3103")

api = HfApi(token=os.environ["HF_TOKEN"])
out_dir = Path("outputs/job_logs")
out_dir.mkdir(parents=True, exist_ok=True)
metrics_path = out_dir / f"{JOB_ID}_metrics.jsonl"
summary_path = out_dir / f"{JOB_ID}_summary.json"
raw_log_path = out_dir / f"{JOB_ID}_raw.log"

DICT_RE = re.compile(r"^\s*\{.*'loss':.*'reward':.*\}\s*$")


def fetch_chunk(timeout_s: int = 60) -> list[str]:
    """Fetch streaming logs for up to `timeout_s` seconds, then stop."""
    out: list[str] = []
    done = threading.Event()

    def reader():
        try:
            for log in api.fetch_job_logs(job_id=JOB_ID, namespace=NAMESPACE, follow=True):
                if done.is_set():
                    break
                out.append(str(log))
        except Exception as exc:
            out.append(f"### Error: {exc}")
        finally:
            done.set()

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    done.set()
    return out


seen_lines: set[str] = set()
if metrics_path.exists():
    for ln in metrics_path.read_text().splitlines():
        if ln.strip():
            seen_lines.add(ln)

print(f"[poll] monitoring {JOB_ID}; poll every {POLL_S}s", flush=True)
last_status = ""
while True:
    info = api.inspect_job(job_id=JOB_ID, namespace=NAMESPACE)
    stage = info.status.stage
    msg = info.status.message or ""
    if stage != last_status:
        print(f"[poll] status -> {stage} ({msg})", flush=True)
        last_status = stage

    if stage in ("RUNNING", "COMPLETED"):
        captured = fetch_chunk(timeout_s=120)
        with raw_log_path.open("a") as raw:
            for line in captured:
                raw.write(line + "\n")
        new_steps = 0
        with metrics_path.open("a") as fh:
            for line in captured:
                if not DICT_RE.match(line):
                    continue
                try:
                    d = ast.literal_eval(line.strip())
                except Exception:
                    continue
                row: dict = {}
                for k, v in d.items():
                    if isinstance(v, str):
                        try:
                            row[k] = float(v)
                        except ValueError:
                            row[k] = v
                    else:
                        row[k] = v
                ser = json.dumps(row, sort_keys=True)
                if ser in seen_lines:
                    continue
                fh.write(ser + "\n")
                seen_lines.add(ser)
                new_steps += 1
        if new_steps:
            print(f"[poll] +{new_steps} new step records (total={len(seen_lines)})", flush=True)
        summary = {
            "job_id": JOB_ID,
            "captured_lines": len(captured),
            "total_step_records": len(seen_lines),
            "stage": stage,
        }
        summary_path.write_text(json.dumps(summary, indent=2, default=str))

    if stage in ("COMPLETED", "ERROR", "CANCELED"):
        print(f"[poll] terminal stage {stage} — exiting", flush=True)
        break

    time.sleep(POLL_S)
