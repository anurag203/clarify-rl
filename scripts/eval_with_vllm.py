#!/usr/bin/env python
"""Run a vLLM-powered eval inside an HF Job and push results to the Hub.

Why this exists
---------------
HF Inference Router does NOT serve fine-tuned community uploads — only
provider-listed models (verified via `model_not_supported` 400 from the
router for our own GRPO checkpoint). So our trained Qwen3 GRPO models
must be evaluated via vLLM that we host ourselves. The cheapest /
cleanest path is one short HF Job per checkpoint that:

  1. Bootstraps the project repo (this file's source repo) from the
     public HF Space `agarwalanu3103/clarify-rl` so it has scenarios,
     `run_eval.py`, and `inference.py` available locally.
  2. Boots the vLLM OpenAI-compatible HTTP server in-process, loading
     the fine-tuned model from its Hub repo.
  3. Connects to the env Space WS exactly like the submission validator.
  4. Replays N eval scenarios via `scripts/run_eval.py --mode api`.
  5. Pushes the results JSON to the model repo's `evals/` folder so the
     submission/validator and `make_plots.py` can find it without us
     shipping artifacts back to the laptop.

Usage (inside an HF Job, via scripts/launch_eval_job.sh):

  HF_TOKEN=hf_xxx \\
  MODEL_NAME=agarwalanu3103/clarify-rl-grpo-qwen3-0-6b \\
  ENV_BASE_URL=https://agarwalanu3103-clarify-rl.hf.space \\
  LIMIT=50 \\
  python scripts/eval_with_vllm.py

Env vars consumed:
  HF_TOKEN          required, write token of the account hosting the eval.
  MODEL_NAME        required, full Hub repo id of the model to evaluate.
  ENV_BASE_URL      env Space URL (default: agarwalanu3103-clarify-rl).
  LIMIT             N scenarios to run (default 50).
  EVAL_LABEL        optional suffix for the output filename (default n{LIMIT}).
  PUSH_TO_REPO      where to upload eval JSON; defaults to MODEL_NAME.
  REPO_SPACE_ID     Space holding `inference.py` + `scripts/` + `scenarios/`.
                    Default: agarwalanu3103/clarify-rl.
  GPU_MEM_UTIL      vLLM gpu memory utilisation (default 0.85).
  MAX_MODEL_LEN     vLLM max model len (default 4096).
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import truststore  # type: ignore[import-not-found]

    truststore.inject_into_ssl()
except ImportError:
    pass


def _read_env() -> dict:
    cfg = {
        "HF_TOKEN": os.environ.get("HF_TOKEN"),
        "MODEL_NAME": os.environ.get("MODEL_NAME"),
        "ENV_BASE_URL": os.environ.get(
            "ENV_BASE_URL", "https://agarwalanu3103-clarify-rl.hf.space"
        ),
        "LIMIT": int(os.environ.get("LIMIT", "50")),
        "EVAL_LABEL": os.environ.get("EVAL_LABEL", ""),
        "PUSH_TO_REPO": os.environ.get("PUSH_TO_REPO", ""),
        "REPO_SPACE_ID": os.environ.get("REPO_SPACE_ID", "agarwalanu3103/clarify-rl"),
        "GPU_MEM_UTIL": float(os.environ.get("GPU_MEM_UTIL", "0.85")),
        "MAX_MODEL_LEN": int(os.environ.get("MAX_MODEL_LEN", "4096")),
        "VLLM_PORT": int(os.environ.get("VLLM_PORT", "8000")),
    }
    if not cfg["HF_TOKEN"]:
        raise SystemExit("HF_TOKEN is required (write token).")
    if not cfg["MODEL_NAME"]:
        raise SystemExit("MODEL_NAME is required (Hub repo id of the trained model).")
    if not cfg["PUSH_TO_REPO"]:
        cfg["PUSH_TO_REPO"] = cfg["MODEL_NAME"]
    if not cfg["EVAL_LABEL"]:
        cfg["EVAL_LABEL"] = f"n{cfg['LIMIT']}"
    return cfg


def _bootstrap_repo(space_id: str, token: str) -> Path:
    """Snapshot the project Space so this job has run_eval.py + scenarios."""
    from huggingface_hub import snapshot_download

    target = Path("/tmp/clarify-rl")
    if target.exists():
        shutil.rmtree(target)
    print(f"[boot] downloading Space {space_id} → {target}", flush=True)
    snapshot_download(
        repo_id=space_id,
        repo_type="space",
        local_dir=str(target),
        token=token,
    )
    # Verify expected files exist.
    must_have = ["inference.py", "scripts/run_eval.py", "scenarios/eval_held_out.json"]
    for rel in must_have:
        if not (target / rel).exists():
            raise FileNotFoundError(f"Bootstrap failed — missing {rel} in Space {space_id}")
    print(f"[boot] repo ready: {sorted(p.name for p in target.iterdir())}", flush=True)
    return target


def _free_port(start: int) -> int:
    p = start
    while p < start + 50:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                p += 1
    raise RuntimeError(f"No free port near {start}")


def _start_vllm(model_name: str, port: int, gpu_mem_util: float, max_len: int):
    log_path = Path("vllm_server.log")
    log = log_path.open("w")
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_mem_util),
        "--max-model-len",
        str(max_len),
        "--dtype",
        "bfloat16",
        "--enforce-eager",
    ]
    print(f"[vllm] launching: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    return proc, log_path


def _wait_for_vllm(port: int, timeout_s: float = 600) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    print(f"[vllm] waiting for {url} (≤{timeout_s:.0f}s) ...", flush=True)
    t0 = time.time()
    last_err = ""
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    body = resp.read().decode()
                    print(f"[vllm] ready after {time.time() - t0:.1f}s — {body[:200]}", flush=True)
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last_err = str(exc)
        time.sleep(5)
    raise RuntimeError(f"vLLM did not start within {timeout_s}s. Last error: {last_err}")


def _run_eval(cfg: dict, repo: Path, port: int) -> Path:
    out_path = repo / "outputs" / f"eval_{Path(cfg['MODEL_NAME']).name.lower()}_{cfg['EVAL_LABEL']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["API_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    env["MODEL_NAME"] = cfg["MODEL_NAME"]
    env["HF_TOKEN"] = cfg["HF_TOKEN"]  # vllm ignores this; openai client wants a key
    env["ENV_BASE_URL"] = cfg["ENV_BASE_URL"]

    cmd = [
        sys.executable,
        str(repo / "scripts" / "run_eval.py"),
        "--mode",
        "api",
        "--out",
        str(out_path),
        "--limit",
        str(cfg["LIMIT"]),
    ]
    print(f"[eval] running: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, env=env, cwd=str(repo))
    if res.returncode != 0:
        raise RuntimeError(f"run_eval.py exited with {res.returncode}")
    return out_path


def _push_to_hub(cfg: dict, eval_json: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=cfg["HF_TOKEN"])
    target = f"evals/{eval_json.name}"
    print(f"[push] uploading {eval_json} → {cfg['PUSH_TO_REPO']}:{target}", flush=True)
    api.upload_file(
        path_or_fileobj=str(eval_json),
        path_in_repo=target,
        repo_id=cfg["PUSH_TO_REPO"],
        repo_type="model",
        commit_message=f"eval: {cfg['EVAL_LABEL']}",
    )
    print(f"[push] done — see https://huggingface.co/{cfg['PUSH_TO_REPO']}/blob/main/{target}", flush=True)


def main() -> None:
    cfg = _read_env()
    print("=" * 70, flush=True)
    print(f"clarify-rl vllm eval | model={cfg['MODEL_NAME']} | n={cfg['LIMIT']}", flush=True)
    print(f"env={cfg['ENV_BASE_URL']}  push_to={cfg['PUSH_TO_REPO']}", flush=True)
    print("=" * 70, flush=True)

    repo = _bootstrap_repo(cfg["REPO_SPACE_ID"], cfg["HF_TOKEN"])

    port = _free_port(cfg["VLLM_PORT"])
    proc, log_path = _start_vllm(cfg["MODEL_NAME"], port, cfg["GPU_MEM_UTIL"], cfg["MAX_MODEL_LEN"])
    try:
        try:
            _wait_for_vllm(port, timeout_s=600)
        except Exception:
            print("[vllm] failed to start. Last 80 log lines:", flush=True)
            try:
                tail = log_path.read_text().splitlines()[-80:]
                print("\n".join(tail), flush=True)
            except Exception:
                pass
            raise

        eval_json = _run_eval(cfg, repo, port)
        _push_to_hub(cfg, eval_json)
        try:
            payload = json.loads(eval_json.read_text())
            print(json.dumps({"summary": payload.get("summary", {})}, indent=2), flush=True)
        except Exception:
            pass
    finally:
        if proc.poll() is None:
            print("[vllm] terminating server ...", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
