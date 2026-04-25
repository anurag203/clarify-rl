#!/usr/bin/env python
"""Evaluate any policy / model on the held-out scenario set.

Two modes:

  policy   Use the deterministic POLICY_PLANS asker from inference.py
           — no LLM, free, deterministic, used as the floor baseline.

  api      Use an OpenAI-compatible chat endpoint (the same path the
           submission validator uses on inference.py). Set:
              MODEL_NAME      e.g. Qwen/Qwen3-0.6B
              API_BASE_URL    e.g. https://router.huggingface.co/v1
              HF_TOKEN        write/read token

Output: a single JSON file with per-scenario scores, breakdowns,
question counts, and aggregate metrics — formatted exactly the way
`scripts/make_plots.py` consumes.

Usage:
  # baseline (deterministic policy)
  python scripts/run_eval.py --mode policy --out outputs/eval_policy.json --limit 100

  # untrained Qwen3-0.6B via HF Inference router
  HF_TOKEN=hf_xxx MODEL_NAME=Qwen/Qwen3-0.6B \\
      python scripts/run_eval.py --mode api --out outputs/eval_qwen3-0.6b_base.json --limit 100

  # trained model via HF Inference Endpoints (you provided the URL)
  API_BASE_URL=https://my-endpoint.endpoints.huggingface.cloud/v1 \\
  MODEL_NAME=clarify-rl-grpo-qwen3-0.6b HF_TOKEN=hf_xxx \\
      python scripts/run_eval.py --mode api --out outputs/eval_qwen3-0.6b_trained.json --limit 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Make the inference.py helpers importable without copy-paste.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))


def _lazy_import_inference():
    """Lazy-import inference.py so `--help` works without openai installed."""
    import inference as _inf  # type: ignore
    return _inf


def _make_ws_url(base_url: str) -> str:
    return base_url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/ws"


async def _ws_reset_with_seed(ws, task_id: str, seed: int) -> dict:
    """Reset env to a specific (task_id, seed) — exact replay of an eval scenario."""
    await ws.send(json.dumps({"type": "reset", "data": {"task_id": task_id, "seed": seed}}))
    resp = json.loads(await ws.recv())
    if resp.get("type") == "error":
        return {"observation": {}, "reward": 0.0, "done": False, "error": resp.get("data", {})}
    data = resp.get("data", {})
    return {
        "observation": data.get("observation", {}),
        "reward": float(data.get("reward", 0.0)),
        "done": bool(data.get("done", False)),
    }


def _parse_observation(obs: dict) -> dict:
    """Pull the canonical tool-result dict out of an MCP observation."""
    result = obs.get("result")
    if isinstance(result, dict):
        if isinstance(result.get("structured_content"), dict):
            return result["structured_content"]
        if isinstance(result.get("data"), dict):
            return result["data"]
        content = result.get("content")
        if isinstance(content, list) and content:
            txt = content[0].get("text", "")
            try:
                parsed = json.loads(txt)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


async def _eval_one_scenario(
    ws,
    scenario: dict,
    mode: str,
    llm_client,
    timeout_s: float,
    inf,
) -> dict:
    """Run a single scenario end-to-end. Returns a result row."""
    seed = scenario["seed"]
    task_id = scenario["task_id"]
    family = scenario.get("family", "")

    t0 = time.time()
    reset = await _ws_reset_with_seed(ws, task_id, seed)
    if "error" in reset:
        return {
            "seed": seed,
            "task_id": task_id,
            "scenario_id": f"seed{seed:05d}_{family}_{task_id}",
            "family": family,
            "request": "",
            "final_score": 0.0,
            "score_breakdown": {},
            "questions_asked": 0,
            "format_pass": False,
            "error": str(reset["error"]),
            "messages": [],
            "trace": [],
            "elapsed_s": time.time() - t0,
        }

    initial_data = _parse_observation(reset["observation"])
    request_text = initial_data.get("request", "")
    max_steps = int(initial_data.get("max_steps", 10))

    messages = [
        {"role": "system", "content": inf.SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"USER REQUEST:\n{request_text}\n\nYou have {max_steps} steps. "
            "Available tools: ask_question(question), propose_plan(plan), get_task_info().\n\n"
            "RESPONSE FORMAT: Reply with ONE function call only, no other text.\n"
            "Examples:\n"
            "  ask_question(\"What is the date?\")\n"
            "  propose_plan('{\"event_type\": \"birthday\", \"date\": \"2024-12-25\"}')\n"
            "  get_task_info()\n"
        )},
    ]

    trace: list[dict] = []
    revealed: dict[str, Any] = {}
    questions_asked = 0
    final_score = 0.0
    score_breakdown: dict[str, float] = {}
    format_pass: Optional[bool] = None
    parse_error: Optional[str] = None
    llm_attempts = 0
    used_policy_step = 0
    done = False

    for step in range(max_steps):
        if time.time() - t0 > timeout_s:
            trace.append({"step": step, "error": "timeout"})
            break

        if mode == "policy":
            tool_name, args = inf._next_policy_action(  # type: ignore[attr-defined]
                task_id, used_policy_step, request_text, revealed
            )
            used_policy_step += 1
        else:  # api
            tool_name, args, fellback, llm_attempts = inf._choose_action(  # type: ignore[attr-defined]
                task_id, messages, llm_client, used_policy_step, llm_attempts, request_text, revealed
            )
            if fellback:
                used_policy_step += 1

        try:
            step_resp = await inf.ws_step(ws, tool_name, args)
        except Exception as exc:  # noqa: BLE001
            trace.append({"step": step, "error": f"ws_step exception: {exc}"})
            break

        obs = step_resp.get("observation", {}) or {}
        result = _parse_observation(obs)
        done = bool(step_resp.get("done"))

        record = {
            "step": step,
            "tool": tool_name,
            "args": args,
            "reward": float(step_resp.get("reward", 0.0)),
            "done": done,
            "result": result,
        }
        trace.append(record)

        format_reminder = (
            "\n\nReminder: Reply with ONE function call only "
            "(ask_question/propose_plan/get_task_info), no other text."
        )
        if tool_name == "ask_question":
            questions_asked += 1
            if isinstance(result, dict) and result.get("field_revealed"):
                fld = result["field_revealed"]
                ans = result.get("answer", "")
                revealed[fld] = ans
            messages.append({"role": "user", "content": json.dumps(result) + format_reminder})
        elif tool_name == "get_task_info":
            messages.append({"role": "user", "content": json.dumps(result) + format_reminder})
        elif tool_name == "propose_plan":
            if isinstance(result, dict):
                final_score = float(result.get("score", step_resp.get("reward", 0.0)))
                score_breakdown = result.get("breakdown", {}) or {}
                parse_error = result.get("parse_error")
                fmt = score_breakdown.get("FormatCheck") or score_breakdown.get("format_check")
                if fmt is not None:
                    format_pass = fmt > 0
            done = True

        if done:
            break

    return {
        "seed": seed,
        "task_id": task_id,
        "scenario_id": f"seed{seed:05d}_{family}_{task_id}",
        "family": family,
        "request": request_text,
        "final_score": final_score,
        "score_breakdown": score_breakdown,
        "questions_asked": questions_asked,
        "format_pass": format_pass,
        "parse_error": parse_error,
        "messages": messages,
        "trace": trace,
        "elapsed_s": time.time() - t0,
    }


async def _run(args) -> dict:
    inf = _lazy_import_inference()

    eval_path = Path(args.scenarios)
    if not eval_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {eval_path}")

    scenarios = json.loads(eval_path.read_text())
    if args.limit and args.limit < len(scenarios):
        scenarios = scenarios[: args.limit]
    print(f"Loaded {len(scenarios)} scenarios from {eval_path}")

    llm_client = None
    if args.mode == "api":
        if not inf.API_KEY:
            raise RuntimeError("api mode requires HF_TOKEN / OPENAI_API_KEY")
        llm_client = inf.create_client()
        if llm_client is None:
            raise RuntimeError("Failed to create OpenAI client (check API_BASE_URL/HF_TOKEN)")
        print(f"Using OpenAI client with base_url={inf.API_BASE_URL} model={inf.MODEL_NAME}")
    else:
        print("Mode: policy (deterministic, no LLM)")

    import websockets

    results: list[dict] = []
    ws_url = _make_ws_url(args.env)
    print(f"Env WS:    {ws_url}")
    print(f"Output to: {args.out}")
    print()

    overall_t0 = time.time()
    async with websockets.connect(
        ws_url, open_timeout=30, close_timeout=10, max_size=2**24
    ) as ws:
        for i, scn in enumerate(scenarios):
            print(f"[{i+1}/{len(scenarios)}] family={scn.get('family','?')} task={scn['task_id']} seed={scn['seed']}", flush=True)
            row = await _eval_one_scenario(ws, scn, args.mode, llm_client, args.timeout, inf)
            results.append(row)
            print(
                f"   score={row['final_score']:.3f} q={row['questions_asked']} fmt={row['format_pass']} "
                f"err={row.get('error') or row.get('parse_error') or ''}",
                flush=True,
            )

    total_s = time.time() - overall_t0

    scores = [r["final_score"] for r in results]
    fmt_passes = [r["format_pass"] for r in results if r["format_pass"] is not None]
    qs = [r["questions_asked"] for r in results]

    summary = {
        "model": inf.MODEL_NAME if args.mode == "api" else None,
        "mode": args.mode,
        "scenarios_total": len(results),
        "elapsed_s": total_s,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "avg_questions": sum(qs) / len(qs) if qs else 0.0,
        "format_pass_rate": (sum(1 for f in fmt_passes if f) / len(fmt_passes)) if fmt_passes else 0.0,
        "completion_rate": sum(1 for r in results if r["final_score"] > 0) / max(1, len(results)),
    }

    payload = {
        "summary": summary,
        "config": {
            "mode": args.mode,
            "model": inf.MODEL_NAME if args.mode == "api" else None,
            "api_base_url": inf.API_BASE_URL if args.mode == "api" else None,
            "env_base_url": args.env,
            "scenarios_file": str(eval_path),
            "limit": args.limit,
        },
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print()
    print(f"Saved {len(results)} results to {out_path}")
    print(f"Avg score:        {summary['avg_score']:.4f}")
    print(f"Format pass rate: {summary['format_pass_rate']:.4f}")
    print(f"Completion rate:  {summary['completion_rate']:.4f}")
    print(f"Avg questions:    {summary['avg_questions']:.2f}")
    print(f"Total elapsed:    {total_s:.1f} s")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("policy", "api"), required=True)
    parser.add_argument(
        "--scenarios",
        default=str(_REPO / "scenarios" / "eval_held_out.json"),
        help="Path to eval scenario JSON (default: scenarios/eval_held_out.json)",
    )
    parser.add_argument("--out", required=True, help="Output JSON file (e.g. outputs/eval_policy.json)")
    parser.add_argument("--limit", type=int, default=None, help="Cap to first N scenarios")
    parser.add_argument(
        "--env",
        default=os.environ.get("ENV_BASE_URL", "https://agarwalanu3103-clarify-rl.hf.space"),
        help="Env Space URL",
    )
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-scenario timeout in seconds")

    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
