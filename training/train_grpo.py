#!/usr/bin/env python
"""ClarifyRL GRPO training — flat script for HF Jobs (and local).

This is the .py twin of `training/train_grpo.ipynb`. Parameterized via env vars
so the same script can launch any model on any HF Jobs flavor:

    Required:
      MODEL_NAME            e.g. Qwen/Qwen3-0.6B
      HF_TOKEN              write token (set via --secrets in HF Jobs)

    Optional (with sensible defaults):
      MAX_STEPS             default 300
      OUTPUT_DIR            default = "clarify-rl-grpo-<model_short>"
      ENV_BASE_URL          default https://agarwalanu3103-clarify-rl.hf.space
      SEED                  default 42
      VLLM_GPU_MEM_UTIL     default auto (0.25 / 0.45 / 0.55 / 0.55 by VRAM tier)
      MAX_COMPLETION_LEN    default auto (768 / 1024 / 1280 / 1536 by VRAM tier)
      VLLM_MAX_MODEL_LEN    default 3072
      NUM_GENERATIONS       default auto (2 / 4 / 8 / 8 by VRAM tier)
      GRAD_ACCUM_STEPS      default 8
      LEARNING_RATE         default 1e-6
      LOGGING_STEPS         default 1
      SAVE_STEPS            default 25
      SAVE_TOTAL_LIMIT      default 4
      WARMUP_STEPS          default 10
      TRACKIO_SPACE_ID      default = OUTPUT_DIR
      SKIP_PUSH             "1" to skip push_to_hub (debug)
      SMOKE_TEST            "1" to run 5 steps with NUM_GEN=2, no push (preflight)
      RESUME_FROM_CKPT      path to checkpoint dir to resume from (e.g. clarify-rl-grpo-X/checkpoint-100)

The script writes the final trainer state to `<OUTPUT_DIR>/trainer_state.json`
and a marker `DONE` file on successful completion — used by `scripts/run_eval.py`
and `scripts/make_plots.py` to know the run actually finished.
"""

from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Disable stale `vllm_ascend` detection BEFORE any TRL import.
#
# Background: vllm 0.17+ ships a stub `vllm_ascend` plugin namespace that
# `importlib.util.find_spec("vllm_ascend")` reports as installed (even on
# x86/CUDA hosts). TRL's `is_vllm_ascend_available()` (in
# `trl/import_utils.py`) is literally:
#
#     def is_vllm_ascend_available() -> bool:
#         return _is_package_available("vllm_ascend")     # → True due to stub
#
# Then `trl/generation/vllm_client.py` (line ~48) does:
#
#     if is_vllm_ascend_available():
#         from vllm_ascend.distributed.device_communicators.pyhccl import ...
#
# That submodule only exists in the real `vllm-ascend` PyPI package, which
# has wheels for cp39/cp310/cp311 only — uv's default Python on HF Jobs is
# cp312, so we can't install it. So the import fails and the worker dies
# before training starts.
#
# We can't sidestep it via dependency pins either:
#   - vllm <= 0.15 pins transformers < 5, but we need transformers @ main
#     for the `environment_factory` chat templates.
#
# Fix: monkey-patch the two detection paths used by TRL's
# `_is_package_available`:
#
#   1. `importlib.util.find_spec("vllm_ascend")` → return None
#   2. `transformers.utils.import_utils._is_package_available("vllm_ascend")`
#      → return False (defence-in-depth, in case any other code path uses
#      the transformers helper directly)
#
# TRL then takes the regular CUDA/NCCL path on a10g/a100/h200, which is
# what we want.
import importlib.util as _importlib_util
import types as _types


# Set of package names we hide from importlib.util.find_spec /
# transformers' _is_package_available. Anything in this set is reported
# as "not installed" to upstream availability checks, while still being
# accessible via direct `import` (because we stub the module in
# sys.modules below).
_BLOCKED_PACKAGES: set[str] = {"vllm_ascend", "llm_blender"}


def _disable_blocked_package_detection() -> None:
    _orig_find_spec = _importlib_util.find_spec

    def _patched_find_spec(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name in _BLOCKED_PACKAGES:
            return None
        for pkg in _BLOCKED_PACKAGES:
            if name.startswith(pkg + "."):
                return None
        return _orig_find_spec(name, *args, **kwargs)

    _importlib_util.find_spec = _patched_find_spec

    try:
        from transformers.utils import import_utils as _tu
    except Exception:  # noqa: BLE001 — transformers not installed at module-eval time
        return

    _orig_is_avail = _tu._is_package_available

    def _patched_is_avail(pkg_name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if pkg_name in _BLOCKED_PACKAGES:
            return_version = kwargs.get("return_version", False)
            if not return_version and args:
                return_version = args[0]
            return (False, "N/A") if return_version else False
        return _orig_is_avail(pkg_name, *args, **kwargs)

    _tu._is_package_available = _patched_is_avail


def _stub_llm_blender() -> None:
    """Stub `llm_blender` so trl's eager import succeeds.

    `trl/trainer/judges.py` does `import llm_blender` at module level. We
    never use the LLM-Blender judge, but trl's lazy-loader still walks
    `callbacks.py → judges.py`, so the import has to succeed. The real
    `llm_blender` package on PyPI imports `TRANSFORMERS_CACHE` from
    `transformers.utils.hub`, but that symbol was removed in transformers
    5.x (we use transformers @ main). Result: ImportError on `import
    llm_blender`, which propagates as `Failed to import trl.trainer.grpo_trainer`.

    Fix: install a fake `llm_blender` module into sys.modules BEFORE trl
    tries to import it. The module is empty — `import llm_blender` will
    succeed without running the real package's `__init__.py`. trl never
    accesses any symbol from it during normal GRPO use, so this is safe.
    """
    if "llm_blender" in sys.modules:
        return
    fake = _types.ModuleType("llm_blender")
    fake.__doc__ = "Stubbed by clarify-rl/training/train_grpo.py to bypass transformers 5.x incompat."
    fake.__file__ = "<stub>"
    sys.modules["llm_blender"] = fake


_disable_blocked_package_detection()
_stub_llm_blender()


# ---------------------------------------------------------------------------
# Env config (do NOT hardcode tokens or secrets — pulled from os.environ).
# ---------------------------------------------------------------------------

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _env(key: str, default: Optional[str] = None, required: bool = False) -> str:
    val = os.environ.get(key)
    if val is None or val == "":
        if required:
            raise RuntimeError(f"Missing required env var: {key}")
        if default is None:
            return ""
        return default
    return val


MODEL_NAME = _env("MODEL_NAME", required=True)
HF_TOKEN = _env("HF_TOKEN", required=True)
ENV_BASE_URL = _env("ENV_BASE_URL", "https://agarwalanu3103-clarify-rl.hf.space")
SEED = int(_env("SEED", "42"))

# SMOKE_TEST forces a fast 5-step run with no Hub push — used as preflight
# before spending real budget. Overrides MAX_STEPS / NUM_GENERATIONS / SKIP_PUSH.
SMOKE_TEST = _env("SMOKE_TEST", "0") == "1"

if SMOKE_TEST:
    MAX_STEPS = 5
else:
    MAX_STEPS = int(_env("MAX_STEPS", "300"))

_model_short = MODEL_NAME.split("/")[-1].lower()
OUTPUT_DIR = _env("OUTPUT_DIR", f"clarify-rl-grpo-{_model_short}")
TRACKIO_SPACE_ID = _env("TRACKIO_SPACE_ID", OUTPUT_DIR)

# NUM_GENERATIONS: GRPO group size — number of completions per prompt.
# CRITICAL: 2 is the floor; with 2 rollouts the advantage often resolves to
# zero when both rollouts produce identical tokens (which is common early in
# training), giving you the "0.000000 loss for the first N steps" pathology.
# 4 makes 6 pairwise comparisons per group; 8 makes 28. Default 0 means
# auto-tune by GPU.
NUM_GENERATIONS = int(_env("NUM_GENERATIONS", "0"))
GRAD_ACCUM_STEPS = int(_env("GRAD_ACCUM_STEPS", "8"))
LEARNING_RATE = float(_env("LEARNING_RATE", "1e-6"))
# KL coefficient against the reference (base) policy. TRL's GRPOConfig default
# is 0.0 (no reference model loaded, no KL term in the loss) — that's what
# Run 1 (0.6B) and Run 2 (1.7B) used, and it explains why Run 2 regressed
# (the 1.7B drifted onto a narrow format-only solution with no anchor pulling
# it back). For Runs 3 and 4 we set BETA=0.2 to keep stronger bases close to
# their priors while the reward still climbs. See docs/blog.md §7b.
BETA = float(_env("BETA", "0.0"))
LOGGING_STEPS = int(_env("LOGGING_STEPS", "1"))
SAVE_STEPS = int(_env("SAVE_STEPS", "25"))
SAVE_TOTAL_LIMIT = int(_env("SAVE_TOTAL_LIMIT", "4"))
WARMUP_STEPS = int(_env("WARMUP_STEPS", "10"))
VLLM_MAX_MODEL_LEN = int(_env("VLLM_MAX_MODEL_LEN", "3072"))
SKIP_PUSH = _env("SKIP_PUSH", "0") == "1" or SMOKE_TEST
RESUME_FROM_CKPT = _env("RESUME_FROM_CKPT", "")

DIFFICULTIES = ["easy", "medium", "hard"]
DIFFICULTY_WEIGHTS = [0.5, 0.3, 0.2]

REQUIRED_KEYS_BY_FAMILY: dict[str, list[str]] = {
    "coding_requirements": ["stack", "scale", "auth", "datastore"],
    "medical_intake": ["primary_symptom", "duration", "severity"],
    "support_triage": ["order_id", "item_issue", "refund_or_replace"],
    "meeting_scheduling": ["participants", "date", "time"],
    "event_planning": ["event_type", "date", "guest_count", "venue"],
}

PROMPT = """You are a helpful assistant that books and plans things for users.
The user's request will be intentionally ambiguous — you do NOT yet have all the information needed to make a good plan.

You have three tools:
  - ask_question(question): ask the user ONE targeted clarifying question (max 6 across the episode).
  - propose_plan(plan): submit your final plan as a JSON STRING with the required fields. This ENDS the episode.
  - get_task_info(): re-read the original user request.

Strategy:
  1. Read the required plan fields listed in the task description.
  2. Use ask_question to ask about EACH required field you do not already know.
  3. When you have enough info, call propose_plan with a JSON string containing ALL required fields.

Rules:
  - Be efficient. Each unnecessary question costs reward.
  - Your plan MUST include every required field listed in the task. Missing fields score zero.
  - NEVER include fields in your plan that you weren't told about. No hallucinating values.
  - The `plan` argument MUST be a JSON STRING (not a dict). Use the exact field names from the required fields list.
"""

# ---------------------------------------------------------------------------
# Auto-tune VRAM-sensitive knobs based on detected GPU.
# ---------------------------------------------------------------------------


def _autotune_for_gpu() -> tuple[float, int, int]:
    """Return (vllm_gpu_memory_utilization, max_completion_length, num_generations).

    Tiered by detected total VRAM. The big-GPU tiers (40+ GB and 80+ GB) are
    intentionally aggressive — leaving 50% of an A100/H100 idle would be a
    waste of the budget given the 22h hackathon deadline.
    """
    user_mem = os.environ.get("VLLM_GPU_MEM_UTIL")
    user_len = os.environ.get("MAX_COMPLETION_LEN")
    user_gen = os.environ.get("NUM_GENERATIONS")
    user_gen_int = int(user_gen) if user_gen and int(user_gen) > 0 else 0

    try:
        import torch
    except Exception:
        return float(user_mem or 0.30), int(user_len or 1024), user_gen_int or 4

    if not torch.cuda.is_available():
        return float(user_mem or 0.30), int(user_len or 1024), user_gen_int or 4

    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    name = torch.cuda.get_device_name(0).lower()
    print(f"Auto-tuning for GPU: {name} ({total_gib:.1f} GiB total)")

    # Tier 1: T4 / 16 GB-class. Conservative — vLLM KV cache eats fast.
    # Tier 2: A10G / L4 / 24 GB-class.
    # Tier 3: A100-40GB / L40S / 40 GB-class.
    # Tier 4: A100-80GB / H100 / 80+ GB-class — go big.
    if total_gib < 18.0:
        mem_util, comp_len, num_gen = 0.25, 768, 2
    elif total_gib < 30.0:
        mem_util, comp_len, num_gen = 0.45, 1024, 4
    elif total_gib < 60.0:
        mem_util, comp_len, num_gen = 0.55, 1280, 8
    else:
        mem_util, comp_len, num_gen = 0.55, 1536, 8

    if SMOKE_TEST:
        # Smoke runs validate the pipeline cheaply — drop num_gen back to 2
        # so the run finishes fast even on a big GPU.
        num_gen = 2

    if user_mem:
        mem_util = float(user_mem)
    if user_len:
        comp_len = int(user_len)
    if user_gen_int > 0:
        num_gen = user_gen_int

    return mem_util, comp_len, num_gen


# ---------------------------------------------------------------------------
# Hugging Face login (env var only — no interactive popup in headless jobs).
# ---------------------------------------------------------------------------


def _login_to_hub() -> None:
    from huggingface_hub import HfApi, login

    login(token=HF_TOKEN, add_to_git_credential=False)
    user = HfApi().whoami()
    print(f"Logged into HF Hub as {user['name']!r}")


# ---------------------------------------------------------------------------
# ClarifyEnv — copied verbatim from the notebook so this script is the
# single source of truth. Keep these in sync if you change either side.
# ---------------------------------------------------------------------------


def _ws_url(base_url: str) -> str:
    return base_url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/ws"


class ClarifyEnv:
    """ClarifyRL env wrapper for TRL's environment_factory.

    One instance per rollout. Holds a WebSocket connection that lives for the
    duration of the episode. After self.done becomes True, the connection is
    closed automatically and the instance can be discarded.
    """

    def __init__(self) -> None:
        from websockets.sync.client import connect as _ws_connect  # noqa: F401

        self._ws_url = _ws_url(ENV_BASE_URL)
        self._ws = None
        self.reward: float = 0.0
        self.done: bool = False
        self.plan_submitted: bool = False
        self._family: str = ""
        self._required_keys: list[str] = []

    def _open(self, retries: int = 3) -> None:
        from websockets.sync.client import connect as _ws_connect

        if self._ws is not None:
            return
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                self._ws = _ws_connect(
                    self._ws_url, open_timeout=30, close_timeout=10, max_size=2**24
                )
                return
            except Exception as exc:
                last_err = exc
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Could not open WS to {self._ws_url}: {last_err}")

    def _close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def _send(self, payload: dict, timeout: float = 30.0) -> dict:
        self._open()
        self._ws.send(json.dumps(payload))
        raw = self._ws.recv(timeout=timeout)
        return json.loads(raw)

    @staticmethod
    def _extract_tool_result(obs_result: Any) -> dict:
        if isinstance(obs_result, dict):
            if "structured_content" in obs_result and isinstance(
                obs_result["structured_content"], dict
            ):
                return obs_result["structured_content"]
            if "data" in obs_result and isinstance(obs_result["data"], dict):
                return obs_result["data"]
            content = obs_result.get("content")
            if isinstance(content, list) and content:
                txt = content[0].get("text", "")
                try:
                    parsed = json.loads(txt)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
            return obs_result
        if isinstance(obs_result, str):
            try:
                parsed = json.loads(obs_result)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {}

    def reset(self, **kwargs) -> Optional[str]:
        """Reset the env, retrying on transient errors (single-tenant capacity / WS close)."""
        self._close()
        self.reward = 0.0
        self.done = False
        self.plan_submitted = False
        self._family = ""
        self._required_keys = []
        task_id = random.choices(DIFFICULTIES, weights=DIFFICULTY_WEIGHTS, k=1)[0]

        max_attempts = 6  # 0.5+1+2+4+8+16 = 31.5 s total
        last_err: str = ""
        for attempt in range(max_attempts):
            backoff = 0.5 * (2**attempt)
            try:
                resp = self._send({"type": "reset", "data": {"task_id": task_id}})
            except Exception as exc:
                last_err = f"connection error: {exc}"
                self._close()
                if attempt < max_attempts - 1:
                    time.sleep(backoff)
                    continue
                break

            if resp.get("type") == "error":
                err_data = resp.get("data", {}) or {}
                err_code = err_data.get("code") if isinstance(err_data, dict) else None
                if err_code == "CAPACITY_REACHED" and attempt < max_attempts - 1:
                    last_err = "capacity reached, retrying"
                    self._close()
                    time.sleep(backoff)
                    continue
                self.done = True
                return f"ERROR resetting env: {err_data}"

            data = resp.get("data", {})
            obs = data.get("observation", {}) or {}
            self.reward = float(data.get("reward") or 0.0)
            self.done = bool(data.get("done") or False)
            info = self._extract_tool_result(obs.get("result"))
            request_text = info.get("request", "")
            family = info.get("family", "")
            max_steps = info.get("max_steps", 10)
            questions_remaining = info.get("questions_remaining", 6)
            self._family = family
            self._required_keys = REQUIRED_KEYS_BY_FAMILY.get(family, [])
            required_keys_str = ", ".join(self._required_keys) if self._required_keys else "unknown"
            return (
                f"USER REQUEST: {request_text}\n"
                f"Task family: {family}\n"
                f"Required plan fields: {required_keys_str}\n"
                f"You have {max_steps} turns and may ask up to {questions_remaining} clarifying questions.\n"
                f"Use the tools to ask about each required field, then call propose_plan with a JSON string containing ALL required fields."
            )

        self.done = True
        return f"ERROR resetting env after {max_attempts} attempts: {last_err}"

    def _step_raw(self, tool_name: str, arguments: dict) -> dict:
        if self.done:
            return {"error": "episode already ended"}
        action = {"type": "call_tool", "tool_name": tool_name, "arguments": arguments}
        try:
            resp = self._send({"type": "step", "data": action})
        except Exception as exc:
            self.done = True
            self._close()
            return {"error": f"WS step failed: {exc}"}
        if resp.get("type") == "error":
            self.done = True
            self._close()
            return {"error": str(resp.get("data"))}
        data = resp.get("data", {})
        obs = data.get("observation", {}) or {}
        self.reward = float(data.get("reward") or 0.0)
        self.done = bool(data.get("done") or False)
        result_dict = self._extract_tool_result(obs.get("result"))
        if self.done:
            self._close()
        return result_dict

    def ask_question(self, question: str) -> str:
        """Ask the user a single clarifying question to gather missing information.

        Args:
            question: The clarifying question to ask, in plain English.

        Returns:
            The user's answer plus how many questions you have left.
        """
        out = self._step_raw("ask_question", {"question": question})
        if "error" in out:
            return f"Error: {out['error']}"
        answer = out.get("answer", "")
        qr = out.get("questions_remaining", "?")
        revealed = out.get("field_revealed")
        suffix = f" [revealed field: {revealed}]" if revealed else ""
        return f'User answered: "{answer}" (questions remaining: {qr}){suffix}'

    def propose_plan(self, plan: str) -> str:
        """Submit your final plan. The plan must be a JSON STRING. Ends the episode.

        Args:
            plan: A JSON-encoded string with the plan fields.
                MUST be a string, not a dict.

        Returns:
            The final score and per-component breakdown.
        """
        self.plan_submitted = True
        if not isinstance(plan, str):
            try:
                plan = json.dumps(plan)
            except Exception:
                plan = str(plan)
        out = self._step_raw("propose_plan", {"plan": plan})
        if "error" in out:
            return f"Error submitting plan: {out['error']}"
        score = out.get("score", 0.0)
        breakdown = out.get("breakdown", {})
        parse_err = out.get("parse_error")
        parts = [f"Final score: {float(score):.3f}"]
        if parse_err:
            parts.append(f"plan parse error: {parse_err}")
        if breakdown:
            parts.append(f"breakdown: {json.dumps(breakdown)}")
        return " | ".join(parts)

    def get_task_info(self) -> str:
        """Re-read the original user request.

        Returns:
            The user's request, task family, required fields, and remaining question budget.
        """
        out = self._step_raw("get_task_info", {})
        if "error" in out:
            return f"Error: {out['error']}"
        family = out.get("family", self._family)
        rk = REQUIRED_KEYS_BY_FAMILY.get(family, self._required_keys)
        required_keys_str = ", ".join(rk) if rk else "unknown"
        return (
            f"Request: {out.get('request', '')}\n"
            f"Family: {family}\n"
            f"Required plan fields: {required_keys_str}\n"
            f"Questions remaining: {out.get('questions_remaining', '?')}"
        )

    def __del__(self):
        try:
            self._close()
        except Exception:
            pass


NO_PLAN_PENALTY = -0.1
PLAN_SUBMISSION_BONUS = 0.05


def reward_func(environments, **kwargs) -> list[float]:
    rewards = []
    for env in environments:
        if env.plan_submitted:
            r = float(env.reward) + PLAN_SUBMISSION_BONUS
        else:
            r = NO_PLAN_PENALTY
        rewards.append(r)
    return rewards


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("ClarifyRL GRPO training")
    print("=" * 70)
    print(f"  Model:           {MODEL_NAME}")
    print(f"  Output dir:      {OUTPUT_DIR}")
    print(f"  Trackio space:   {TRACKIO_SPACE_ID}")
    print(f"  Env base URL:    {ENV_BASE_URL}")
    print(f"  Max steps:       {MAX_STEPS}")
    print(f"  Seed:            {SEED}")
    print(f"  Num generations: {NUM_GENERATIONS or 'auto'}")
    print(f"  Grad accum:      {GRAD_ACCUM_STEPS}")
    print(f"  Smoke test:      {SMOKE_TEST}")
    print(f"  Resume from:     {RESUME_FROM_CKPT or '(no resume)'}")
    print(f"  Push to hub:     {not SKIP_PUSH}")
    print("=" * 70)

    random.seed(SEED)
    _login_to_hub()

    # --- Sanity-check transformers / dependencies ------------------------
    import transformers
    from packaging.version import Version

    if Version(transformers.__version__) < Version("5.2.0"):
        raise RuntimeError(
            f"transformers {transformers.__version__} is too old for `environment_factory`. "
            "Install transformers from main: pip install -U git+https://github.com/huggingface/transformers.git@main"
        )

    import importlib.util

    for _pkg in ("jmespath", "bitsandbytes", "trl", "vllm", "trackio", "websockets"):
        if importlib.util.find_spec(_pkg) is None:
            raise RuntimeError(f"{_pkg} is not installed")
    print(f"transformers {transformers.__version__} | jmespath/bitsandbytes/trl/vllm OK")

    # --- GPU memory cleanup before trainer init --------------------------
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        free_b, total_b = torch.cuda.mem_get_info()
        free_gib = free_b / 1024**3
        total_gib = total_b / 1024**3
        print(f"GPU memory before trainer init: {free_gib:.2f} / {total_gib:.2f} GiB free")
        if free_gib < 3.0:
            raise RuntimeError(
                f"Only {free_gib:.2f} GiB free — not enough headroom. Aborting fail-fast."
            )
    else:
        print("WARNING: no CUDA device — training will be unusably slow")

    # --- Resolve auto-tuned knobs first so dataset sizing is correct -----
    from trl import GRPOConfig, GRPOTrainer

    vllm_mem_util, max_comp_len, num_gen = _autotune_for_gpu()
    print(
        f"vllm_gpu_memory_utilization={vllm_mem_util} | "
        f"max_completion_length={max_comp_len} | num_generations={num_gen}"
    )

    # --- Build dataset (one row per rollout, prompt is just the system) --
    from datasets import Dataset

    num_train_episodes = max(2000, MAX_STEPS * num_gen * GRAD_ACCUM_STEPS * 2)
    dataset = Dataset.from_dict(
        {"prompt": [[{"role": "system", "content": PROMPT}] for _ in range(num_train_episodes)]}
    )
    print(f"Dataset: {len(dataset)} episodes (≈ {MAX_STEPS * num_gen * GRAD_ACCUM_STEPS} consumed)")

    # --- GRPOConfig ------------------------------------------------------
    # Some optional knobs (chat_template_kwargs, log_completions,
    # num_completions_to_print, vllm_max_model_length) only exist in newer
    # TRL versions. Build the kwargs dict and filter out anything the
    # installed `GRPOConfig` doesn't accept, instead of crashing.
    import dataclasses as _dc
    import trl as _trl  # noqa: PLC0415

    print(f"trl version: {getattr(_trl, '__version__', '?')}")
    _grpo_field_names = {f.name for f in _dc.fields(GRPOConfig)}

    _grpo_kwargs: dict[str, Any] = {
        "num_train_epochs": 1,
        "learning_rate": LEARNING_RATE,
        "beta": BETA,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "per_device_train_batch_size": 1,
        "warmup_steps": WARMUP_STEPS,
        "max_steps": MAX_STEPS,
        "optim": "adamw_8bit",
        "max_grad_norm": 1.0,
        "seed": SEED,
        "num_generations": num_gen,
        "max_completion_length": max_comp_len,
        "log_completions": True,
        "num_completions_to_print": 2,
        "chat_template_kwargs": {"enable_thinking": False},
        "use_vllm": True,
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": vllm_mem_util,
        "vllm_max_model_length": VLLM_MAX_MODEL_LEN,
        "output_dir": OUTPUT_DIR,
        "report_to": "trackio",
        "trackio_space_id": TRACKIO_SPACE_ID,
        "logging_steps": LOGGING_STEPS,
        "save_steps": SAVE_STEPS,
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "gradient_checkpointing": True,
        "push_to_hub": not SKIP_PUSH,
    }

    _dropped = [k for k in _grpo_kwargs if k not in _grpo_field_names]
    if _dropped:
        print(f"WARNING: dropping {_dropped} — not supported in this TRL version")
        for k in _dropped:
            _grpo_kwargs.pop(k)

    grpo_config = GRPOConfig(**_grpo_kwargs)

    # --- Patch jupyter OutStream.fileno (no-op outside Jupyter) ----------
    try:
        from ipykernel.iostream import OutStream
        OutStream.fileno = lambda self: sys.__stdout__.fileno()
    except Exception:
        pass

    # --- Build & train ---------------------------------------------------
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=grpo_config,
        environment_factory=ClarifyEnv,
    )
    print("Trainer initialised, starting train()...")

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.name} | Total: {gpu.total_memory / 1024**3:.2f} GiB")

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # --- Resume logic: if RESUME_FROM_CKPT is set or a checkpoint already
    # lives under OUTPUT_DIR, pass it to trainer.train(). This makes a job
    # restart pick up where the previous run died, instead of redoing work.
    resume_arg: Any = False
    if RESUME_FROM_CKPT:
        if Path(RESUME_FROM_CKPT).exists():
            resume_arg = RESUME_FROM_CKPT
            print(f"Resuming from explicit checkpoint: {RESUME_FROM_CKPT}")
        else:
            print(f"WARNING: RESUME_FROM_CKPT='{RESUME_FROM_CKPT}' does not exist, ignoring")
    else:
        existing_ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if existing_ckpts:
            resume_arg = True
            print(f"Auto-resume: found {len(existing_ckpts)} existing checkpoint(s), latest={existing_ckpts[-1].name}")

    t0 = time.time()
    train_failed = False
    failure_reason = ""
    trainer_stats = None
    try:
        trainer_stats = trainer.train(resume_from_checkpoint=resume_arg) if resume_arg else trainer.train()
    except torch.cuda.OutOfMemoryError as oom:
        train_failed = True
        failure_reason = f"OOM: {oom}"
        print(f"\nFATAL: GPU OOM during training: {oom}")
        print("Hint: reduce MAX_COMPLETION_LEN or NUM_GENERATIONS for next launch.")
    except KeyboardInterrupt:
        train_failed = True
        failure_reason = "interrupted"
        print("\nInterrupted by user.")
    except Exception as exc:
        train_failed = True
        failure_reason = f"{type(exc).__name__}: {exc}"
        print(f"\nFATAL: training raised {failure_reason}")

    train_secs = time.time() - t0
    print(f"\nTraining loop exited in {train_secs:.1f} s ({train_secs / 60:.1f} min) | failed={train_failed}")
    if trainer_stats is not None:
        print(f"Stats: {trainer_stats}")

    # --- Save & push (always — even on failure we keep the partial model) ---
    try:
        trainer.save_model(OUTPUT_DIR)
        print(f"Saved model to {OUTPUT_DIR}")
    except Exception as exc:
        print(f"WARNING: save_model failed: {exc}")

    # Persist log_history for downstream plots — TRL writes this anyway,
    # but writing it explicitly makes the structure stable across versions.
    log_history = list(getattr(trainer.state, "log_history", []) or [])
    (out / "log_history.json").write_text(json.dumps(log_history, indent=2))
    print(f"Wrote {len(log_history)} log entries to {out / 'log_history.json'}")

    summary = {
        "model": MODEL_NAME,
        "max_steps": MAX_STEPS,
        "num_generations": num_gen,
        "vllm_gpu_memory_utilization": vllm_mem_util,
        "max_completion_length": max_comp_len,
        "train_seconds": train_secs,
        "stats": str(trainer_stats) if trainer_stats is not None else None,
        "failed": train_failed,
        "failure_reason": failure_reason,
        "output_dir": OUTPUT_DIR,
        "trackio_space_id": TRACKIO_SPACE_ID,
        "num_log_entries": len(log_history),
        "smoke_test": SMOKE_TEST,
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2))

    if not SKIP_PUSH and not train_failed:
        try:
            trainer.push_to_hub()
            print(f"Pushed {OUTPUT_DIR} to the HF Hub")
        except Exception as exc:
            print(f"WARNING: push_to_hub failed: {exc} — model is still saved locally")

    if train_failed:
        (out / "FAILED").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n{failure_reason}\n")
        print(f"FAILED — wrote {out / 'FAILED'} marker. Exit 2.")
        sys.exit(2)

    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    print("DONE — exit 0")


if __name__ == "__main__":
    main()
