"""
Per-step shaping rewards + plan parsing helpers.

The composable rubric in `server.rubrics` only fires on the terminal step
(`propose_plan`). GRPO benefits from a denser signal during exploration, so
the env emits small per-step shaping rewards on `ask_question` calls.

This module is the thin glue between raw tool-call results and reward floats.
"""

from __future__ import annotations

import json
from typing import Any, Optional


REWARD_REVEAL_NEW_FIELD: float = 0.05
REWARD_NO_USEFUL_INFO: float = 0.02
PENALTY_DUPLICATE_QUESTION: float = -0.02
PENALTY_OVER_CAP: float = -0.05


def parse_plan(plan_str: Any) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if plan_str is None:
        return None, "plan is null"
    if isinstance(plan_str, dict):
        return plan_str, None
    if not isinstance(plan_str, str):
        return None, f"plan must be a JSON string, got {type(plan_str).__name__}"
    text = plan_str.strip()
    if not text:
        return None, "plan is empty"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg}"
    if not isinstance(parsed, dict):
        return None, f"plan must be a JSON object, got {type(parsed).__name__}"
    return parsed, None


def ask_question_reward(
    *,
    over_cap: bool,
    is_duplicate_field: bool,
    revealed_new_field: bool,
) -> float:
    if over_cap:
        return PENALTY_OVER_CAP
    if is_duplicate_field:
        return PENALTY_DUPLICATE_QUESTION
    if revealed_new_field:
        return REWARD_REVEAL_NEW_FIELD
    return REWARD_NO_USEFUL_INFO


__all__ = [
    "REWARD_REVEAL_NEW_FIELD",
    "REWARD_NO_USEFUL_INFO",
    "PENALTY_DUPLICATE_QUESTION",
    "PENALTY_OVER_CAP",
    "parse_plan",
    "ask_question_reward",
]
