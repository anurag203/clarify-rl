"""
ClarifyEnvironment — OpenEnv MCPEnvironment for the ClarifyRL task.

Three MCP tools:
- `get_task_info()` — free, returns the original ambiguous request and meta
- `ask_question(question)` — costs 1 from the 6-question budget
- `propose_plan(plan)` — terminal; runs the composable rubric
"""

from __future__ import annotations

import random
from typing import Any, Optional

from fastmcp import FastMCP

from openenv.core.env_server.interfaces import EnvironmentMetadata
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from server.grader import (
    PENALTY_OVER_CAP,
    ask_question_reward,
    parse_plan,
)
from server.rubrics import RubricContext, build_rubric, score_breakdown
from server.scenarios import Scenario, generate
from server.user_simulator import answer

from models import ClarifyState


_INSTRUCTIONS = (
    "Ask clarifying questions via ask_question(question) — you have a 6-question budget. "
    "Then submit your final plan via propose_plan(plan) where plan is a JSON string "
    "object containing the required keys for the task family. "
    "Avoid hallucinating values for fields you never asked about."
)


class ClarifyEnvironment(MCPEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, max_questions: int = 6) -> None:
        mcp_server = FastMCP("clarify_rl")

        def get_task_info() -> dict[str, Any]:
            return self._tool_get_task_info()

        def ask_question(question: str) -> dict[str, Any]:
            return self._tool_ask_question(question)

        def propose_plan(plan: str) -> dict[str, Any]:
            return self._tool_propose_plan(plan)

        mcp_server.tool()(get_task_info)
        mcp_server.tool()(ask_question)
        mcp_server.tool()(propose_plan)

        super().__init__(mcp_server=mcp_server)
        self.rubric = build_rubric()

        self._default_max_questions: int = max_questions
        self._scenario: Optional[Scenario] = None
        self._asked_field_keys: set[str] = set()
        self._public_state: ClarifyState = ClarifyState()
        self._last_step_reward: float = 0.0
        self._last_step_done: bool = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CallToolObservation:
        task_id = kwargs.get("task_id", "medium")
        if seed is None:
            seed = random.randint(0, 10**9)

        sc = generate(seed=seed, task_id=task_id)
        self._scenario = sc
        self._asked_field_keys = set()
        self._last_step_reward = 0.0
        self._last_step_done = False

        self._public_state = ClarifyState(
            episode_id=episode_id,
            step_count=0,
            task_id=sc["task_id"],
            task_title=sc["task_title"],
            questions_asked=[],
            questions_remaining=sc["max_questions"],
            answers_received=[],
            fields_revealed=[],
            plan_submitted=False,
            episode_done=False,
            final_score=None,
            score_breakdown=None,
        )

        result = {
            "type": "task",
            "request": sc["request"],
            "task_id": sc["task_id"],
            "task_title": sc["task_title"],
            "family": sc["family"],
            "max_steps": sc["max_steps"],
            "questions_remaining": sc["max_questions"],
            "instructions": _INSTRUCTIONS,
        }
        return CallToolObservation(
            tool_name="reset",
            result=result,
            done=False,
            reward=0.0,
        )

    def _patch_obs(self, obs: CallToolObservation, action: Any) -> CallToolObservation:
        if not isinstance(action, CallToolAction):
            return obs
        obs.reward = self._last_step_reward
        obs.done = self._last_step_done
        self._public_state.step_count = self._public_state.step_count + 1
        if self._last_step_done:
            self._public_state.episode_done = True
        sc = self._scenario
        if sc and self._public_state.step_count >= sc["max_steps"] and not obs.done:
            obs.done = True
            self._public_state.episode_done = True
        return obs

    def step(
        self,
        action: Any,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CallToolObservation:
        obs = super().step(action, timeout_s=timeout_s, **kwargs)
        return self._patch_obs(obs, action)

    async def step_async(
        self,
        action: Any,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CallToolObservation:
        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)
        return self._patch_obs(obs, action)

    def _step_impl(
        self,
        action: Any,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CallToolObservation:
        del timeout_s, kwargs
        return CallToolObservation(
            tool_name=getattr(action, "tool_name", "unknown"),
            result=None,
            done=False,
            reward=0.0,
        )

    @property
    def state(self) -> ClarifyState:
        return self._public_state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="ClarifyRL — AskBeforeYouAct",
            description=(
                "Train LLMs to ask clarifying questions instead of hallucinating. "
                "Five task families (coding / medical-intake / support-triage / meeting / event), "
                "rule-based simulator, composable rubric."
            ),
            version="0.1.0",
            author="Team Bhole Chature",
        )

    def _require_scenario(self) -> Scenario:
        if self._scenario is None:
            raise RuntimeError("Environment must be reset() before tool calls.")
        return self._scenario

    def _guard_episode_done(self) -> Optional[dict[str, Any]]:
        if self._public_state.episode_done:
            self._last_step_reward = 0.0
            self._last_step_done = True
            return {"error": "episode already ended", "episode_done": True}
        return None

    def _tool_get_task_info(self) -> dict[str, Any]:
        sc = self._require_scenario()
        blocked = self._guard_episode_done()
        if blocked:
            return blocked
        self._last_step_reward = 0.0
        self._last_step_done = False
        return {
            "request": sc["request"],
            "task_id": sc["task_id"],
            "task_title": sc["task_title"],
            "family": sc["family"],
            "questions_remaining": self._public_state.questions_remaining,
            "instructions": _INSTRUCTIONS,
        }

    def _tool_ask_question(self, question: str) -> dict[str, Any]:
        sc = self._require_scenario()
        st = self._public_state

        blocked = self._guard_episode_done()
        if blocked:
            return blocked

        question = question[:200]

        if st.questions_remaining <= 0:
            self._last_step_reward = PENALTY_OVER_CAP
            self._last_step_done = True
            return {
                "answer": "(no more questions allowed)",
                "questions_remaining": 0,
                "field_revealed": None,
                "duplicate": False,
                "over_cap": True,
            }

        text, matched = answer(question, sc["hidden_profile"], sc["family"])
        is_duplicate = matched is not None and matched in self._asked_field_keys
        revealed_new = matched is not None and not is_duplicate

        if revealed_new:
            self._asked_field_keys.add(matched)
            st.fields_revealed = sorted(self._asked_field_keys)

        st.questions_asked = st.questions_asked + [question]
        st.answers_received = st.answers_received + [text]
        st.questions_remaining = st.questions_remaining - 1

        self._last_step_reward = ask_question_reward(
            over_cap=False,
            is_duplicate_field=is_duplicate,
            revealed_new_field=revealed_new,
        )
        self._last_step_done = False

        return {
            "answer": text,
            "questions_remaining": st.questions_remaining,
            "field_revealed": matched if revealed_new else None,
            "duplicate": is_duplicate,
            "over_cap": False,
        }

    def _tool_propose_plan(self, plan: str) -> dict[str, Any]:
        sc = self._require_scenario()
        st = self._public_state

        blocked = self._guard_episode_done()
        if blocked:
            return blocked

        parsed, parse_err = parse_plan(plan)
        ctx = RubricContext(
            family=sc["family"],
            hidden_profile=sc["hidden_profile"],
            critical_fields=frozenset(sc["critical_fields"]),
            required_keys=tuple(sc["required_keys"]),
            asked_field_keys=frozenset(self._asked_field_keys),
            questions_asked_count=len(st.questions_asked),
            max_questions=sc["max_questions"],
            parsed_plan=parsed,
            parse_error=parse_err,
        )

        score = float(self.rubric(action=None, observation=ctx))
        breakdown = score_breakdown(self.rubric)

        self._last_step_reward = score
        self._last_step_done = True

        st.plan_submitted = True
        st.episode_done = True
        st.final_score = score
        st.score_breakdown = breakdown

        return {
            "type": "resolution",
            "score": score,
            "breakdown": breakdown,
            "expected_profile": sc["hidden_profile"],
            "critical_fields": list(sc["critical_fields"]),
            "required_keys": list(sc["required_keys"]),
            "submitted_plan": parsed,
            "parse_error": parse_err,
            "questions_asked": len(st.questions_asked),
            "fields_revealed": sorted(self._asked_field_keys),
        }


__all__ = ["ClarifyEnvironment"]
