"""Integration tests for ClarifyEnvironment — full episode flows.

Requires full openenv stack installed. Run with:
    python -m pytest tests/test_environment.py -v
Skip with:
    python -m pytest -m 'not integration'
"""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.integration

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from server.clarify_environment import ClarifyEnvironment
from server.scenarios import REQUIRED_KEYS_BY_FAMILY


@pytest.fixture
def env():
    e = ClarifyEnvironment()
    e.reset(seed=7, task_id="medium")
    return e


def _call(env, tool, **kwargs):
    action = CallToolAction(tool_name=tool, arguments=kwargs)
    return env.step(action)


class TestReset:
    def test_returns_observation(self):
        e = ClarifyEnvironment()
        obs = e.reset(seed=42, task_id="easy")
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.result["type"] == "task"

    def test_state_after_reset(self):
        e = ClarifyEnvironment()
        e.reset(seed=42, task_id="medium")
        s = e.state
        assert s.step_count == 0
        assert s.questions_remaining == 6
        assert s.episode_done is False
        assert s.plan_submitted is False

    @pytest.mark.parametrize("diff", ["easy", "medium", "hard"])
    def test_all_difficulties(self, diff):
        e = ClarifyEnvironment()
        obs = e.reset(seed=1, task_id=diff)
        assert obs.result["task_id"] == diff


class TestListTools:
    def test_returns_three_tools(self, env):
        obs = env.step(ListToolsAction())
        names = [t.name for t in obs.tools]
        assert set(names) == {"get_task_info", "ask_question", "propose_plan"}


class TestGetTaskInfo:
    def test_free_action(self, env):
        obs = _call(env, "get_task_info")
        assert obs.reward == 0.0
        assert obs.done is False
        data = obs.result.data
        assert "request" in data
        assert "family" in data

    def test_step_count_increments(self, env):
        _call(env, "get_task_info")
        assert env.state.step_count == 1


class TestAskQuestion:
    def test_reveals_field(self, env):
        obs = _call(env, "ask_question", question="what is the order id?")
        data = obs.result.data
        assert data["field_revealed"] is not None
        assert obs.reward > 0

    def test_budget_decrements(self, env):
        _call(env, "ask_question", question="what is the order id?")
        assert env.state.questions_remaining == 5

    def test_duplicate_penalty(self, env):
        _call(env, "ask_question", question="what is the order id?")
        obs2 = _call(env, "ask_question", question="tell me order id again?")
        assert obs2.reward < 0
        assert obs2.result.data["duplicate"] is True

    def test_unknown_question_small_reward(self, env):
        obs = _call(env, "ask_question", question="tell me about your cat")
        assert obs.reward > 0
        assert obs.result.data["field_revealed"] is None

    def test_over_cap(self, env):
        for i in range(6):
            _call(env, "ask_question", question=f"question {i}")
        obs = _call(env, "ask_question", question="one more")
        assert obs.result.data["over_cap"] is True
        assert obs.done is True

    def test_truncates_long_question(self, env):
        long_q = "x" * 500
        obs = _call(env, "ask_question", question=long_q)
        assert obs.done is False


class TestProposePlan:
    def test_terminates_episode(self, env):
        plan = json.dumps({"order_id": "#1", "item_issue": "late", "refund_or_replace": "refund"})
        obs = _call(env, "propose_plan", plan=plan)
        assert obs.done is True
        assert env.state.episode_done is True
        assert env.state.plan_submitted is True
        assert env.state.final_score is not None

    def test_bad_json_zero_score(self, env):
        obs = _call(env, "propose_plan", plan="not json")
        assert obs.done is True
        assert env.state.final_score == 0.0

    def test_missing_keys_zero_score(self, env):
        obs = _call(env, "propose_plan", plan='{"order_id": "#1"}')
        assert obs.done is True
        assert env.state.final_score == 0.0

    def test_breakdown_populated(self, env):
        plan = json.dumps({"order_id": "#1", "item_issue": "late", "refund_or_replace": "refund"})
        _call(env, "propose_plan", plan=plan)
        bd = env.state.score_breakdown
        assert "FormatCheckRubric" in bd


class TestEpisodeDoneGuard:
    def test_no_ask_after_plan(self, env):
        plan = json.dumps({"order_id": "#1", "item_issue": "late", "refund_or_replace": "refund"})
        _call(env, "propose_plan", plan=plan)
        obs = _call(env, "ask_question", question="more info?")
        assert obs.done is True
        assert obs.result.data.get("error") == "episode already ended"

    def test_no_plan_after_plan(self, env):
        plan = json.dumps({"order_id": "#1", "item_issue": "late", "refund_or_replace": "refund"})
        _call(env, "propose_plan", plan=plan)
        obs = _call(env, "propose_plan", plan=plan)
        assert obs.done is True
        assert obs.result.data.get("error") == "episode already ended"

    def test_no_info_after_plan(self, env):
        plan = json.dumps({"order_id": "#1", "item_issue": "late", "refund_or_replace": "refund"})
        _call(env, "propose_plan", plan=plan)
        obs = _call(env, "get_task_info")
        assert obs.done is True


class TestMaxSteps:
    def test_max_steps_enforced(self):
        e = ClarifyEnvironment()
        e.reset(seed=7, task_id="easy")
        for i in range(10):
            obs = _call(e, "get_task_info")
            if obs.done:
                assert e.state.step_count <= 8
                return
        pytest.fail("max_steps not enforced within 10 calls")


class TestStepCount:
    def test_increments_each_call(self, env):
        for i in range(1, 4):
            _call(env, "get_task_info")
            assert env.state.step_count == i


class TestOraclePolicy:
    def test_oracle_scores_high(self):
        for seed in range(10):
            e = ClarifyEnvironment()
            obs = e.reset(seed=seed, task_id="medium")
            family = obs.result["family"]
            profile = e._scenario["hidden_profile"]
            critical = e._scenario["critical_fields"]
            required_keys = REQUIRED_KEYS_BY_FAMILY[family]

            for cf in critical:
                kw = cf.replace("_", " ")
                _call(e, "ask_question", question=f"what is the {kw}?")

            plan = json.dumps(profile)
            obs = _call(e, "propose_plan", plan=plan)
            assert obs.done is True
            score = e.state.final_score
            assert score is not None
            assert score > 0.5, f"seed={seed} family={family} oracle score={score} too low"
