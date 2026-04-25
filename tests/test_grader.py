"""Tests for server.grader — plan parsing + per-step shaping rewards."""
from __future__ import annotations

from server.grader import (
    PENALTY_DUPLICATE_QUESTION,
    PENALTY_OVER_CAP,
    REWARD_NO_USEFUL_INFO,
    REWARD_REVEAL_NEW_FIELD,
    ask_question_reward,
    parse_plan,
)


class TestParsePlan:
    def test_valid_json(self):
        plan, err = parse_plan('{"a": 1}')
        assert plan == {"a": 1}
        assert err is None

    def test_dict_passthrough(self):
        plan, err = parse_plan({"a": 1})
        assert plan == {"a": 1}
        assert err is None

    def test_null(self):
        plan, err = parse_plan(None)
        assert plan is None
        assert "null" in err

    def test_empty_string(self):
        plan, err = parse_plan("")
        assert plan is None
        assert "empty" in err

    def test_invalid_json(self):
        plan, err = parse_plan("{bad}")
        assert plan is None
        assert "invalid JSON" in err

    def test_json_array_rejected(self):
        plan, err = parse_plan("[1, 2]")
        assert plan is None
        assert "object" in err

    def test_non_string(self):
        plan, err = parse_plan(42)
        assert plan is None
        assert "JSON string" in err


class TestAskQuestionReward:
    def test_over_cap(self):
        assert ask_question_reward(over_cap=True, is_duplicate_field=False, revealed_new_field=False) == PENALTY_OVER_CAP

    def test_duplicate(self):
        assert ask_question_reward(over_cap=False, is_duplicate_field=True, revealed_new_field=False) == PENALTY_DUPLICATE_QUESTION

    def test_reveal(self):
        assert ask_question_reward(over_cap=False, is_duplicate_field=False, revealed_new_field=True) == REWARD_REVEAL_NEW_FIELD

    def test_no_info(self):
        assert ask_question_reward(over_cap=False, is_duplicate_field=False, revealed_new_field=False) == REWARD_NO_USEFUL_INFO

    def test_reward_magnitudes(self):
        assert REWARD_REVEAL_NEW_FIELD > REWARD_NO_USEFUL_INFO > 0
        assert PENALTY_DUPLICATE_QUESTION < 0
        assert PENALTY_OVER_CAP < PENALTY_DUPLICATE_QUESTION
