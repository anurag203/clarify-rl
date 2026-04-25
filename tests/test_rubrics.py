"""Tests for server.rubrics — composable rubric components.

Requires openenv stack installed. Run with:
    python -m pytest tests/test_rubrics.py -v
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

from server.rubrics import (
    FieldMatchRubric,
    FormatCheckRubric,
    HallucinationCheckRubric,
    InfoGainRubric,
    QuestionEfficiencyRubric,
    RubricContext,
    build_rubric,
    score_breakdown,
)


def _ctx(**overrides) -> RubricContext:
    defaults = dict(
        family="support_triage",
        hidden_profile={"order_id": "#4521", "item_issue": "wrong-item", "refund_or_replace": "refund"},
        critical_fields=frozenset({"order_id", "item_issue", "refund_or_replace"}),
        required_keys=("order_id", "item_issue", "refund_or_replace"),
        asked_field_keys=frozenset({"order_id", "item_issue", "refund_or_replace"}),
        questions_asked_count=3,
        max_questions=6,
        parsed_plan={"order_id": "#4521", "item_issue": "wrong-item", "refund_or_replace": "refund"},
        parse_error=None,
    )
    defaults.update(overrides)
    return RubricContext(**defaults)


class TestFormatCheck:
    r = FormatCheckRubric()

    def test_valid_plan(self):
        assert self.r(None, _ctx()) == 1.0

    def test_missing_key(self):
        plan = {"order_id": "#4521", "item_issue": "wrong-item"}
        assert self.r(None, _ctx(parsed_plan=plan)) == 0.0

    def test_empty_value(self):
        plan = {"order_id": "", "item_issue": "wrong-item", "refund_or_replace": "refund"}
        assert self.r(None, _ctx(parsed_plan=plan)) == 0.0

    def test_null_value(self):
        plan = {"order_id": None, "item_issue": "wrong-item", "refund_or_replace": "refund"}
        assert self.r(None, _ctx(parsed_plan=plan)) == 0.0

    def test_none_plan(self):
        assert self.r(None, _ctx(parsed_plan=None)) == 0.0

    def test_extra_keys_ok(self):
        plan = {"order_id": "#4521", "item_issue": "wrong-item", "refund_or_replace": "refund", "extra": "ok"}
        assert self.r(None, _ctx(parsed_plan=plan)) == 1.0


class TestFieldMatch:
    r = FieldMatchRubric()

    def test_perfect_match(self):
        assert self.r(None, _ctx()) == 1.0

    def test_no_match(self):
        plan = {"order_id": "WRONG", "item_issue": "WRONG", "refund_or_replace": "WRONG"}
        assert self.r(None, _ctx(parsed_plan=plan)) == 0.0

    def test_partial_match(self):
        plan = {"order_id": "#4521", "item_issue": "WRONG", "refund_or_replace": "WRONG"}
        score = self.r(None, _ctx(parsed_plan=plan))
        assert 0.0 < score < 1.0

    def test_fuzzy_numeric(self):
        ctx = _ctx(
            hidden_profile={"guest_count": 20},
            critical_fields=frozenset({"guest_count"}),
            required_keys=("guest_count",),
            parsed_plan={"guest_count": "20"},
            family="event_planning",
        )
        assert self.r(None, ctx) == 1.0


class TestInfoGain:
    r = InfoGainRubric()

    def test_all_revealed(self):
        assert self.r(None, _ctx()) == 1.0

    def test_none_revealed(self):
        assert self.r(None, _ctx(asked_field_keys=frozenset())) == 0.0

    def test_partial(self):
        asked = frozenset({"order_id"})
        score = self.r(None, _ctx(asked_field_keys=asked))
        assert abs(score - 1 / 3) < 0.01


class TestQuestionEfficiency:
    r = QuestionEfficiencyRubric()

    def test_zero_questions(self):
        assert self.r(None, _ctx(questions_asked_count=0)) == 1.0

    def test_max_questions(self):
        assert self.r(None, _ctx(questions_asked_count=6)) == 0.0

    def test_half(self):
        assert abs(self.r(None, _ctx(questions_asked_count=3)) - 0.5) < 0.01


class TestHallucinationCheck:
    r = HallucinationCheckRubric()

    def test_all_asked(self):
        assert self.r(None, _ctx()) == 1.0

    def test_none_asked(self):
        score = self.r(None, _ctx(asked_field_keys=frozenset()))
        assert score == 0.0

    def test_partial(self):
        asked = frozenset({"order_id"})
        score = self.r(None, _ctx(asked_field_keys=asked))
        assert 0.0 < score < 1.0


class TestComposedRubric:
    def test_oracle_high_score(self):
        rubric = build_rubric()
        score = float(rubric(None, _ctx()))
        assert score > 0.75

    def test_bad_format_zero(self):
        rubric = build_rubric()
        score = float(rubric(None, _ctx(parsed_plan=None)))
        assert score == 0.0

    def test_no_questions_still_scores(self):
        rubric = build_rubric()
        ctx = _ctx(asked_field_keys=frozenset(), questions_asked_count=0)
        score = float(rubric(None, ctx))
        assert score < 0.5

    def test_breakdown_keys(self):
        rubric = build_rubric()
        rubric(None, _ctx())
        bd = score_breakdown(rubric)
        expected_keys = {
            "FormatCheckRubric", "FieldMatchRubric", "InfoGainRubric",
            "QuestionEfficiencyRubric", "HallucinationCheckRubric",
        }
        assert set(bd.keys()) == expected_keys
