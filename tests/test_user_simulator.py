"""Tests for server.user_simulator — keyword matching + answer generation."""
from __future__ import annotations

import pytest

from server.scenarios import FIELD_VOCAB, TASK_FIELDS
from server.user_simulator import FIELD_KEYWORDS, answer, format_answer, match_field


class TestFieldKeywordCoverage:
    @pytest.mark.parametrize("family", TASK_FIELDS.keys())
    def test_every_field_reachable(self, family):
        for field in TASK_FIELDS[family]:
            assert field in FIELD_KEYWORDS, (
                f"FIELD_KEYWORDS missing entry for {family}.{field}"
            )
            assert len(FIELD_KEYWORDS[field]) > 0, (
                f"FIELD_KEYWORDS[{field}] has no keywords"
            )


class TestMatchField:
    def test_stack_matches(self):
        assert match_field("what stack should I use?", ["stack", "scale"]) == "stack"

    def test_no_match(self):
        assert match_field("tell me a joke", ["stack", "scale"]) is None

    def test_longest_keyword_wins(self):
        result = match_field("what's wrong with the order?", ["item_issue", "order_id"])
        assert result == "item_issue"

    def test_restricted_to_allowed_keys(self):
        assert match_field("what stack?", ["scale", "auth"]) is None


class TestFormatAnswer:
    def test_ends_with_punctuation(self):
        text = format_answer("stack", "python+fastapi", "coding_requirements")
        assert text.endswith(".")

    def test_value_appears(self):
        text = format_answer("order_id", "#4521", "support_triage")
        assert "#4521" in text


class TestAnswer:
    def test_known_field(self):
        profile = {"stack": "python+fastapi", "scale": "1k users"}
        text, matched = answer("what stack?", profile, "coding_requirements")
        assert matched == "stack"
        assert "python+fastapi" in text

    def test_unknown_question(self):
        profile = {"stack": "go+gin"}
        text, matched = answer("tell me about your cat", profile, "coding_requirements")
        assert matched is None
        assert "preference" in text.lower() or "don't" in text.lower()

    @pytest.mark.parametrize("family", TASK_FIELDS.keys())
    def test_all_family_fields_answerable(self, family):
        profile = {f: FIELD_VOCAB[f][0] for f in TASK_FIELDS[family]}
        for field in TASK_FIELDS[family]:
            kw = FIELD_KEYWORDS[field][0]
            q = f"What is the {kw}?"
            _, matched = answer(q, profile, family)
            assert matched == field, (
                f"Q='{q}' for {family}.{field} matched {matched} instead"
            )
