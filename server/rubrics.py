"""
Composable rubric for ClarifyRL terminal-step scoring.

Top-level shape:

    Sequential(
        Gate(FormatCheckRubric, threshold=0.5),
        WeightedSum([FieldMatch 0.50, InfoGain 0.20, Efficiency 0.15, Hallucination 0.15]),
    )

Each component reads the per-episode ground truth from a `RubricContext` object
that the environment builds at the terminal step and passes in place of the
public `CallToolObservation`. The agent never sees this context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from openenv.core.rubrics import Gate, Rubric, Sequential, WeightedSum

from server.scenarios import FIELD_VOCAB, TASK_FIELDS


CRITICAL_BONUS_WEIGHT: float = 2.0
NON_CRITICAL_WEIGHT: float = 1.0


@dataclass
class RubricContext:
    family: str
    hidden_profile: dict[str, Any]
    critical_fields: frozenset[str]
    required_keys: tuple[str, ...]
    asked_field_keys: frozenset[str]
    questions_asked_count: int
    max_questions: int
    parsed_plan: Optional[dict[str, Any]]
    parse_error: Optional[str] = None


def _normalize(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("-", " ").replace("_", " ").split())


_VALUE_SYNONYMS: dict[str, frozenset[str]] = {
    "veg": frozenset({"veg", "vegetarian"}),
    "vegetarian": frozenset({"veg", "vegetarian"}),
    "google meet": frozenset({"google meet", "gmeet", "meet"}),
    "in person": frozenset({"in person", "inperson", "offline"}),
}


def _values_equal(submitted: Any, expected: Any) -> bool:
    a = _normalize(submitted)
    b = _normalize(expected)
    if a == b:
        return True
    if a in _VALUE_SYNONYMS and b in _VALUE_SYNONYMS[a]:
        return True
    if b in _VALUE_SYNONYMS and a in _VALUE_SYNONYMS[b]:
        return True
    try:
        return float(a) == float(b)
    except (ValueError, TypeError):
        return False


class FormatCheckRubric(Rubric):
    def forward(self, action: Any, observation: Any) -> float:
        del action
        ctx: RubricContext = observation
        plan = ctx.parsed_plan
        if not isinstance(plan, dict):
            return 0.0
        for key in ctx.required_keys:
            if key not in plan:
                return 0.0
            value = plan[key]
            if value is None:
                return 0.0
            if isinstance(value, str) and not value.strip():
                return 0.0
        return 1.0


class FieldMatchRubric(Rubric):
    def forward(self, action: Any, observation: Any) -> float:
        del action
        ctx: RubricContext = observation
        plan = ctx.parsed_plan or {}
        profile = ctx.hidden_profile
        if not profile:
            return 1.0
        total_weight = 0.0
        matched_weight = 0.0
        for field_key, expected in profile.items():
            w = CRITICAL_BONUS_WEIGHT if field_key in ctx.critical_fields else NON_CRITICAL_WEIGHT
            total_weight += w
            if field_key in plan and _values_equal(plan[field_key], expected):
                matched_weight += w
        return matched_weight / total_weight if total_weight else 0.0


class InfoGainRubric(Rubric):
    def forward(self, action: Any, observation: Any) -> float:
        del action
        ctx: RubricContext = observation
        critical = ctx.critical_fields
        if not critical:
            return 1.0
        revealed = ctx.asked_field_keys & critical
        return len(revealed) / len(critical)


class QuestionEfficiencyRubric(Rubric):
    def forward(self, action: Any, observation: Any) -> float:
        del action
        ctx: RubricContext = observation
        if ctx.max_questions <= 0:
            return 1.0
        used = max(0, ctx.questions_asked_count)
        return max(0.0, 1.0 - used / ctx.max_questions)


class HallucinationCheckRubric(Rubric):
    def forward(self, action: Any, observation: Any) -> float:
        del action
        ctx: RubricContext = observation
        plan = ctx.parsed_plan or {}
        scope = set(TASK_FIELDS.get(ctx.family, ())) & set(FIELD_VOCAB)
        plan_fields_in_scope = {k for k in plan if k in scope}
        if not plan_fields_in_scope:
            return 1.0
        hallucinated = plan_fields_in_scope - ctx.asked_field_keys
        return 1.0 - len(hallucinated) / len(plan_fields_in_scope)


def build_rubric() -> Rubric:
    return Sequential(
        Gate(FormatCheckRubric(), threshold=0.5),
        WeightedSum(
            rubrics=[
                FieldMatchRubric(),
                InfoGainRubric(),
                QuestionEfficiencyRubric(),
                HallucinationCheckRubric(),
            ],
            weights=[0.50, 0.20, 0.15, 0.15],
        ),
    )


def score_breakdown(rubric: Rubric) -> dict[str, float]:
    out: dict[str, float] = {}
    for _name, sub in rubric.named_rubrics():
        score = getattr(sub, "last_score", None)
        if score is not None and isinstance(sub, (FormatCheckRubric, FieldMatchRubric, InfoGainRubric, QuestionEfficiencyRubric, HallucinationCheckRubric)):
            out[type(sub).__name__] = float(score)
    return out


__all__ = [
    "RubricContext",
    "FormatCheckRubric",
    "FieldMatchRubric",
    "InfoGainRubric",
    "QuestionEfficiencyRubric",
    "HallucinationCheckRubric",
    "build_rubric",
    "score_breakdown",
]
