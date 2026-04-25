"""
Procedural scenario generator for ClarifyRL.

A scenario is the ground-truth of one episode: a deliberately-vague request,
a hidden user profile (the answer key), the subset of profile fields that
are critical for the task, and the per-task budget.

Pure-Python, deterministic given (seed, task_id), no external deps.
"""

from __future__ import annotations

import random
from typing import Any, TypedDict


FAMILIES: tuple[str, ...] = (
    "coding_requirements",
    "medical_intake",
    "support_triage",
    "meeting_scheduling",
    "event_planning",
)


DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")


FIELD_VOCAB: dict[str, list[Any]] = {
    "stack": ["python+fastapi", "node+express", "go+gin", "rust+axum", "ruby+rails"],
    "scale": ["<100 users", "1k users", "10k users", "1M users"],
    "auth": ["none", "api-key", "jwt", "oauth2", "saml-sso"],
    "datastore": ["postgres", "mysql", "mongodb", "sqlite", "redis-only"],
    "deployment_target": ["aws-lambda", "kubernetes", "fly.io", "heroku", "on-prem"],
    "language_version": ["python3.11", "python3.12", "node18", "node20"],
    "test_coverage": ["none", "unit-only", "unit+integration", "full-e2e"],

    "primary_symptom": ["headache", "fever", "cough", "fatigue", "nausea", "rash", "pain"],
    "duration": ["<1 hour", "1-24 hours", "1-7 days", "1-4 weeks", "chronic"],
    "severity": ["mild", "moderate", "severe"],
    "age_band": ["child", "teen", "adult", "senior"],
    "prior_conditions": ["none", "diabetes", "hypertension", "asthma", "other"],
    "medications": ["none", "prescription", "otc", "both"],

    "order_id": ["#4521", "#7830", "#1199", "#9027", "none-provided"],
    "item_issue": ["wrong-item", "damaged", "missing", "late", "never-arrived"],
    "refund_or_replace": ["refund", "replace", "store-credit", "unsure"],
    "urgency": ["low", "medium", "high"],
    "channel_preferred": ["email", "phone", "chat"],

    "participants": ["whole team", "just me and X", "leadership", "external client"],
    "date": ["today", "tomorrow", "this week", "next week"],
    "time": ["morning", "afternoon", "evening", "flexible"],
    "duration_minutes": [15, 30, 45, 60, 90],
    "platform": ["zoom", "google-meet", "in-person", "phone"],

    "event_type": ["birthday", "farewell", "anniversary", "team-building", "baby-shower"],
    "guest_count": [5, 10, 20, 50, 100],
    "venue": ["restaurant", "home", "office", "outdoor", "rented hall"],
    "budget_band": ["<$100", "$100-500", "$500-2000", "$2000+"],
    "theme": ["casual", "formal", "themed", "surprise"],
    "dietary_constraints": ["none", "vegetarian", "vegan", "mixed"],
}


TASK_FIELDS: dict[str, list[str]] = {
    "coding_requirements": [
        "stack", "scale", "auth", "datastore",
        "deployment_target", "language_version", "test_coverage",
    ],
    "medical_intake": [
        "primary_symptom", "duration", "severity",
        "age_band", "prior_conditions", "medications",
    ],
    "support_triage": [
        "order_id", "item_issue", "refund_or_replace",
        "urgency", "channel_preferred",
    ],
    "meeting_scheduling": [
        "participants", "date", "time", "duration_minutes", "platform",
    ],
    "event_planning": [
        "event_type", "guest_count", "venue", "date",
        "budget_band", "theme", "dietary_constraints",
    ],
}


CRITICAL_BY_FAMILY: dict[str, list[str]] = {
    "coding_requirements": ["stack", "scale", "auth", "datastore", "deployment_target"],
    "medical_intake": ["primary_symptom", "duration", "severity", "age_band"],
    "support_triage": ["order_id", "item_issue", "refund_or_replace", "urgency"],
    "meeting_scheduling": ["participants", "date", "time", "duration_minutes"],
    "event_planning": ["event_type", "guest_count", "date", "venue"],
}


REQUIRED_KEYS_BY_FAMILY: dict[str, list[str]] = {
    "coding_requirements": ["stack", "scale", "auth", "datastore"],
    "medical_intake": ["primary_symptom", "duration", "severity"],
    "support_triage": ["order_id", "item_issue", "refund_or_replace"],
    "meeting_scheduling": ["participants", "date", "time"],
    "event_planning": ["event_type", "date", "guest_count", "venue"],
}


REQUEST_TEMPLATES: dict[str, list[str]] = {
    "coding_requirements": [
        "Build me an API.",
        "Write a service for me.",
        "I need a backend.",
        "Set up an endpoint.",
        "Make me a web service.",
    ],
    "medical_intake": [
        "I'm not feeling well.",
        "Something's off with my health.",
        "I have a problem.",
        "I need some help.",
    ],
    "support_triage": [
        "My order is wrong.",
        "There's an issue with my purchase.",
        "Something went wrong.",
        "I need help with an order.",
    ],
    "meeting_scheduling": [
        "Schedule a sync.",
        "Set up a meeting.",
        "Get a call on the calendar.",
    ],
    "event_planning": [
        "Plan a birthday party.",
        "Organize a team event.",
        "Set up a celebration.",
    ],
}


N_FIELDS_RANGE_BY_DIFFICULTY: dict[str, tuple[int, int]] = {
    "easy": (2, 3),
    "medium": (4, 5),
    "hard": (6, 7),
}


MIN_CRITICAL_BY_DIFFICULTY: dict[str, int] = {
    "easy": 2,
    "medium": 3,
    "hard": 4,
}


MAX_STEPS_BY_DIFFICULTY: dict[str, int] = {
    "easy": 8,
    "medium": 10,
    "hard": 12,
}


TASK_TITLE_BY_DIFFICULTY: dict[str, str] = {
    "easy": "Mild Ambiguity",
    "medium": "Moderate Ambiguity",
    "hard": "High Ambiguity",
}


MAX_QUESTIONS: int = 6


class Scenario(TypedDict):
    family: str
    task_id: str
    task_title: str
    request: str
    hidden_profile: dict[str, Any]
    critical_fields: list[str]
    required_keys: list[str]
    field_keys: list[str]
    max_steps: int
    max_questions: int


def generate(seed: int, task_id: str = "medium") -> Scenario:
    if task_id not in N_FIELDS_RANGE_BY_DIFFICULTY:
        raise ValueError(
            f"task_id must be one of {DIFFICULTIES!r}, got {task_id!r}"
        )

    rng = random.Random(seed)

    family = rng.choice(FAMILIES)
    valid_fields = TASK_FIELDS[family]
    critical_pool = CRITICAL_BY_FAMILY[family]
    required_keys = REQUIRED_KEYS_BY_FAMILY[family]

    n_lo, n_hi = N_FIELDS_RANGE_BY_DIFFICULTY[task_id]
    target_n = rng.randint(n_lo, n_hi)
    target_n = min(target_n, len(valid_fields))

    if task_id == "easy":
        pool = critical_pool
        n = min(target_n, len(pool))
        chosen_fields = rng.sample(pool, k=n)
    else:
        chosen_fields = list(required_keys)
        remaining = [f for f in valid_fields if f not in chosen_fields]
        extra_n = max(0, target_n - len(chosen_fields))
        if extra_n > 0 and remaining:
            chosen_fields += rng.sample(remaining, k=min(extra_n, len(remaining)))

    profile: dict[str, Any] = {f: rng.choice(FIELD_VOCAB[f]) for f in chosen_fields}

    min_critical = min(
        MIN_CRITICAL_BY_DIFFICULTY[task_id],
        len(critical_pool),
        target_n,
    )
    critical_present = set(chosen_fields) & set(critical_pool)
    if len(critical_present) < min_critical:
        deficit = min_critical - len(critical_present)
        candidates = [f for f in critical_pool if f not in critical_present]
        extras = rng.sample(candidates, k=min(deficit, len(candidates)))
        for f in extras:
            if f not in profile:
                profile[f] = rng.choice(FIELD_VOCAB[f])
                chosen_fields.append(f)
            critical_present.add(f)

    template = rng.choice(REQUEST_TEMPLATES[family])

    return Scenario(
        family=family,
        task_id=task_id,
        task_title=TASK_TITLE_BY_DIFFICULTY[task_id],
        request=template,
        hidden_profile=profile,
        critical_fields=sorted(critical_present),
        required_keys=list(required_keys),
        field_keys=sorted(profile.keys()),
        max_steps=MAX_STEPS_BY_DIFFICULTY[task_id],
        max_questions=MAX_QUESTIONS,
    )


__all__ = [
    "FAMILIES",
    "DIFFICULTIES",
    "FIELD_VOCAB",
    "TASK_FIELDS",
    "CRITICAL_BY_FAMILY",
    "REQUIRED_KEYS_BY_FAMILY",
    "REQUEST_TEMPLATES",
    "N_FIELDS_RANGE_BY_DIFFICULTY",
    "MIN_CRITICAL_BY_DIFFICULTY",
    "MAX_STEPS_BY_DIFFICULTY",
    "TASK_TITLE_BY_DIFFICULTY",
    "MAX_QUESTIONS",
    "Scenario",
    "generate",
]
