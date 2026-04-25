"""
Rule-based user simulator for ClarifyRL.

Given a free-text clarifying question + the hidden profile + the task family,
return a short natural-language answer and the profile field that was revealed
(or None if the question didn't match any field the user knows).

Pure-Python, deterministic, sub-millisecond. No LLM call.
"""

from __future__ import annotations

from typing import Any, Optional


FIELD_KEYWORDS: dict[str, list[str]] = {
    "stack": ["stack", "language", "framework", "tech", "what to build it in"],
    "scale": ["scale", "users", "traffic", "load", "rps", "concurrent"],
    "auth": ["auth", "authentication", "login", "sso", "jwt", "oauth"],
    "datastore": ["database", "db", "storage", "persist", "data store"],
    "deployment_target": ["deploy", "host", "hosting", "cloud", "aws", "kubernetes", "where to run"],
    "language_version": ["version", "python version", "node version", "runtime version"],
    "test_coverage": ["test", "coverage", "testing", "qa"],

    "primary_symptom": ["symptom", "what's wrong", "what hurts", "how do you feel", "what is the issue"],
    "duration": ["how long", "since when", "duration", "when did", "started"],
    "severity": ["severe", "mild", "intense", "how bad", "severity", "scale of pain"],
    "age_band": ["age", "how old", "young", "elderly", "child", "adult"],
    "prior_conditions": ["history", "prior condition", "medical history", "pre-existing", "chronic"],
    "medications": ["medication", "meds", "drugs", "prescription", "taking anything"],

    "order_id": ["order id", "order number", "order #", "reference", "tracking", "which order"],
    "item_issue": ["what's wrong with", "what happened", "damaged", "missing", "wrong", "issue with the order", "problem with"],
    "refund_or_replace": ["refund", "replace", "return", "credit", "what would you like", "resolution"],
    "urgency": ["when do you need", "need by", "urgent", "asap", "how soon", "urgency"],
    "channel_preferred": ["contact", "reach you", "email or phone", "how should we", "channel"],

    "participants": ["who", "participants", "attend", "join", "invite", "attendees"],
    "date": ["what day", "which day", "date", "when (day)", "what date"],
    "time": ["what time", "which time", "hour", "morning or afternoon"],
    "duration_minutes": ["how long", "duration", "minutes", "length"],
    "platform": ["zoom", "platform", "in person", "in-person", "where (online)", "virtual or"],

    "event_type": ["what kind of event", "kind", "type of event", "occasion"],
    "guest_count": ["how many", "guest", "headcount", "size", "people"],
    "venue": ["where", "venue", "location", "place"],
    "budget_band": ["budget", "cost", "spend", "price", "how much"],
    "theme": ["theme", "vibe", "style", "formal or casual"],
    "dietary_constraints": ["diet", "vegetarian", "vegan", "food restriction", "allergies", "dietary"],
}


_FIELD_PHRASING: dict[str, str] = {
    "stack": "I'd like to use {value}",
    "scale": "Expecting around {value}",
    "auth": "Auth should be {value}",
    "datastore": "Use {value}",
    "deployment_target": "Deploy to {value}",
    "language_version": "Use {value}",
    "test_coverage": "{value} tests",

    "primary_symptom": "It's a {value}",
    "duration": "About {value}",
    "severity": "I'd say {value}",
    "age_band": "I'm a {value}",
    "prior_conditions": "{value}",
    "medications": "{value}",

    "order_id": "Order {value}",
    "item_issue": "{value}",
    "refund_or_replace": "I'd prefer a {value}",
    "urgency": "Urgency is {value}",
    "channel_preferred": "Please reach me by {value}",

    "participants": "{value}",
    "date": "{value}",
    "time": "{value}",
    "duration_minutes": "{value} minutes",
    "platform": "{value}",

    "event_type": "A {value}",
    "guest_count": "About {value} people",
    "venue": "At a {value}",
    "budget_band": "Budget around {value}",
    "theme": "{value}",
    "dietary_constraints": "{value}",
}


_NO_MATCH_REPLIES: dict[str, str] = {
    "coding_requirements": "I don't have a strong preference on that — pick something reasonable.",
    "medical_intake": "I'm not sure about that, sorry.",
    "support_triage": "I don't really know — does it matter?",
    "meeting_scheduling": "No preference, you choose.",
    "event_planning": "Up to you on that one.",
}


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def match_field(question: str, allowed_keys: list[str]) -> Optional[str]:
    q = _normalize(question)
    best_score = -1
    best_field: Optional[str] = None
    for field_key in allowed_keys:
        for kw in FIELD_KEYWORDS.get(field_key, ()):
            if kw in q and len(kw) > best_score:
                best_score = len(kw)
                best_field = field_key
    return best_field


def format_answer(field_key: str, value: Any, family: str) -> str:
    del family
    phrasing = _FIELD_PHRASING.get(field_key, "{value}")
    text = phrasing.format(value=value).strip()
    if not text.endswith((".", "!", "?")):
        text += "."
    return text


def answer(
    question: str,
    hidden_profile: dict[str, Any],
    family: str,
) -> tuple[str, Optional[str]]:
    profile_keys = list(hidden_profile.keys())
    matched = match_field(question, profile_keys)
    if matched is None:
        return _NO_MATCH_REPLIES.get(family, "I don't know."), None
    return format_answer(matched, hidden_profile[matched], family), matched


__all__ = [
    "FIELD_KEYWORDS",
    "match_field",
    "format_answer",
    "answer",
]
