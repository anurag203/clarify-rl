#!/usr/bin/env python
"""Run-1 specific trace demo: trained 0.6B (300 steps) vs Qwen3-4B-Instruct.

The trained model doesn't beat the larger instruct baseline on `final_score`
(both struggle to clear FormatCheck on held-out), but it DOES win clearly on
question efficiency, runtime, and avoidance of `<think>` token-waste. This
script picks a few comparable scenarios that make those wins legible.

Picks:
  - A scenario where 4B-Instruct hits the 6-question cap but the trained
    model hands back a plan in <=2 steps (efficiency win).
  - A scenario where 4B-Instruct succeeded (score > 0) — shows the gap that
    more training will close.
  - A scenario where the trained model attempted a plan but with the wrong
    schema — illustrates the failure mode for the writeup.

Usage:
    python scripts/make_trace_demo_run1.py \
        --base outputs/eval_qwen3-4b-instruct_n30.json \
        --trained outputs/run1_artifacts/eval_clarify-rl-grpo-qwen3-0-6b_n50_thinkfix.json \
        --out docs/trace_demo_run1.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _index(eval_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["scenario_id"]: r for r in eval_dict.get("results", [])}


def _format_breakdown(bd: dict[str, float] | None) -> str:
    if not bd:
        return "_no breakdown_"
    parts = []
    for k in ("FormatCheckRubric", "FieldMatchRubric", "InfoGainRubric", "QuestionEfficiencyRubric", "HallucinationCheckRubric"):
        v = bd.get(k)
        if v is None:
            continue
        try:
            parts.append(f"{k.replace('Rubric','')}={float(v):.2f}")
        except (TypeError, ValueError):
            parts.append(f"{k}=?")
    return " · ".join(parts) if parts else "_no breakdown_"


def _render_messages(messages: list[dict[str, Any]] | None, max_chars: int = 350) -> str:
    if not messages:
        return "_(no messages captured)_"
    lines: list[str] = []
    for m in messages[:30]:
        role = m.get("role", "?")
        content = (m.get("content") or "").strip()
        if len(content) > max_chars:
            content = content[: max_chars - 1] + "…"
        if not content:
            continue
        lines.append(f"**{role}**: {content}")
    return "\n\n".join(lines)


def _pick_scenarios(
    base_idx: dict[str, dict[str, Any]],
    trained_idx: dict[str, dict[str, Any]],
) -> list[tuple[str, str]]:
    """Return list of (scenario_id, label) tuples — picks 3 illustrative cases."""
    common = sorted(set(base_idx) & set(trained_idx))
    picks: list[tuple[str, str]] = []

    base_winners = sorted(
        [sid for sid in common if (base_idx[sid].get("final_score") or 0) > 0],
        key=lambda s: -(base_idx[s].get("final_score") or 0),
    )
    if base_winners:
        picks.append((base_winners[0], "Untrained 4B WIN — gap that more training closes"))

    eff = sorted(
        common,
        key=lambda s: (
            (trained_idx[s].get("questions_asked") or 0)
            - (base_idx[s].get("questions_asked") or 0)
        ),
    )
    for sid in eff:
        if sid in {p[0] for p in picks}:
            continue
        delta = (base_idx[sid].get("questions_asked") or 0) - (trained_idx[sid].get("questions_asked") or 0)
        if delta >= 3:
            picks.append((sid, "Question efficiency win — trained asks far fewer"))
            break

    for sid in common:
        if sid in {p[0] for p in picks}:
            continue
        b = base_idx[sid]
        t = trained_idx[sid]
        if (t.get("questions_asked") or 0) >= 1 and (t.get("final_score") or 0) == 0:
            picks.append((sid, "Failure mode — trained submits wrong-schema plan"))
            break

    if len(picks) < 3 and common:
        for sid in common:
            if sid not in {p[0] for p in picks}:
                picks.append((sid, "Comparable scenario"))
            if len(picks) >= 3:
                break
    return picks[:3]


def _emit(out_path: Path, base: dict, trained: dict, picks: list[tuple[str, str]]) -> None:
    base_idx = _index(base)
    trained_idx = _index(trained)

    base_label = "Qwen3-4B-Instruct (untrained)"
    trained_label = "Qwen3-0.6B GRPO (300 steps, run-1)"

    parts: list[str] = []
    parts.append(f"# Run-1 trace demo — {trained_label} vs {base_label}")
    parts.append("")
    parts.append("Three illustrative scenarios from the held-out eval set. Both models fail FormatCheck most of the time, but the trained 0.6B already shows clear behavioural shifts: fewer questions, no `<think>` token-waste, faster turnaround.")
    parts.append("")
    parts.append(f"- Base: `{base.get('summary', {}).get('model', 'Qwen/Qwen3-4B-Instruct-2507')}` (n={base.get('summary', {}).get('scenarios_total', '?')})")
    parts.append(f"- Trained: `{trained.get('summary', {}).get('model', 'agarwalanu3103/clarify-rl-grpo-qwen3-0-6b')}` (n={trained.get('summary', {}).get('scenarios_total', '?')})")
    parts.append(f"- Avg questions: base **{base.get('summary', {}).get('avg_questions', '?'):.2f}**, trained **{trained.get('summary', {}).get('avg_questions', '?'):.2f}**")
    parts.append(f"- Eval runtime: base **{base.get('summary', {}).get('elapsed_s', 0):.0f}s** / 30 scenarios, trained **{trained.get('summary', {}).get('elapsed_s', 0):.0f}s** / 50 scenarios")
    parts.append("")
    parts.append("---")
    parts.append("")

    for i, (sid, label) in enumerate(picks, 1):
        b = base_idx[sid]
        t = trained_idx[sid]

        family = b.get("family", "?")
        difficulty = b.get("task_id", b.get("difficulty", "?"))
        request_full = (b.get("request") or t.get("request") or "")
        request = request_full.split("USER REQUEST:")[-1].strip()
        request = request.split("You have")[0].strip()
        if len(request) > 200:
            request = request[:200] + "…"

        parts.append(f"## {i}. {label}")
        parts.append("")
        parts.append(f"**Scenario:** `{sid}` — family=`{family}`, task=`{difficulty}`")
        parts.append("")
        parts.append(f"**Request:** {request}")
        parts.append("")
        parts.append("| Run | Score | Q's asked | Runtime | Rubric breakdown |")
        parts.append("|-----|-------|-----------|---------|------------------|")
        parts.append(
            f"| {base_label} | **{float(b.get('final_score', 0.0)):.2f}** | "
            f"{b.get('questions_asked', 0)} | "
            f"{b.get('elapsed_s', 0):.1f}s | "
            f"{_format_breakdown(b.get('score_breakdown'))} |"
        )
        parts.append(
            f"| {trained_label} | **{float(t.get('final_score', 0.0)):.2f}** | "
            f"{t.get('questions_asked', 0)} | "
            f"{t.get('elapsed_s', 0):.1f}s | "
            f"{_format_breakdown(t.get('score_breakdown'))} |"
        )
        parts.append("")
        parts.append(f"<details><summary><b>{base_label} trace</b></summary>")
        parts.append("")
        parts.append(_render_messages(b.get("messages")))
        parts.append("")
        parts.append("</details>")
        parts.append("")
        parts.append(f"<details><summary><b>{trained_label} trace</b></summary>")
        parts.append("")
        parts.append(_render_messages(t.get("messages")))
        parts.append("")
        parts.append("</details>")
        parts.append("")
        parts.append("---")
        parts.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))
    print(f"[ok] wrote {out_path}  ({len(picks)} scenarios, {out_path.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--trained", required=True)
    parser.add_argument("--out", default="docs/trace_demo_run1.md")
    args = parser.parse_args()

    base = _load(args.base)
    trained = _load(args.trained)
    picks = _pick_scenarios(_index(base), _index(trained))
    if not picks:
        print("[warn] No comparable scenarios found")
        return
    _emit(Path(args.out), base, trained, picks)


if __name__ == "__main__":
    main()
