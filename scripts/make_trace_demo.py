#!/usr/bin/env python
"""Generate a side-by-side untrained-vs-trained trace demo for the blog/README.

Reads two eval JSONs produced by `run_eval.py`, picks N scenarios where the
trained model dramatically outscored the base (or where the trained model
asked-and-the-base-hallucinated), and emits a markdown file with a clean
two-column comparison + rubric breakdown.

Usage:
    python scripts/make_trace_demo.py \\
        --base outputs/eval_qwen3-1.7b_base.json \\
        --trained outputs/eval_qwen3-1.7b_trained.json \\
        --out docs/trace_demo.md \\
        --n 3
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
    for k in ("FieldMatch", "InfoGain", "QuestionEfficiency", "HallucinationCheck"):
        v = bd.get(k, bd.get(k.lower(), 0.0))
        try:
            parts.append(f"{k}={float(v):.2f}")
        except (TypeError, ValueError):
            parts.append(f"{k}=?")
    return " · ".join(parts)


def _render_messages(messages: list[dict[str, Any]] | None, max_chars: int = 400) -> str:
    if not messages:
        return "_(no messages captured)_"
    lines: list[str] = []
    for m in messages[:30]:
        role = m.get("role", "?")
        content = (m.get("content") or "").strip()
        if len(content) > max_chars:
            content = content[: max_chars - 1] + "…"
        if not content:
            tool = m.get("tool_calls") or []
            if tool:
                names = ", ".join(t.get("name", "?") for t in tool)
                content = f"_[tool: {names}]_"
            else:
                continue
        lines.append(f"**{role}**: {content}")
    return "\n\n".join(lines)


def _pick_demo_scenarios(
    base: dict[str, dict[str, Any]],
    trained: dict[str, dict[str, Any]],
    n: int,
) -> list[str]:
    common = set(base) & set(trained)
    diffs: list[tuple[str, float, dict[str, Any], dict[str, Any]]] = []
    for sid in common:
        b, t = base[sid], trained[sid]
        b_score = float(b.get("final_score", 0.0))
        t_score = float(t.get("final_score", 0.0))
        delta = t_score - b_score
        diffs.append((sid, delta, b, t))

    diffs.sort(key=lambda x: -x[1])
    seen_families: set[str] = set()
    picks: list[str] = []
    for sid, delta, b, _t in diffs:
        if delta <= 0.05:
            break
        fam = b.get("family", "?")
        if fam in seen_families:
            continue
        seen_families.add(fam)
        picks.append(sid)
        if len(picks) >= n:
            break

    if len(picks) < n:
        for sid, delta, _b, _t in diffs:
            if sid in picks:
                continue
            if delta > 0:
                picks.append(sid)
            if len(picks) >= n:
                break
    return picks


def _emit(
    out_path: Path,
    base: dict[str, Any],
    trained: dict[str, Any],
    picks: list[str],
) -> None:
    base_idx = _index(base)
    trained_idx = _index(trained)

    base_label = base.get("label", "untrained")
    trained_label = trained.get("label", "trained")

    parts: list[str] = []
    parts.append(f"# Two-trace demo — {base_label} vs {trained_label}")
    parts.append("")
    parts.append(f"_{len(picks)} scenarios where the trained model substantially outperformed the base._")
    parts.append("")
    parts.append("Each row shows: the ambiguous request → the agent's full message trace → final rubric breakdown.")
    parts.append("")

    for i, sid in enumerate(picks, 1):
        b = base_idx[sid]
        t = trained_idx[sid]

        family = b.get("family", "?")
        difficulty = b.get("difficulty", "?")
        request = (b.get("request") or t.get("request") or "_(no request captured)_").strip()
        if len(request) > 240:
            request = request[:240] + "…"

        parts.append(f"## {i}. `{sid}` — `{family}` (`{difficulty}`)")
        parts.append("")
        parts.append(f"**Request**: {request}")
        parts.append("")
        parts.append(
            "| Run | Score | Q's asked | Format pass | Rubric breakdown |"
        )
        parts.append("|-----|-------|-----------|-------------|------------------|")
        parts.append(
            f"| {base_label} | **{float(b.get('final_score', 0.0)):.2f}** | "
            f"{b.get('questions_asked', 0)} | "
            f"{'✓' if b.get('format_pass') else '✗'} | "
            f"{_format_breakdown(b.get('score_breakdown'))} |"
        )
        parts.append(
            f"| {trained_label} | **{float(t.get('final_score', 0.0)):.2f}** | "
            f"{t.get('questions_asked', 0)} | "
            f"{'✓' if t.get('format_pass') else '✗'} | "
            f"{_format_breakdown(t.get('score_breakdown'))} |"
        )
        parts.append("")
        parts.append(f"**{base_label} trace:**")
        parts.append("")
        parts.append(_render_messages(b.get("messages")))
        parts.append("")
        parts.append(f"**{trained_label} trace:**")
        parts.append("")
        parts.append(_render_messages(t.get("messages")))
        parts.append("")
        parts.append("---")
        parts.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))
    print(f"[ok] wrote {out_path} with {len(picks)} demo scenarios")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="Path to base eval JSON")
    parser.add_argument("--trained", required=True, help="Path to trained eval JSON")
    parser.add_argument("--out", default="docs/trace_demo.md", help="Output markdown path")
    parser.add_argument("--n", type=int, default=3, help="Number of demo scenarios to include")
    args = parser.parse_args()

    base = _load(args.base)
    trained = _load(args.trained)
    base_idx = _index(base)
    trained_idx = _index(trained)
    picks = _pick_demo_scenarios(base_idx, trained_idx, args.n)
    if not picks:
        print("[warn] No scenarios where trained > base by >0.05 — nothing to demo")
        return
    _emit(Path(args.out), base, trained, picks)


if __name__ == "__main__":
    main()
