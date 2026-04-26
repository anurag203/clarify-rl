#!/usr/bin/env python
"""Hackathon-narrative comparison plots that go beyond `make_plots.py`.

Given the eval JSONs that `refresh_all_plots.sh` downloads from each model
repo, this script renders three artefacts targeted at judges:

1. ``06_same_base_delta.png`` — per-family delta (GRPO − base) for each
   model size, exposing where RL helps vs. hurts at each scale. This is the
   most important hackathon plot: it tells the "scale-dependent training
   response" story directly.

2. ``07_runs_summary_table.png`` — clean text table of every run's
   aggregate score, format pass rate, and per-family numbers. Ships as a
   PNG so it can drop straight into the README.

3. ``runs_summary.json`` — machine-readable version of the same table for
   downstream tooling (the blog post inlines it).

Inputs are auto-discovered from ``outputs/run_artifacts/`` so the script
stays in lock-step with whatever has actually been pushed to the Hub by
``refresh_all_plots.sh``. Anything that isn't there yet (e.g. Run 3 / Run
4 evals while training is still in flight) is just omitted from the
matrix — every plot degrades gracefully.

Why this lives outside ``make_plots.py``: ``make_plots.py`` is the
generic per-eval comparison primitive; ``compare_runs.py`` is the
opinionated, run-aware orchestrator that knows the relationship between
the runs (same model size → "delta", different model sizes → side-by-
side).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RunSpec:
    """One row of the comparison table.

    ``base_label`` cross-references the matching base entry by ``label`` so
    the same-base delta plot can pair them. ``base_label=None`` means this
    row IS itself a base.
    """

    label: str
    eval_path: Path
    base_label: str | None = None
    color: str = "tab:blue"


# Ordered for legend stability in plots and rows in the summary table.
RUN_SPECS: list[RunSpec] = [
    RunSpec(
        label="0.6B base",
        eval_path=Path("outputs/run_artifacts/v4/evals/eval_qwen3-0.6b_n50_v4.json"),
        color="tab:gray",
    ),
    RunSpec(
        label="0.6B GRPO (Run 1)",
        eval_path=Path("outputs/run_artifacts/v4/evals/eval_clarify-rl-grpo-qwen3-0-6b_n50_v4.json"),
        base_label="0.6B base",
        color="tab:blue",
    ),
    RunSpec(
        label="1.7B base",
        eval_path=Path("outputs/run_artifacts/v4/evals/eval_qwen3-1.7b_n50_v4.json"),
        color="dimgray",
    ),
    RunSpec(
        label="1.7B GRPO no-KL (Run 2)",
        eval_path=Path("outputs/run_artifacts/1.7B/evals/eval_clarify-rl-grpo-qwen3-1-7b_n50.json"),
        base_label="1.7B base",
        color="tab:orange",
    ),
    RunSpec(
        label="1.7B GRPO +KL (Run 4)",
        # Auto-resolved from outputs/run_artifacts/1.7B-KL/evals/<latest>.json
        eval_path=Path("outputs/run_artifacts/1.7B-KL/evals"),
        base_label="1.7B base",
        color="tab:green",
    ),
    RunSpec(
        label="1.7B GRPO fixed (Run 6)",
        eval_path=Path("outputs/run_artifacts/1.7B-Run6/evals"),
        base_label="1.7B base",
        color="#0d47a1",
    ),
    RunSpec(
        label="4B base",
        eval_path=Path("outputs/run_artifacts/4B-base/evals"),
        color="darkgray",
    ),
    RunSpec(
        label="4B GRPO (Run 3)",
        eval_path=Path("outputs/run_artifacts/4B/evals"),
        base_label="4B base",
        color="tab:purple",
    ),
    RunSpec(
        label="4B-instruct",
        eval_path=Path("outputs/eval_qwen3-4b-instruct_n50_v4.json"),
        color="tab:red",
    ),
]


def _resolve_eval_path(spec: RunSpec) -> Path | None:
    """If ``spec.eval_path`` is a directory, pick the most-recently-modified
    eval JSON inside it. Otherwise return the file as-is. Missing → None.
    """
    p = spec.eval_path
    if not p.exists():
        return None
    if p.is_file():
        return p
    if p.is_dir():
        candidates = sorted(
            p.glob("eval_*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None
    return None


def _load_summary_and_results(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text())
    return data.get("summary", {}), data.get("results", [])


def _per_family_means(results: list[dict]) -> dict[str, float]:
    """Mean final_score per task family. Treats unknown family as ``"?"``."""
    by_fam: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_fam[r.get("family", "?")].append(float(r.get("final_score", 0.0)))
    return {fam: statistics.mean(scores) if scores else 0.0 for fam, scores in by_fam.items()}


def _per_family_max(results: list[dict]) -> dict[str, float]:
    by_fam: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_fam[r.get("family", "?")].append(float(r.get("final_score", 0.0)))
    return {fam: max(scores) if scores else 0.0 for fam, scores in by_fam.items()}


# ---------------------------------------------------------------------------
# Plot 6 — same-base delta chart
# ---------------------------------------------------------------------------


def _all_families(specs: dict[str, dict]) -> list[str]:
    fams: set[str] = set()
    for entry in specs.values():
        fams.update(entry["family_means"].keys())
    return sorted(fams)


def _delta_panel(ax, specs: dict[str, dict], pairs, families, metric_key: str, ylabel: str, title: str) -> None:
    n_families = len(families)
    n_pairs = len(pairs)
    width = 0.8 / max(1, n_pairs)
    x = list(range(n_families))
    for i, (trained_label, entry) in enumerate(pairs):
        base_entry = specs[entry["base_label"]]
        delta = {
            fam: entry[metric_key].get(fam, 0.0) - base_entry[metric_key].get(fam, 0.0)
            for fam in families
        }
        ax.bar(
            [xi + (i - (n_pairs - 1) / 2) * width for xi in x],
            [delta[fam] for fam in families],
            width=width,
            label=trained_label,
            color=entry["color"],
            edgecolor="black",
            linewidth=0.5,
        )
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")


def plot_same_base_delta(specs: dict[str, dict], out_path: Path) -> None:
    """For every (trained, base) pair, plot Δ = trained − base on TWO panels:
    left = mean per family (avg behaviour), right = max per family (peak
    capability). The right panel exposes the "capability concentration"
    finding: Run 2's mean regressed on meeting_scheduling, but its max went
    *up*, so it learned a narrower-but-stronger solver for that family.
    """
    pairs = [
        (label, entry)
        for label, entry in specs.items()
        if entry["base_label"] and entry["base_label"] in specs
    ]
    if not pairs:
        print("[skip] same-base delta — no trained vs base pairs available yet")
        return

    import matplotlib.pyplot as plt

    families = _all_families(specs)
    fig, (ax_mean, ax_max) = plt.subplots(1, 2, figsize=(max(13, len(families) * 2.0), 5.5), sharey=False)
    _delta_panel(
        ax_mean, specs, pairs, families,
        metric_key="family_means",
        ylabel="Δ avg score (GRPO − same-size base)",
        title="(a) Average behaviour\npositive = GRPO consistently helps, negative = regression",
    )
    _delta_panel(
        ax_max, specs, pairs, families,
        metric_key="family_max",
        ylabel="Δ max score (GRPO − same-size base)",
        title="(b) Peak capability\npositive = GRPO unlocks higher ceiling on at least 1 scenario",
    )

    handles, labels = ax_mean.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(pairs)), fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Where GRPO helps vs. hurts, per task family", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_path}")


# ---------------------------------------------------------------------------
# Plot 7 — summary table as PNG
# ---------------------------------------------------------------------------


def render_summary_table(specs: dict[str, dict], out_path: Path) -> dict:
    """Render the runs_summary as a PNG (suitable for README embed) AND
    return the same data as a dict so the JSON sibling can be written.
    """
    families = _all_families(specs)

    rows: list[dict] = []
    for label, entry in specs.items():
        s = entry["summary"]
        row = {
            "label": label,
            "model": s.get("model", "?"),
            "n": s.get("scenarios_total", "?"),
            "avg_score": float(s.get("avg_score", 0.0)),
            "format_pass_rate": float(s.get("format_pass_rate", 0.0) or 0.0),
            "completion_rate": float(s.get("completion_rate", 0.0) or 0.0),
            **{f"fam_{fam}": entry["family_means"].get(fam, 0.0) for fam in families},
            **{f"max_{fam}": entry["family_max"].get(fam, 0.0) for fam in families},
        }
        rows.append(row)

    summary = {"families": families, "rows": rows}

    # Render text as PNG
    import matplotlib.pyplot as plt
    from matplotlib import patches

    headers = ["Run", "n", "avg", "fmt%"] + [fam for fam in families]
    body: list[list[str]] = []
    for row in rows:
        body.append([
            row["label"],
            str(row["n"]),
            f"{row['avg_score']:.4f}",
            f"{row['format_pass_rate'] * 100:.0f}%",
            *[f"{row[f'fam_{fam}']:.3f}" for fam in families],
        ])

    n_cols = len(headers)
    n_rows = len(body) + 1
    char_widths = [max(len(headers[c]), max((len(b[c]) for b in body), default=0)) for c in range(n_cols)]
    total_chars = sum(char_widths)
    rel_widths = [w / total_chars for w in char_widths]
    fig_width = max(13, sum(char_widths) * 0.13)

    fig, ax = plt.subplots(figsize=(fig_width, 0.6 * n_rows + 0.8))
    ax.axis("off")
    table = ax.table(
        cellText=body,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=rel_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    for c in range(n_cols):
        table[(0, c)].set_facecolor("#cccccc")
        table[(0, c)].set_text_props(weight="bold")

    # highlight winning rows per column
    for c, fam in enumerate(headers):
        if c < 4:
            continue
        col_vals = [row[f"fam_{fam}"] for row in rows]
        if not col_vals:
            continue
        max_v = max(col_vals)
        if max_v <= 0:
            continue
        for r, row in enumerate(rows, start=1):
            if abs(row[f"fam_{fam}"] - max_v) < 1e-9:
                table[(r, c)].set_facecolor("#cdeac0")  # green
                table[(r, c)].set_text_props(weight="bold")

    ax.set_title("ClarifyRL — per-run × per-family scoreboard (n=50, eval v4)\nGreen cell = best score in that family", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_path}")

    return summary


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="plots", help="Directory for PNGs + runs_summary.json")
    args = p.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs: dict[str, dict] = {}
    for spec in RUN_SPECS:
        path = _resolve_eval_path(spec)
        if path is None:
            print(f"[skip] {spec.label}: no eval JSON yet at {spec.eval_path}")
            continue
        summary, results = _load_summary_and_results(path)
        specs[spec.label] = {
            "summary": summary,
            "family_means": _per_family_means(results),
            "family_max": _per_family_max(results),
            "base_label": spec.base_label,
            "color": spec.color,
            "eval_path": str(path),
        }

    if not specs:
        print("[err] no eval JSONs found at all — nothing to plot")
        return

    print(f"\n[load] {len(specs)} eval JSON(s):")
    for lbl, entry in specs.items():
        print(f"  - {lbl}: {entry['eval_path']}")
    print()

    plot_same_base_delta(specs, out_dir / "06_same_base_delta.png")
    summary = render_summary_table(specs, out_dir / "07_runs_summary_table.png")

    json_path = out_dir / "runs_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[ok] {json_path}")
    print()
    print(f"All comparison artifacts written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
