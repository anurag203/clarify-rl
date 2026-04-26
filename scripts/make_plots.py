#!/usr/bin/env python
"""Generate the 5 submission plots from training logs + eval JSONs.

Inputs (all optional — script emits whatever it can):
  --log-history       <output_dir>/log_history.json (from train_grpo.py)
  --eval-policy       outputs/eval_policy.json
  --eval-base         outputs/eval_<model>_base.json
  --eval-trained      outputs/eval_<model>_trained.json
  --out-dir           plots/  (default)

Output PNGs in --out-dir:
  01_reward_loss_curves.png   reward + loss vs training step
  02_per_family_bars.png      avg score per task family (policy / base / trained)
  03_component_breakdown.png  per-rubric-component bars (FieldMatch / InfoGain / ...)
  04_before_after.png         scatter + grouped bar of policy vs base vs trained
  05_question_efficiency.png  hist of questions asked per scenario (policy / base / trained)

If multiple sources are present, each plot overlays the comparable series.
If a source is missing, the plot quietly omits that series and continues.

Usage:
  python scripts/make_plots.py \\
      --log-history clarify-rl-grpo-qwen3-0.6b/log_history.json \\
      --eval-policy outputs/eval_policy.json \\
      --eval-base outputs/eval_qwen3-0.6b_base.json \\
      --eval-trained outputs/eval_qwen3-0.6b_trained.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

# Lazy-import matplotlib only at runtime — keeps `--help` snappy and avoids
# import errors when only a subset of inputs is present.


def _load_json(path: str | None) -> Any:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[skip] {path} not found")
        return None
    return json.loads(p.read_text())


def _safe_mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


# Stable label -> color map. Same label = same color in every plot, no
# collisions across the 8 series we render in the submission deck. Values
# are picked from matplotlib's tab20 + a couple of overrides to keep the
# headline trios (Run 1 / Run 2 / Run 4) clearly distinct.
_LABEL_COLORS: dict[str, str] = {
    "policy (deterministic)":   "#9e9e9e",   # neutral grey
    "policy (baseline)":        "#9e9e9e",
    "0.6B base":                "#ffb74d",   # warm orange
    "0.6B GRPO (Run 1)":        "#1f77b4",   # strong blue
    "1.7B base":                "#66bb6a",   # mid green
    "1.7B GRPO no-KL (Run 2)":  "#e53935",   # red — the regression run
    "1.7B GRPO +KL (Run 4)":    "#2e7d32",   # deep green — KL-anchored hero
    "4B base":                  "#5e35b1",   # purple — ceiling marker
    "4B-instruct":              "#00838f",   # teal
    "4B GRPO (Run 3)":          "#ff6f00",   # amber
    "untrained":                "#ffb74d",
    "trained":                  "#1f77b4",
}

# Fallback palette when a label isn't pre-mapped. tab20-style colors
# already chosen to contrast with the named ones above.
_FALLBACK_COLORS: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _color_for(label: str, fallback_index: int = 0) -> str:
    """Return a stable, distinct color for `label`."""
    if label in _LABEL_COLORS:
        return _LABEL_COLORS[label]
    return _FALLBACK_COLORS[fallback_index % len(_FALLBACK_COLORS)]


def _autoscale_top(values: list[float], floor: float = 0.10, headroom: float = 1.20) -> float:
    """Pick a y-axis top so the data fills the visible range.

    `floor` is the minimum top we want even when values are tiny (so a single
    very low bar doesn't end up at 100% of the axis and look misleading).
    """
    if not values:
        return 1.0
    top = max(values) * headroom
    return max(top, floor)


# ---------------------------------------------------------------------------
# Plot 1 — Reward + Loss curves
# ---------------------------------------------------------------------------


def plot_reward_loss_curves(log_history: list[dict], out_path: Path) -> None:
    if not log_history:
        print("[skip] reward/loss curves — no log_history")
        return

    import matplotlib.pyplot as plt

    steps_loss: list[int] = []
    losses: list[float] = []
    steps_rew: list[int] = []
    rewards: list[float] = []

    for row in log_history:
        step = row.get("step")
        if step is None:
            continue
        if "loss" in row and isinstance(row["loss"], (int, float)):
            steps_loss.append(step)
            losses.append(float(row["loss"]))
        if "reward" in row and isinstance(row["reward"], (int, float)):
            steps_rew.append(step)
            rewards.append(float(row["reward"]))
        elif "rewards/reward_func/mean" in row:
            steps_rew.append(step)
            rewards.append(float(row["rewards/reward_func/mean"]))

    if not steps_loss and not steps_rew:
        print("[skip] reward/loss curves — log_history has no loss or reward")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if steps_loss:
        axes[0].plot(steps_loss, losses, color="tab:red", lw=1.5)
        axes[0].set_title("Training loss (advantage-weighted)")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.3)

    if steps_rew:
        axes[1].plot(steps_rew, rewards, color="tab:blue", lw=1.5)
        # Rolling mean for trend
        window = max(1, len(rewards) // 20)
        if window >= 3 and len(rewards) >= window:
            roll = [
                _safe_mean(rewards[max(0, i - window): i + 1])
                for i in range(len(rewards))
            ]
            axes[1].plot(steps_rew, roll, color="tab:blue", lw=2.5, alpha=0.4, label=f"rolling-mean ({window})")
            axes[1].legend()
        axes[1].set_title("Reward per training step")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Reward (rubric score)")
        axes[1].grid(alpha=0.3)

    fig.suptitle("ClarifyRL GRPO training")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[ok] {out_path}")


def plot_reward_loss_curves_multi(runs: dict[str, list[dict]], out_path: Path) -> None:
    """Overlay multiple training runs on the same reward/loss axes."""
    runs = {label: hist for label, hist in runs.items() if hist}
    if not runs:
        print("[skip] reward/loss curves \u2014 no runs")
        return

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]

    for i, (label, hist) in enumerate(runs.items()):
        steps_loss: list[int] = []
        losses: list[float] = []
        steps_rew: list[int] = []
        rewards: list[float] = []
        for row in hist:
            step = row.get("step")
            if step is None:
                continue
            if "loss" in row and isinstance(row["loss"], (int, float)):
                steps_loss.append(step)
                losses.append(float(row["loss"]))
            if "reward" in row and isinstance(row["reward"], (int, float)):
                steps_rew.append(step)
                rewards.append(float(row["reward"]))
            elif "rewards/reward_func/mean" in row:
                steps_rew.append(step)
                rewards.append(float(row["rewards/reward_func/mean"]))

        color = palette[i % len(palette)]
        if steps_loss:
            axes[0].plot(steps_loss, losses, color=color, lw=1.5, label=label, alpha=0.8)
        if steps_rew:
            axes[1].plot(steps_rew, rewards, color=color, lw=1.5, alpha=0.55)
            window = max(3, len(rewards) // 20)
            if len(rewards) >= window:
                roll = [
                    _safe_mean(rewards[max(0, j - window): j + 1])
                    for j in range(len(rewards))
                ]
                axes[1].plot(steps_rew, roll, color=color, lw=2.5, label=f"{label} (rolling-{window})")

    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Reward per training step")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Reward (rubric mean)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("ClarifyRL GRPO training curves")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[ok] {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — Per-family bars
# ---------------------------------------------------------------------------


def plot_per_family_bars(evals: dict[str, dict | None], out_path: Path) -> None:
    series = {label: ev for label, ev in evals.items() if ev}
    if not series:
        print("[skip] per-family — no eval JSONs")
        return

    import matplotlib.pyplot as plt

    families = sorted({
        r.get("family", "?")
        for ev in series.values()
        for r in ev.get("results", [])
    })

    matrix: dict[str, dict[str, float]] = {}
    for label, ev in series.items():
        per_family = defaultdict(list)
        for r in ev.get("results", []):
            per_family[r.get("family", "?")].append(r.get("final_score", 0.0))
        matrix[label] = {fam: _safe_mean(per_family.get(fam, [])) for fam in families}

    n_groups = len(families)
    n_series = len(series)
    width = 0.8 / max(1, n_series)
    x = list(range(n_groups))

    fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.4), 5))
    all_values: list[float] = []
    for i, (label, scores) in enumerate(matrix.items()):
        vals = [scores[fam] for fam in families]
        all_values.extend(vals)
        ax.bar(
            [xi + (i - (n_series - 1) / 2) * width for xi in x],
            vals,
            width=width,
            label=label,
            color=_color_for(label, i),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=15, ha="right")
    ax.set_ylabel("Avg final score (n=50, eval v4)")
    ax.set_title("Avg score per task family — base vs trained, all model sizes")
    ax.set_ylim(0.0, _autoscale_top(all_values, floor=0.40))
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[ok] {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 — Per-component breakdown
# ---------------------------------------------------------------------------


def plot_component_breakdown(evals: dict[str, dict | None], out_path: Path) -> None:
    series = {label: ev for label, ev in evals.items() if ev}
    if not series:
        print("[skip] component breakdown — no eval JSONs")
        return

    import matplotlib.pyplot as plt

    # Real keys in the eval JSON are e.g. `FieldMatchRubric` (with suffix).
    # We display the short name on the axis but look up `<short>Rubric` first.
    component_names = ["FormatCheck", "FieldMatch", "InfoGain", "QuestionEfficiency", "HallucinationCheck"]

    def _bd_get(bd: dict, c: str) -> float | None:
        for key in (f"{c}Rubric", c, c.lower(), c.replace("Check", "")):
            v = bd.get(key)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
        return None

    # Average ONLY across scenarios where a score breakdown was actually
    # produced (i.e. the rubric ran). Format-failed scenarios with `{}` get
    # filtered out so the bars represent "given the rubric ran, here is each
    # component's contribution" — which is what the README claims.
    matrix: dict[str, dict[str, float]] = {}
    coverage: dict[str, int] = {}
    for label, ev in series.items():
        sums: dict[str, list[float]] = {c: [] for c in component_names}
        n_scored = 0
        for r in ev.get("results", []):
            bd = r.get("score_breakdown") or {}
            if not bd:
                continue
            n_scored += 1
            for c in component_names:
                v = _bd_get(bd, c)
                if v is not None:
                    sums[c].append(v)
        matrix[label] = {c: _safe_mean(sums[c]) for c in component_names}
        coverage[label] = n_scored

    n_comps = len(component_names)
    n_series = len(series)
    width = 0.8 / max(1, n_series)
    x = list(range(n_comps))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    all_values: list[float] = []
    for i, (label, scores) in enumerate(matrix.items()):
        vals = [scores[c] for c in component_names]
        all_values.extend(vals)
        n = coverage.get(label, 0)
        legend_label = f"{label} (n_scored={n}/{len(series.get(label, {}).get('results', []))})"
        ax.bar(
            [xi + (i - (n_series - 1) / 2) * width for xi in x],
            vals,
            width=width,
            label=legend_label,
            color=_color_for(label, i),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(component_names, rotation=10, ha="right")
    ax.set_ylabel("Avg component score (when rubric ran)")
    ax.set_title("Reward breakdown by rubric component (conditional on score being computed)")
    ax.set_ylim(0.0, _autoscale_top(all_values, floor=1.0))
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[ok] {out_path}")


# ---------------------------------------------------------------------------
# Plot 4 — Before / after summary
# ---------------------------------------------------------------------------


def plot_before_after(evals: dict[str, dict | None], out_path: Path) -> None:
    series = {label: ev for label, ev in evals.items() if ev}
    if not series:
        print("[skip] before/after — no eval JSONs")
        return

    import matplotlib.pyplot as plt

    metrics = {
        "avg_score": "Avg final score",
        "completion_rate": "Completion rate",
    }
    n_series = len(series)
    n_metrics = len(metrics)
    width = 0.8 / max(1, n_series)
    x = list(range(n_metrics))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    all_values: list[float] = []
    for i, (label, ev) in enumerate(series.items()):
        s = ev.get("summary", {}) or {}
        vals = [float(s.get(k, 0.0)) for k in metrics]
        all_values.extend(vals)
        bars = ax.bar(
            [xi + (i - (n_series - 1) / 2) * width for xi in x],
            vals,
            width=width,
            label=label,
            color=_color_for(label, i),
            edgecolor="white",
            linewidth=0.5,
        )
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.values()), rotation=0, ha="center")
    ax.set_ylim(0.0, _autoscale_top(all_values, floor=0.30))
    ax.set_ylabel("Score / rate (n=50, eval v4)")
    ax.set_title("Aggregate eval metrics — base vs trained, all model sizes")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[ok] {out_path}")


# ---------------------------------------------------------------------------
# Plot 5 — Question efficiency histogram
# ---------------------------------------------------------------------------


def plot_question_efficiency(evals: dict[str, dict | None], out_path: Path) -> None:
    series = {label: ev for label, ev in evals.items() if ev}
    if not series:
        print("[skip] question hist — no eval JSONs")
        return

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = list(range(0, 8))  # 0..7 questions

    for i, (label, ev) in enumerate(series.items()):
        qs = [r.get("questions_asked", 0) for r in ev.get("results", [])]
        ax.hist(
            qs,
            bins=bins,
            alpha=0.55,
            label=f"{label} (mean={_safe_mean(qs):.2f})",
            color=_color_for(label, i),
            edgecolor="black",
            align="left",
        )

    ax.set_xticks(list(range(0, 7)))
    ax.set_xlabel("Questions asked per scenario")
    ax.set_ylabel("Count")
    ax.set_title("Question efficiency (lower is better, given same score)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[ok] {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-history", action="append", default=[],
                        help="LABEL=PATH log_history.json (can repeat for multi-run)")
    parser.add_argument("--eval-policy", default=None, help="Path to outputs/eval_policy.json")
    parser.add_argument("--eval-base", default=None, help="Path to outputs/eval_<model>_base.json")
    parser.add_argument("--eval-trained", default=None, help="Path to outputs/eval_<model>_trained.json")
    parser.add_argument("--eval", action="append", default=[],
                        help="LABEL=PATH eval JSON (can repeat to add extra series)")
    parser.add_argument("--out-dir", default="plots", help="Output directory for PNGs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evals: dict[str, dict | None] = {}
    if args.eval_policy:
        evals["policy (baseline)"] = _load_json(args.eval_policy)
    if args.eval_base:
        evals["untrained"] = _load_json(args.eval_base)
    if args.eval_trained:
        evals["trained"] = _load_json(args.eval_trained)
    for spec in args.eval:
        if "=" not in spec:
            print(f"[warn] --eval expects LABEL=PATH, got {spec!r}; skipping")
            continue
        # rpartition: labels may legitimately contain '=' (e.g. "Run X, beta=0")
        label, _, path = spec.rpartition("=")
        label = label.strip()
        path = path.strip()
        loaded = _load_json(path)
        if loaded is not None:
            evals[label] = loaded

    if args.log_history and len(args.log_history) == 1 and "=" not in args.log_history[0]:
        log_history = _load_json(args.log_history[0]) or []
        plot_reward_loss_curves(log_history, out_dir / "01_reward_loss_curves.png")
    elif args.log_history:
        labelled: dict[str, list] = {}
        for spec in args.log_history:
            # Use rpartition so labels can contain '=' (e.g. "Run X, beta=0").
            # Path is always the trailing token after the LAST '='.
            label, sep, path = spec.rpartition("=")
            if not sep:
                # Fallback: no '=' at all → treat the whole thing as a path
                label, path = path, label
            data = _load_json(path.strip())
            if data is not None:
                labelled[label.strip()] = data
        plot_reward_loss_curves_multi(labelled, out_dir / "01_reward_loss_curves.png")
    else:
        print("[skip] reward/loss curves \u2014 no log_history")

    plot_per_family_bars(evals, out_dir / "02_per_family_bars.png")
    plot_component_breakdown(evals, out_dir / "03_component_breakdown.png")
    plot_before_after(evals, out_dir / "04_before_after.png")
    plot_question_efficiency(evals, out_dir / "05_question_efficiency.png")

    print()
    print(f"All plots written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
