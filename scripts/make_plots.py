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
    palette = ["tab:gray", "tab:orange", "tab:blue", "tab:green", "tab:red"]
    for i, (label, scores) in enumerate(matrix.items()):
        ax.bar(
            [xi + (i - (n_series - 1) / 2) * width for xi in x],
            [scores[fam] for fam in families],
            width=width,
            label=label,
            color=palette[i % len(palette)],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=15, ha="right")
    ax.set_ylabel("Avg final score")
    ax.set_title("Avg score per task family")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
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

    component_names = ["FieldMatch", "InfoGain", "QuestionEfficiency", "HallucinationCheck"]

    matrix: dict[str, dict[str, float]] = {}
    for label, ev in series.items():
        sums: dict[str, list[float]] = {c: [] for c in component_names}
        for r in ev.get("results", []):
            bd = r.get("score_breakdown") or {}
            for c in component_names:
                v = bd.get(c)
                if v is None:
                    v = bd.get(c.lower()) or bd.get(c.replace("Check", "")) or 0.0
                try:
                    sums[c].append(float(v))
                except (TypeError, ValueError):
                    sums[c].append(0.0)
        matrix[label] = {c: _safe_mean(sums[c]) for c in component_names}

    n_comps = len(component_names)
    n_series = len(series)
    width = 0.8 / max(1, n_series)
    x = list(range(n_comps))

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = ["tab:gray", "tab:orange", "tab:blue", "tab:green", "tab:red"]
    for i, (label, scores) in enumerate(matrix.items()):
        ax.bar(
            [xi + (i - (n_series - 1) / 2) * width for xi in x],
            [scores[c] for c in component_names],
            width=width,
            label=label,
            color=palette[i % len(palette)],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(component_names, rotation=10, ha="right")
    ax.set_ylabel("Avg component score")
    ax.set_title("Reward breakdown by rubric component")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
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
        "format_pass_rate": "Format pass rate",
        "completion_rate": "Completion rate",
    }
    n_series = len(series)
    n_metrics = len(metrics)
    width = 0.8 / max(1, n_series)
    x = list(range(n_metrics))

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = ["tab:gray", "tab:orange", "tab:blue", "tab:green", "tab:red"]
    for i, (label, ev) in enumerate(series.items()):
        s = ev.get("summary", {}) or {}
        vals = [float(s.get(k, 0.0)) for k in metrics]
        ax.bar(
            [xi + (i - (n_series - 1) / 2) * width for xi in x],
            vals,
            width=width,
            label=label,
            color=palette[i % len(palette)],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.values()), rotation=10, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score / rate")
    ax.set_title("Aggregate eval metrics")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
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

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = ["tab:gray", "tab:orange", "tab:blue", "tab:green", "tab:red"]
    bins = list(range(0, 8))  # 0..7 questions

    for i, (label, ev) in enumerate(series.items()):
        qs = [r.get("questions_asked", 0) for r in ev.get("results", [])]
        ax.hist(
            qs,
            bins=bins,
            alpha=0.55,
            label=f"{label} (mean={_safe_mean(qs):.2f})",
            color=palette[i % len(palette)],
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
