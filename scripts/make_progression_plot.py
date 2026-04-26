#!/usr/bin/env python
"""Generate the headline "training improves the model" plots.

Produces:
  08_training_progression.png
      LEFT:  smoothed reward over training step (all runs) with 1.7B base
             reference line so judges see the policy gradient working.
      RIGHT: same-model eval before/after pairs with delta arrows.

  09_training_diagnostics.png
      LEFT:  reward std over training step (convergence signal).
      RIGHT: mean completion length over step (behaviour shift).

Inputs:
  --log-history LABEL=PATH   log_history.json per run (repeat)
  --summary     PATH         plots/runs_summary.json
  --out-dir     PATH         plots/ (default)
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

_LABEL_COLORS: dict[str, str] = {
    "0.6B base":                "#ffb74d",
    "0.6B GRPO (Run 1)":        "#1f77b4",
    "1.7B base":                "#66bb6a",
    "1.7B GRPO no-KL (Run 2)":  "#e53935",
    "1.7B GRPO +KL (Run 4)":    "#2e7d32",
    "1.7B GRPO fixed (Run 6)":  "#0d47a1",
    "1.7B GRPO best (Run 7)":   "#ff6f00",
    "4B base":                  "#5e35b1",
    "4B-instruct":              "#00838f",
    "4B GRPO (Run 3)":          "#ff6f00",
}

_FALLBACK = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
             "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _color(label: str, i: int = 0) -> str:
    return _LABEL_COLORS.get(label, _FALLBACK[i % len(_FALLBACK)])


def _rolling(vals: list[float], w: int) -> list[float]:
    out: list[float] = []
    for i in range(len(vals)):
        chunk = vals[max(0, i - w + 1):i + 1]
        out.append(statistics.mean(chunk) if chunk else 0.0)
    return out


def _extract_series(hist: list[dict]) -> dict[str, list]:
    """Pull step, reward, reward_std, completion length, and kl from log_history."""
    steps, rewards, reward_stds, comp_lens, kls = [], [], [], [], []
    for row in hist:
        step = row.get("step")
        if step is None:
            continue
        rew = row.get("reward") or row.get("rewards/reward_func/mean")
        if rew is None:
            continue
        steps.append(int(step))
        rewards.append(float(rew))
        reward_stds.append(float(row.get("reward_std") or row.get("rewards/reward_func/std") or 0.0))
        comp_lens.append(float(row.get("completions/mean_length", 0.0)))
        kls.append(float(row.get("kl", 0.0)))
    return {"steps": steps, "rewards": rewards, "reward_stds": reward_stds,
            "comp_lens": comp_lens, "kls": kls}


_BEFORE_AFTER_PAIRS: list[tuple[str, str]] = [
    ("0.6B base", "0.6B GRPO (Run 1)"),
    ("1.7B base", "1.7B GRPO no-KL (Run 2)"),
    ("1.7B base", "1.7B GRPO +KL (Run 4)"),
    ("1.7B base", "1.7B GRPO fixed (Run 6)"),
    ("1.7B base", "1.7B GRPO best (Run 7)"),
]


def plot_08(runs: dict[str, dict], summary_rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, (ax_rew, ax_ba) = plt.subplots(1, 2, figsize=(16, 5.5),
                                         gridspec_kw={"width_ratios": [1.2, 1]})

    # --- LEFT: Smoothed reward over step ---
    window = 30
    for i, (label, data) in enumerate(runs.items()):
        s = data["series"]
        if not s["steps"]:
            continue
        raw = s["rewards"]
        smooth = _rolling(raw, window)
        col = _color(label, i)
        ax_rew.plot(s["steps"], raw, color=col, alpha=0.18, lw=0.8)
        ax_rew.plot(s["steps"], smooth, color=col, lw=2.5,
                    label=f"{label} (rolling-{window})")
        if smooth:
            ax_rew.annotate(f"{smooth[-1]:.4f}",
                            xy=(s["steps"][-1], smooth[-1]),
                            fontsize=8, fontweight="bold", color=col,
                            xytext=(5, 5), textcoords="offset points")

    base_17b_avg = 0.0
    for row in summary_rows:
        if row.get("label") == "1.7B base":
            base_17b_avg = row.get("avg_score", 0.0)
            break
    if base_17b_avg > 0:
        ax_rew.axhline(base_17b_avg, color="#66bb6a", ls="--", lw=1.5, alpha=0.7)
        ax_rew.text(5, base_17b_avg + 0.002, f"1.7B base eval avg = {base_17b_avg:.3f}",
                    fontsize=8, color="#66bb6a", fontstyle="italic")

    ax_rew.set_xlabel("Training step", fontsize=11)
    ax_rew.set_ylabel("Mean rubric reward", fontsize=11)
    ax_rew.set_title("Reward climbs over training\n(policy gradient is working)", fontsize=12)
    ax_rew.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax_rew.grid(alpha=0.3)
    ax_rew.set_ylim(bottom=-0.005)

    # --- RIGHT: Before/after eval bars ---
    by_label = {r["label"]: r for r in summary_rows}
    pairs = [(b, t) for b, t in _BEFORE_AFTER_PAIRS
             if b in by_label and t in by_label]
    n = len(pairs)
    x_pos = list(range(n))
    bar_w = 0.35

    for idx, (base_lbl, trained_lbl) in enumerate(pairs):
        base_score = by_label[base_lbl]["avg_score"]
        trained_score = by_label[trained_lbl]["avg_score"]
        delta = trained_score - base_score

        ax_ba.bar(idx - bar_w / 2, base_score, bar_w,
                  color="#bdbdbd", edgecolor="white", linewidth=0.5)
        ax_ba.bar(idx + bar_w / 2, trained_score, bar_w,
                  color=_color(trained_lbl, idx), edgecolor="white", linewidth=0.5)

        top = max(base_score, trained_score)
        sign = "+" if delta >= 0 else ""
        ax_ba.text(idx, top + 0.008, f"{sign}{delta:.3f}",
                   ha="center", va="bottom", fontsize=9, fontweight="bold",
                   color="#2e7d32" if delta >= 0 else "#c62828")
        ax_ba.text(idx - bar_w / 2, base_score + 0.002, f"{base_score:.3f}",
                   ha="center", va="bottom", fontsize=7, color="#616161")
        ax_ba.text(idx + bar_w / 2, trained_score + 0.002, f"{trained_score:.3f}",
                   ha="center", va="bottom", fontsize=7, color="#212121")

    pair_labels = []
    for b, t in pairs:
        short_t = t.split("(")[-1].rstrip(")")
        size = b.split(" ")[0]
        pair_labels.append(f"{size}\n{short_t}")
    ax_ba.set_xticks(x_pos)
    ax_ba.set_xticklabels(pair_labels, fontsize=9)
    ax_ba.set_ylabel("Avg eval score (n=50)", fontsize=11)
    ax_ba.set_title("Eval score: base (grey) vs trained (color)\nDelta labeled above each pair", fontsize=12)
    ax_ba.grid(alpha=0.3, axis="y")

    vals = [by_label[b]["avg_score"] for b, _ in pairs] + [by_label[t]["avg_score"] for _, t in pairs]
    top_val = max(vals) if vals else 0.1
    ax_ba.set_ylim(0, top_val * 1.4 + 0.02)

    grey_patch = mpatches.Patch(color="#bdbdbd", label="Base (untrained)")
    trained_patch = mpatches.Patch(color="#1f77b4", label="After GRPO")
    ax_ba.legend(handles=[grey_patch, trained_patch], fontsize=8, loc="upper right")

    fig.suptitle("ClarifyRL — Training progression and evaluation improvement", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_path}")


def plot_09(runs: dict[str, dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_std, ax_len) = plt.subplots(1, 2, figsize=(14, 5))
    window = 20

    for i, (label, data) in enumerate(runs.items()):
        s = data["series"]
        if not s["steps"]:
            continue
        col = _color(label, i)

        smooth_std = _rolling(s["reward_stds"], window)
        ax_std.plot(s["steps"], smooth_std, color=col, lw=2, label=f"{label} (rolling-{window})")

        smooth_len = _rolling(s["comp_lens"], window)
        ax_len.plot(s["steps"], smooth_len, color=col, lw=2, label=label)

    ax_std.set_xlabel("Training step")
    ax_std.set_ylabel("Reward std (within batch)")
    ax_std.set_title("Reward variance over training\n(shrinking = policy converging)")
    ax_std.legend(fontsize=8)
    ax_std.grid(alpha=0.3)

    ax_len.set_xlabel("Training step")
    ax_len.set_ylabel("Mean completion length (tokens)")
    ax_len.set_title("Completion length over training\n(tracks output verbosity shift)")
    ax_len.legend(fontsize=8)
    ax_len.grid(alpha=0.3)

    fig.suptitle("ClarifyRL — Training diagnostics", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-history", action="append", default=[],
                   help="LABEL=PATH (can repeat)")
    p.add_argument("--summary", default="plots/runs_summary.json",
                   help="Path to runs_summary.json")
    p.add_argument("--out-dir", default="plots")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, dict] = {}
    for spec in args.log_history:
        label, _, path = spec.rpartition("=")
        if not label:
            label, path = path, label
        path = path.strip()
        label = label.strip()
        p_path = Path(path)
        if not p_path.exists():
            print(f"[skip] {label}: {path} not found")
            continue
        hist = json.loads(p_path.read_text())
        runs[label] = {"series": _extract_series(hist)}

    summary_rows: list[dict] = []
    sp = Path(args.summary)
    if sp.exists():
        summary_rows = json.loads(sp.read_text()).get("rows", [])
    else:
        print(f"[warn] {args.summary} not found — before/after panel will be empty")

    if runs:
        plot_08(runs, summary_rows, out_dir / "08_training_progression.png")
        plot_09(runs, out_dir / "09_training_diagnostics.png")
    else:
        print("[skip] no log_history files provided")


if __name__ == "__main__":
    main()
