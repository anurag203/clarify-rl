"""Rich Gradio Blocks UI for the ClarifyRL HF Space.

Replaces the bare-bones auto-generated OpenEnv Gradio with a 4-tab
dashboard that judges can explore without leaving the Space URL.
"""

from __future__ import annotations

import json
import asyncio
from pathlib import Path
from typing import Any

import gradio as gr

_ROOT = Path(__file__).resolve().parent.parent
_PLOTS = _ROOT / "plots"
_SCENARIOS = _ROOT / "scenarios" / "eval_held_out.json"


def _load_summary_table() -> str:
    """Build a markdown table from runs_summary.json."""
    p = _PLOTS / "runs_summary.json"
    if not p.exists():
        return "*runs_summary.json not found*"
    data = json.loads(p.read_text())
    rows = data.get("rows", [])
    lines = [
        "| Model | Avg Score | Completion Rate | Event Planning | Meeting Scheduling |",
        "|-------|-----------|-----------------|----------------|-------------------|",
    ]
    for r in rows:
        label = r["label"]
        avg = f'{r["avg_score"]:.4f}'
        comp = f'{r["completion_rate"]:.0%}'
        ep = f'{r.get("fam_event_planning", 0):.3f}'
        ms = f'{r.get("fam_meeting_scheduling", 0):.3f}'
        lines.append(f"| {label} | {avg} | {comp} | {ep} | {ms} |")
    return "\n".join(lines)


def _plot_path(name: str) -> str | None:
    p = _PLOTS / name
    return str(p) if p.exists() else None


# ── Tab 1: Overview ──────────────────────────────────────────────────────

_OVERVIEW_MD = """
# ClarifyRL — AskBeforeYouAct

> Train LLMs to **ask clarifying questions** instead of hallucinating.
>
> **Theme #5 Wild Card** · Teaching *epistemic humility* as an AI-safety primitive.

**Team Bhole Chature** (Anurag Agarwal + Kanan Agarwal)
Meta OpenEnv Hackathon Grand Finale, Apr 25-26 2026, Bangalore

---

## The Problem

Today's LLMs default to **hallucinating answers to ambiguous requests** instead of asking
what they don't know. We built an RL environment that **rewards the opposite reflex**:
admit uncertainty, ask the *right* question, then act on grounded information.

## The Environment

Each episode: a deliberately vague request (e.g. "Plan a birthday party") with a hidden
user profile. The agent has 3 MCP tools: `ask_question`, `propose_plan`, `get_task_info`.
A 5-component composable rubric scores the final plan on format, field-match, info-gain,
question efficiency, and hallucination avoidance.

**5 task families**: coding requirements, medical intake, support triage, meeting scheduling, event planning.

## Key Links

| Resource | Link |
|----------|------|
| GitHub | [github.com/anurag203/clarify-rl](https://github.com/anurag203/clarify-rl) |
| Colab Notebook | [Open in Colab](https://colab.research.google.com/github/anurag203/clarify-rl/blob/main/training/train_grpo.ipynb) |
| Blog / Writeup | [docs/blog.md](https://huggingface.co/spaces/agarwalanu3103/clarify-rl/blob/main/docs/blog.md) |
| Trained Model (Run 6) | [Kanan2005/clarify-rl-grpo-qwen3-1-7b-run6](https://huggingface.co/Kanan2005/clarify-rl-grpo-qwen3-1-7b-run6) |
| Trained Model (Run 4) | [anurag203/clarify-rl-run4-qwen3-1.7b-beta0.2](https://huggingface.co/anurag203/clarify-rl-run4-qwen3-1.7b-beta0.2) |
| Interactive Demo | [clarify-rl-demo](https://huggingface.co/spaces/anurag203/clarify-rl-demo) |

---

## Headline Results (n=50 held-out scenarios)
"""

# ── Tab 2: Live Demo ─────────────────────────────────────────────────────

def _load_sample_traces() -> list[dict[str, Any]]:
    """Load a handful of eval results for replay."""
    evals_dir = _ROOT / "outputs" / "run_artifacts"
    traces: list[dict] = []
    for edir in sorted(evals_dir.glob("*/evals/eval_*.json")):
        try:
            data = json.loads(edir.read_text())
            results = data.get("results", [])
            for r in results[:3]:
                if r.get("final_score", 0) > 0:
                    traces.append(r)
                    if len(traces) >= 8:
                        return traces
        except Exception:
            continue
    return traces


def _format_trace_as_chat(trace: dict) -> list[list]:
    """Convert an eval trace into Gradio chatbot tuple format [(user, bot), ...]."""
    pairs: list[list] = []
    scenario_id = trace.get("scenario_id", "unknown")
    family = trace.get("family", "unknown")
    score = trace.get("final_score", 0)

    pairs.append([None,
        f"**Scenario**: `{scenario_id}` | **Family**: `{family}` | **Final Score**: `{score:.3f}`"])

    turns = trace.get("turns", trace.get("conversation", []))
    if isinstance(turns, list):
        pending_user = None
        for t in turns:
            role = t.get("role", "system")
            content = t.get("content", str(t))[:500]
            if role in ("user", "tool_result", "environment"):
                pending_user = content
            elif role in ("assistant", "agent", "model"):
                pairs.append([pending_user, content])
                pending_user = None
        if pending_user:
            pairs.append([pending_user, None])

    breakdown = trace.get("score_breakdown", {})
    if breakdown:
        bd_lines = "\n".join(f"  - **{k}**: {v:.3f}" for k, v in breakdown.items())
        pairs.append([None, f"**Rubric Breakdown**:\n{bd_lines}"])

    return pairs


async def _run_live_episode(difficulty: str) -> list[list]:
    """Run a live episode against the local env via WebSocket."""
    import websockets
    pairs: list[list] = []
    try:
        async with websockets.connect("ws://localhost:8000/ws", open_timeout=5) as ws:
            await ws.send(json.dumps({"type": "reset", "data": {"task_id": difficulty}}))
            resp = json.loads(await ws.recv())
            data = resp.get("data", {})
            obs = data.get("observation", {})
            result_raw = obs.get("result", "")
            try:
                info = json.loads(result_raw) if isinstance(result_raw, str) else result_raw
            except (json.JSONDecodeError, TypeError):
                info = {}

            request = info.get("request", str(info))
            family = info.get("family", "unknown")
            max_steps = info.get("max_steps", 8)

            pairs.append([
                f"**New Episode** ({difficulty})",
                f"**Family**: `{family}`\n**Request**: \"{request}\"\n**Budget**: {max_steps} steps",
            ])

            for step in range(1, max_steps + 1):
                action = {"type": "step", "data": {
                    "type": "call_tool",
                    "tool_name": "get_task_info",
                    "arguments": {},
                }}
                await ws.send(json.dumps(action))
                step_resp = json.loads(await ws.recv())
                step_data = step_resp.get("data", {})
                step_obs = step_data.get("observation", {})
                reward = step_data.get("reward", 0)
                done = step_data.get("done", False)
                result_str = step_obs.get("result", "")

                pairs.append([
                    f"Step {step}: `get_task_info`",
                    f"Reward: `{reward}` | Done: `{done}`\n```\n{str(result_str)[:300]}\n```",
                ])

                if done:
                    break
    except Exception as exc:
        pairs.append([None, f"Connection error: {exc}"])
    return pairs


# ── Tab 3: API & Docker ──────────────────────────────────────────────────

_API_MD = """
## Environment API

This Space exposes a full OpenEnv-compatible environment. All endpoints are live right now.

### Health Check

```bash
curl https://agarwalanu3103-clarify-rl.hf.space/health
# {"status": "healthy"}
```

### Reset (start a new episode)

```bash
curl -X POST https://agarwalanu3103-clarify-rl.hf.space/reset \\
  -H 'Content-Type: application/json' \\
  -d '{"task_id": "medium"}'
```

Returns a `CallToolObservation` with the vague user request, task family, and step budget.

### Step (take an action)

```bash
curl -X POST https://agarwalanu3103-clarify-rl.hf.space/step \\
  -H 'Content-Type: application/json' \\
  -d '{
    "type": "call_tool",
    "tool_name": "ask_question",
    "arguments": {"question": "What is the budget?"}
  }'
```

### WebSocket (for training / streaming)

```python
import websockets, json, asyncio

async def demo():
    async with websockets.connect("wss://agarwalanu3103-clarify-rl.hf.space/ws") as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_id": "easy"}}))
        print(json.loads(await ws.recv()))

asyncio.run(demo())
```

### Available Tools

| Tool | Arguments | Description |
|------|-----------|-------------|
| `ask_question` | `{"question": "..."}` | Ask one clarifying question (max 6 per episode) |
| `propose_plan` | `{"plan": '{"field": "value", ...}'}` | Submit final plan as JSON string. Ends episode. |
| `get_task_info` | `{}` | Re-read the original user request |

---

## Run Locally with Docker

```bash
git clone https://github.com/anurag203/clarify-rl.git
cd clarify-rl
docker build -t clarify-rl .
docker run -p 8000:8000 clarify-rl
# Visit http://localhost:8000
```

Or with pip:

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

## openenv.yaml

```yaml
spec_version: 1
name: clarify_rl
type: space
runtime: fastapi
app: server.app:app
port: 8000

tasks:
  - id: easy
    title: "Mild Ambiguity (2-3 fields)"
    max_steps: 8
    difficulty: easy
  - id: medium
    title: "Moderate Ambiguity (4-5 fields)"
    max_steps: 10
    difficulty: medium
  - id: hard
    title: "High Ambiguity (6-7 fields)"
    max_steps: 12
    difficulty: hard
```

## Composable Rubric

```
Sequential(
  Gate(FormatCheck, threshold=0.5),
  WeightedSum([
    FieldMatch        0.50,   # plan fields match hidden profile
    InfoGain          0.20,   # questions revealed critical fields
    QuestionEfficiency 0.15,  # fewer questions = better
    HallucinationCheck 0.15,  # no fabricated values
  ])
)
```
"""

# ── Tab 4: Training Results ──────────────────────────────────────────────

_RESULTS_MD = """
## Training Progression

The plot below shows reward climbing over 300-400 training steps for all GRPO runs.
Run 6 (dark blue) has the healthiest reward curve — non-zero from step 1, peaking at 0.27.
This was achieved by fixing 4 root causes in the training pipeline:

1. **Example contamination**: removed misleading field-name example from the prompt
2. **Sparse reward**: added plan-submission bonus and no-plan penalty
3. **Missing required keys**: surfaced required field names in the observation
4. **Role mismatch**: aligned training and eval prompt formats

## KL-Anchor Ablation

The central finding: **GRPO without a KL anchor causes catastrophic capability collapse**.
Run 2 (beta=0) destroyed event_planning (0.138 -> 0.000). Run 4 (beta=0.2) recovered it
and **beat the base** (0.175 vs 0.138). Run 6 (beta=1.0, fixed pipeline) nearly matches
base aggregate while having the strongest training signal.

## Same-Base Delta

The delta plot shows per-family score change (trained - base) for each run.
Positive = training helped. Negative = training hurt.
"""


# ── Build the full UI ────────────────────────────────────────────────────

def build_gradio_ui() -> gr.Blocks:
    """Construct the 4-tab Gradio Blocks dashboard."""

    try:
        theme = gr.themes.Soft(primary_hue="blue", secondary_hue="purple")
    except Exception:
        theme = None

    with gr.Blocks(
        theme=theme,
        title="ClarifyRL — AskBeforeYouAct",
        css="""
        .hero-img img { max-height: 400px; object-fit: contain; }
        .plot-img img { max-height: 500px; object-fit: contain; }
        footer { display: none !important; }
        """,
    ) as demo:

        gr.Markdown(
            "# ClarifyRL — AskBeforeYouAct\n"
            "> Train LLMs to ask clarifying questions instead of hallucinating. "
            "| [GitHub](https://github.com/anurag203/clarify-rl) "
            "| [Blog](https://huggingface.co/spaces/agarwalanu3103/clarify-rl/blob/main/docs/blog.md) "
            "| [Colab](https://colab.research.google.com/github/anurag203/clarify-rl/blob/main/training/train_grpo.ipynb)"
        )

        with gr.Tabs():

            # ── TAB 1: Overview ──────────────────────────────────────
            with gr.TabItem("Overview", id="overview"):
                gr.Markdown(_OVERVIEW_MD)

                hero = _plot_path("08_training_progression.png")
                if hero:
                    gr.Image(hero, label="Training Progression & Eval Results",
                             elem_classes=["hero-img"])

                gr.Markdown("### Score Table (all runs, n=50 held-out)")
                gr.Markdown(_load_summary_table())

                summary_img = _plot_path("07_runs_summary_table.png")
                if summary_img:
                    gr.Image(summary_img, label="Runs Summary Table",
                             elem_classes=["plot-img"])

            # ── TAB 2: Live Demo ─────────────────────────────────────
            with gr.TabItem("Live Demo", id="demo"):
                gr.Markdown(
                    "## Try the Environment\n\n"
                    "Run a live episode against the ClarifyRL environment. "
                    "Select a difficulty and click **Run Episode** to see "
                    "the env reset and step through tool calls.\n\n"
                    "*Note: This runs the environment only (no trained model). "
                    "It demonstrates the API interaction loop.*"
                )

                with gr.Row():
                    difficulty = gr.Dropdown(
                        choices=["easy", "medium", "hard"],
                        value="medium",
                        label="Difficulty",
                        scale=1,
                    )
                    run_btn = gr.Button("Run Episode", variant="primary", scale=1)

                chatbot = gr.Chatbot(
                    label="Episode Trace",
                    height=450,
                )

                def run_episode(diff: str):
                    try:
                        return asyncio.run(_run_live_episode(diff))
                    except Exception as exc:
                        return [[None, f"Error: {exc}"]]

                run_btn.click(fn=run_episode, inputs=[difficulty], outputs=[chatbot])

                gr.Markdown("---\n### Pre-recorded Eval Traces (scored episodes)")
                trace_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select a trace to view",
                    interactive=True,
                )
                trace_chat = gr.Chatbot(
                    label="Eval Trace Replay",
                    height=400,
                )

                _traces = _load_sample_traces()
                if _traces:
                    trace_labels = [
                        f"{t.get('scenario_id', f'trace-{i}')} (score={t.get('final_score', 0):.3f})"
                        for i, t in enumerate(_traces)
                    ]
                    trace_dropdown.choices = trace_labels

                    def show_trace(selection: str):
                        if not selection:
                            return []
                        idx = next(
                            (i for i, lb in enumerate(trace_labels) if lb == selection),
                            None,
                        )
                        if idx is None:
                            return []
                        return _format_trace_as_chat(_traces[idx])

                    trace_dropdown.change(fn=show_trace, inputs=[trace_dropdown], outputs=[trace_chat])

            # ── TAB 3: API & Docker ──────────────────────────────────
            with gr.TabItem("API & Docker", id="api"):
                gr.Markdown(_API_MD)

            # ── TAB 4: Training Results ──────────────────────────────
            with gr.TabItem("Training Results", id="results"):
                gr.Markdown(_RESULTS_MD)

                plot_sections = [
                    ("Training Progression", "08_training_progression.png"),
                    ("Training Diagnostics (Convergence)", "09_training_diagnostics.png"),
                    ("Reward & KL Curves", "01_reward_loss_curves.png"),
                    ("Same-Base Delta (where RL helps vs hurts)", "06_same_base_delta.png"),
                    ("Per-Family Scores", "02_per_family_bars.png"),
                    ("Rubric Component Breakdown", "03_component_breakdown.png"),
                    ("Before vs After", "04_before_after.png"),
                    ("Question Efficiency", "05_question_efficiency.png"),
                ]

                for title, fname in plot_sections:
                    p = _plot_path(fname)
                    if p:
                        gr.Markdown(f"### {title}")
                        gr.Image(p, show_label=False,
                                 elem_classes=["plot-img"])

    return demo
