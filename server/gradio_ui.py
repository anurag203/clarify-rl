"""Rich Gradio Blocks UI for the ClarifyRL HF Space.

Dark neon cyberpunk theme. CSS injected via gr.HTML("<style>") to work
with Gradio 6.x where css= param on gr.Blocks is ignored when using
mount_gradio_app().
"""

from __future__ import annotations

import json
import asyncio
from pathlib import Path
from typing import Any

import gradio as gr

_ROOT = Path(__file__).resolve().parent.parent
_PLOTS = _ROOT / "plots"

# ---------------------------------------------------------------------------
# CSS — injected via gr.HTML so it works with mount_gradio_app + Gradio 6.x
# ---------------------------------------------------------------------------
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Force dark background everywhere ─────────────────────── */
html, body, main,
.gradio-container, .gradio-container.gradio-container-6-13-0,
div[class*="gradio"], .contain, .app,
[data-testid="blocks-container"] {
    background: #0a0a1a !important;
    color: #e0e0ff !important;
}

footer, .footer { display: none !important; }

/* ── Tabs ─────────────────────────────────────────────────── */
[role="tablist"] {
    background: #111128 !important;
    border-radius: 12px 12px 0 0 !important;
    border-bottom: 2px solid #1e1e4a !important;
    padding: 4px !important;
}
[role="tab"] {
    color: #8888bb !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.82em !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    padding: 10px 18px !important;
    transition: all 0.3s ease;
}
[role="tab"]:hover {
    color: #00f0ff !important;
    background: rgba(0,240,255,0.08) !important;
}
[role="tab"][aria-selected="true"] {
    color: #00f0ff !important;
    background: rgba(0,240,255,0.12) !important;
    border-color: #00f0ff !important;
    box-shadow: 0 0 15px rgba(0,240,255,0.2), inset 0 0 15px rgba(0,240,255,0.05);
}
[role="tabpanel"] {
    background: #0a0a1a !important;
    border: none !important;
}

/* ── Markdown text ────────────────────────────────────────── */
.markdown-text, .prose, .md, [data-testid="markdown"],
.markdown-text p, .prose p {
    color: #e0e0ff !important;
}
.markdown-text h1, .prose h1 { font-family: 'Orbitron', monospace !important; color: #00f0ff !important; text-shadow: 0 0 10px rgba(0,240,255,0.3); font-size: 1.7em !important; border-bottom: 1px solid #1e1e4a; padding-bottom: 8px; }
.markdown-text h2, .prose h2 { font-family: 'Orbitron', monospace !important; color: #00f0ff !important; text-shadow: 0 0 8px rgba(0,240,255,0.25); font-size: 1.3em !important; }
.markdown-text h3, .prose h3 { font-family: 'Orbitron', monospace !important; color: #39ff14 !important; text-shadow: 0 0 8px rgba(57,255,20,0.2); font-size: 1.1em !important; }
.markdown-text strong, .prose strong { color: #fffc00 !important; }
.markdown-text a, .prose a { color: #00f0ff !important; text-decoration: none; border-bottom: 1px solid rgba(0,240,255,0.3); transition: all 0.2s; }
.markdown-text a:hover, .prose a:hover { color: #ff00e5 !important; border-color: #ff00e5; text-shadow: 0 0 8px rgba(255,0,229,0.4); }
.markdown-text code, .prose code {
    background: rgba(0,240,255,0.1) !important;
    color: #00f0ff !important;
    border: 1px solid rgba(0,240,255,0.2) !important;
    border-radius: 4px; padding: 1px 6px;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.88em;
}
.markdown-text pre, .prose pre {
    background: #0d0d2b !important;
    border: 1px solid #1e1e4a !important;
    border-radius: 8px !important;
}
.markdown-text pre code, .prose pre code {
    background: transparent !important; border: none !important;
    color: #39ff14 !important;
}
.markdown-text blockquote, .prose blockquote {
    border-left: 3px solid #ff00e5 !important;
    background: rgba(255,0,229,0.05) !important;
    padding: 10px 16px !important; border-radius: 0 8px 8px 0;
    color: #8888bb !important;
}
.markdown-text hr, .prose hr {
    border: none !important; height: 1px !important;
    background: linear-gradient(90deg, transparent, #00f0ff, #ff00e5, transparent) !important;
    margin: 24px 0 !important;
}

/* ── Tables ───────────────────────────────────────────────── */
.markdown-text table, .prose table { border-collapse: collapse; width: 100%; }
.markdown-text th, .prose th {
    background: rgba(0,240,255,0.1) !important; color: #00f0ff !important;
    font-family: 'Orbitron', monospace !important; font-size: 0.78em;
    text-transform: uppercase; letter-spacing: 1px;
    padding: 10px 12px !important; border-bottom: 2px solid #00f0ff !important;
}
.markdown-text td, .prose td {
    padding: 8px 12px !important; border-bottom: 1px solid #1e1e4a !important;
    color: #e0e0ff !important;
}
.markdown-text tr:hover td, .prose tr:hover td { background: rgba(0,240,255,0.04) !important; }

/* ── Images ───────────────────────────────────────────────── */
.image-container, [data-testid="image"] {
    border: 1px solid #1e1e4a !important; border-radius: 12px !important;
    overflow: hidden; background: #0a0a1a !important;
    box-shadow: 0 0 20px rgba(0,240,255,0.08);
    transition: box-shadow 0.3s ease;
}
.image-container:hover, [data-testid="image"]:hover {
    box-shadow: 0 0 30px rgba(0,240,255,0.18), 0 0 60px rgba(255,0,229,0.08);
}
.image-container img, [data-testid="image"] img {
    border-radius: 12px;
}

/* ── Buttons ──────────────────────────────────────────────── */
button.primary, [data-testid="button"].primary,
button[variant="primary"] {
    background: linear-gradient(135deg, #00f0ff, #ff00e5) !important;
    color: #fff !important; font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important; letter-spacing: 1px; text-transform: uppercase;
    border: none !important; border-radius: 8px !important;
    box-shadow: 0 0 20px rgba(0,240,255,0.3), 0 0 40px rgba(255,0,229,0.15);
    transition: all 0.3s ease;
}
button.primary:hover, [data-testid="button"].primary:hover {
    box-shadow: 0 0 30px rgba(0,240,255,0.5), 0 0 60px rgba(255,0,229,0.3);
    transform: translateY(-1px);
}

/* ── Inputs / dropdowns ───────────────────────────────────── */
input, select, textarea, [data-testid="textbox"],
.border-none, .dropdown-arrow, .wrap {
    background: #111128 !important; color: #e0e0ff !important;
    border-color: #1e1e4a !important; border-radius: 8px !important;
}
label, .label-text, [data-testid="label-text"] {
    color: #8888bb !important;
}

/* ── Chatbot ──────────────────────────────────────────────── */
[data-testid="chatbot"], .chatbot {
    background: #111128 !important;
    border: 1px solid #1e1e4a !important; border-radius: 12px !important;
}

/* ── Neon pulse animation for live dot ────────────────────── */
@keyframes neonPulse {
    0%, 100% { box-shadow: 0 0 4px #39ff14; }
    50% { box-shadow: 0 0 16px #39ff14, 0 0 30px rgba(57,255,20,0.3); }
}
@keyframes gradientShift {
    0%, 100% { background-position: 0% center; }
    50% { background-position: 200% center; }
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plot_path(name: str) -> str | None:
    p = _PLOTS / name
    return str(p) if p.exists() else None


def _load_summary_table() -> str:
    p = _PLOTS / "runs_summary.json"
    if not p.exists():
        return "*runs_summary.json not found*"
    data = json.loads(p.read_text())
    rows = data.get("rows", [])
    lines = [
        "| Model | Avg Score | Completion | Event Planning | Meeting Sched. |",
        "|-------|-----------|------------|----------------|----------------|",
    ]
    for r in rows:
        lines.append(f"| {r['label']} | {r['avg_score']:.4f} | {r['completion_rate']:.0%} | {r.get('fam_event_planning',0):.3f} | {r.get('fam_meeting_scheduling',0):.3f} |")
    return "\n".join(lines)


def _header_html() -> str:
    return """
    <div style="
        text-align:center; padding:32px 20px 24px;
        background: linear-gradient(135deg, #0a0a2e 0%, #1a0a3e 50%, #0a0a2e 100%);
        border-bottom: 2px solid #00f0ff;
        box-shadow: 0 4px 30px rgba(0,240,255,0.15);
        border-radius: 12px; margin-bottom: 12px;
    ">
        <h1 style="
            font-family: 'Orbitron', monospace; font-size: 2.6em; font-weight: 900;
            background: linear-gradient(90deg, #00f0ff, #ff00e5, #00f0ff);
            background-size: 200% auto;
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            animation: gradientShift 4s ease infinite;
            margin: 0 0 10px 0; letter-spacing: 3px;
        ">CLARIFY RL</h1>
        <p style="color:#c0c0ff; font-size:1.05em; margin:4px 0; font-family:'Inter',sans-serif;">
            <span style="
                display:inline-block; width:9px; height:9px; background:#39ff14;
                border-radius:50%; margin-right:8px; vertical-align:middle;
                animation: neonPulse 2s ease infinite;
            "></span>
            Train LLMs to <strong style="color:#fff;">ask clarifying questions</strong> instead of hallucinating
        </p>
        <p style="color:#8888bb; font-size:0.85em; margin-top:10px; font-family:'Inter',sans-serif;">
            <a href="https://github.com/anurag203/clarify-rl" style="color:#00f0ff; text-decoration:none; font-weight:600;">GitHub</a>
            &nbsp;&bull;&nbsp;
            <a href="https://huggingface.co/spaces/agarwalanu3103/clarify-rl/blob/main/Blog.md" style="color:#00f0ff; text-decoration:none; font-weight:600;">Blog</a>
            &nbsp;&bull;&nbsp;
            <a href="https://colab.research.google.com/github/anurag203/clarify-rl/blob/main/training/train_grpo.ipynb" style="color:#00f0ff; text-decoration:none; font-weight:600;">Colab</a>
            &nbsp;&bull;&nbsp;
            <a href="https://huggingface.co/spaces/anurag203/clarify-rl-demo" style="color:#00f0ff; text-decoration:none; font-weight:600;">Demo</a>
        </p>
    </div>
    """


def _stat_cards_html() -> str:
    p = _PLOTS / "runs_summary.json"
    if not p.exists():
        return ""
    data = json.loads(p.read_text())
    rows = data.get("rows", [])
    best = max((r for r in rows if "GRPO" in r.get("label", "")), key=lambda r: r.get("avg_score", 0), default=None)
    base = next((r for r in rows if r.get("label") == "1.7B base"), None)
    ceiling = next((r for r in rows if r.get("label") == "4B base"), None)
    n_runs = sum(1 for r in rows if "GRPO" in r.get("label", ""))

    def card(value, label, color, glow):
        return f"""
        <div style="
            flex:1; min-width:140px; background:#111128;
            border:1px solid #1e1e4a; border-radius:12px;
            padding:18px 14px; text-align:center;
            transition: all 0.3s ease; cursor:default;
        " onmouseover="this.style.borderColor='{color}'; this.style.boxShadow='0 0 20px {glow}';"
           onmouseout="this.style.borderColor='#1e1e4a'; this.style.boxShadow='none';">
            <div style="font-family:'Orbitron',monospace; font-size:1.9em; font-weight:900;
                color:{color}; text-shadow: 0 0 10px {glow};">{value}</div>
            <div style="font-size:0.72em; color:#8888bb; text-transform:uppercase;
                letter-spacing:1.5px; margin-top:5px; font-family:'Inter',sans-serif;">{label}</div>
        </div>"""

    cards = []
    if best:
        cards.append(card(f"{best['avg_score']:.3f}", "Best Trained", "#00f0ff", "rgba(0,240,255,0.4)"))
    if base:
        cards.append(card(f"{base['avg_score']:.3f}", "1.7B Base", "#ff00e5", "rgba(255,0,229,0.4)"))
    if ceiling:
        cards.append(card(f"{ceiling['avg_score']:.3f}", "4B Ceiling", "#39ff14", "rgba(57,255,20,0.4)"))
    cards.append(card(str(n_runs), "GRPO Runs", "#fffc00", "rgba(255,252,0,0.4)"))
    cards.append(card("5", "Task Families", "#ff6b00", "rgba(255,107,0,0.4)"))

    return f'<div style="display:flex; gap:14px; margin:14px 0; flex-wrap:wrap;">{"".join(cards)}</div>'


# ---------------------------------------------------------------------------
# Tab content
# ---------------------------------------------------------------------------

_OVERVIEW_MD = """
# ClarifyRL

> Train LLMs to **ask clarifying questions** instead of hallucinating.
>
> **Theme #5 Wild Card** &middot; Teaching *epistemic humility* as an AI-safety primitive.

**Team Bhole Chature** (Anurag Agarwal + Kanan Agarwal) &middot; Meta OpenEnv Hackathon Grand Finale, Apr 25-26 2026

---

## The Problem

Today's LLMs default to **hallucinating answers to ambiguous requests** instead of asking
what they don't know. We built an RL environment that **rewards the opposite reflex**:
admit uncertainty, ask the *right* question, then act on grounded information.

## The Environment

Each episode: a deliberately vague request (e.g. *"Plan a birthday party"*) with a **hidden
user profile**. The agent has 3 MCP tools: `ask_question`, `propose_plan`, `get_task_info`.
A **5-component composable rubric** scores the final plan.

**5 task families**: coding requirements, medical intake, support triage, meeting scheduling, event planning.

---

## Key Links

| Resource | Link |
|----------|------|
| GitHub | [github.com/anurag203/clarify-rl](https://github.com/anurag203/clarify-rl) |
| Blog / Writeup | [Blog.md](https://huggingface.co/spaces/agarwalanu3103/clarify-rl/blob/main/Blog.md) |
| Colab Notebook | [Open in Colab](https://colab.research.google.com/github/anurag203/clarify-rl/blob/main/training/train_grpo.ipynb) |
| Trained Model (Run 6) | [clarify-rl-grpo-qwen3-1-7b-run6](https://huggingface.co/Kanan2005/clarify-rl-grpo-qwen3-1-7b-run6) |
| Trained Model (Run 4) | [clarify-rl-run4-qwen3-1.7b-beta0.2](https://huggingface.co/anurag203/clarify-rl-run4-qwen3-1.7b-beta0.2) |
| Interactive Demo | [clarify-rl-demo](https://huggingface.co/spaces/anurag203/clarify-rl-demo) |

---

## Headline Results (n=50 held-out scenarios)
"""

_API_MD = """
## Environment API

This Space exposes a **live** OpenEnv-compatible environment. All endpoints are active right now.

---

### Health Check
```bash
curl https://agarwalanu3103-clarify-rl.hf.space/health
```

### Reset (start a new episode)
```bash
curl -X POST https://agarwalanu3103-clarify-rl.hf.space/reset \\
  -H 'Content-Type: application/json' \\
  -d '{"task_id": "medium"}'
```

### Step (take an action)
```bash
curl -X POST https://agarwalanu3103-clarify-rl.hf.space/step \\
  -H 'Content-Type: application/json' \\
  -d '{"type":"call_tool","tool_name":"ask_question","arguments":{"question":"What is the budget?"}}'
```

### WebSocket
```python
import websockets, json, asyncio

async def demo():
    async with websockets.connect("wss://agarwalanu3103-clarify-rl.hf.space/ws") as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_id": "easy"}}))
        print(json.loads(await ws.recv()))

asyncio.run(demo())
```

---

### Available Tools

| Tool | Arguments | Description |
|------|-----------|-------------|
| `ask_question` | `{"question": "..."}` | Ask one clarifying question (max 6/episode) |
| `propose_plan` | `{"plan": '{"key":"val"}'}` | Submit final plan as JSON string. **Ends episode.** |
| `get_task_info` | `{}` | Re-read the original user request |

---

## Run Locally with Docker
```bash
git clone https://github.com/anurag203/clarify-rl.git
cd clarify-rl
docker build -t clarify-rl .
docker run -p 7860:7860 clarify-rl
```

## openenv.yaml
```yaml
spec_version: 1
name: clarify_rl
type: space
runtime: fastapi
app: server.app:app
port: 7860
tasks:
  - id: easy
    title: "Mild Ambiguity (2-3 fields)"
    max_steps: 8
  - id: medium
    title: "Moderate Ambiguity (4-5 fields)"
    max_steps: 10
  - id: hard
    title: "High Ambiguity (6-7 fields)"
    max_steps: 12
```

## Composable Rubric
```
Sequential(
  Gate(FormatCheck, threshold=0.5),
  WeightedSum([
    FieldMatch         0.50
    InfoGain           0.20
    QuestionEfficiency 0.15
    HallucinationCheck 0.15
  ])
)
```
"""

_RESULTS_MD = """
## Training Progression

Run 6 (dark blue) has the healthiest reward curve: **non-zero from step 1**, peaking at **0.27**.
Run 7 (beta=0.3, lr=1e-6) is training with rewards reaching **0.73** at step 115.

### Root causes fixed in Run 6

1. **Example contamination** &mdash; removed misleading field-name example
2. **Sparse reward** &mdash; added plan-submission bonus and no-plan penalty
3. **Missing required keys** &mdash; surfaced required field names in the observation
4. **Role mismatch** &mdash; aligned training and eval prompt formats

---

## KL-Anchor Ablation (5-point beta sweep)

| Beta | Run | Avg Score | Key finding |
|------|-----|-----------|-------------|
| 0.0 | Run 2 | 0.029 | Catastrophic collapse on event_planning |
| 0.2 | Run 4 | 0.056 | Recovered event_planning, beats base (0.175 vs 0.138) |
| 0.3 | Run 7 | *training* | Training reward 0.48-0.73 (highest ever) |
| 0.5 | Run 5 | *canceled* | Reward stuck at 0 (pre-fix pipeline) |
| 1.0 | Run 6 | 0.061 | Nearly matches base. Strongest training signal pre-Run 7. |

---
"""


# ---------------------------------------------------------------------------
# Live demo
# ---------------------------------------------------------------------------

def _load_sample_traces() -> list[dict[str, Any]]:
    evals_dir = _ROOT / "outputs" / "run_artifacts"
    traces: list[dict] = []
    for edir in sorted(evals_dir.glob("*/evals/eval_*.json")):
        try:
            data = json.loads(edir.read_text())
            for r in data.get("results", [])[:3]:
                if r.get("final_score", 0) > 0:
                    traces.append(r)
                    if len(traces) >= 8:
                        return traces
        except Exception:
            continue
    return traces


def _format_trace_as_chat(trace: dict) -> list[list]:
    pairs: list[list] = []
    sid = trace.get("scenario_id", "unknown")
    fam = trace.get("family", "unknown")
    score = trace.get("final_score", 0)
    pairs.append([None, f"**{sid}** | Family: `{fam}` | Score: **{score:.3f}**"])
    turns = trace.get("turns", trace.get("conversation", []))
    if isinstance(turns, list):
        pending = None
        for t in turns:
            role = t.get("role", "system")
            content = t.get("content", str(t))[:500]
            if role in ("user", "tool_result", "environment"):
                pending = content
            elif role in ("assistant", "agent", "model"):
                pairs.append([pending, content])
                pending = None
        if pending:
            pairs.append([pending, None])
    bd = trace.get("score_breakdown", {})
    if bd:
        pairs.append([None, "**Rubric**: " + " | ".join(f"**{k}**: {v:.3f}" for k, v in bd.items())])
    return pairs


async def _run_live_episode(difficulty: str) -> list[list]:
    import websockets
    pairs: list[list] = []
    try:
        async with websockets.connect("ws://localhost:7860/ws", open_timeout=5) as ws:
            await ws.send(json.dumps({"type": "reset", "data": {"task_id": difficulty}}))
            resp = json.loads(await ws.recv())
            data = resp.get("data", {})
            obs = data.get("observation", {})
            raw = obs.get("result", "")
            try:
                info = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                info = {}
            req = info.get("request", str(info))
            fam = info.get("family", "unknown")
            mx = info.get("max_steps", 8)
            pairs.append([f"Start ({difficulty})", f"**{fam}**: \"{req}\"\nBudget: {mx} steps"])
            for step in range(1, mx + 1):
                action = {"type": "step", "data": {"type": "call_tool", "tool_name": "get_task_info", "arguments": {}}}
                await ws.send(json.dumps(action))
                sr = json.loads(await ws.recv())
                sd = sr.get("data", {})
                rw = sd.get("reward", 0)
                dn = sd.get("done", False)
                rs = sd.get("observation", {}).get("result", "")
                pairs.append([f"Step {step}: `get_task_info`", f"Reward: `{rw}` | Done: `{dn}`\n```\n{str(rs)[:300]}\n```"])
                if dn:
                    break
    except Exception as exc:
        pairs.append([None, f"Connection error: {exc}"])
    return pairs


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_gradio_ui() -> gr.Blocks:

    with gr.Blocks(title="ClarifyRL — AskBeforeYouAct") as demo:

        # Inject CSS via HTML tag — the only reliable method with mount_gradio_app + Gradio 6.x
        gr.HTML(f"<style>{_CSS}</style>")

        # Header with inline styles as fallback
        gr.HTML(_header_html())

        # Stat cards with inline styles
        gr.HTML(_stat_cards_html())

        with gr.Tabs():

            with gr.TabItem("Overview"):
                gr.Markdown(_OVERVIEW_MD)
                hero = _plot_path("08_training_progression.png")
                if hero:
                    gr.Image(hero, label="Training Progression & Eval Results")
                gr.Markdown("### Score Table (all runs, n=50 held-out)")
                gr.Markdown(_load_summary_table())
                si = _plot_path("07_runs_summary_table.png")
                if si:
                    gr.Image(si, label="Runs Summary")

            with gr.TabItem("Live Demo"):
                gr.Markdown(
                    "## Try the Environment\n\n"
                    "Run a **live episode** against the ClarifyRL environment. "
                    "Select difficulty and click **Run Episode**."
                )
                with gr.Row():
                    difficulty = gr.Dropdown(["easy", "medium", "hard"], value="medium", label="Difficulty", scale=1)
                    run_btn = gr.Button("Run Episode", variant="primary", scale=1)
                chatbot = gr.Chatbot(label="Episode Trace", height=450)

                def run_ep(diff):
                    try:
                        return asyncio.run(_run_live_episode(diff))
                    except Exception as exc:
                        return [[None, f"Error: {exc}"]]

                run_btn.click(fn=run_ep, inputs=[difficulty], outputs=[chatbot])

                gr.Markdown("---\n### Pre-recorded Eval Traces")
                tdd = gr.Dropdown(choices=[], label="Select a scored trace", interactive=True)
                tchat = gr.Chatbot(label="Eval Trace Replay", height=400)
                _traces = _load_sample_traces()
                if _traces:
                    _labels = [f"{t.get('scenario_id', f't-{i}')} (score={t.get('final_score',0):.3f})" for i, t in enumerate(_traces)]
                    tdd.choices = _labels
                    def _show(sel):
                        if not sel: return []
                        idx = next((i for i, l in enumerate(_labels) if l == sel), None)
                        return _format_trace_as_chat(_traces[idx]) if idx is not None else []
                    tdd.change(fn=_show, inputs=[tdd], outputs=[tchat])

            with gr.TabItem("API & Docker"):
                gr.Markdown(_API_MD)

            with gr.TabItem("Training Results"):
                gr.Markdown(_RESULTS_MD)
                for title, fname in [
                    ("Training Progression", "08_training_progression.png"),
                    ("Training Diagnostics", "09_training_diagnostics.png"),
                    ("Reward & KL Curves", "01_reward_loss_curves.png"),
                    ("Same-Base Delta", "06_same_base_delta.png"),
                    ("Per-Family Scores", "02_per_family_bars.png"),
                    ("Component Breakdown", "03_component_breakdown.png"),
                    ("Before vs After", "04_before_after.png"),
                    ("Question Efficiency", "05_question_efficiency.png"),
                ]:
                    p = _plot_path(fname)
                    if p:
                        gr.Markdown(f"### {title}")
                        gr.Image(p, show_label=False)

    return demo
