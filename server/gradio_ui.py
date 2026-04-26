"""Rich Gradio Blocks UI for the ClarifyRL HF Space.

Replaces the bare-bones auto-generated OpenEnv Gradio with a 4-tab
dashboard that judges can explore without leaving the Space URL.

Dark neon theme with glowing accents for visual impact.
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

# ---------------------------------------------------------------------------
# Neon dark CSS
# ---------------------------------------------------------------------------
_NEON_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --neon-cyan: #00f0ff;
    --neon-magenta: #ff00e5;
    --neon-green: #39ff14;
    --neon-yellow: #fffc00;
    --neon-orange: #ff6b00;
    --bg-dark: #0a0a1a;
    --bg-card: #111128;
    --bg-card-hover: #1a1a3e;
    --border-glow: #1e1e4a;
    --text-primary: #e0e0ff;
    --text-muted: #8888bb;
}

/* ── Global ────────────────────────────────────────────────── */
.gradio-container {
    background: var(--bg-dark) !important;
    font-family: 'Inter', sans-serif !important;
    max-width: 1200px !important;
}

.dark .gradio-container, .gradio-container {
    background: var(--bg-dark) !important;
}

footer { display: none !important; }

/* ── Neon header bar ──────────────────────────────────────── */
.neon-header {
    text-align: center;
    padding: 30px 20px 20px;
    margin-bottom: 10px;
    background: linear-gradient(135deg, #0a0a2e 0%, #1a0a3e 50%, #0a0a2e 100%);
    border-bottom: 2px solid var(--neon-cyan);
    box-shadow: 0 4px 30px rgba(0, 240, 255, 0.15);
    border-radius: 12px;
}
.neon-header h1 {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 2.4em !important;
    font-weight: 900 !important;
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-magenta), var(--neon-cyan));
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: neonShift 4s ease infinite;
    text-shadow: none;
    margin: 0 0 8px 0;
    letter-spacing: 2px;
}
.neon-header p {
    color: var(--text-muted) !important;
    font-size: 1em;
    margin: 4px 0;
}
.neon-header a {
    color: var(--neon-cyan) !important;
    text-decoration: none;
    font-weight: 600;
    border-bottom: 1px solid rgba(0,240,255,0.3);
    transition: all 0.3s ease;
}
.neon-header a:hover {
    color: var(--neon-magenta) !important;
    border-bottom-color: var(--neon-magenta);
    text-shadow: 0 0 8px rgba(255,0,229,0.5);
}

@keyframes neonShift {
    0%, 100% { background-position: 0% center; }
    50% { background-position: 200% center; }
}

/* ── Tabs ─────────────────────────────────────────────────── */
.tabs > .tab-nav {
    background: var(--bg-card) !important;
    border-radius: 12px 12px 0 0 !important;
    border-bottom: 2px solid var(--border-glow) !important;
    padding: 4px !important;
}
.tabs > .tab-nav > button {
    color: var(--text-muted) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.85em !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    margin: 2px !important;
    transition: all 0.3s ease;
}
.tabs > .tab-nav > button:hover {
    color: var(--neon-cyan) !important;
    background: rgba(0, 240, 255, 0.08) !important;
    border-color: rgba(0, 240, 255, 0.3) !important;
}
.tabs > .tab-nav > button.selected {
    color: var(--neon-cyan) !important;
    background: rgba(0, 240, 255, 0.12) !important;
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.2), inset 0 0 15px rgba(0, 240, 255, 0.05);
}

.tabitem {
    background: var(--bg-dark) !important;
    padding: 20px !important;
}

/* ── Markdown inside tabs ─────────────────────────────────── */
.prose, .markdown-text, .md, .gradio-markdown {
    color: var(--text-primary) !important;
}
.prose h1, .prose h2, .prose h3,
.markdown-text h1, .markdown-text h2, .markdown-text h3 {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--neon-cyan) !important;
    text-shadow: 0 0 10px rgba(0, 240, 255, 0.3);
}
.prose h1 { font-size: 1.8em !important; border-bottom: 1px solid var(--border-glow); padding-bottom: 8px; }
.prose h2 { font-size: 1.4em !important; }
.prose h3 { font-size: 1.15em !important; color: var(--neon-green) !important; text-shadow: 0 0 8px rgba(57,255,20,0.2); }
.prose strong { color: var(--neon-yellow) !important; }
.prose code {
    background: rgba(0, 240, 255, 0.1) !important;
    color: var(--neon-cyan) !important;
    border: 1px solid rgba(0, 240, 255, 0.2) !important;
    border-radius: 4px;
    padding: 1px 6px;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9em;
}
.prose pre {
    background: #0d0d2b !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 8px !important;
    box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
}
.prose pre code {
    background: transparent !important;
    border: none !important;
    color: var(--neon-green) !important;
}
.prose a { color: var(--neon-cyan) !important; transition: color 0.2s; }
.prose a:hover { color: var(--neon-magenta) !important; text-shadow: 0 0 8px rgba(255,0,229,0.4); }
.prose blockquote {
    border-left: 3px solid var(--neon-magenta) !important;
    background: rgba(255, 0, 229, 0.05) !important;
    padding: 10px 16px !important;
    border-radius: 0 8px 8px 0;
    color: var(--text-muted) !important;
}
.prose table { border-collapse: collapse; width: 100%; }
.prose table th {
    background: rgba(0, 240, 255, 0.1) !important;
    color: var(--neon-cyan) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 10px 12px !important;
    border-bottom: 2px solid var(--neon-cyan) !important;
}
.prose table td {
    padding: 8px 12px !important;
    border-bottom: 1px solid var(--border-glow) !important;
    color: var(--text-primary) !important;
}
.prose table tr:hover td {
    background: rgba(0, 240, 255, 0.04) !important;
}
.prose hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--neon-cyan), var(--neon-magenta), transparent) !important;
    margin: 24px 0 !important;
}

/* ── Images ───────────────────────────────────────────────── */
.hero-img, .plot-img {
    border: 1px solid var(--border-glow) !important;
    border-radius: 12px !important;
    overflow: hidden;
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.08);
    transition: box-shadow 0.3s ease;
}
.hero-img:hover, .plot-img:hover {
    box-shadow: 0 0 30px rgba(0, 240, 255, 0.18), 0 0 60px rgba(255, 0, 229, 0.08);
}
.hero-img img, .plot-img img {
    max-height: 500px;
    object-fit: contain;
    border-radius: 12px;
}

/* ── Buttons ──────────────────────────────────────────────── */
.primary.svelte-1ogfvm8, button.primary {
    background: linear-gradient(135deg, var(--neon-cyan), var(--neon-magenta)) !important;
    color: #fff !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.3), 0 0 40px rgba(255, 0, 229, 0.15);
    transition: all 0.3s ease;
}
.primary.svelte-1ogfvm8:hover, button.primary:hover {
    box-shadow: 0 0 30px rgba(0, 240, 255, 0.5), 0 0 60px rgba(255, 0, 229, 0.3);
    transform: translateY(-1px);
}

/* ── Dropdown / inputs ────────────────────────────────────── */
.wrap.svelte-1ogfvm8, .container > .wrap, select, input,
.border-none, .dropdown-arrow {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-glow) !important;
    border-radius: 8px !important;
}

/* ── Chatbot ──────────────────────────────────────────────── */
.chatbot {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 12px !important;
}
.message.user { background: rgba(0, 240, 255, 0.08) !important; border-left: 3px solid var(--neon-cyan) !important; }
.message.bot  { background: rgba(255, 0, 229, 0.05) !important; border-left: 3px solid var(--neon-magenta) !important; }

/* ── Neon stat cards ──────────────────────────────────────── */
.stat-row { display: flex; gap: 16px; margin: 16px 0; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 160px;
    background: var(--bg-card);
    border: 1px solid var(--border-glow);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    transition: all 0.3s ease;
}
.stat-card:hover {
    border-color: var(--neon-cyan);
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.15);
    transform: translateY(-2px);
}
.stat-card .stat-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.8em;
    font-weight: 900;
    color: var(--neon-cyan);
    text-shadow: 0 0 10px rgba(0, 240, 255, 0.4);
}
.stat-card .stat-label {
    font-size: 0.75em;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}
.stat-card.magenta .stat-value { color: var(--neon-magenta); text-shadow: 0 0 10px rgba(255,0,229,0.4); }
.stat-card.green .stat-value { color: var(--neon-green); text-shadow: 0 0 10px rgba(57,255,20,0.4); }
.stat-card.yellow .stat-value { color: var(--neon-yellow); text-shadow: 0 0 10px rgba(255,252,0,0.4); }
.stat-card.orange .stat-value { color: var(--neon-orange); text-shadow: 0 0 10px rgba(255,107,0,0.4); }

/* ── Glow pulse on the title ──────────────────────────────── */
.pulse-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--neon-green);
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 2s ease infinite;
    vertical-align: middle;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 4px var(--neon-green); }
    50% { box-shadow: 0 0 16px var(--neon-green), 0 0 30px rgba(57,255,20,0.3); }
}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _stat_cards_html() -> str:
    p = _PLOTS / "runs_summary.json"
    if not p.exists():
        return ""
    data = json.loads(p.read_text())
    rows = data.get("rows", [])
    best_trained = max(
        (r for r in rows if "GRPO" in r.get("label", "")),
        key=lambda r: r.get("avg_score", 0),
        default=None,
    )
    base_1_7b = next((r for r in rows if r.get("label") == "1.7B base"), None)
    ceiling = next((r for r in rows if r.get("label") == "4B base"), None)
    n_runs = sum(1 for r in rows if "GRPO" in r.get("label", ""))

    cards = []
    if best_trained:
        cards.append(f'<div class="stat-card"><div class="stat-value">{best_trained["avg_score"]:.3f}</div><div class="stat-label">Best Trained Score</div></div>')
    if base_1_7b:
        cards.append(f'<div class="stat-card magenta"><div class="stat-value">{base_1_7b["avg_score"]:.3f}</div><div class="stat-label">1.7B Base Score</div></div>')
    if ceiling:
        cards.append(f'<div class="stat-card green"><div class="stat-value">{ceiling["avg_score"]:.3f}</div><div class="stat-label">4B Ceiling</div></div>')
    cards.append(f'<div class="stat-card yellow"><div class="stat-value">{n_runs}</div><div class="stat-label">GRPO Runs</div></div>')
    cards.append(f'<div class="stat-card orange"><div class="stat-value">5</div><div class="stat-label">Task Families</div></div>')

    return f'<div class="stat-row">{"".join(cards)}</div>'


# ---------------------------------------------------------------------------
# Tab content: Markdown
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
A **5-component composable rubric** scores the final plan on format, field-match, info-gain,
question efficiency, and hallucination avoidance.

**5 task families**: coding requirements, medical intake, support triage, meeting scheduling, event planning.

---

## Key Links

| Resource | Link |
|----------|------|
| GitHub | [github.com/anurag203/clarify-rl](https://github.com/anurag203/clarify-rl) |
| Colab Notebook | [Open in Colab](https://colab.research.google.com/github/anurag203/clarify-rl/blob/main/training/train_grpo.ipynb) |
| Blog / Writeup | [docs/blog.md](https://huggingface.co/spaces/agarwalanu3103/clarify-rl/blob/main/docs/blog.md) |
| Trained Model (Run 6) | [clarify-rl-grpo-qwen3-1-7b-run6](https://huggingface.co/Kanan2005/clarify-rl-grpo-qwen3-1-7b-run6) |
| Trained Model (Run 4) | [clarify-rl-run4-qwen3-1.7b-beta0.2](https://huggingface.co/anurag203/clarify-rl-run4-qwen3-1.7b-beta0.2) |
| Interactive Demo | [clarify-rl-demo](https://huggingface.co/spaces/anurag203/clarify-rl-demo) |

---

## Headline Results (n=50 held-out scenarios)
"""

_API_MD = """
## Environment API

<div class="pulse-dot"></div> This Space exposes a **live** OpenEnv-compatible environment. All endpoints are active right now.

---

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
    async with websockets.connect(
        "wss://agarwalanu3103-clarify-rl.hf.space/ws"
    ) as ws:
        await ws.send(json.dumps({
            "type": "reset",
            "data": {"task_id": "easy"}
        }))
        print(json.loads(await ws.recv()))

asyncio.run(demo())
```

---

### Available Tools

| Tool | Arguments | Description |
|------|-----------|-------------|
| `ask_question` | `{"question": "..."}` | Ask one clarifying question (max 6/episode) |
| `propose_plan` | `{"plan": '{"field": "val"}'}` | Submit final plan as JSON string. **Ends episode.** |
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
  - id: medium
    title: "Moderate Ambiguity (4-5 fields)"
    max_steps: 10
  - id: hard
    title: "High Ambiguity (6-7 fields)"
    max_steps: 12
```

---

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

The plots below show reward climbing over 300-400 training steps for all GRPO runs.
**Run 6** (dark blue) has the healthiest reward curve: non-zero from step 1, peaking at **0.27**.

### Root causes fixed in Run 6

1. **Example contamination** &mdash; removed misleading field-name example from the prompt
2. **Sparse reward** &mdash; added plan-submission bonus and no-plan penalty
3. **Missing required keys** &mdash; surfaced required field names in the observation
4. **Role mismatch** &mdash; aligned training and eval prompt formats

---

## KL-Anchor Ablation

**GRPO without a KL anchor causes catastrophic capability collapse.**

- Run 2 (`beta=0`) destroyed event_planning: `0.138 -> 0.000`
- Run 4 (`beta=0.2`) recovered it and **beat the base**: `0.175 vs 0.138`
- Run 6 (`beta=1.0`, fixed pipeline) nearly matches base aggregate with the strongest training signal

---
"""


# ---------------------------------------------------------------------------
# Live demo helpers
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
        lines = "\n".join(f"  - **{k}**: {v:.3f}" for k, v in bd.items())
        pairs.append([None, f"**Rubric Breakdown**:\n{lines}"])
    return pairs


async def _run_live_episode(difficulty: str) -> list[list]:
    import websockets
    pairs: list[list] = []
    try:
        async with websockets.connect("ws://localhost:8000/ws", open_timeout=5) as ws:
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
                so = sd.get("observation", {})
                rw = sd.get("reward", 0)
                dn = sd.get("done", False)
                rs = so.get("result", "")
                pairs.append([f"Step {step}: `get_task_info`", f"Reward: `{rw}` | Done: `{dn}`\n```\n{str(rs)[:300]}\n```"])
                if dn:
                    break
    except Exception as exc:
        pairs.append([None, f"Connection error: {exc}"])
    return pairs


# ---------------------------------------------------------------------------
# Build the Blocks app
# ---------------------------------------------------------------------------

def build_gradio_ui() -> gr.Blocks:

    with gr.Blocks(
        title="ClarifyRL — AskBeforeYouAct",
        css=_NEON_CSS,
    ) as demo:

        # ── Neon header ──────────────────────────────────────────
        gr.HTML("""
        <div class="neon-header">
            <h1>CLARIFY RL</h1>
            <p><span class="pulse-dot"></span>
               Train LLMs to <strong>ask clarifying questions</strong> instead of hallucinating
            </p>
            <p style="margin-top:8px; font-size:0.85em;">
                <a href="https://github.com/anurag203/clarify-rl">GitHub</a> &nbsp;&bull;&nbsp;
                <a href="https://huggingface.co/spaces/agarwalanu3103/clarify-rl/blob/main/docs/blog.md">Blog</a> &nbsp;&bull;&nbsp;
                <a href="https://colab.research.google.com/github/anurag203/clarify-rl/blob/main/training/train_grpo.ipynb">Colab</a> &nbsp;&bull;&nbsp;
                <a href="https://huggingface.co/spaces/anurag203/clarify-rl-demo">Demo</a>
            </p>
        </div>
        """)

        # ── Stat cards ───────────────────────────────────────────
        gr.HTML(_stat_cards_html())

        with gr.Tabs():

            # ── TAB 1: Overview ──────────────────────────────────
            with gr.TabItem("Overview", id="overview"):
                gr.Markdown(_OVERVIEW_MD)

                hero = _plot_path("08_training_progression.png")
                if hero:
                    gr.Image(hero, label="Training Progression & Eval Results", elem_classes=["hero-img"])

                gr.Markdown("### Score Table (all runs, n=50 held-out)")
                gr.Markdown(_load_summary_table())

                summary_img = _plot_path("07_runs_summary_table.png")
                if summary_img:
                    gr.Image(summary_img, label="Runs Summary", elem_classes=["plot-img"])

            # ── TAB 2: Live Demo ─────────────────────────────────
            with gr.TabItem("Live Demo", id="demo"):
                gr.Markdown(
                    "## Try the Environment\n\n"
                    "Run a **live episode** against the ClarifyRL environment. "
                    "Select difficulty and click **Run Episode**.\n\n"
                    "*Runs the environment only (no trained model). Demonstrates the API loop.*"
                )
                with gr.Row():
                    difficulty = gr.Dropdown(["easy", "medium", "hard"], value="medium", label="Difficulty", scale=1)
                    run_btn = gr.Button("Run Episode", variant="primary", scale=1)

                chatbot = gr.Chatbot(label="Episode Trace", height=450)

                def run_episode(diff):
                    try:
                        return asyncio.run(_run_live_episode(diff))
                    except Exception as exc:
                        return [[None, f"Error: {exc}"]]

                run_btn.click(fn=run_episode, inputs=[difficulty], outputs=[chatbot])

                gr.Markdown("---\n### Pre-recorded Eval Traces")
                trace_dd = gr.Dropdown(choices=[], label="Select a scored trace", interactive=True)
                trace_chat = gr.Chatbot(label="Eval Trace Replay", height=400)

                _traces = _load_sample_traces()
                if _traces:
                    _labels = [f"{t.get('scenario_id', f't-{i}')} (score={t.get('final_score',0):.3f})" for i, t in enumerate(_traces)]
                    trace_dd.choices = _labels

                    def _show(sel):
                        if not sel:
                            return []
                        idx = next((i for i, l in enumerate(_labels) if l == sel), None)
                        return _format_trace_as_chat(_traces[idx]) if idx is not None else []

                    trace_dd.change(fn=_show, inputs=[trace_dd], outputs=[trace_chat])

            # ── TAB 3: API & Docker ──────────────────────────────
            with gr.TabItem("API & Docker", id="api"):
                gr.Markdown(_API_MD)

            # ── TAB 4: Training Results ──────────────────────────
            with gr.TabItem("Training Results", id="results"):
                gr.Markdown(_RESULTS_MD)
                for title, fname in [
                    ("Training Progression", "08_training_progression.png"),
                    ("Training Diagnostics", "09_training_diagnostics.png"),
                    ("Reward & KL Curves", "01_reward_loss_curves.png"),
                    ("Same-Base Delta", "06_same_base_delta.png"),
                    ("Per-Family Scores", "02_per_family_bars.png"),
                    ("Rubric Component Breakdown", "03_component_breakdown.png"),
                    ("Before vs After", "04_before_after.png"),
                    ("Question Efficiency", "05_question_efficiency.png"),
                ]:
                    p = _plot_path(fname)
                    if p:
                        gr.Markdown(f"### {title}")
                        gr.Image(p, show_label=False, elem_classes=["plot-img"])

    return demo
