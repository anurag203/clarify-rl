# 02 — Architecture

## High-level System

```
┌────────────────────────────────────────────────────────────────────┐
│                      TRAINING (Colab T4)                           │
│  ┌──────────────┐    rollouts    ┌──────────────────────────────┐ │
│  │ Qwen2.5-1.5B │ ──────────────▶│ ClarifyEnv (in-process or    │ │
│  │  + Unsloth   │   ◀──reward───  │  via HTTP to HF Space)       │ │
│  │  + TRL GRPO  │                 └──────────────────────────────┘ │
│  └──────────────┘                                                  │
│         │                                                          │
│         ▼                                                          │
│   plots/*.png  + LoRA adapter                                      │
└────────────────────────────────────────────────────────────────────┘
                          │
                          ▼ (push)
┌────────────────────────────────────────────────────────────────────┐
│                  HF SPACE (clarify-rl)                             │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Dockerfile → uvicorn server.app:app on :8000                 │ │
│  │                                                                │ │
│  │   FastAPI (from openenv.create_app):                          │ │
│  │     - POST /reset, /step, /state                              │ │
│  │     - WS /mcp  (MCP JSON-RPC)                                 │ │
│  │     - GET /health, /metadata, /schema                         │ │
│  │     - Gradio UI at /                                          │ │
│  │                                                                │ │
│  │   ClarifyEnvironment(MCPEnvironment):                         │ │
│  │     ├─ reset() → new scenario                                 │ │
│  │     ├─ step() → MCP tool dispatch + reward                    │ │
│  │     ├─ rubric: Sequential(Gate(FormatCheck), WeightedSum...)  │ │
│  │     ├─ profile_generator: ProfileGenerator                    │ │
│  │     └─ user_simulator: UserSimulator (rule-based)             │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
                          ▲
                          │ HTTP / WebSocket
┌────────────────────────────────────────────────────────────────────┐
│                  CLIENT (Judge / Demo)                             │
│  ClarifyClient(MCPToolClient).reset() → list_tools() → call_tool() │
└────────────────────────────────────────────────────────────────────┘
```

## File-system Layout

```
clarify-rl/
├── docs/                          # ALL planning docs (this folder)
├── openenv.yaml                   # Manifest: spec_version, tasks, runtime
├── pyproject.toml                 # Dependencies: openenv-core, fastapi, etc.
├── Dockerfile                     # python:3.11-slim + uv install
├── README.md                      # Public-facing: pitch, results, all links
├── blog.md                        # HF blog post (markdown)
├── __init__.py                    # Package exports
├── models.py                      # ClarifyState(State), action/obs aliases
├── client.py                      # ClarifyClient(MCPToolClient)
├── inference.py                   # Standalone baseline eval script (validator artifact)
├── server/
│   ├── __init__.py
│   ├── app.py                     # create_app(ClarifyEnvironment, ...)
│   ├── clarify_environment.py     # MCPEnvironment subclass + tool registration
│   ├── rubrics.py                 # Rubric subclasses (5 components)
│   ├── profile_generator.py       # Procedural scenario generator
│   ├── user_simulator.py          # Rule-based user (Q → answer)
│   └── grader.py                  # Per-step shaping reward + plan parser
├── scenarios/
│   ├── templates.json             # Task type templates (5 task types)
│   └── eval_held_out.json         # 100 held-out eval scenarios (frozen seed)
├── plots/
│   ├── reward_curve.png           # MUST commit (validator)
│   ├── loss_curve.png             # MUST commit (validator)
│   ├── per_task_bars.png          # Baseline vs trained per task type
│   └── per_component.png          # Rubric component breakdown
├── training/
│   └── train_grpo.ipynb           # Colab GRPO training notebook
├── scripts/
│   ├── evaluate.py                # baseline-vs-trained eval driver
│   ├── generate_scenarios.py      # offline scenario generation
│   └── make_plots.py              # turn eval JSON → .png
├── tests/
│   ├── test_environment.py
│   ├── test_rubrics.py
│   └── test_profile_generator.py
├── outputs/                       # local-only (gitignored)
└── .gitignore
```

## Data Flow per Episode

```
1. Client calls reset(task_id="medium")
2. ProfileGenerator.generate(seed, difficulty) → (request_text, hidden_profile)
3. Env stores hidden_profile internally; returns observation:
   {
     "type": "task",
     "request": "Book me dinner for tomorrow",
     "hint": "5-6 hidden preferences. You have 6 questions.",
     "max_steps": 10
   }
4. LOOP for up to max_steps:
   a. Agent calls one of:
      - ask_question(question="What's your budget?")
      - propose_plan(plan='{"cuisine":"indian", "budget":"$50", ...}')
      - get_task_info()
   b. If ask_question:
      - questions_remaining -= 1
      - UserSimulator.answer(question, hidden_profile) → text
      - State updated; reward = +0.05 (relevant) or -0.02 (duplicate)
   c. If propose_plan:
      - Plan parsed as JSON
      - Episode terminates: done=True
      - Rubric evaluates: Sequential(Gate(FormatCheck), WeightedSum(4 components))
      - reward = final_score (0.0-1.0)
   d. If 6 questions exhausted without plan: forced terminal with reward=0
5. Final observation: terminal info with score breakdown
```

## Component Responsibilities

| Component | Owns | Calls |
|-----------|------|-------|
| `ClarifyEnvironment` | episode lifecycle, state, tool dispatch | profile_generator, user_simulator, rubric |
| `ProfileGenerator` | scenario sampling, request templating | (none) |
| `UserSimulator` | Q→A rule-based mapping | (none) |
| `rubrics.py` | 5 Rubric subclasses + composition | (none) |
| `grader.py` | per-step shaping reward, plan parsing | (none) |
| `app.py` | FastAPI app, root route | OpenEnv `create_app` |
| `client.py` | Thin MCP client | OpenEnv `MCPToolClient` |
| `inference.py` | Baseline eval orchestration | OpenAI API + ClarifyClient |
| `train_grpo.ipynb` | GRPO training loop | Unsloth + TRL + ClarifyClient |
| `evaluate.py` | Baseline vs trained eval | ClarifyClient + plotting |

## Boundaries (Strict)

- `client.py` MUST NOT import anything from `server/`
- `server/` MUST NOT import from `client.py`
- `models.py` is shared (state types, action/obs aliases)
- `rubrics.py` operates only on `(action, observation)` — never reaches into env internals

## OpenEnv API Touchpoints

```python
# Base classes
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import State, EnvironmentMetadata
from openenv.core.env_server.mcp_types import (
    CallToolAction, CallToolObservation,
    ListToolsAction, ToolError, ToolErrorType
)
from openenv.core.env_server.http_server import create_app
from openenv.core.mcp_client import MCPToolClient
from openenv.core.rubrics import (
    Rubric, WeightedSum, Gate, Sequential
)

# FastMCP for tool registration
from fastmcp import FastMCP
```

## Concurrency Model

- `SUPPORTS_CONCURRENT_SESSIONS = False` (default)
- Single-session env per server instance
- For training: spawn N processes each running their own env
  (Unsloth GRPO with batch=4 prompts × 4 completions = 16 rollouts in parallel via async)
- For HF Space hosting: single session is fine (judges call it 1-by-1)
