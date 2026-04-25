---
trigger: always_on
description: ClarifyRL project facts and agent behavior rules — auto-loaded into every chat
---

# ClarifyRL — agent rules (auto-loaded)

## What this project is (one sentence)

ClarifyRL is an OpenEnv environment that trains LLMs to **ask clarifying questions before acting** instead of hallucinating, submitted to the **Meta OpenEnv Hackathon Grand Finale** (Apr 25–26, 2026, Bangalore).

## LOCKED decisions — DO NOT pivot without explicit user approval

- **Idea**: ClarifyRL / AskBeforeYouAct — train epistemic humility via RL
- **Theme pitch**: #5 Wild Card (primary) + #3.2 Personalized + #2 Long-Horizon (secondary)
- **Headline metric**: hallucination rate ~90% → ~3% on 100 held-out scenarios
- **5 task families**: `coding_requirements`, `medical_intake`, `support_triage`, `meeting_scheduling`, `event_planning`
- **Stack**: OpenEnv 0.2.2 + MCPEnvironment + FastMCP + Unsloth + TRL GRPO + Qwen2.5-1.5B-Instruct
- **Compute**: HF Jobs `t4-small` (primary) + free Colab T4 (fallback)
- **Starter notebook**: fork of TRL `openenv_wordle_grpo.ipynb`
- **Rubric architecture**: `Sequential(Gate(FormatCheck), WeightedSum([FieldMatch 0.50, InfoGain 0.20, QuestionEfficiency 0.15, HallucinationCheck 0.15]))`
- **MCP tool names**: `ask_question`, `propose_plan`, `get_task_info` (NOT `reset`/`step`/`state`/`close` — those are reserved)
- **Team**: Bhole Chature (Anurag Agarwal + Kanan Agarwal)
- **Submission deadline**: Apr 26, 5:00 PM IST

## At chat start, you MUST

1. Read `clarify-rl/docs/STATUS.md` to know what's done / in-progress / next-up
2. Read the last 3 entries in `clarify-rl/docs/SESSION_LOG.md` to know what prior agents did
3. Briefly confirm understanding (2-3 bullets) before proposing work
4. NOT re-litigate any LOCKED decision above unless the user explicitly asks

## At chat end, you MUST

1. Append a `SESSION_LOG.md` entry: timestamp + 3-bullet summary of what you did
2. Update `STATUS.md`: bump `last-completed`, `in-progress`, `next-step`, `blockers`
3. Tell the user "STATUS + SESSION_LOG updated"

## Doc structure (read in this order if new)

| # | Path | Purpose |
|---|------|---------|
| Onboarding | `clarify-rl/docs/AGENT_ONBOARDING.md` | Paste-this-first for non-Windsurf agents |
| State | `clarify-rl/docs/STATUS.md` | What's true right now |
| History | `clarify-rl/docs/SESSION_LOG.md` | Append-only session history |
| 00 | `clarify-rl/docs/00-overview.md` | Pitch + problem framing |
| 01 | `clarify-rl/docs/01-requirements.md` | V1-V12 validators + functional reqs |
| 02 | `clarify-rl/docs/02-architecture.md` | System design |
| 03 | `clarify-rl/docs/03-environment-spec.md` | Env state/actions/observations |
| 04 | `clarify-rl/docs/04-rubric-design.md` | Composable rubric details |
| 05 | `clarify-rl/docs/05-scenario-design.md` | Procedural scenarios + 5 families |
| 06 | `clarify-rl/docs/06-training-plan.md` | GRPO + Unsloth + HF Jobs |
| 07 | `clarify-rl/docs/07-deployment.md` | HF Space + submission checklist |
| 08 | `clarify-rl/docs/08-timeline.md` | 48h sprint plan |
| 09 | `clarify-rl/docs/09-risks.md` | Risk register |
| 10 | `clarify-rl/docs/10-positioning.md` | Pitch / framing for judges |

## Coding conventions

- Match existing OpenEnv reference patterns (see `envs/` of meta-pytorch/OpenEnv repo)
- Client (`client.py`) NEVER imports from `server/` — strict separation
- All env code in `clarify-rl/server/`; client in `clarify-rl/client.py`
- Type hints everywhere; dataclasses for actions/observations
- No new files at repo root unless necessary
- Don't add comments or docstrings unless the user asks
- Match the Python style already present in `models.py` and `client.py`

## Behavior rules

- Be terse and direct; no acknowledgment phrases
- Prefer minimal upstream fixes over downstream workarounds
- For non-trivial work, draft a short plan; keep one step in progress; mark done as you go
- If unsure between two paths, ask one focused question; don't guess
- Never delete or weaken tests without explicit direction
- Never run destructive commands (rm, force push, db drops) without user approval
- Lint warnings about markdown table compact-style and blank-lines-around-lists are pre-existing across all docs — ignore them
