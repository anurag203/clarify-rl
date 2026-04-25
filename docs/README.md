# ClarifyRL — Documentation Index

Project: **ClarifyRL** — Train LLMs to ask clarifying questions instead of hallucinating.
Hackathon: **Meta OpenEnv Hackathon Grand Finale**, Apr 25-26, 2026, Bangalore.
Team: **Bhole Chature** (Anurag Agarwal + Kanan Agarwal).

> **New agent / new chat?** Read in this order:
>
> 1. [`AGENT_ONBOARDING.md`](./AGENT_ONBOARDING.md) — paste-this-first for non-Windsurf agents
> 2. [`STATUS.md`](./STATUS.md) — what's true right now
> 3. [`SESSION_LOG.md`](./SESSION_LOG.md) — last 3 entries, what prior agents did
> 4. Then the design docs below.

## Read in this order

| # | Doc | What it covers |
|---|-----|----------------|
| 00 | [overview.md](./00-overview.md) | Pitch, problem statement, why this idea wins |
| 01 | [requirements.md](./01-requirements.md) | Functional, non-functional, hackathon validator requirements |
| 02 | [architecture.md](./02-architecture.md) | System architecture, components, data flow |
| 03 | [environment-spec.md](./03-environment-spec.md) | OpenEnv env design: state, actions, observations, MCP tools |
| 04 | [rubric-design.md](./04-rubric-design.md) | 5-component composable rubric, weights, anti-hacking |
| 05 | [scenario-design.md](./05-scenario-design.md) | Profile schema, task types, user simulator |
| 06 | [training-plan.md](./06-training-plan.md) | GRPO + Unsloth config, baseline, eval methodology |
| 07 | [deployment.md](./07-deployment.md) | HF Space, Colab, README, submission checklist |
| 08 | [timeline.md](./08-timeline.md) | Hour-by-hour 48h sprint plan + team split |
| 09 | [risks.md](./09-risks.md) | Risk register + mitigations + fallback plans |

## Lock-status

- ✅ **Idea LOCKED**: ClarifyRL (AskBeforeYouAct) — train epistemic humility via RL
- ✅ **Theme LOCKED**: **#5 Wild Card (primary)** + 3.2 Personalized + 2 Long-Horizon (secondary)
- ✅ **Task families LOCKED**: 3 high-stakes (coding, medical-intake, support) + 2 personal (meeting, event)
- ✅ **Stack LOCKED**: OpenEnv 0.2.2 + MCPEnvironment + FastMCP + Unsloth + TRL GRPO + Qwen2.5-1.5B
- ✅ **Compute LOCKED**: Colab free T4 + $30 HF inference credits + M3 Pro 18GB
- ✅ **Docs LOCKED**: Positioning sharpened with AI-safety framing
- ⏳ **Code**: Scaffolding done, env + rubric pending

## Headline metric

> **Hallucination rate: ~90% baseline → ~3% trained** (on 100 held-out scenarios across 5 task families).

Secondary metrics: plan satisfaction 27% → 85%, field-match F1 0.20 → 0.92, avg clarifying questions 0.4 → 2.7.
