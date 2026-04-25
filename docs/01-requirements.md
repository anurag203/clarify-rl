# 01 — Requirements

Three layers of requirements: **must-have for validator**, **must-have for winning**, **nice-to-have**.

## 1. Hackathon Validator (Mandatory — auto-reject if missing)

These are checked automatically before a human ever sees the submission.

| ID | Requirement | Status |
|----|-------------|--------|
| V1 | Public Hugging Face Space at submitted URL (incognito-testable) | ⏳ |
| V2 | Valid `openenv.yaml` (parseable, spec_version: 1) | ⏳ |
| V3 | Environment subclasses `Environment` or `MCPEnvironment` | ⏳ |
| V4 | Implements Gym-style `reset()` / `step()` / `state` | ⏳ |
| V5 | `plots/loss_curve.png` committed to repo | ⏳ |
| V6 | `plots/reward_curve.png` committed to repo | ⏳ |
| V7 | Runnable Colab notebook for training (linked) | ⏳ |
| V8 | Training script uses Unsloth or HF TRL | ⏳ |
| V9 | README links HF Space + Colab + writeup with embedded plots | ⏳ |
| V10 | Writeup (HF blog OR YouTube ≤2 min) linked from README | ⏳ |
| V11 | No reserved tool names (`reset`, `step`, `state`, `close` for MCP) | ✅ (planned: `ask_question`, `propose_plan`, `get_task_info`) |
| V12 | Client never imports server internals | ⏳ (will use thin `MCPToolClient` subclass) |

## 2. Functional Requirements (Core product)

### F1 — Environment

- F1.1: Environment loads on `reset(task_id="easy"|"medium"|"hard")`
- F1.2: Each `reset()` generates a fresh procedural scenario (request + hidden profile)
- F1.3: Agent has access to 3 MCP tools: `ask_question`, `propose_plan`, `get_task_info`
- F1.4: Episode terminates on `propose_plan` OR after 6 questions OR after `max_steps`
- F1.5: Per-step reward returned in `observation.reward`
- F1.6: Final episode reward = composable rubric score (0.0-1.0)
- F1.7: `state` property always reflects current episode tracking

### F2 — Profile Generator

- F2.1: Generates profiles with 2-12 hidden fields per scenario
- F2.2: 5 task families (3 high-stakes + 2 personal): `coding_requirements` / `medical_intake` / `support_triage` / `meeting_scheduling` / `event_planning`
- F2.3: Each field has a value drawn from a constrained vocabulary
- F2.4: Difficulty level controls number of hidden fields (easy: 2-3, med: 5-6, hard: 8+)
- F2.5: Pure-Python deterministic when seeded (for reproducibility)

### F3 — User Simulator

- F3.1: Takes a question text + hidden profile, returns an answer
- F3.2: Keyword/intent matching maps Q to profile field
- F3.3: If Q matches a known field → return profile value
- F3.4: If Q matches no field → "I don't have a strong preference"
- F3.5: Latency: <50ms (rule-based, pure Python — no LLM call during training)

### F4 — Rubric System

- F4.1: Uses OpenEnv's `Rubric` base class hierarchy
- F4.2: Composes `Sequential(Gate(FormatCheck), WeightedSum([4 components]))`
- F4.3: Each component returns 0.0-1.0 (higher = better)
- F4.4: Weights sum to 1.0 in `WeightedSum`
- F4.5: All components introspectable via `env.rubric.named_rubrics()`

### F5 — Training

- F5.1: GRPO via TRL on Qwen2.5-1.5B-Instruct (Unsloth 4-bit)
- F5.2: Rollouts call the OpenEnv environment (NOT a static dataset)
- F5.3: Training runs ≥300 steps with visible upward reward curve
- F5.4: Loss + reward curves saved as `.png` after run

### F6 — Evaluation

- F6.1: Baseline (untrained) eval on 100 held-out scenarios
- F6.2: Trained eval on the same 100 scenarios
- F6.3: Report aggregate + per-family + per-difficulty + per-component metrics (incl. headline **hallucination rate**)
- F6.4: Generate before/after bar chart as `.png`

## 3. Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| N1 | Rollout latency (single episode) | ≤3 sec on M3 Pro / Colab T4 |
| N2 | Local server startup | ≤5 sec |
| N3 | Memory footprint at runtime | ≤500 MB (env only, no model) |
| N4 | Number of training scenarios | ≥200 procedurally generated |
| N5 | Number of held-out eval scenarios | ≥100 (disjoint from training) |
| N6 | Reward curve must show monotonic-ish improvement | from baseline to ≥3x |
| N7 | Code style follows existing OpenEnv reference patterns | (see SRE env) |
| N8 | All deliverable links work from logged-out browser | 100% |

## 4. What We Are NOT Building

To stay scope-disciplined for 48h:

- ❌ Multi-agent self-play (would need 2x rollouts, no time)
- ❌ Real LLM-based user simulator during training (too slow)
- ❌ Sandboxed code execution
- ❌ Custom GPU kernels or model architecture changes
- ❌ Web UI beyond the auto-generated Gradio one OpenEnv provides
- ❌ Database backend (everything is in-memory or JSON file)
- ❌ Multi-language support (English only)

## 5. Acceptance Criteria (Done = winning-ready)

The project is "done" when ALL of these hold:

- [ ] `curl http://localhost:8000/health` returns 200 from a fresh container
- [ ] HF Space loads in incognito browser and Gradio UI works
- [ ] Colab notebook runs end-to-end without manual intervention
- [ ] `plots/reward_curve.png` shows visible upward trend
- [ ] `plots/loss_curve.png` shows visible downward trend
- [ ] Trained model achieves ≥2.5x baseline on held-out plan_satisfaction
- [ ] Hallucination rate drops from ~90% baseline to ≤10% trained on held-out set
- [ ] README has all 4 deliverable links + embedded plots + 1-line captions
- [ ] Two-trace demo (baseline vs trained on same prompt) recorded and linked
- [ ] All 12 validator items (V1-V12) pass
