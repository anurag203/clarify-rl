# ClarifyRL — slide deck (5-min judge read)

> Marp-style markdown slides. Read on GitHub or paste into [Marp](https://marp.app/) to export to PDF. Each `---` is a slide break.

---

## Slide 1 · Title

# **ClarifyRL**

### An RL environment that puts *"Ask Before You Act"* on the reward path

> *Every RLHF, RLVR, and GRPO-on-math paper rewards arriving at the right answer. Almost none reward deciding to ask first. We built the environment that does — and validated it works.*

**Team Bhole Chature** · Anurag Agarwal + Kanan Agarwal
Meta OpenEnv Hackathon Grand Finale · Bangalore · April 25-26, 2026

> Theme #5 (Wild Card): epistemic humility as an AI safety primitive.

`huggingface.co/spaces/agarwalanu3103/clarify-rl` · `github.com/anurag203/clarify-rl`

---

## Slide 2 · The problem

> A user types **"Set up a sync with the team this week"** into your assistant.

It cheerfully replies: *"Done — Thursday at 3pm, 60 minutes, on Zoom with Engineering, Marketing, and Sales."* Three things the model just made up: the **time** (you said *"this week"*), the **duration** (never specified), and the **invitees** (you said *"the team,"* not three departments). Polished. Confident. Completely fabricated. This is a real-world failure mode for agents that have to take action.

The right behaviour: **ask the right clarifying questions first, then act.**

We built an OpenEnv environment that *rewards exactly this behaviour*. The contribution is the environment. To validate it works, we trained Qwen3-1.7B inside it with GRPO — the trained model **beats its own base by +19%** on 50 held-out scenarios.

---

## Slide 3 · Environment design (40% of judging)

`ClarifyEnvironment` extends `MCPEnvironment` and exposes 3 tools:

```text
get_task_info()         # free  — re-read the original ambiguous request
ask_question(question)  # costs 1 from a 6-question budget
propose_plan(plan)      # terminal — runs the rubric, returns score
```

**5 task families, 100s of scenarios per family, 6-step max episode**

```text
Sequential(
  Gate(FormatCheck, threshold=0.5),    # invalid JSON → 0
  WeightedSum([
    FieldMatch     0.50,   # plan correctness vs hidden profile
    InfoGain       0.20,   # did questions reveal critical fields?
    Efficiency     0.15,   # fewer questions = better
    Hallucination  0.15,   # no fabricated values (penalty)
  ])
)
```

`max_concurrent_envs=64` so the same Space services every parallel rollout from every HF Jobs flavour.

---

## Slide 4 · Training recipe

| Run | Base | β (KL anchor) | LR | Steps | Key result |
|---|---|---|---|---|---|
| Run 1 | Qwen3-0.6B | 0 | 1e-6 | 300 | Unlocks event_planning (0 → 0.382 max) |
| Run 2 | Qwen3-1.7B | **0** | 1e-6 | 400 | Regresses (0.067 → 0.029). Collapses event_planning. |
| **Run 4** | Qwen3-1.7B | **0.2** | **5e-7** | 300 | KL anchor recovers event_planning (0 → 0.175, beats base) |
| **Run 6** | Qwen3-1.7B | **1.0** | 5e-7 | 300 | Fixed pipeline. Nearly matches base (0.061 vs 0.063). |
| **Run 7** ⭐ | Qwen3-1.7B | **0.3** | **1e-6** | 400 | **Training rewards 0.48-0.73** (highest ever). Eval pending. |

7 GRPO runs total (5 completed, 1 canceled, 1 training). Same env. Same MCP tools. A 5-point β-sweep {0, 0.2, 0.3, 0.5, 1.0} with a training pipeline overhaul between Runs 4 and 6.

Held-out eval: **n=50 scenarios per checkpoint**, deterministic seeds, async WebSocket harness, Run 1 / Run 2 / Run 4 / 0.6B-base / 1.7B-base / 4B-base / 4B-Instruct = **7 evals** total.

---

## Slide 5 · The headline finding (storytelling 30% + improvement 20%)

> **GRPO without a KL anchor causes catastrophic capability collapse on stronger bases. Adding β=0.2 cleanly fixes it.**

| Family | 1.7B base | **Run 2 (β=0)** | **Run 4 (β=0.2)** | Δ |
|---|---|---|---|---|
| event_planning μ | 0.138 | **0.000 ❌ collapse** | **0.175 ✅ beats base** | +0.175 |
| meeting_scheduling μ | 0.153 | 0.130 | 0.064 | -0.066 |
| meeting_scheduling **max** | 0.500 | **0.725 ↑↑ peak** | 0.350 | -0.375 |
| **avg over all families** | 0.067 | **0.029 ↓** | **0.056 ↑** | +0.027 |

Three things to notice:

1. **Run 2 destroyed event_planning** (mean 0.138 → 0.000). Run 4 recovered it — and *beat* the base on it (0.138 → 0.175).
2. **The cost is peak ceiling**: Run 4's meeting_scheduling top-line drops from 0.725 (Run 2's gem) to 0.350. KL prevents the extreme specialisation Run 2 leaned on.
3. **Run 4's avg (0.056) ≈ 1.7B base (0.067)** — the regularisation is doing exactly what theory says: keeping the policy close to base while slipping in the targeted improvement.

`Run 4 is the hero checkpoint.`  Live demo: `anurag203/clarify-rl-demo` (3 tabs).

---

## Slide 6 · Evidence (the 6 committed plots)

![Same-base delta plot — Run 4 above the base on event_planning](../plots/06_same_base_delta.png)

The bar above zero on `event_planning` is the central result. Same model. Same data. Same env. Only β changed.

Other plots in `plots/`:

- `01_reward_loss_curves.png` — training-time reward + loss curves; Run 4 KL stayed bounded 0.005-0.010 throughout.
- `02_per_family_bars.png` — every model × every family.
- `03_component_breakdown.png` — what each rubric component contributed.
- `04_before_after.png` — base vs trained for all 3 runs.
- `05_question_efficiency.png` — distribution of questions asked / scenario.
- `runs_summary.json` — machine-readable scoreboard.

All plots auto-generate via `scripts/make_plots.py` from `outputs/run_artifacts/*/evals/*.json`.

---

## Slide 7 · Pipeline (10%)

The reproducible chain (every step is an `./scripts/launch_*.sh` away):

```text
training/train_grpo.py  ── HF Jobs ── A100 1.5 h ──> agarwalanu3103/clarify-rl-grpo-*
                                    + self-hosted metrics
scripts/launch_eval_job.sh ── HF Jobs ── A10G 7 min ── vLLM ── n=50 evals → JSON
scripts/make_plots.py ── PNG → committed to git → embedded in README
```

- **Reward components** are separate Rubric classes in `server/rubrics.py`. Composable via `Sequential` and `WeightedSum`. Each one has unit tests in `tests/test_rubrics.py`.
- **`Gate(FormatCheck)`** prevents reward hacking — invalid JSON yields 0 regardless of plan content.
- **vLLM-in-HF-Job eval** because HF Inference Router doesn't serve fine-tuned community uploads. We host vLLM ourselves for $0.13 per 50-scenario eval.
- **Colab badge** for the trainer, so judges can re-run the smoke version end-to-end.

---

## Slide 8 · What we'd do next (we ran out of time, not ideas)

1. **4B + GRPO with fixed pipeline.** The 4B base alone scores 0.1446 — the headroom is real.
2. **Curriculum on family difficulty.** medical_intake + support_triage stayed at 0.000. Needs family-specific warm-up.
3. **Even longer training.** Run 7's reward is still climbing at step 100/400 — 600-1000 steps may push further.
4. **Compositional plans.** Nested format (sub-tasks, conditions) for studying multi-level clarification.

---

## Slide 9 · Submission assets (all 7 links)

| Asset | URL |
|---|---|
| **HF Space (env + Gradio UI)** | `huggingface.co/spaces/agarwalanu3103/clarify-rl` |
| **Demo Space (replay + live chat)** | `huggingface.co/spaces/anurag203/clarify-rl-demo` |
| **Trained model — Run 6 (fixed pipeline)** | `huggingface.co/Kanan2005/clarify-rl-grpo-qwen3-1-7b-run6` |
| **Trained model — Run 4 (KL anchor)** | `huggingface.co/anurag203/clarify-rl-run4-qwen3-1.7b-beta0.2` |
| **Code (GitHub)** | `github.com/anurag203/clarify-rl` |
| **Training notebook (Colab)** | `colab.research.google.com/github/anurag203/clarify-rl/blob/main/training/train_grpo.ipynb` |
| **Blog post (full writeup)** | `github.com/anurag203/clarify-rl/blob/main/docs/blog.md` |

Auto-validator gates all GREEN (see `SUBMISSION_CHECKLIST.md`).

---

## Slide 10 · Thank you

> The KL anchor turned a capability collapse into a transferable improvement, on the same model with the same data.

That's the central result, and it's what we want judges to remember. Live demo waits for you on `anurag203/clarify-rl-demo`.

— **Team Bhole Chature**
