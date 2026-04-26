# ClarifyRL: Teaching Small LLMs to Ask Before They Act

You text your assistant: *"Plan a birthday party."*

It comes back with a venue you never mentioned, a guest list it made up, and a budget pulled from thin air. Everything looks polished. Everything is wrong.

This is the default mode of every LLM today. They are trained to sound confident, not to say *"wait — how many people are coming?"*

We thought: what if we could train that reflex? Not the confidence. The pause. The question.

So we built **ClarifyRL** — an RL environment where the only way to score well is to **ask the right questions first**, then act on what you actually learned. We trained a 1.7B model across 7 GRPO runs, discovered that a simple KL anchor prevents catastrophic forgetting, and diagnosed 4 hidden bugs in our own training pipeline along the way.

Here is what we found.

**Team Bhole Chature** — Anurag Agarwal + Kanan Agarwal · Meta OpenEnv Hackathon Grand Finale, Apr 25-26 2026

> **Links:** [GitHub](https://github.com/anurag203/clarify-rl) · [Env Space](https://huggingface.co/spaces/agarwalanu3103/clarify-rl) · [Interactive Demo](https://huggingface.co/spaces/anurag203/clarify-rl-demo) · [Colab](https://colab.research.google.com/github/anurag203/clarify-rl/blob/main/training/train_grpo.ipynb) · [Run 6 model](https://huggingface.co/Kanan2005/clarify-rl-grpo-qwen3-1-7b-run6) · [Run 4 model](https://huggingface.co/anurag203/clarify-rl-run4-qwen3-1.7b-beta0.2)

![Training progression and evaluation improvement](../plots/08_training_progression.png)

---

## 1. The problem

You ask a model *"schedule a sync"* and it gives you a meeting at 2pm, 30 minutes, in a room you have never booked. It guessed every field. None of it came from you.

This happens because LLMs are trained to produce answers, not to notice when they do not have the information. RLHF rewards confident-sounding outputs. Saying "I don't know, let me ask" is punished, not rewarded.

We wanted to flip that. Make the model earn its score by asking the right questions first, then planning based on real answers. Not guesses. Not hallucinations. Real information from the user.

That is ClarifyRL.

## 2. The environment

ClarifyRL is an [OpenEnv 0.2.2](https://github.com/meta-pytorch/OpenEnv) environment, deployed as an HF Space (Docker + FastMCP). Each episode:

1. **Hidden profile.** A 12-field user profile is sampled (e.g. `{start_time: "2pm", duration: "30min", meeting_type: "video", ...}`).
2. **Vague request.** The agent only sees a *deliberately ambiguous* surface form ("Schedule a sync"). Critical fields are missing.
3. **Tools.** The agent has three MCP tools:
   - `ask_question(q)` — costs 1 of a 6-question budget; returns the user's natural-language answer plus which field was revealed.
   - `propose_plan(plan)` — submits a JSON string with the agent's chosen fields. **Ends the episode.**
   - `get_task_info()` — re-reads the original request (free).
4. **Reward.** A 5-component composable rubric scores the submitted plan:

   ```
   Sequential(
     Gate(FormatCheck, threshold=0.5),
     WeightedSum([
       FieldMatch     0.50,   # plan correctness vs hidden profile
       InfoGain       0.20,   # did questions reveal critical fields?
       Efficiency     0.15,   # fewer questions = better
       Hallucination  0.15,   # no fabricated values in plan
     ])
   )
   ```

Five task families exercise different ambiguity surfaces: `coding_requirements`, `medical_intake`, `support_triage`, `meeting_scheduling`, `event_planning`. The full eval set is 100 scenarios with deterministic seeds (`scenarios/eval_held_out.json`).

## 3. Why GRPO

We chose [Group-Relative Policy Optimization](https://arxiv.org/abs/2402.03300) because it eliminates the need for a value-function critic — important when:
- The reward signal arrives once at episode end (sparse).
- Episodes have variable length (1-7 turns).
- Rollouts contain mixed tool calls and free text.

GRPO computes per-rollout advantages by comparing each rollout's reward to the group mean, normalized by group standard deviation. **Critical lesson:** with `num_generations=2` (the default in many tutorials), advantage often resolves to exactly 0 when both rollouts produce identical token sequences early in training — giving you a `0.000000 loss` pathology for the first 15-20 steps. Bumping to `num_generations=4` (6 pairwise comparisons) or `8` (28 pairwise comparisons) per group fixes this immediately.

## 4. Training setup

| Run | Model | β (KL) | LR | num_gen | Steps | Account | Status |
|---|---|---|---|---|---|---|---|
| 1 | Qwen3-0.6B | 0.0 | 1e-6 | 8 | 300 | A (agarwalanu3103) | done — eval'd |
| 2 | Qwen3-1.7B | 0.0 | 1e-6 | 8 | 400 | A | done — eval'd |
| 3 | Qwen3-4B | 0.0 | 1e-6 | 2 | 300 | B (Kanan2005) | canceled (queue) |
| **4** | **Qwen3-1.7B** | **0.2** | **5e-7** | 8 | 300 | C (2022uec1542) | **done — eval'd** |
| 5 | Qwen3-1.7B | 0.5 | 5e-7 | 8 | 300 | C | canceled (reward stuck at 0) |
| **6** | **Qwen3-1.7B** | **1.0** | **5e-7** | 8 | 300 | B (Kanan2005) | **done — eval'd (fixed pipeline)** |
| **7** | **Qwen3-1.7B** | **0.3** | **1e-6** | 8 | 400 | A (agarwalanu3103) | **training** |

Run 4 is the controlled twin of Run 2 — same 1.7B model, same env, same compute envelope — except for two changes: TRL `beta=0.2` KL anchor and half the learning rate (5e-7 vs 1e-6). The question we wanted answered: does KL regularization prevent the family-narrowing we saw in Run 2 while preserving its `meeting_scheduling` peak? **Verdict: yes on the first, no on the second** (see § 5.2). The KL stayed bounded between 0.005-0.010 throughout training, confirming the anchor was actively pulling against drift. Run 3 was meant as a 4B-scale confirmation of the same finding but sat in the HF Jobs scheduling queue for 48 minutes on Account B's a100-large; with Run 4 already in hand we canceled it rather than push into the submission deadline (see §7b). All trained runs use the same env Space (Account A), which we configured for `max_concurrent_envs=64` to support concurrent rollout streams. Other knobs:

- TRL `GRPOConfig`: `learning_rate=1e-6`, `gradient_accumulation_steps=8`, `optim="adamw_8bit"`, `gradient_checkpointing=True`.
- vLLM colocate (`vllm_mode="colocate"`, `vllm_gpu_memory_utilization=0.55` for 80 GB tier).
- Chat template: Qwen3 base (we tried Llama-3 and Qwen2.5-Instruct early; both fail TRL's `add_response_schema` — the chat templates don't support tool-use schema injection).
- Reward: passed straight from `ClarifyEnvironment.step` into the `reward_func` via TRL's `environment_factory` API, so the RL loop sees real rubric scores from the live env.

## 5. Results — both runs, n=50 v4 fair eval

### 5.1 Training reward curves

![Reward and loss curves](../plots/01_reward_loss_curves.png)

Both runs show the optimizer climbing honestly:

- **Run 1 (0.6B, 300 steps)**: reward grows from 0.006 (first 10 steps) to 0.045 (last 10 steps) — **7×** growth.
- **Run 2 (1.7B, 400 steps)**: reward grows from 0.0017 (first 30 steps) to 0.022 (last 30 steps) — **13×** growth.

The 1.7B starts further from format-following (its zero-shot reward is essentially zero), takes longer to "decide" what behavior to commit to, and plateaus at a *lower* reward than the 0.6B. That's already a hint of what we'll see in held-out eval: the 1.7B's solution is narrower, even though its base is broader.

> **Training metrics — self-hosted.** All reward curves, KL divergence, completion length, and reward std are plotted from `log_history.json` files committed to this repo. The headline progression plot is [`plots/08_training_progression.png`](../plots/08_training_progression.png); KL + per-step reward lives in [`plots/01_reward_loss_curves.png`](../plots/01_reward_loss_curves.png); convergence diagnostics in [`plots/09_training_diagnostics.png`](../plots/09_training_diagnostics.png). The Run 4 (β=0.2) KL panel shows the term staying bounded at 0.005-0.010 throughout the entire 300 steps — that's the anchor doing its job.

### 5.2 Held-out evaluation: aggregate and per-family

![Aggregate eval metrics](../plots/04_before_after.png)

| Model | Size | Avg score | Completion | Best score | Avg Qs |
|---|---|---|---|---|---|
| Random policy | n/a | 0.0000 | 0% | 0.000 | 3.96 |
| Qwen3-0.6B base | 0.6B | 0.0000 | 0% | 0.000 | 2.84 |
| **Qwen3-0.6B GRPO (Run 1, β=0)** | 0.6B | **0.0076** ↑ | 2% | **0.382** | 4.20 |
| Qwen3-1.7B base | 1.7B | 0.0669 | 18% | 0.522 | 5.20 |
| **Qwen3-1.7B GRPO (Run 2, β=0)** | 1.7B | 0.0286 ↓ | 6% | **0.725** | 5.70 |
| **Qwen3-1.7B GRPO (Run 4, β=0.2)** | 1.7B | **0.0560** ✅ | 14% | 0.510 | _._ |
| Qwen3-4B-Instruct | 4B | 0.0399 | 6% | 0.757 | 4.84 |
| **Qwen3-4B base** ← real ceiling | 4B | **0.1446** | **24%** | **0.819** | 5.10 |

Per-family breakdown — this is the table that earns the headline:

![Per-family scores](../plots/02_per_family_bars.png)

| Model | event_planning | meeting_scheduling | medical_intake | support_triage |
|---|---|---|---|---|
| Qwen3-0.6B base | 0/12 | 0/11 | 0/15 | 0/12 |
| **Qwen3-0.6B GRPO (Run 1, β=0)** | **1/12** (avg 0.032, max 0.382) ✅ | 0/11 | 0/15 | 0/12 |
| Qwen3-1.7B base | 4/12 (avg 0.138, max 0.522) | 5/11 (avg 0.153, max 0.500) | 0/15 | 0/12 |
| Qwen3-1.7B GRPO (Run 2, β=0) | **0/12 (avg 0.000)** ❌ | 3/11 (avg 0.130, **max 0.725 ↑↑**) | 0/15 | 0/12 |
| **Qwen3-1.7B GRPO (Run 4, β=0.2)** | **5/12 (avg 0.175 ✅, max 0.510)** | 1/11 (avg 0.064, max 0.350) | 0/15 | 0/12 |
| Qwen3-4B-Instruct | 3/12 (avg 0.166, max 0.757) | 0/11 | 0/15 | 0/12 |
| **Qwen3-4B base** | **6/12 (avg 0.340, max 0.795)** | **2/11 (avg 0.287, max 0.819)** | 0/15 | 0/12 |

![Same-base delta (Run - Base) for each family](../plots/06_same_base_delta.png)

The plot above is the per-family delta (trained run − same-size base) for each of the three trained runs we have. It shows the KL anchor effect at a glance: on `event_planning`, Run 4 (green, +KL) sits *above* the same-size base — the only post-RL bar in positive territory at 1.7B — while Run 2 (orange, no-KL) crashes to −0.138. Same model, same env, same steps; a single hyperparameter (β) flips the sign. That's the entire hackathon thesis in one chart.

Six honest observations:

1. **The KL anchor cleanly fixed Run 2's regression.** Run 2 (β=0) wiped out `event_planning` (0.138 → 0.000 mean). Run 4 (β=0.2, half LR, *same model*) recovered it to **0.175 — beating the base** (0.138). The same training data, same number of steps, same env, but with a regularizer pulling against drift. This is the cleanest controlled comparison in the table.
2. **The cost of the anchor is the peak.** Run 2's gem was the 0.725 max on `meeting_scheduling` (highest score by a trained model). Run 4 dropped it to 0.350. β=0.2 is doing exactly what β does: stopping the policy from over-committing to one family's solution. For a hackathon, **breadth + recovery beats one peak**, and Run 4's average (0.056) is nearly back to 1.7B base (0.067) while Run 2 was stuck at 0.029.
3. **GRPO on the 0.6B unlocks a capability the base couldn't reach.** The base 0.6B never scored anything; the trained 0.6B scored on `event_planning` (1/12, max 0.382). This is the only sub-1B model in our experiments that has ever produced a non-zero plan score in this env.
4. **Medical intake and support triage are unreachable.** All six trained/base models score 0/27 on these two families. The fields are too coupled (`[order_id, item_issue, refund_or_replace]`) — one wrong guess collapses the plan. Future work: per-family curricula or hierarchical scaffolding.
5. **The real ceiling is Qwen3-4B *base*, not 4B-Instruct.** 4B base (no RL) scores avg 0.1446 and tops 0.819 on `meeting_scheduling` — the highest single-scenario score we've seen at any size. Instruct-tuning *hurt* the 4B (4B-Inst avg 0.0399, max 0.757). For a clarification-style multi-turn task, Qwen3's instruction-SFT seems to weaken the kind of patient field-by-field reasoning the rubric rewards. Whether RL can lift this strong base remains an open question for us — see §7b for why we punted Run 3.
6. **Reward magnitude tells the right story when read in context.** Run 4's training reward peaked at 0.114 (vs Run 2's ~0.029) but the *eval* score is comparable across both — because the KL anchor is keeping the model honest in a way reward magnitude alone can't measure. The KL series (mean 0.005-0.010 throughout) confirmed the regularizer was actively engaged.

### 5.3 Question-efficiency

![Questions asked per scenario](../plots/05_question_efficiency.png)

| Model | Mean Qs | Distribution shape |
|---|---|---|
| Random policy | 3.96 | flat U[0,6] |
| 0.6B base | **2.84** | bimodal at 0 and 5 — many "give up immediately" outcomes |
| 0.6B trained (Run 1) | 4.20 | bimodal at 1 and 5 — uses budget more deliberately |
| 1.7B base | 5.20 | concentrated at 5-6 — leans on "ask until forced" |
| 1.7B trained (Run 2) | 5.70 | shifted further toward the 6-cap |
| 4B-Instruct | 4.84 | broad, 4-6 dominant |

Counter-intuitive read: 0.6B base has the *lowest* mean question count, but that's because it fails to call any tool on most scenarios — its low Qs is unrecoverable parser failure, not efficiency. The trained 0.6B raises the mean to 4.20 because it actually *attempts* tasks now. The 1.7B trained moves slightly toward the 6-question cap, consistent with a model that's more committed to `ask_question(...)` cycling but now misses the "stop and propose" timing on `event_planning`.

### 5.4 What changed mechanically (trace observations)

- **No more `<think>` token-waste anywhere.** Qwen3 ships with reasoning ON by default, which on a 300-token budget burns the entire reply inside `<think>...</think>` and never reaches the TOOL/ARGS block we parse. Disabling via `chat_template_kwargs={"enable_thinking": False}` (mirrored at train *and* eval time) collapsed eval runtime from "never completes within budget" to ~0.7s per scenario for 0.6B and ~2.3s for 1.7B.
- **0.6B: format adherence emerges in the right places.** Trained 0.6B emits balanced `ask_question("...")` then `propose_plan({...})` for the scenarios where it scores. The base 0.6B emits free text or invalid syntax in those same scenarios.
- **1.7B: format adherence emerges *too eagerly*.** Trained 1.7B starts with proper tool calls but truncates the question loop earlier than the base, jumping to `propose_plan({...})` before key fields are revealed. On `event_planning` this collapses to empty/sparse plans (0% pass), even though the base 1.7B happily asks longer and gets fields right.

The trained 0.6B's required-keys mapping is also still rough — it sometimes emits a `coding_requirements`-style plan into a different family, or `{}`. This is consistent with a reward curve still climbing at step 300; closing the FormatCheck threshold (the 0% pass everywhere shows the structural gate is still untripped) needs either more steps with a richer reward, or a structural pre-trained scaffold. We discuss this in section 9.

See [`docs/trace_demo.md`](trace_demo.md) for the full `seed10004_event_planning_hard` trace where the trained 0.6B scores 0.382 vs the base's 0.000.

## 6. The eval-pipeline bug saga (the real story behind the v4 numbers)

We initially saw 0/50 across **every** model — trained, base, instruct-tuned, all of them. That's not a model problem; that's an eval-pipeline problem. We dug in and found **five distinct bugs** silently flattening every score to 0. The v4 numbers in section 5.2 are the first measurement that survives all of them.

1. **Parser bug (function-call form).** The trained 0.6B emits `function_call(arg="value")` style with **nested parens** in question text (e.g. `ask_question("What is your budget? (in USD)")`). Our original `parse_tool_call` used a naive regex that stopped at the first `)`, mangling 100% of the trained model's outputs. **Fix:** replaced with a balanced-paren scanner (`_find_balanced_func_call`) plus dedicated `_parse_positional_args` that handles `key="value"`, `key={json}`, and bare positional args.
2. **Parser bug (prefix form).** The same trained model also emits `ASK: {"question": "..."}` and `PROPOSE: {"plan": "..."}` for ~30% of its outputs (a habit picked up during GRPO training because the rubric rewards both forms equally). The original parser didn't recognize the prefix form at all. **Fix:** added `_parse_prefixed_call` with a `_PREFIX_TO_TOOL` mapping for `ASK / Q / QUESTION → ask_question`, `PROPOSE / PLAN → propose_plan`, `INFO / TASK_INFO → get_task_info`.
3. **Parser bug (commas in quotes).** `ask_question("What is X (e.g., birthday)?")` was being split on the comma inside the quoted string, truncating the question to `"What is X (e.g."`. **Fix:** wrote `_split_top_level_commas` that respects quotes, parens, brackets, and braces simultaneously.
4. **Prompt example contamination.** Our eval `SYSTEM_PROMPT` had `propose_plan(plan='{"stack": "python+fastapi", "scale": "1k users"}')` as an illustrative example. Qwen3-1.7B base **literally copied that plan verbatim for every scenario regardless of family** — we saw 50/50 event-planning tasks emit the software-stack plan. **Fix:** aligned the eval prompt char-for-character with the training prompt so the model has zero distribution shift.
5. **Conversational drift on Instruct models.** Qwen3-4B-Instruct would emit valid tool calls for the first 2-3 turns then drift to natural language ("Let me think about what date might work…"). **Fix:** modified `scripts/run_eval.py` to inject `RESPONSE FORMAT: Reply with ONE function call only, no other text.` into the initial user message and append a 2-line reminder to every observation reply.

A sixth issue, mostly mechanical: the env Space was rejecting concurrent eval clients with `CAPACITY_REACHED (8/8 sessions active)`. We bumped `max_concurrent_envs` from 8 → 64 in `server/app.py`, since each `ClarifyEnvironment` is in-memory with no shared state.

The reason this is worth a whole section: **without these fixes, "GRPO doesn't train this model" was the wrong conclusion — the right conclusion was "we couldn't measure what GRPO was doing."** Five compounding eval bugs is a classic RL-systems failure mode and easy to miss when every individual rubric component looks correct in isolation.

## 7. What worked

1. **`num_generations=4-8` instead of 2**: single biggest quality lever. Fixed the `0.000000` loss pathology we saw on the first Colab smoke run.
2. **Composable rubric over a single scalar**: the `Sequential(Gate, WeightedSum(...))` shape lets us debug exactly which axis the model is failing on at a per-rollout level.
3. **One env Space for all rollouts**: `max_concurrent_envs=64` + `SUPPORTS_CONCURRENT_SESSIONS=True` saved us from cloning the Space three times. The env is stateless across instances, so each parallel HF Job opens its own WS sessions against the same Space.
4. **`chat_template_kwargs={"enable_thinking": False}` mirrored at train AND eval time**: see section 5.4. Without this, every run wastes the token budget on `<think>` traces.
5. **The vLLM-in-HF-Job eval pipeline.** `scripts/launch_eval_job.sh` + `scripts/eval_with_vllm.py` host our own OpenAI-compatible vLLM server in a one-shot HF Job per checkpoint, because HF Inference Router doesn't serve fine-tuned community uploads. Each n=50 eval costs ~$0.13 and finishes in 2-7 min wall.

## 7b. What we'd change next time

The most important takeaway from the per-family table is that our reward shape was wrong for stronger bases — and Run 4 confirmed that the missing piece was the KL anchor, not the optimizer or the task. Beyond that:

1. **Family-aware required-keys reward.** Right now `FormatCheck` uses a single regex that's too tight (0% pass everywhere). A learned per-family schema check (or a softer "did all required keys appear in the propose payload" signal) would let the gate trip and the rest of the rubric carry weight.
2. **KL anchor for stronger bases — confirmed.** 1.7B is small enough that 1e-6 LR with no KL penalty pulled the model off its useful prior fast. Run 4 directly tested this — same model as Run 2 with `beta=0.2` and half the LR. **Result: event_planning recovered to 0.175 (beats 1.7B base 0.138), aggregate jumped 0.029 → 0.056, KL stayed bounded 0.005-0.010 throughout.** The trade-off was losing Run 2's meeting_scheduling peak (0.725 → 0.350). Next sweep: β ∈ {0.05, 0.10} with the same half-LR to find the sweet spot that keeps both the recovery *and* the peak.
3. **Curriculum over families.** All five families share a single batch. Trained 0.6B ended up only learning `event_planning` because that's the one whose scenarios it could partially solve — a curriculum that warms up on the easiest family then mixes harder ones in would likely close the medical/support-triage gap.
4. **Scale-aware hyperparams.** Our 0.6B and 1.7B configs were identical (`num_gen=8, lr=1e-6, max_steps≈300-400`). The honest reading of section 5 is that GRPO worked for 0.6B because the model had nothing to lose, and hurt 1.7B because it had something to lose. We had a Run 3 queued to test this at 4B scale (β=0, lr=1e-6, num_gen=2, max_completion_len=768 to fit a single A100-80GB). It sat in HF Jobs' SCHEDULING queue for 48 minutes on Account B's a100-large; with Run 4 already in hand we cancelled rather than push into the deadline. The natural next-step experiment — given Run 4 — would be **4B + β=0.2 + half-LR**, which is what we'd run on a stable h200 if we had another 24 hours.

## 7c. Training pipeline overhaul (Runs 5-7)

After the initial 4-run KL ablation, we did a deep analysis of *why* eval scores were consistently below the base model. We found **4 root causes** in the training pipeline that were silently undermining every run:

1. **Example contamination in the prompt.** The training `PROMPT` included `propose_plan(plan='{"start_time": "2pm", "duration": "30min"}')` as an illustration. These are *meeting-specific keys that don't match any family's required fields*. Run 5 logs confirmed the model was literally copying `start_time`/`duration` for event_planning tasks. FormatCheck failed, reward = 0.
2. **Reward misalignment on timeout.** When an episode timed out without `propose_plan`, `env.reward` retained the last shaping reward (+0.02 to +0.05), teaching the model "just keep asking questions" was better than submitting a plan that might score 0. We added `NO_PLAN_PENALTY = -0.1` and `PLAN_SUBMISSION_BONUS = +0.05`.
3. **Missing required-keys hint.** The reset observation told the model the task family but not the required keys. A 1.7B model cannot memorize which keys belong to which of 5 families from scratch. We added `Required plan fields: event_type, date, guest_count, venue` to the observation.
4. **Train/eval role mismatch.** Training used `user` role for the system prompt; eval used `system` role. We aligned both.

**Run 5** (beta=0.5, old pipeline) confirmed the problem: reward stuck at 0 on 8/10 steps. Canceled.

**Run 6** (beta=1.0, fixed pipeline) proved the fixes work: reward was non-zero from step 1 (0.12), peaked at 0.27, and every training step had `frac_reward_zero_std: 0`. Eval: avg_score=0.061, nearly matching 1.7B base (0.063 on same v5 prompts). But beta=1.0 was too conservative to *beat* the base.

**Run 7** (beta=0.3, lr=1e-6, 400 steps, fixed pipeline) is the current best attempt: training rewards of **0.48-0.73** at step 90/400. With a moderate KL anchor and the fixed pipeline preventing collapse, this run has the highest training reward of any run in the project. Eval pending.

## 8. What didn't work (and why we kept the lessons)

1. **Llama-3.x-Instruct, Qwen2.5-Instruct.** Both fail TRL's `add_response_schema` because their chat templates don't support tool-use schema injection. Stayed in the Qwen3 family.
2. **HF Inference Router for fine-tuned uploads.** Returns `model_not_supported` 400 — Router only serves provider-listed models. We host our own vLLM in a one-shot HF Job (`scripts/launch_eval_job.sh` → `scripts/eval_with_vllm.py`) per checkpoint. Cost per eval: $0.13 / 50 scenarios.
3. **`num_generations=2`.** Tutorials use this as a default. Don't. The variance is too low; advantages collapse to zero.
4. **Free-form rewards instead of a structured rubric.** We tried a single "did the agent do well" reward early in development. The model overfit to format compliance and ignored hallucination. The 5-component rubric forces it to balance.
5. **TRL pre-1.0 + `chat_template_kwargs`.** TRL versions <1.0 don't have the `chat_template_kwargs` config field — the older `GRPOConfig.__init__` rejects it as an unknown kwarg. Newer TRL versions also drop the eager `mergekit` import that conflicts irrecoverably with vLLM's pydantic pin. **Pin `trl[vllm]>=1.0` explicitly** in your launch script and defensively filter unsupported `GRPOConfig` kwargs by reflecting on `dataclasses.fields(GRPOConfig)` — this saved us when uv resolved an older TRL than expected.
6. **`vllm_ascend` shadow plugin on x86 hosts.** vllm 0.10+ ships a `vllm_ascend` plugin namespace stub that `importlib.util.find_spec` reports as installed even on plain CUDA. TRL's `is_vllm_ascend_available()` then triggers an import that fails because the actual `vllm-ascend` package only has cp39/310/311 wheels. **Monkey-patch `importlib.util.find_spec` and `transformers.utils.import_utils._is_package_available`** to hide it (and `llm_blender`, which transformers 5.x's API changes also break).
7. **Qwen3 default thinking mode.** Burns the full token budget on `<think>` traces during eval, never emitting a `TOOL:` line. Disable with `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` in the OpenAI client.

## 9. Reproducing

```bash
git clone https://github.com/anurag203/clarify-rl
cd clarify-rl
pip install -e .

# Run the env locally
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Smoke training (5 steps, no Hub push)
HF_TOKEN=hf_xxx SMOKE=1 ./scripts/launch_hf_job.sh Qwen/Qwen3-0.6B a100-large

# Real training across 3 HF accounts in parallel
HF_TOKEN_1=... HF_TOKEN_2=... HF_TOKEN_3=... ./scripts/launch_all.sh

# Eval a Hub checkpoint via vLLM-in-HF-Job
HF_TOKEN=... ./scripts/launch_eval_job.sh \
    --model agarwalanu3103/clarify-rl-grpo-qwen3-0-6b \
    --flavor a10g-small \
    --limit 50

# Plots after eval (one-shot)
HF_TOKEN=... ./scripts/refresh_all_plots.sh
```

Or open the [training notebook in Colab](https://colab.research.google.com/github/anurag203/clarify-rl/blob/main/training/train_grpo.ipynb).

## 10. Future work

- **Family-aware reward shape. ✅ DONE in Run 6-7.** Our 0% format pass rate everywhere showed the structural gate was the wrong gate for tool-call models. We added `REQUIRED_KEYS_BY_FAMILY` mapping to the training script and surfaced required fields in the observation. Combined with `NO_PLAN_PENALTY` and `PLAN_SUBMISSION_BONUS`, this transformed the reward landscape: Run 6 had non-zero rewards from step 1, and Run 7 reached rewards of 0.48-0.73 (vs 0.01 max in Run 4).
- **KL-penalized GRPO on stronger bases. ✅ ANSWERED.** Run 4 was exactly this experiment, and the answer is yes: `beta=0.2` restored `event_planning` (0.000 → 0.175, beats base) and recovered avg score (0.029 → 0.056). **Update:** Run 6 (beta=1.0, fixed pipeline) nearly matched the base (0.061 vs 0.063). Run 7 (beta=0.3, lr=1e-6, 400 steps) is training with the strongest reward signal yet.
- **β sweep. ✅ IN PROGRESS.** We now have data at β ∈ {0, 0.2, 0.3, 0.5, 1.0}, forming a 5-point ablation. Early results suggest β=0.3 with the fixed pipeline is the sweet spot: strong enough to prevent collapse, loose enough to let the model improve beyond the base.
- **Cross-family generalization.** Current task families are sampled at training time. Holding out a family entirely (e.g., train on 4, eval on the 5th) would test whether the asking-behavior generalizes or memorizes.
- **Hard scenarios.** Right now `eval_held_out.json` is 50/30/20% easy/medium/hard. A `super_hard` tier with adversarial vagueness ("do the thing") would test whether the trained models degrade gracefully or collapse.
- **Multi-turn ambiguity resolution.** Currently each question reveals one field. A more realistic env would have the user respond ambiguously sometimes, requiring follow-up questions.

## 11. Acknowledgments

Built on [Meta's OpenEnv](https://github.com/meta-pytorch/OpenEnv) and [Hugging Face TRL](https://github.com/huggingface/trl). The starter notebook was [TRL's `openenv_wordle_grpo.ipynb`](https://github.com/huggingface/trl). Thanks to the Meta + HF teams for shipping production-grade RL environments and the GRPO-with-vLLM-colocate path that made same-day parallel runs feasible.
