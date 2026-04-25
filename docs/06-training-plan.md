# 06 — Training Plan

GRPO + Unsloth on Qwen2.5-1.5B-Instruct. **Primary compute path: HF Jobs `t4-small`** (uses on-site hackathon credits). **Backup path: free Colab T4**.

## Starter Template (don't reinvent)

We fork the official TRL OpenEnv starter notebook rather than writing GRPO setup from scratch. Choose ONE:

| Template | Source | Why we picked it |
|----------|--------|------------------|
| `openenv_wordle_grpo.ipynb` | [trl/examples/notebooks](https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_wordle_grpo.ipynb) | Multi-turn rollout w/ env, closest to our episode shape |
| `openenv_sudoku_grpo.ipynb` | [trl/examples/notebooks](https://github.com/huggingface/trl/blob/main/examples/notebooks/openenv_sudoku_grpo.ipynb) | Cleaner reward func plumbing |
| `unsloth_2048.ipynb` | [meta-pytorch/OpenEnv tutorial/examples](https://github.com/meta-pytorch/OpenEnv/blob/main/tutorial/examples/unsloth_2048.ipynb) | Unsloth-specific reference |

**Decision: start from `openenv_wordle_grpo.ipynb`** (multi-turn shape matches our `ask → answer → ... → propose_plan` trajectory most closely). Replace its env client + reward fn with our `ClarifyClient` and `env_reward_fn`.

## Why these choices

| Choice | Why |
|--------|-----|
| **Qwen2.5-1.5B-Instruct** | Small enough for T4 with Unsloth 4-bit (~6GB VRAM); strong instruction-following baseline |
| **Unsloth** | 2-3x faster training, 50% less memory than vanilla Transformers |
| **TRL GRPO** | Group-Relative Policy Optimization — designed for LLMs, no value model needed |
| **4-bit QLoRA** | Fits in T4 16GB VRAM with batch=4 prompts × 4 completions |
| **HF Jobs `t4-small`** | Official hackathon compute; on-site credits cover full run; survives session timeouts |
| **Colab free T4** | $0 fallback; ≥1.5h per session; sufficient for 500-800 GRPO steps |

## Compute Paths

### Path A — HF Jobs (preferred once on-site credits arrive)

```bash
hf auth login   # uses on-site-issued HF token
hf jobs hardware   # confirms t4-small available
hf jobs uv run --with trl --with unsloth --flavor t4-small \
  -s HF_TOKEN -- training/train_grpo.py
```

### Path B — Colab free T4 (fallback / smoke test)

Open `training/train_grpo.ipynb` in Colab → Runtime → GPU (T4) → Run All. Same script, run interactively

## Hyperparameters

```python
TRAINING_CONFIG = {
    # Model
    "model_name": "unsloth/Qwen2.5-1.5B-Instruct",
    "load_in_4bit": True,
    "max_seq_length": 4096,

    # LoRA
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],

    # GRPO
    "num_generations": 4,                 # completions per prompt
    "per_device_train_batch_size": 4,    # prompts per step
    "gradient_accumulation_steps": 1,
    "learning_rate": 5e-6,
    "warmup_steps": 10,
    "max_steps": 600,
    "max_prompt_length": 1024,
    "max_completion_length": 512,
    "temperature": 0.9,
    "top_p": 0.95,
    "beta": 0.04,                         # KL penalty coefficient
    "logging_steps": 5,
    "save_steps": 100,

    # Reproducibility
    "seed": 42,
}
```

## Rollout Function (the heart of training)

```python
def rollout(prompt: str, env: ClarifyClient) -> tuple[str, float, dict]:
    """Run a full episode: returns (full_completion_str, reward, metadata)."""
    obs = env.reset(task_id=random.choice(["easy","medium","hard"]))
    request = json.loads(obs.result)["request"]

    conversation = [
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"user","content": f"Request: {request}\n\nAvailable tools: ask_question(question), propose_plan(plan_json), get_task_info()"},
    ]
    completion_text = ""
    cumulative_reward = 0.0

    for step in range(MAX_STEPS_PER_EPISODE):  # =10
        # Generate next agent action
        out = model.generate(conversation, max_new_tokens=200)
        action_text = out.split("Action:")[-1].strip()
        completion_text += action_text + "\n"
        # Parse action
        tool_name, args = parse_action(action_text)
        # Step env
        step_obs = env.call_tool(tool_name, **args)
        cumulative_reward += step_obs.reward or 0.0
        # Append observation to conversation
        conversation.append({"role":"assistant","content":action_text})
        conversation.append({"role":"user","content": f"Observation: {step_obs.result}"})
        if step_obs.done:
            break

    return completion_text, cumulative_reward, {"steps": step+1}
```

## Reward Function for GRPO

GRPO needs `reward_funcs: List[Callable[[completions], List[float]]]`.

```python
def env_reward_fn(prompts: list, completions: list, **kwargs) -> list[float]:
    """For each completion, run rollout and return cumulative reward."""
    with ClarifyClient(base_url=ENV_URL) as env:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            # Re-execute the trajectory in env to get true reward
            r = replay_in_env(prompt, completion, env)
            rewards.append(r)
    return rewards
```

For efficiency, we'll embed the env in-process (Python module import) rather than HTTP during training; the HTTP server is for the demo/judging.

## System Prompt (frozen during training)

```text
You are a helpful assistant that books and plans things for users.
When you receive a request, you may not have all the information needed.
You can:
1. ASK clarifying questions using the ask_question(question) tool (max 6 total)
2. PROPOSE a final plan using propose_plan(plan_json) when you have enough info
3. GET the task description again using get_task_info()

Always emit ONE action per turn in this format:
Action: tool_name(arg="value")

Be efficient: ask only what you NEED, then propose a plan.
Do NOT include preferences in the plan that you weren't told about.
```

## Training Curriculum

Mix 3 difficulty levels in each batch:
- 50% easy
- 30% medium
- 20% hard

This keeps gradient signal strong (easy samples succeed) while pushing on hard cases.

## Eval Methodology

```python
def evaluate(model, n_scenarios=100):
    results = {"easy":[], "medium":[], "hard":[]}
    for seed in range(10000, 10000+n_scenarios):
        scenario = load_held_out(seed)
        result = rollout_with_model(model, scenario)
        results[scenario["task_id"]].append(result)
    return aggregate(results)
```

Reports:
- Aggregate plan_satisfaction (= average final rubric score)
- Per-difficulty plan_satisfaction
- Per-task-type plan_satisfaction
- Per-component breakdown (field_match, info_gain, efficiency, hallucination)
- Average questions asked per episode (efficiency metric)
- % episodes with ≥0.7 score (success threshold)

## Plot Generation

After eval, `scripts/make_plots.py` generates:

| Plot | Content | Caption |
|------|---------|---------|
| `reward_curve.png` | x=training step, y=mean episode reward (rolling 20-step window) | "Mean episode reward during GRPO training: 0.27 → 0.84 over 600 steps." |
| `loss_curve.png` | x=training step, y=GRPO policy loss | "GRPO policy loss decreasing as policy improves." |
| `per_task_bars.png` | grouped bars: baseline vs trained per task type | "Trained model dominates baseline across all 5 task types." |
| `per_component.png` | per-component bars: baseline vs trained | "Field-match jumps from 0.20→0.92; hallucination penalty from 0.10→0.97." |
| `eval_dist.png` | histogram of per-episode rewards (baseline vs trained) | "Distribution of rewards on 100 held-out scenarios shifts right after training." |

All plots: matplotlib, dpi=120, with axis labels, title, legend, savefig(`plots/X.png`).

## Compute Budget

| Phase | Time on T4 | Notes |
|-------|-----------|-------|
| Model load + LoRA setup | 3 min | |
| 100 GRPO steps (smoke test) | 15 min | verify reward curve trends up |
| 600 GRPO steps (full run) | ~90 min | within Colab free session limit |
| Eval (100 scenarios × 2 models) | 15 min | sequential |
| **Total** | **~2h** | well within budget |

If reward curve flat at step 100 → stop, debug rubric/scenarios, retry. Don't burn 90min on a broken run.

## Sample-Inspection Cadence (anti reward-hacking, per Help Guide §8 + §15)

Every **50 GRPO steps**, dump 4 random rollouts to `logs/samples_step_{N}.jsonl` and eyeball:

- Are questions getting more targeted (vs filler)?
- Is the plan JSON well-formed?
- Any sign of reward hacking (e.g. always asking 0 Qs and submitting empty plan, or asking the same Q 6 times)?
- Is the assistant ever inventing values it was never told?

If reward is rising but generations look degenerate → **stop training**, tighten the offending rubric, restart. Watching only the scalar reward curve is the #1 way to fool yourself.

## Saving the Trained Model (LoRA gotcha, per Help Guide §16)

**DO NOT** load the 4-bit base, upcast to 16-bit, then merge LoRA naïvely — that path damages quality. Use one of these instead:

```python
# Preferred: keep adapters separate, push to Hub
model.save_pretrained("clarify-rl-lora")
model.push_to_hub("anurag203/clarify-rl-lora")

# OR: use Unsloth's official merged-save path
model.save_pretrained_merged("clarify-rl-merged", tokenizer, save_method="merged_16bit")
```

Immediately after saving, run a 5-prompt smoke eval against the saved artifact — don't discover a broken save Sunday afternoon.

## Sanity Checks (run before full training)

1. Random policy on env → reward ~0.20 average ✅
2. Always-propose-with-empty-plan → format gate triggers, reward = 0 ✅
3. Ask 6 random Qs then propose → reward ~0.30 ✅
4. Oracle policy (asks for every critical field, then proposes correct plan) → reward ~0.95 ✅

If any of these fail, the env is broken — fix before training.

## Reproducibility

- Random seeds fixed at every layer (rng, torch, numpy, transformers)
- Held-out eval scenarios stored as `scenarios/eval_held_out.json` (committed)
- Training logs include git commit hash + config hash
- Final LoRA adapter pushed to HF Hub for re-evaluation
