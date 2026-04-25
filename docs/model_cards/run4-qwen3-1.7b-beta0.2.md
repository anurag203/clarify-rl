---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
base_model_relation: finetune
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- text-generation
- conversational
- reinforcement-learning
- rlhf
- rl
- grpo
- trl
- kl-anchor
- agentic
- tool-use
- clarifying-questions
- ask-vs-guess
datasets:
- agarwalanu3103/clarify-rl
model-index:
- name: clarify-rl-run4-qwen3-1.7b-beta0.2
  results:
  - task:
      type: text-generation
      name: ClarifyRL multi-turn ask-or-guess
    dataset:
      type: held_out_scenarios
      name: ClarifyRL eval (50 held-out scenarios, n=50 v4)
    metrics:
    - type: avg_score
      value: 0.0560
      name: avg_score (μ over 4 families × 50 scenarios)
    - type: completion_rate
      value: 0.14
      name: completion_rate
    - type: family_event_planning_mean
      value: 0.175
      name: event_planning μ
    - type: family_meeting_scheduling_mean
      value: 0.064
      name: meeting_scheduling μ
---

# ClarifyRL — Run 4 — Qwen3-1.7B GRPO (β=0.2 KL anchor)

> **The hero checkpoint of the ClarifyRL hackathon submission.** This is
> Qwen3-1.7B trained with TRL GRPO and an explicit **KL anchor at β=0.2**
> against the frozen base, which is the single change that turned a
> capability collapse (Run 2, β=0) into a measurable improvement on the
> held-out eval *while preserving breadth across families*.

- **Live demo (replay + CPU live chat):** [`anurag203/clarify-rl-demo`](https://huggingface.co/spaces/anurag203/clarify-rl-demo)
- **Code (training, eval, plots):** [`github.com/anurag203/clarify-rl`](https://github.com/anurag203/clarify-rl)
- **W&B dashboard (all 3 runs live):** [`anuragagarwal203-cisco/clarify-rl`](https://wandb.ai/anuragagarwal203-cisco/clarify-rl)
- **Environment Space (OpenEnv MCP server):** [`agarwalanu3103/clarify-rl`](https://huggingface.co/spaces/agarwalanu3103/clarify-rl)
- **Hackathon write-up (blog):** [`docs/blog.md`](https://github.com/anurag203/clarify-rl/blob/main/docs/blog.md)

## Model summary

| Field | Value |
| --- | --- |
| Base model | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) (NOT the instruct tune) |
| Algorithm | TRL GRPO (Group Relative Policy Optimization) |
| KL anchor (β) | **0.2** (vs Run 2 = 0.0) |
| Learning rate | 5e-7, cosine decay → 1.7e-9 |
| Steps | 300 |
| Wall time | 78.2 min on a single A100 (HF Jobs `a100-large`) |
| Optimizer | adamw_torch_fused |
| Generations / step | 4 (with vLLM sampler) |
| Max completion length | 768 tokens |
| Reward stack | `OutputCorrectnessRubric` (0.6) + `EfficiencyRubric` (0.2) + `FormatCheckRubric` (0.2) over 5 task families |
| Cost | **~$1.80** of HF Jobs credit |

The training and eval are fully reproducible from
[`training/train_grpo.py`](https://github.com/anurag203/clarify-rl/blob/main/training/train_grpo.py)
with `BETA=0.2 LEARNING_RATE=5e-7 NUM_STEPS=300`.

## What this model does

ClarifyRL trains a small LLM to **ask before it acts**. Given a deliberately
under-specified user request (e.g. *"set up a celebration"*), the agent has
6 question budget and must choose between:

- `ask_question(question)` — pull a single missing field from the user simulator
- `propose_plan(plan)` — emit the final structured JSON plan
- `get_task_info()` — re-read the brief

Reward depends on (a) format correctness, (b) field overlap of the final
plan vs the hidden ground-truth profile, and (c) efficiency (fewer
questions for higher-quality plans). 5 task families: `event_planning`,
`medical_intake`, `meeting_scheduling`, `support_triage`,
`coding`. Held-out eval uses 50 scenarios with the same 4-family coverage
as training.

## The headline result — KL anchor turns the model around

Same base, same data, same step count. The only difference is β.

| Eval metric (n=50, held-out) | 1.7B base | Run 2 (β=0) | **Run 4 (β=0.2) ✅** |
| --- | ---: | ---: | ---: |
| avg_score (μ across 4 families) | 0.067 | 0.029 ↓ | **0.056** ✅ |
| completion_rate | 0.18 | 0.06 | **0.14** |
| event_planning μ | 0.138 | **0.000 ❌** | **0.175 ✅✅** |
| event_planning max | 0.522 | 0.000 | 0.510 |
| meeting_scheduling μ | 0.153 | 0.130 | 0.064 ↓ |
| meeting_scheduling max | 0.500 | **0.725** | 0.350 |

Three observations the hackathon write-up leans on:

1. **GRPO without anchor causes capability collapse.** Run 2 (β=0) drove
   `event_planning` from 0.138 → **0.000** mean while inflating one peak
   in `meeting_scheduling`. The model traded breadth for an
   exploit-and-overfit that the held-out eval flags immediately.
2. **GRPO with KL anchor cleanly improves the protected family.**
   Run 4 (β=0.2, lr=5e-7) on the same model recovered avg_score to
   **0.056** AND **beat the base on `event_planning`** (0.138 → **0.175**).
   The anchor literally fixed Run 2's regression *without* extra data.
3. **The cost is peak capability.** Run 4 dropped
   `meeting_scheduling` max from 0.725 (Run 2's gem) to 0.350. KL prevents
   the kind of extreme specialization Run 2 leaned on. That's the
   trade-off, stated honestly.

For the same-base delta plot
([`plots/06_same_base_delta.png`](https://raw.githubusercontent.com/anurag203/clarify-rl/main/plots/06_same_base_delta.png)),
see the full blog at
[`docs/blog.md`](https://github.com/anurag203/clarify-rl/blob/main/docs/blog.md).

## Intended use

- **Research / hackathon** — reproduce the KL-anchor ablation on a small
  reasoner.
- **Demo / education** — illustrate that a 1.7B param model can be steered
  toward an *ask-first* policy with a tiny RL budget when KL is enforced.
- **Drop-in replacement for `Qwen/Qwen3-1.7B`** in agentic, multi-turn,
  tool-using settings where the agent should clarify ambiguous requests
  rather than hallucinate fields.

## Out-of-scope use

- General chat assistant. The reward shaping is highly specific to the
  ClarifyRL ask-or-guess setting; do not expect calibration or RLHF-style
  helpfulness on open-ended prompts.
- Production / safety-critical / medical / legal. The model has **no** RLHF
  safety alignment; the `medical_intake` family in the eval is a *task
  scaffold*, not a clinical reasoning task.
- Anything outside English.

## How to use it

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo = "anurag203/clarify-rl-run4-qwen3-1.7b-beta0.2"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    repo,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

SYSTEM = (
    "You are an agent that must complete a task by asking clarifying "
    "questions and then proposing a structured JSON plan. Tools: "
    "ask_question(question), propose_plan(plan), get_task_info(). "
    "You have a 6-question budget. Output ONLY one tool call per turn."
)
USER = "Set up a celebration."

prompt = tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
    tokenize=False,
    add_generation_prompt=True,
    chat_template_kwargs={"enable_thinking": False},
)
out = mdl.generate(
    **tok(prompt, return_tensors="pt").to(mdl.device),
    max_new_tokens=120,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tok.eos_token_id,
)
print(tok.decode(out[0][tok(prompt, return_tensors="pt")["input_ids"].shape[1]:],
                 skip_special_tokens=True))
# expected first turn: ask_question("What is the date of the celebration?")
```

For the full multi-turn agent loop with tool parsing, see
[`scripts/eval_agent.py`](https://github.com/anurag203/clarify-rl/blob/main/scripts/eval_agent.py)
or the live tab of the
[`anurag203/clarify-rl-demo`](https://huggingface.co/spaces/anurag203/clarify-rl-demo) Space.

## Training data

Held-out eval uses
[`scenarios/eval_held_out.json`](https://github.com/anurag203/clarify-rl/blob/main/scenarios/eval_held_out.json)
(50 scenarios across 4 families). The training data is procedurally generated
inside the OpenEnv MCP environment and is *not* a static dump — the user
simulator answers each `ask_question` with a sampled ground-truth profile,
so the same scenario can yield different conversations across rollouts.

See
[`scenarios/`](https://github.com/anurag203/clarify-rl/tree/main/scenarios)
and
[`docs/05-scenario-design.md`](https://github.com/anurag203/clarify-rl/blob/main/docs/05-scenario-design.md)
for the full scenario taxonomy.

## Training procedure

| Hyperparameter | Value |
| --- | --- |
| Algorithm | GRPO (TRL ≥ 1.0) |
| `beta` (KL coefficient) | **0.2** |
| Learning rate | 5e-7 (cosine to ~1.7e-9) |
| Total steps | 300 |
| Generations / step | 4 |
| Group size | 4 |
| Max completion length | 768 |
| Sampler | vLLM, `temperature=1.0`, `top_p=1.0` |
| Reference model | Frozen `Qwen/Qwen3-1.7B` (loaded once, never updated) |
| Hardware | 1× A100 80GB (HF Jobs `a100-large`) |
| Wall time | 78.2 min |
| Final reward | 0.0050 mean / 0.114 max |
| KL stayed bounded | 0.005 – 0.010 throughout (the anchor did its job) |
| LR cosine-decayed | yes, to 1.7e-9 (verified in W&B) |

The full 300-entry `log_history.json` is committed at
[`outputs/run_artifacts/1.7B-KL/`](https://github.com/anurag203/clarify-rl/tree/main/outputs/run_artifacts/1.7B-KL)
and is mirrored to
[`wandb.ai/anuragagarwal203-cisco/clarify-rl`](https://wandb.ai/anuragagarwal203-cisco/clarify-rl)
under run name `run4-1p7b-kl-anchor`.

## Evaluation

| Family | 1.7B base μ | Run 2 (β=0) μ | **Run 4 (β=0.2) μ** |
| --- | ---: | ---: | ---: |
| event_planning | 0.138 | 0.000 | **0.175** |
| meeting_scheduling | 0.153 | 0.130 | 0.064 |
| medical_intake | 0.000 | 0.000 | 0.000 |
| support_triage | 0.000 | 0.000 | 0.000 |
| **avg_score (μ)** | **0.067** | **0.029** | **0.056** |
| **completion_rate** | 18% | 6% | **14%** |

Eval methodology: 50 held-out scenarios per family, 5 families, scored by
the same rubric stack used at training time. Eval runs are reproducible
from
[`scripts/run_eval.py`](https://github.com/anurag203/clarify-rl/blob/main/scripts/run_eval.py)
with the JSON profile committed at
[`outputs/run_artifacts/1.7B-KL/evals/`](https://github.com/anurag203/clarify-rl/tree/main/outputs/run_artifacts/1.7B-KL).

## Limitations

- 1.7B parameters — not a strong reasoner. Even Run 4's mean (0.056) is
  below 4B base's mean (0.145). RL helped, but parameter count still wins.
- Format-pass rate is **0%** across all evaluated runs because the eval
  rubric expects strict JSON keys; even Run 4 occasionally proposes plans
  that don't quite match the schema. We deliberately leave the format
  rubric strict because format failure is a common failure mode of small
  RLHF'd models.
- `medical_intake` and `support_triage` remain at 0 across all 1.7B and
  0.6B variants — these families need either a stronger base or scenario
  redesign. Logged as future work in
  [`docs/blog.md` §7b](https://github.com/anurag203/clarify-rl/blob/main/docs/blog.md).
- 4B GRPO (Run 3) was canceled in HF Jobs queue at 48 min — the anchor
  finding has not yet been confirmed at 4B scale.

## Citation

If you build on this, please cite the GitHub repository and the W&B
project:

```bibtex
@misc{agarwal2026clarifyrl,
  author       = {Agarwal, Anurag},
  title        = {ClarifyRL: Teaching small LLMs to ask before they act,
                  with KL-anchored GRPO},
  year         = {2026},
  howpublished = {\url{https://github.com/anurag203/clarify-rl}},
  note         = {Hackathon submission, Apr 26 2026.}
}
```

Built on top of [TRL](https://github.com/huggingface/trl) (GRPO trainer),
[Qwen3](https://huggingface.co/Qwen) (base model), and the [OpenEnv MCP
environment](https://huggingface.co/spaces/agarwalanu3103/clarify-rl).

## License

Apache-2.0 — same as the upstream Qwen3-1.7B base.
