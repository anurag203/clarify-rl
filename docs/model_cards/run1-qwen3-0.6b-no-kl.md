---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
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
- agentic
- tool-use
- clarifying-questions
- weak-base
- ablation
model-index:
- name: clarify-rl-run1-qwen3-0.6b-no-kl
  results:
  - task:
      type: text-generation
      name: ClarifyRL multi-turn ask-or-guess
    dataset:
      type: held_out_scenarios
      name: ClarifyRL eval (50 held-out scenarios, n=50 v4)
    metrics:
    - type: avg_score
      value: 0.0076
      name: avg_score (μ over 4 families × 50 scenarios)
    - type: completion_rate
      value: 0.02
      name: completion_rate
    - type: family_event_planning_max
      value: 0.382
      name: event_planning max (new capability vs 0.0 in 0.6B base)
---

# ClarifyRL — Run 1 — Qwen3-0.6B GRPO (β=0, weak-base baseline)

> **The "RL unlocks weak bases" data point.** Qwen3-0.6B base scores
> *exactly* 0.0 on every family in our held-out eval — it cannot finish
> the multi-turn ask-or-guess task at all. After 300 steps of GRPO with
> the same reward stack we used at 1.7B and 4B, the same 0.6B model goes
> from 0 → **0.382 max** on `event_planning`. That's not a great number,
> but it's the difference between *zero capability* and *some capability*.

This is one of the three trained checkpoints in the ClarifyRL hackathon
submission. The full story sits at the parent project; this card is the
0.6B leg of the model-size ablation.

- **Live demo (replay viewer + CPU live chat):** [`anurag203/clarify-rl-demo`](https://huggingface.co/spaces/anurag203/clarify-rl-demo)
- **Code (training, eval, plots):** [`github.com/anurag203/clarify-rl`](https://github.com/anurag203/clarify-rl)
- **W&B dashboard:** [`anuragagarwal203-cisco/clarify-rl`](https://wandb.ai/anuragagarwal203-cisco/clarify-rl) (run name: `run1-0p6b-no-kl`)
- **Hackathon write-up:** [`docs/blog.md`](https://github.com/anurag203/clarify-rl/blob/main/docs/blog.md)

## Model summary

| Field | Value |
| --- | --- |
| Base model | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) |
| Algorithm | TRL GRPO (Group Relative Policy Optimization) |
| KL anchor (β) | **0.0** |
| Learning rate | 1e-6 |
| Steps | 300 |
| Wall time | ~50 min on a single A100 (HF Jobs `a100-large`) |
| Reward stack | `OutputCorrectnessRubric` + `EfficiencyRubric` + `FormatCheckRubric` |
| Cost | ~$1.08 of HF Jobs credit |

## What this checkpoint demonstrates

| Family | 0.6B base μ | **Run 1 μ** | 0.6B base max | **Run 1 max** |
| --- | ---: | ---: | ---: | ---: |
| event_planning | 0.000 | **0.032 ↑** | 0.000 | **0.382 ↑↑** |
| meeting_scheduling | 0.000 | 0.000 | 0.000 | 0.000 |
| medical_intake | 0.000 | 0.000 | 0.000 | 0.000 |
| support_triage | 0.000 | 0.000 | 0.000 | 0.000 |
| **avg_score (μ)** | **0.000** | **0.008** | — | — |
| **completion_rate** | 0% | **2%** | — | — |

The mean is almost rounding error, but the *maximum* on `event_planning`
moved from 0.0 → 0.382. That's GRPO finding a single rollout where 0.6B
managed to clarify the ambiguous request, ask the right question, and
produce a syntactically valid plan — something the base model never
succeeded at across 50 held-out scenarios.

This is the canonical "RL unlocks new capability in weak bases" data
point in the
[`docs/blog.md`](https://github.com/anurag203/clarify-rl/blob/main/docs/blog.md)
write-up. The KL-anchor finding is at 1.7B (Run 2 vs
[Run 4](https://huggingface.co/anurag203/clarify-rl-run4-qwen3-1.7b-beta0.2)),
but Run 1 is the size-ablation anchor: the same recipe scales monotonically
in capability with parameter count.

## Intended use

- **Ablation / research only.** Do not deploy this for any user-facing
  task. Mean score on the held-out eval is 0.008.
- Reproducing the parameter-size ablation at the bottom of the
  ClarifyRL blog (0.6B → 1.7B → 4B base).
- Investigating GRPO dynamics on a tiny base.

## Out-of-scope use

- Production / general chat / safety-critical anything. This is a 600M
  param base trained on a tiny budget.
- Any task outside the 5 ClarifyRL families.

## How to use it

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo = "anurag203/clarify-rl-run1-qwen3-0.6b-no-kl"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    repo, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True,
)
# See scripts/eval_agent.py for the full multi-turn agent driver.
```

## Training procedure

Reproducible from
[`training/train_grpo.py`](https://github.com/anurag203/clarify-rl/blob/main/training/train_grpo.py)
with:

```bash
BETA=0 LEARNING_RATE=1e-6 NUM_STEPS=300 NUM_GENERATIONS=4 \
MAX_COMPLETION_LEN=768 BASE_MODEL=Qwen/Qwen3-0.6B \
python training/train_grpo.py
```

The full `log_history.json` is committed at
[`outputs/run_artifacts/0.6B/`](https://github.com/anurag203/clarify-rl/tree/main/outputs/run_artifacts/0.6B).

## Limitations

- 600M params on a 5-family multi-turn agentic task is *under-powered*.
  Even with RL, format compliance is 0% and only 2% of held-out
  scenarios complete.
- 3 of 4 families remain at 0 across both base and Run 1. Mean is driven
  entirely by `event_planning`.
- No KL anchor. Run 4 (1.7B + β=0.2) is the right reference for "RL
  done well."

## Citation

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

## License

Apache-2.0 — same as the upstream Qwen3-0.6B base.
