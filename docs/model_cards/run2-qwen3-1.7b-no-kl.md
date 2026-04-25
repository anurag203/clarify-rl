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
- no-kl-anchor
- agentic
- tool-use
- clarifying-questions
- ablation
- regression-checkpoint
model-index:
- name: clarify-rl-run2-qwen3-1.7b-no-kl
  results:
  - task:
      type: text-generation
      name: ClarifyRL multi-turn ask-or-guess
    dataset:
      type: held_out_scenarios
      name: ClarifyRL eval (50 held-out scenarios, n=50 v4)
    metrics:
    - type: avg_score
      value: 0.0286
      name: avg_score (μ over 4 families × 50 scenarios)
    - type: completion_rate
      value: 0.06
      name: completion_rate
    - type: family_event_planning_mean
      value: 0.000
      name: event_planning μ (regression — was 0.138 in base)
    - type: family_meeting_scheduling_max
      value: 0.725
      name: meeting_scheduling max (the lone peak)
---

# ClarifyRL — Run 2 — Qwen3-1.7B GRPO (β=0, no KL anchor) — *the regression checkpoint*

> **This is the negative-result checkpoint.** Run 2 is the same recipe as
> [Run 4](https://huggingface.co/anurag203/clarify-rl-run4-qwen3-1.7b-beta0.2)
> with the *only* difference being `beta=0` (no KL anchor against the
> frozen base). It demonstrates **capability collapse**: the model's
> mean score dropped on 3 of 4 families while it found one peak in
> `meeting_scheduling`. We publish this checkpoint deliberately as the
> "before" half of the hackathon ablation.

If you want the working model, use
[`anurag203/clarify-rl-run4-qwen3-1.7b-beta0.2`](https://huggingface.co/anurag203/clarify-rl-run4-qwen3-1.7b-beta0.2)
instead.

- **Live demo (replay + CPU live chat):** [`anurag203/clarify-rl-demo`](https://huggingface.co/spaces/anurag203/clarify-rl-demo) — has a Run-2-vs-Run-4 side-by-side tab
- **Code (training, eval, plots):** [`github.com/anurag203/clarify-rl`](https://github.com/anurag203/clarify-rl)
- **W&B dashboard:** [`anuragagarwal203-cisco/clarify-rl`](https://wandb.ai/anuragagarwal203-cisco/clarify-rl) (run name: `run2-1p7b-no-kl`)
- **Hackathon write-up:** [`docs/blog.md`](https://github.com/anurag203/clarify-rl/blob/main/docs/blog.md)

## Why publish a regression?

Hackathon evidence works *both ways*. Run 2 is half of a
counter-factual: same base, same data, same step count, only β
changes. By publishing both the regression (β=0, this card) and the
recovery (β=0.2,
[Run 4](https://huggingface.co/anurag203/clarify-rl-run4-qwen3-1.7b-beta0.2)),
we let judges and other researchers verify the central thesis end-to-end
*on the actual weights*, not just on plots.

| Eval metric (n=50, held-out) | 1.7B base | **Run 2 (β=0) ← this checkpoint** | Run 4 (β=0.2) |
| --- | ---: | ---: | ---: |
| avg_score (μ across 4 families) | 0.067 | **0.029 ↓** | 0.056 ✅ |
| completion_rate | 18% | **6% ↓** | 14% |
| event_planning μ | 0.138 | **0.000 ❌** | 0.175 ✅ |
| meeting_scheduling μ | 0.153 | 0.130 | 0.064 |
| meeting_scheduling max | 0.500 | **0.725 ↑** | 0.350 |

**Read the table this way:** without a KL anchor, GRPO traded *broad
competence* for *one extreme peak* in `meeting_scheduling`. The peak is
real (max 0.725 is the highest across all evaluated 1.7B variants), but
the mean across families collapsed.

## Model summary

| Field | Value |
| --- | --- |
| Base model | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) |
| Algorithm | TRL GRPO (Group Relative Policy Optimization) |
| KL anchor (β) | **0.0** (intentionally — this is the "no anchor" arm of the ablation) |
| Learning rate | 1e-6 (vs Run 4's 5e-7) |
| Steps | 300 |
| Wall time | ~70 min on a single A100 (HF Jobs `a100-large`) |
| Generations / step | 4 |
| Reward stack | `OutputCorrectnessRubric` (0.6) + `EfficiencyRubric` (0.2) + `FormatCheckRubric` (0.2) |
| Cost | ~$2.21 of HF Jobs credit |

## Where the weights live

This `anurag203/*` repo hosts the **rich card / metadata only**. The actual
300-step Run 2 weights are checkpointed at
[`agarwalanu3103/clarify-rl-grpo-qwen3-1-7b`](https://huggingface.co/agarwalanu3103/clarify-rl-grpo-qwen3-1-7b)
on the training account. A unified-namespace mirror is in flight; in the
meantime download from the upstream repo directly:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo = "agarwalanu3103/clarify-rl-grpo-qwen3-1-7b"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    repo, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True,
)
# … same agent loop as Run 4. See scripts/eval_agent.py for the full driver.
```

## Intended use

- **Ablation comparison only.** The model exists to be compared against
  Run 4 (β=0.2) on the same eval; that's the entire reason it is public.
- Reproducing the
  [`docs/blog.md`](https://github.com/anurag203/clarify-rl/blob/main/docs/blog.md)
  KL-anchor finding from raw weights.
- Diagnosing what *capability collapse* looks like in a small RL'd
  reasoner.

## Out-of-scope use

- Anything resembling production. This is a deliberately broken
  checkpoint, kept for science.
- Anything that depends on broad multi-family competence — see the
  per-family table.

## Training procedure

Same env / reward / scaffolding as Run 4. The reproducible command is:

```bash
BETA=0 LEARNING_RATE=1e-6 NUM_STEPS=300 NUM_GENERATIONS=4 \
MAX_COMPLETION_LEN=768 \
python training/train_grpo.py
```

See
[`training/train_grpo.py`](https://github.com/anurag203/clarify-rl/blob/main/training/train_grpo.py).
The 300-step `log_history.json` is committed at
[`outputs/run_artifacts/1.7B-noKL/`](https://github.com/anurag203/clarify-rl/tree/main/outputs/run_artifacts/1.7B-noKL).

## Evaluation

| Family | 1.7B base μ | **Run 2 (β=0) μ** | Run 4 (β=0.2) μ |
| --- | ---: | ---: | ---: |
| event_planning | 0.138 | **0.000 ❌** | 0.175 ✅ |
| meeting_scheduling | 0.153 | **0.130** | 0.064 |
| medical_intake | 0.000 | 0.000 | 0.000 |
| support_triage | 0.000 | 0.000 | 0.000 |
| **avg_score (μ)** | **0.067** | **0.029** | **0.056** |
| **completion_rate** | 18% | **6%** | 14% |
| **format_pass_rate** | 0% | 0% | 0% |

## Limitations

- This model is **intentionally** worse than the base on average.
  Don't deploy it.
- See the Run 4 model card for the full discussion; everything in the
  "Limitations" section there applies more strongly here.

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

Apache-2.0 — same as the upstream Qwen3-1.7B base.
