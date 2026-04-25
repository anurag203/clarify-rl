# Submission Checklist — ClarifyRL

> Last validated: 2026-04-26 04:35 IST. All P0 auto-validator gates **GREEN** from a logged-out browser. Submission is locked-in for the 5:00 PM IST deadline.

## Auto-validator gates (Discord rules)

| Gate | Status | Evidence |
|---|---|---|
| Public HF Space, accessible logged-out | ✅ | `https://huggingface.co/spaces/agarwalanu3103/clarify-rl` returns HTTP 200; `runtime.stage = RUNNING` |
| Space exposes a valid `openenv.yaml` | ✅ | `…/raw/main/openenv.yaml` returns HTTP 200, valid `spec_version: 1` schema |
| Env serves `/health` and `/reset` | ✅ | `/health` → `{"status":"healthy"}`; `/reset` → valid `CallToolObservation` with task data |
| Public training script (Colab-runnable) | ✅ | `https://colab.research.google.com/github/anurag203/clarify-rl/blob/main/training/train_grpo.ipynb` returns HTTP 200; raw notebook on GitHub returns HTTP 200, 38 KB |
| README links to: HF Space, Colab, GitHub repo, blog/video | ✅ | All 5 deliverable links present in README badge stack + asset table |
| Training evidence as committed image files | ✅ | 6 PNGs committed under `plots/` and rendering on the env Space (each returns HTTP 200 with proper file size) |
| Plots embedded with `![]()` in README | ✅ | 12 image embeds in env Space README; all resolve via `/resolve/main/plots/...` |
| Public GitHub repo linked from README | ✅ | `https://github.com/anurag203/clarify-rl` returns HTTP 200; same commit graph as HF Space |
| Trained model artifact public on HF Hub | ✅ | All 3 model repos public + have rich model cards (`anurag203/clarify-rl-run1/2/4-...`); Run 4 weights also at training-account repo |

## Deliverable links (paste these into the submission form)

| Submission form field | URL |
|---|---|
| **Hugging Face Space (env)** | https://huggingface.co/spaces/agarwalanu3103/clarify-rl |
| **Colab notebook** | https://colab.research.google.com/github/anurag203/clarify-rl/blob/main/training/train_grpo.ipynb |
| **Code repository** | https://github.com/anurag203/clarify-rl |
| **Blog (HF-hosted markdown)** | https://huggingface.co/spaces/agarwalanu3103/clarify-rl/blob/main/docs/blog.md |
| **Interactive demo (replay + live chat)** | https://huggingface.co/spaces/anurag203/clarify-rl-demo |
| **W&B dashboard** | https://wandb.ai/anuragagarwal203-cisco/clarify-rl |
| **Hero trained model — Run 4 (β=0.2 KL anchor)** | https://huggingface.co/anurag203/clarify-rl-run4-qwen3-1.7b-beta0.2 |

## Required files at repo root

- `inference.py` ✅
- `openenv.yaml` ✅ (spec_version 1, fastapi runtime, 3 task difficulties)
- `Dockerfile` ✅
- `README.md` ✅ (5 badges + asset table + 6 embedded plots + headline + per-family table)
- `pyproject.toml` ✅
- `server/` ✅ (`app.py`, `clarify_environment.py`, `rubric.py`, `scenarios.py`, plus rubric components)
- `scenarios/` ✅
- `plots/*.png` ✅ (committed to both GitHub and HF Space)
- `docs/blog.md` ✅ (27 KB writeup, the `--video / blog` deliverable)
- `docs/model_cards/` ✅ (rich cards for Run 1 / Run 2 / Run 4 — also pushed to HF model repos)

## What Must Be True Before Submit (verified)

- [x] `pytest -m 'not integration' -q` passes locally
- [x] Public HF Space responds on `/health` and `/reset` from the public Space URL
- [x] `openenv.yaml` parses (auto-validator does this)
- [x] No secrets are committed (`.git/config` cleaned, `.env` is `.gitignore`d)
- [x] Plots embedded with markdown `![]()` syntax and resolve from raw README
- [x] Colab badge URL opens the training notebook in a fresh Colab session
- [x] All 3 trained model repos public + have rich model cards
- [x] W&B dashboard publicly viewable

## Inference Modes (used in demo + eval)

- `BASELINE_MODE=policy` — deterministic scripted agent (no LLM needed) — used as control
- `BASELINE_MODE=hybrid` — LLM with policy fallback (default in demo Space)
- `BASELINE_MODE=llm` — pure LLM (used by the eval pipeline running vLLM-in-HF-Job)

## Push Reminder

Top-level tree must look like:

```text
Dockerfile
README.md
inference.py
openenv.yaml
pyproject.toml
server/
scenarios/
plots/
docs/
training/
tests/
scripts/
```

## How to verify our submission in 60 seconds (judges)

1. Open https://huggingface.co/spaces/agarwalanu3103/clarify-rl — confirm Space loads (cpu-basic, ~30 s warm-up).
2. From a terminal: `curl -s https://agarwalanu3103-clarify-rl.hf.space/reset -X POST -H 'Content-Type: application/json' --data '{}' | python -m json.tool` — confirm a real `CallToolObservation` with a task pops out.
3. Click the Colab badge — confirm `training/train_grpo.ipynb` opens.
4. Open the demo Space https://huggingface.co/spaces/anurag203/clarify-rl-demo — try the Replay or Chat tab. Live chat uses CPU inference of Run 4, so first response can take ~30-60 s.
5. Open the W&B project https://wandb.ai/anuragagarwal203-cisco/clarify-rl — confirm 3 runs (run1 / run2 / run4) are present with reward + loss curves.
