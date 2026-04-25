# STATUS — live project state

> **Update this file at the END of every session.** Keep it short (≤ 80 lines). For history, see `SESSION_LOG.md`.

**Last updated**: 2026-04-25 14:50 IST — Cascade (Windsurf), Phase 4 deploy + baseline eval

## Current phase

**Phase 4.5 — HF Space live, policy baseline confirmed. Ready for LLM baseline + training.**

HF Space: https://huggingface.co/spaces/agarwalanu3103/clarify-rl (RUNNING, Docker, /health OK)

## Last completed

- `inference.py` rewritten to submission format (OpenAI client, WebSocket, structured logs, 3 modes)
- HF Space deployed to `agarwalanu3103/clarify-rl` — Docker build successful, `/health` + `/reset` verified
- Policy baseline ran: all 3 tasks end-to-end, per-question rewards confirmed (0.02-0.05)
- `scenarios/eval_held_out.json` generated (300 scenarios, seeds 10000-10099)
- `.dockerignore`, `SUBMISSION_CHECKLIST.md` added
- `pyproject.toml`: `openai`, `websockets`, `huggingface_hub` deps added

## In progress

- Nothing actively coding

## Next step (default if user just says "continue")

1. Run hybrid baseline with LLM (HF_TOKEN + Qwen2.5-1.5B-Instruct)
2. Fork `openenv_wordle_grpo.ipynb` → `training/train_grpo.ipynb`
3. First real training run (100 GRPO steps on Colab T4 or HF Jobs)
4. Evaluate trained model vs baseline

## Open questions / blockers

- Policy scores 0.00 because it submits empty plan — expected, LLM baseline will be meaningful
- SSL cert issue on local macOS (fixed with `truststore`); not an issue in Docker/HF Space

## Files most recently touched

- `inference.py` — complete rewrite (OpenAI client, WS, structured logs)
- `Dockerfile` — simplified (single-step install, no lockfile)
- `.dockerignore`, `SUBMISSION_CHECKLIST.md` — created
- `pyproject.toml` — added openai, websockets deps
- `scripts/generate_eval_set.py` — created
- `scenarios/eval_held_out.json` — generated

## Locked decisions (mirror of `.windsurf/rules/clarify-rl.md` — do NOT pivot)

- Idea: ClarifyRL — train asking-vs-guessing via RL
- Theme: #5 Wild Card (primary) + #3.2 + #2
- Headline: hallucination 90% → 3%
- 5 families: coding / medical-intake / support-triage / meeting / event
- Stack: OpenEnv 0.2.2 + MCPEnvironment + Unsloth + TRL GRPO + Qwen2.5-1.5B
- Compute: HF Jobs t4-small primary, Colab T4 fallback
- Starter notebook: fork TRL `openenv_wordle_grpo.ipynb`
- MCP tools: `ask_question`, `propose_plan`, `get_task_info`
- Deadline: Apr 26, 2026, 5:00 PM IST
