# STATUS — live project state

> **Update this file at the END of every session.** Keep it short (≤ 80 lines). For history, see `SESSION_LOG.md`.

**Last updated**: 2026-04-25 15:30 IST — Cascade (Windsurf), gap analysis + regression tests session

## Current phase

**Phase 3.7 — Core env hardened with tests. Ready for Phase 4 (deploy + baseline eval).**

Hackathon timeline: inside 48-hour build window. Phases 1-3 complete. Env hardened with episode-done guard, 200-char cap, 48 unit tests (all green). Next: push to HF Space, inference.py, eval set.

## Last completed

- Full gap analysis: spec 03/04/05 cross-referenced against implementation code
- **Fix**: `_guard_episode_done()` — blocks tool calls after episode ends (was unguarded)
- **Fix**: `ask_question` now truncates to 200 chars (spec 03 requirement)
- **Fix**: `05-scenario-design.md` difficulty table synced with code (medium=4-5, hard=6-7)
- **Fix**: `__init__.py` made import-safe (try/except) so tests run without openenv
- **Tests**: 4 test modules (48 unit tests all green):
  - `test_scenarios.py` — generation, reproducibility, field coverage, required keys, difficulty ranges
  - `test_grader.py` — parse_plan, ask_question_reward, reward magnitudes
  - `test_user_simulator.py` — keyword coverage, field matching, all-family reachability
  - `test_environment.py` — integration tests (marked `@integration`, need openenv to run)
  - `test_rubrics.py` — rubric components + composition (marked `@integration`)
- **Verified**: every TASK_FIELDS key has matching FIELD_KEYWORDS entry (all 5 families)
- pytest config added to `pyproject.toml` (importlib mode, integration marker)

## In progress

- Nothing actively coding — ready for Phase 4

## Next step (default if user just says "continue")

1. Push to HF Space, verify public access incognito
2. Write `inference.py` — baseline eval (Qwen2.5-1.5B-Instruct via HF Inference API)
3. Generate `scenarios/eval_held_out.json` (frozen seeds 10000-10099)
4. Fork `openenv_wordle_grpo.ipynb` → `training/train_grpo.ipynb`
5. First real training run (100 GRPO steps on Colab T4)

## Open questions / blockers

- None as of last update.

## Files most recently touched

- `server/clarify_environment.py` — `_guard_episode_done()`, 200-char cap
- `__init__.py` — import-safe try/except
- `docs/05-scenario-design.md` — difficulty table synced
- `tests/test_scenarios.py`, `test_grader.py`, `test_user_simulator.py`, `test_rubrics.py`, `test_environment.py` — created
- `tests/conftest.py`, `conftest.py` — pytest config
- `pyproject.toml` — pytest ini_options

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
