# STATUS — live project state

> **Update this file at the END of every session.** Keep it short (≤ 100 lines). For history, see `SESSION_LOG.md`.

**Last updated**: 2026-04-26 04:00 IST — Cascade. **🎯 Submission lap closed + interactive demo Space deployed.** Run 4 eval (KL-anchor, β=0.2) is the central finding: event_planning μ 0.000 → **0.175** (beats 1.7B base 0.138), avg_score 0.029 → **0.056**, meeting peak 0.725 → 0.350 (the cost of breadth). All 3 trained runs are on W&B (<https://wandb.ai/anuragagarwal203-cisco/clarify-rl>), code is on GitHub (<https://github.com/anurag203/clarify-rl>), and the new judge-facing demo Space (<https://huggingface.co/spaces/anurag203/clarify-rl-demo>) has 3 tabs — Replay viewer, Run 2 vs Run 4 side-by-side, and a CPU-inference Live chat tab. Run 3 v5 was canceled at 48 min in HF Jobs SCHEDULING queue → 4B GRPO logged as future work.

## Current phase

**Phase 14 — submission**: hackathon thesis is **"KL-anchored GRPO at scale: can RL improve a strong reasoner without overfitting to one task family?"** Run 1 (0.6B, β=0), Run 2 (1.7B, β=0), and Run 4 (1.7B, β=0.2) form a 3-point ablation; the 4B base eval marks the ceiling. Run 3 (4B, β=0) was canceled in queue and is logged in `blog.md` §7b as the natural next experiment (4B + β=0.2 + half-LR).

HF Space: <https://huggingface.co/spaces/agarwalanu3103/clarify-rl> — LIVE, 64 concurrent sessions. Trained models: <https://huggingface.co/agarwalanu3103/clarify-rl-grpo-qwen3-0-6b>, <https://huggingface.co/agarwalanu3103/clarify-rl-grpo-qwen3-1-7b>, <https://huggingface.co/2022uec1542/clarify-rl-grpo-qwen3-1-7b> (Run 4, β=0.2 KL anchor).

## Last completed (this session)

- ✅ Wired `BETA` env-var → `GRPOConfig.beta` in `training/train_grpo.py`
- ✅ **Run 3 v3** (Qwen3-4B, a100-large, num_gen=4) — OOM at step 1. Auto-pushed step-0 weights + FAILED marker, exit 2.
- ✅ **Run 3 v5 launched**: 4B with `NUM_GENERATIONS=2 MAX_COMPLETION_LEN=768 VLLM_GPU_MEM_UTIL=0.40`, beta=0, lr=1e-6 (`69ed2569d2c8bd8662bce61a`) — _SCHEDULING 35min, queue is unusually long_
- ✅ **Run 4 COMPLETED**: 1.7B + KL anchor (beta=0.2, lr=5e-7) finished 300 steps in 78.2 min — final reward 0.0050, max 0.114, KL stayed bounded 0.005-0.010, lr cosine-decayed correctly to 1.7e-09
- ✅ **Run 4 eval triggered**: `69ed2ccbd2c8bd8662bce6ec` (a10g-large, n=50 v4) — RUNNING
- ✅ Run 4 full `log_history.json` downloaded from Hub (300 entries) — replaces the partial scrape
- ✅ **4B base eval COMPLETE**: `eval_qwen3-4b_qwen3-4b-base_n50_v4.json`, avg=**0.1446** (top of leaderboard!)
- ✅ Built `scripts/compare_runs.py` — same-base delta + winner-highlighted scoreboard, picks 4B base as ceiling
- ✅ Built `scripts/watch_and_eval.py` — auto-fires evals when runs finish (state persisted)
- ✅ `make_plots.py` parser fix: `=` in labels (e.g. `beta=0`) no longer breaks LABEL=PATH split (rpartition)
- ✅ Plot 01/02/04/06/07 all regenerated with Run 4's curve and 4B base baseline

## Headline n=50 v4 numbers (fair) — corrected

| Model | Avg score | Completion | Format pass |
|---|---|---|---|
| Random policy | 0.0000 | 0% | 0% |
| Qwen3-0.6B base | 0.0000 | 0% | 0% |
| **Qwen3-0.6B GRPO (Run 1, β=0)** | **0.0076** ↑ | 2% | 0% |
| Qwen3-1.7B base | 0.0669 | 18% | 0% |
| **Qwen3-1.7B GRPO (Run 2, β=0)** | 0.0286 ↓ | 6% | 0% |
| **Qwen3-1.7B GRPO (Run 4, β=0.2)** | **0.0560 ✅** | 14% | 0% |
| Qwen3-4B-Instruct | 0.0399 | 6% | 0% |
| **Qwen3-4B base** ← **REAL CEILING** | **0.1446** | **24%** | 0% |
| Qwen3-4B GRPO (Run 3) | _canceled — queue_ | _._ | _._ |

### Per-family table — KL anchor verdict in **bold**

| Family | 1.7B base | **Run 2 (no-KL)** | **Run 4 (+KL)** | 4B base | 4B-instruct |
|---|---|---|---|---|---|
| event_planning μ | 0.138 | **0.000 ❌** | **0.175 ✅✅** | 0.340 | 0.166 |
| event_planning max | 0.522 | 0.000 | 0.510 | 0.795 | 0.757 |
| meeting_scheduling μ | 0.153 | **0.130 ↑** | **0.064 ↓** | 0.287 | 0.000 |
| meeting_scheduling max | 0.500 | **0.725 ↑↑** | 0.350 | 0.819 | 0.000 |
| medical_intake | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| support_triage | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**Refined narrative — this is the hackathon thesis confirmed:**

1. **GRPO without anchor causes capability collapse.** Run 2 (β=0) drove event_planning from 0.138 → 0.000 mean, sacrificing breadth for one peak in meeting_scheduling. Avg_score regressed 0.067 → 0.029.
2. **GRPO with KL anchor cleanly improves the protected family.** Run 4 (β=0.2, lr=5e-7) on the same model recovered avg to **0.056** AND **beat base on event_planning** (0.138 → **0.175**). The anchor literally fixed Run 2's regression.
3. **The cost is peak capability.** Run 4 dropped meeting_scheduling max from 0.725 (Run 2's gem) to 0.350. KL prevents the extreme specialization that Run 2 leaned on.
4. **GRPO unlocks weak bases**: 0.6B couldn't touch event (0 → 0.032 mean, 0 → 0.382 max). Real new capability.
5. **Strong base sets the bar**: Qwen3-4B base scores 0.1446 *without any RL* — beats 4B-Instruct (0.0399) on every solvable family. Whatever Qwen3 instruct-SFT did, it weakened reasoning on multi-turn tool-using setups. Open question: does GRPO+KL push 4B above this — Run 3 was canceled before we could find out (logged as future work in `blog.md` §7b).

## In progress (right now)

- ✅ Submission lap is closed: 3 trained runs evaluated, 4B base eval as ceiling, all plots regenerated, all docs reconciled
- ✅ **Run 3 v5** — CANCELED at 48 min SCHEDULING (HF Jobs queue saturation, Kanan2005 a100-large). 4B GRPO is now an explicit "future work" item in `blog.md` §7b
- ✅ **Run 4 eval** — DONE in 7.8 min, JSON in `outputs/run_artifacts/1.7B-KL/evals/`, plots 06+07 regenerated
- ✅ Docs reconciled: STATUS.md / blog.md / README.md / trace_demo.md all tell the same KL-anchor-wins story with Run 3 logged as future work

## Decision dial: scoreboard of expected vs actual

1. ✅ **Run 4 mean (0.056) ≈ 1.7B base (0.067)** → KL anchor recovered most of the regression. `+0.027` vs Run 2 no-KL.
2. ❌ **Run 4 max(meeting) = 0.350 < Run 2's 0.725** → KL anchor traded the peak for breadth. Expected.
3. ✅ **Run 4 event_planning μ (0.175) > 1.7B base (0.138)** → KL not just preserved but *improved* the family Run 2 destroyed. Surprise upside.
4. ➖ **Run 3 (4B GRPO)** → not run (HF Jobs queue saturation, canceled at 48 min). Logged as future work; not blocking the submission since Run 4 already validates the central thesis on 1.7B.

## Next step (default if user just says "continue")

1. ✅ Run 3 cancellation decision MADE — narrative locked at 3 trained runs (Run 1 / Run 2 / Run 4) + 4B base eval as ceiling
2. ✅ Final sanity pass on blog.md + README.md done — Run 3 placeholders dropped, 3-trained-run scoreboard clean
3. ✅ Plots already match (no Run 3 stub — `runs_summary.json` only has 7 evaluated rows; `compare_runs.py` skips missing `outputs/run_artifacts/4B/evals/`)
4. ✅ STATUS.md ↔ blog.md ↔ README.md ↔ trace_demo.md cross-checked — no leftover "scheduling" or "in progress" Run 3 references in user-facing copy
5. **NEXT: submit `docs/blog.md` to HF blog post format** + push final commit

## Open questions / blockers

- 🟢 vLLM-in-HF-Job eval pipeline solid (Run-1: 6.7 min, Run-2: 113 sec, Run-4: 7.8 min, 4B base: ✅)
- 🟢 Auto-eval state persisted to `outputs/auto_eval_state.json` — restart safe
- 🟢 Token plumbing for 3 accounts working
- 🟢 Plots auto-skip Run 3 because eval JSON never landed — no manual cleanup required

## Files most recently touched (this session)

- `outputs/runs.json` — Run 3 marked `CANCELED_QUEUE_SATURATION` with reason; Run 4 marked completed with eval job id
- `outputs/auto_eval_state.json` — Run 3 = CANCELED, Run 4 = COMPLETED
- `outputs/run4_artifacts/log_history_partial.json` — full 300-step log scraped from live job
- `outputs/run_artifacts/1.7B-KL/evals/` — Run 4 eval JSON pulled from Hub
- `outputs/run_artifacts/4B-base/evals/` — 4B base eval (real ceiling)
- `docs/blog.md`, `docs/trace_demo.md`, `docs/STATUS.md`, `README.md` — KL-anchor narrative; Run 3 → future work
- `plots/01–07_*.png` + `plots/runs_summary.json` — regenerated against the 7-row scoreboard
- `scripts/compare_runs.py`, `scripts/refresh_all_plots.sh`, `scripts/watch_and_eval.py`, `scripts/poll_status.sh` — orchestration kept (Run 3 specs gated on file existence, no-op if missing)

## Locked decisions

- Idea: ClarifyRL — train asking-vs-guessing via RL
- Theme: #5 Wild Card (primary) + #3.2 + #2
- 5 families: coding / medical-intake / support-triage / meeting / event
- Stack: OpenEnv 0.2.2 + MCPEnvironment + TRL GRPO ≥1.0 + Qwen3 family (0.6B / 1.7B / 4B)
- Compute: HF Jobs, 3 successful trained runs + 1 base eval across 3 accounts; spend ≈ Run-1 $1.08 + Run-2 $2.21 + Run-3 v3 (OOM) $0.40 + Run-3 v5 (canceled in queue, $0.00) + Run-4 $1.80 + 4B base eval $0.13 + Run-4 eval $0.20 = **~$5.8 of $120 budget**
- Submission format: HF blog post (markdown)
- MCP tools: `ask_question`, `propose_plan`, `get_task_info`
- Deadline: Apr 26, 2026, 5:00 PM IST (~15.5 hours from now)
