# 08 — Hour-by-hour Timeline

48-hour sprint. Hard deadline: **Apr 26, 5:00 PM IST**.

## Day 1 — Saturday, Apr 25

### 7:00–9:00 AM — Reporting & Opening (mandatory attendance)

- 7:00 AM check-in at Scaler Electronic City campus
- Breakfast (until 9 AM)
- Opening ceremony — listen for: Cursor credits, HF credits ($30?), final theme clarifications

### 9:00–10:00 AM — Setup

- Both team members:
  - Verify Colab account works, T4 reachable
  - Verify HF account, get on-site credits redeemed
  - **Run `hf auth login` + `hf jobs hardware`** to confirm `t4-small` access works (per opening-deck p.78-80)
  - Verify GitHub access
- Anurag: create empty `clarify-rl` HF Space (CPU basic, public, Docker)
- Kanan: create empty `clarify-rl` GitHub repo

### 10:00 AM–12:00 PM — Phase 1: Scaffold ✅ DONE

- All scaffold files in `clarify-rl/`:
  - openenv.yaml, pyproject.toml, Dockerfile, __init__.py, models.py, client.py, server/__init__.py
- ✅ Done. Move directly to Phase 2.

### 12:00–2:30 PM — Phase 2: Core environment (Anurag-led) ✅ DONE

- `server/scenarios.py` — procedural scenario generation
- `server/user_simulator.py` — rule-based Q→A
- `server/clarify_environment.py` — MCPEnvironment subclass with 3 tools + step_async
- `server/app.py` — FastAPI app via `create_app(...)`
- Local smoke test: `uvicorn server.app:app` + all 3 smoke scripts pass
- Lunch can be eaten while building (ladoos, biryani provided)

### 2:30–4:00 PM — Phase 3: Rubrics + per-step grader (Anurag-led) ✅ DONE

- `server/rubrics.py` — 5 Rubric subclasses, composed via Sequential + Gate + WeightedSum
- `server/grader.py` — per-step shaping reward + plan parser
- Wire rubric into `ClarifyEnvironment.__init__`
- Verified: oracle policy → ~0.89, random → low, blank plan → 0

### 4:00–5:30 PM — Phase 4: Deploy + baseline eval ← NEXT

- Push to HF Space, verify public access incognito
- `inference.py` — baseline eval script (Qwen2.5-1.5B-Instruct via HF Inference API)
- Run baseline on 100 held-out scenarios → save `outputs/baseline.json`
- Generate `scenarios/eval_held_out.json` (frozen seeds 10000-10099)

### 5:30–7:00 PM — Phase 5: Training notebook (Kanan-led, in parallel)

While Anurag builds env: Kanan **forks `openenv_wordle_grpo.ipynb`** (TRL official, opening-deck p.73) and adapts it.
- `training/train_grpo.ipynb` — fork of `openenv_wordle_grpo.ipynb` → swap env client to `ClarifyClient`, swap reward fn to ours
- Verify on Colab T4 first; switch to HF Jobs `t4-small` once smoke run passes
- Mock rollout with dummy env until real env is ready

### 7:00–9:00 PM — Phase 6: First real training run

- Dinner overlap (gulab jamun expected)
- Run 100 GRPO steps (Colab T4 first, HF Jobs `t4-small` once stable)
- Inspect reward curve: should trend up
- **Eyeball 4 random rollouts** (sample-inspection cadence per `06-training-plan.md`) — reward up but degenerate generations = reward hacking, stop & fix
- If flat: debug (likely rubric or rollout parsing)

### 9:00–11:00 PM — Phase 7: Iterate

- If first run looked good: continue with 300 more steps
- If reward hacking observed: tighten rubric, restart
- Generate first preliminary plots

### 11:00 PM–1:00 AM — Phase 8: Overnight prep

- Start the LONG training run (600 steps)
- One person monitors checkpoints, the other sleeps in shifts
- Begin README.md draft (use 07-deployment.md as template)

## Day 2 — Sunday, Apr 26

### 1:00–7:00 AM — Sleep shifts + monitoring

- Long run completes (~90min on T4) → if Colab kills session, resume from checkpoint
- Both should have at least 4-5 hours rest

### 7:00–9:00 AM — Phase 9: Final eval + plots

- Run trained model on held-out 100 scenarios
- Save `outputs/trained.json`
- Generate ALL 5 plots: reward_curve, loss_curve, per_task_bars, per_component, eval_dist
- Commit `plots/*.png` to repo

### 9:00–11:00 AM — Phase 10: Demo content (Kanan-led)

- Record demo screen captures (baseline vs trained traces)
- Edit 2-min video (OBS → ffmpeg → upload YouTube unlisted)
- OR write `blog.md` HF blog post (parallel option)
- Both ideally — videos pop in pitches

### 11:00 AM–1:00 PM — Phase 11: Polish + README

- Finalize root `README.md` with:
  - Pitch + headline metric
  - Embedded plots with captions
  - Result table
  - Architecture diagram (mermaid or ASCII)
  - All 4 deliverable links
- Polish blog.md if writing one
- Final commit

### 1:00–3:00 PM — Phase 12: Final validation sweep

- Run through `docs/07-deployment.md` validation checklist (every line)
- Test HF Space from incognito
- Re-run Colab notebook end-to-end on fresh runtime
- Verify all README links open correctly
- Fix any breakage

### 3:00–4:30 PM — Phase 13: Submit

- Fill Google Form with all 4 URLs
- Double-check each URL works from incognito
- Confirm submission

### 4:30–5:00 PM — BUFFER (do not skip)

- Re-verify submission record
- Take screenshots of submission confirmation
- Final coffee. Breathe.

### 5:00 PM — DEADLINE 🚨

No more changes accepted under any circumstances.

### 5:00 PM onward — Closing & networking

## Team Split — Bhole Chature

| Phase | Anurag | Kanan |
|-------|--------|-------|
| 1 | scaffold review | scaffold review |
| 2 | core env (clarify_environment.py) | scenarios.py + user_simulator.py |
| 3 | rubrics.py + grader.py | unit tests for env + rubric |
| 4 | HF Space deploy | inference.py + baseline eval |
| 5 | bug-fix env on issues | train_grpo.ipynb (Colab) |
| 6 | monitor first training run | inspect outputs, sample logs |
| 7 | iterate on env/rubric | iterate on training config |
| 8 | overnight checkpoint mgmt | sleep shift A |
| 9 | sleep shift B | final eval + plots |
| 10 | README + repo polish | demo video + blog.md |
| 11 | architecture writeup | submission link audit |
| 12 | final validation sweep | final validation sweep |
| 13 | hits submit | screenshot + confirm |

Both pair on critical risk items: training debug, validation sweep.

## Anti-bus-factor Rule

**Both team members must know how to:**
- Push to HF Space + GitHub
- Restart the Colab runtime
- Run the env locally
- Run the eval script

If one is stuck/AFK at submission time, the other should be able to ship.

## Energy Management

- Snack/coffee every 2 hours
- 5-min walks each hour
- No alcohol/Red Bull spirals
- Hydration > caffeine
