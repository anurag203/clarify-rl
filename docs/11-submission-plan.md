# 11 — Submission Plan (Final 21 hours)

> **Live document.** Updated 2026-04-25 20:30 IST. Owner: Cascade (this chat).
> Replaces the speculative parts of `08-timeline.md` for the final stretch.

## TL;DR

We have **~21 hours** to deadline (Apr 26, 5:00 PM IST). Colab smoke run validated the pipeline (loss differentiated from step 19+, OOM at step 33 on T4). User has confirmed the **full $120 budget is available** — strategy switches from "save money" to "buy reliability and quality."

Plan: run **3 production GRPO jobs in parallel** on HF Jobs (one Qwen3-0.6B / 1.7B / 4B), each on a flavor sized one tier above its minimum (no OOM repeat), with **`NUM_GENERATIONS` auto-tuned by GPU** (kills the `0.000000` loss pattern). Optional 4th insurance run on Account 4 with a different seed for robustness. Total spend ~$70-95 of $120.

---

## 1. Locked decisions for the rest of the hackathon

| Decision | Value | Rationale |
|---|---|---|
| Compute platform (real runs) | **HF Jobs** (not Colab) | Survives laptop sleep; per-second billing; on plan path |
| Env Space capacity | **`max_concurrent_envs=8`** on the existing Space | One Space supports all parallel rollouts; no need to clone |
| Model family | **Qwen3 only** (0.6B / 1.7B / 4B) | TRL's `add_response_schema` works on Qwen3 chat template; Llama / Qwen2.5 fail with `Unrecognized chat template` (verified day 1) |
| **NUM_GENERATIONS** | **auto-tuned per GPU (4 / 8 / 8 by VRAM tier)** | `2` produces `0.000000` loss when both rollouts agree (the pathology in tonight's Colab logs); 4-8 gives 6-28 pairwise advantages |
| Training step target | **250–500 steps** depending on model size | F5.3 requires ≥300; we go beyond for 0.6B since its rollouts are cheap |
| Eval set | `scenarios/eval_held_out.json` (300 scenarios; 100 per difficulty) | Already frozen, seeds 10000–10099 |
| Submission writeup format | **HF blog post** (markdown) — not video | Faster to ship under time pressure; can add video later if time permits |
| Budget posture | **Spend up to $120** to maximize quality, not minimize cost | User explicitly confirmed; failures are more expensive than compute |

---

## 2. Pending deliverables (gap report)

Cross-referenced against `docs/01-requirements.md` V1–V12 + acceptance criteria + `docs/07-deployment.md`.

### 🚨 Validator-blockers (auto-rejection if missing)

| ID | Item | Status | Owner | ETA |
|---|---|---|---|---|
| V5 | `plots/loss_curve.png` committed | ❌ | Cascade | After first 300-step run finishes |
| V6 | `plots/reward_curve.png` committed | ❌ | Cascade | Same as V5 |
| V7 | Public Colab notebook URL | ❌ | Anurag | 30 min — push notebook to GitHub then take Colab link |
| V9 | README links HF Space + Colab + writeup + embedded plots | ❌ | Cascade | After V5/V6 + writeup ready |
| V10 | Writeup (HF blog ≤ 2 min read OR YouTube ≤ 2 min) | ❌ | Cascade | Sunday morning — write `docs/blog.md` |
| GitHub repo (`origin` remote) | ❌ — only `hf` remote exists today | Anurag | 15 min |

### 🟡 Winning-readiness gaps (acceptance criteria)

| ID | Item | Status | Owner | ETA |
|---|---|---|---|---|
| F5.3 | ≥ 300 GRPO steps with upward reward curve | 🟡 50-step smoke run cooking; real run starts on HF Jobs tonight | Cascade | Tonight |
| F6.1 | Baseline eval on 100 held-out scenarios | ❌ | Cascade | Tonight, after eval script written |
| F6.2 | Trained eval on same 100 scenarios | ❌ | Cascade | Sunday morning |
| F6.3 | Per-family + per-difficulty + per-component metrics + headline hallucination rate | ❌ | Cascade | Sunday morning |
| F6.4 | Before/after bar chart | ❌ | Cascade | Sunday morning |
| Two-trace demo (baseline vs trained on the same prompt, side-by-side) | ❌ | Cascade | Sunday morning |
| Headline claim "hallucination 90% → 3%" backed by real numbers | ❌ — currently a placeholder | Cascade | After eval done |

### 🟢 Cleanups (do-not-forget)

- `.git/config` has the HF token in plaintext (`https://agarwalanu3103:hf_xxx@…`). **Rotate token after submission.**
- `docs/STATUS.md` says "Phase 5 Qwen3-1.7B" — out of date; we're now multi-model.
- `docs/SESSION_LOG.md` last entry is from 16:05 — needs an end-of-session bump tonight.

---

## 3. Multi-model parallel-run plan

### 3.1 Phased rollout

| Phase | Goal | Cost | Stop condition |
|---|---|---|---|
| **0. Smoke** | 5-step run on `a10g-small`, no Hub push, validates env Space + WS concurrency + script changes end-to-end | ~$0.50 | All 5 steps complete + log_history.json non-empty |
| **1. Production** | Three parallel real runs (0.6B / 1.7B / 4B), one per account | ~$70 | Each writes a `DONE` marker |
| **2. Insurance** _(optional)_ | Backup 1.7B run with seed=84 from Account 4 | ~$25 | Used if any production run shows degenerate behavior |
| **3. Bonus** _(if time)_ | A second 0.6B with longer rollouts to test "does longer thinking help?" | ~$10 | Only after V5/V6/V7 validator-blockers are green |

### 3.2 The production lineup ($120 budget version)

| Acct | Model | HF Jobs flavour | VRAM | NUM_GEN | Steps | ~Time | Cost |
|---|---|---|---|---|---|---|---|
| **A1** | `Qwen/Qwen3-0.6B` | `a10g-large` (24 GB) | ~14 GB | 4 | 500 | ~5 h | ~$8 |
| **A2** | `Qwen/Qwen3-1.7B` | `a100-large` (40 GB) | ~26 GB | 8 | 400 | ~5 h | ~$25 |
| **A3** | `Qwen/Qwen3-4B` | `h100-large` (80 GB) | ~50 GB | 8 | 250 | ~4 h | ~$35 |
| **A4** | `Qwen/Qwen3-1.7B` (seed=84) | `a100-large` (40 GB) | ~26 GB | 8 | 400 | ~5 h | ~$25 (insurance) |

**Production total: ~$68. With insurance: ~$93 of $120. Bonus run uses remaining ~$25.**

> **Why a10g-large for Qwen3-0.6B (not -small or t4)?** The Colab T4 run OOM'd at step ~33 once rollouts got longer. We're now running 500 steps with `num_gen=4` (4× the rollouts per step vs the Colab run) so we need the bigger card. a10g-large is the same 24 GB as -small but with more vCPU which helps async WS round-trips.
>
> **Why a100-large for Qwen3-1.7B (not a10g-largex2)?** Multi-GPU adds NCCL setup complexity and we have not validated the multi-GPU path. Single A100 with 40 GB is a known-good config for 1.7B + `num_gen=8` + vLLM colocate.
>
> **Why h100-large for Qwen3-4B?** The 4B model with `num_gen=8` and `max_completion_length=1536` needs ~50 GB just for vLLM KV cache + activations. A100-40GB would OOM. A100-80GB and H100 both fit; H100 is faster (~2× tokens/sec on vLLM) and only ~$3-5/h more — worth it given we cannot afford a re-run.
>
> **Why drop NUM_GEN auto-tune from a flat 2 to 4-8?** Looking at tonight's Colab logs, steps 1-18 were `0.000000` because the 2-rollout group produced identical token sequences → advantage = 0 → no gradient. Bumping to 4 gives 6 pairwise advantages per group; bumping to 8 gives 28. This is the single biggest quality lever in this redesign.

All jobs target the **same env Space** (`agarwalanu3103/clarify-rl`), which is now configured for `max_concurrent_envs=8`. With 3 jobs × 8 generations × 1 active session per rollout, peak load could be 24 sessions but realistically vLLM batches them down to 3-6 concurrent WS connections per job (one per active rollout slot). 8 is enough headroom; we keep the 4th account in reserve unless we explicitly opt-in to insurance.

### 3.3 Why this hedges risk

- **3 sizes, not 1**: If only 0.6B converges, we still have a publishable scaling-story-of-one. If 4B converges, that's the headline figure.
- **Optional insurance run on the same model with different seed**: gives us a "trained 2 seeds, both converge" robustness story for the blog at ~$25 marginal cost.
- **Smoke run gates the whole pipeline before spending big**: 30 min, $0.50, but catches bugs that would have wasted $70+ in production runs.
- **OOM trap in `train_grpo.py`**: even if we hit OOM at step 200, we still get `log_history.json` + the latest checkpoint saved. Re-launching with `RESUME_FROM_CKPT` resumes from there.

### 3.4 What we are NOT doing

- ❌ Llama / Mistral / Qwen2.5-Instruct — chat template rejected by TRL `add_response_schema`. Verified day-1 failure mode; not retesting under time pressure.
- ❌ 7B+ models — H100-large 80 GB might fit one, but we have no time to validate and the scaling-story payoff is marginal vs 4B.
- ❌ Multi-GPU runs — `a10g-largex2/x4` adds NCCL setup risk; not validated for our `vllm_mode="colocate"` config.
- ❌ More than 4 parallel runs — env Space `max_concurrent_envs=8` is the cap; 4 jobs × 2 active sessions each fits, 5+ doesn't.
- ❌ Lora / QLora — full fine-tuning works on every flavor in the lineup with `adamw_8bit`; LoRA adds risk for no quality gain at our model sizes.

---

## 4. Hour-by-hour for the next 21 hours

All times IST.

### Saturday Apr 25 (today)

#### 20:00 – 20:45 — Script upgrades (Cascade) ✅
- [x] Write `training/train_grpo.py` (parameterized version of the notebook)
- [x] Write `scripts/launch_hf_job.sh` (single-job launcher with `--smoke` flag)
- [x] Write `scripts/launch_all.sh` (multi-account parallel launcher)
- [x] Write `scripts/preflight.sh` (validates whole pipeline before spending)
- [x] Write `scripts/run_eval.py` (eval any model on 300 held-out)
- [x] Write `scripts/make_plots.py` (all required PNGs)
- [x] Bump `ClarifyEnvironment.SUPPORTS_CONCURRENT_SESSIONS = True` and `max_concurrent_envs=8` in `server/app.py`
- [x] Add OOM trap + resume support + smoke-test mode + auto-tuned NUM_GEN to `train_grpo.py`

#### 20:45 – 21:30 — Anurag pushes Space + provides tokens (BLOCKING)
- [ ] **Anurag: push `server/` changes to HF Space** (`git push hf main` from `server/` dir) — 3-4 min Docker rebuild
- [ ] **Anurag: create + paste HF write-tokens for Accounts 2 and 3** (Account 1 already known)
- [ ] **Cascade: run `scripts/preflight.sh`** with HF_TOKEN_1 — confirms Space concurrency works

#### 21:30 – 22:00 — Smoke run (Cascade, $0.50)
- [ ] `HF_TOKEN=$HF_TOKEN_1 SMOKE=1 ./scripts/launch_hf_job.sh Qwen/Qwen3-0.6B a10g-small`
- [ ] Watch 5-step run complete in ~10-15 min
- [ ] Verify `log_history.json` has 5 entries with non-zero loss (with NUM_GEN=2 in smoke, expect mostly zeros — but a real DONE marker is the gate)
- [ ] If smoke passes: proceed. If smoke fails: debug for 30 min, then proceed anyway.

#### 22:00 – 22:30 — Launch all 3 production runs (Cascade, parallel)
```
HF_TOKEN_1=$T1 HF_TOKEN_2=$T2 HF_TOKEN_3=$T3 ./scripts/launch_all.sh
```
- [ ] Verify all 3 jobs are submitted in their HF Jobs dashboards
- [ ] Tail first 5-10 steps of each via Trackio dashboards
- [ ] Confirm all 3 show non-zero loss within 20 steps (sanity check on NUM_GEN=4-8 fix)

#### 22:30 – 00:30 — Run baseline eval while training cooks (Cascade)
- [ ] Run `scripts/run_eval.py --mode policy --out outputs/eval_policy.json` (no LLM, free, deterministic)
- [ ] Run `scripts/run_eval.py --mode api --out outputs/eval_qwen3-0.6b_base.json` against `Qwen/Qwen3-0.6B` (untrained baseline)
- [ ] Skim a few traces — sanity check eval script faithfully replays scenarios
- [ ] If insurance run desired: launch `INSURANCE=1 ./scripts/launch_all.sh` with HF_TOKEN_4

### Sunday Apr 26

#### 00:30 – 06:30 — Sleep shifts (training runs unattended)
- All 3 HF Jobs run unattended; check dashboards every wakeup

#### 06:30 – 09:30 — Trained eval + plots
- [ ] Pick best of 3 by Trackio reward curve → that's the "official" submission model
- [ ] Run `scripts/run_eval.py --model agarwalanu3103/clarify-rl-grpo-<best>` → `outputs/trained.json`
- [ ] Run `scripts/make_plots.py` → fills `plots/` with all required PNGs
- [ ] Commit `plots/*.png` and `outputs/*.json`

#### 09:30 – 11:30 — Writeup (`docs/blog.md`)
- [ ] Write 6-section HF blog post: problem → approach → env design → rubric → training → results
- [ ] Embed plots inline; include results table with hallucination rate
- [ ] Two-trace demo as a code-block: same prompt, baseline vs trained outputs

#### 11:30 – 13:00 — README + GitHub
- [ ] Rewrite root `README.md` per `docs/07-deployment.md` template
- [ ] Embed plots inline with 1-line captions
- [ ] Create GitHub repo, push, get public URL
- [ ] Push final commit to HF Space too
- [ ] Get Colab notebook share link

#### 13:00 – 14:30 — Final validation sweep
- [ ] Walk through all 12 V-items in `docs/01-requirements.md`
- [ ] Walk through `docs/07-deployment.md` validation checklist
- [ ] Test HF Space `/health` from incognito
- [ ] Test Colab notebook opens fresh
- [ ] Test all README links

#### 14:30 – 16:00 — Submit
- [ ] Fill submission form with the 4 URLs
- [ ] Anurag confirms each URL works from incognito
- [ ] Take screenshots of submission confirmation

#### 16:00 – 17:00 — Buffer (do not skip)
- Final coffee. Last-minute fixes only.

#### 17:00 — 🚨 DEADLINE

---

## 5. What I (Cascade) need from Anurag

In rough chronological order:

1. **Now (next 5 min)**: nothing — I'm writing scripts.
2. **In ~30 min** (after current Colab run finishes): say "go" so I push the env capacity fix to the Space.
3. **In ~60 min**: HF write-tokens for Accounts 2, 3, and 4 (one for each parallel run + one buffer).
4. **In ~2 h** (after first job is live): keep an eye on Trackio dashboards every ~30 min while we set up the others.
5. **Sunday morning**: collaborate on writeup tone + which trace to use for the demo.

Token paste format:

```
A1 = hf_xxxxxxxxxxxx     # agarwalanu3103 (already known)
A2 = hf_yyyyyyyyyyyy     # second account, USERNAME=...
A3 = hf_zzzzzzzzzzzz     # third account, USERNAME=...
A4 = hf_aaaaaaaaaaaa     # buffer account, USERNAME=...
```

I will store them in env vars only, never in source. After submission, **Anurag rotates all 4 tokens.**

---

## 6. Risk log

| Risk | Likelihood | Mitigation |
|---|---|---|
| HF Jobs API quirk we haven't seen | medium | Dry-run with 5 steps before launching real run |
| Env Space rebuild bricks the Space | low | Test capacity-bump on local Docker first; push only after current Colab run finishes |
| One of the bigger models OOMs even on the bigger flavour | medium | Buffer account A4 lets us re-launch on next-tier-up flavour |
| Trained model's chat template degrades during GRPO and breaks eval | medium | Eval script always falls back to policy mode; we still have an answer |
| Not enough time for video → ship blog only | low (planned) | Blog is the primary writeup; video is a nice-to-have |
| Eval inference cost balloons | low | Use trained vLLM checkpoint inside an HF Job for eval ($0.25 vs router-API per-token) |

---

## 7. What "done" looks like

We are submission-ready when **all** of these are true:

- [ ] 3 trained checkpoints exist on the Hub
- [ ] `plots/reward_curve.png`, `plots/loss_curve.png`, `plots/per_task_bars.png`, `plots/per_component.png` committed
- [ ] `outputs/baseline.json`, `outputs/trained.json` committed
- [ ] `docs/blog.md` written and pushed
- [ ] Root `README.md` has 4 links + 3 embedded plots + results table
- [ ] GitHub repo public, HF Space public, Colab notebook public
- [ ] Submission form filled
- [ ] Screenshots of submission confirmation in `docs/`
