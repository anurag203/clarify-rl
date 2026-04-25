# 09 — Risk Register & Mitigations

Ranked by likelihood × impact. Top of list = address first.

## R1 — Reward curve goes flat (HIGH likelihood, HIGH impact)

**Symptom**: After 100 GRPO steps, mean episode reward stays at baseline (~0.25).

**Causes**:
- Reward signal too sparse
- Per-step shaping too small relative to terminal reward
- Rollout parsing broken (model outputs gibberish, parser silently fails)
- KL coefficient (β) too high → policy can't move

**Mitigations**:
- Sanity-check rollout parser: print 5 random completions + parsed actions
- Verify shaping rewards firing: log per-step reward by action type
- Reduce β to 0.01
- Increase shaping reward magnitude (×2)
- Simplify rubric: drop InfoGain temporarily, use only FieldMatch
- Pre-warm with SFT on synthetic "ask first" trajectories (1-2 epochs)

**Time to detect**: 15 min (smoke test of 100 steps)
**Time to fix**: 30-60 min

## R2 — Reward hacking (HIGH likelihood, MEDIUM impact)

**Symptom**: Reward curve climbs but qualitative outputs are gibberish/repetitive.

**Likely hacks**:
- Always ask same generic question 6 times then submit empty plan
- Submit JSON with all profile field keys but garbage values
- Output the same action token over and over

**Mitigations**:
- Duplicate-Q penalty (already in plan)
- HallucinationCheckRubric (already in plan)
- FormatCheck Gate with strict schema (already in plan)
- Add EntropyRubric: penalize repeated actions (component if needed)
- Manual inspection of 10 trained outputs every 100 steps

**Time to detect**: 100 GRPO steps + manual inspection (15 min)
**Time to fix**: 30 min (add penalty component)

## R3 — Colab session times out mid-training (MEDIUM, MEDIUM)

**Symptom**: Long training run gets killed by Colab free-tier session limits.

**Mitigations**:
- Save LoRA checkpoint every 100 steps
- Always run training in resumable form (TRL supports resume from checkpoint)
- Plan training in 100-step chunks, not one mega-run
- Have second Google account ready for backup

**Time to detect**: live
**Time to fix**: 5 min (resume from last checkpoint)

## R4 — HF Space build fails (MEDIUM, HIGH)

**Symptom**: `git push space main` succeeds but Space build errors out.

**Common causes**:
- Dockerfile issues (missing deps, wrong Python version)
- pyproject.toml resolution failure
- HF Space hardware mismatch

**Mitigations**:
- Test Docker build LOCALLY before pushing: `docker build -t clarify-rl . && docker run -p 8000:8000 clarify-rl`
- Mirror EXACT Dockerfile from working SRE env (which we know builds)
- Push minimal stub Space FIRST (just FastAPI hello world), confirm builds, then layer on env
- Keep Space build logs open in browser tab while pushing

**Time to detect**: 5-10 min (HF build logs)
**Time to fix**: 15-30 min (Docker iteration)

## R5 — Validator rejects submission (LOW likelihood, FATAL impact)

**Symptom**: Auto-validator marks submission incomplete; never reaches human judges.

**Mitigations**:
- Run through every item in `docs/07-deployment.md` checklist
- 1-hour pre-deadline buffer for fixes
- Test ALL deliverable links from incognito browser
- Make sure plots are committed as files, not just in notebook outputs

**Time to detect**: post-submission (TOO LATE — must validate before)
**Time to fix**: depends on what's missing

## R6 — Training takes too long on T4 (LOW, MEDIUM)

**Symptom**: 600 GRPO steps take >2 hours; eats into Day 2 schedule.

**Mitigations**:
- Use Unsloth (we already are)
- Use 4-bit quantization (we already are)
- Reduce max_seq_length to 2048 if needed
- Reduce num_generations to 2 (instead of 4)
- Stop at 300 steps if curve is good — quality > quantity

**Time to detect**: 30 min into training (extrapolate)
**Time to fix**: tune config, restart from checkpoint

## R7 — Rubric doesn't separate good from bad (LOW, HIGH) — ✅ VERIFIED OK

**Symptom**: Even oracle policy gets ~0.5; even random policy gets ~0.5.

**Causes**:
- Weights wrong, components average out
- FormatCheck too lenient
- HallucinationCheck too punitive

**Mitigations**:
- Run sanity policies BEFORE training:
  - Random: should get ~0.20
  - Oracle (asks all critical Qs, perfect plan): should get ~0.95
  - Blank plan: should get 0.0
- If gap is small, retune weights and component logic before training

**Current status**: Oracle scores ~0.89 via `smoke_env.py` (FormatCheck=1.0, FieldMatch=1.0, InfoGain=1.0, Efficiency=0.5, Hallucination=0.75). Gap is healthy.

**Time to detect**: 10 min (sanity script)
**Time to fix**: 30-60 min

## R8 — Profile generator produces unsolvable scenarios (LOW, MEDIUM) — ✅ MITIGATED

**Symptom**: Even oracle can't get high score on some scenarios.

**Causes**:
- Field vocabulary too sparse → user simulator returns wrong field
- Critical fields not always present
- Request template too vague to even hint at task type

**Mitigations**:
- Validate generator: 100 random scenarios → oracle scores them → all should be ≥0.7
- Add task_type hint to every request template (subtle, e.g. "dinner" → restaurant)
- Ensure FIELD_KEYWORDS covers all profile fields

**Fix applied**: `scenarios.py` now always includes `required_keys` in the profile for medium/hard difficulty. Hard range adjusted to (6,7) to match actual field pool sizes (max 7).

**Time to detect**: 5 min (sanity check)
**Time to fix**: 15-30 min

## R9 — One team member becomes unavailable (LOW, HIGH)

**Symptom**: Anurag or Kanan can't continue (illness, technical issues, lost device).

**Mitigations**:
- Both can git-push to both remotes
- Both have HF + GitHub credentials
- Both have Colab access
- Pair-program critical sections (env, rubric)

**Time to detect**: live
**Time to fix**: depends, but project should continue

## R10 — Last-minute organizational changes (LOW, VARIABLE)

**Symptom**: Submission form changes, deadline shifts, theme reinterpretations announced.

**Mitigations**:
- Monitor Discord every 2 hours
- Both team members on Discord notifications
- Have a Plan B for each deliverable (video OR blog, not both required)

## Fallback Plans (graceful degradation)

If we run out of time:

1. **Cut difficulty levels**: Ship only "medium" task — still scores well on Storytelling
2. **Cut task types**: Ship 3 of 5 task types instead of all 5
3. **Cut training**: Use Unsloth pre-trained on synthetic SFT data, skip GRPO. Worse story but still ships.
4. **Cut video**: Ship blog post only.
5. **Cut blog**: Ship video only.

The core ship is: **HF Space + Colab + plots + README**. Everything else is bonus.

## Risk Score Summary

| ID | Risk | L | I | Score |
|----|------|---|---|-------|
| R1 | Reward curve flat | H | H | 9 |
| R2 | Reward hacking | H | M | 6 |
| R3 | Colab timeout | M | M | 4 |
| R4 | HF Space build fail | M | H | 6 |
| R5 | Validator rejection | L | F | 5 |
| R6 | Training too slow | L | M | 2 |
| R7 | Rubric doesn't separate | L | H | 3 |
| R8 | Bad scenarios | L | M | 2 |
| R9 | Team member down | L | H | 3 |
| R10 | Org changes | L | V | 1 |

L=likelihood, I=impact, F=fatal.

**Top 3 to actively mitigate during build**: R1, R2, R4.
