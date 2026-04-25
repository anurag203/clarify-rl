# What you (Anurag) need to do — GUI-only tasks

> **🎯 SUBMISSION-READY (2026-04-26 03:10 IST).** All GUI gates below are cleared.
> Live state lives in [`STATUS.md`](STATUS.md); experiment writeup in [`blog.md`](blog.md); submission lineup in the [`README.md`](../README.md) headline. **This file is frozen as a historical to-do list — it does *not* reflect current state.**

Last updated: 2026-04-25 16:25 UTC (smoke green, prod run 1 launched on account 1) — *frozen at this snapshot*

The hackathon submission requires a few steps that **must** happen through a
web browser (account creation, Pro upgrades, model license accepts, copying
write tokens, creating GitHub repos). Everything else — terminal commands, git
pushes, code fixes, launching HF Jobs, running evals — I (the agent) can do
from this terminal session.

---

## Status snapshot (frozen — historical only)

| Item                                | Status      | Owner |
|-------------------------------------|-------------|-------|
| Env Space (`agarwalanu3103/clarify-rl`) | LIVE, 8-session capacity, verified | done |
| Smoke test job (`69ece82a...e9b8`)  | PASSED — 5/5 steps, loss=-0.099, no pathology | done |
| Production run 1 — Qwen3-0.6B / A100-large / 300 steps (`69ece976...df48`) | ✅ COMPLETED + EVAL DONE — Run 1 in `runs.json` | done |
| Production run 2 — Qwen3-1.7B (β=0)       | ✅ COMPLETED + EVAL DONE — Run 2 in `runs.json` | done |
| Production run 4 — Qwen3-1.7B (β=0.2 KL anchor) | ✅ COMPLETED + EVAL DONE — Run 4 in `runs.json` (the central finding) | done |
| Production run 3 — Qwen3-4B (β=0)         | ❌ CANCELED at 48 min in HF Jobs SCHEDULING queue — logged as future work in `blog.md` §7b | done |
| GitHub repo for code + Colab badge  | submission asset table in `README.md` | done |

---

## You-only (must use a browser)

### A. (Most important) Two more HF accounts with write tokens

We have 3 HF accounts to spread the $120 budget across so the 3 model sizes
can train in parallel:

| # | Username           | Token starts with         | Status         |
|---|--------------------|---------------------------|----------------|
| 1 | `agarwalanu3103`   | `<redacted>`              | LIVE — running run 1 |
| 2 | `____` (account 2) | `____` (need write token) | **TO DO**      |
| 3 | `____` (account 3) | `____` (need write token) | **TO DO**      |

Steps for accounts 2 and 3:

1. Sign in to <https://huggingface.co> with the second/third account.
2. Settings → Billing → make sure the account has **Compute credits**
   (prepaid is fine — you don't need Pro). Each account needs ~$30 of credit
   for the bigger runs.
3. Settings → **Access Tokens** → "New token"
   - Name: `clarify-rl-train-write`
   - Type: **Write**
   - Expiry: 7 days (we only need it for ~24h of training)
4. Visit <https://huggingface.co/Qwen/Qwen3-1.7B> and **click Agree** on the
   license card (and the same for <https://huggingface.co/Qwen/Qwen3-4B>).
   You only need to do this once *per account*.
5. Paste into chat with me as:
   - `HF_TOKEN_2=hf_xxxxxx, USERNAME_2=<their HF username>`
   - `HF_TOKEN_3=hf_xxxxxx, USERNAME_3=<their HF username>`

I will NEVER store these in `.git/config` — they go straight into HF Jobs as
encrypted secrets and are removed from this terminal afterward.

### B. (Smaller — also for account 1) accept Qwen3 license

For your existing account 1, please briefly visit:

- <https://huggingface.co/Qwen/Qwen3-0.6B> — click **Agree** (probably needed for run 1 push)
- <https://huggingface.co/Qwen/Qwen3-1.7B> — click **Agree**
- <https://huggingface.co/Qwen/Qwen3-4B>   — click **Agree**

Without this the model push at end-of-training will fail with a 403, even
though the download (which uses caching) might succeed.

### C. Create a public GitHub repo for the code (one-time, ~2 min)

The hackathon validator requires a public GitHub repo URL with a Colab badge
on the README. We need ONE public repo on github.com (a personal account is
fine — this isn't sensitive code).

1. <https://github.com/new>
2. Repo name: `clarify-rl` (or anything)
3. **Public**, no README/license/.gitignore (we already have all three).
4. After clicking Create, **paste me the SSH or HTTPS clone URL**. Examples:
   - `git@github.com:yourgithubuser/clarify-rl.git`
   - `https://github.com/yourgithubuser/clarify-rl.git`

I will:
- `git remote add github <url>`
- `git push -u github master`
- update the Colab badge in `README.md` to point at the right repo
- commit and push the badge fix

### D. (After all 3 trainings push) public model cards

Once HF Jobs pushes the trained checkpoints to
`<username>/clarify-rl-grpo-{0.6b,1.7b,4b}`, you'll need to:

1. Visit each model repo (3 total)
2. Settings → Visibility → flip to **Public**

This is the only way the hackathon validator (and reviewers) can fetch them.
I cannot flip visibility from the terminal.

---

## I (the agent) am handling these — no GUI needed

You don't need to do any of this; I'm running them from the terminal:

- ✅ All `git add` / `git commit` / `git push` (to GitHub and HF Spaces)
- ✅ All `hf jobs uv run …` launches and monitoring
- ✅ All `python scripts/run_eval.py …` and `scripts/make_plots.py` runs
- ✅ All edits to `train_grpo.py`, `launch_hf_job.sh`, `run_eval.py`, etc.
- ✅ Pushed env Space changes (Space at commit `895f00d`, 8 concurrent sessions verified)
- ✅ Smoke test on account 1 (PASSED — see status table above)
- ✅ Production run 1 launched (`69ece976d2c8bd8662bcdf48`) — monitoring now
- 🔜 Will launch run 2 (Qwen3-1.7B) on account 2 the moment you paste its token
- 🔜 Will launch run 3 (Qwen3-4B)   on account 3 the moment you paste its token
- 🔜 Will launch a 4th "insurance" run on whichever account finishes first
- 🔜 Will run post-training eval and generate plots
- 🔜 Will write `docs/blog.md` body once eval numbers are in
- 🔜 Will update `README.md` with final model card links

---

## Right-now blocker

The pipeline is fully validated end-to-end. Run 1 is training on account 1
right now. To unblock the parallel scale-up I need from you:

1. **HF write tokens + usernames for accounts 2 and 3** (see section A)
2. **A public GitHub repo URL** (see section C)
3. **Click "Agree" on the Qwen3 license cards** for each account (sections A.4 and B)

Paste them into chat in any order. I'll do everything else.

---

## What it'll cost (rough projection of $120 budget)

| Run                    | Flavor       | Hourly | Wall time | Cost  |
|------------------------|--------------|--------|-----------|-------|
| Smoke (already done)   | a10g-small   | $1     | 0.06h     | <$0.10 |
| Run 1: Qwen3-0.6B 300st | a100-large   | $2.50  | ~1h       | ~$2.50 |
| Run 2: Qwen3-1.7B 300st | a100-large   | $2.50  | ~1.5h     | ~$3.75 |
| Run 3: Qwen3-4B 300st   | h200         | $5     | ~2h       | ~$10  |
| Insurance run (Qwen3-1.7B, seed 84, optional) | a100-large | $2.50 | 1.5h | ~$3.75 |
| Eval sweep (3 trained + 3 base + policy on local) | n/a | n/a | 0 | $0 |
| **Total**              |              |        |           | **~$20** |

Even with a generous 3x safety margin we stay well under $60 of $120, so we
have headroom to redo any run that fails or extend MAX_STEPS to 500 if early
plots show training is still climbing.
