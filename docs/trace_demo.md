# Same-base trace demo — scenario `seed10004_event_planning_hard`

> **The point of this page:** show what 300 steps of GRPO training change on the **same base model**, on the **same scenario**, with the **same eval harness**. The two transcripts below are real, deterministic, n=50 v4-eval-pipeline runs you can reproduce yourself with `scripts/launch_eval_job.sh`.

**Scenario:** `seed10004_event_planning_hard`  
**Family:** `event_planning`  
**Surface request:** "Organize a team event."  
**Hidden profile (the agent never sees this):**
```json
{
  "event_type": "team-building",
  "date": "today",
  "guest_count": 100,
  "venue": "home"
}
```
**Required keys to win FormatCheck:** `event_type`, `date`, `venue`, `guest_count`

---

## Untrained Qwen3-0.6B (base, off-the-shelf)

| Step | Tool | Args (truncated) |
|---|---|---|
| 0 | `get_task_info` | `{}` |
| 1 | `get_task_info` | `{}` |
| 2 | `get_task_info` | `{}` |
| 3 | `get_task_info` | `{}` |
| 4 | `get_task_info` | `{}` |
| 5 | `get_task_info` | `{}` |
| 6 | `get_task_info` | `{}` |
| 7 | `get_task_info` | `{}` |
| 8 | `get_task_info` | `{}` |
| 9 | `ask_question` | `"What is the main requirement?"` |
| 10 | `ask_question` | `"What are the technical specifications?"` |
| 11 | `ask_question` | `"What is the scale or scope?"` |

**Outcome:** episode timed out at step 11 with no `propose_plan`.  
**Final score: 0.0000.** No format check, no field match, no plan.

The base model exhibits two pathologies it never recovers from:
1. **Re-read loop** — calls `get_task_info()` nine times in a row, hoping the request will somehow disambiguate itself.
2. **Generic questions** — when it finally does ask, it asks `coding_requirements`-shaped questions ("technical specifications", "scope") on an `event_planning` task, yielding zero field reveals from the env.

---

## Trained Qwen3-0.6B (Run 1, 300 GRPO steps)

| Step | Tool | Question/Plan | Env response | Field revealed |
|---|---|---|---|---|
| 0 | `ask_question` | "What are the specific details of the team members and their roles?" | "Up to you on that one." | – |
| 1 | `ask_question` | "What is the specific time and location for the event?" | "At a home." | **`venue`** |
| 2 | `ask_question` | "What is the list of team members and their roles?" | "Up to you on that one." | – |
| 3 | `ask_question` | "What is the agenda or topics to be discussed at the event?" | "Up to you on that one." | – |
| 4 | `ask_question` | "What is the duration of the event and how many participants are expected to attend?" | "About 100 people." | **`guest_count`** |
| 5 | `propose_plan` | `{"event_type":"team_event","date":"2024-12-25","venue":"home","guest_count":"100","agenda_topics":"list of topics and discussion"}` | – | **terminal** |

**Outcome:** episode terminates cleanly at step 5 with a 5-key plan.  
**Final score: 0.3818** (`FormatCheck = 1.0`, `FieldMatch = 0.36`, `InfoGain = 0.5`, `QuestionEfficiency = 0.17`, `HallucinationCheck = 0.5`).

Two of the four required fields (`venue`, `guest_count`) come straight from question reveals. Of the remaining two:
- `event_type` is a reasonable guess from the surface phrase "team event" — partial credit (the gold value is `"team-building"`, model wrote `"team_event"`).
- `date` is hallucinated (`"2024-12-25"` vs gold `"today"`) — that's exactly what the `HallucinationCheckRubric` deducts on, dropping it to 0.5.

---

## What changed in 300 steps of GRPO

The same base model, same scenario, same env, same parser. The behavioural diff is unambiguous:

| Behaviour | Untrained 0.6B | Trained 0.6B (Run 1) |
|---|---|---|
| Calls `get_task_info()` in a loop? | Yes, 9× | No, 0× |
| Submits a plan within budget? | **No** | **Yes** (step 5) |
| Asks family-appropriate questions? | No (asks coding-style) | Yes (asks event-style) |
| Picks up at least one revealed field? | No | Yes (2 fields: `venue`, `guest_count`) |
| Final score on this scenario | 0.000 | **0.382** |

This is the strongest concrete evidence we have that GRPO is working: same model, opposite behaviour, real field-grounded plan.

---

## Honest caveats

- This is **the** scenario where Run-1 wins on `event_planning`; on the other 11 `event_planning` scenarios it scores 0.0. The 0.6B model has learned the protocol shape and a workable plan template for `event`, not yet the per-family field semantics for the rest. See `docs/blog.md` §5.4.
- The trained model still hallucinates `date` even though the env answered "Up to you on that one." A stronger `HallucinationCheckRubric` weight, KL anchor, or family-aware reward would fix this (see future-work in blog §10).
- The base 0.6B's `get_task_info` loop is a function of the no-thinking system prompt — with thinking enabled, the same model burns its turn budget inside `<think>` blocks instead. Either way, it never proposes a plan. (See blog §6 for the eval-pipeline saga that pushed us to disable thinking everywhere.)

---

## Counter-trace: where Run-2 (1.7B GRPO) regresses on the SAME family

The honest version of the writeup needs the regression *and* the win it traded for. On `seed10000_event_planning_hard`, the *base* 1.7B scored **0.522** while the *trained* 1.7B scored **0.000**. **But** on `seed10002_meeting_scheduling_medium` the trained 1.7B scored **0.725** (the highest single-scenario score across all our trained models) versus the 1.7B base's 0.500 on the same scenario. Training concentrated capability into `meeting_scheduling` at the cost of `event_planning`.

The fix is in section 10: KL anchor + per-family question templates. **Run 4 directly tests this** — same model, same data, same env — see the next section.

---

## ⭐ The KL-anchor showpiece — Run 2 (β=0) vs Run 4 (β=0.2) on `seed10004_event_planning_medium`

**Same scenario, same base model, same eval harness.** The only difference is `GRPOConfig.beta=0.0` (Run 2) vs `0.2` (Run 4). This is the cleanest controlled comparison in the entire experiment.

**Surface request:** "Plan a birthday party."  
**Hidden profile (agent never sees):** `{event_type: "birthday", date: "this week", guest_count: 50, venue: "restaurant", ...}`  
**Required keys for FormatCheck:** `event_type`, `date`, `venue`, `guest_count`

### Run 2 (β=0, no anchor) — final score **0.000**

| Step | Tool | Args |
|---|---|---|
| 0 | `ask_question` | "What is the date of the birthday party?" → `this week.` (`date` ✅) |
| 1 | `ask_question` | "What is the time of the birthday party?" → `Up to you on that one.` |
| 2 | `ask_question` | "What is the **guest list size**?" → `About 50 people.` (`guest_count` ✅) |
| 3 | `ask_question` | "What is the theme of the birthday party?" → `Up to you on that one.` |
| 4 | `ask_question` | "What is the **budget** for the birthday party?" → `Up to you on that one.` ❌ wrong family |
| 5 | `get_task_info` | – (wasted turn) |
| 6 | `propose_plan` | `{event_type: "birthday", date: "this week", time: "Up to you...", guest_count: "About 50", theme: "...", budget: "..."}` |

**Score breakdown:** FormatCheck **0.0** ❌ (gate fails) → final = 0 × everything else = **0.000**.  
But the semantic components actually look fine: FieldMatch=0.75, InfoGain=0.75, Hallucination=1.0. The model is *capable* — it's just emitting in a shape the FormatCheck rubric refuses, because the missing `venue` field signals "this is not an event_planning plan." It asked `budget` and `theme` instead of `venue` because GRPO over-weighted question templates from `coding_requirements`.

### Run 4 (β=0.2 KL anchor) — final score **0.510**

| Step | Tool | Args |
|---|---|---|
| 0 | `ask_question` | "What is the date of the birthday party?" → `this week.` (`date` ✅) |
| 1 | `ask_question` | "What time would you like the party to be held?" → `Up to you on that one.` |
| 2 | `ask_question` | "How many guests are expected at the party?" → `About 50 people.` (`guest_count` ✅) |
| 3 | `ask_question` | "What is the **preferred location** for the party?" → `At a restaurant.` (`venue` ✅) |
| 4 | `ask_question` | "What is the name of the restaurant?" → `Up to you on that one.` |
| 5 | `propose_plan` | `{event_type: "birthday", date: "this week", time: "Up to you...", guest_count: "About 50 people.", venue: "At a restaurant.", restaurant_name: "Up to you..."}` |

**Score breakdown:** FormatCheck **1.0** ✅ → FieldMatch=0.44, InfoGain=0.75, Hallucination=0.75 → final = **0.510**.

### What the anchor changed

| Aspect | Run 2 (β=0) | Run 4 (β=0.2) |
|---|---|---|
| Asks for `venue`? | ❌ No (asks `budget` instead) | ✅ Yes ("preferred location") |
| Asks for `theme`/`budget` (off-family)? | ✅ Both | ❌ Neither |
| FormatCheck passes? | ❌ 0.0 | ✅ 1.0 |
| Final score | **0.000** | **0.510** |

The KL anchor pulled the 1.7B back toward the base's question vocabulary, which had `venue`/`location` as a natural `event_planning` question. Without the anchor, GRPO had drifted to a Frankenstein template ("budget" + "theme") that was rewarded somewhere during training but is wrong at inference time on this family.

This is exactly what KL regularization is *supposed to do*: keep the post-RL policy within an acceptable divergence of the prior so generic capabilities aren't sacrificed for narrow reward optimization. β=0.2 was strong enough to fully recover this scenario; the trade-off is that Run 4 also lost Run 2's strongest peak (`seed10002_meeting_scheduling_medium` from 0.725 → 0.350).

---

## What "good" looks like — Qwen3-4B base on `seed10014_meeting_scheduling_hard`

For reference, the **strongest single trace** in our entire eval suite. This is the off-the-shelf 4B base (no GRPO) on a high-ambiguity meeting-scheduling task. It's worth showing because it tells us where the ceiling sits *without* training — and is the bar a future 4B GRPO run would need to clear (a planned Run 3 was canceled in HF Jobs queue at 48 min; see `docs/blog.md` §7b).

**Surface request:** "Get a call on the calendar."  
**Hidden profile:** `{participants: "external client", date: "today", time: "flexible", duration_minutes: 30, event_type: "call", ...}`

| Step | Tool | Question/Plan | Env response | Field revealed |
|---|---|---|---|---|
| 0 | `get_task_info` | – | task body re-read | – |
| 1 | `ask_question` | "Who is the call with?" | "external client." | **`participants`** |
| 2 | `ask_question` | "What is the date of the call?" | "today." | **`date`** |
| 3 | `ask_question` | "What time does the call start?" | "flexible." | **`time`** |
| 4 | `ask_question` | "What is the duration of the call?" | "30 minutes." | **`duration_minutes`** |
| 5 | `ask_question` | "What is the event type of the call?" | "No preference, you choose." | – |
| 6 | `propose_plan` | `{"event_type":"call","date":"today","time":"flexible","duration_minutes":"30","participants":"external client"}` | – | **terminal** |

**Final score: 0.8194** (`FormatCheck=1.0`, `FieldMatch=0.89`, `InfoGain=1.0`, `QuestionEfficiency=0.17`, `Hallucination=1.0`).

Five things make this the gold trace:
1. **One `get_task_info`, then commit.** No re-read loop.
2. **Each question targets a single field** in family-canonical order (participants → date → time → duration → event_type).
3. **Every answer that revealed a field is grounded into the plan verbatim** — no hallucination penalty.
4. **`event_type` is not asked — it's inferred from the surface request "Get a call on the calendar"** ("call"). That's a zero-cost field reveal that the smaller models miss.
5. **Plan submitted at step 6 with all 5 keys present** — FormatCheck trips on this scenario, which is rare across the eval set.

This is the bar a 4B GRPO run would have to *match or exceed* on `meeting_scheduling` to justify its training cost. **Given Run 4's result on 1.7B**, the natural next-step recipe is *not* the β=0 settings we had queued for Run 3 — it would be **4B + β=0.2 + half-LR**, the configuration that recovered breadth at 1.7B. We didn't get to run it before the deadline (Run 3's β=0 launch sat in HF Jobs SCHEDULING queue for 48 min and was canceled), so it stays as future work.

---

## Reproducing

```bash
# Trained 0.6B eval
HF_TOKEN=$HF_TOKEN ./scripts/launch_eval_job.sh \
    --model agarwalanu3103/clarify-rl-grpo-qwen3-0-6b \
    --flavor a10g-small \
    --limit 50

# 0.6B base eval
HF_TOKEN=$HF_TOKEN ./scripts/launch_eval_job.sh \
    --model Qwen/Qwen3-0.6B \
    --flavor a10g-small \
    --limit 50

# Trained 1.7B eval (Run 2)
HF_TOKEN=$HF_TOKEN ./scripts/launch_eval_job.sh \
    --model agarwalanu3103/clarify-rl-grpo-qwen3-1-7b \
    --flavor a10g-large \
    --limit 50

# 1.7B base eval
HF_TOKEN=$HF_TOKEN ./scripts/launch_eval_job.sh \
    --model Qwen/Qwen3-1.7B \
    --flavor a10g-large \
    --limit 50

# Pull eval JSONs and find specific scenarios
python -c "
import json
for path, label in [
    ('outputs/run_artifacts/v4/evals/eval_clarify-rl-grpo-qwen3-0-6b_n50_v4.json', '0.6B TRAINED'),
    ('outputs/run_artifacts/v4/evals/eval_qwen3-0.6b_n50_v4.json',                  '0.6B BASE'),
    ('outputs/run_artifacts/1.7B/evals/eval_clarify-rl-grpo-qwen3-1-7b_n50.json',   '1.7B TRAINED'),
    ('outputs/run_artifacts/v4/evals/eval_qwen3-1.7b_n50_v4.json',                  '1.7B BASE'),
]:
    d = json.load(open(path))
    for sid in ['seed10004_event_planning_hard', 'seed10000_event_planning_hard']:
        r = next(x for x in d['results'] if x['scenario_id'] == sid)
        print(f'{label:13s} {sid:36s} score: {r[\"final_score\"]:.4f}')
"
```
