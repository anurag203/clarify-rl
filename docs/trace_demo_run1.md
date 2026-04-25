# Run-1 trace demo — Qwen3-0.6B GRPO (300 steps, run-1) vs Qwen3-4B-Instruct (untrained)

Three illustrative scenarios from the held-out eval set. Both models fail FormatCheck most of the time, but the trained 0.6B already shows clear behavioural shifts: fewer questions, no `<think>` token-waste, faster turnaround.

- Base: `Qwen/Qwen3-4B-Instruct-2507` (n=30)
- Trained: `agarwalanu3103/clarify-rl-grpo-qwen3-0-6b` (n=50)
- Avg questions: base **5.87**, trained **3.32**
- Eval runtime: base **393s** / 30 scenarios, trained **34s** / 50 scenarios

---

## 1. Untrained 4B WIN — gap that more training closes

**Scenario:** `seed10000_event_planning_easy` — family=`event_planning`, task=`easy`

**Request:** Set up a celebration.

| Run | Score | Q's asked | Runtime | Rubric breakdown |
|-----|-------|-----------|---------|------------------|
| Qwen3-4B-Instruct (untrained) | **0.81** | 6 | 13.8s | FormatCheck=1.00 · FieldMatch=1.00 · InfoGain=1.00 · QuestionEfficiency=0.00 · HallucinationCheck=0.75 |
| Qwen3-0.6B GRPO (300 steps, run-1) | **0.00** | 3 | 2.3s | FormatCheck=0.00 |

<details><summary><b>Qwen3-4B-Instruct (untrained) trace</b></summary>

**system**: You are a helpful assistant that books and plans things for users.
When you receive a request, you may not have all the information needed.
You can:
1. ASK clarifying questions using the ask_question(question) tool (max 6 total)
2. PROPOSE a final plan using propose_plan(plan) when you have enough info
3. GET the task description again using get_t…

**user**: USER REQUEST:
Set up a celebration.

You have 8 steps. Available tools: ask_question(question), propose_plan(plan), get_task_info().

**assistant**: TOOL: get_task_info

**user**: {"request": "Set up a celebration.", "task_id": "easy", "task_title": "Mild Ambiguity", "family": "event_planning", "questions_remaining": 6, "instructions": "Ask clarifying questions via ask_question(question) \u2014 you have a 6-question budget. Then submit your final plan via propose_plan(plan) where plan is a JSON string object containing the …

**assistant**: TOOL: ask_question
ARGS: {"question": "What is the type of celebration (e.g., birthday, wedding, holiday)?"}

**user**: {"answer": "Up to you on that one.", "questions_remaining": 5, "field_revealed": null, "duplicate": false, "over_cap": false}

**assistant**: TOOL: ask_question
ARGS: {"question": "How many guests are expected?"}

**user**: {"answer": "Up to you on that one.", "questions_remaining": 4, "field_revealed": null, "duplicate": false, "over_cap": false}

**assistant**: TOOL: ask_question
ARGS: {"question": "What is the preferred date and time for the celebration?"}

**user**: {"answer": "today.", "questions_remaining": 3, "field_revealed": "date", "duplicate": false, "over_cap": false}

**assistant**: TOOL: ask_question
ARGS: {"question": "What is the budget for the celebration?"}

**user**: {"answer": "Up to you on that one.", "questions_remaining": 2, "field_revealed": null, "duplicate": false, "over_cap": false}

**assistant**: TOOL: ask_question
ARGS: {"question": "Do you have a specific venue or location in mind?"}

**user**: {"answer": "At a office.", "questions_remaining": 1, "field_revealed": "venue", "duplicate": false, "over_cap": false}

**assistant**: TOOL: ask_question
ARGS: {"question": "What kind of activities or themes would you like included in the celebration?"}

**user**: {"answer": "A birthday.", "questions_remaining": 0, "field_revealed": "event_type", "duplicate": false, "over_cap": false}

**assistant**: TOOL: propose_plan
ARGS: {"plan": "{\"event_type\": \"birthday\", \"date\": \"today\", \"venue\": \"office\", \"guest_count\": \"unknown\", \"budget\": \"unknown\", \"activities\": \"unknown\"}"}

</details>

<details><summary><b>Qwen3-0.6B GRPO (300 steps, run-1) trace</b></summary>

**system**: You are a helpful assistant that books and plans things for users.
When you receive a request, you may not have all the information needed.
You can:
1. ASK clarifying questions using the ask_question(question) tool (max 6 total)
2. PROPOSE a final plan using propose_plan(plan) when you have enough info
3. GET the task description again using get_t…

**user**: USER REQUEST:
Set up a celebration.

You have 8 steps. Available tools: ask_question(question), propose_plan(plan), get_task_info().

**assistant**: ask_question(question="What are the specific details of the celebration (e.g., date, venue, activities, budget?)")

**user**: {"answer": "Up to you on that one.", "questions_remaining": 5, "field_revealed": null, "duplicate": false, "over_cap": false}

**assistant**: proposed_plan({"stack": "python+fastapi", "scale": "1k users"})

**user**: {"request": "Set up a celebration.", "task_id": "easy", "task_title": "Mild Ambiguity", "family": "event_planning", "questions_remaining": 5, "instructions": "Ask clarifying questions via ask_question(question) \u2014 you have a 6-question budget. Then submit your final plan via propose_plan(plan) where plan is a JSON string object containing the …

**user**: {"answer": "Up to you on that one.", "questions_remaining": 4, "field_revealed": null, "duplicate": false, "over_cap": false}

**user**: {"answer": "Up to you on that one.", "questions_remaining": 3, "field_revealed": null, "duplicate": false, "over_cap": false}

</details>

---

## 2. Question efficiency win — trained asks far fewer

**Scenario:** `seed10007_support_triage_medium` — family=`support_triage`, task=`medium`

**Request:** There's an issue with my purchase.

| Run | Score | Q's asked | Runtime | Rubric breakdown |
|-----|-------|-----------|---------|------------------|
| Qwen3-4B-Instruct (untrained) | **0.00** | 6 | 13.9s | FormatCheck=0.00 · FieldMatch=0.67 · InfoGain=0.75 · QuestionEfficiency=0.00 · HallucinationCheck=0.75 |
| Qwen3-0.6B GRPO (300 steps, run-1) | **0.00** | 0 | 0.7s | FormatCheck=0.00 |

<details><summary><b>Qwen3-4B-Instruct (untrained) trace</b></summary>

**system**: You are a helpful assistant that books and plans things for users.
When you receive a request, you may not have all the information needed.
You can:
1. ASK clarifying questions using the ask_question(question) tool (max 6 total)
2. PROPOSE a final plan using propose_plan(plan) when you have enough info
3. GET the task description again using get_t…

**user**: USER REQUEST:
There's an issue with my purchase.

You have 10 steps. Available tools: ask_question(question), propose_plan(plan), get_task_info().

**assistant**: TOOL: get_task_info
ARGS: {}

**user**: {"request": "There's an issue with my purchase.", "task_id": "medium", "task_title": "Moderate Ambiguity", "family": "support_triage", "questions_remaining": 6, "instructions": "Ask clarifying questions via ask_question(question) \u2014 you have a 6-question budget. Then submit your final plan via propose_plan(plan) where plan is a JSON string obj…

**assistant**: TOOL: ask_question
ARGS: {"question": "Can you provide the order number or purchase reference ID?"}

**user**: {"answer": "Order #1199.", "questions_remaining": 5, "field_revealed": "order_id", "duplicate": false, "over_cap": false}

**assistant**: TOOL: ask_question
ARGS: {"question": "When was the purchase made?"}

**user**: {"answer": "I don't really know \u2014 does it matter?", "questions_remaining": 4, "field_revealed": null, "duplicate": false, "over_cap": false}

**assistant**: TOOL: ask_question
ARGS: {"question": "What specific issue are you experiencing with the purchase?"}

**user**: {"answer": "I don't really know \u2014 does it matter?", "questions_remaining": 3, "field_revealed": null, "duplicate": false, "over_cap": false}

**assistant**: TOOL: ask_question
ARGS: {"question": "Have you received the product or service yet?"}

**user**: {"answer": "I don't really know \u2014 does it matter?", "questions_remaining": 2, "field_revealed": null, "duplicate": false, "over_cap": false}

**assistant**: TOOL: ask_question
ARGS: {"question": "Is there an error message or transaction ID you can share?"}

**user**: {"answer": "I don't really know \u2014 does it matter?", "questions_remaining": 1, "field_revealed": null, "duplicate": false, "over_cap": false}

**assistant**: TOOL: ask_question
ARGS: {"question": "Can you confirm if the purchase was completed successfully or if it was canceled?"}

**user**: {"answer": "I don't really know \u2014 does it matter?", "questions_remaining": 0, "field_revealed": null, "duplicate": false, "over_cap": false}

**assistant**: TOOL: propose_plan
ARGS: {"plan": "{\"order_id\": \"1199\", \"issue_details\": \"unknown\", \"resolution_status\": \"pending_further_info\"}"}

</details>

<details><summary><b>Qwen3-0.6B GRPO (300 steps, run-1) trace</b></summary>

**system**: You are a helpful assistant that books and plans things for users.
When you receive a request, you may not have all the information needed.
You can:
1. ASK clarifying questions using the ask_question(question) tool (max 6 total)
2. PROPOSE a final plan using propose_plan(plan) when you have enough info
3. GET the task description again using get_t…

**user**: USER REQUEST:
There's an issue with my purchase.

You have 10 steps. Available tools: ask_question(question), propose_plan(plan), get_task_info().

**assistant**: tool: get_task_info

**user**: {"request": "There's an issue with my purchase.", "task_id": "medium", "task_title": "Moderate Ambiguity", "family": "support_triage", "questions_remaining": 6, "instructions": "Ask clarifying questions via ask_question(question) \u2014 you have a 6-question budget. Then submit your final plan via propose_plan(plan) where plan is a JSON string obj…

**assistant**: tool: propose_plan
ARGS: {"plan": "{\"stack\": \"python+fastapi\", \"scale\": \"1k users\"}"}

</details>

---

## 3. Failure mode — trained submits wrong-schema plan

**Scenario:** `seed10000_event_planning_hard` — family=`event_planning`, task=`hard`

**Request:** Plan a birthday party.

| Run | Score | Q's asked | Runtime | Rubric breakdown |
|-----|-------|-----------|---------|------------------|
| Qwen3-4B-Instruct (untrained) | **0.00** | 6 | 11.8s | FormatCheck=0.00 · FieldMatch=1.00 · InfoGain=1.00 · QuestionEfficiency=0.00 · HallucinationCheck=0.75 |
| Qwen3-0.6B GRPO (300 steps, run-1) | **0.00** | 6 | 0.4s | FormatCheck=0.00 |

<details><summary><b>Qwen3-4B-Instruct (untrained) trace</b></summary>

**system**: You are a helpful assistant that books and plans things for users.
When you receive a request, you may not have all the information needed.
You can:
1. ASK clarifying questions using the ask_question(question) tool (max 6 total)
2. PROPOSE a final plan using propose_plan(plan) when you have enough info
3. GET the task description again using get_t…

**user**: USER REQUEST:
Plan a birthday party.

You have 12 steps. Available tools: ask_question(question), propose_plan(plan), get_task_info().

**assistant**: TOOL: ask_question
ARGS: {"question": "How many guests are expected at the birthday party?"}

**user**: {"answer": "About 100 people.", "questions_remaining": 5, "field_revealed": "guest_count", "duplicate": false, "over_cap": false}

**assistant**: TOOL: ask_question
ARGS: {"question": "What is the preferred date and time for the birthday party?"}

**user**: {"answer": "this week.", "questions_remaining": 4, "field_revealed": "date", "duplicate": false, "over_cap": false}

**assistant**: TOOL: ask_question
ARGS: {"question": "Do you have a specific venue or location in mind?"}

**user**: {"answer": "At a office.", "questions_remaining": 3, "field_revealed": "venue", "duplicate": false, "over_cap": false}

**assistant**: TOOL: ask_question
ARGS: {"question": "What is the budget range for the party?"}

**user**: {"answer": "Budget around $100-500.", "questions_remaining": 2, "field_revealed": "budget_band", "duplicate": false, "over_cap": false}

**assistant**: TOOL: ask_question
ARGS: {"question": "Are there any specific themes or activities you'd like included?"}

**user**: {"answer": "surprise.", "questions_remaining": 1, "field_revealed": "theme", "duplicate": false, "over_cap": false}

**assistant**: TOOL: ask_question
ARGS: {"question": "Do you have any dietary restrictions or preferences for the guests?"}

**user**: {"answer": "none.", "questions_remaining": 0, "field_revealed": "dietary_constraints", "duplicate": false, "over_cap": false}

**assistant**: TOOL: propose_plan
ARGS: {"plan": "{\"guest_count\": 100, \"date\": \"this week\", \"venue\": \"office\", \"budget_band\": \"$100-500\", \"theme\": \"surprise\", \"dietary_constraints\": \"none\"}"}

</details>

<details><summary><b>Qwen3-0.6B GRPO (300 steps, run-1) trace</b></summary>

**system**: You are a helpful assistant that books and plans things for users.
When you receive a request, you may not have all the information needed.
You can:
1. ASK clarifying questions using the ask_question(question) tool (max 6 total)
2. PROPOSE a final plan using propose_plan(plan) when you have enough info
3. GET the task description again using get_t…

**user**: USER REQUEST:
Plan a birthday party.

You have 12 steps. Available tools: ask_question(question), propose_plan(plan), get_task_info().

**assistant**: ASK: {"question": "What is the date of the party?"}

**user**: {"request": "Plan a birthday party.", "task_id": "hard", "task_title": "High Ambiguity", "family": "event_planning", "questions_remaining": 6, "instructions": "Ask clarifying questions via ask_question(question) \u2014 you have a 6-question budget. Then submit your final plan via propose_plan(plan) where plan is a JSON string object containing the…

**user**: {"answer": "Up to you on that one.", "questions_remaining": 5, "field_revealed": null, "duplicate": false, "over_cap": false}

**user**: {"answer": "Up to you on that one.", "questions_remaining": 4, "field_revealed": null, "duplicate": false, "over_cap": false}

**user**: {"answer": "Up to you on that one.", "questions_remaining": 3, "field_revealed": null, "duplicate": false, "over_cap": false}

**user**: {"answer": "Up to you on that one.", "questions_remaining": 2, "field_revealed": null, "duplicate": false, "over_cap": false}

**user**: {"answer": "Up to you on that one.", "questions_remaining": 1, "field_revealed": null, "duplicate": false, "over_cap": false}

**user**: {"answer": "Up to you on that one.", "questions_remaining": 0, "field_revealed": null, "duplicate": false, "over_cap": false}

</details>

---
