# 04 — Rubric Design

The reward system uses OpenEnv's official `Rubric` system with **composable primitives**.

## Top-level Composition

```python
from openenv.core.rubrics import Rubric, WeightedSum, Gate, Sequential

clarify_rubric = Sequential(
    Gate(FormatCheckRubric(), threshold=0.5),     # HARD GATE
    WeightedSum(
        rubrics=[
            FieldMatchRubric(),         # 0.50  primary correctness
            InfoGainRubric(),           # 0.20  did Qs reveal critical fields
            QuestionEfficiencyRubric(), # 0.15  fewer Qs = better
            HallucinationCheckRubric(), # 0.15  no fabrication
        ],
        weights=[0.50, 0.20, 0.15, 0.15],
    ),
)
```

Behavior:

- **If format invalid** (FormatCheck < 0.5) → `Gate` returns 0 → `Sequential` returns 0
- **If format valid** → `WeightedSum` evaluates 4 components → returns weighted average

## Component 1 — FormatCheckRubric (gate)

Checks the submitted plan is well-formed JSON with required keys for the task type.

```python
class FormatCheckRubric(Rubric):
    def forward(self, action: CallToolAction, obs: CallToolObservation) -> float:
        # Plan parsed from action.arguments["plan"] (string of JSON)
        # Required keys vary by task_type (read from obs.metadata)
        # Returns 1.0 if all required keys present and types valid, else 0.0
```

Required keys per task family (5 families — see `docs/05-scenario-design.md`):

| Family | Required keys |
|--------|---------------|
| `coding_requirements` | stack, scale, auth, datastore |
| `medical_intake` | primary_symptom, duration, severity |
| `support_triage` | order_id, item_issue, refund_or_replace |
| `meeting_scheduling` | participants, date, time |
| `event_planning` | event_type, date, guest_count, venue |

If ANY required key is missing or types don't match → 0.0 (gate triggers, total reward = 0).

## Component 2 — FieldMatchRubric (weight 0.50)

The PRIMARY signal: how well does the submitted plan match the hidden profile?

```python
class FieldMatchRubric(Rubric):
    def forward(self, action, obs) -> float:
        plan = parse_plan(action.arguments["plan"])
        profile = obs.metadata["hidden_profile"]   # exposed at terminal step only
        
        # Per-field weighted F1
        # weights: critical fields (budget, dietary, group_size) = 2x, others = 1x
        matches = 0.0
        total_weight = 0.0
        for field, expected_value in profile.items():
            w = 2.0 if field in CRITICAL_FIELDS else 1.0
            total_weight += w
            if field in plan and field_matches(plan[field], expected_value):
                matches += w
        return matches / total_weight if total_weight else 0.0
```

`field_matches()` does fuzzy comparison: case-insensitive, semantic equivalence for known synonyms (e.g. "veg" == "vegetarian"), tolerance for numeric ranges.

## Component 3 — InfoGainRubric (weight 0.20)

How many of the *critical* hidden fields did the agent's questions reveal?

```python
class InfoGainRubric(Rubric):
    def forward(self, action, obs) -> float:
        revealed: set[str] = obs.metadata["fields_revealed"]
        critical: set[str] = obs.metadata["critical_fields"]   # subset of profile
        
        if not critical:
            return 1.0
        return len(revealed & critical) / len(critical)
```

This rewards asking the *right* questions, not just any questions.

## Component 4 — QuestionEfficiencyRubric (weight 0.15)

Penalizes using too many of the 6-question budget. Linear decay.

```python
class QuestionEfficiencyRubric(Rubric):
    def forward(self, action, obs) -> float:
        n_asked: int = obs.metadata["questions_asked_count"]
        max_q: int = 6
        # 0 Qs → 1.0;  6 Qs → 0.0;  3 Qs → 0.5
        return max(0.0, 1.0 - n_asked / max_q)
```

## Component 5 — HallucinationCheckRubric (weight 0.15)

The plan must only contain values that could be derived from asked questions.

```python
class HallucinationCheckRubric(Rubric):
    def forward(self, action, obs) -> float:
        plan = parse_plan(action.arguments["plan"])
        asked_field_keys: set[str] = obs.metadata["asked_field_keys"]
        critical_fields_in_plan = {k for k in plan if k in HIDDEN_FIELDS}
        
        if not critical_fields_in_plan:
            return 1.0
        # Penalize plan fields whose key was never asked about
        hallucinated = critical_fields_in_plan - asked_field_keys
        return 1.0 - len(hallucinated) / len(critical_fields_in_plan)
```

So if the agent guesses 4 fields but only asked about 2, score = 1 - 2/4 = 0.5.

## Score Range Examples

| Scenario | Format | Field | Info | Eff | Hallu | Total |
|----------|:-----:|:----:|:---:|:---:|:----:|:----:|
| Bad format | 0.0 | — | — | — | — | **0.0** (gated) |
| Asks nothing, guesses everything | 1.0 | 0.20 | 0.00 | 1.00 | 0.00 | **0.25** |
| Asks all 6, gets everything right | 1.0 | 1.00 | 1.00 | 0.00 | 1.00 | **0.80** |
| Asks 3 critical Qs, perfect plan | 1.0 | 0.95 | 1.00 | 0.50 | 1.00 | **0.92** |
| Asks 2 Qs, gets critical right | 1.0 | 0.85 | 1.00 | 0.67 | 1.00 | **0.88** |

The rubric naturally produces a sharp gradient: lazy guessing → ~0.25, smart asking → ~0.85+. This drives a clean reward curve during training.

## Per-step Shaping Rewards (separate from rubric)

The composable rubric only fires on terminal step. Per-step shaping is computed inline in `step()` for GRPO training signal:

| Action | Outcome | Shaping reward |
|--------|---------|----------------|
| `ask_question` | Reveals new field | +0.05 |
| `ask_question` | Reveals nothing useful | +0.02 |
| `ask_question` | Near-duplicate of prior Q | -0.02 |
| `ask_question` | Over-cap (no Qs left) | -0.05 + done |
| `get_task_info` | (free action) | 0.0 |
| `propose_plan` | Triggers rubric | rubric score (0-1) |

GRPO sees the cumulative reward = (sum of shaping rewards) + (terminal rubric score). This gives both dense exploration signal AND the strong terminal target.

## Anti-Hacking Defenses

| Hack attempt | Defense |
|--------------|---------|
| "Just guess everything blindly" | Hallucination penalty + low FieldMatch |
| "Ask all 6 Qs as filler" | QuestionEfficiency tanks |
| "Submit empty/malformed plan" | FormatCheck gate → 0 |
| "Repeat same Q to game rewards" | Duplicate-Q penalty (per-step) |
| "Ask irrelevant Qs" | InfoGain stays low |
| "Memorize one task type" | 5 task types × procedural fields = combinatorial diversity |

## Visualization for Demo

The per-component breakdown becomes a **per-component bar chart** for the demo video:

```
        Baseline      Trained
Format    1.00   ▰▰▰   1.00   ▰▰▰
Field     0.20   ▰        0.92   ▰▰▰▰▰
Info      0.10   ▰        0.88   ▰▰▰▰▰
Eff       0.95   ▰▰▰▰▰    0.55   ▰▰▰
Hallu     0.10   ▰        0.97   ▰▰▰▰▰
─────────────────────────────────────
TOTAL     0.27           0.87
```

This is **far more compelling** than a single number, and showcases the composable rubric system to judges.
