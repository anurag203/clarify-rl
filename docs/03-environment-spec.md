# 03 — Environment Specification

The OpenEnv-compliant `ClarifyEnvironment` definition.

## Class Hierarchy

```
Environment (ABC)
  └── MCPEnvironment            # FastMCP-integrated base
        └── ClarifyEnvironment   # ours
```

## State (`ClarifyState(State)`)

Pydantic model in `models.py`:

```python
class ClarifyState(State):
    episode_id: Optional[str]            # from base State
    step_count: int                      # from base State
    task_id: str                         # "easy" | "medium" | "hard"
    task_title: str                      # human-readable
    questions_asked: list[str]           # all Qs asked this episode
    questions_remaining: int             # cap = 6
    answers_received: list[str]          # parallel to questions_asked
    fields_revealed: list[str]           # profile fields covered by Qs
    plan_submitted: bool
    episode_done: bool
    final_score: Optional[float]         # set on terminal step
    score_breakdown: Optional[dict]      # per-component rubric scores
```

`State` is exposed via the `state` property of the env. The HTTP `/state` endpoint serializes it to JSON.

## Internal (Hidden) Episode State

Stored on the env instance, NEVER in `ClarifyState` (which is exposed to client):

```python
self._hidden_profile: dict[str, Any]      # ground truth (12 fields)
self._scenario: dict                       # full scenario record
self._max_questions: int = 6
self._asked_field_keys: set[str]          # for hallucination check
self._proposed_plan: Optional[dict]       # parsed JSON when submitted
```

## Action Space

`ClarifyAction = CallToolAction` (MCP standard). Three valid tools:

### Tool 1 — `get_task_info()`

Returns the original ambiguous request + meta-info.

- Args: none
- Returns: JSON string with `request`, `task_id`, `questions_remaining`, `instructions`
- Reward: 0.0
- Side-effects: none (free action; can be called multiple times)

### Tool 2 — `ask_question(question: str)`

Ask a clarifying question. Costs 1 from the 6-question budget.

- Args: `question` (str, max 200 chars)
- Returns: JSON string with `answer`, `questions_remaining`, `field_revealed` (or null)
- Reward (per-step shaping):
  - +0.05 if Q reveals a new profile field
  - +0.02 if Q reveals nothing useful
  - -0.02 if Q is a near-duplicate of a previous Q
  - -0.05 if questions_remaining was already 0 (over-cap)
- Side-effects:
  - `questions_asked.append(q)`, `answers_received.append(a)`
  - `questions_remaining -= 1`
  - If reveals field: `fields_revealed.append(field_key)`, `_asked_field_keys.add(field_key)`

### Tool 3 — `propose_plan(plan: str)`

Submit the final plan as a JSON string. Ends the episode.

- Args: `plan` (str, must parse as JSON object with required keys depending on task type)
- Returns: JSON string with `score`, `breakdown` (per-component), `expected_profile`, `correct`
- Reward: composable rubric score (0.0-1.0)
- Side-effects:
  - `plan_submitted = True`, `episode_done = True`
  - `final_score`, `score_breakdown` populated

## Observation Space

`ClarifyObservation = CallToolObservation` (MCP standard). All observations have:

- `tool_name: str` — name of the tool just called (or `"reset"` / `"timeout"`)
- `result: str` — JSON-serialized payload (tool-specific)
- `error: Optional[ToolError]` — set only on transport/framework errors
- `done: bool` — true when episode terminates
- `reward: float` — per-step or terminal reward

### Observation Result Schemas (the JSON inside `result`)

**reset() result:**
```json
{
  "type": "task",
  "request": "Book me dinner tomorrow",
  "task_id": "medium",
  "task_title": "Moderate Ambiguity",
  "max_steps": 10,
  "questions_remaining": 6,
  "instructions": "Ask clarifying questions, then propose_plan(...) in JSON"
}
```

**ask_question() result:**
```json
{
  "answer": "I'm vegetarian and dairy-free.",
  "questions_remaining": 5,
  "field_revealed": "dietary_restrictions"
}
```

**propose_plan() result (terminal):**
```json
{
  "type": "resolution",
  "score": 0.83,
  "breakdown": {
    "format_check": 1.0,
    "field_match_f1": 0.85,
    "info_gain": 0.80,
    "question_efficiency": 0.83,
    "hallucination_check": 0.95
  },
  "expected_profile": { "...": "..." },
  "submitted_plan": { "...": "..." },
  "correct": false,
  "questions_asked": 3
}
```

## Reset Contract

```python
def reset(
    self,
    seed: Optional[int] = None,
    episode_id: Optional[str] = None,
    task_id: str = "medium",   # via **kwargs
) -> CallToolObservation:
    """
    1. Generate new scenario via ProfileGenerator(seed, task_id)
    2. Reset all internal trackers + ClarifyState
    3. Reset rubric trajectory state (self._reset_rubric())
    4. Return observation with type="task"
    """
```

## Step Contract

```python
def step(
    self,
    action: Action,
    timeout_s: Optional[float] = None,
    **kwargs,
) -> CallToolObservation:
    """
    Routes:
    - ListToolsAction → return tool list (free, no step counted)
    - CallToolAction → dispatch to ask_question/propose_plan/get_task_info
    - Other → error observation
    
    Always:
    - Increment step_count
    - Compute per-step shaping reward
    - On terminal: invoke rubric, populate breakdown
    - Honor max_steps from openenv.yaml
    """
```

## State Property

```python
@property
def state(self) -> ClarifyState:
    """Return the current public-facing state. Hidden profile NOT included."""
```

Called by `/state` HTTP endpoint and the OpenEnv framework's introspection.

## get_metadata Override

```python
def get_metadata(self) -> EnvironmentMetadata:
    return EnvironmentMetadata(
        name="ClarifyRL — AskBeforeYouAct",
        description="Train LLMs to ask clarifying questions instead of hallucinating.",
        readme_content=<read README.md>,
        version="0.1.0",
        author="Team Bhole Chature",
        documentation_url="https://huggingface.co/spaces/anurag203/clarify-rl",
    )
```

## Validation Hooks

The env exposes its rubric via `self.rubric`:

```python
# Judges or eval scripts can introspect:
for name, sub_rubric in env.rubric.named_rubrics():
    print(f"{name}: {sub_rubric.last_score}")
```

This satisfies the "composable rubrics > monolithic" judging criterion.
