# 05 — Scenario Design

How we generate scenarios procedurally with **zero data collection**.

## 5 Task Families (mix of high-stakes + personal)

We deliberately blend **3 high-stakes** (frontier AI relevance) + **2 personal** (universal relatability) families. This gives us storytelling range and signals "trains real-world AI agents" instead of "toy demo."

| # | Family | Stakes | Example Request | Field Pool | Critical Fields |
|---|--------|--------|----------------|-----------|-----------------|
| 1 | `coding_requirements` | high | "Build me an API." | 10 | stack, scale, auth, datastore, deployment_target |
| 2 | `medical_intake` | high | "I'm not feeling well." | 9 | primary_symptom, duration, severity, age_band |
| 3 | `support_triage` | high | "My order is wrong." | 9 | order_id, item_issue, refund_or_replace, urgency |
| 4 | `meeting_scheduling` | low | "Schedule a sync." | 8 | participants, date, time, duration |
| 5 | `event_planning` | low | "Plan a birthday party." | 9 | event_type, guest_count, date, venue |

> **Note**: `medical_intake` is **non-diagnostic intake only** (gathering presenting info, not advising). We do not claim clinical safety.

## Hidden Field Schema (universe)

All possible field keys + value vocabularies. The generator picks 2-12 of these per scenario based on family + difficulty.

```python
FIELD_VOCAB = {
    # Coding requirements
    "stack":              ["python+fastapi","node+express","go+gin","rust+axum","ruby+rails"],
    "scale":              ["<100 users","1k users","10k users","1M users"],
    "auth":               ["none","api-key","jwt","oauth2","saml-sso"],
    "datastore":          ["postgres","mysql","mongodb","sqlite","redis-only"],
    "deployment_target":  ["aws-lambda","kubernetes","fly.io","heroku","on-prem"],
    "language_version":   ["python3.11","python3.12","node18","node20"],
    "test_coverage":      ["none","unit-only","unit+integration","full-e2e"],

    # Medical intake (NON-DIAGNOSTIC — for triage info-gathering only)
    "primary_symptom":    ["headache","fever","cough","fatigue","nausea","rash","pain"],
    "duration":           ["<1 hour","1-24 hours","1-7 days","1-4 weeks","chronic"],
    "severity":           ["mild","moderate","severe"],
    "age_band":           ["child","teen","adult","senior"],
    "prior_conditions":   ["none","diabetes","hypertension","asthma","other"],
    "medications":        ["none","prescription","otc","both"],

    # Customer support
    "order_id":           ["#XXXX","none-provided"],
    "item_issue":         ["wrong-item","damaged","missing","late","never-arrived"],
    "refund_or_replace":  ["refund","replace","store-credit","unsure"],
    "urgency":            ["low","medium","high"],
    "channel_preferred":  ["email","phone","chat"],

    # Personal: meeting
    "participants":       ["whole team","just me and X","leadership","external client"],
    "date":               ["today","tomorrow","this week","next week"],
    "time":               ["morning","afternoon","evening","flexible"],
    "duration_minutes":   [15,30,45,60,90],
    "platform":           ["zoom","google-meet","in-person","phone"],

    # Personal: event
    "event_type":         ["birthday","farewell","anniversary","team-building","baby-shower"],
    "guest_count":        [5,10,20,50,100],
    "venue":              ["restaurant","home","office","outdoor","rented hall"],
    "budget_band":        ["<$100","$100-500","$500-2000","$2000+"],
    "theme":              ["casual","formal","themed","surprise"],
    "dietary_constraints":["none","vegetarian","vegan","mixed"],
}
```

## Difficulty Tiers

| Difficulty | # hidden fields | Min critical | Max steps |
|-----------|-----------------|--------------|-----------|
| easy      | 2-3             | all critical | 8         |
| medium    | 4-5             | 3 critical   | 10        |
| hard      | 6-7             | 4 critical   | 12        |

Higher difficulty = more fields agent could waste questions on, harder to prioritize.

## Procedural Generation Algorithm

```python
def generate(seed: int, task_id: str = "medium") -> Scenario:
    rng = random.Random(seed)

    # 1. Pick task family (uniform across all 5)
    family = rng.choice([
        "coding_requirements","medical_intake","support_triage",
        "meeting_scheduling","event_planning",
    ])

    # 2. Pick difficulty target
    n_fields = {"easy": rng.randint(2,3),
                "medium": rng.randint(5,6),
                "hard": rng.randint(8,12)}[task_id]

    # 3. Sample field set from family-allowed fields
    valid_fields = TASK_FIELDS[family]
    chosen_fields = rng.sample(valid_fields, k=min(n_fields, len(valid_fields)))

    # 4. Sample values
    profile = {f: rng.choice(FIELD_VOCAB[f]) for f in chosen_fields}

    # 5. Pick a request template (intentionally vague)
    template = rng.choice(REQUEST_TEMPLATES[family])
    request_text = template

    # 6. Determine critical subset
    critical = set(chosen_fields) & set(CRITICAL_BY_FAMILY[family])
    if len(critical) < MIN_CRITICAL[task_id]:
        extra = rng.sample(
            list(set(CRITICAL_BY_FAMILY[family]) - critical),
            k=MIN_CRITICAL[task_id] - len(critical),
        )
        for f in extra:
            chosen_fields.append(f)
            profile[f] = rng.choice(FIELD_VOCAB[f])
            critical.add(f)

    return {
        "family": family,
        "task_id": task_id,
        "request": request_text,
        "hidden_profile": profile,
        "critical_fields": list(critical),
        "field_keys": list(profile.keys()),
        "max_steps": MAX_STEPS_BY_TASK[task_id],
    }
```

## Request Templates (intentionally vague)

```python
REQUEST_TEMPLATES = {
    "coding_requirements": [
        "Build me an API.",
        "Write a service for me.",
        "I need a backend.",
        "Set up an endpoint.",
        "Make me a web service.",
    ],
    "medical_intake": [
        "I'm not feeling well.",
        "Something's off with my health.",
        "I have a problem.",
        "I need some help.",
    ],
    "support_triage": [
        "My order is wrong.",
        "There's an issue with my purchase.",
        "Something went wrong.",
        "I need help with an order.",
    ],
    "meeting_scheduling": [
        "Schedule a sync.",
        "Set up a meeting.",
        "Get a call on the calendar.",
    ],
    "event_planning": [
        "Plan a birthday party.",
        "Organize a team event.",
        "Set up a celebration.",
    ],
}
```

Notice: NONE of the request texts include any actual values. The agent MUST ask to get them.

## Output Tone (per family)

The user simulator's answer style is family-aware to keep traces feeling real:

| Family | User-sim tone |
|--------|--------------|
| `coding_requirements` | Technical: *"Stack should be python+fastapi."* |
| `medical_intake` | Personal/uncertain: *"It's been about 3 days."* |
| `support_triage` | Frustrated: *"Order #4521. Item arrived damaged."* |
| `meeting_scheduling` | Casual: *"Just me and Priya, tomorrow afternoon works."* |
| `event_planning` | Excited: *"Birthday party for ~20 people, casual vibe."* |

This makes demo traces feel **alive** in the video, not robotic.

## User Simulator (rule-based, fast)

The simulator answers questions from the hidden profile via keyword matching against `FIELD_KEYWORDS`. See `server/user_simulator.py` for the full keyword map. Latency <1ms per call (pure Python dict scan).

### FIELD_KEYWORDS (mapping Q → field) — abbreviated

```python
FIELD_KEYWORDS = {
    # coding
    "stack":              ["stack","language","framework","tech"],
    "scale":              ["scale","users","traffic","load","rps"],
    "auth":               ["auth","authentication","login","sso","jwt"],
    "datastore":          ["database","db","storage","persist"],
    "deployment_target":  ["deploy","host","cloud","aws","kubernetes"],
    # medical
    "primary_symptom":    ["symptom","what's wrong","what hurts","how do you feel"],
    "duration":           ["how long","since when","duration","when did"],
    "severity":           ["severe","mild","intense","how bad","scale"],
    "age_band":           ["age","how old","young","elderly"],
    # support
    "order_id":           ["order","order id","number","reference"],
    "item_issue":         ["what's wrong","damaged","missing","wrong"],
    "refund_or_replace":  ["refund","replace","return","credit"],
    "urgency":            ["when","need by","urgent","asap"],
    # meeting
    "participants":       ["who","participants","attend","join"],
    "date":               ["when","date","day"],
    "time":               ["time","hour","what time"],
    "duration_minutes":   ["how long","duration","minutes"],
    # event
    "event_type":         ["kind","what kind","type"],
    "guest_count":        ["how many","guest","headcount","size"],
    "venue":              ["where","venue","location"],
    "budget_band":        ["budget","cost","spend","price"],
}
```

## Eval Set Strategy

- **Training set**: scenarios generated with seeds 0-999 (live during training)
- **Held-out eval set**: scenarios with seeds 10000-10099 (fixed, in `scenarios/eval_held_out.json`)
- Eval set is **stratified across families and difficulties**: 20 scenarios per family × balanced easy/medium/hard
- This guarantees evaluation is on truly unseen scenarios across ALL 5 families → robust generalization story

## Anti-Memorization

- Profile values randomly sampled from per-field vocabularies
- Combinatorial space: **5 families × ~10 fields × ~6 values × difficulty variation ≈ tens of millions of unique scenarios**
- Agent CANNOT memorize; must learn the meta-skill of "ask before acting" and which fields are critical per family
