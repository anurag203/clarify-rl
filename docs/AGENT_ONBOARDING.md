# Agent Onboarding — paste this at the start of any new chat

> Use this file when starting a new chat with an agent that does **not** auto-load `.windsurf/rules/` (e.g. Cursor, ChatGPT, Claude desktop, web Claude). Inside Windsurf, `.windsurf/rules/clarify-rl.md` loads automatically — you can skip this.

## The prompt to paste

```
Before doing anything else, read these files IN ORDER and confirm you've read them:

1. clarify-rl/docs/AGENT_ONBOARDING.md   (this file)
2. clarify-rl/docs/STATUS.md             (current state)
3. clarify-rl/docs/SESSION_LOG.md        (last 3 entries only)
4. clarify-rl/docs/00-overview.md        (the pitch)
5. clarify-rl/docs/README.md             (doc index)

After reading, give me a 5-bullet summary of:
- What ClarifyRL is
- Current phase + last-completed task
- What's in progress
- What's the next step
- Any open blockers

Then ask me what to work on. Do NOT pivot any LOCKED decisions listed below
without my explicit approval. At the end of our session, append a SESSION_LOG
entry and update STATUS.md.
```

---

## LOCKED decisions (do not pivot)

- **Idea**: ClarifyRL / AskBeforeYouAct — train epistemic humility via RL
- **Theme**: #5 Wild Card (primary) + #3.2 Personalized + #2 Long-Horizon (secondary)
- **Headline metric**: hallucination rate ~90% → ~3% on 100 held-out scenarios
- **5 task families**: `coding_requirements`, `medical_intake`, `support_triage`, `meeting_scheduling`, `event_planning`
- **Stack**: OpenEnv 0.2.2 + MCPEnvironment + FastMCP + Unsloth + TRL GRPO + Qwen2.5-1.5B-Instruct
- **Compute**: HF Jobs `t4-small` (primary) + free Colab T4 (fallback)
- **Starter notebook**: fork of TRL `openenv_wordle_grpo.ipynb`
- **Rubric**: `Sequential(Gate(FormatCheck), WeightedSum([FieldMatch 0.50, InfoGain 0.20, QuestionEfficiency 0.15, HallucinationCheck 0.15]))`
- **MCP tool names**: `ask_question`, `propose_plan`, `get_task_info`
- **Team**: Bhole Chature (Anurag + Kanan)
- **Deadline**: Apr 26, 2026, 5:00 PM IST

## End-of-session ritual (mandatory)

1. Append entry to `clarify-rl/docs/SESSION_LOG.md`:

    ```markdown
    ## YYYY-MM-DD HH:MM IST — <agent name / chat tag>
    - Did: <bullet>
    - Did: <bullet>
    - Did: <bullet>
    - Decisions: <any new locked decision> (or "none")
    - Next: <what should happen next>
    ```

2. Update `clarify-rl/docs/STATUS.md` — bump `last-completed`, `in-progress`, `next-step`, `blockers`.

3. Confirm in chat: *"STATUS + SESSION_LOG updated."*

## Repo map (high level)

```
hackathon_proj/
└── clarify-rl/
    ├── .windsurf/rules/clarify-rl.md   ← auto-loaded by Windsurf
    ├── docs/                            ← all design docs (00-10) + STATUS + SESSION_LOG
    ├── server/                          ← env + rubrics + simulator (in progress)
    ├── client.py                        ← thin client (must NOT import server)
    ├── models.py                        ← action/observation dataclasses
    ├── openenv.yaml                     ← OpenEnv manifest
    ├── pyproject.toml
    ├── Dockerfile
    └── hackathon_data/                  ← official hackathon docs (FAQs, themes, ceremony PDF)
```

## Where to find what

- **"What does the project do?"** → `clarify-rl/docs/00-overview.md`
- **"What's done / in progress?"** → `clarify-rl/docs/STATUS.md`
- **"What did the last agent change?"** → `clarify-rl/docs/SESSION_LOG.md`
- **"What's the rubric design?"** → `clarify-rl/docs/04-rubric-design.md`
- **"How do scenarios get generated?"** → `clarify-rl/docs/05-scenario-design.md`
- **"What's the training plan?"** → `clarify-rl/docs/06-training-plan.md`
- **"How is this pitched to judges?"** → `clarify-rl/docs/10-positioning.md`
- **"What did the hackathon organizers say?"** → `clarify-rl/hackathon_data/`
