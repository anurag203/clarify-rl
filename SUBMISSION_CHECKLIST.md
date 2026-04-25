# Submission Checklist

## Required files at repo root

- `inference.py`
- `openenv.yaml`
- `Dockerfile`
- `README.md`
- `pyproject.toml`
- `server/`
- `scenarios/`

## What Must Be True Before Submit

- `pytest -m 'not integration' -q` passes
- `openenv validate .` passes
- Hugging Face Space is updated with the latest files
- The Space responds on `/health` and `/reset`
- No secrets are committed or left in `.env`

## Inference Modes

- `BASELINE_MODE=policy` — deterministic scripted agent (no LLM needed)
- `BASELINE_MODE=hybrid` — LLM with policy fallback (default)
- `BASELINE_MODE=llm` — pure LLM (not recommended without credits)

## Push Reminder

Top-level tree must look like:

```text
Dockerfile
README.md
inference.py
openenv.yaml
pyproject.toml
server/
scenarios/
tests/
scripts/
```
