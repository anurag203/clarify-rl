"""
FastAPI entry point for the ClarifyRL OpenEnv environment.

`openenv.yaml` references this module as `server.app:app`.
Run locally with:  uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import gradio as gr
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from server.clarify_environment import ClarifyEnvironment
from server.gradio_ui import build_gradio_ui


# max_concurrent_envs=64: each ClarifyEnvironment is in-memory only (no GPU
# / no shared state — see SUPPORTS_CONCURRENT_SESSIONS in clarify_environment.py),
# so we can comfortably fan out to multiple parallel HF Jobs evals + lingering
# orphan sessions from disconnected eval clients without hitting CAPACITY_REACHED.
app = create_app(
    env=ClarifyEnvironment,
    action_cls=CallToolAction,
    observation_cls=CallToolObservation,
    env_name="clarify_rl",
    max_concurrent_envs=64,
)

_gradio_demo = build_gradio_ui()
app = gr.mount_gradio_app(app, _gradio_demo, path="/")


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
