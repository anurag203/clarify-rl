"""
FastAPI entry point for the ClarifyRL OpenEnv environment.

`openenv.yaml` references this module as `server.app:app`.
Run locally with:  uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from server.clarify_environment import ClarifyEnvironment


# max_concurrent_envs=8 supports up to 4 parallel training jobs simultaneously
# (each job uses 2 sessions for num_generations=2 rollouts, +margin). Each
# session gets a fresh ClarifyEnvironment instance — see SUPPORTS_CONCURRENT_SESSIONS
# in clarify_environment.py for safety reasoning.
app = create_app(
    env=ClarifyEnvironment,
    action_cls=CallToolAction,
    observation_cls=CallToolObservation,
    env_name="clarify_rl",
    max_concurrent_envs=8,
)


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
