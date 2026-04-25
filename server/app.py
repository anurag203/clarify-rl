"""
FastAPI entry point for the ClarifyRL OpenEnv environment.

`openenv.yaml` references this module as `server.app:app`.
Run locally with:  uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from server.clarify_environment import ClarifyEnvironment


app = create_app(
    env=ClarifyEnvironment,
    action_cls=CallToolAction,
    observation_cls=CallToolObservation,
    env_name="clarify_rl",
)


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
