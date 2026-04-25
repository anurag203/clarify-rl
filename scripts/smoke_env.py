"""Smoke test for ClarifyEnvironment: reset -> oracle ask -> perfect plan."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.clarify_environment import ClarifyEnvironment
from server.user_simulator import FIELD_KEYWORDS
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction


def main() -> None:
    env = ClarifyEnvironment()

    print("=== reset(seed=7, task_id=medium) ===")
    obs = env.reset(seed=7, task_id="medium")
    print("reset.result:")
    print(json.dumps(obs.result, indent=2, default=str))
    print("state:", env.state.model_dump())

    print("\n=== list tools ===")
    tools_obs = env.step(ListToolsAction())
    print("tools:", [t.name for t in tools_obs.tools])

    print("\n=== call get_task_info ===")
    obs = env.step(CallToolAction(tool_name="get_task_info", arguments={}))
    print(f"reward={obs.reward}  done={obs.done}")
    print("result:", obs.result)

    print("\n=== ask each critical question (oracle) ===")
    crit = env._scenario["critical_fields"]
    print("critical fields:", crit)
    for fkey in crit:
        kw = FIELD_KEYWORDS[fkey][0]
        question = f"what is the {kw}?"
        obs = env.step(
            CallToolAction(
                tool_name="ask_question",
                arguments={"question": question},
            )
        )
        print(f"  Q='{question}'  reward={obs.reward}  done={obs.done}")
        print(f"    result={obs.result}")

    print("\nstate after asking:", env.state.model_dump())

    print("\n=== submit perfect plan ===")
    plan = json.dumps(env._scenario["hidden_profile"])
    obs = env.step(
        CallToolAction(
            tool_name="propose_plan",
            arguments={"plan": plan},
        )
    )
    print(f"reward={obs.reward:.3f}  done={obs.done}")
    print("result:")
    print(json.dumps(obs.result, indent=2, default=str))

    print("\nfinal state:", env.state.model_dump())


if __name__ == "__main__":
    main()
