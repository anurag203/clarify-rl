"""
End-to-end test through ClarifyClient (the path TRL training actually uses).

Server stays a single env instance per WebSocket session; reset+step+state
all share state. Run uvicorn first, then this script.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from client import ClarifyClient


def main() -> int:
    env = ClarifyClient(base_url="http://127.0.0.1:7860").sync()
    with env:
        print("--- list tools ---")
        tools = env.list_tools()
        print([t.name for t in tools])

        print("\n--- reset(seed=7, task_id=medium) ---")
        obs = env.reset(seed=7, task_id="medium")
        print(f"reward={obs.reward}  done={obs.done}")
        print("observation:", obs.observation)

        print("\n--- ask_question: order id ---")
        result = env.call_tool("ask_question", question="what is the order id?")
        print("result:", result)

        print("\n--- ask_question: item issue ---")
        result = env.call_tool("ask_question", question="what's wrong with the order?")
        print("result:", result)

        print("\n--- ask_question: refund/replace ---")
        result = env.call_tool("ask_question", question="refund or replace?")
        print("result:", result)

        print("\n--- ask_question: urgency ---")
        result = env.call_tool("ask_question", question="when do you need this?")
        print("result:", result)

        print("\n--- propose_plan ---")
        plan = json.dumps({
            "order_id": "#4521",
            "item_issue": "wrong-item",
            "refund_or_replace": "replace",
            "urgency": "high",
        })
        result = env.call_tool("propose_plan", plan=plan)
        print("result:", result)

        print("\n--- final state ---")
        state = env.state()
        print(state)

    return 0


if __name__ == "__main__":
    sys.exit(main())
