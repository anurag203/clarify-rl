"""
ClarifyRL — Baseline Inference Script
======================================

MANDATORY:
- Before submitting, ensure the following variables are defined in your
  environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root
  directory of the project.
- Participants must use OpenAI Client for all LLM calls using above variables.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import textwrap
import time
from typing import Optional

try:
    import truststore; truststore.inject_into_ssl()
except ImportError:
    pass

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
BASELINE_MODE = os.getenv("BASELINE_MODE", "hybrid").lower()

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

TEMPERATURE = 0.7
MAX_TOKENS = 300
MAX_LLM_STEPS_PER_TASK = int(os.getenv("MAX_LLM_STEPS_PER_TASK", "8"))
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a helpful assistant that books and plans things for users.
    When you receive a request, you may not have all the information needed.
    You can:
    1. ASK clarifying questions using the ask_question(question) tool (max 6 total)
    2. PROPOSE a final plan using propose_plan(plan) when you have enough info
    3. GET the task description again using get_task_info()

    RESPOND WITH EXACTLY ONE TOOL CALL PER TURN:
    TOOL: tool_name
    ARGS: {"arg1": "value1"}

    Examples:
    TOOL: ask_question
    ARGS: {"question": "What is your budget?"}

    TOOL: propose_plan
    ARGS: {"plan": "{\\"stack\\": \\"python+fastapi\\", \\"scale\\": \\"1k users\\"}"}

    TOOL: get_task_info
    ARGS: {}

    Be efficient: ask only what you NEED, then propose a plan.
    Do NOT include preferences in the plan that you weren't told about.
""")

POLICY_PLANS = {
    "easy": [
        ("get_task_info", {}),
        ("ask_question", {"question": "What is the main requirement?"}),
        ("ask_question", {"question": "Any specific preferences or constraints?"}),
    ],
    "medium": [
        ("get_task_info", {}),
        ("ask_question", {"question": "What is the main requirement?"}),
        ("ask_question", {"question": "What are the specific details needed?"}),
        ("ask_question", {"question": "Any constraints or preferences?"}),
        ("ask_question", {"question": "What is the timeline or deadline?"}),
    ],
    "hard": [
        ("get_task_info", {}),
        ("ask_question", {"question": "What is the main requirement?"}),
        ("ask_question", {"question": "What are the technical specifications?"}),
        ("ask_question", {"question": "What is the scale or scope?"}),
        ("ask_question", {"question": "Any constraints or limitations?"}),
        ("ask_question", {"question": "What is the timeline?"}),
        ("ask_question", {"question": "Any other preferences?"}),
    ],
}


def create_client() -> Optional[OpenAI]:
    if BASELINE_MODE == "policy":
        return None
    if not API_KEY:
        print("[DEBUG] No API key found; policy fallback will be used.", flush=True)
        return None
    try:
        return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        print(f"[DEBUG] Failed to create OpenAI client: {exc}", flush=True)
        return None


def parse_tool_call(response_text: str) -> tuple[Optional[str], dict]:
    cleaned = _strip_reasoning(response_text)
    tool_match = re.search(r"TOOL:\s*(\w+)", cleaned, re.IGNORECASE)
    raw_args = _extract_args_block(cleaned)

    if tool_match:
        tool_name = tool_match.group(1).strip()
        args = {}
        if raw_args:
            args = _load_json_like(raw_args)
        return tool_name, args

    json_tool_name, json_tool_args = _parse_json_tool_call(cleaned)
    if json_tool_name:
        return json_tool_name, json_tool_args

    func_match = re.search(r"(\w+)\s*\(([^)]*)\)", cleaned)
    if func_match:
        tool_name = func_match.group(1).strip()
        if tool_name in ("ask_question", "propose_plan", "get_task_info"):
            raw_args_str = func_match.group(2).strip()
            args = _parse_positional_args(tool_name, raw_args_str)
            return tool_name, args

    action_match = re.search(
        r'Action:\s*(\w+)\((?:(\w+)\s*=\s*["\'](.+?)["\']|([^)]*))\)',
        cleaned, re.DOTALL,
    )
    if action_match:
        tool_name = action_match.group(1)
        if action_match.group(2) and action_match.group(3) is not None:
            key = action_match.group(2)
            val = action_match.group(3).replace('\\"', '"').replace("\\'", "'")
            return tool_name, {key: val}
        elif action_match.group(4):
            raw = action_match.group(4).strip()
            if "=" in raw:
                k, _, v = raw.partition("=")
                return tool_name, {k.strip(): v.strip().strip("\"'")}
            return tool_name, {}

    return None, {}


def _strip_reasoning(response_text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("```json", "```")
    cleaned = cleaned.replace("```tool", "```")
    return cleaned.strip()


def _extract_args_block(response_text: str) -> Optional[str]:
    args_marker = re.search(r"ARGS:\s*", response_text, re.IGNORECASE)
    if not args_marker:
        return None
    start = response_text.find("{", args_marker.end())
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(response_text)):
        char = response_text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return response_text[start:index + 1]
    return None


def _candidate_json_objects(text: str) -> list[str]:
    candidates = []
    start = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start = index
                depth = 1
            continue
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:index + 1])
                start = None
    return candidates


def _load_json_like(raw: str) -> dict:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        normalized = raw.strip()
        normalized = re.sub(r"(\w+)\s*=", r'"\1": ', normalized)
        normalized = normalized.replace("'", '"')
        try:
            parsed = json.loads(normalized)
        except json.JSONDecodeError:
            return _parse_args_fallback(raw)
    return parsed if isinstance(parsed, dict) else {}


def _parse_json_tool_call(response_text: str) -> tuple[Optional[str], dict]:
    for candidate in _candidate_json_objects(response_text):
        parsed = _load_json_like(candidate)
        if not parsed:
            continue
        tool_name = (
            parsed.get("tool") or parsed.get("tool_name")
            or parsed.get("name") or parsed.get("action")
        )
        if not isinstance(tool_name, str):
            continue
        args = parsed.get("args") or parsed.get("arguments") or parsed.get("parameters") or {}
        if isinstance(args, str) and args.strip().startswith("{"):
            args = _load_json_like(args)
        if not isinstance(args, dict):
            args = {}
        return tool_name.strip(), args
    return None, {}


def _parse_args_fallback(raw: str) -> dict:
    args = {}
    for match in re.finditer(r'"(\w+)"\s*:\s*"([^"]*)"', raw):
        args[match.group(1)] = match.group(2)
    for match in re.finditer(r'"(\w+)"\s*:\s*(\d+)', raw):
        args[match.group(1)] = int(match.group(2))
    return args


def _parse_positional_args(tool_name: str, raw_args: str) -> dict:
    if not raw_args:
        return {}
    parts = [p.strip().strip("'\"") for p in raw_args.split(",")]
    arg_map = {
        "ask_question": ["question"],
        "propose_plan": ["plan"],
    }
    param_names = arg_map.get(tool_name, [])
    args = {}
    for i, part in enumerate(parts):
        if i < len(param_names):
            args[param_names[i]] = part
    return args


def _parse_result_field(obs: dict) -> str:
    result_raw = obs.get("result", "")
    if not result_raw:
        return str(obs)
    try:
        parsed = json.loads(result_raw)
        if isinstance(parsed, dict) and "tool_result" in parsed:
            return parsed["tool_result"]
        return json.dumps(parsed, indent=2)
    except (json.JSONDecodeError, TypeError):
        return str(result_raw)


def _next_policy_action(
    task_id: str, step_index: int, request_text: str, revealed: dict,
) -> tuple[str, dict]:
    plan = POLICY_PLANS.get(task_id, POLICY_PLANS["medium"])
    if step_index < len(plan):
        return plan[step_index]
    return ("propose_plan", {"plan": json.dumps(revealed)})


def _choose_action(
    task_id: str,
    messages: list[dict],
    llm_client: Optional[OpenAI],
    step_index: int,
    llm_attempts: int,
    request_text: str,
    revealed: dict,
) -> tuple[str, dict, bool, int]:
    policy_action = _next_policy_action(task_id, step_index, request_text, revealed)

    if BASELINE_MODE == "policy" or llm_client is None:
        return policy_action[0], policy_action[1], True, llm_attempts

    if llm_attempts >= MAX_LLM_STEPS_PER_TASK:
        return policy_action[0], policy_action[1], True, llm_attempts

    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        assistant_msg = response.choices[0].message.content or ""
        llm_attempts += 1
    except Exception as exc:
        print(f"    LLM unavailable, switching to policy: {exc}")
        return policy_action[0], policy_action[1], True, MAX_LLM_STEPS_PER_TASK

    tool_name, args = parse_tool_call(assistant_msg)
    if tool_name and tool_name in ("ask_question", "propose_plan", "get_task_info"):
        messages.append({"role": "assistant", "content": assistant_msg})
        return tool_name, args, False, llm_attempts

    if tool_name:
        print(f"    LLM suggested unknown tool {tool_name}; using policy instead.")
    else:
        print("    Could not parse tool call; using policy instead.")
    messages.append({"role": "assistant", "content": assistant_msg})
    return policy_action[0], policy_action[1], True, MAX_LLM_STEPS_PER_TASK


def _get_ws_url() -> str:
    ws_url = ENV_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
    return f"{ws_url}/ws"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str] = None,
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


async def ws_reset(ws, task_id: str) -> dict:
    await ws.send(json.dumps({"type": "reset", "data": {"task_id": task_id}}))
    resp = json.loads(await ws.recv())
    if resp.get("type") == "error":
        return {"observation": {}, "reward": 0.0, "done": False, "error": resp.get("data", {})}
    data = resp.get("data", {})
    return {
        "observation": data.get("observation", {}),
        "reward": data.get("reward", 0.0),
        "done": data.get("done", False),
    }


async def ws_step(ws, tool_name: str, args: dict) -> dict:
    action = {"type": "call_tool", "tool_name": tool_name, "arguments": args}
    await ws.send(json.dumps({"type": "step", "data": action}))
    resp = json.loads(await ws.recv())
    if resp.get("type") == "error":
        return {
            "observation": {"result": json.dumps({"error": resp.get("data", {}).get("message", "Unknown error")})},
            "reward": 0.0,
            "done": False,
        }
    data = resp.get("data", {})
    return {
        "observation": data.get("observation", {}),
        "reward": data.get("reward", 0.0),
        "done": data.get("done", False),
    }


def wait_for_server(base_url: str, timeout: int = 120) -> bool:
    import urllib.request
    import urllib.error
    import ssl

    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except ImportError:
        pass

    urls = [f"{base_url}/health", f"{base_url}/"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        for url in urls:
            try:
                req = urllib.request.urlopen(url, timeout=5, context=ctx)
                if req.status == 200:
                    return True
            except Exception:
                pass
        time.sleep(2)
    return False


async def run_task_async(llm_client: Optional[OpenAI], task_id: str, task_title: str) -> float:
    rewards_list: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env="clarify_rl", model=MODEL_NAME or "policy")

    try:
        import websockets
        print(f"\nTask: {task_id} ({task_title})")
        print("-" * 50)

        ws_url = _get_ws_url()
        async with websockets.connect(ws_url, open_timeout=30, close_timeout=10) as ws:
            reset_result = await ws_reset(ws, task_id)
            obs = reset_result.get("observation", {})

            initial_result = obs.get("result", "")
            try:
                initial_data = json.loads(initial_result) if initial_result else {}
            except (json.JSONDecodeError, TypeError):
                initial_data = {}

            request_text = initial_data.get("request", str(initial_data))
            max_steps = initial_data.get("max_steps", 10)

            initial_context = (
                f"USER REQUEST:\n{request_text}\n\n"
                f"You have {max_steps} steps. Available tools: "
                f"ask_question(question), propose_plan(plan), get_task_info()\n"
                f"Ask clarifying questions to gather missing info, then propose a plan."
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": initial_context},
            ]

            task_step_budget = max_steps
            llm_attempts = 0
            revealed: dict = {}

            for step in range(1, task_step_budget + 1):
                tool_name, args, used_policy, llm_attempts = _choose_action(
                    task_id, messages, llm_client, step - 1, llm_attempts,
                    request_text, revealed,
                )

                args_str = json.dumps(args) if args else "{}"
                action_str = f"{tool_name}({args_str})"
                source = "policy" if used_policy else "llm"
                print(f"  Step {step}: [{source}] {action_str}")

                result = await ws_step(ws, tool_name, args)
                obs_data = result.get("observation", {})
                reward = result.get("reward", 0.0)
                done = result.get("done", False)
                tool_result = _parse_result_field(obs_data)

                try:
                    result_parsed = json.loads(tool_result) if isinstance(tool_result, str) else tool_result
                    if isinstance(result_parsed, dict):
                        for k, v in result_parsed.items():
                            if k not in ("error", "episode_done", "questions_remaining", "fields_revealed"):
                                revealed[k] = v
                except (json.JSONDecodeError, TypeError):
                    pass

                rewards_list.append(reward)
                steps_taken = step

                log_step(step=step, action=action_str, reward=reward, done=done, error=None)

                if len(str(tool_result)) > 1500:
                    tool_result = str(tool_result)[:1500] + "... [truncated]"

                if used_policy:
                    messages.append({
                        "role": "assistant",
                        "content": f"TOOL: {tool_name}\nARGS: {json.dumps(args)}",
                    })
                messages.append({
                    "role": "user",
                    "content": f"Tool result:\n{tool_result}\n\nReward: {reward}\nSteps remaining: {max_steps - step}",
                })

                if done:
                    try:
                        terminal_data = json.loads(obs_data.get("result", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        terminal_data = {}
                    score = terminal_data.get("final_score", terminal_data.get("score", reward))
                    if score is None:
                        score = reward
                    success = score >= SUCCESS_SCORE_THRESHOLD
                    breakdown = terminal_data.get("score_breakdown", {})
                    print(f"  --> Episode ended. Score: {score}")
                    if breakdown:
                        for comp, val in breakdown.items():
                            print(f"      {comp}: {val}")
                    break
            else:
                score = sum(rewards_list) if rewards_list else 0.0
                score = min(max(score, 0.0), 1.0)
                success = score >= SUCCESS_SCORE_THRESHOLD
                print(f"  --> Max steps reached. Score: {score}")

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards_list)

    return score


def main():
    print("=" * 60)
    print("  ClarifyRL — Baseline Inference")
    print("=" * 60)
    print(f"Mode: {BASELINE_MODE}")
    print(f"API: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Environment: {ENV_BASE_URL}")

    tasks = [
        ("easy", "Mild Ambiguity (2-3 fields)"),
        ("medium", "Moderate Ambiguity (4-5 fields)"),
        ("hard", "High Ambiguity (6-7 fields)"),
    ]

    print("\nWaiting for environment server...", flush=True)
    server_ok = wait_for_server(ENV_BASE_URL)
    if not server_ok:
        print("ERROR: Environment server not reachable.", flush=True)
        for task_id, title in tasks:
            log_start(task=task_id, env="clarify_rl", model=MODEL_NAME or "policy")
            log_end(success=False, steps=0, score=0.0, rewards=[])
        print("Emitted zero-score logs for all tasks. Exiting.", flush=True)
        sys.exit(0)
    print("Server is ready.\n", flush=True)

    llm_client = create_client()

    task_timeout = 300

    scores = {}
    for task_id, title in tasks:
        try:
            score = asyncio.run(
                asyncio.wait_for(run_task_async(llm_client, task_id, title), timeout=task_timeout)
            )
        except asyncio.TimeoutError:
            print(f"[DEBUG] Task {task_id} timed out after {task_timeout}s", flush=True)
            log_start(task=task_id, env="clarify_rl", model=MODEL_NAME or "policy")
            log_end(success=False, steps=0, score=0.0, rewards=[])
            score = 0.0
        except Exception as exc:
            print(f"[DEBUG] Task {task_id} crashed: {exc}", flush=True)
            score = 0.0
        scores[task_id] = score

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for task_id, title in tasks:
        print(f"  {task_id:<8s} ({title}): {scores.get(task_id, 0.0):.2f}")

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n  Average: {avg:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[DEBUG] Fatal error in main: {exc}", flush=True)
        sys.exit(0)
