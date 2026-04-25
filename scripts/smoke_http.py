"""HTTP smoke test for the running ClarifyRL server."""

import json
import sys
import time
from urllib import error, request


BASE = "http://127.0.0.1:8000"


def _post(path: str, payload: dict) -> tuple[int, str]:
    req = request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.read().decode()
    except error.HTTPError as e:
        return e.code, e.read().decode()


def _get(path: str) -> tuple[int, str]:
    try:
        with request.urlopen(BASE + path, timeout=15) as resp:
            return resp.status, resp.read().decode()
    except error.HTTPError as e:
        return e.code, e.read().decode()


def _wait_until_up(timeout_s: int = 20) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with request.urlopen(BASE + "/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (error.URLError, ConnectionError, OSError):
            time.sleep(0.5)
    return False


def main() -> int:
    if not _wait_until_up():
        print("ERROR: server did not respond to /health within 20s")
        return 1

    print("--- /health ---")
    code, body = _get("/health")
    print(f"HTTP {code}\n{body}\n")

    print("--- /reset ---")
    code, body = _post("/reset", {"seed": 7, "task_id": "medium"})
    print(f"HTTP {code}\n{body}\n")

    print("--- /step ask_question ---")
    code, body = _post(
        "/step",
        {
            "action": {
                "tool_name": "ask_question",
                "arguments": {"question": "what is the order id?"},
            }
        },
    )
    print(f"HTTP {code}\n{body}\n")

    print("--- /state ---")
    code, body = _get("/state")
    print(f"HTTP {code}\n{body}\n")

    print("--- /metadata ---")
    code, body = _get("/metadata")
    print(f"HTTP {code}\n{body}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
