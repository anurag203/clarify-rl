#!/usr/bin/env bash
# ClarifyRL preflight — sanity-check the full pipeline BEFORE spending real
# budget on parallel HF Jobs runs. Catches:
#   - hf CLI not installed
#   - HF_TOKEN missing / invalid
#   - env Space not reachable / wrong concurrency config
#   - TRL/transformers/jmespath/bnb local import errors (your laptop)
#   - max_concurrent_envs not yet bumped on the live Space
#
# Usage:
#   HF_TOKEN=hf_xxx ./scripts/preflight.sh
#
# Exits with code 0 on success, non-zero with a clear diagnostic on failure.

set -euo pipefail

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { printf "${GREEN}[OK]${NC}    %s\n" "$1"; }
fail() { printf "${RED}[FAIL]${NC}  %s\n" "$1"; exit 1; }
warn() { printf "${YELLOW}[WARN]${NC}  %s\n" "$1"; }

: "${ENV_BASE_URL:=https://agarwalanu3103-clarify-rl.hf.space}"

echo "====================================================================="
echo "ClarifyRL preflight"
echo "  Env Space: $ENV_BASE_URL"
echo "====================================================================="

# 1. hf CLI present
if ! command -v hf >/dev/null 2>&1; then
    fail "hf CLI not installed. Run: pip install -U 'huggingface_hub[cli]'"
fi
ok "hf CLI present: $(hf --version 2>&1 | head -1)"

# 2. HF_TOKEN set + valid
if [ -z "${HF_TOKEN:-}" ]; then
    fail "HF_TOKEN env var is not set"
fi
if ! HF_HUB_DISABLE_PROGRESS_BARS=1 hf auth whoami >/dev/null 2>&1; then
    fail "HF_TOKEN appears invalid (hf auth whoami failed)"
fi
WHOAMI="$(HF_HUB_DISABLE_PROGRESS_BARS=1 hf auth whoami 2>&1 | head -1)"
ok "HF_TOKEN valid: $WHOAMI"

# 3. Space reachable. The OpenEnv server exposes /health (not /healthz).
HEALTH_HTTP="$(curl -sS -o /dev/null -w "%{http_code}" --max-time 10 "$ENV_BASE_URL/health" 2>/dev/null || echo "000")"
if [ "$HEALTH_HTTP" != "200" ]; then
    fail "Env Space at $ENV_BASE_URL/health returned HTTP $HEALTH_HTTP (expected 200)"
fi
ok "Env Space /health returned 200"

# 4. Probe the WS endpoint with a quick reset to confirm capacity > 1
WS_URL="$(echo "$ENV_BASE_URL" | sed 's|^https|wss|; s|^http|ws|')/ws"
# Use the project's venv python so truststore is available on macOS w/ corp proxies.
PYBIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/bin/python"
[ -x "$PYBIN" ] || PYBIN="python3"
"$PYBIN" - <<PY
import asyncio, json, sys
try:
    import truststore  # noqa: F401
    truststore.inject_into_ssl()
except Exception:
    pass
try:
    import websockets
except ImportError:
    print("[WARN]  python \`websockets\` not installed — skipping WS probe.")
    print("        $PYBIN -m pip install websockets to enable concurrency probe.")
    sys.exit(0)

async def probe():
    async def one(_name):
        async with websockets.connect("$WS_URL", open_timeout=20) as ws:
            await ws.send(json.dumps({"type": "reset", "data": {"task_id": "easy"}}))
            return json.loads(await asyncio.wait_for(ws.recv(), timeout=20))

    r1 = await one("seq")
    if r1.get("type") == "error":
        print(f"[FAIL]  WS reset returned error: {r1}")
        sys.exit(1)
    print(f"[OK]    Sequential reset ok (type={r1.get('type')})")

    results = await asyncio.gather(*(one(f"par{i}") for i in range(3)), return_exceptions=True)
    errs = [r for r in results
            if isinstance(r, Exception) or (isinstance(r, dict) and r.get("type") == "error")]
    if errs:
        print(f"[WARN]  Concurrent probe: {len(errs)}/3 failed — push server/ changes & wait for rebuild")
        for e in errs:
            print(f"        - {e}")
        sys.exit(3)
    print("[OK]    3 concurrent WS sessions succeeded — max_concurrent_envs is live")

asyncio.run(probe())
PY
ws_rc=$?
if [ "$ws_rc" -eq 3 ]; then
    warn "Space concurrency probe failed — push server/ changes before parallel runs!"
elif [ "$ws_rc" -ne 0 ]; then
    fail "WS probe script exited $ws_rc"
fi

# 5. Local Python syntax + AST check of the training script (no heavy deps required).
if "$PYBIN" -c "
import ast, pathlib
src = pathlib.Path('training/train_grpo.py').read_text()
ast.parse(src)
print('train_grpo.py is valid Python')
" 2>&1; then
    ok "training/train_grpo.py parses cleanly"
else
    fail "training/train_grpo.py has syntax errors"
fi

# 6. run_eval.py / make_plots.py --help (light imports OK)
if "$PYBIN" scripts/run_eval.py --help >/dev/null 2>&1; then
    ok "scripts/run_eval.py --help works"
else
    fail "scripts/run_eval.py raised on --help"
fi
if "$PYBIN" scripts/make_plots.py --help >/dev/null 2>&1; then
    ok "scripts/make_plots.py --help works"
else
    fail "scripts/make_plots.py raised on --help"
fi

echo
echo "====================================================================="
echo "Preflight complete. You're cleared to launch."
echo "====================================================================="
