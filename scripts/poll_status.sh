#!/usr/bin/env bash
# Quick-glance poll of Run 3 v5 + Run 4 status. Used during the hackathon submission
# window when the watcher script is overkill. Reads HF tokens from env vars.
#
# Usage:
#   HF_TOKEN_KANAN=hf_... HF_TOKEN_MNIT=hf_... bash scripts/poll_status.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${HF_TOKEN_KANAN:?need HF_TOKEN_KANAN}"
: "${HF_TOKEN_MNIT:?need HF_TOKEN_MNIT}"

.venv/bin/python -c "
import os, dateutil.parser as dp, datetime, truststore
truststore.inject_into_ssl()
from huggingface_hub import HfApi

now = datetime.datetime.now(datetime.timezone.utc)
def show(name, tok_env, jid):
    api = HfApi(token=os.environ[tok_env])
    j = api.inspect_job(job_id=jid)
    created = dp.parse(str(j.created_at))
    el = (now - created).total_seconds() / 60
    msg = j.status.message or '-'
    print(f'{name}: stage={j.status.stage} elapsed={el:5.1f}min  msg={msg}')

show('Run3 v5 (4B+GRPO @ Kanan)  ', 'HF_TOKEN_KANAN', '69ed2569d2c8bd8662bce61a')
show('Run4    (1.7B+KL @ MNIT)   ', 'HF_TOKEN_MNIT',  '69ed1a3fd70108f37acdee5e')
"
