#!/usr/bin/env bash
set -euo pipefail

cd /home/fanfan/projects/dubfilm
mkdir -p out logs

# Load .env defaults if present
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Usage: ./run_local_once.sh [input_video]
INPUT="${1:-/home/fanfan/projects/dubfilm/in/e62512f8-a22e-4c1f-adaa-7bfe305e4e3f.mp4}"
# Keep explicit/.env value; final fallback whisper.
export TRANSCRIBE_PROVIDER="${TRANSCRIBE_PROVIDER:-whisper}"
export DUB_LOCAL_INPUT="$INPUT"

for i in 1 2 3; do
  echo "[run_local_once] attempt $i/3 input=$INPUT provider=$TRANSCRIBE_PROVIDER"
  if timeout 2400 ./.venv/bin/python -u run_local_dub.py 2>&1 | tee -a logs/local_run.log; then
    echo "[run_local_once] done"
    ls -lh out | tail -n 5
    exit 0
  fi
  echo "[run_local_once] failed attempt $i"
  sleep $((i*3))
done

echo "[run_local_once] failed after retries. Check logs/local_run.log"
exit 1
