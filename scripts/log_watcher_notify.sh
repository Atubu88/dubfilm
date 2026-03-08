#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$PROJECT_DIR/logs/dubfilm.log"
STATE_FILE="$PROJECT_DIR/tmp/log_watcher.last"
RUNTIME_LOG="$PROJECT_DIR/logs/log_watcher_notify.log"
ENV_FILE="$PROJECT_DIR/.env"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/tmp"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$ENV_FILE"; set +a
fi

BOT_TOKEN="${BOT_TOKEN:-}"
CHAT_ID="${LOG_WATCH_CHAT_ID:-732402669}"

if [[ -z "$BOT_TOKEN" ]]; then
  echo "[$(date -Iseconds)] ERROR: BOT_TOKEN is empty" | tee -a "$RUNTIME_LOG"
  exit 1
fi

touch "$STATE_FILE"
LAST_LINE="$(cat "$STATE_FILE" 2>/dev/null || true)"

send_msg() {
  local text="$1"
  curl -sS -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
    -d "chat_id=${CHAT_ID}" \
    --data-urlencode "text=${text}" >/dev/null || true
}

# On start marker
send_msg "🪝 Лог-вотчер запущен. Буду писать после каждого VIDEO_DONE."
echo "[$(date -Iseconds)] watcher started, log=$LOG_FILE chat=$CHAT_ID" >> "$RUNTIME_LOG"

# Read appended lines only
if [[ -f "$LOG_FILE" ]]; then
  tail -n 0 -F "$LOG_FILE"
else
  # wait for file to appear
  touch "$LOG_FILE"
  tail -n 0 -F "$LOG_FILE"
fi | while IFS= read -r line; do
  [[ "$line" == *"VIDEO_DONE"* ]] || continue

  # simple dedup in case of restarts/rotations
  if [[ "$line" == "$LAST_LINE" ]]; then
    continue
  fi

  LAST_LINE="$line"
  printf '%s' "$LAST_LINE" > "$STATE_FILE"

  mode="unknown"
  send_kind="unknown"
  path=""
  [[ "$line" =~ mode=([^[:space:]]+) ]] && mode="${BASH_REMATCH[1]}"
  [[ "$line" =~ send=([^[:space:]]+) ]] && send_kind="${BASH_REMATCH[1]}"
  [[ "$line" =~ path=(.*)$ ]] && path="${BASH_REMATCH[1]}"

  msg="✅ VIDEO_DONE\nmode=${mode}\nsend=${send_kind}"
  if [[ -n "$path" ]]; then
    msg+="\nfile=${path}"
  fi

  send_msg "$msg"
  echo "[$(date -Iseconds)] notified: $line" >> "$RUNTIME_LOG"
done
