#!/usr/bin/env bash
# Self-healing SSH tunnel to the Nexus gateway: 127.0.0.1:13000 -> PC-B:3000.
#
# WHY THIS IS ITS OWN SCRIPT (2026-07-27): the supervisor used to be an inline
# subshell inside start_server.sh, so every app restart / stray `kill` took it
# down with the app — and it wrote to /dev/null, so nothing ever showed that it
# had stopped reconnecting. The app then silently ran on the PUBLIC Cloudflare
# route, which costs ~+2s on BOTH the LLM first sentence and the TTS first chunk
# (measured: TTFA 0.78s with the tunnel vs 2.7-5.0s without). Now it:
#   * runs detached from the app (survives server restarts),
#   * LOGS everything to logs/tunnel.log,
#   * verifies the tunnel by an actual HTTP request, not just "ssh is running",
#   * alternates the VPN IP and the public host (either may be the reachable one).
# Retry spacing stays >=20s: hammering :22 is what tripped PC-B's border SSH
# rate-limit once and locked us out of both routes for an hour.
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
LOG="logs/tunnel.log"
PIDFILE="logs/tunnel.pid"
HOSTS=(10.66.137.91 aiccore.unomaha.edu)
KEY="$HOME/.ssh/pcb_gesture"

# Single instance (Git Bash has no pgrep, so use a pidfile + kill -0 probe).
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
  echo "supervisor already running (pid $(cat "$PIDFILE"))"
  exit 0
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

healthy() { curl -s -m 3 -o /dev/null "http://127.0.0.1:13000/"; }

log "[tunnel-supervisor] started (pid $$)"
while true; do
  if healthy; then
    sleep 15
    continue
  fi
  log "[tunnel] DOWN — reconnecting"
  for host in "${HOSTS[@]}"; do
    if healthy; then break; fi
    log "[tunnel] dialing $host"
    # Foreground: ssh exits when the link dies (ServerAlive*), and the loop
    # immediately tries again. Errors go to the log, never /dev/null.
    ssh -F none -i "$KEY" -o StrictHostKeyChecking=accept-new \
      -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=2 \
      -o ConnectTimeout=8 -N -L 13000:127.0.0.1:3000 "ai-ccore@$host" >> "$LOG" 2>&1
    log "[tunnel] ssh to $host exited (rc=$?)"
    sleep 20
  done
done
