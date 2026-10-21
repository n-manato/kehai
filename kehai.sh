#!/usr/bin/env bash
# KEHAI on PC-B: start / stop / status / log.
#
#   ./kehai.sh up        gateway + ActionPlan + vLLM, then start (what @reboot runs)
#   ./kehai.sh start     detached; safe to run twice (refuses if already up)
#   ./kehai.sh stop
#   ./kehai.sh restart
#   ./kehai.sh status
#   ./kehai.sh log       follow the server log
#
# The browser side lives on PC-A: pwsh scripts/pcb-app-tunnel.ps1, then
# http://127.0.0.1:12395/ . Nothing here needs a display or a sound card —
# this box has neither.
set -u
cd "$(dirname "$0")"

PORT=12394
# NOTE the bracket: [r] keeps this pattern from matching the shell running it.
# Without it `pkill -f run_server.py` kills its own ssh/bash wrapper (whose
# command line contains the string) and the real server survives.
PAT='[r]un_server.py'

pid() { pgrep -f "$PAT" | head -1; }

case "${1:-status}" in
  start)
    if [ -n "$(pid)" ]; then echo "already running (pid $(pid))"; exit 0; fi
    mkdir -p logs cache
    setsid --fork nohup bash start_server.sh </dev/null >>logs/server.log 2>&1
    echo "starting… (log: logs/server.log)"
    for _ in $(seq 40); do
      ss -ltn 2>/dev/null | grep -q ":$PORT " && { echo "up on :$PORT"; exit 0; }
      sleep 1
    done
    echo "did NOT come up within 40s — check logs/server.log"; exit 1
    ;;
  stop)
    p="$(pid)"
    [ -z "$p" ] && { echo "not running"; exit 0; }
    kill "$p"; sleep 3
    [ -n "$(pid)" ] && kill -9 "$(pid)"
    echo "stopped"
    ;;
  up) mkdir -p logs; bash scripts/pcb-stack-up.sh 2>&1 | tee -a logs/boot.log ;;
  restart) "$0" stop && "$0" start ;;
  status)
    p="$(pid)"
    if [ -n "$p" ]; then echo "running (pid $p)"; else echo "NOT running"; fi
    ss -ltn 2>/dev/null | grep ":$PORT " || echo "no listener on :$PORT"
    # The gateway is what actually speaks/hears; a live server in front of a
    # dead gateway is the "she won't talk" shape, so report both.
    code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $(cat .motion_gateway_key 2>/dev/null)" http://127.0.0.1:3000/v1/models)
    echo "nexus gateway :3000 -> HTTP $code (200 = healthy)"
    ;;
  log) tail -f logs/server.log ;;
  *) echo "usage: $0 {up|start|stop|restart|status|log}"; exit 2 ;;
esac
