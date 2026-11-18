#!/usr/bin/env bash
# KEHAI on PC-B: start / stop / status / log.
#
#   ./kehai.sh up        gateway + ActionPlan + vLLM, then start (what @reboot runs)
#   ./kehai.sh start     detached; safe to run twice (refuses if already up)
#   ./kehai.sh stop
#   ./kehai.sh restart
#   ./kehai.sh status
#   ./kehai.sh log       follow the server log
#   ./kehai.sh mode {cascade|omni|status}
#                        switch the brain (scripts/pcb-mode.sh): cascade = Whisper ->
#                        Qwen3.8 -> GSV (original); omni = Qwen3-Omni hears the audio.
#                        Remembered in .mode across reboots.
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
  mode) shift; exec bash scripts/pcb-mode.sh "$@" ;;
  status)
    p="$(pid)"
    if [ -n "$p" ]; then echo "running (pid $p)"; else echo "NOT running"; fi
    echo "mode: $([ -f .mode ] && tr -d '[:space:]' < .mode || echo cascade)"
    ss -ltn 2>/dev/null | grep ":$PORT " || echo "no listener on :$PORT"
    # The gateway is what actually speaks/hears; a live server in front of a
    # dead gateway is the "she won't talk" shape, so report both.
    code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $(cat .motion_gateway_key 2>/dev/null)" http://127.0.0.1:3000/v1/models)
    echo "nexus gateway :3000 -> HTTP $code (200 = healthy)"
    # The parts she cannot work without. Qwen3-TTS was stopped from outside on
    # 2026-11-07 23:30 and she sat there silent for two hours, every reply's
    # audio failing with nothing on the screen to say so.
    if [ "$(tr -d '[:space:]' < .mode 2>/dev/null)" = "omni" ]; then
      for chk in "8808:Thinker (hears+thinks)" "8809:Qwen3-TTS (her voice)"; do
        port=${chk%%:*}; name=${chk#*:}
        c=$(curl -s -o /dev/null -m 3 -w '%{http_code}' http://127.0.0.1:$port/v1/models)
        [ "$c" = 200 ] && echo "$name :$port -> up" || echo "$name :$port -> DOWN (HTTP $c) — ./kehai.sh up"
      done
    fi
    ;;
  log) tail -f logs/server.log ;;
  *) echo "usage: $0 {up|start|stop|restart|status|log|mode {cascade|omni|status}}"; exit 2 ;;
esac
