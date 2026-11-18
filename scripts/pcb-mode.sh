#!/usr/bin/env bash
# PC-B: switch the brain behind KEHAI, and remember the choice across reboots.
#
#   ./kehai.sh mode cascade   Whisper -> Qwen3.8-27B (vLLM :8807, TP=2) -> GPT-SoVITS
#                             (the original pipeline; the rollback target)
#   ./kehai.sh mode omni      Whisper (turn control only) -> Qwen3-Omni-30B INT4
#                             (vLLM-Omni :8808, Thinker on GPU1, Talker on GPU0)
#                             -> GPT-SoVITS (Stage 1) or the Talker (Stage 2)
#   ./kehai.sh mode status
#
# Everything the OTHER mode needs on the GPUs is stopped first, so the cards are
# free for this one. The choice lands in .mode, which start_server.sh (env for
# the app) and scripts/pcb-stack-up.sh (@reboot) read. Services owned by the
# Nexus gateway (GPT-SoVITS, GestureLSM) are switched through its DB: the
# gateway worker starts whatever is enabled and kills whatever is not
# (AICCORE-Nexus/src/lib/worker.ts), so stopping them by hand would not stick.
set -u
KEHAI="$(cd "$(dirname "$0")/.." && pwd)"
NEXUS="$HOME/Documents/DEV/AICCORE-Nexus"
DB="$NEXUS/prisma/dev.db"
MODE_FILE="$KEHAI/.mode"
OMNI_DIR="$NEXUS/stack/vllm-omni"
OMNI_LOG="$OMNI_DIR/server.log"

log() { echo "$(date -Is) [mode] $*"; }
ok() { curl -sf -o /dev/null -m 4 "$1"; }
wait_for() {  # url seconds label
  for _ in $(seq "$2"); do ok "$1" && { log "$3 is up"; return 0; }; sleep 1; done
  log "$3 did NOT come up within ${2}s"; return 1
}
wait_gone() {  # pgrep-pattern seconds label
  for _ in $(seq "$2"); do pgrep -f "$1" >/dev/null || { log "$3 stopped"; return 0; }; sleep 1; done
  log "$3 still running after ${2}s — killing hard"; pkill -9 -f "$1"; sleep 2
}
current() { local m=""; [ -f "$MODE_FILE" ] && m="$(tr -d '[:space:]' < "$MODE_FILE")"; echo "${m:-cascade}"; }

# ModelRuntimeConfig.enabled for a gateway-managed model. One backup per day.
set_enabled() {  # modelId 0|1
  local bak="$DB.bak-mode-$(date +%Y%m%d)"
  [ -f "$bak" ] || cp "$DB" "$bak"
  sqlite3 "$DB" "UPDATE ModelRuntimeConfig SET enabled=$2 WHERE modelId='$1';"
  log "gateway DB: $1 enabled=$2"
}

# GPT-SoVITS' card. keepalive.sh (the gateway's launcher for it) reads
# stack/voice/.gsv_gpu; killing the server makes the gateway worker relaunch it
# there (enabled=1). omni needs GPU1 for the Thinker alone: with GSV's 2.1 GiB
# beside it the Thinker started, answered two warm-up calls and died of OOM on
# its first real forward (2026-11-02, GPU1 at 23.8/24 GB).
move_gsv() {  # gpu-index
  local want="$1" have
  have="$(cat "$NEXUS/stack/voice/.gsv_gpu" 2>/dev/null | tr -d '[:space:]')"; have="${have:-1}"
  echo "$want" > "$NEXUS/stack/voice/.gsv_gpu"
  if [ "$have" = "$want" ] && ok http://127.0.0.1:9880/docs; then return 0; fi
  log "moving GPT-SoVITS to GPU$want (was GPU$have)"
  set_enabled gpt-sovits 1
  pkill -f '[a]pi_v2.py .*-p 9880' 2>/dev/null
  for _ in $(seq 20); do pgrep -f '[a]pi_v2.py .*-p 9880' >/dev/null || break; sleep 1; done
  # Run the gateway's own launcher ourselves: its worker judges the voice stack
  # by voice_service (:9881), which survives the GSV kill, so it never noticed
  # GSV was gone (2026-11-02: 180 s and no restart). keepalive.sh is idempotent
  # and reads the new .gsv_gpu. (Its pgrep would match a caller whose command
  # line contains "api_v2.py ... -p 9880" — this script's does not.)
  (cd "$NEXUS/stack/voice" && setsid --fork nohup bash keepalive.sh </dev/null >>/tmp/voice_keepalive.log 2>&1)
  wait_for http://127.0.0.1:9880/docs 180 "GPT-SoVITS on GPU$want" || log "GSV did not return — see /tmp/gsv_api.log"
}

# omni's reply voice: OMNI_TTS, else Qwen3-TTS when a fine-tuned speaker is
# installed (the same rule as start_server.sh).
omni_tts() {
  local t="${OMNI_TTS:-}"
  if [ -z "$t" ]; then
    if [ -s "$NEXUS/stack/qwen3-tts/models/current.speaker" ]; then t=qwen3tts; else t=gsv; fi
  fi
  echo "$t"
}

# cascade's reply voice (2026-11-04): Qwen3-TTS when a fine-tuned speaker is
# installed (faster first audio, the voice the user chose), else GPT-SoVITS.
# CASCADE_TTS=gsv|qwen3tts overrides. Persisted in .cascade-tts so the reboot
# bring-up (pcb-stack-up.sh) and start_server.sh lay the cards out the same way.
cascade_tts() {
  local t="${CASCADE_TTS:-}"
  if [ -z "$t" ]; then
    if [ -s "$NEXUS/stack/qwen3-tts/models/current.speaker" ]; then t=qwen3tts; else t=gsv; fi
  fi
  echo "$t"
}

# GPT-SoVITS off the GPU while nothing speaks through it (omni with Qwen3-TTS,
# 2026-11-04: the user does not want idle models on the GPUs; GSV had served
# nothing but KEHAI's start-up warm-up for a day). enabled=0 keeps the gateway
# from relaunching it; the PID-pattern kill frees its 2.2 GiB now.
stop_gsv() {
  set_enabled gpt-sovits 0
  pkill -f '[a]pi_v2.py .*-p 9880' 2>/dev/null
  for _ in $(seq 20); do pgrep -f '[a]pi_v2.py .*-p 9880' >/dev/null || break; sleep 1; done
  pgrep -f '[a]pi_v2.py .*-p 9880' >/dev/null && log "GPT-SoVITS still running" || log "GPT-SoVITS stopped (gateway: enabled=0)"
}

# A real generation, not just /v1/models: the 5th start (2026-11-02) answered
# the model list, served two warm-ups, then the engine died of OOM on the
# first forward and every turn after that spoke an error message.
omni_smoke() {
  curl -sf -m 120 http://127.0.0.1:8808/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3-omni","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"temperature":0}' \
    | grep -q '"content"'
}

# The two vLLM servers share the words "vllm serve"; tell them apart by port.
# `[v]` keeps pgrep from matching this script's own command line.
PAT_CASCADE='[v]llm serve.*--port 8807'
PAT_OMNI='[v]llm serve.*--port 8808'
PAT_TTS='[v]llm serve.*--port 8809'   # Qwen3-TTS (omni mode TTS, GPU0)
TTS_DIR="$NEXUS/stack/qwen3-tts"
PAT_AP='[a]ctionplan_service.py'

stop_cascade_gpu() {
  if pgrep -f "$PAT_CASCADE" >/dev/null; then
    log "stopping vLLM (Qwen3.8-27B, :8807)"; pkill -f "$PAT_CASCADE"; wait_gone "$PAT_CASCADE" 30 "vLLM :8807"
  fi
  if pgrep -f "$PAT_AP" >/dev/null; then
    log "stopping ActionPlan (:8893)"; pkill -f "$PAT_AP"; wait_gone "$PAT_AP" 20 ActionPlan
  fi
}
stop_omni_gpu() {
  # The wrapper too (it may not have exec'd vllm yet). Anchored to the exact
  # command line the launcher produces: a looser '[s]erve-qwen3-omni' matched a
  # caller's shell whose command line merely MENTIONED the file and killed it
  # mid-switch (2026-11-02) — the self-kill the HANDOFF warns about, again.
  pkill -f '^bash \./serve-qwen3-omni\.sh' 2>/dev/null
  if pgrep -f "$PAT_OMNI" >/dev/null; then
    log "stopping vLLM-Omni (:8808)"; pkill -f "$PAT_OMNI"; wait_gone "$PAT_OMNI" 30 "vLLM-Omni :8808"
  fi
}
stop_tts_gpu() {
  pkill -f '^bash \./serve-qwen3-tts\.sh' 2>/dev/null
  if pgrep -f "$PAT_TTS" >/dev/null; then
    log "stopping Qwen3-TTS (:8809)"; pkill -f "$PAT_TTS"; wait_gone "$PAT_TTS" 30 "Qwen3-TTS :8809"
  fi
}
start_tts_gpu() {  # [gpu-index, default 0]
  local gpu="${1:-0}"
  [ -e "$TTS_DIR/serve-qwen3-tts.sh" ] || ln -s "$KEHAI/scripts/pcb-vllm/serve-qwen3-tts.sh" "$TTS_DIR/serve-qwen3-tts.sh"
  if ok http://127.0.0.1:8809/v1/models; then log "Qwen3-TTS already up"; return 0; fi
  log "starting Qwen3-TTS (:8809, GPU$gpu) -> $TTS_DIR/server.log"
  (cd "$TTS_DIR" && QWEN3TTS_GPU="$gpu" setsid --fork bash -c 'exec ./serve-qwen3-tts.sh </dev/null >>server.log 2>&1')
  wait_for http://127.0.0.1:8809/v1/models 600 "Qwen3-TTS" || log "Qwen3-TTS did not come up — OMNI_TTS=gsv keeps working; see $TTS_DIR/server.log"
}

# Engine workers (VLLM::EngineCore / VLLM::Worker_*) normally die with their
# server; with ALL servers down any survivor is a leak holding VRAM.
reap_workers() {
  if ! pgrep -f "$PAT_CASCADE" >/dev/null && ! pgrep -f "$PAT_OMNI" >/dev/null && ! pgrep -f "$PAT_TTS" >/dev/null; then
    if pgrep -f '^VLLM::' >/dev/null; then log "reaping stray VLLM:: workers"; pkill -9 -f '^VLLM::'; sleep 2; fi
  fi
}

status() {
  echo "mode: $(current)   (.mode)"
  echo "kehai: $("$KEHAI/kehai.sh" status 2>/dev/null | head -1)"
  for p in 8807:vLLM-Qwen3.8 8808:vLLM-Omni 8809:Qwen3-TTS 8893:ActionPlan 8802:GestureLSM 9880:GPT-SoVITS 8000:Whisper 3000:gateway; do
    port="${p%%:*}"; name="${p#*:}"
    if ss -ltn 2>/dev/null | grep -q ":$port "; then echo "  :$port $name  up"; else echo "  :$port $name  -"; fi
  done
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  GPU /'
  sqlite3 -readonly "$DB" "select modelId||' enabled='||enabled from ModelRuntimeConfig where modelId in ('gpt-sovits','gesturelsm-1');" 2>/dev/null | sed 's/^/  gateway: /'
}

case "${1:-status}" in
  status) status ;;

  omni)
    STAGE="${OMNI_STAGE:-1}"
    [ -x "$OMNI_DIR/.venv/bin/vllm" ] || { log "no venv at $OMNI_DIR/.venv — see docs/PCB-HANDOFF.md (omni)"; exit 1; }
    [ -e "$OMNI_DIR/serve-qwen3-omni.sh" ] || ln -s "$KEHAI/scripts/pcb-vllm/serve-qwen3-omni.sh" "$OMNI_DIR/serve-qwen3-omni.sh"
    "$KEHAI/kehai.sh" stop
    stop_cascade_gpu
    # Coming from cascade with the Qwen3-TTS voice, the TTS sits on GPU1 — the
    # Thinker's card. Stop it; it is started again on GPU0 further down.
    if pgrep -af "$PAT_TTS" | grep -q '"devices":"1"'; then
      log "Qwen3-TTS is on GPU1 (cascade layout) — moving it to GPU0"; stop_tts_gpu
    fi
    # A previous omni attempt that never answered is a leak, not a server.
    ok http://127.0.0.1:8808/v1/models || stop_omni_gpu
    reap_workers
    # GestureLSM (real-time co-speech motion) runs in omni mode since 2026-11-05
    # (user: Iris moves by generated motion, not KAWAII). It lives on GPU0
    # (gesture-service/run.sh, GESTURE_GPU 0): never beside the Thinker on GPU1.
    # GPT-SoVITS only when it is the reply voice (OMNI_TTS=gsv, Stage 1).
    set_enabled gesturelsm-1 1
    if [ "$STAGE" = "2" ] || [ "$(omni_tts)" != "gsv" ]; then stop_gsv; else move_gsv 0; fi
    if ok http://127.0.0.1:8808/v1/models && omni_smoke; then log "vLLM-Omni already up"; else
      ok http://127.0.0.1:8808/v1/models && { log "vLLM-Omni answers but cannot generate — restarting it"; stop_omni_gpu; reap_workers; }
      log "starting vLLM-Omni (Qwen3-Omni-30B INT4, stage=$STAGE) -> $OMNI_LOG"
      (cd "$OMNI_DIR" && OMNI_STAGE="$STAGE" setsid --fork bash -c 'exec ./serve-qwen3-omni.sh </dev/null >>server.log 2>&1')
      # Poll the server, and bail as soon as its process is gone (a startup
      # crash shows within a minute; waiting the full 15 min for it is downtime).
      # "Gone" only counts once the `vllm serve` command line has been SEEN:
      # the wrapper spends its first seconds in bash/python before exec'ing
      # vllm, and judging it absent then (2026-11-02) fell back to cascade on
      # top of a server that was still starting — the two then fought over
      # GPU1 and the omni worker died of OOM.
      up=0; seen=0; n=0
      for _ in $(seq 900); do
        ok http://127.0.0.1:8808/v1/models && { up=1; break; }
        if pgrep -f "$PAT_OMNI" >/dev/null; then seen=1
        elif [ "$seen" = 1 ] || [ "$n" -ge 60 ]; then break
        fi
        n=$((n + 1)); sleep 1
      done
      if [ "$up" = 1 ] && ! omni_smoke; then
        log "vLLM-Omni is listening but a test generation failed"; up=0
      fi
      if [ "$up" != 1 ]; then
        log "vLLM-Omni did not come up — see $OMNI_LOG"
        grep -E "Error|error:" "$OMNI_LOG" | grep -vE "File |rope_parameters|\^\^\^" | tail -3
        # Never leave the box without a brain: go back to the pipeline that works.
        log "falling back to cascade"
        exec "$0" cascade
      fi
      log "vLLM-Omni is up (test generation OK)"
    fi
    start_tts_gpu
    echo omni > "$MODE_FILE"
    [ "$STAGE" = "2" ] && echo 2 > "$KEHAI/.mode-stage" || rm -f "$KEHAI/.mode-stage"
    "$KEHAI/kehai.sh" start
    status
    ;;

  cascade)
    "$KEHAI/kehai.sh" stop
    stop_omni_gpu
    stop_tts_gpu
    reap_workers
    CTTS="$(cascade_tts)"
    if [ "$CTTS" = "qwen3tts" ]; then
      # Qwen3.8-27B (TP=2, ~13.2 GB per card) + Qwen3-TTS on GPU1 (7.1 GB).
      # That leaves no room for ActionPlan (5.5 GB, GPU1), GestureLSM or GSV:
      # the body runs as in omni (KAWAII clip dictionary). 2026-11-04.
      set_enabled gesturelsm-1 0
      stop_gsv
      pgrep -f "$PAT_AP" >/dev/null && { log "stopping ActionPlan (:8893)"; pkill -f "$PAT_AP"; wait_gone "$PAT_AP" 20 ActionPlan; }
      start_tts_gpu 1
    else
      set_enabled gesturelsm-1 1
      # GSV back on GPU1 BEFORE vLLM :8807 takes its share of both cards.
      move_gsv 1
    fi
    echo "$CTTS" > "$KEHAI/.cascade-tts"
    echo cascade > "$MODE_FILE"
    rm -f "$KEHAI/.mode-stage"
    # The original bring-up: ActionPlan -> STT -> vLLM :8807 -> KEHAI (idempotent).
    # KEHAI comes up last, after GSV (it remembers the character's weights as
    # applied and will not resend them to a restarted GSV).
    "$KEHAI/kehai.sh" up
    status
    ;;

  *) echo "usage: $0 {cascade|omni|status}   (OMNI_STAGE=2 for the Talker)"; exit 2 ;;
esac
