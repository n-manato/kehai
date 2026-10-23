#!/usr/bin/env bash
# Start the KEHAI server ON PC-B (aiccore, Ubuntu) with the env vars the
# VRM/co-speech motion needs.
#
# This is the PC-B variant of PC-A's start_server.sh. The difference that
# matters: on PC-A every model call left the machine (SSH tunnel :13000 ->
# PC-B:3000, public gateway as fallback), so that script had to launch and
# supervise a tunnel. Here the Nexus gateway is a LOOPBACK service on this same
# box (next-server listening on *:3000), so:
#   - there is NO tunnel to start, and no tunnel that can silently die
#   - MOTION_GATEWAY_BASE / GSV_STREAM_URL / conf.yaml all point at 127.0.0.1:3000
#   - the public gateway stays configured only as an automatic fallback
#
# Everything else is deliberately identical to PC-A: the env vars below are what
# make her speak and move. Started without them (i.e. bare `python run_server.py`)
# the LLM still answers, so the chat LOOKS alive, but TTS/STT/gestures are all
# dead — the symptom reads as 「話しかけても喋らない」rather than as a startup
# mistake. Always start via this script.
#
# Run from the repo root:  bash start_server.sh
# Detached (normal use):   setsid --fork nohup bash start_server.sh </dev/null >>logs/server.log 2>&1
set -e
cd "$(dirname "$0")"

# uvx lives in ~/.local/bin and is how conf.yaml's MCP servers ("time",
# "ddg-search") are launched. A login shell has it; setsid/nohup does not.
export PATH="$HOME/.local/bin:$PATH"

export GESTURE_ENABLED=1                              # enable co-speech gestures (master switch)
export MOTION_TEXT_STREAM=1                           # TEXT-driven STREAMING gestures (ActionPlan on GPU1 via /v1/motion/stream):
                                                      # fires from the LLM sentence text (no TTS wait), clip grows while it plays.
                                                      # Set 0 to fall back to the audio-driven GestureLSM path (/v1/gesture).
# Co-speech pacing: ESTIMATED speech seconds between text-driven gesture fires.
# Each fire also carries the GestureLSM upgrade, so this sets how many GestureLSM
# clips a turn gets. 4.0 left the body static ~53% of speaking time; 1.5 cut that
# to ~31% for +93ms TTFA. See PC-A's copy for the full measurement notes.
export MOTION_TEXT_SPACING=1.5
export GSV_STREAM=1                                   # low-latency streaming TTS via gateway /v1/audio/stream (GSV :9880 raw PCM)
# vLLM runs with --max-model-len 24576. The persona system prompt alone is ~12.2k
# real tokens, so the budget below leaves ~3k for generation at that cap.
export LLM_CTX_BUDGET=21000
export LLM_CTX_MAX=24576
export MOTION_GATEWAY_KEY="$(cat .motion_gateway_key 2>/dev/null || echo)"  # gateway bearer key (also used by gsv_stream.py)
export ELEVENLABS_API_KEY="$(cat .elevenlabs_key 2>/dev/null || echo)"      # asr/elevenlabs_asr.py (shadow challenger)
export BACKCHANNEL="${BACKCHANNEL:-1}"   # 相槌 on turn end (utils/backchannel.py); 0 = off
export LAN_HTTPS_PORT="${LAN_HTTPS_PORT:-12443}"  # https for LAN devices (PROTO cable) so the mic works; "" = off
export LAN_HTTPS_IPS="${LAN_HTTPS_IPS:-192.168.50.1}"  # PROTO-only LAN (enp179s0f1); goes into the cert

# LOOPBACK gateway. No tunnel, no Cloudflare, no LAN hop.
export MOTION_GATEWAY_BASE="${MOTION_GATEWAY_BASE:-http://127.0.0.1:3000/v1}"
export GSV_STREAM_URL="${GSV_STREAM_URL:-http://127.0.0.1:3000/v1/audio/stream}"
# Kept as an automatic fallback for the gesture/motion paths only — if the local
# gateway is down the public one usually is too (same box), but it costs nothing.
export MOTION_GATEWAY_FALLBACK="https://gpu.aiccore-uno.ai/v1"

# Warm the On Demand voice + gesture stacks in the background: after a reboot or
# an eviction the first request pays the ~13s cold boot — pay it now, not
# mid-conversation (a cold gesture lands after the frontend stops polling).
# Every probe uses `curl -f`: without it a gateway 502/503 still exits 0, and on
# 2026-10-21 all four lines printed "warm" while every backend was down.
GW="$MOTION_GATEWAY_BASE"
if [ -n "$MOTION_GATEWAY_KEY" ]; then
  mkdir -p cache
  # LLM (vLLM, :8807): NOT On Demand and the gateway never starts it (scripts/pcb-stack-up.sh does).
  # Probed directly — /v1/models answers without running a generation.
  ( curl -sf -m 5 http://127.0.0.1:8807/v1/models -o /dev/null \
      && echo "[warmup] LLM (vLLM) is up" \
      || echo "[warmup] LLM DOWN — run ./kehai.sh up" ) &
  ( curl -sf -m 120 -X POST "$GW/audio/speech" \
      -H "Authorization: Bearer $MOTION_GATEWAY_KEY" -H "Content-Type: application/json" \
      -d '{"model":"gpt-sovits","input":"起動しました"}' -o /dev/null \
      && echo "[warmup] TTS stack is warm" || echo "[warmup] TTS warmup failed (will cold-start on first use)" ) &
  (
    python3 - <<'PY' 2>/dev/null
import wave
w = wave.open("cache/warm_silent6s.wav", "wb")
w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
w.writeframes(b"\x00\x00" * 96000)
w.close()
PY
    [ -f cache/warm_silent6s.wav ] || exit 0
    curl -sf -m 120 -X POST "$GW/gesture?format=joints" \
      -H "Authorization: Bearer $MOTION_GATEWAY_KEY" \
      -F "audio=@cache/warm_silent6s.wav;type=audio/wav" -F "text=起動" -o /dev/null \
      && echo "[warmup] gesture model is warm" || echo "[warmup] gesture warmup failed (will cold-start on first use)"
    # Text-streaming motion (ActionPlan, resident on GPU1): NOT On Demand — if the
    # box rebooted, scripts/pcb-stack-up.sh relaunches it. Probe it so a dead service is
    # visible at startup instead of as silently-missing gestures.
    curl -sf -m 30 -X POST "$GW/motion/stream" \
      -H "Authorization: Bearer $MOTION_GATEWAY_KEY" -H "Content-Type: application/json" \
      -d '{"text":"a person stands","seconds":1,"session":"warmup","chain":false}' -o /dev/null \
      && echo "[warmup] text-motion (ActionPlan) is up" \
      || echo "[warmup] text-motion DOWN — run ./kehai.sh up"
    # STT (faster-whisper on GPU1 via the gateway sidecar): the first request
    # after a sidecar restart pays the model load — pay it here instead.
    # MUST match conf.yaml's gateway_whisper_asr.model. Warming a different id
    # warms a different model: the real first turn would then cold-load (4.2s
    # measured) against the client's 5s loopback timeout, and a miss there falls
    # over to the PUBLIC gateway — a silent trip to the internet for a service
    # running on this same box.
    curl -sf -m 120 -X POST "$GW/audio/transcriptions" \
      -H "Authorization: Bearer $MOTION_GATEWAY_KEY" \
      -F "file=@cache/warm_silent6s.wav;type=audio/wav" -F "model=faster-whisper-large-v3-full" -o cache/warm_stt.json \
      && ! grep -q "Mock transcription" cache/warm_stt.json \
      && echo "[warmup] STT model is warm" \
      || echo "[warmup] STT DOWN (gateway answers with a mock transcript when its sidecar is down) — run ./kehai.sh up"
  ) &
fi

echo "GESTURE_ENABLED=$GESTURE_ENABLED  MOTION_GATEWAY_KEY=${MOTION_GATEWAY_KEY:+<set>}  BASE=$MOTION_GATEWAY_BASE"

# The venv is python3.11 (system python3 here is 3.8; the app needs >=3.10).
PYEXE=".venv/bin/python"
[ -x "$PYEXE" ] || { echo "ERROR: $PYEXE missing — create it with: python3.11 -m venv .venv"; exit 1; }
exec "$PYEXE" run_server.py "$@"
