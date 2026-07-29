#!/usr/bin/env bash
# Start the Open-LLM-VTuber server with the env vars the VRM/co-speech motion needs.
# GESTURE_ENABLED defaults to OFF in gesture_client.py — without it there are NO
# body gestures (the frontend polls /motion/clip-smpl and only ever gets 202).
# Run from the repo root:  bash start_server.sh   (add & or nohup for detached)
set -e
cd "$(dirname "$0")"

export GESTURE_ENABLED=1                              # enable co-speech gestures (master switch)
export MOTION_TEXT_STREAM=1                           # TEXT-driven STREAMING gestures (ActionPlan on PC-B GPU1 via /v1/motion/stream):
                                                      # fires from the LLM sentence text (no TTS wait), first frames ~0.75s over the
                                                      # public gateway, clip grows while it plays. Set 0 to fall back to the
                                                      # audio-driven GestureLSM path (/v1/gesture).
# Co-speech pacing: how many ESTIMATED speech seconds between text-driven gesture fires.
# Each fire is also what carries the GestureLSM upgrade, so this really sets how many
# GestureLSM clips a turn gets. Measured 2026-07-26 on the English pipeline:
#   4.0 (old default) -> 8 sentences produced only 2 clips (2 x 4.23s of coverage for ~11s
#        of speech), so the body was STATIC 52.8% of the time it was speaking.
#   1.5                -> GestureLSM playing 4.8s -> 10.0s, static 52.8% -> 30.8%.
# Cost measured over 12 turns: TTFA total 1547ms -> 1640ms (+93ms, ~6%) and
# tts_first_chunk 906ms -> 914ms, i.e. the shared PC-B GPU absorbed the extra gesture
# calls without slowing the voice. The old 4.0 existed to stop KAWAII clips replacing each
# other every 1-2s; the frontend's gestureLSMActive() guard now blocks that instead.
export MOTION_TEXT_SPACING=1.5
export GSV_STREAM=1                                   # true low-latency streaming TTS via gateway /v1/audio/stream (GSV :9880 raw PCM)
# 2026-07-22: vLLM restarted with --max-model-len 24576 (32768 OOM'd during CUDA
# graph capture; 24576 fits). The Rem persona system prompt alone is ~12.2k REAL
# tokens, so with the old 16384 cap only ~2.5k remained for history+tool results
# (fetched content was being dropped). Budget = total input (system+messages)
# estimate; ~21000 leaves ~3k for generation at the 24576 cap.
export LLM_CTX_BUDGET=21000
export LLM_CTX_MAX=24576
export MOTION_GATEWAY_KEY="$(cat .motion_gateway_key 2>/dev/null || echo)"  # gateway bearer key (also used by gsv_stream.py for the stream route)
# 2026-07-20: SSH tunnels retired — everything (gestures AND TTS) goes through the
# public Nexus gateway. The gateway On-Demand-starts the PC-B model services on the
# first request and keeps them resident. Gestures over the public route measure
# ~2.8s vs the old LAN tunnels' ~1.5s; TTS measures the same as the tunnel (~1.3-2.5s).
# 2026-07-21: overridable again — the streaming-TTS setup tunnels gestures directly
# to PC-B's gesture service (-L 18802:127.0.0.1:8802 -> http://127.0.0.1:18802):
# LAN ~1.5s/clip vs public ~2.8s decides whether co-speech clips keep up with
# streamed playback (~3s/sentence). Public gateway stays as automatic fallback.
# 2026-07-22: LAN TUNNEL to the Nexus gateway itself (ssh -L 13000 -> PC-B
# 127.0.0.1:3000). PC-B :22 is already open, so NO firewall change / LAN
# exposure; auth + routing stay on the gateway. Measured vs Cloudflare:
# motion ttfb 0.50->0.15s, TTS first byte 0.40->0.18s, STT 1.8->1.1s.
# The supervisor below keeps it alive; MOTION_GATEWAY_FALLBACK keeps the
# public route as automatic fallback (gesture/motion paths).
# Supervisor notes (2026-07-26 incident): the old loop retried ONLY the VPN IP
# every ~7s; when the VPN dropped it hammered PC-B:22 for hours and tripped the
# border SSH rate-limit (both routes then REFUSED connections — even though the
# gateway itself was fine). Now: alternate VPN IP -> public hostname, and wait
# 20s after every attempt (blocked-streak lesson: space retries or the reset
# persists). While the tunnel is down the app self-heals: LLM/ASR/streaming-TTS/
# motion all fall back to the public gateway automatically.
# 2026-07-27: the supervisor moved OUT of this script into
# scripts/tunnel_supervisor.sh and now runs DETACHED. As an inline subshell it
# died with every app restart, after which the app silently fell back to the
# public Cloudflare route — measured cost ~+2s on BOTH llm_first_sentence and
# tts_first_chunk (TTFA 0.78s tunnelled vs 2.7-5.0s public). It also logs to
# logs/tunnel.log instead of /dev/null so a stuck tunnel is visible.
if [ -f logs/tunnel.pid ] && kill -0 "$(cat logs/tunnel.pid 2>/dev/null)" 2>/dev/null; then
  echo "[tunnel] supervisor already running (pid $(cat logs/tunnel.pid))"
else
  nohup bash scripts/tunnel_supervisor.sh </dev/null >/dev/null 2>&1 &
  disown 2>/dev/null || true
  echo "[tunnel] supervisor started (detached; log: logs/tunnel.log)"
fi
export MOTION_GATEWAY_BASE="${MOTION_GATEWAY_BASE:-http://127.0.0.1:13000/v1}"
export MOTION_GATEWAY_FALLBACK="https://gpu.aiccore-uno.ai/v1"
export GSV_STREAM_URL="${GSV_STREAM_URL:-http://127.0.0.1:13000/v1/audio/stream}"
# OPENAI_API_KEY must already be in the environment (for openai_llm).

# Warm the On Demand voice + gesture stacks in the background: after a PC-B
# reboot/eviction the first gateway request pays the ~13s cold boot — pay it now,
# not mid-conversation (a cold gesture lands after the frontend stops polling).
if [ -n "$MOTION_GATEWAY_KEY" ]; then
  ( curl -s -m 120 -X POST "https://gpu.aiccore-uno.ai/v1/audio/speech" \
      -H "Authorization: Bearer $MOTION_GATEWAY_KEY" -H "Content-Type: application/json" \
      -d '{"model":"gpt-sovits","input":"起動しました"}' -o /dev/null \
      && echo "[warmup] TTS stack is warm" || echo "[warmup] TTS warmup failed (will cold-start on first use)" ) &
  (
    python - <<'PY' 2>/dev/null
import wave
w = wave.open("cache/warm_silent6s.wav", "wb")
w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
w.writeframes(b"\x00\x00" * 96000)
w.close()
PY
    [ -f cache/warm_silent6s.wav ] || exit 0
    curl -s -m 120 -X POST "https://gpu.aiccore-uno.ai/v1/gesture?format=joints" \
      -H "Authorization: Bearer $MOTION_GATEWAY_KEY" \
      -F "audio=@cache/warm_silent6s.wav;type=audio/wav" -F "text=起動" -o /dev/null \
      && echo "[warmup] gesture model is warm" || echo "[warmup] gesture warmup failed (will cold-start on first use)"
    # Text-streaming motion (ActionPlan, resident on PC-B GPU1): not On Demand —
    # if PC-B rebooted it must be relaunched by hand. Probe it so a dead service
    # is visible at startup instead of as silently-missing gestures.
    curl -s -m 30 -X POST "https://gpu.aiccore-uno.ai/v1/motion/stream" \
      -H "Authorization: Bearer $MOTION_GATEWAY_KEY" -H "Content-Type: application/json" \
      -d '{"text":"a person stands","seconds":1,"session":"warmup","chain":false}' -o /dev/null \
      && echo "[warmup] text-motion (ActionPlan) is up" \
      || echo "[warmup] text-motion DOWN — on PC-B: cd ~/Documents/DEV/ActionPlan-Code && CUDA_VISIBLE_DEVICES=0 setsid --fork nohup ./venv/bin/python actionplan_service.py --port 8893 </dev/null >/tmp/ap_service.log 2>&1"
    # STT (faster-whisper on PC-B GPU1 via the gateway sidecar): first request
    # after a sidecar restart pays the model load (~30s) — pay it here instead.
    curl -s -m 120 -X POST "https://gpu.aiccore-uno.ai/v1/audio/transcriptions" \
      -H "Authorization: Bearer $MOTION_GATEWAY_KEY" \
      -F "file=@cache/warm_silent6s.wav;type=audio/wav" -F "model=faster-whisper-large-v3" -o /dev/null \
      && echo "[warmup] STT model is warm" || echo "[warmup] STT warmup failed (will cold-start on first use)"
  ) &
fi

echo "GESTURE_ENABLED=$GESTURE_ENABLED  MOTION_GATEWAY_KEY=${MOTION_GATEWAY_KEY:+<set>}  BASE=$MOTION_GATEWAY_BASE"
# Use the venv python if it has the package installed, else the system python
# (this repo's deps are currently installed in the system Python, not .venv).
PYEXE=".venv/Scripts/python.exe"
"$PYEXE" -c "import open_llm_vtuber" >/dev/null 2>&1 || PYEXE="python"
exec "$PYEXE" run_server.py "$@"
