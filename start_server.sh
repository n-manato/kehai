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
# The render host's display (scripts/render-host.sh writes it): the GPU X server :58 since
# 2026-11-05, Xvfb :57 before. render_stream.py captures RENDER_DISPLAY.
if [ -z "${RENDER_DISPLAY:-}" ] && [ -f .render-display ]; then
  export RENDER_DISPLAY="$(cat .render-display)"
fi
# The capture render_stream.py uses (kehai-glgrab: window texture -> NVENC). Without it
# the capture falls back to x11grab, whose X readback freezes the render whenever a
# CUDA client has work queued on GPU0 (2026-11-05). Build it if it is missing.
if [ ! -x scripts/pcb-render-x/glgrab/kehai-glgrab ]; then
  bash scripts/pcb-render-x/glgrab/build.sh || echo "[start] kehai-glgrab did not build — capture falls back to x11grab"
fi

# uvx lives in ~/.local/bin and is how conf.yaml's MCP servers ("time",
# "ddg-search") are launched. A login shell has it; setsid/nohup does not.
export PATH="$HOME/.local/bin:$PATH"

# Brain switch (./kehai.sh mode, scripts/pcb-mode.sh). `.mode` holds cascade
# (default: Whisper -> Qwen3.8 on :8807 -> GSV) or omni (Qwen3-Omni on :8808
# hears the turn's audio; Whisper stays for turn control). Everything below that
# differs between the two hangs off KEHAI_MODE; conf.yaml is shared.
export KEHAI_MODE="$([ -f .mode ] && tr -d '[:space:]' < .mode)"
export KEHAI_MODE="${KEHAI_MODE:-cascade}"

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
if [ "$KEHAI_MODE" = "omni" ]; then
  # conf.yaml's openai_omni_llm block replaces the character's llm_provider
  # (agent_factory honours this); the Thinker is served with max_model_len 16384
  # (INT4 weights ~17 GB + KV on one 3090), so the budget shrinks with it.
  export LLM_PROVIDER_OVERRIDE="${LLM_PROVIDER_OVERRIDE:-openai_omni_llm}"
  # Thinker context 24k since 2026-11-04 (serve-qwen3-omni.sh: KV 3 GB).
  export LLM_CTX_BUDGET=20000
  export LLM_CTX_MAX=24576
  # GestureLSM generates the co-speech motion in omni mode since 2026-11-05
  # (gateway DB enabled by pcb-mode.sh, on GPU0). Characters with
  # gesture_lsm_only (Iris) use it alone; the others keep KAWAII + it.
  # MOTION_LSM_UPGRADE=0 turns the generation off again. Stage 2 also drops GSV.
  export MOTION_LSM_UPGRADE="${MOTION_LSM_UPGRADE:-1}"
  export BODY_LITE=1   # ActionPlan stopped (pcb-mode.sh)
  # The speculative reply starts on the PAUSE with the audio itself (no wait
  # for the segment ASR), so the end-of-turn silence can shrink from 0.9 s
  # (6+22 frames) to ~0.6 s (6+13): the reply has ~0.4 s of head start either
  # way, and the semantic grace still holds mid-thought endings.
  export SPECULATE_AUDIO_FIRST="${SPECULATE_AUDIO_FIRST:-1}"
  # TTS backend in omni mode: gsv (GPT-SoVITS fine-tune, first chunk 0.4-1.2 s)
  # or qwen3tts (vLLM-Omni :8809, zero-shot clone of the same reference clip,
  # first chunk ~0.12 s). pcb-mode.sh starts the :8809 server in omni mode.
  # With a fine-tuned Qwen3-TTS voice in place (stack/qwen3-tts/models/current,
  # speaker name in models/current.speaker) the default is that voice; until
  # then GSV. OMNI_TTS=gsv|qwen3tts overrides either way.
  _TTSM="$HOME/Documents/DEV/AICCORE-Nexus/stack/qwen3-tts/models"
  export QWEN3TTS_SPEAKER="${QWEN3TTS_SPEAKER:-$([ -f "$_TTSM/current.speaker" ] && tr -d '[:space:]' < "$_TTSM/current.speaker")}"
  # 2026-11-04 06:00: Qwen3-TTS again by default with the retrained Aura voice
  # (stack/qwen3-tts/finetune/aura/output_r9/checkpoint-epoch-3 via models/current:
  # talker-only fine-tune, sub-talker left at Base — the first checkpoints
  # re-rolled the voice per sentence or never stopped; see docs/PCB-HANDOFF.md
  # "Qwen3-TTS の学習↔試験の周回"). Gate: scripts/qwen3tts_gate.py. Short
  # sentences are merged with the next one in tts_manager (QWEN3TTS_MERGE_SHORT,
  # default 16 chars) because a lone "Good evening." still comes out bright.
  # OMNI_TTS=gsv puts GPT-SoVITS back (first audio 0.5 s instead of 0.05 s).
  if [ -n "$QWEN3TTS_SPEAKER" ]; then export OMNI_TTS="${OMNI_TTS:-qwen3tts}"; else export OMNI_TTS="${OMNI_TTS:-gsv}"; fi
  # Whisper fills the history (the Thinker reads EARLIER turns as this text),
  # the chat panel and the noise screen; the Thinker hears the current turn's
  # audio itself. Since the fast release (SPEC_FAST_RELEASE) the reply no
  # longer waits for it, so the accurate setting is back (2026-11-04):
  # large-v3 (full, not turbo) + beam 5, as conf.yaml / cascade. turbo+greedy
  # wrote "Thank you." for coughs and knocks in 29 of 67 turns (2026-11-03).
  # Only utterances under SPEC_FAST_RELEASE_MIN_S still wait for the text.
  export ASR_MODEL_OVERRIDE="${ASR_MODEL_OVERRIDE:-faster-whisper-large-v3-full}"
  export ASR_BEAM_OVERRIDE="${ASR_BEAM_OVERRIDE:-5}"
  export QWEN3TTS_URL="${QWEN3TTS_URL:-http://127.0.0.1:8809/v1/audio/speech}"
  # 2026-11-04: 13 (0.61 s) -> 9. The transcript no longer gates the release
  # (SPEC_FAST_RELEASE below), so the endpoint can sit under the ASR's 0.5-0.65 s.
  export VAD_END_MISSES="${VAD_END_MISSES:-9}"
  # The pause (segment close) that starts the speculative reply: 4 frames =
  # 128 ms (conf.yaml: 6 = 192 ms). Shorter = the Thinker starts earlier;
  # the cost is more speculations dropped when the speaker resumes.
  export VAD_PAUSE_MISSES="${VAD_PAUSE_MISSES:-4}"
  # End-of-turn model (vad/smart_turn.py, Pipecat Smart Turn v3.2, CPU ONNX):
  # at each pause, a confident "finished" (>= SMART_TURN_RELEASE) releases the
  # reply before the VAD endpoint. OFF by default: measured 2026-11-04 over 9+9
  # PROTO-path turns it made the first audio LATER (505 vs 453 ms median) —
  # running in KEHAI's process it competes with the code reading the Thinker's
  # stream, even on one nice+10 thread — and released early in only 4 of 9.
  # Worth retrying out of process. SMART_TURN=1 to try it.
  export SMART_TURN="${SMART_TURN:-0}"
  export SMART_TURN_RELEASE="${SMART_TURN_RELEASE:-0.8}"
  # What the Thinker is given per turn (wav + context tail), newest 60, for
  # "she says my words make no sense" (basic_memory_agent._dump_omni_turn).
  export OMNI_AUDIO_DUMP="${OMNI_AUDIO_DUMP-logs/omni_audio}"
  export SPEC_FAST_RELEASE="${SPEC_FAST_RELEASE:-1}"
  # The server VAD listens to the PROTO viewer's mic directly (render client),
  # not to the render tab's copy that lags it 81-111 ms. 0 = the old path.
  export VAD_VIEWER_DIRECT="${VAD_VIEWER_DIRECT:-1}"
  # The reply's PCM goes to the PROTO viewer straight from the server
  # (render_stream.ViewerMirror), not via the render tab's playback + tap
  # (40-80 ms). The tab still plays it for lip-sync; its tapped copy is dropped.
  export VIEWER_PCM_DIRECT="${VIEWER_PCM_DIRECT:-1}"
  # Who is she answering? The Thinker looks at the PROTO camera frame at the
  # speech onset and each segment close (70-125 ms a look, off the reply path;
  # src/kehai/omni_addressee.py) and the turn is dropped unless someone is
  # talking to her — with a fresh frame, nobody in view is "not to her" too
  # (user's spec, 2026-11-03). No camera / frozen feed = answer as before.
  # Judged frames + scores land in logs/addressee/ for review
  # (OMNI_ADDRESSEE_DUMP="" stops that). OMNI_ADDRESSEE=0 turns it off.
  export OMNI_ADDRESSEE="${OMNI_ADDRESSEE:-1}"
  export OMNI_ADDRESSEE_MIN="${OMNI_ADDRESSEE_MIN:-5}"
  export OMNI_ADDRESSEE_DUMP="${OMNI_ADDRESSEE_DUMP-logs/addressee}"
  # Who is talking (src/kehai/active_speaker.py, LR-ASD on the lips of the faces in
  # front, CPU): answers are for a visitor whose lips move with the voice, not for
  # whatever words are heard (the TV, a phone, someone out of view). ASD=0 = off.
  # 2026-11-09: off. On the PROTO (4 kHz mic with noise suppression) the visitor
  # talking straight at her scored -0.2..-2.5 at every audio/video offset tried.
  export ASD="${ASD:-0}"
  # Mouth movement from the picture alone (src/kehai/lip_motion.py): log only.
  export LIP="${LIP:-1}"
else
  # cascade (Qwen3.8-27B reading Whisper's text). Its voice is chosen by
  # ./kehai.sh mode cascade and kept in .cascade-tts: qwen3tts = the same
  # Qwen3-TTS Aura voice as omni (on GPU1), with the body as in omni (no
  # ActionPlan / GestureLSM / GSV — no room on the cards). 2026-11-04.
  if [ "$(tr -d '[:space:]' < .cascade-tts 2>/dev/null)" = "qwen3tts" ]; then
    _TTSM="$HOME/Documents/DEV/AICCORE-Nexus/stack/qwen3-tts/models"
    export QWEN3TTS_SPEAKER="${QWEN3TTS_SPEAKER:-$([ -f "$_TTSM/current.speaker" ] && tr -d '[:space:]' < "$_TTSM/current.speaker")}"
    export OMNI_TTS="${OMNI_TTS:-qwen3tts}"   # name kept: service_context reads it in both modes
    export QWEN3TTS_URL="${QWEN3TTS_URL:-http://127.0.0.1:8809/v1/audio/speech}"
    export MOTION_LSM_UPGRADE="${MOTION_LSM_UPGRADE:-0}"
    export BODY_LITE=1
  fi
  # Brain-independent paths built for omni: VAD on the viewer's own mic, and
  # the reply's PCM straight to the viewer.
  export VAD_VIEWER_DIRECT="${VAD_VIEWER_DIRECT:-1}"
  export VIEWER_PCM_DIRECT="${VIEWER_PCM_DIRECT:-1}"
fi
# No public-gateway fallback on PC-B. The fallback exists for PC-A's SSH tunnel;
# here https://gpu.aiccore-uno.ai is this same box's gateway behind Cloudflare,
# so it cannot help when the local one is slow or down. With it on, the client
# also used a 12s read timeout: a long prefill on Qwen3.8 (Rem's first turn
# ~10s, a 20k-token compaction ~17s) timed out, and every request for the next
# 5 minutes went out through Cloudflare (2026-10-23: 502s). Empty = disabled.
export LLM_BASE_FALLBACK=""
# Same for the streaming TTS (tts/gsv_stream.py): its fallback is this box again.
export GSV_STREAM_FALLBACK=""
# And for STT (asr/gateway_whisper_asr.py). With a fallback configured the
# loopback request is cut at 5s; on 2026-10-23 21:22 one slow transcription went
# on to the public route and waited 30s for a 502 — the turn was lost.
export ASR_PUBLIC_FALLBACK=""
export MOTION_GATEWAY_KEY="$(cat .motion_gateway_key 2>/dev/null || echo)"  # gateway bearer key (also used by gsv_stream.py)
export ELEVENLABS_API_KEY="$(cat .elevenlabs_key 2>/dev/null || echo)"      # asr/elevenlabs_asr.py (shadow challenger)
# 相槌 on turn end (utils/backchannel.py); 0 = off. Off with Qwen3-TTS: its
# reply audio lands 0.3-0.5 s after the endpoint, and with the fast release
# (SPEC_FAST_RELEASE) the 0.56 s "Mm-hm." clip was sent on every turn and
# overlapped the reply's first words (2026-11-04). It covered GSV's 0.5-1 s gap.
if [ "${OMNI_TTS:-}" = "qwen3tts" ]; then export BACKCHANNEL="${BACKCHANNEL:-0}"; else export BACKCHANNEL="${BACKCHANNEL:-1}"; fi
# Addressee check OFF (2026-10-23, user's call): judging "not to her" from faces
# kept dropping real turns at the PROTO — the camera points high, so faces sit at
# the frame's bottom edge or out of it, and its feed sometimes freezes. 1 = on.
export ADDRESSEE="${ADDRESSEE:-0}"       # answer only speech said while someone faces the PROTO camera (face_detect.py); 0 = off
# Proto M2 (the small PROTO, from 2026-11-09): people stand 0.4-0.5 m (camera
# estimate) from it; the user stepping back was still answered at 0.9 and 1.2
# ("reacts from quite far away"). The estimate is uncalibrated for this camera.
# The big PROTO used the code default, 2.0.
# 2026-11-12: chosen per PROTO from the viewer's camera (face_detect.set_viewer_camera:
# "ProtoCam" = M2 -> 0.8, else 2.0). Set ADDRESSEE_NEAR_M only to force one value.
export ADDRESSEE_NEAR_M="${ADDRESSEE_NEAR_M:-}"
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
  # LLM: NOT On Demand and the gateway never starts it (scripts/pcb-stack-up.sh /
  # pcb-mode.sh do). Probed directly — /v1/models answers without a generation.
  if [ "$KEHAI_MODE" = "omni" ]; then
    ( curl -sf -m 5 http://127.0.0.1:8808/v1/models -o /dev/null \
        && echo "[warmup] LLM (vLLM-Omni :8808) is up" \
        || echo "[warmup] LLM DOWN — run ./kehai.sh mode omni" ) &
  else
    ( curl -sf -m 5 http://127.0.0.1:8807/v1/models -o /dev/null \
        && echo "[warmup] LLM (vLLM) is up" \
        || echo "[warmup] LLM DOWN — run ./kehai.sh up" ) &
  fi
  # GPT-SoVITS is warmed only when it is the reply voice. With Qwen3-TTS
  # (omni) it is stopped by pcb-mode.sh and this request would only fail.
  if [ "${OMNI_TTS:-gsv}" = "gsv" ]; then
  ( curl -sf -m 120 -X POST "$GW/audio/speech" \
      -H "Authorization: Bearer $MOTION_GATEWAY_KEY" -H "Content-Type: application/json" \
      -d '{"model":"gpt-sovits","input":"起動しました"}' -o /dev/null \
      && echo "[warmup] TTS stack is warm" || echo "[warmup] TTS warmup failed — if GPT-SoVITS is stopped: OMNI_TTS=gsv ./kehai.sh mode omni" ) &
  fi
  # Qwen3-TTS JIT-compiles its Triton sampling kernels on the FIRST synthesis
  # after the server starts: that request took 4.5 s, the next 0.35 s, and the
  # first reply after a mode switch waited 4.2 s for its first PCM (2026-11-04).
  # Pay it here with the same request shape the replies use.
  if [ "${OMNI_TTS:-gsv}" = "qwen3tts" ] && [ -n "${QWEN3TTS_SPEAKER:-}" ]; then
  ( t0=$(date +%s.%N)
    curl -sf -m 120 -X POST "${QWEN3TTS_URL:-http://127.0.0.1:8809/v1/audio/speech}" -H "Content-Type: application/json" \
      -d '{"model":"qwen3-tts","input":"Systems ready. How can I help you today?","task_type":"CustomVoice","voice":"'"$QWEN3TTS_SPEAKER"'","language":"English","response_format":"pcm","stream_format":"audio","seed":1234,"max_new_tokens":110}' -o /dev/null \
      && echo "[warmup] Qwen3-TTS is warm ($(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}') s)" \
      || echo "[warmup] Qwen3-TTS warmup failed — is :8809 up? (./kehai.sh mode status)" ) &
  fi
  (
    python3 - <<'PY' 2>/dev/null
import wave
w = wave.open("cache/warm_silent6s.wav", "wb")
w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
w.writeframes(b"\x00\x00" * 96000)
w.close()
PY
    [ -f cache/warm_silent6s.wav ] || exit 0
    # GestureLSM is On Demand: the first /gesture request starts it, so warm it
    # whenever the generation is on (omni too since 2026-11-05).
    if [ "${MOTION_LSM_UPGRADE:-1}" = "1" ] || [ -z "${BODY_LITE:-}" ]; then
    curl -sf -m 180 -X POST "$GW/gesture?format=joints" \
      -H "Authorization: Bearer $MOTION_GATEWAY_KEY" \
      -F "audio=@cache/warm_silent6s.wav;type=audio/wav" -F "text=起動" -o /dev/null \
      && echo "[warmup] gesture model is warm" || echo "[warmup] gesture warmup failed (will cold-start on first use)"
    fi
    # ActionPlan is stopped on purpose in omni mode (pcb-mode.sh): probing it there
    # only printed a misleading "DOWN — run ./kehai.sh up".
    if [ -z "${BODY_LITE:-}" ]; then
    # Text-streaming motion (ActionPlan, resident on GPU1): NOT On Demand — if the
    # box rebooted, scripts/pcb-stack-up.sh relaunches it. Probe it so a dead service is
    # visible at startup instead of as silently-missing gestures.
    curl -sf -m 30 -X POST "$GW/motion/stream" \
      -H "Authorization: Bearer $MOTION_GATEWAY_KEY" -H "Content-Type: application/json" \
      -d '{"text":"a person stands","seconds":1,"session":"warmup","chain":false}' -o /dev/null \
      && echo "[warmup] text-motion (ActionPlan) is up" \
      || echo "[warmup] text-motion DOWN — run ./kehai.sh up"
    fi
    # STT (faster-whisper on GPU1 via the gateway sidecar): the first request
    # after a sidecar restart pays the model load — pay it here instead.
    # MUST match conf.yaml's gateway_whisper_asr.model. Warming a different id
    # warms a different model: the real first turn would then cold-load (4.2s
    # measured) against the client's 5s loopback timeout, and a miss there falls
    # over to the PUBLIC gateway — a silent trip to the internet for a service
    # running on this same box.
    curl -sf -m 120 -X POST "$GW/audio/transcriptions" \
      -H "Authorization: Bearer $MOTION_GATEWAY_KEY" \
      -F "file=@cache/warm_silent6s.wav;type=audio/wav" -F "model=${ASR_MODEL_OVERRIDE:-faster-whisper-large-v3-full}" -o cache/warm_stt.json \
      && ! grep -q "Mock transcription" cache/warm_stt.json \
      && echo "[warmup] STT model is warm" \
      || echo "[warmup] STT DOWN (gateway answers with a mock transcript when its sidecar is down) — run ./kehai.sh up"
  ) &
fi

echo "GESTURE_ENABLED=$GESTURE_ENABLED  MOTION_GATEWAY_KEY=${MOTION_GATEWAY_KEY:+<set>}  BASE=$MOTION_GATEWAY_BASE  MODE=$KEHAI_MODE${LLM_PROVIDER_OVERRIDE:+ (llm=$LLM_PROVIDER_OVERRIDE)}${OMNI_ADDRESSEE:+  OMNI_ADDRESSEE=$OMNI_ADDRESSEE(min $OMNI_ADDRESSEE_MIN)}${ASD:+  ASD=$ASD}${LIP:+ LIP=$LIP}"

# The venv is python3.11 (system python3 here is 3.8; the app needs >=3.10).
PYEXE=".venv/bin/python"
[ -x "$PYEXE" ] || { echo "ERROR: $PYEXE missing — create it with: python3.11 -m venv .venv"; exit 1; }
exec "$PYEXE" run_server.py "$@"
