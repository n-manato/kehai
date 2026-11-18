#!/usr/bin/env bash
# Serve ONE Qwen3-TTS checkpoint on :8809 and run the acceptance gate against
# it (scripts/qwen3tts_gate.py), then stop the server again. The training ↔
# test loop of 2026-11-03: GPU0 cannot hold the trainer (11.5 GiB) and the
# server (8.8 GiB) beside Chrome / Whisper / GSV at the same time, so each
# checkpoint is served only for the ~1 min the gate needs.
#
#   scripts/pcb-vllm/qwen3tts-gate-ckpt.sh <checkpoint dir> [gate args...]
#   e.g. ... finetune/aura/output_r1/checkpoint-epoch-6 --save-dir /tmp/x
#
# Exit code is the gate's (0 = PASS). The server is stopped by PID, never by
# name (pkill -f '^VLLM::' would also kill the Thinker's workers).
set -u
CKPT="$(cd "$1" && pwd)"; shift
KEHAI="$(cd "$(dirname "$0")/../.." && pwd)"
TTS="$HOME/Documents/DEV/AICCORE-Nexus/stack/qwen3-tts"

if pgrep -f '[v]llm serve.*--port 8809' >/dev/null; then
  echo "a :8809 server is already running — stop it first (kill \$(pgrep -f '[v]llm serve.*--port 8809'))" >&2
  exit 2
fi
[ -e "$CKPT/speech_tokenizer/model.safetensors" ] || {
  echo "linking speech tokenizer weights into $CKPT"
  ln -s "$TTS/models/Qwen3-TTS-12Hz-1.7B-Base/speech_tokenizer/model.safetensors" "$CKPT/speech_tokenizer/model.safetensors"
}
echo "[gate-ckpt] serving $CKPT"
(cd "$TTS" && QWEN3TTS_MODEL="$CKPT" setsid --fork bash -c 'exec ./serve-qwen3-tts.sh </dev/null >>server.log 2>&1')
for _ in $(seq 240); do curl -sf -m 2 -o /dev/null http://127.0.0.1:8809/v1/models && break; sleep 1; done
PID="$(pgrep -f '[v]llm serve.*--port 8809' | head -1)"
if ! curl -sf -m 2 -o /dev/null http://127.0.0.1:8809/v1/models; then
  echo "[gate-ckpt] server did not come up (see $TTS/server.log)" >&2
  [ -n "$PID" ] && kill "$PID"
  exit 3
fi
echo "[gate-ckpt] up (pid $PID); running the gate"
(cd "$KEHAI" && .venv/bin/python scripts/qwen3tts_gate.py "$@")
RC=$?
echo "[gate-ckpt] stopping server $PID"
kill "$PID"; timeout 60 tail --pid="$PID" -f /dev/null
exit $RC
