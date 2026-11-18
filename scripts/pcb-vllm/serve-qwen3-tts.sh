#!/usr/bin/env bash
# Qwen3-TTS-12Hz-1.7B-Base on :8809 for PC-B `omni` mode — the streaming TTS
# (OMNI_TTS=qwen3tts). Versioned here; AICCORE-Nexus/stack/qwen3-tts/
# serve-qwen3-tts.sh is a symlink (Nexus ignores stack/). Runs in the SAME
# vLLM-Omni venv as the Thinker (stack/vllm-omni/.venv); weights in
# stack/qwen3-tts/models (4.5 GB + 0.7 GB tokenizer).
#
# GPU0, beside render Chrome (3.9 GB) and Whisper (1.8 GB): stage 0 (talker)
# 0.25 x 24 GB with the KV sized explicitly (profiling on a shared card
# under-estimated it and the server died at start, 2026-11-03), stage 1
# (codec) 0.08. Measured first PCM byte 0.11-0.13 s, 4x real time.
# --allowed-local-media-path: KEHAI sends the reference clip as file:// (the
# character's GSV ref under stack/voice/refs); without it the server answers
# 500 "Cannot load local files" and every sentence is silent (2026-11-03).
set -eu
cd "$HOME/Documents/DEV/AICCORE-Nexus/stack/qwen3-tts"
VENV="$HOME/Documents/DEV/AICCORE-Nexus/stack/vllm-omni/.venv"
export PATH="$VENV/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
# models/current -> a fine-tuned CustomVoice checkpoint (finetune/aura/output/
# checkpoint-epoch-N, speaker baked in) when one has been made; else the Base
# model with zero-shot cloning from the character's reference clip.
if [ -z "${QWEN3TTS_MODEL:-}" ] && [ -e models/current/config.json ]; then
  QWEN3TTS_MODEL=models/current
fi
MODEL="${QWEN3TTS_MODEL:-models/Qwen3-TTS-12Hz-1.7B-Base}"
PORT="${QWEN3TTS_PORT:-8809}"
# A local copy of vllm_omni/deploy/qwen3_tts.yaml with smaller codec chunks
# (see comments inside) wins over the package default when present.
if [ -f qwen3_tts.deploy.yaml ]; then DEPLOY="$PWD/qwen3_tts.deploy.yaml"; else
DEPLOY="$(python - <<'PY'
import os, vllm_omni
print(os.path.join(os.path.dirname(vllm_omni.__file__), "deploy", "qwen3_tts.yaml"))
PY
)"
fi
unset CUDA_VISIBLE_DEVICES
# GPU0 in omni (GPU1 is the Thinker's); GPU1 in cascade with the Qwen3-TTS
# voice (GPU0 then holds half of Qwen3.8-27B beside the render and Whisper).
GPU="${QWEN3TTS_GPU:-0}"
# On GPU0, join CUDA MPS when it runs (scripts/pcb-mps.sh): its two stage processes then
# share one context with GestureLSM and Whisper, and the 60 fps render keeps its turn on
# the GPU (2026-11-05). No MPS daemon = no change.
if [ "$GPU" = 0 ]; then eval "$(CUDA_VISIBLE_DEVICES=0 "$HOME/Documents/DEV/KEHAI/scripts/pcb-mps.sh" env)"; fi
echo "[serve-qwen3-tts] MPS: ${CUDA_MPS_PIPE_DIRECTORY:-off}"
# Talker pacing (tts_pace/sitecustomize.py): sleep this long after each talker step so the
# render on the same GPU gets its turn. 0 = off.
export KEHAI_TTS_STEP_SLEEP_MS="${KEHAI_TTS_STEP_SLEEP_MS:-0}"
export PYTHONPATH="$HOME/Documents/DEV/KEHAI/scripts/pcb-vllm/tts_pace${PYTHONPATH:+:$PYTHONPATH}"
echo "[serve-qwen3-tts] talker step sleep: ${KEHAI_TTS_STEP_SLEEP_MS} ms, eager: ${QWEN3TTS_EAGER:-0}"
# QWEN3TTS_EAGER=0|1|talker|code2wav: run that stage without CUDA graphs (experiment,
# 2026-11-05: the TTS starves the 60 fps render on the same GPU; graphs are suspected).
# QWEN3TTS_EAGER=piecewise: the talker keeps CUDA graphs but only PIECEWISE (split at the
# attention ops), leaving gaps between pieces.
E0=false; E1=false; C0=""
# Default: full CUDA graphs (~4x real time, first audio ~0.06 s). Earlier on 2026-11-05
# full graphs held the 60 fps render at 1-5 fps and PIECEWISE (~2x real time) was the
# default on GPU0 — but the starving was the x11grab capture: its X readback waited for
# the GPU to drain the TTS's queued graphs and held Chrome's presents. With the GPU
# capture (render_stream: kehai-glgrab) the render holds 60 with full graphs. Use
# QWEN3TTS_EAGER=piecewise again if the capture is RENDER_CAPTURE=x11grab.
QWEN3TTS_EAGER="${QWEN3TTS_EAGER:-0}"
case "$QWEN3TTS_EAGER" in
  1) E0=true; E1=true ;; talker) E0=true ;; code2wav) E1=true ;;
  piecewise) C0=',"compilation_config":{"cudagraph_mode":"PIECEWISE"}' ;;
esac
OVERRIDES='{"0":{"devices":"'"$GPU"'","gpu_memory_utilization":0.25,"kv_cache_memory_bytes":700000000,"max_num_seqs":4,"enforce_eager":'"$E0"''"$C0"'},"1":{"devices":"'"$GPU"'","gpu_memory_utilization":0.08,"enforce_eager":'"$E1"'}}'
echo "[serve-qwen3-tts] model=$MODEL port=$PORT"
exec vllm serve "$MODEL" --omni --host 127.0.0.1 --port "$PORT" \
  --served-model-name qwen3-tts \
  --deploy-config "$DEPLOY" --stage-overrides "$OVERRIDES" \
  --allowed-local-media-path "$HOME/Documents/DEV/AICCORE-Nexus/stack/voice/refs" "$@"
