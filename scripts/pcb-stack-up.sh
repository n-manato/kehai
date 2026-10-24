#!/usr/bin/env bash
# Bring up everything KEHAI depends on on PC-B, then KEHAI itself. Idempotent:
# anything already answering is left alone. Run by `./kehai.sh up` and by the
# crontab @reboot entry — after a reboot nothing else starts these.
#
# Order and placement are deliberate (all hit on 2026-10-21):
#   gateway (:3000)  — owns the TTS / gesture lifecycle; those are NOT started here.
#                      If they stay down, check ModelRuntimeConfig.enabled in its DB.
#   ActionPlan       — GPU1. Left on GPU0 it takes the room GestureLSM (pinned to
#                      GPU0) needs, and the gateway answers gesture with 507 gpu_full.
#   STT (:8000)      — GPU0. Without it the gateway returns mock transcripts.
#   vLLM (:8807)     — needs the venv's bin on PATH: flashinfer JIT-builds with ninja.
#   KEHAI            — last, so its warmup probes see a complete stack.
set -u
export PATH="/usr/local/bin:/usr/bin:/bin"
NEXUS="$HOME/Documents/DEV/AICCORE-Nexus"
KEHAI="$(cd "$(dirname "$0")/.." && pwd)"

log() { echo "$(date -Is) [stack-up] $*"; }
ok() { curl -sf -o /dev/null -m 4 "$1"; }
wait_for() {  # url seconds label
  for _ in $(seq "$2"); do ok "$1" && { log "$3 is up"; return 0; }; sleep 1; done
  log "$3 did NOT come up within ${2}s"; return 1
}

# At @reboot the NVIDIA driver can still be initialising.
for _ in $(seq 120); do nvidia-smi >/dev/null 2>&1 && break; sleep 1; done

if ok http://127.0.0.1:3000/; then log "gateway already up"; else
  log "starting gateway"
  (cd "$NEXUS" && setsid --fork bash -c 'exec npm start </dev/null >/tmp/nexus.log 2>&1')
  wait_for http://127.0.0.1:3000/ 60 gateway
fi

# `[a]` / `[v]` keep pgrep from matching this script's own command line.
if ok http://127.0.0.1:8893/health || pgrep -f '[a]ctionplan_service.py' >/dev/null; then
  log "ActionPlan already up"
else
  log "starting ActionPlan on GPU1"
  (cd "$NEXUS/stack/ActionPlan-Code" && CUDA_VISIBLE_DEVICES=1 setsid --fork nohup ./venv/bin/python \
    actionplan_service.py --port 8893 </dev/null >/tmp/ap_service.log 2>&1)
  wait_for http://127.0.0.1:8893/health 120 ActionPlan
fi

# STT sidecar (faster-whisper behind the gateway). When it is down the gateway
# does not fail — it answers every transcription with a canned "Mock
# transcription..." sentence, so the mic looks alive and nothing is understood.
# GPU0: GPU1 is ~95% full with GSV/gesture/ActionPlan and large-v3 needs ~3.4 GB.
if ok http://127.0.0.1:8000/health; then log "STT sidecar already up"; else
  log "starting STT sidecar on GPU0"
  (cd "$NEXUS/model-service" && CUDA_VISIBLE_DEVICES=0 setsid --fork bash -c 'exec bash run.sh </dev/null >>/tmp/model-service.log 2>&1')
  wait_for http://127.0.0.1:8000/health 60 "STT sidecar"
fi

# 2026-10-23: Qwen3.8-27B (stack/vllm-next). It answers to the old id
# "qwen3-30b-a3b" too, so the gateway and conf.yaml are unchanged. The flags and
# the reasons for them live in scripts/pcb-vllm/serve-qwen38.sh.
# The previous model (Qwen3-30B-A3B) was DELETED for disk space the same day.
# Rolling back means rebuilding stack/vllm-voice first — a venv with vllm==0.11.0
# and `hf download stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ --local-dir
# models/qwen3-30b-a3b-awq` — then serving it with:
#   vllm serve models/qwen3-30b-a3b-awq --host 127.0.0.1 --port 8807 --served-model-name qwen3-30b-a3b --tensor-parallel-size 2 --max-model-len 24576 --max-num-seqs 4 --gpu-memory-utilization 0.52 --enable-auto-tool-choice --tool-call-parser hermes --enable-prefix-caching
if ok http://127.0.0.1:8807/v1/models; then log "vLLM already up"; else
  if ! pgrep -f '[v]llm serve' >/dev/null; then
    log "starting vLLM (Qwen3.8-27B)"
    (cd "$NEXUS/stack/vllm-next" && setsid --fork bash -c 'exec ./serve-qwen38.sh </dev/null >>vllm.log 2>&1')
  fi
  wait_for http://127.0.0.1:8807/v1/models 600 vLLM
fi

"$KEHAI/kehai.sh" start
