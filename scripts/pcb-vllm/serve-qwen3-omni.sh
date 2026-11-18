#!/usr/bin/env bash
# Qwen3-Omni-30B-A3B-Instruct on :8808 for PC-B `omni` mode (./kehai.sh mode omni).
# Versioned here; AICCORE-Nexus/stack/vllm-omni/serve-qwen3-omni.sh is a symlink
# to this file (Nexus ignores stack/). The venv (uv, Python 3.12, vLLM 0.32.0
# +cu129, vllm-omni 0.32.0) and the weights live in stack/vllm-omni.
#
# Weights: Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound — the Thinker (30B
# MoE) is INT4 W4A16 (group 128), the Talker and Code2Wav ride along in bf16.
# vLLM-Omni lists AutoRound for the Thinker stage on Ampere SM80+; nothing
# quantizes the Talker, which is why stages 1/2 clear quantization_config.
#
# Placement (no NVLink, no P2P here — stages talk through /dev/shm, not GPUs):
#   stage 0 Thinker   -> GPU1 (empty in omni mode: vLLM TP=2, ActionPlan,
#                        GestureLSM are stopped by pcb-mode.sh)   ~20 GB
#   stage 1 Talker    -> GPU0 (shares with render Chrome 3.9 GB + Whisper 1.8 GB)
#   stage 2 Code2Wav  -> GPU0
# gpu_memory_utilization is a fraction of the WHOLE card and vLLM checks that
# much is FREE at start, so GPU0's two stages must stay under ~0.7 together.
#
#   OMNI_STAGE=1 (default)  Thinker only: text out, GPT-SoVITS still speaks.
#   OMNI_STAGE=2            + Talker/Code2Wav: the model speaks (audio_output:true
#                           in conf.yaml's openai_omni_llm block to use it).
set -eu
cd "$HOME/Documents/DEV/AICCORE-Nexus/stack/vllm-omni"
export PATH="$PWD/.venv/bin:$PATH"
# Same trap as serve-qwen38.sh: FlashInfer JIT-builds its sampler with the
# system nvcc (12.4) and dies on its flags; vLLM's own sampler needs no build.
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL="${OMNI_MODEL:-models/qwen3-omni-int4-autoround}"
STAGE="${OMNI_STAGE:-1}"
PORT="${OMNI_PORT:-8808}"

# Stage ids are physical GPU indices: CUDA_VISIBLE_DEVICES is deliberately unset.
# The shipped thinking YAML spreads the Thinker over "0,1" with TP=2 and runs
# eager (verified on 2x H100); here it is ONE 3090 (no NVLink/P2P, nothing to
# gain from TP on a 3B-active MoE) with CUDA graphs on.
unset CUDA_VISIBLE_DEVICES
# moe_backend: vllm-omni forces "triton" for Qwen3-Omni unless told otherwise
# (engine/stage_init_utils.py), and vLLM's Triton WNA16 MoE kernel refuses this
# INT4 checkpoint ("expert bias is not supported", 2026-11-02). Marlin has no
# such limit on Ampere, so say so explicitly.
# Memory (3rd start, 2026-11-02): weights + profiling peak took ~19.1 GiB of the
# 0.85 x 23.57 GiB budget and left 0.89 GiB for KV, short of the 1.5 GiB a
# 16k context needs. 0.90 (21.2 GiB, still under the 21.4 GiB free next to
# GSV's 2.1 GiB) and a 4096-token prefill chunk (half the activation peak;
# a 12k persona prefills in three chunks) buy the KV back.
# 4th start: with 0.90 the profiler computed 2.37 GiB for KV and then failed to
# allocate it (2.35 GiB free — GSV's 2.1 GiB sits on the same card). As in
# serve-qwen38.sh, size the KV explicitly instead of trusting profiling on a
# shared card: 2.0 GB covers the 1.5 GiB a 16k context needs.
# CUDA graphs for PREFILL steps too (2026-11-04). vLLM sizes the graph grid
# from max_num_seqs: min(2 x max_num_seqs, 512) = 4 tokens here, so every
# prefill step of more than 4 new tokens (this turn's audio, the time line,
# the new user message) ran graph-less: 97 ms to the first token for ~10 new
# tokens on a cached prefix vs 49 ms for an exact repeat. Explicit sizes up to
# 512 tokens (attention stays outside the graph: PIECEWISE for mixed steps).
# OMNI_GRAPH_SIZES="" restores the default.
GRAPH_SIZES="${OMNI_GRAPH_SIZES-1,2,4,8,16,24,32,48,64,96,128,160,192,256,320,384,448,512}"
COMP=""
if [ -n "$GRAPH_SIZES" ]; then COMP=",\"compilation_config\":{\"cudagraph_capture_sizes\":[${GRAPH_SIZES}]}"; fi
# Context 16k -> 24k (2026-11-04): KV 2.0 -> 3.0 GB = ~30k tokens at 96 KiB
# per token (48 layers x 4 KV heads x 128 x K,V x bf16), room for one full
# 24k request beside a short one. GPU1 ~22.7 of 24 GiB; utilization 0.95 is
# only checked as FREE at start, and GPU1 is empty then.
THINKER='"0":{"devices":"1","tensor_parallel_size":1,"gpu_memory_utilization":0.95,"kv_cache_memory_bytes":3000000000,"max_model_len":24576,"max_num_batched_tokens":4096,"max_num_seqs":2,"enforce_eager":false,"moe_backend":"marlin"'"${COMP}"'}'
if [ "$STAGE" = "1" ]; then
  # Thinker-only deploy config shipped with vllm-omni (no Talker, no Code2Wav).
  DEPLOY="$(python - <<'PY'
import os, vllm_omni
print(os.path.join(os.path.dirname(vllm_omni.__file__), "deploy", "qwen3_omni_moe_thinking.yaml"))
PY
)"
  DEPLOY_ARGS=(--deploy-config "$DEPLOY")
  OVERRIDES="{${THINKER}}"
else
  DEPLOY_ARGS=()
  OVERRIDES="{${THINKER},\"1\":{\"devices\":\"0\",\"gpu_memory_utilization\":0.35,\"hf_overrides\":{\"quantization_config\":null}},\"2\":{\"devices\":\"0\",\"gpu_memory_utilization\":0.10,\"hf_overrides\":{\"quantization_config\":null}}}"
fi

echo "[serve-qwen3-omni] model=$MODEL stage=$STAGE port=$PORT"
echo "[serve-qwen3-omni] overrides=$OVERRIDES"
exec vllm serve "$MODEL" --omni \
  --host 127.0.0.1 --port "$PORT" \
  --served-model-name qwen3-omni \
  "${DEPLOY_ARGS[@]}" \
  --stage-overrides "$OVERRIDES" \
  --enable-auto-tool-choice --tool-call-parser hermes \
  "$@"
