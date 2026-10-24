#!/usr/bin/env bash
# Qwen3.8-27B (4-bit, AutoRound W4A16 + BF16 MTP head) on :8807 for PC-B.
# Versioned here; AICCORE-Nexus/stack/vllm-next/serve-qwen38.sh is a symlink to
# this file (Nexus ignores stack/). The venv and the weights live in vllm-next.
#
# --served-model-name keeps "qwen3-30b-a3b" as an alias so the gateway
# (VLLM_MODELS, routes that id to :8807) and KEHAI's conf.yaml work unchanged.
#
# Memory: GPUs are shared with Whisper / render Chrome (GPU0) and GSV /
# ActionPlan / gesturelsm (GPU1). KV is sized explicitly because profiling on a
# busy card misjudges it; the hybrid GDN model keeps KV in only 16 of 64 layers.
#
# Thinking is turned off server-side (voice latency); the client sends nothing.
# chat_template_kehai.jinja (next to this file) = the model's template with two
# changes, both for requests KEHAI really sends:
#   - a system turn mid-conversation is rendered instead of rejected. KEHAI puts
#     "Current local time" before EVERY user turn, so stock Qwen3.8 failed every
#     turn with 400 "System message must be at the beginning."
#   - no user turn is fine when add_generation_prompt is false: the prefix warm
#     (openai_compatible_llm.warm_prefix) of a brand-new conversation sends the
#     system prompt alone.
set -eu
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$HOME/Documents/DEV/AICCORE-Nexus/stack/vllm-next"
export PATH="$PWD/.venv/bin:$PATH"   # flashinfer JIT needs ninja from the venv
# FlashInfer JIT-builds its sampling kernels with whatever nvcc it finds, which
# here is the system /usr/bin/nvcc (12.4) — too old for its flags ("Unknown
# option '--compress-mode=size'"), so engine init dies. vLLM's own PyTorch
# sampler needs no build.
export VLLM_USE_FLASHINFER_SAMPLER=0
exec vllm serve models/qwen3.8-27b-int4 \
  --host 127.0.0.1 --port 8807 \
  --served-model-name qwen3.8-27b qwen3-30b-a3b \
  --tensor-parallel-size 2 \
  --max-model-len 24576 --max-num-seqs 4 \
  --gpu-memory-utilization 0.55 --kv-cache-memory-bytes 3G \
  --language-model-only \
  --enable-prefix-caching --prefix-match-unit 16 --mamba-ssm-cache-dtype bfloat16 \
  --chat-template "$HERE/chat_template_kehai.jinja" \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
