#!/usr/bin/env bash
# CUDA MPS for GPU0 (2026-11-05, "60 FPS にしたい").
#
# GPU0 runs the render (Chrome + the GPU X server + NVENC) beside three CUDA services —
# Qwen3-TTS (two vLLM-Omni stage processes), GestureLSM and Whisper. The GPU time-slices
# per CONTEXT: one busy CUDA process left the 60 fps render ~30 fps, three froze it for
# seconds (real replies: 1-9 fps for ~10 s). Under MPS those services share ONE context,
# so three busy ones cost the render what one does (measured 27-32 fps at 100% load,
# vs 0-15 without). The graphics processes are not MPS clients.
#
#   scripts/pcb-mps.sh start|stop|status
#   eval "$(scripts/pcb-mps.sh env)"   # in a GPU0 service's launcher: join MPS if it runs
#
# Short /tmp path on purpose: the control socket path must fit in a Unix socket name
# (~108 bytes). Under a long directory the clients silently ran WITHOUT MPS.
set -u
MPS_DIR="${KEHAI_MPS_DIR:-/tmp/mps-gpu0}"
PIPE="$MPS_DIR/pipe"; LOG="$MPS_DIR/log"
running() { [ -S "$PIPE/control" ] && pgrep -u "$(id -u)" -f '^nvidia-cuda-mps-control' >/dev/null; }
case "${1:-status}" in
  start)
    if running; then echo "MPS (GPU0) already running"; exit 0; fi
    mkdir -p "$PIPE" "$LOG"
    # GPU0 = PCI 0x1A in PCI order (nvidia-smi 0); the server exposes only that GPU, so a
    # client's CUDA_VISIBLE_DEVICES=0 means it, and GPU1 (the Thinker) is never touched.
    CUDA_MPS_PIPE_DIRECTORY="$PIPE" CUDA_MPS_LOG_DIRECTORY="$LOG" \
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 nvidia-cuda-mps-control -d
    sleep 0.5; running && echo "MPS (GPU0) started: $PIPE" || { echo "MPS did not start — see $LOG"; exit 1; }
    ;;
  stop)
    if ! running; then echo "MPS not running"; exit 0; fi
    echo quit | CUDA_MPS_PIPE_DIRECTORY="$PIPE" nvidia-cuda-mps-control; echo "MPS stopped"
    ;;
  env)
    # Printed for eval: join MPS only when it runs and the service targets GPU0.
    # KEHAI_MPS_SKIP=1: keep this one service out (experiments / a service that misbehaves under MPS).
    if [ "${KEHAI_MPS_SKIP:-0}" != "1" ] && running && [ "${CUDA_VISIBLE_DEVICES:-0}" = "0" ]; then
      echo "export CUDA_MPS_PIPE_DIRECTORY='$PIPE' CUDA_MPS_LOG_DIRECTORY='$LOG'"
    fi
    ;;
  status)
    if running; then echo "MPS (GPU0) running: $PIPE"; grep -c "NEW CLIENT" "$LOG/control.log" 2>/dev/null | sed 's/^/  clients so far: /'; else echo "MPS not running"; fi
    ;;
  *) echo "usage: $0 {start|stop|status|env}"; exit 2 ;;
esac
