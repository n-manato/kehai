#!/usr/bin/env bash
# Persistent SSH tunnels: localhost:18802/18803/18804 -> PC-B gesturelsm instances
# 127.0.0.1:8802/8803/8804 (behind, they run in parallel on the idle GPU).
#
# Why THREE separate SSH connections (not one tunnel with 3 -L forwards): a single SSH
# connection serializes concurrent transfers, which negates the multi-instance
# throughput (each gesture is ~170KB gzipped; under load one tunnel made them ~3.5s
# instead of ~1.5s). Separate connections transfer in parallel. The backend round-robins
# across the three ports (gesture_client._bases()). Uses key ~/.ssh/pcb_gesture (no
# password). Each tunnel auto-reconnects. Launched by start_server.sh.
#
# Why the LAN at all: the public tunnel (gpu.aiccore-uno.ai) adds ~2s relay latency and
# strips gzip; PC-B is on the LAN but only port 22 is open, so we forward over SSH.
set -u
KEY="$HOME/.ssh/pcb_gesture"
# LAN first (fast, needs the VPN), then the public hostname (works without the VPN,
# slightly higher latency). ConnectTimeout keeps the dead-route attempt short.
HOSTS=("ai-ccore@10.66.137.91" "ai-ccore@aiccore.unomaha.edu")

tunnel() {
  local L=$1 R=$2 H
  while true; do
    for H in "${HOSTS[@]}"; do
      ssh -i "$KEY" \
          -o StrictHostKeyChecking=accept-new \
          -o ConnectTimeout=6 \
          -o ServerAliveInterval=20 -o ServerAliveCountMax=3 \
          -o ExitOnForwardFailure=yes -N \
          -L "127.0.0.1:$L:127.0.0.1:$R" "$H"
      echo "[pcb_tunnel] $L->$R via $H dropped/failed; trying next..." >&2
      sleep 2
    done
  done
}

tunnel 18802 8802 &
tunnel 18803 8803 &
tunnel 18804 8804 &
tunnel 18806 9880 &  # GPT-SoVITS voice service (P0: Rem TTS on PC-B GPU1). 18806 because a
                     # LOCAL PC-A GSV instance squatted 18805 (kept as manual fallback).
wait
