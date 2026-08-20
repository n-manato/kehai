<#
.SYNOPSIS
  Keep an SSH tunnel open from this PC to the KEHAI server running on PC-B.

.DESCRIPTION
  KEHAI now also runs on PC-B (aiccore, /home/ai-ccore/Documents/DEV/KEHAI),
  where the Nexus gateway is a loopback service — LLM, STT, TTS and motion all
  stay on that box. Only :22 is reachable from here (measured: :3000 and every
  other port are firewalled), so the browser reaches it through this tunnel.

      this PC  https://127.0.0.1:12395  --ssh:22-->  PC-B  127.0.0.1:12394

  Deliberately a DIFFERENT local port from PC-A's own server (:12394): the two
  instances then have separate browser origins, so localStorage (model/look
  presets, the webrtc flag) and the microphone grant never bleed across. The
  cost is one extra mic permission click the first time.

  PC-B serves plain HTTP (enable_https: false), unlike PC-A. That is not a
  downgrade here: the hop is already encrypted by SSH, and 127.0.0.1 counts as
  a secure context in the browser even over http, so the microphone still
  works. It also removes the cert interstitial that would otherwise appear on
  EVERY start (the server wipes cache/ on exit, so the self-signed cert is
  regenerated each run).

  NOTE: WebRTC audio cannot traverse this tunnel — aiortc only offers PC-B's
  host candidates and the browser cannot reach them. The frontend detects that
  and falls back to the WebSocket audio path on its own, which works but loses
  the browser's echo cancellation. Use headphones with the PC-B instance.

  Start the server on PC-B first:
      ssh aiccore-RTX3090 'cd ~/Documents/DEV/KEHAI && setsid --fork nohup bash start_server.sh </dev/null >>logs/server.log 2>&1'

.EXAMPLE
  pwsh scripts/pcb-app-tunnel.ps1
  # then open http://127.0.0.1:12395/
#>
param(
  [int]$LocalPort = 12395,
  [int]$RemotePort = 12394,
  [string]$SshHost = 'aiccore-RTX3090'
)

$ErrorActionPreference = 'Continue'

# A listener already on the local port is almost always a previous run of this
# script, and a second one would just fail to bind every retry. Say so and stop
# instead of looping on an error the user cannot see (the ssh output is quiet).
$busy = Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue
if ($busy) {
  Write-Host "Port $LocalPort already has a listener (PID $($busy[0].OwningProcess)) — a tunnel is probably already up." -ForegroundColor Yellow
  Write-Host "Open http://127.0.0.1:$LocalPort/ , or stop that process first."
  exit 1
}

Write-Host "KEHAI @ PC-B tunnel: http://127.0.0.1:$LocalPort/  ->  ${SshHost}:$RemotePort" -ForegroundColor Cyan
Write-Host "Ctrl+C to stop. Reconnects automatically if the link drops.`n"

while ($true) {
  $started = Get-Date
  # -N: no remote command, just the forward. ExitOnForwardFailure so a port that
  # cannot bind ends the attempt loudly instead of leaving a live-looking ssh
  # session with no forward behind it.
  ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 `
      -L "${LocalPort}:127.0.0.1:${RemotePort}" $SshHost

  $lasted = [int]((Get-Date) - $started).TotalSeconds
  # Space out retries. A tight loop against this host's :22 has tripped the
  # border SSH rate-limit before (2026-07-26), which then blocks BOTH routes for
  # hours — the failure mode looks like PC-B being down when it is fine.
  $wait = if ($lasted -lt 10) { 20 } else { 5 }
  Write-Host "[tunnel] link ended after ${lasted}s — reconnecting in ${wait}s..." -ForegroundColor DarkGray
  Start-Sleep -Seconds $wait
}
