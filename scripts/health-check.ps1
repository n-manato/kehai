<#
.SYNOPSIS
  Answer "why isn't she speaking?" in about ten seconds.

.DESCRIPTION
  Every silent-failure this project has hit looks identical from the outside —
  the chat is alive, the badge says 接続済み, and no audio ever comes. The cause
  has been a different link in the chain each time, and none of them announce
  themselves. This walks the chain in the order that fails most often and prints
  what to do about the first thing that is broken.

  Deliberately does NOT touch the server or the browser: it only reads state, so
  it is safe to run while a conversation is in progress.

.EXAMPLE
  pwsh scripts/health-check.ps1
#>

$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)

$fails = @()
function Ok   ($m) { Write-Host "  OK   $m" -ForegroundColor Green }
function Bad  ($m, $fix) { Write-Host "  FAIL $m" -ForegroundColor Red; $script:fails += $fix }
function Warn ($m) { Write-Host "  WARN $m" -ForegroundColor Yellow }

Write-Host "`n=== 1. app server (:12394) ===" -ForegroundColor Cyan
$srv = Get-NetTCPConnection -LocalPort 12394 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($srv) {
  Ok "listening (pid $($srv.OwningProcess))"
  # A genuine double-start is two INDEPENDENT trees. A normal launch already
  # shows two processes — the venv interpreter re-execs the system one and stays
  # as its parent — so counting processes alone reports a false alarm every time.
  $procs = @(Get-CimInstance Win32_Process -Filter "Name LIKE '%python%'" -ErrorAction SilentlyContinue |
             Where-Object { $_.CommandLine -match 'run_server' })
  $ids = $procs | ForEach-Object { $_.ProcessId }
  $roots = @($procs | Where-Object { $ids -notcontains $_.ParentProcessId })
  if ($roots.Count -gt 1) {
    Warn "$($roots.Count) independent run_server instances (pids $($roots.ProcessId -join ', ')) — only one owns the port; stop the others"
  }
  # NOTE: do not try to tell "started via start_server.sh" from the command line.
  # The script ends in `exec python`, which REPLACES the shell, so the surviving
  # process always looks like a bare `python run_server.py`. Section 2 is the
  # authoritative check — it reads what the script actually printed.
} else {
  Bad "nothing listening on 12394" "Start it:  Start-Process 'C:\Program Files\Git\bin\bash.exe' -ArgumentList 'start_server.sh' -WorkingDirectory (Get-Location) -RedirectStandardOutput server.log -RedirectStandardError server.err.log -WindowStyle Hidden"
}

Write-Host "`n=== 2. startup environment ===" -ForegroundColor Cyan
if (Test-Path server.log) {
  $envLine = Get-Content server.log -TotalCount 40 | Where-Object { $_ -match 'GESTURE_ENABLED=' } | Select-Object -Last 1
  if ($envLine) {
    if ($envLine -match 'MOTION_GATEWAY_KEY=<set>') { Ok $envLine.Trim() }
    else { Bad "gateway key NOT set: $($envLine.Trim())" "Restart via start_server.sh — ASR sends no auth without MOTION_GATEWAY_KEY." }
  } else {
    Bad "no 'GESTURE_ENABLED=' line in server.log" "The server was not started by start_server.sh. Restart it that way."
  }
} else { Warn "no server.log (started without output redirection?)" }

Write-Host "`n=== 3. SSH tunnel (:13000) ===" -ForegroundColor Cyan
$tun = Get-NetTCPConnection -LocalPort 13000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($tun) { Ok "listening (pid $($tun.OwningProcess))" }
else { Bad "port 13000 has no listener — conf.yaml points BOTH TTS and ASR at it" "See logs/tunnel.log. Either fix the tunnel, or temporarily point openai_tts.base_url and gateway_whisper_asr.base_url at https://gpu.aiccore-uno.ai/v1" }

# The tunnel being up right now is not the same as it STAYING up. Without the
# supervisor resident, the next drop is permanent and silent.
#
# Checked by command line, NOT by logs/tunnel.pid: that file holds an msys pid
# from Git Bash, which is a different numbering from the Windows pid, so
# Get-Process on it reports "dead" for a perfectly healthy supervisor.
$sup = @(Get-CimInstance Win32_Process -Filter "Name='bash.exe'" -ErrorAction SilentlyContinue |
         Where-Object { $_.CommandLine -match 'tunnel_supervisor' })
if ($sup.Count -gt 0) { Ok "supervisor resident (pid $($sup[0].ProcessId)) — a drop will self-heal" }
else { Bad "tunnel supervisor NOT running — the tunnel will not come back on its own" "Start it: Start-Process 'C:\Program Files\Git\bin\bash.exe' -ArgumentList 'scripts/tunnel_supervisor.sh' -WorkingDirectory (Get-Location) -WindowStyle Hidden" }

Write-Host "`n=== 4. voice stack (does TTS actually return audio?) ===" -ForegroundColor Cyan
if (Test-Path .motion_gateway_key) {
  $key = (Get-Content .motion_gateway_key -Raw).Trim()
  foreach ($u in @("http://127.0.0.1:13000/v1/audio/speech", "https://gpu.aiccore-uno.ai/v1/audio/speech")) {
    try {
      $sw = [Diagnostics.Stopwatch]::StartNew()
      $r = Invoke-WebRequest -Uri $u -Method POST -Headers @{ Authorization = "Bearer $key" } `
             -ContentType "application/json" -Body '{"model":"gpt-sovits","input":"test"}' -TimeoutSec 90
      $sw.Stop()
      Ok "$u -> HTTP $($r.StatusCode), $($r.RawContentLength) bytes, $([int]$sw.Elapsed.TotalMilliseconds) ms"
    } catch {
      if ($u -match '127.0.0.1') { Bad "$u -> $($_.Exception.Message)" "The tunnel is bound but not forwarding. Restart the supervisor: bash scripts/tunnel_supervisor.sh" }
      else { Bad "$u -> $($_.Exception.Message)" "The public gateway is unreachable too — check the network / PC-B." }
    }
  }
} else { Bad ".motion_gateway_key missing" "Restore the key file at the repo root." }

Write-Host "`n=== 5. did the last turn actually speak? ===" -ForegroundColor Cyan
if (Test-Path server.err.log) {
  $tail = Get-Content server.err.log -Tail 400
  $resp = $tail | Select-String -Pattern 'AI response:' | Select-Object -Last 1
  $used = $tail | Select-String -Pattern 'motion-usage' | Select-Object -Last 1
  $strm = $tail | Select-String -Pattern 'TRUE TTS STREAMING enabled' | Select-Object -Last 1
  if ($strm) { Ok ($strm.Line -replace '.*\| ', '') } else { Warn "no streaming-TTS line — GSV_STREAM may be off" }
  if ($resp) { Ok ("last reply: " + (($resp.Line -replace '.*AI response: ', '').Substring(0, [Math]::Min(70, ($resp.Line -replace '.*AI response: ', '').Length)))) }
  else { Warn "no 'AI response' in the recent log — no turn has completed lately" }
  if ($used) { Ok ($used.Line -replace '^.*?\[motion-usage\]', '[motion-usage]') }
} else { Warn "no server.err.log — the app's own log goes to stderr, redirect it" }

Write-Host "`n=== 6. browser-side (cannot be checked from here) ===" -ForegroundColor Cyan
Write-Host "  If everything above is OK and she is still silent, it is the PAGE:" -ForegroundColor Gray
Write-Host "   - URL must have NO ?debug= — that freezes the clock and nothing plays." -ForegroundColor Gray
Write-Host "   - Hard-reload (Ctrl+Shift+R) after any rebuild or server restart," -ForegroundColor Gray
Write-Host "     which also re-establishes the WebRTC mic track." -ForegroundColor Gray
Write-Host "   - Microphone must be allowed FOR THIS EXACT ORIGIN. https://127.0.0.1" -ForegroundColor Gray
Write-Host "     and https://localhost are different origins with separate grants." -ForegroundColor Gray
Write-Host "   - Check in DevTools console:" -ForegroundColor Gray
Write-Host "       (await navigator.permissions.query({name:'microphone'})).state   // want 'granted'" -ForegroundColor Gray
Write-Host "       window.__AI_DEBUG__.state().frozen                                // want false" -ForegroundColor Gray

if ($fails.Count -eq 0) {
  Write-Host "`nServer side is healthy. If she is silent, work through section 6.`n" -ForegroundColor Green
} else {
  Write-Host "`n--- what to fix, in order ---" -ForegroundColor Yellow
  $i = 1; foreach ($f in $fails) { Write-Host "$i. $f" -ForegroundColor Yellow; $i++ }
  Write-Host ""
}
