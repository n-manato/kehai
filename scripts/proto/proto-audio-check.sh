#!/usr/bin/env bash
# Run this ON THE PROTO (not on PC-B). Read-only: it changes no setting.
#
# Why: every visitor's voice reaches KEHAI cut at 4 kHz (telephone band) — the
# consonants that tell "weather" from "radio" are gone before any software here
# sees the sound (measured 2026-10-23 and again on 11-06 recordings: the energy
# just above 4 kHz is 29-30 dB below the energy just under it; full-band speech is
# within ~8 dB). Chrome's own processing was ruled out. The browser uses the
# "Default" input, a virtual source, and the PROTO's audio has a LADSPA chain
# (SC4 compressor + Simple amplifier). This prints how that input is built and
# MEASURES each input: speak normally for the 5 s each one records.
#
#   bash proto-audio-check.sh 2>&1 | tee proto-audio.txt     # then send proto-audio.txt
set -u
have() { command -v "$1" >/dev/null 2>&1; }

echo "== audio server"
have pactl && pactl info | grep -E "Server Name|Server Version|Default Sink|Default Source|Default Sample"
echo
echo "== inputs (sources)"
pactl list short sources
echo
echo "== source details"
pactl list sources | grep -E "^Source #|^[[:space:]]+(Name|Description|Sample Specification|Owner Module|Active Port):"
echo
echo "== modules that process or re-route sound"
pactl list short modules | grep -Ei "ladspa|echo|filter|remap|loopback|virtual|null|rnnoise|webrtc|combine|resample" || echo "(none)"
if have wpctl; then
  echo
  echo "== PipeWire"
  wpctl status | sed -n '/^Audio/,/^Video/p'
fi
echo
echo "== ALSA capture hardware"
have arecord && arecord -l

if have parec && have python3; then
  for src in $(pactl list short sources | awk '{print $2}' | grep -v '\.monitor$'); do
    echo
    echo "== $src: recording 5 s — speak now"
    tmp=$(mktemp /tmp/proto_mic_XXXX.raw)
    timeout 5 parec -d "$src" --format=s16le --rate=16000 --channels=1 --raw >"$tmp" 2>/dev/null
    python3 - "$tmp" <<'PY'
import math, struct, sys
d = open(sys.argv[1], "rb").read()
n = len(d) // 2
if n < 16000:
    print("  (too little audio — source silent or busy)")
    sys.exit()
x = struct.unpack("<%dh" % n, d[: 2 * n])[: 3 * 16000]
def band(f0, f1, step=100):
    # mean Goertzel power over f0..f1 (no numpy needed)
    tot, cnt = 0.0, 0
    for f in range(f0, f1, step):
        k = 2 * math.cos(2 * math.pi * f / 16000)
        s1 = s2 = 0.0
        for v in x:
            s0 = v + k * s1 - s2
            s2, s1 = s1, s0
        tot += s1 * s1 + s2 * s2 - k * s1 * s2
        cnt += 1
    return tot / cnt
peak = max(abs(v) for v in x)
lo, hi = band(3400, 3900), band(4100, 4600)
edge = 10 * math.log10(hi / lo) if lo > 0 and hi > 0 else float("nan")
verdict = "CUT AT 4 kHz (telephone band)" if edge < -12 else "full band"
print(f"  peak {20 * math.log10(max(peak, 1) / 32768):.0f} dBFS; 4.1-4.6 kHz vs 3.4-3.9 kHz: {edge:+.1f} dB -> {verdict}")
PY
    rm -f "$tmp"
  done
else
  echo
  echo "(parec or python3 missing: no measurement — the listing above is still useful)"
fi

cat <<'TXT'

== What the result means
- An input that says "full band" while the browser's "Default" says "CUT AT 4 kHz":
  point the browser at that input (PROTO viewer URL: add ?micid=<part of its name>,
  or make it the system default with: pactl set-default-source <name>).
- Every input "CUT AT 4 kHz": the microphone hardware or its driver mode is
  narrow-band (e.g. a Bluetooth headset profile, an 8 kHz USB mode). A USB
  microphone array with echo cancellation is the fix (see docs/PCB-HANDOFF.md).
TXT
