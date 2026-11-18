#!/bin/bash
# Build kehai-glgrab (GPU-side capture of the render window -> NVENC H.264).
# Needs gcc, libX11/libGL headers (installed) and the driver's libXcomposite /
# libnvidia-encode at run time. nvEncodeAPI.h is vendored in include/ (NVIDIA, MIT
# notice in the file; from FFmpeg/nv-codec-headers n12.2.72.0).
set -e
cd "$(dirname "$0")"
gcc -O2 -Wall -o kehai-glgrab.tmp kehai_glgrab.c -lX11 -lGL -ldl /usr/lib/x86_64-linux-gnu/libXcomposite.so.1
mv kehai-glgrab.tmp kehai-glgrab
echo "built $(pwd)/kehai-glgrab"
