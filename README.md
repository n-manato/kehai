<h1 align="center">KEHAI</h1>

<p align="center"><i>気配 — the sense that someone is there.</i></p>

KEHAI is a voice-interactive AI companion with a real-time 3D body. You speak, she
answers out loud in her own voice, and the avatar moves while she talks — co-speech
gestures, facial expression and lip sync driven by what is actually being said.

ENGLISH README | [日本語 README](./README.JP.md)

---

## What it is

- **Full-duplex voice.** Microphone in over WebRTC, streaming TTS out. Server-side
  VAD segments speech, ASR overlaps the segment, and generation starts on a
  semantic end-of-turn rather than a fixed silence timeout.
- **A body, not a portrait.** VRM (and Live2D) avatars with a pre-generated motion
  library, emotion-driven talk loops, blinking and expression morphs, and hair /
  skirt physics.
- **Characters are configuration.** Persona, voice, spoken language, avatar and
  motion profile all switch together from a single YAML file in `characters/`.
- **Runs on your own hardware.** This repo is the app (PC-A). The heavy models —
  LLM, GPT-SoVITS TTS, Whisper STT, motion generation — run behind a private
  gateway on the GPU box (PC-B).

## Requirements

- Python 3.10–3.12
- Node.js + npm (only if you rebuild the frontend)
- `OPENAI_API_KEY` in the environment
- Access to the GPU gateway (`MOTION_GATEWAY_KEY`, read from `.motion_gateway_key`)

## Quick start

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip && pip install -r requirements.txt

bash start_server.sh            # add --verbose for debug logging
```

Then open **https://127.0.0.1:12394/** and allow microphone access.

> **Always start with `start_server.sh`, never `python run_server.py`.**
> Only the script sets `MOTION_GATEWAY_KEY`, `GESTURE_ENABLED`, `GSV_STREAM` and
> the LLM context limits, and starts the tunnel supervisor that owns port 13000.
> Started bare, the LLM still replies — so the chat looks alive — but every TTS
> and STT call fails, and the symptom reads as "she won't speak."

A healthy start prints, as its first line:

```
GESTURE_ENABLED=1  MOTION_GATEWAY_KEY=<set>  BASE=http://127.0.0.1:13000/v1
```

### When she won't speak

```powershell
pwsh scripts/health-check.ps1
```

Read-only and safe to run mid-conversation. It walks server → startup env →
tunnel → voice stack → last turn and prints the fix for the first broken link.
If the server side is healthy, the fault is in the page: a `?debug=` URL freezes
the clock, a rebuild or restart needs a hard reload to re-establish the WebRTC
mic track, and microphone grants are per-origin — `https://127.0.0.1` and
`https://localhost` are **not** the same origin.

## Configuration

| What | Where |
| --- | --- |
| Main config | `conf.yaml` |
| Defaults / templates | `config_templates/` |
| Characters (persona + voice + language + avatar) | `characters/` |
| Persona prompts | `prompts/personas/` |
| Avatars | `live2d-models/`, VRM assets under `frontend/` |
| Motion library | `motion-lib-kawaii/` |

The default character is Miku (English) — an AI-CCORE secretarial android.
`characters/rem_ja.yaml` switches persona, voice, language and GSV weights back
to Rem (Japanese) in one click.

## Layout

```
run_server.py            entry point (launched by start_server.sh)
src/kehai/               backend: server, websocket handler, service context
  agent/ asr/ tts/ vad/  pluggable engines, one factory each
  conversations/         turn orchestration and TTS streaming
  motion/                motion library, retargeting, SMPL -> avatar
  config_manager/        typed, validated config models
frontend/                built web UI (served; do not hand-edit)
frontend-src/            web UI source (React + Vite + Cubism WebSDK)
scripts/                 build, health-check, tunnel supervisor
```

## Development

```bash
ruff check .                      # lint
ruff format .                     # format
pre-commit run --all-files
pwsh scripts/build-frontend.ps1   # rebuild + deploy the web UI
```

After a frontend build, hard-reload the browser (Ctrl+Shift+R).

## Credits

Built on [Open-LLM-VTuber](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)
(MIT) and its web frontend. See `LICENSE` and `frontend-src/LICENSE` — the
frontend carries its own license with commercial-use terms.
