# CLAUDE.md

> **このチェックアウトは PC-B (aiccore) 側のインスタンスです。**
> 作業を始める前に必ず読むこと → @docs/PCB-HANDOFF.md
> （この箱固有の制約・起動確認の手順・踏みやすい罠がまとまっている。
> 下の記述は PC-A 前提のものが混ざっているので、食い違ったら HANDOFF が正）


This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KEHAI is a voice-interactive AI companion with Live2D avatar support that runs completely offline. It's a cross-platform Python application supporting real-time voice conversations, visual perception, and Live2D character animations. The project features modular architecture for LLM, ASR (Automatic Speech Recognition), TTS (Text-to-Speech), and other components.

## Essential Commands

### Development Setup
- **Create venv**: `python -m venv .venv`
- **Activate venv**: `.venv\\Scripts\\Activate.ps1`
- **Install dependencies**: `python -m pip install --upgrade pip && pip install -r requirements.txt`
- **Run server**: `bash start_server.sh` — NOT `python run_server.py`.
  `run_server.py` on its own starts a server that **cannot speak or listen**.
  `conf.yaml` points `openai_tts` and `gateway_whisper_asr` at
  `http://127.0.0.1:13000` (the SSH tunnel to the Nexus gateway), and
  `gateway_whisper_asr.api_key` is deliberately empty so the key comes from
  `MOTION_GATEWAY_KEY`. Only `start_server.sh` sets that key, sets
  `GSV_STREAM` / `GESTURE_ENABLED` / `LLM_CTX_*`, and starts the tunnel
  supervisor that owns port 13000. Started bare, the LLM still answers — so the
  chat looks alive — but every TTS and STT call fails and the symptom reads as
  「話しかけても喋らない」rather than as a startup mistake.
  Requires `OPENAI_API_KEY` in the environment.
- **Run with verbose logging**: `bash start_server.sh --verbose`
- **Diagnose "she won't speak"**: `pwsh scripts/health-check.ps1` — walks server,
  startup env, tunnel, voice stack and the last turn, and prints the fix for the
  first broken link. Read-only, safe to run mid-conversation. If it reports the
  server side healthy, the fault is in the PAGE (see section 6 of its output):
  a `?debug=` URL freezes the clock so nothing plays, a rebuild or server restart
  needs a hard reload to re-establish the WebRTC mic track, and microphone grants
  are per-origin — `https://127.0.0.1` and `https://localhost` are NOT the same
  origin. **This app is used at `https://127.0.0.1:12394/`.**
- **Check it came up correctly**: the first log line must read
  `GESTURE_ENABLED=1  MOTION_GATEWAY_KEY=<set>  BASE=http://127.0.0.1:13000/v1`,
  and port 13000 must have a listener (`Get-NetTCPConnection -LocalPort 13000`).
  If the tunnel is down, `logs/tunnel.log` says why; the public gateway
  `https://gpu.aiccore-uno.ai/v1` is the manual fallback for the two base_urls.
- **Update project**: `uv run upgrade.py`

### Running on PC-B instead (2026-08-06)

A second, independent install lives on the GPU box at
`aiccore-RTX3090:~/Documents/DEV/KEHAI`. Same code, its own `conf.yaml` and
`start_server.sh` — **the two are not meant to be synced**, and this PC's copy
still works exactly as documented above.

Why it exists: on PC-B the Nexus gateway is a LOOPBACK service (`*:3000`), so
LLM / STT / TTS / motion stop crossing the network entirely. No tunnel
supervisor, no `:13000`, no Cloudflare fallback in the hot path.

    # 1. server (PC-B) — has no display and no sound card; it only serves
    ssh aiccore-RTX3090 'cd ~/Documents/DEV/KEHAI && ./kehai.sh start'
    #    ./kehai.sh {start|stop|restart|status|log}   (status also probes :3000)

    # 2. tunnel (here) — only :22 reaches PC-B; every other port is firewalled
    pwsh scripts/pcb-app-tunnel.ps1

    # 3. browser (here)
    http://127.0.0.1:12395/

Three things about that URL are deliberate:

- **:12395, not :12394** — a separate browser origin from this PC's own server,
  so localStorage (look presets, the `webrtc` flag) and the microphone grant
  never bleed between the two instances.
- **http, not https** — PC-B runs with `enable_https: false`. The hop is already
  encrypted by SSH, `127.0.0.1` is a secure context even over http (so the mic
  works), and it avoids a cert interstitial on *every* start: `clean_cache()`
  deletes `cache/` on exit, so the self-signed cert is regenerated each run.
- **WebRTC audio will not work over the tunnel** — aiortc only offers PC-B's own
  host candidates and the browser cannot reach them, so ICE fails and the
  frontend falls back to the WebSocket audio path by itself. Functional, but the
  browser's echo cancellation is gone: **use headphones**. To skip the failed
  attempt, set `localStorage.webrtc='0'` on the :12395 origin only. Getting
  WebRTC back means opening PC-B's firewall or putting this PC on its tailnet.

### Code Quality
- **Lint code**: `ruff check .`
- **Format code**: `ruff format .`
- **Run pre-commit hooks**: `pre-commit run --all-files`

### Server Configuration
- **Main config file**: `conf.yaml` (user configuration)
- **Default configs**: `config_templates/conf.default.yaml` and `config_templates/conf.ZH.default.yaml`
- **Character configs**: `characters/` directory (YAML files)

## Architecture Overview

### Core Components

**WebSocket Server** (`src/kehai/server.py`):
- FastAPI-based server handling WebSocket connections
- Serves frontend, Live2D models, and static assets
- Supports both main client and proxy WebSocket endpoints

**Service Context** (`src/kehai/service_context.py`):
- Central dependency injection container
- Manages all engines (LLM, ASR, TTS, VAD, etc.)
- Each WebSocket connection gets its own service context instance

**WebSocket Handler** (`src/kehai/websocket_handler.py`):
- Routes WebSocket messages to appropriate handlers
- Manages client connections, groups, and conversation state
- Handles audio data, conversation triggers, and Live2D interactions

### Modular Engine System

The project uses a factory pattern for all AI engines:

**Agent System** (`src/kehai/agent/`):
- `agent_factory.py` - Factory for creating different agent types
- `agents/` - Various agent implementations (basic_memory, hume_ai, letta, mem0)
- `stateless_llm/` - Stateless LLM implementations (Claude, OpenAI, Ollama, etc.)

**ASR Engines** (`src/kehai/asr/`):
- Support for multiple ASR backends: Sherpa-ONNX, FunASR, Faster-Whisper, OpenAI Whisper, etc.
- Factory pattern for engine selection based on configuration

**TTS Engines** (`src/kehai/tts/`):
- Multiple TTS options: Azure TTS, Edge TTS, MeloTTS, CosyVoice, GPT-SoVITS, etc.
- Configurable voice cloning and multi-language support

**VAD (Voice Activity Detection)** (`src/kehai/vad/`):
- Silero VAD for detecting speech activity
- Essential for voice interruption without feedback loops

### Configuration Management

**Config System** (`src/kehai/config_manager/`):
- Type-safe configuration classes for each component
- Automatic validation and loading from YAML files
- Support for multiple character configurations and config switching

### Conversation System

**Conversation Handling** (`src/kehai/conversations/`):
- `conversation_handler.py` - Main conversation orchestration
- `single_conversation.py` - Individual user conversations
- `group_conversation.py` - Multi-user group conversations
- `tts_manager.py` - Audio streaming and TTS management

### MCP (Model Context Protocol) Integration

**MCP System** (`src/kehai/mcpp/`):
- Tool execution and server registry
- JSON detection and parameter extraction
- Integration with various MCP servers for extended functionality

## Key Development Patterns

### Error Handling
The codebase uses the missing `_cleanup_failed_connection` method pattern - when implementing new WebSocket handlers, ensure proper cleanup methods are implemented.

### Live2D Integration
- Models stored in `live2d-models/` directory
- Each model has its own `.model3.json` configuration
- Expression and motion control through WebSocket messages

### Audio Processing
- Real-time audio streaming through WebSocket
- Voice interruption support without headphones
- Multi-format audio support with proper codec handling

### Multi-language Support
- Character configurations support multiple languages
- TTS translation capabilities (speak in different language than input)
- I18n system for UI elements

## Important File Locations

- **Entry point**: `run_server.py`
- **Main server**: `src/kehai/server.py`
- **WebSocket routing**: `src/kehai/routes.py`
- **Configuration**: `conf.yaml` (user), `config_templates/` (defaults)
- **Frontend (built)**: `frontend/` — served static build; do not hand-edit
- **Frontend (source)**: `frontend-src/` — embedded KEHAI-Web source (React/Vite/TS + Cubism WebSDK) with this project's customizations. Rebuild & deploy with `pwsh scripts/build-frontend.ps1`
- **Live2D models**: `live2d-models/`
- **Character definitions**: `characters/`
- **Chat history**: `chat_history/`
- **Cache**: `cache/` (audio files, temporary data)

## Development Guidelines

### Adding New Engines
1. Create interface in appropriate directory (e.g., `asr_interface.py`)
2. Implement concrete class following existing patterns
3. Add to factory class (e.g., `asr_factory.py`)
4. Update configuration classes in `config_manager/`
5. Add configuration options to default YAML files

### WebSocket Message Handling
1. Add message type to `MessageType` enum in `websocket_handler.py`
2. Create handler method following `_handle_*` pattern
3. Register in `_init_message_handlers()` dictionary
4. Ensure proper error handling and client response

### Configuration Changes
- Always update both default config templates
- Maintain backward compatibility when possible
- Use the upgrade system for breaking changes
- Validate configurations in respective config manager classes

## Testing and Quality Assurance

The project uses:
- **Ruff** for linting and formatting (configured in `pyproject.toml`)
- **Pre-commit hooks** for automated quality checks
- **GitHub Actions** for CI/CD (`.github/workflows/`)
- Manual testing through web interface and desktop client

## Package Management

Uses **uv** (modern Python package manager):
- Dependencies defined in `pyproject.toml`
- Lock file: `uv.lock`
- Generated requirements: `requirements.txt` (auto-generated)
- Optional dependencies for specific features (e.g., `bilibili` extra)

## OpenAI Quick Start

- Set `OPENAI_API_KEY` in your shell before starting the server.
- `conf.yaml` is configured to use `openai_llm` for chat and `openai_tts` for speech.
- The current config uses local Whisper ASR, so first launch may download the `base` model into `models/whisper`.
