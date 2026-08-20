import os
import sys
import time
import atexit
import socket
import asyncio
import argparse
import threading
import subprocess
from pathlib import Path
from urllib.parse import urlparse
import tomli
import uvicorn
from loguru import logger
from upgrade_codes.upgrade_manager import UpgradeManager

from src.kehai.server import WebSocketServer
from src.kehai.config_manager import Config, read_yaml, validate_config

os.environ["HF_HOME"] = str(Path(__file__).parent / "models")
os.environ["MODELSCOPE_CACHE"] = str(Path(__file__).parent / "models")

upgrade_manager = UpgradeManager()


def get_version() -> str:
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    return pyproject["project"]["version"]


def init_logger(console_log_level: str = "INFO") -> None:
    logger.remove()
    # Console output
    logger.add(
        sys.stderr,
        level=console_log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {message}",
        colorize=True,
    )

    # File output
    logger.add(
        "logs/debug_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message} | {extra}",
        backtrace=True,
        diagnose=True,
    )


def check_frontend_submodule(lang=None):
    """
    Check if the frontend submodule is initialized. If not, attempt to initialize it.
    If initialization fails, log an error message.
    """
    if lang is None:
        lang = upgrade_manager.lang

    frontend_path = Path(__file__).parent / "frontend" / "index.html"
    if not frontend_path.exists():
        if lang == "zh":
            logger.warning("未找到前端子模块，正在尝试初始化子模块...")
        else:
            logger.warning(
                "Frontend submodule not found, attempting to initialize submodules..."
            )

        try:
            subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"], check=True
            )
            if frontend_path.exists():
                if lang == "zh":
                    logger.info("👍 前端子模块（和其他子模块）初始化成功。")
                else:
                    logger.info(
                        "👍 Frontend submodule (and other submodules) initialized successfully."
                    )
            else:
                if lang == "zh":
                    logger.critical(
                        '子模块初始化失败。\n你之后可能会在浏览器中看到 {{"detail":"Not Found"}} 的错误提示。请检查我们的快速入门指南和常见问题页面以获取更多信息。'
                    )
                    logger.error(
                        "初始化子模块后，前端文件仍然缺失。\n"
                        + "你是否手动更改或删除了 `frontend` 文件夹？\n"
                        + "它是一个 Git 子模块 - 你不应该直接修改它。\n"
                        + "如果你这样做了，请使用 `git restore frontend` 丢弃你的更改，然后再试一次。\n"
                    )
                else:
                    logger.critical(
                        'Failed to initialize submodules. \nYou might see {{"detail":"Not Found"}} in your browser. Please check our quick start guide and common issues page from our documentation.'
                    )
                    logger.error(
                        "Frontend files are still missing after submodule initialization.\n"
                        + "Did you manually change or delete the `frontend` folder?  \n"
                        + "It's a Git submodule — you shouldn't modify it directly.  \n"
                        + "If you did, discard your changes with `git restore frontend`, then try again.\n"
                    )
        except Exception as e:
            if lang == "zh":
                logger.critical(
                    f'初始化子模块失败: {e}。\n怀疑你跟 GitHub 之间有网络问题。你之后可能会在浏览器中看到 {{"detail":"Not Found"}} 的错误提示。请检查我们的快速入门指南和常见问题页面以获取更多信息。\n'
                )
            else:
                logger.critical(
                    f'Failed to initialize submodules: {e}. \nYou might see {{"detail":"Not Found"}} in your browser. Please check our quick start guide and common issues page from our documentation.\n'
                )


# Handle to the auto-launched GPT-SoVITS process (if we started one).
_sovits_proc: subprocess.Popen | None = None


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """True if something is already listening on host:port."""
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    try:
        with socket.create_connection((probe_host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _warmup_sovits(api_url: str, gpt_cfg: dict) -> None:
    """
    Fire one throwaway synthesis so the user's FIRST real sentence isn't slow.

    The first inference after the model loads (or after a long idle) JIT-compiles
    CUDA kernels and can take 6-15s even for a 5-character sentence — exactly the
    "TTS is slow" symptom seen on the opening line of a reply. Synthesizing a tiny
    phrase here pays that one-time cost up front on a phrase nobody hears.

    Best-effort: any failure is logged and ignored (voice still works, the first
    real sentence is just cold).
    """
    import requests

    ref = gpt_cfg.get("ref_audio_path", "")
    if not ref:
        return  # can't synthesize without a reference clip; skip silently
    lang = gpt_cfg.get("text_lang", "ja")
    params = {
        "text": "ウォームアップ。" if lang == "ja" else "Warming up.",
        "text_lang": lang,
        "ref_audio_path": ref,
        "prompt_lang": gpt_cfg.get("prompt_lang", lang),
        "prompt_text": gpt_cfg.get("prompt_text", ""),
        "media_type": "wav",
        "streaming_mode": "false",
    }
    try:
        started = time.monotonic()
        resp = requests.get(api_url, params=params, timeout=120)
        if resp.status_code == 200:
            logger.info(
                f"🔥 GPT-SoVITS warmed up in {time.monotonic() - started:.1f}s "
                "— the first sentence will be fast now."
            )
        else:
            logger.warning(
                f"GPT-SoVITS warmup returned HTTP {resp.status_code}; skipping."
            )
    except Exception as e:
        logger.warning(f"GPT-SoVITS warmup failed (non-fatal): {e}")


def _watch_sovits_ready(
    host: str,
    port: int,
    api_url: str | None = None,
    gpt_cfg: dict | None = None,
    timeout: int = 180,
) -> None:
    """Poll the TTS port until GPT-SoVITS is serving, then warm it up."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if _sovits_proc is not None and _sovits_proc.poll() is not None:
            logger.error(
                f"GPT-SoVITS exited early (code {_sovits_proc.returncode}). "
                "Run scripts/start-gpt-sovits.cmd manually to see the error."
            )
            return
        if _port_open(host, port):
            logger.info(f"✅ GPT-SoVITS is ready on {host}:{port}.")
            if api_url and gpt_cfg:
                _warmup_sovits(api_url, gpt_cfg)
            return
        time.sleep(1.0)
    logger.warning(
        f"GPT-SoVITS still not responding after {timeout}s — voice may be "
        "unavailable until it finishes loading."
    )


def _stop_gpt_sovits() -> None:
    """Terminate the auto-launched GPT-SoVITS process tree on shutdown."""
    global _sovits_proc
    if _sovits_proc is not None and _sovits_proc.poll() is None:
        logger.info("Stopping GPT-SoVITS...")
        try:
            if os.name == "nt":
                # Kill the whole tree — api_v2 spawns inference workers.
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(_sovits_proc.pid)],
                    capture_output=True,
                )
            else:
                _sovits_proc.terminate()
                _sovits_proc.wait(timeout=10)
        except Exception as e:
            logger.warning(f"Error stopping GPT-SoVITS: {e}")
    _sovits_proc = None


def start_gpt_sovits_if_needed(raw_config: dict) -> None:
    """
    Auto-launch the local GPT-SoVITS TTS server so the user only has to run
    run_server.py. Mirrors scripts/start-gpt-sovits.cmd (conda env on PATH +
    PYTHONPATH). No-op when TTS isn't local gpt_sovits, when something is
    already serving the port, or when the bundled env/repo isn't present.
    """
    global _sovits_proc
    tts = raw_config.get("character_config", {}).get("tts_config", {})
    if tts.get("tts_model") != "gpt_sovits_tts":
        return

    api_url = tts.get("gpt_sovits", {}).get("api_url", "http://127.0.0.1:9880/tts")
    parsed = urlparse(api_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 9880

    if host not in ("127.0.0.1", "localhost", "::1"):
        logger.info(
            f"GPT-SoVITS api_url points to a remote host ({host}); not auto-launching."
        )
        return

    if _port_open(host, port):
        logger.info(
            f"GPT-SoVITS already running on {host}:{port} — reusing it."
        )
        # A reused instance may have gone cold during a long idle; warm it so the
        # first sentence after this restart is fast too.
        threading.Thread(
            target=_warmup_sovits,
            args=(api_url, tts.get("gpt_sovits", {})),
            daemon=True,
        ).start()
        return

    root = Path(__file__).parent
    env_dir = root / "third_party" / "miniconda3" / "envs" / "GPTSoVits"
    sovits_dir = root / "third_party" / "GPT-SoVITS"
    python_exe = env_dir / ("python.exe" if os.name == "nt" else "bin/python")

    if not python_exe.exists() or not sovits_dir.exists():
        logger.warning(
            "Could not find the bundled GPT-SoVITS under third_party/ — "
            "start it manually (scripts/start-gpt-sovits.cmd). Skipping auto-launch."
        )
        return

    # Replicate the .cmd's environment: prepend the conda env dirs to PATH so the
    # native deps resolve, and put the GPT-SoVITS repo on PYTHONPATH.
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        [
            str(env_dir),
            str(env_dir / "Library" / "bin"),
            str(env_dir / "Scripts"),
            env.get("PATH", ""),
        ]
    )
    env["PYTHONPATH"] = str(sovits_dir)

    cmd = [
        str(python_exe),
        "api_v2.py",
        "-a",
        host,
        "-p",
        str(port),
        "-c",
        os.path.join("GPT_SoVITS", "configs", "tts_infer.yaml"),
    ]
    logger.info(
        f"Launching GPT-SoVITS on {host}:{port} (loading models may take ~10-60s)..."
    )
    creationflags = (
        subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    )
    try:
        _sovits_proc = subprocess.Popen(
            cmd, cwd=str(sovits_dir), env=env, creationflags=creationflags
        )
    except Exception as e:
        logger.error(f"Failed to launch GPT-SoVITS: {e}")
        return

    atexit.register(_stop_gpt_sovits)
    threading.Thread(
        target=_watch_sovits_ready,
        args=(host, port, api_url, tts.get("gpt_sovits", {})),
        daemon=True,
    ).start()


def parse_args():
    parser = argparse.ArgumentParser(description="KEHAI Server")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--hf_mirror", action="store_true", help="Use Hugging Face mirror"
    )
    parser.add_argument(
        "--no-sovits",
        action="store_true",
        help="Don't auto-launch the bundled GPT-SoVITS TTS server",
    )
    return parser.parse_args()


@logger.catch
def run(console_log_level: str, auto_sovits: bool = True):
    init_logger(console_log_level)
    logger.info(f"KEHAI, version v{get_version()}")

    # Get selected language
    lang = upgrade_manager.lang

    # Check if the frontend submodule is initialized
    check_frontend_submodule(lang)

    # Sync user config with default config
    try:
        upgrade_manager.sync_user_config()
    except Exception as e:
        logger.error(f"Error syncing user config: {e}")

    atexit.register(WebSocketServer.clean_cache)

    # Load configurations from yaml file
    raw_config = read_yaml("conf.yaml")
    config: Config = validate_config(raw_config)
    server_config = config.system_config

    # Auto-launch the bundled GPT-SoVITS TTS server (unless disabled) so the
    # user only needs to run this one command. It warms up in the background
    # while the rest of the server context loads.
    if auto_sovits:
        start_gpt_sovits_if_needed(raw_config)

    if server_config.enable_proxy:
        logger.info("Proxy mode enabled - /proxy-ws endpoint will be available")

    # Initialize the WebSocket server (synchronous part)
    server = WebSocketServer(config=config)

    # Perform asynchronous initialization (loading context, etc.)
    logger.info("Initializing server context...")
    try:
        asyncio.run(server.initialize())
        logger.info("Server context initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize server context: {e}")
        sys.exit(1)  # Exit if initialization fails

    # Prepare HTTPS (needed for mic/camera/screen access from other PCs on the LAN)
    ssl_kwargs = {}
    scheme = "http"
    if getattr(server_config, "enable_https", False):
        from src.kehai.ssl_utils import ensure_self_signed_cert

        certfile, keyfile = ensure_self_signed_cert(
            server_config.ssl_certfile, server_config.ssl_keyfile
        )
        ssl_kwargs = {"ssl_certfile": certfile, "ssl_keyfile": keyfile}
        scheme = "https"

    # Run the Uvicorn server
    logger.info(
        f"Starting server on {scheme}://{server_config.host}:{server_config.port}"
    )
    uvicorn.run(
        app=server.app,
        host=server_config.host,
        port=server_config.port,
        log_level=console_log_level.lower(),
        # On Ctrl+C uvicorn waits for open connections to close. The client
        # WebSocket (/client-ws) stays in a receive loop until the browser tab
        # disconnects, so without a bound the shutdown hangs forever on
        # "Waiting for connections to close." Force-close after a few seconds.
        timeout_graceful_shutdown=5,
        **ssl_kwargs,
    )


if __name__ == "__main__":
    args = parse_args()
    console_log_level = "DEBUG" if args.verbose else "INFO"
    if args.verbose:
        logger.info("Running in verbose mode")
    else:
        logger.info(
            "Running in standard mode. For detailed debug logs, use: uv run run_server.py --verbose"
        )
    if args.hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    run(console_log_level=console_log_level, auto_sovits=not args.no_sovits)
