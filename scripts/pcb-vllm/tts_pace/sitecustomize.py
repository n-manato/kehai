"""Pace the Qwen3-TTS talker so the 60 fps render can use GPU0 (KEHAI, 2026-11-05).

Loaded only by serve-qwen3-tts.sh (PYTHONPATH). The talker (vLLM-Omni stage 0) runs its
decode steps as CUDA-graph replays back to back; while it does, the render on the same
GPU got 1-5 fps (Chrome's compositor waited ~1 s for its frame). Running the talker
eagerly kept the render at 51-60 fps but left the TTS at ~1.4x real time — too thin a
margin (the viewer ran dry at sentence starts). Here the talker keeps its graphs and
sleeps KEHAI_TTS_STEP_SLEEP_MS after each executed step, which hands the GPU back to
the render between steps. 0 / unset = no change. Patches vLLM's EngineCoreProc after it
is imported (an import hook), so nothing is touched in processes that never load it.
"""
import os
import sys

_SLEEP = float(os.environ.get("KEHAI_TTS_STEP_SLEEP_MS", "0") or 0) / 1000.0

if _SLEEP > 0:
    import importlib.abc
    import importlib.machinery
    import time

    def _patch(core):
        import multiprocessing

        orig = core.EngineCoreProc._process_engine_step

        def _paced(self):
            executed = orig(self)
            if executed and "stage0" in multiprocessing.current_process().name:
                time.sleep(_SLEEP)
            return executed

        core.EngineCoreProc._process_engine_step = _paced

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "vllm.v1.engine.core":
                return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return spec
            run = spec.loader.exec_module

            def exec_module(module):
                run(module)
                _patch(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Hook())
