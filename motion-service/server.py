"""FastAPI(+WS) front door for the HY-Motion live motion service (PC-B).

Run:  CUDA_VISIBLE_DEVICES=0,1 python server.py --host 0.0.0.0 --port 8800

The HTTP/WS contract is wired and the server STARTS even before the model is
implemented (endpoints return 503 until `runner.load()` is done), so PC-A
integration and the API shape can be tested independently of the GPU work.
"""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import prompts
from hy_motion_runner import HYMotionRunner
from pool import ClipPool

STATE: dict = {"runners": [], "pool": None}


class MotionRequest(BaseModel):
    emotion: str = "neutral"
    prompt: str | None = None
    duration: float | None = None
    seed: int | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    # One resident runner per visible GPU.
    runners = [HYMotionRunner(device=f"cuda:{i}") for i in range(config.NUM_WORKERS)]
    for r in runners:
        try:
            r.load()  # TODO(PC-B): real load; raises NotImplementedError in scaffold
        except NotImplementedError:
            print("[server] runner not implemented yet — serving 503 until wired")
    STATE["runners"] = runners
    STATE["pool"] = ClipPool(runners)
    if any(r.ready for r in runners):
        STATE["pool"].start_background_refill()
    yield
    if STATE["pool"]:
        STATE["pool"].stop()


app = FastAPI(title="HY-Motion live motion service", lifespan=lifespan)

# The browser (PC-A frontend) fetches /pool/{emotion}/next cross-origin, and
# THREE.MMDLoader uses XHR which enforces CORS — allow it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _vmd_response(path) -> Response:
    data = path.read_bytes()
    return Response(content=data, media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{path.name}"'})


@app.get("/healthz")
def healthz():
    pool = STATE["pool"]
    return {
        "status": "ok",
        "runner_ready": any(r.ready for r in STATE["runners"]),
        "workers": config.NUM_WORKERS,
        "pool": pool.depth() if pool else {},
    }


@app.get("/pool/{emotion}/next")
def pool_next(emotion: str):
    """Live main path: instantly return a ready clip for `emotion`."""
    if emotion not in prompts.POOL_EMOTIONS:
        raise HTTPException(404, f"unknown emotion '{emotion}'")
    pool = STATE["pool"]
    clip = pool.next_clip(emotion) if pool else None
    if clip is None:
        # Pool empty (warming up or model not wired). PC-A falls back to idle.
        raise HTTPException(503, "no ready clip; keep current/idle")
    return _vmd_response(clip)


@app.post("/motion")
def post_motion(req: MotionRequest):
    """On-demand: return a matching clip now, else accept and generate async.

    TODO(PC-B): when wired, enqueue a high-priority generation, return
    202 {job_id}, and push completion over /events.
    """
    pool = STATE["pool"]
    clip = pool.next_clip(req.emotion) if pool else None
    if clip is not None:
        return _vmd_response(clip)
    raise HTTPException(503, "no ready clip; async generation not wired yet")


@app.websocket("/events")
async def events(ws: WebSocket):
    """Push job completions: {type:'job_done', job_id, url}. TODO(PC-B): wire."""
    await ws.accept()
    try:
        while True:
            await ws.receive_text()  # keepalive; real impl pushes job events
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=config.HOST)
    ap.add_argument("--port", type=int, default=config.PORT)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
