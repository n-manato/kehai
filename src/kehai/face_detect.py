"""Find the faces in the PROTO's camera frames, on the server.

The render tab's head tracker needs to know where the people in front of the
PROTO are, and which of them is closest. Doing that in the render tab failed
twice over:

- MediaPipe's detectors shrink the frame to 128x128, so in the PROTO's wide
  640x360 view a visitor's face is a few pixels across. Measured on frames of
  the real hall: nobody past ~1.3 m was found at all.
- Tiling the frame fixes the range but costs ~50 ms of detection per camera
  frame, and the render tab is busy drawing 4K at 60 fps. In a worker it was
  worse (~800 ms): every GL sync waits behind the render.

YuNet (OpenCV zoo, MIT) takes the frame at its own size, finds the same faces
out to 4 m at score ~0.93 in ~10 ms of CPU here, and its eye keypoints scale
as 1/distance almost exactly — which is what deciding "closest" rests on. The
camera JPEGs already pass through this server (render_stream.py), so only the
face list goes on to the render tab.

Model: face_detection_yunet_2023mar.onnx, vendored beside this file (230 KB).

It also keeps `ATTENTION`: frame by frame, whether someone near the PROTO is
facing it. The voice turn reads that back for the span of the utterance to tell
speech meant for her from people talking among themselves
(websocket_handler._finalize_vad_turn) — a lookup into frames already seen, so it
costs the reply nothing.
"""

import math
import os
import time
from collections import deque
from pathlib import Path

import av
import numpy as np
from loguru import logger

MODEL = Path(__file__).with_name("face_detection_yunet_2023mar.onnx")
SIZE = 640  # the model's fixed input: 640x640
STRIDES = (8, 16, 32)

# Facing = the nose sits near the midline between the eyes. Measured as
# (nose-to-left-eye - nose-to-right-eye) / their sum on photos of known pose:
# frontal 0.03-0.17, ~25-32 degrees turned 0.36-0.47. 0.30 is ~20 degrees.
# (The eyes themselves are a few pixels at this range; head pose is what a
# 640x360 frame can actually see, and people turn their head to whoever they
# are talking to.)
FACING_ASYM = float(os.getenv("FACE_FACING_ASYM", "0.30"))
# For the distance estimate: the camera's horizontal FOV (same assumption as the
# frontend's headCamFovDeg) and the average adult pupil distance.
HFOV_DEG = float(os.getenv("FACE_HFOV_DEG", "60"))
IPD_M = 0.063
# Addressee check on (websocket_handler gates turns, the render tab gates
# barge-in on the per-frame `attn` it is sent). ADDRESSEE=0 turns both off.
ADDRESSEE = os.getenv("ADDRESSEE", "1") == "1"


class FaceDetector:
    def __init__(self, min_score: float = 0.7):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        # Two threads: ~10 ms a frame at 15 fps, without taking cores from the
        # encoder and the rest of the box.
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(MODEL), opts, providers=["CPUExecutionProvider"]
        )
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.decoder = av.CodecContext.create("mjpeg", "r")
        self.min_score = min_score
        self.input = np.zeros((1, 3, SIZE, SIZE), np.float32)

    def detect(self, jpeg: bytes) -> dict:
        """JPEG in; frame size and faces out, normalised to the frame.

        Each face: s = score, box = [x, y, w, h], eyes = [[x, y], [x, y]] (eye
        centres). Not thread-safe; the caller keeps one detection in flight.
        """
        frame = self.decoder.decode(av.Packet(jpeg))[0]
        w, h = frame.width, frame.height
        # Fit the longer side to the model's input; the rest stays zero padding.
        k = min(SIZE / w, SIZE / h, 1.0)
        sw, sh = max(1, round(w * k)), max(1, round(h * k))
        bgr = frame.to_ndarray(width=sw, height=sh, format="bgr24")
        self.input.fill(0)
        self.input[0, :, :sh, :sw] = bgr.transpose(2, 0, 1)
        out = dict(zip(self.outputs, self.session.run(None, {"input": self.input})))

        cands = []
        for s in STRIDES:
            n = SIZE // s
            cls = np.clip(out[f"cls_{s}"][0, :, 0], 0, 1)
            obj = np.clip(out[f"obj_{s}"][0, :, 0], 0, 1)
            score = np.sqrt(cls * obj)
            for i in np.nonzero(score >= self.min_score)[0]:
                r, c = divmod(int(i), n)
                b = out[f"bbox_{s}"][0, i]
                kp = out[f"kps_{s}"][0, i]
                cx, cy = (c + b[0]) * s, (r + b[1]) * s
                bw, bh = float(np.exp(b[2]) * s), float(np.exp(b[3]) * s)
                cands.append(
                    (
                        float(score[i]),
                        [float(cx - bw / 2), float(cy - bh / 2), bw, bh],
                        [
                            [float((c + kp[0]) * s), float((r + kp[1]) * s)],
                            [float((c + kp[2]) * s), float((r + kp[3]) * s)],
                        ],
                        float((c + kp[4]) * s),  # nose x
                    )
                )

        # One face is found at several grid cells and strides: keep the best.
        cands.sort(key=lambda f: -f[0])
        kept = []
        for f in cands:
            if all(_iou(f[1], g[1]) < 0.3 for g in kept):
                kept.append(f)

        focal = sw / (2 * math.tan(math.radians(HFOV_DEG) / 2))
        faces = []
        for score, box, eyes, nose_x in kept:
            (ax, ay), (bx, by) = sorted(eyes)  # left, right in the image
            eye_px = max(math.hypot(bx - ax, by - ay), 1.0)
            # Signed: past either eye (|yaw| > 1) is a profile.
            yaw = ((nose_x - ax) - (bx - nose_x)) / max(bx - ax, 1.0)
            faces.append(
                {
                    "s": round(score, 3),
                    "box": [box[0] / sw, box[1] / sh, box[2] / sw, box[3] / sh],
                    "eyes": [[x / sw, y / sh] for x, y in eyes],
                    "yaw": round(yaw, 2),
                    "facing": abs(yaw) < FACING_ASYM,
                    "d": round(IPD_M * focal / eye_px, 2),
                }
            )
        return {"w": w, "h": h, "faces": faces}


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


class Attention:
    """Whether someone near the PROTO is facing it, per camera frame.

    One camera, one log: the PROTO install has a single viewer camera, so this
    is process-wide rather than per client.
    """

    # Only people this close count — further away they are passing, not talking.
    MAX_M = float(os.getenv("ADDRESSEE_MAX_M", "3.0"))
    # Fraction of the utterance's frames with someone facing her.
    # (The render tab's barge-in check mirrors these two: head-tracker.ts.)
    MIN_FRAC = float(os.getenv("ADDRESSEE_MIN_FRAC", "0.3"))
    # People look, then speak: the window opens this long before the speech.
    LEAD_S = 1.0

    def __init__(self, keep_s: float = 60.0):
        self.keep_s = keep_s
        # (monotonic time, someone near is facing, nearest distance or None)
        self.frames: deque[tuple[float, bool, float | None]] = deque()

    def record(self, faces: list[dict]) -> bool:
        """Log one frame; returns whether someone near was facing her in it."""
        now = time.monotonic()
        near = [f for f in faces if f["d"] <= self.MAX_M]
        facing = any(f["facing"] for f in near)
        self.frames.append((now, facing, min((f["d"] for f in near), default=None)))
        while self.frames and now - self.frames[0][0] > self.keep_s:
            self.frames.popleft()
        return facing

    def judge(self, t0: float, t1: float) -> tuple[bool | None, str]:
        """Was the speech between t0 and t1 (monotonic) addressed to her?

        None = no camera to judge by (not running, or stale): the caller
        answers as it always did. Fails OPEN on purpose — a reply nobody asked
        for is a small embarrassment, ignoring someone who did is not.
        """
        if not self.frames or t1 - self.frames[-1][0] > 1.5:
            return None, "no camera"
        win = [f for f in self.frames if t0 - self.LEAD_S <= f[0] <= t1]
        if len(win) < 3:
            return None, f"only {len(win)} camera frames"
        facing = sum(1 for f in win if f[1])
        near = [f[2] for f in win if f[2] is not None]
        detail = f"facing {facing}/{len(win)} frames, " + (
            f"nearest {min(near):.1f} m"
            if near
            else f"nobody within {self.MAX_M:.1f} m"
        )
        return facing / len(win) >= self.MIN_FRAC, detail


ATTENTION = Attention()


def load() -> FaceDetector | None:
    """The detector, or None (logged) if it cannot run — the caller then falls
    back to sending the frames on for the render tab to detect in itself."""
    try:
        return FaceDetector()
    except Exception as err:
        logger.warning(
            f"[faces] server-side face detection unavailable ({err}); render tab will detect"
        )
        return None
