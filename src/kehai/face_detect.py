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
"""

from pathlib import Path

import av
import numpy as np
from loguru import logger

MODEL = Path(__file__).with_name("face_detection_yunet_2023mar.onnx")
SIZE = 640  # the model's fixed input: 640x640
STRIDES = (8, 16, 32)


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
                    )
                )

        # One face is found at several grid cells and strides: keep the best.
        cands.sort(key=lambda f: -f[0])
        kept = []
        for f in cands:
            if all(_iou(f[1], g[1]) < 0.3 for g in kept):
                kept.append(f)

        return {
            "w": w,
            "h": h,
            "faces": [
                {
                    "s": round(score, 3),
                    "box": [box[0] / sw, box[1] / sh, box[2] / sw, box[3] / sh],
                    "eyes": [[x / sw, y / sh] for x, y in eyes],
                }
                for score, box, eyes in kept
            ],
        }


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


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
