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
# 2026-10-23: 0.30 -> 0.40 (~27 degrees). Past ~2.5 m the face is small and the
# estimate flickers: a person talking straight at her from 2.6-3.0 m was
# "facing" in only 2-8% of frames at 0.30. See Attention.MIN_FRAC.
FACING_ASYM = float(os.getenv("FACE_FACING_ASYM", "0.40"))
# For the distance estimate: the camera's horizontal FOV (same assumption as the
# frontend's headCamFovDeg) and the average adult pupil distance.
# 2026-11-09: 60 -> 100. The user standing 1.5 m in front of the PROTO read
# 2.1-4.0 m at 60 and a median of 1.7-2.0 m at 90 (the camera is a wide one);
# every metre below is on the 100 scale (0.48x the old 60 one).
HFOV_DEG = float(os.getenv("FACE_HFOV_DEG", "100"))
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
        # The same picture in grayscale, for active_speaker's lip crops.
        self.last_gray = (bgr[..., 0] * 0.114 + bgr[..., 1] * 0.587 + bgr[..., 2] * 0.299).astype(np.uint8)
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
    # 2026-10-23: 3.0 -> 3.5 — people at the PROTO stood 2.6-3.0 m back and
    # were ignored as "nobody within 3.0 m"; the distance is from an
    # uncalibrated FOV, so the edge needs slack.
    # 2026-11-04: 3.5 -> 4.0. A visitor standing at the PROTO was estimated at
    # 2.8-3.6 m (the FOV is uncalibrated; the far desk reads 5-7 m), and the
    # 3.6 m frames did not count.
    # 2026-11-09: 4.0 -> 2.5 with the FOV fixed (HFOV_DEG 60 -> 100 shrinks every
    # distance to 0.48x: 4.0 on the old scale is 1.9 m).
    MAX_M = float(os.getenv("ADDRESSEE_MAX_M", "2.5"))
    # Fraction of the utterance's frames with someone facing her.
    # (The render tab's barge-in check mirrors these two: head-tracker.ts.)
    # 2026-10-23: 0.3 -> 0.12. One person talking to her at the PROTO from
    # 1.6-2.8 m measured 9-100% over 12 utterances; 0.3 dropped three of them
    # (9%, 16%, 20%) — among them "can you tell me today's weather?" — because at
    # that distance the face is small and the facing call flickers. Someone
    # turned away to talk to another person stays near 0.
    # Later that evening 0.12 -> 0.05 (with FACING_ASYM 0.40, MAX_M 3.5): read-aloud
    # test sentences from 2.6-3.0 m were dropped at 1/45, 4/51, 4/76 frames and
    # only answered when repeated. A few glances during the utterance now do.
    MIN_FRAC = float(os.getenv("ADDRESSEE_MIN_FRAC", "0.05"))
    # People look, then speak: the window opens this long before the speech.
    LEAD_S = 1.0
    ARRIVE_GAP_S = float(os.getenv("VISITOR_GAP_S", "20"))

    def __init__(self, keep_s: float = 60.0):
        self.keep_s = keep_s
        # (monotonic time, someone near is facing, nearest distance or None,
        #  ((distance, facing), ...) of the faces within MAX_M)
        self.frames: deque[tuple] = deque()
        self.xs: deque[tuple] = deque(maxlen=1800)
        self._near_at = 0.0    # when someone was last within NEAR_M
        self.arrived_at = 0.0  # when someone last came within NEAR_M after ARRIVE_GAP_S of nobody

    def record(self, faces: list[dict]) -> bool | None:
        """Log one frame; returns whether someone near was facing her in it, or
        None when nobody near is in view (the render tab's barge-in check then
        leaves the frame out, as judge() does)."""
        now = time.monotonic()
        near = [f for f in faces if f["d"] <= self.MAX_M]
        facing = any(f["facing"] for f in near)
        self.frames.append(
            (now, facing, min((f["d"] for f in near), default=None), tuple((f["d"], bool(f["facing"])) for f in near))
        )
        while self.frames and now - self.frames[0][0] > self.keep_s:
            self.frames.popleft()
        # Where across the picture the nearest face is (0 left .. 1 right), for
        # matching a voice's direction to the person (voice_direction.py).
        if near:
            f0 = min(near, key=lambda f: f["d"])
            self.xs.append((now, f0["box"][0] + f0["box"][2] / 2))
        # Someone walking up after the place in front of her stood empty: a new
        # visitor (websocket_handler starts a new conversation for them).
        if any(f["d"] <= self.NEAR_M for f in near):
            if self._near_at and now - self._near_at >= self.ARRIVE_GAP_S:
                self.arrived_at = now
            self._near_at = now
        return facing if near else None

    def face_x(self, t0: float, t1: float) -> float | None:
        """Median picture position (0..1) of the nearest face over t0..t1."""
        v = sorted(x for t, x in self.xs if t0 <= t <= t1)
        return v[len(v) // 2] if len(v) >= 3 else None

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
        if not near:
            # Nobody in view is NOT "not to her". The camera sees a slice of the
            # space in front of the PROTO: on 2026-10-23 a person 2 m away had
            # their face at the very bottom of the frame (y 0.81-0.93) and was
            # out of it whenever they stepped closer or looked down, and every
            # word was dropped as "nobody within 3.5 m". The voice has already
            # passed the VAD's proximity gate, so answer, as with no camera.
            return None, f"nobody within {self.MAX_M:.1f} m in view — answering"
        # Out of the frames in which someone near WAS visible: a face drifting in
        # and out at the frame's edge should not dilute the ones it faced her in.
        return facing / len(near) >= self.MIN_FRAC, (
            f"facing {facing}/{len(near)} frames with someone near, nearest {min(near):.1f} m"
        )


    # "Only someone who has stopped close in front of her" (2026-11-09, the user's
    # choice for a crowded venue with a TV beside the PROTO, where neither the
    # words, the lips nor the microphone could tell a visitor from the TV): a
    # face within NEAR_M in DWELL_FRAC of the frames from DWELL_S before the speech
    # to its end, facing her in MIN_FRAC of those. Distances come from an
    # uncalibrated FOV; on the PROTO a visitor talking to her read 2.6-3.1 m.
    # 2026-11-09: 3.3 -> 2.0 m, real metres now (the user stood 1.5 m away).
    NEAR_M = float(os.getenv("ADDRESSEE_NEAR_M") or "2.0")
    DWELL_S = float(os.getenv("ADDRESSEE_DWELL_S", "1.0"))
    DWELL_FRAC = float(os.getenv("ADDRESSEE_DWELL_FRAC", "0.6"))
    ENGAGED_FACING = float(os.getenv("ADDRESSEE_ENGAGED_FACING", "0.25"))
    EDGE_FRAC = float(os.getenv("ADDRESSEE_EDGE_FRAC", "0.15"))  # 0 = off

    def engaged(self, t0: float, t1: float) -> tuple[bool | None, str]:
        """Was the speech between t0 and t1 said by someone who had stopped close
        in front of her? None = no camera to tell (stale or not running)."""
        if not self.frames or t1 - self.frames[-1][0] > 1.5:
            return None, "no camera"
        win = [f for f in self.frames if t0 - self.DWELL_S <= f[0] <= t1]
        if len(win) < 5:
            return None, f"only {len(win)} camera frames"
        # One frame's distance swings +-30 % (the eyes are ~15 px apart in the
        # 640-wide picture), so "close" is the nearest face's MEDIAN over the
        # window, and "stopped" is a face being in view in DWELL_FRAC of it.
        # Before (2026-11-09 02:35), each frame had to read close by itself: the
        # user 1.5 m away read 2.1-4.0 m and 4 of their 7 turns were dropped.
        seen_fr = [f for f in win if f[2] is not None]
        med = float(np.median([f[2] for f in seen_fr])) if seen_fr else None
        close = seen_fr if med is not None and med <= self.NEAR_M else []
        seen = (f"nearest median {med:.1f} m, {np.percentile([f[2] for f in seen_fr], 10):.1f}-"
                f"{np.percentile([f[2] for f in seen_fr], 90):.1f}") if seen_fr else "no face"
        if len(close) < self.DWELL_FRAC * len(win):
            # Someone who stepped right up can leave the frame (the camera looks
            # up; at the PROTO a face sits at the bottom edge): close and still
            # there just before, and no face at all since — they are the one.
            before = [f for f in self.frames if t0 - 3.0 <= f[0] < t0]
            last_close = max((f[0] for f in before if any(d <= self.NEAR_M for d, _ in f[3])), default=None)
            if (
                before and last_close is not None
                and sum(1 for f in before if any(d <= self.NEAR_M for d, _ in f[3])) >= self.DWELL_FRAC * len(before) * 0.5
                and all(not f[3] for f in self.frames if f[0] > last_close)
            ):
                return True, f"was close until {t0 - last_close:.1f} s before, then out of view (stepped closer)"
            # Close, but the face is only caught now and then: standing right at
            # the big PROTO a visitor sits on the frame's bottom / side edge
            # (2026-11-13 09:44:25, 1.1 m away, a face in 6 of 34 frames —
            # dropped). Every face seen was close (the median is), enough of
            # them to be no flicker, and facing her in half of those.
            if (
                self.EDGE_FRAC > 0
                and len(close) >= max(3, self.EDGE_FRAC * len(win))
                and sum(1 for f in close if f[1]) >= 0.5 * len(close)
            ):
                return True, (
                    f"someone close, mostly out of the camera's view, facing her "
                    f"(close {len(close)}/{len(win)} frames, {seen})"
                )
            return False, f"nobody stopped within {self.NEAR_M:.1f} m ({len(close)}/{len(win)} frames, {seen})"
        facing = sum(1 for f in close if f[1])
        # Talking to her, they look at her a good part of the time; MIN_FRAC
        # (5 %, from when facing was read at 0.30 on a mis-scaled distance) let
        # someone turned to a friend or to the robot next door through. Today's
        # 48 turns she rightly took had 33-100 % facing, one greeting 20 %.
        ok = facing >= self.ENGAGED_FACING * len(close)
        return ok, (
            f"{'someone stopped close, facing her' if ok else 'someone close but turned away'}"
            f" (close {len(close)}/{len(win)} frames, facing {facing}, {seen})"
        )

    def left_between(self, t0: float, t1: float, gap_s: float = 3.0) -> bool:
        """Was nobody within NEAR_M for gap_s or longer somewhere in t0..t1 (the
        visitor walked off)? False without camera frames to tell."""
        last_near = t0
        seen = False
        for f in self.frames:
            if f[0] < t0 or f[0] > t1:
                continue
            seen = True
            if any(d <= self.NEAR_M for d, _ in f[3]):
                last_near = f[0]
            elif f[0] - last_near >= gap_s:
                return True
        return seen and t1 - last_near >= gap_s and bool(self.frames) and self.frames[-1][0] >= t1 - 1.5

    def empty_for(self, s: float = 3.0) -> bool:
        """The camera is running and nobody has been within MAX_M for s seconds."""
        now = time.monotonic()
        if not self.frames or now - self.frames[-1][0] > 1.5 or now - self.frames[0][0] < s:
            return False
        for f in reversed(self.frames):
            if now - f[0] > s:
                break
            if f[2] is not None:
                return False
        return True

    def presence(self, t0: float, t1: float) -> tuple[str | None, str]:
        """Who was in front while t0..t1 was said: "facing" (someone near,
        facing her), "away" (someone near, not facing her), "nobody" (no one
        within MAX_M in view), or None with no camera to tell."""
        v, why = self.judge(t0, t1)
        if v is True:
            return "facing", why
        if v is False:
            return "away", why
        return ("nobody" if why.startswith("nobody") else None), why


ATTENTION = Attention()


# Which PROTO is on the other end decides how near "stopped in front" is. The small
# Proto M2 (camera "ProtoCam1") has people at 0.4-0.5 m and answered the user from
# 1.2 m ("reacts from quite far away", 2026-11-09) -> 0.8 m; the big PROTO has them
# at 1.0-1.6 m, and a fixed 0.8 there ignored every "Hi, how are you?" (2026-11-12
# 23:20). The two are swapped in and out, so the viewer's camera picks it; an
# explicit ADDRESSEE_NEAR_M still wins.
_NEAR_SMALL_M = float(os.getenv("ADDRESSEE_NEAR_M_SMALL", "0.8"))
_NEAR_BIG_M = float(os.getenv("ADDRESSEE_NEAR_M_BIG", "2.0"))


def set_viewer_camera(label: str | None) -> None:
    if os.getenv("ADDRESSEE_NEAR_M"):
        return
    small = bool(label) and "protocam" in label.lower()
    near = _NEAR_SMALL_M if small else _NEAR_BIG_M
    if near != Attention.NEAR_M:
        logger.info(
            f"[addressee] camera {label!r}: {'small Proto M2' if small else 'big PROTO'} — "
            f"'stopped in front' is within {near:.1f} m (was {Attention.NEAR_M:.1f})"
        )
        Attention.NEAR_M = near


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
