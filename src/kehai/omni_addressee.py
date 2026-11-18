"""Is the person speaking talking to HER? Judged by Qwen3-Omni from the PROTO's
camera frame (omni mode) — ahead of the YuNet head-orientation rule in
face_detect.Attention, which stays as the fallback.

Cost, measured 2026-11-03 on the running Thinker (:8808, INT4, GPU1): a 640x360
frame is ~245 tokens; with the short system prompt below in the prefix cache a
verdict takes 70-125 ms, and one fired in the middle of a reply slows that
reply by ~3% (113 -> 110 tok/s). So the Thinker is asked at the moments that
matter — the speech onset and each segment close — and never on the reply path:
the endpoint reads the verdicts already in and only waits, bounded, for one
still in flight.

STRICT by design (the user's spec, 2026-11-03: "answer when someone is talking
to the PROTO"): with a FRESH camera frame, a score under OMNI_ADDRESSEE_MIN is
"not to her" — nobody in view included. The empty room scores 0. With no
camera, a frozen feed (render_stream flags it) or the server down, judge()
answers None and the caller falls back to the YuNet rule / answering as before,
so a viewer without a camera (a laptop browser) is never muted by this.

Env (start_server.sh sets the omni-mode defaults):
  OMNI_ADDRESSEE=1        on/off
  OMNI_ADDRESSEE_MIN=5    score 0-9 at or above which the speech is "to her"
  OMNI_ADDRESSEE_DUMP=dir save each judged frame as <time>_s<score>_<why>.jpg
                          (newest 300 kept) so misjudgements can be reviewed;
                          "" = off
  OMNI_ADDRESSEE_URL      the Thinker's chat/completions (default :8808)
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import time
from collections import deque
from pathlib import Path

import httpx
from loguru import logger

ENABLED = os.getenv("OMNI_ADDRESSEE", "0") == "1"
MIN_SCORE = int(os.getenv("OMNI_ADDRESSEE_MIN", "") or 5)
URL = os.getenv("OMNI_ADDRESSEE_URL", "http://127.0.0.1:8808/v1/chat/completions")
MODEL = os.getenv("OMNI_ADDRESSEE_MODEL", "qwen3-omni")
DUMP_DIR = os.getenv("OMNI_ADDRESSEE_DUMP", "")
DUMP_KEEP = 300
# A frame older than this is not "what the camera sees now".
FRESH_S = 1.5
# People look, then speak: a look this long before the speech counts for it.
LEAD_S = 1.0
KEEP_S = 60.0

# Kept short and fixed so it lives in the Thinker's prefix cache; every look
# then pays only for the frame itself. Checked 2026-11-03: empty room 0, a
# person facing the camera 9, two people turned to each other 0, an anime still
# 0 (a plain "how sure" wording without the explicit rules gave the pair 7).
_SYSTEM = (
    "You are the eyes of a robot avatar on a public display; the camera is on the "
    "robot, so a person looking into the camera is looking at the robot. Someone is "
    "speaking right now. Rate how sure you are that the speaker is talking TO the "
    "robot. 9 = a person faces the camera and addresses it (even if only partly in "
    "view at the bottom or edge). 0 = nobody in view; or two or more people are "
    "turned toward EACH OTHER (talking among themselves); or the person looks at a "
    "phone, a screen, or elsewhere, or is walking past. Use 1-8 only when the "
    "picture is genuinely ambiguous. Answer with one digit 0-9 only."
)


class Judge:
    """Process-wide, like face_detect.ATTENTION: one PROTO, one camera."""

    def __init__(self) -> None:
        # (monotonic time the verdict came in, score 0-9, why it was asked)
        self.verdicts: deque[tuple[float, int, str]] = deque()
        self.inflight: asyncio.Task | None = None
        self._client: httpx.AsyncClient | None = None
        self.calls = 0
        self.fails = 0
        self.ms = 0.0
        self._dumped = 0

    # --- the frame -----------------------------------------------------
    @staticmethod
    def fresh_frame() -> bytes:
        """The viewer camera's latest frame, or b"" when there is none worth
        judging by (no camera, stale, or the feed is frozen on one picture)."""
        from .render_stream import hub  # lazy: render_stream imports face_detect, not us

        if not hub.cam_last or hub.cam_frozen_logged:
            return b""
        if time.time() - hub.cam_at > FRESH_S:
            return b""
        return hub.cam_last

    # --- asking ----------------------------------------------------------
    def check(self, why: str) -> asyncio.Task | None:
        """Ask for a verdict on the current frame. One look at a time: while one
        is in flight its task is returned instead. None = nothing to judge by."""
        if not ENABLED:
            return None
        if self.inflight is not None and not self.inflight.done():
            return self.inflight
        frame = self.fresh_frame()
        if not frame:
            return None
        self.inflight = asyncio.create_task(self._ask(frame, why))
        return self.inflight

    async def _ask(self, frame: bytes, why: str) -> int | None:
        t0 = time.monotonic()
        text = ""
        try:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=httpx.Timeout(3.0, connect=0.5))
            body = {
                "model": MODEL,
                "max_tokens": 1,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64,"
                                    + base64.b64encode(frame).decode()
                                },
                            },
                            {"type": "text", "text": "Score 0-9:"},
                        ],
                    },
                ],
            }
            r = await self._client.post(URL, json=body)
            r.raise_for_status()
            text = (r.json()["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:  # noqa: BLE001 — a failed look is "no verdict"
            self.fails += 1
            logger.warning(f"[addressee] omni look failed ({why}): {e!r}")
            return None
        ms = (time.monotonic() - t0) * 1000
        self.calls += 1
        self.ms += ms
        m = re.search(r"\d", text)
        if not m:
            logger.warning(f"[addressee] omni answered {text!r}, not a score ({why})")
            return None
        score = int(m.group())
        now = time.monotonic()
        self.verdicts.append((now, score, why))
        while self.verdicts and now - self.verdicts[0][0] > KEEP_S:
            self.verdicts.popleft()
        logger.info(f"[addressee] omni {score}/9 in {ms:.0f} ms ({why})")
        if DUMP_DIR:
            self._dump(frame, score, why)
        return score

    def _dump(self, frame: bytes, score: int, why: str) -> None:
        try:
            d = Path(DUMP_DIR)
            d.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S") + f".{int((time.time() % 1) * 1000):03d}"
            (d / f"{stamp}_s{score}_{why}.jpg").write_bytes(frame)
            self._dumped += 1
            if self._dumped % 20 == 1:
                files = sorted(d.glob("*.jpg"))
                for old in files[: max(0, len(files) - DUMP_KEEP)]:
                    old.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001 — review material only
            logger.debug(f"[addressee] dump failed: {e!r}")

    # --- judging ---------------------------------------------------------
    def judge(self, t0: float, t1: float) -> tuple[bool | None, str]:
        """Was the speech between t0 and t1 (monotonic) addressed to her, from
        the looks already in? None = no look to judge by (caller falls back)."""
        if not ENABLED:
            return None, "omni addressee off"
        win = [v for v in self.verdicts if t0 - LEAD_S <= v[0] <= t1]
        if not win:
            return None, "no omni look yet"
        best = max(v[1] for v in win)
        return best >= MIN_SCORE, f"omni score {best}/9 over {len(win)} look(s)"

    async def settle(self, t0: float, timeout: float = 0.3) -> tuple[bool | None, str]:
        """judge() for the utterance that began at t0, waiting — bounded — for a
        look still in flight when none has landed since t0. The lead-in
        verdicts alone would answer for the PREVIOUS speaker: a visitor's aside
        scored 0 half a second ago, then the user turned to her — her onset read
        that stale 0, skipped the barge-in and talked over them. With no look in
        flight and a fresh frame, one is taken now (a turn with no segment
        close). On timeout the lead-in verdict, if any, stands."""
        v = self.judge(t0, time.monotonic())
        if not ENABLED:
            return v
        since_t0 = any(t >= t0 for t, _, _ in self.verdicts)
        task = self.inflight if (self.inflight is not None and not self.inflight.done()) else None
        if v[0] is not None and (since_t0 or task is None):
            return v
        if task is None:
            task = self.check("endpoint")
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout)
            except asyncio.TimeoutError:
                if v[0] is not None:
                    return v
                return None, f"omni look not back within {timeout * 1000:.0f} ms"
            except Exception:  # noqa: BLE001 — _ask logs it
                pass
        return self.judge(t0, time.monotonic())

    def stats(self) -> str:
        if not self.calls:
            return "omni looks: none"
        return f"omni looks: {self.calls} ({self.ms / self.calls:.0f} ms avg, {self.fails} failed)"


JUDGE = Judge()


# --- the EARS (2026-11-04) ---------------------------------------------------
# Whether a stretch of speech is said TO her, judged from the speech itself.
#
# Why: the camera look above cannot tell. On 2026-11-04 afternoon the same
# scene — two people at the far desk, backs to the camera, talking to each other
# — scored 0 and 9 on consecutive frames, and every chat in the room (a
# presentation, "thank you"s, phone talk) was answered, mostly with "I am not
# sure what you mean". YuNet fails open when nobody near is in view, and the
# viewer's AGC lifts distant voices over the 68 dB onset gate, so loudness says
# nothing about distance either.
#
# Measured on that afternoon's 60 room recordings (all chatter or noise) and 30
# questions to her rendered like the PROTO mic (3.9 kHz band limit, room reverb,
# the room's noise at 10-18 dB SNR, the same level): chatter 0/9 on all 60,
# questions 9/9 on 28 of 30 (misses: a far "Who made you?", a near "Thank you,
# Iris."). 45-52 ms a call. Giving it the robot's last reply as text made it
# WORSE (15 of 60 chats then scored 9), so it hears the audio alone.
AUDIO_ENABLED = os.getenv("OMNI_ADDRESSEE_AUDIO", "1" if ENABLED else "0") == "1"
AUDIO_MIN = int(os.getenv("OMNI_ADDRESSEE_AUDIO_MIN", "") or 5)

_AUDIO_SYSTEM = (
    "You are the ears of a robot receptionist standing in a busy room. You hear one "
    "stretch of speech picked up by its microphone. Rate how sure you are that this "
    "speech is said TO the robot: a question, a request, a greeting, or a reply meant "
    "for it. 9 = clearly addressed to the robot. 0 = people talking to each other, a "
    "presentation, a phone call, background chatter, thanks or goodbyes between other "
    "people, a fragment of a longer conversation, or noise with no clear words. Answer "
    "with one digit 0-9 only."
)


class AudioJudge:
    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self.calls = 0
        self.fails = 0

    async def score(self, wav_b64: str, why: str = "") -> int | None:
        """0-9 for a 16 kHz WAV (base64), or None when the Thinker cannot say."""
        if not (AUDIO_ENABLED and wav_b64):
            return None
        t0 = time.monotonic()
        try:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=httpx.Timeout(2.0, connect=0.5))
            body = {
                "model": MODEL,
                "max_tokens": 1,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": _AUDIO_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + wav_b64}},
                            {"type": "text", "text": "Score 0-9:"},
                        ],
                    },
                ],
            }
            r = await self._client.post(URL, json=body)
            r.raise_for_status()
            text = (r.json()["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:  # noqa: BLE001 — no verdict; the caller falls back
            self.fails += 1
            logger.warning(f"[addressee] ears failed ({why}): {e!r}")
            return None
        m = re.search(r"\d", text)
        if not m:
            logger.warning(f"[addressee] ears answered {text!r}, not a score ({why})")
            return None
        self.calls += 1
        s = int(m.group())
        # base64 of 16 kHz int16 mono: 4/3 chars per byte, 32 kB a second
        secs = max(0.0, (len(wav_b64) * 3 / 4 - 44) / 32000)
        logger.info(f"[addressee] ears {s}/9 for {secs:.1f}s of speech in {(time.monotonic() - t0) * 1000:.0f} ms ({why})")
        return s


AUDIO = AudioJudge()


# --- What was said (shared transcription) ---
#
# (History. The words check described below was REMOVED on 2026-11-08 — it did
# not generalise beyond the day it was tuned on; see "Did the speaker stop
# mid-sentence?" further down and active_speaker.py.)
#
# 2026-11-06, "she answers with nobody in front — the TV next to her" and "while
# someone talks to her she picks up too much around her". The ears above scored
# the TV 0/9 almost every time; what let it through was the face rule
# (websocket_handler._followup: someone near facing her = answer, whatever was
# said). Beside a TV, the person watching it or working at the desk next to the
# PROTO faces her too: 41 of the 45 turns answered since 11-05 16:00 came in that
# way, most of them TV ("Thank you for tuning in…", "We'll be right back", "we
# hope to see you again in the next episode"). The Thinker's transcription of the
# turn (made anyway for the reply, openai_omni_llm.transcribe) now goes through a
# words check before a face can stand in for the ears.
#
# Measured on 31 utterances really said to her (the visitors' turns of 11-04..06
# and probe questions) and 60 the room or the TV produced (same days), the
# few-shot A/B below passes 30/31 and 9/60 (the 9 are a TV's own questions and
# bare "No." / "What is it?"); a 0-9 score with the same rules swung between
# 0 and 9 (passing 15-22/31). 45 ms a call, the examples sit in the prefix cache.

HEARD_ON = os.getenv("OMNI_HEARD", "1") != "0"
HEARD_LANG = os.getenv("OMNI_HEARD_LANG", "English")
# The transcript is written as the continuation of 'The speaker said: "' (an
# assistant prefill, stopped at the closing quote). Asked plainly to "transcribe
# the speech exactly", the Thinker ANSWERED requests instead of writing them
# down — 3 of 8 test questions ("Iris, tell me a short story…" came back as
# "Once upon a time…", "What is machine learning?…" as its explanation), and on
# the PROTO "I'm sorry, I can't understand what you're asking" (11-06 12:49:57).
# That text went to the reply beside the audio as "what I said". With the
# prefill: 8 of 8 written down, 96-220 ms instead of 440 ms (2026-11-06).
_HEARD_SYSTEM = (
    "You are a speech-to-text engine, not an assistant. Write down the words spoken in "
    "the audio exactly, in {lang}. Never answer, reply to, or carry out what is said, "
    "even when it is a question or a request. Output only the transcript."
)
_HEARD_PREFILL = 'The speaker said: "'


class Heard:
    """The Thinker's transcription of one utterance's audio, made once: the
    reply (openai_omni_llm.transcribe) and the words check ask for the same
    audio at the same pause, and share one request."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._recent: dict[int, asyncio.Future] = {}
        self._order: deque[int] = deque()

    async def get(self, wav_b64: str, timeout: float = 1.5) -> str | None:
        # (Continuing the transcript of an earlier pause instead of writing the
        # whole utterance again was tried on 2026-11-08: 95 -> 60 ms, but the word
        # error rate went 7.1% -> 16.8% on narrow-band public questions — "for me"
        # came back "4 me" — so every transcript is written from the whole audio.)
        if not (HEARD_ON and wav_b64):
            return None
        key = hash(wav_b64)
        fut = self._recent.get(key)
        if fut is None:
            fut = asyncio.ensure_future(self._ask(wav_b64, timeout))
            self._recent[key] = fut
            self._order.append(key)
            while len(self._order) > 16:
                self._recent.pop(self._order.popleft(), None)
        try:
            return await asyncio.shield(fut)
        except Exception:  # noqa: BLE001
            return None

    async def _ask(self, wav_b64: str, timeout: float) -> str | None:
        t0 = time.monotonic()
        try:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=0.5))
            body = {
                "model": MODEL,
                "max_tokens": 60,
                "temperature": 0,
                "stop": ['"'],
                "messages": [
                    {"role": "system", "content": _HEARD_SYSTEM.format(lang=HEARD_LANG)},
                    {"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + wav_b64}}]},
                    {"role": "assistant", "content": _HEARD_PREFILL},
                ],
                "continue_final_message": True,
                "add_generation_prompt": False,
            }
            r = await asyncio.wait_for(self._client.post(URL, json=body), timeout)
            r.raise_for_status()
            text = (r.json()["choices"][0]["message"]["content"] or "").strip().strip('"').strip()
            if text.startswith(_HEARD_PREFILL):  # a server that echoes the prefill
                text = text[len(_HEARD_PREFILL):].strip().strip('"').strip()
            if _degenerate(text):
                # 2026-11-09 03:02: on a murmur ("Mm-hmm") the Thinker looped the
                # prefill — "The speaker said: 'The speaker said: '..." — and that
                # went into memory and the unfinished check. Whisper's text stands.
                logger.info(f"[omni] heard (discarded, looping) in {(time.monotonic() - t0) * 1000:.0f} ms: {text[:60]!r}")
                return None

        except Exception as e:  # noqa: BLE001 — the reply then hears the audio alone
            logger.warning(f"[omni] heard: transcription failed ({e!r})")
            return None
        logger.info(f"[omni] heard in {(time.monotonic() - t0) * 1000:.0f} ms: {text[:80]!r}")
        return text or None


def _degenerate(text: str) -> bool:
    """A transcript that is the prompt echoed or one phrase looping."""
    if "speaker said" in text.lower():
        return True
    w = re.findall(r"[\w']+", text.lower())
    grams = [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
    return bool(grams) and max(grams.count(g) for g in set(grams)) >= 3 and len(set(grams)) < 0.5 * len(grams)


HEARD = Heard()


# When the reply waits for the transcript. Writing it out costs 90-240 ms before
# the reply can start (2026-11-08, 3 probe questions: first audio 0.47 s without
# it, 0.64 s with it). It is there because the PROTO's microphone delivers
# telephone-band audio (nothing above 4 kHz — consonants gone), on which the
# Thinker misheard questions it then answered (2026-11-04: written down first, 8
# of 9 replies matched the question). On full-band audio it heard them directly.
# So the reply waits for it only when the utterance's audio is band-limited, by
# measurement, not by who speaks or what is said: the cliff a 4 kHz cut leaves —
# mean energy at 4.1-4.6 kHz against 3.4-3.9 kHz. Real PROTO recordings -29/-30 dB,
# a 3.9 kHz room simulation -17..-23 dB, full-band speech (synthetic voices, a
# YouTube clip) -7.5..+8.5 dB. (The plain share of energy above 4 kHz overlapped:
# a full-band sentence with few fricatives measured 7e-4.) A fixed microphone
# thus makes her ~0.17 s quicker by itself. OMNI_HEARD_CLIFF_DB=off = always wait.
_CLIFF_ENV = os.getenv("OMNI_HEARD_CLIFF_DB", "-12")
HEARD_CLIFF_DB = None if _CLIFF_ENV.strip().lower() in ("off", "") else float(_CLIFF_ENV)


def band_cliff_db(wav_b64: str) -> float | None:
    """dB of 4.1-4.6 kHz against 3.4-3.9 kHz in a 16 kHz WAV (base64): about 0
    for full-band speech, -20 dB or lower for audio cut at 4 kHz."""
    try:
        import numpy as np

        x = np.frombuffer(base64.b64decode(wav_b64)[44:], dtype=np.int16).astype(np.float32)
        x = x[-16000 * 4:]
        if len(x) < 1600:
            return None
        X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        f = np.fft.rfftfreq(len(x), 1 / 16000)
        below = X[(f > 3400) & (f < 3900)].mean()
        above = X[(f > 4100) & (f < 4600)].mean()
        return float(10 * np.log10(above / below)) if below > 0 and above > 0 else None
    except Exception:  # noqa: BLE001
        return None


def reply_needs_heard(wav_b64: str) -> bool:
    """Should the reply wait for the Thinker's transcript of this audio?"""
    if HEARD_CLIFF_DB is None:
        return True
    c = band_cliff_db(wav_b64)
    if c is None:
        return True
    full = c > HEARD_CLIFF_DB
    logger.info(
        f"[omni] audio band edge {c:+.1f} dB at 4 kHz: "
        + ("full-band — the reply hears it directly" if full else "cut at 4 kHz — the reply waits for the transcript")
    )
    return not full



# --- Did the speaker stop mid-sentence? -------------------------------------
#
# 2026-11-08. Until now the words were also asked "is this said to Iris?"
# (a few-shot A/B on the transcript). Its examples were written from one day's TV
# lines and one visitor's sentences, and on public data that has nothing to do
# with the PROTO room it passed only 62/150 ordinary requests to an assistant
# (MASSIVE en-US test: "compose email", "should I bring a jacket tonight") and
# 47/150 lines of people talking to each other (DailyDialog). Whom speech is for
# is not in its words — a TV, a podcast and two visitors chatting say the same
# kinds of sentences a visitor says to her — so that judge is gone; who is
# talking is now seen (active_speaker.py). The words keep the one question they
# can answer for any speaker and topic: is the sentence finished?
#
# Grammar, not a word list: the hand list of English endings it replaces ("the",
# "about", "tell me about", "so i", ...) caught 66/200 sentences cut at a random
# word in public dialogue and assistant data; this catches 148/200 (most of the
# rest end where a sentence can end: "My name is Gao", "Thank you"), 142/200 and
# 139/200 with a "." or "?" stuck on the end the way a transcriber does, and
# holds 6/200 finished sentences by mistake (a held one waits ~1.5 s for more).
# The examples are generic (no PROTO-room lines); the measurement set is DailyDialog
# + MASSIVE test data the examples were not drawn from. 40-60 ms a call.

COMPLETE_ON = os.getenv("OMNI_COMPLETENESS", "1") != "0"
_COMPLETE_SYSTEM = (
    "Speech-to-text of what someone said before a pause. Transcripts are often lowercase with no punctuation, "
    "and the transcriber often puts a period or question mark at the end of a phrase that was cut off — so judge only "
    "the WORDS. Answer I if the last sentence breaks off: it ends on a word that cannot end a sentence (an article "
    "like 'the', a preposition like 'to'/'about'/'for', a conjunction like 'and'/'but', an auxiliary like 'can'/'is' "
    "without its verb, a possessive like 'my', 'very', or a subject with no verb like 'I' / 'My name'). Answer C "
    "otherwise — any phrase that can end there, including commands and questions without punctuation, greetings, "
    "thanks and one-word answers. Many speakers are not native: wrong word order, missing articles or verbs, odd "
    "word choice and fillers like 'uh' / 'um' in the middle do NOT make a sentence broken off. Look only at how it "
    "ENDS: if the last word can end a sentence (a noun, a verb, an adjective, 'me', 'you', 'it', 'here', 'please'), "
    "answer C however clumsy the rest is. One letter."
)
_COMPLETE_SHOTS = [
    ("where can i find the", "I"), ("where can i find the library", "C"), ("Thank you very.", "I"),
    ("thanks a lot", "C"), ("I would like to?", "I"), ("yes please", "C"), ("My name.", "I"),
    ("turn off the lights in the kitchen", "C"), ("Could you show me how it works?", "C"),
    ("So what I mean is.", "I"), ("what time does it", "I"), ("what time does it open", "C"),
    # Learner English (2026-11-09): 9 of 30 finished sentences like these were held as
    # broken off, none of 30 native ones — each costing the visitor a 1.5 s wait.
    ("Where I can buy ticket?", "C"), ("Please, uh, tell me how it is made.", "C"), ("Where I can buy the", "I"),
    ("You can help me little?", "C"), ("Okay, um, I want ask about your job.", "C"), ("Okay, um, I want ask about", "I"),
    ("Why it is so expensive?", "C"), ("Can you say me more simple?", "C"),
]


class Completeness:
    """unfinished(text): True = stopped mid-sentence, False = could end there,
    None = no verdict (the caller falls back to its own rules)."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._prefix = [{"role": "system", "content": _COMPLETE_SYSTEM}]
        for said, a in _COMPLETE_SHOTS:
            self._prefix += [{"role": "user", "content": f'Said: "{said}"'}, {"role": "assistant", "content": a}]
        self._recent: dict[str, asyncio.Future] = {}
        self._order: deque[str] = deque()

    async def unfinished(self, text: str, why: str = "") -> bool | None:
        text = (text or "").strip()
        if not (COMPLETE_ON and text):
            return None
        fut = self._recent.get(text)
        if fut is None:
            fut = asyncio.ensure_future(self._ask(text, why))
            self._recent[text] = fut
            self._order.append(text)
            while len(self._order) > 32:
                self._recent.pop(self._order.popleft(), None)
        try:
            return await asyncio.shield(fut)
        except Exception:  # noqa: BLE001
            return None

    async def _ask(self, text: str, why: str) -> bool | None:
        t0 = time.monotonic()
        try:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=httpx.Timeout(1.5, connect=0.5))
            body = {
                "model": MODEL, "max_tokens": 1, "temperature": 0,
                "messages": self._prefix + [{"role": "user", "content": f'Said: "{text}"'}],
            }
            r = await self._client.post(URL, json=body)
            r.raise_for_status()
            a = (r.json()["choices"][0]["message"]["content"] or "").strip().upper()[:1]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[turn] completeness check failed ({why}): {e!r}")
            return None
        if a not in ("I", "C"):
            return None
        logger.info(
            f"[turn] words {'UNFINISHED' if a == 'I' else 'finished'} in {(time.monotonic() - t0) * 1000:.0f} ms"
            f"{' (' + why + ')' if why else ''}: {text[-70:]!r}"
        )
        return a == "I"

    async def judge_audio(self, wav_b64: str, why: str = "") -> tuple[bool | None, str]:
        """Transcribe (shared with the reply, HEARD) then judge: (unfinished, heard)."""
        heard = await HEARD.get(wav_b64)
        if not heard:
            return None, ""
        return await self.unfinished(heard, why), heard


COMPLETE = Completeness()
