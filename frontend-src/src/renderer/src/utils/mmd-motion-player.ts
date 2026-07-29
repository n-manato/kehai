/* eslint-disable no-underscore-dangle */
import * as THREE from "three";
import { MMDLoader } from "three/examples/jsm/loaders/MMDLoader.js";

const LS_MOTION_URL = "mmdMotionServiceUrl";

// Pool emotions the PC-B motion-service serves (must match motion-service/prompts.py).
const POOL_EMOTIONS = new Set([
  "neutral", "happy", "sad", "thinking", "surprised", "angry", "shy",
]);

/**
 * Map a backend motion-group name (e.g. "Joy_big", "Sadness_worry") to a pool
 * emotion. Mirrors emotionFromMotionGroup() in the Live2D talking-motion layer
 * so MMD and Live2D agree on the emotion vocabulary.
 */
export function mmdEmotionFromGroup(group?: string | null): string {
  if (!group) return "neutral";
  const g = group.toLowerCase();
  if (g.startsWith("joy")) return "happy";
  if (g.startsWith("sadness") || g.startsWith("sad")) return "sad";
  if (g.startsWith("anger") || g.startsWith("angry")) return "angry";
  if (g.startsWith("surprise")) return "surprised";
  if (g.startsWith("thinking") || g === "look") return "thinking";
  if (g.startsWith("shy")) return "shy";
  return "neutral";
}

/**
 * Map a raw emotion KEYWORD (backend `actions.emotions`, e.g. "joy_grin",
 * "anger_pout", "sadness_tearful", "surprise_amazed", "smug") to one of the 7
 * warm-pool BODY emotions. Finer keyword variants collapse to their family —
 * the HY-Motion pool only has the 7 — while the FACE keeps the full granularity
 * (see mmd-expression resolveFacePreset).
 */
export function mmdEmotionFromKeyword(keyword?: string | null): string {
  if (!keyword) return "neutral";
  const k = keyword.toLowerCase();
  if (k.startsWith("joy") || k === "smug" || k === "smirk" || k.startsWith("love"))
    return "happy";
  if (k.startsWith("sad")) return "sad";
  if (k.startsWith("anger") || k.startsWith("angry") || k === "disgust")
    return "angry";
  if (k.startsWith("surprise") || k === "fear") return "surprised";
  if (k.startsWith("think")) return "thinking";
  if (k.startsWith("shy")) return "shy";
  return "neutral";
}

/**
 * Plays MMD `.vmd` motion clips on the loaded `.pmx` at runtime — these are the
 * HY-Motion clips the PC-B motion-service generates and retargets to MMD bones.
 *
 * Design (see docs/motion-service-plan.md):
 *  - Uses its OWN THREE.AnimationMixer so the viewer's MMDAnimationHelper stays
 *    physics-only. The viewer calls `update(delta)` each frame BEFORE
 *    `helper.update(delta)`, so ammo physics (hair/skirt) reacts to the posed
 *    bones — same order MMDAnimationHelper uses internally.
 *  - While a clip is active the viewer GATES its procedural idle (skips the
 *    head/neck/upper-body bone writes) so there is exactly one writer per bone.
 *    Idle resumes (and blends back) once the clip has fully faded out.
 *  - Crossfades between successive clips; caches parsed clips by url.
 *
 * Singleton (mirrors `mmdLipSync` / `mmdExpression`) so the always-mounted audio
 * / websocket layer can drive playback without prop drilling.
 */
class MMDMotionPlayer {
  private mesh: THREE.SkinnedMesh | null = null;

  private mixer: THREE.AnimationMixer | null = null;

  private loader: MMDLoader | null = null;

  private current: THREE.AnimationAction | null = null;

  // True from the moment a clip starts until it has fully faded out — the
  // viewer gates idle on this, and we only advance the mixer while it's true.
  private active = false;

  private stopTimer: ReturnType<typeof setTimeout> | null = null;

  // Parsed AnimationClips by url (avoid re-fetching/parsing the same vmd).
  // Pool responses ROTATE server-side, so pool urls are never cached (caching
  // by url pinned the first clip of each emotion for the whole session).
  private cache = new Map<string, THREE.AnimationClip>();

  // Monotonic id of the latest play() request — a slow load finishing after a
  // newer play() must not clobber the newer clip.
  private loadSeq = 0;

  // Monotonic id of the latest gesture poll — a newer sentence's playGesture
  // cancels an older sentence's still-polling gesture fetch.
  private gestureSeq = 0;

  // Pre-fetched co-speech gesture clips (id -> VMD bytes), filled while the PRIOR
  // sentence plays (prefetchGesture) so THIS sentence's gesture is already in hand
  // at its audio start and plays aligned from ~frame 0. Consumed by playGesture.
  private gestureReady = new Map<string, ArrayBuffer>();

  // PC-B motion-service base url (e.g. http://PC-B:8800). Empty = disabled.
  private baseUrl = "";

  // Last emotion we started a clip for (debug/introspection only).
  private lastEmotion = "";

  // Throttle for the idle heartbeat (pumpIdle).
  private lastIdlePump = 0;

  // Seamless-loop state: HY-Motion clips do NOT start and end in the same pose
  // (measured loop-seam 0.3-0.6m/joint), so a hard LoopRepeat SNAPS every cycle.
  // Instead we crossfade the clip into a fresh copy of ITSELF near the end, so
  // the end pose blends smoothly back into the start pose (no visible jump).
  private currentClip: THREE.AnimationClip | null = null;

  private loopActive = false; // whether the current clip is a looping (ping-pong) clip

  setMesh(mesh: THREE.SkinnedMesh) {
    this.mesh = mesh;
    this.mixer = new THREE.AnimationMixer(mesh);
    if (!this.loader) this.loader = new MMDLoader();
    this.current = null;
    this.active = false;
    this.lastEmotion = "";
  }

  /** Set + persist the PC-B motion-service base url. "" disables live motion. */
  setBaseUrl(url: string) {
    this.baseUrl = (url || "").replace(/\/+$/, "");
    try {
      localStorage.setItem(LS_MOTION_URL, this.baseUrl);
    } catch {
      /* ignore */
    }
  }

  getBaseUrl(): string {
    if (!this.baseUrl) {
      try {
        this.baseUrl = localStorage.getItem(LS_MOTION_URL) || "";
      } catch {
        /* ignore */
      }
    }
    return this.baseUrl;
  }

  /**
   * Fetch a `.vmd` (aborting after `timeoutMs` so the body never hangs on a slow
   * gateway fallback) and play it. Status 200 -> play sized; 202 (bespoke still
   * generating) or any failure/timeout -> `onMiss()`. Speech is never blocked —
   * the audio is already playing when this runs.
   */
  private fetchAndPlay(
    url: string,
    opts: {
      durationSec?: number;
      loop?: boolean;
      fade?: number;
      onMiss?: () => void;
      timeoutMs?: number;
    } = {},
  ) {
    if (!this.mesh) return;
    const ctrl = new AbortController();
    const to = setTimeout(() => ctrl.abort(), opts.timeoutMs ?? 4000);
    fetch(url, { signal: ctrl.signal })
      .then((r) =>
        r.status === 200 ? r.arrayBuffer() : Promise.reject(new Error(String(r.status))),
      )
      .then((buf) => {
        const objUrl = URL.createObjectURL(new Blob([buf]));
        this.play(objUrl, {
          durationSec: opts.durationSec,
          loop: opts.loop,
          fade: opts.fade,
          cache: false,
          revoke: objUrl,
        });
      })
      .catch(() => opts.onMiss?.())
      .finally(() => clearTimeout(to));
  }

  /**
   * BESPOKE per-sentence clip: fetch `/motion/clip/{id}` (generated in parallel
   * with TTS). Ready -> play sized to the sentence; not ready (202) / fail ->
   * `onMiss()` (caller falls back to the fresh pool buffer).
   */
  playClip(
    url: string,
    opts: { durationSec?: number; fade?: number; onMiss?: () => void } = {},
  ) {
    this.fetchAndPlay(url, {
      durationSec: opts.durationSec,
      fade: opts.fade,
      onMiss: opts.onMiss,
    });
  }

  /**
   * Poll `/motion/clip/{id}` until ready (200) and stash the VMD bytes so a LATER
   * playGesture(id) starts instantly. Called when the audio task is RECEIVED (while
   * the PRIOR sentence is still playing), exploiting the parallel/per-sentence TTS
   * so the gesture is generated ahead and in hand by the time this sentence plays.
   * Fetch-only (no mesh needed); safe to call before the model loads.
   */
  /** True if this sentence's gesture is already pre-fetched (plays aligned, so the
   *  caller can SKIP the warm talk loop and crossfade straight from the prior one). */
  hasReadyGesture(id: string): boolean {
    return !!id && this.gestureReady.has(id);
  }

  prefetchGesture(url: string, id: string, opts: { tries?: number; intervalMs?: number } = {}) {
    if (!id || this.gestureReady.has(id)) return;
    const tries = opts.tries ?? 12;
    const intervalMs = opts.intervalMs ?? 700;
    let attempt = 0;
    if (this.gestureReady.size > 8) this.gestureReady.clear(); // bound (stale ids)
    const poll = () => {
      const ctrl = new AbortController();
      const to = setTimeout(() => ctrl.abort(), 4000);
      fetch(url, { signal: ctrl.signal })
        .then((r) => {
          if (r.status === 200) return r.arrayBuffer();
          if (r.status === 202) return null; // still generating -> retry
          throw new Error(String(r.status));
        })
        .then((buf) => {
          if (buf) this.gestureReady.set(id, buf);
          else if (++attempt < tries) setTimeout(poll, intervalMs);
        })
        .catch(() => {})
        .finally(() => clearTimeout(to));
    };
    poll();
  }

  /**
   * Audio-driven CO-SPEECH gesture, SYNCED to the audio. The gesture was generated
   * from THIS sentence's finished audio (GestureLSM, 30fps) so gesture-time ==
   * audio-time. We (a) use the pre-fetched clip if present (ready at sentence start
   * -> aligned from ~frame 0), else POLL `/motion/clip/{id}` until 200; then (b)
   * play it FORWARD-ONCE at 1.0x SEEKED to `startAtSec()` (the audio's current
   * position) so it lands in step with speech — instant sync-in for late arrivals.
   * `guard()` drops a clip whose sentence already ended (don't smear a stale gesture
   * over the next sentence). The caller keeps a warm talk loop playing underneath,
   * which this crossfades over. A newer sentence's playGesture supersedes this poll.
   */
  playGesture(
    url: string,
    opts: {
      id?: string;
      startAtSec?: () => number;
      guard?: () => boolean;
      tries?: number;
      intervalMs?: number;
    } = {},
  ) {
    if (!this.mesh) return;
    const startAtSec = opts.startAtSec ?? (() => 0);
    const guard = opts.guard ?? (() => true);
    const tries = opts.tries ?? 8;
    const intervalMs = opts.intervalMs ?? 700;
    const gseq = ++this.gestureSeq;

    const playBuf = (buf: ArrayBuffer) => {
      if (gseq !== this.gestureSeq || !this.mesh) return; // superseded / torn down
      if (!guard()) return; // this sentence already ended -> don't smear it forward
      const objUrl = URL.createObjectURL(new Blob([buf]));
      // FORWARD-ONCE (LoopOnce + clampWhenFinished); 1.0x so it stays in step with
      // the audio it was made from; SEEK to the audio's current position so start
      // (or mid-sentence sync-in) aligns. The silence-padded tail settles to rest
      // and the next sentence's gesture crossfades over it.
      this.play(objUrl, {
        loop: false,
        cache: false,
        revoke: objUrl,
        speed: 1.0,
        startAt: startAtSec(),
      });
    };

    // Fast path: pre-fetched during the prior sentence -> play immediately, aligned.
    const ready = opts.id ? this.gestureReady.get(opts.id) : undefined;
    if (ready) {
      this.gestureReady.delete(opts.id!);
      playBuf(ready);
      return;
    }

    let attempt = 0;
    const poll = () => {
      if (gseq !== this.gestureSeq || !this.mesh) return; // superseded / torn down
      const ctrl = new AbortController();
      const to = setTimeout(() => ctrl.abort(), 4000);
      fetch(url, { signal: ctrl.signal })
        .then((r) => {
          if (r.status === 200) return r.arrayBuffer();
          if (r.status === 202) return null; // still generating -> retry
          throw new Error(String(r.status));
        })
        .then((buf) => {
          if (gseq !== this.gestureSeq) return; // a newer gesture won the race
          if (buf) playBuf(buf);
          else if (++attempt < tries) setTimeout(poll, intervalMs);
        })
        .catch(() => {
          /* fetch failed/aborted — the warm loop keeps playing; give up quietly */
        })
        .finally(() => clearTimeout(to));
    };
    poll();
  }

  /** Cancel any in-flight gesture poll (e.g. on reply end / interruption). */
  cancelGesture() {
    this.gestureSeq += 1;
  }

  /**
   * Play a fresh POOL-buffer clip for a body KEY and LOOP it. The key is a
   * content gesture intent ("point"/"count"/"shrug"...) or one of the 7 emotions;
   * the backend's /motion/pool resolves anything unknown to neutral. Same key
   * keeps the CURRENT loop (does not re-fetch) and only crossfades to a fresh
   * clip when the key CHANGES — so the body switches exactly when the gesture or
   * emotion changes, and holds a steady loop otherwise (not restless).
   */
  playEmotion(key: string | null | undefined) {
    if (!this.mesh) return;
    const k = key || "neutral";
    if (k === this.lastEmotion && this.active) return; // keep the current loop
    this.lastEmotion = k;
    const base = this.getBaseUrl();
    this.fetchAndPlay(`${base}/motion/pool/${encodeURIComponent(k)}`, {
      loop: true,
      onMiss: () => {
        if (this.lastEmotion === k) this.lastEmotion = ""; // let the next retry
      },
    });
  }

  /**
   * Idle heartbeat: while idle (no active reply clip) keep a fresh neutral pool
   * clip looping, refreshed every few seconds so idle body motion stays alive and
   * varied ("continuously generated"). Self-throttled; call each frame with
   * whether the app is idle. No-op while a reply clip is active.
   */
  pumpIdle(isIdle: boolean) {
    if (!this.mesh || !isIdle || this.active) return;
    const now = Date.now();
    if (now - this.lastIdlePump < 6000) return;
    this.lastIdlePump = now;
    const base = this.getBaseUrl();
    this.fetchAndPlay(`${base}/motion/pool/neutral`, { loop: true });
  }

  /** A clip is currently playing or fading out (viewer gates idle on this). */
  isActive() {
    return this.active;
  }

  /** Load a `.vmd` and crossfade to it, sized to `durationSec` if given. */
  play(
    url: string,
    opts: {
      loop?: boolean;
      fade?: number;
      cache?: boolean;
      durationSec?: number;
      revoke?: string;
      onError?: () => void;
      speed?: number;
      startAt?: number;
    } = {},
  ) {
    const { mesh, mixer, loader } = this;
    if (!mesh || !mixer || !loader) {
      console.warn("[MMD] motion player: no mesh registered");
      if (opts.revoke) URL.revokeObjectURL(opts.revoke); // don't leak the blob
      return;
    }
    const fade = opts.fade ?? 0.3;
    // Conversation/idle clips play at NATURAL SPEED and LOOP — the known-good
    // baseline. A clip loops to fill a long sentence and is cut by the next
    // sentence's crossfade for a short one, so it never freezes on a clamped last
    // frame or plays at a distorted speed. (An earlier attempt to timeScale-
    // stretch each clip to the exact sentence length made the motion look wrong.)
    const loopIt = opts.loop ?? true;
    const cacheable = opts.cache ?? true;
    const seq = ++this.loadSeq;

    const start = (clip: THREE.AnimationClip) => {
      if (!this.mixer || !this.mesh) return;
      if (this.stopTimer) {
        clearTimeout(this.stopTimer);
        this.stopTimer = null;
      }
      const action = this.mixer.clipAction(clip);
      action.reset();
      // CO-SPEECH SYNC: seek into the clip so its frame aligns with the audio's
      // current playback position (the gesture was generated from THIS audio at
      // 30fps, so gesture-time == audio-time). Ready-at-sentence-start -> ~0
      // (aligned); late-arriving -> seek forward (sync-in). Clamp inside the clip.
      if (opts.startAt && opts.startAt > 0) {
        action.time = Math.min(opts.startAt, Math.max(0, clip.duration - 0.02));
      }
      if (loopIt) {
        // Seamless loop with NO seam: ping-pong plays the clip forward then
        // backward, so the turnaround pose is IDENTICAL (C0-continuous) — no snap,
        // and none of the joint-slide the old self-crossfade produced trying to
        // bridge the 0.3-0.6m gap between these non-cyclic HY-Motion clips' start
        // and end poses ("継ぎ目/カクつき"). Reversed playback reads naturally for
        // conversational hand gestures, and it needs no per-cycle clip cloning.
        action.setLoop(THREE.LoopPingPong, Infinity);
        action.clampWhenFinished = false;
      } else {
        // Bespoke one-shot sized to a sentence: play once and hold the last frame.
        action.setLoop(THREE.LoopOnce, 1);
        action.clampWhenFinished = true;
      }
      action.enabled = true;
      action.setEffectiveTimeScale(opts.speed ?? 1);
      action.setEffectiveWeight(1);
      action.play();
      const prev = this.current;
      if (prev && prev !== action) {
        // Crossfade from the previous clip (both play; weight blends over `fade`).
        prev.crossFadeTo(action, fade, false);
        // Uncached clips are new objects on every fetch — release the faded-out
        // one's mixer bindings or a long session leaks clips.
        const prevClip = prev.getClip();
        setTimeout(() => {
          if (this.mixer && this.current !== prev) {
            prev.stop();
            const isCached = [...this.cache.values()].includes(prevClip);
            if (!isCached) this.mixer.uncacheClip(prevClip);
          }
        }, fade * 1000 + 100);
      } else {
        action.fadeIn(fade);
      }
      this.current = action;
      this.currentClip = clip;
      this.loopActive = loopIt;
      this.active = true;
    };

    const cached = cacheable ? this.cache.get(url) : undefined;
    if (cached) {
      start(cached);
      if (opts.revoke) URL.revokeObjectURL(opts.revoke);
      return;
    }
    // MMDLoader.loadAnimation(url, mesh, onLoad) -> AnimationClip (bone + morph tracks).
    loader.loadAnimation(
      url,
      mesh,
      (clip: unknown) => {
        if (opts.revoke) URL.revokeObjectURL(opts.revoke);
        if (seq !== this.loadSeq) return; // superseded by a newer play()
        const c = clip as THREE.AnimationClip;
        if (cacheable) this.cache.set(url, c);
        start(c);
      },
      undefined,
      (e: unknown) => {
        if (opts.revoke) URL.revokeObjectURL(opts.revoke);
        console.error("[MMD] vmd load error:", e);
        if (seq === this.loadSeq) opts.onError?.();
      },
    );
  }

  /** Fade the current clip out and hand control back to procedural idle. */
  stop(fade = 0.3) {
    this.gestureSeq += 1; // cancel any in-flight gesture poll (reply end / interrupt)
    this.loopActive = false; // stop seam-looping while fading out
    if (!this.current) {
      this.active = false;
      return;
    }
    this.current.fadeOut(fade);
    if (this.stopTimer) clearTimeout(this.stopTimer);
    // Keep `active` (idle stays gated, mixer keeps advancing) until the fade is
    // done, so idle and the fading clip never write the same bone at once.
    this.stopTimer = setTimeout(
      () => {
        this.current?.stop();
        this.current = null;
        this.active = false;
        this.lastEmotion = "";
        this.stopTimer = null;
      },
      Math.max(0, fade * 1000),
    );
  }

  /** Advance the clip mixer. Call once per frame, before helper.update(). */
  update(delta: number) {
    if (!this.mixer || !this.active) return;
    // Looping clips ping-pong natively (see start()), so update() just advances
    // the mixer — no per-cycle clip cloning / self-crossfade to babysit.
    this.mixer.update(delta);
  }

  clearMesh(mesh: THREE.SkinnedMesh) {
    if (this.mesh === mesh) {
      if (this.stopTimer) {
        clearTimeout(this.stopTimer);
        this.stopTimer = null;
      }
      this.mixer?.stopAllAction();
      this.mesh = null;
      this.mixer = null;
      this.current = null;
      this.currentClip = null;
      this.loopActive = false;
      this.active = false;
      this.lastEmotion = "";
      this.cache.clear();
    }
  }
}

export const mmdMotionPlayer = new MMDMotionPlayer();
