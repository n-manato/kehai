/* eslint-disable no-underscore-dangle */
import * as THREE from "three";
import { SPACE } from "../space/space-config";
import { taa } from "../space/taa";

/**
 * `window.__AI_DEBUG__` — the hooks that let an automated viewer (Claude driving
 * chrome-devtools-mcp, or a Playwright visual test) photograph this scene and get
 * the SAME pixels every time.
 *
 * Why this exists: a screenshot of a live three.js scene is not evidence of
 * anything on its own. Between two captures the VRM has blinked, breathed, moved
 * its gaze, advanced an idle clip, and every `uTime` shader has rolled forward, so
 * an image diff reports a change that says nothing about the code that changed.
 * Freezing the clock is what turns "a picture of the app" into a measurement.
 *
 * The whole thing hangs off ONE interception point. The render loop derives its
 * timestep once, in vrm-viewer's `animate()`, and hands that dt to the motion
 * player, the spring bones, the Bullet hair sim, the environment, and every shader
 * uniform downstream. Route that single number through `step()` and the entire
 * frame is under control — no per-system freeze flags, and nothing to keep in sync
 * as systems are added.
 *
 * Two clocks needed separating. Most systems integrate dt, but blink and gaze
 * schedule themselves against wall-clock `performance.now()`, so a dt of 0 would
 * leave them running. `now()` is the debug-aware replacement: it returns the
 * simulated clock once the debug clock is driving, and the real one otherwise.
 *
 * SEEKING IS FORWARD-ONLY, and that is a deliberate limitation rather than an
 * unfinished feature. Spring-bone and Bullet rigid-body state is path-dependent —
 * there is no inverse for "the hair settled like this" — so a backwards seek could
 * only ever produce a plausible-looking lie. `?debug=1&t=2` replays from zero in
 * exact 1/60 steps instead, which is reproducible to the bit on every load.
 *
 * Costs nothing when unused: `step()` is a branch on a boolean that stays false,
 * and `random()` falls through to `Math.random` until a seed is set.
 */

interface AttachCtx {
  camera: THREE.PerspectiveCamera;
  controls: { target: THREE.Vector3; enabled: boolean; update: () => void };
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  /** Re-solve the framing after a preset change (viewer's own camera solver). */
  reframe?: () => void;
}

/** Named camera setups. `dist` is metres back from the target on +Z. */
const CAMERA_PRESETS: Record<string, { ty: number; dist: number }> = {
  // The shipping desktop framing — matches vrm-viewer's own defaults exactly, so
  // "front" is the shot the user actually looks at rather than a debug-only view.
  front: { ty: 0.98, dist: 2.4 },
  final: { ty: 0.98, dist: 2.4 },
  fullbody: { ty: 0.85, dist: 4.2 },
  face: { ty: 1.36, dist: 0.85 },
  // Pulled back and centred on the ring system, for judging the console alone.
  hud: { ty: 1.15, dist: 3.6 },
};

/** One step per frame, always this size. A seek therefore costs its own duration
 *  in wall time — cheap enough for the 1-3 s marks a screenshot actually uses,
 *  and it keeps the loop untouched, which matters more than seek speed here. */
const FIXED_STEP = 1 / 60;

class AiDebug {
  private ctx: AttachCtx | null = null;

  private frozen = false;

  /** Simulated time in ms. Authoritative once `driving` is true. */
  private simMs = 0;

  /** True once freeze/setTime has taken over; `now()` switches source on it. */
  private driving = false;

  /** Seek target in ms, or null when not seeking. */
  private seekTo: number | null = null;

  private seekResolvers: (() => void)[] = [];

  private rngState: number | null = null;

  /**
   * Rolling window of REAL frame deltas in ms, for `frameStats()`. ~4s at 60fps.
   *
   * Recorded here because `step()` is the one place the true wall-clock delta is
   * known, and it is called once per frame whether or not the clock is frozen —
   * so the cost of a render can be measured on a FROZEN scene, which is the only
   * way to compare two settings without the animation itself moving the number.
   */
  private frameMs: number[] = [];

  /** Single-step mode: the loop renders only when a frame has been asked for. */
  private paused = false;

  private pending = 0;

  private stepping = false;

  private frameResolvers: (() => void)[] = [];

  // --- clock -------------------------------------------------------------

  /**
   * Convert the real frame delta into the delta the scene should advance by.
   * Call exactly once per frame, as early as possible in the loop, with the
   * RAW delta — the long-stall clamp lives in here now.
   */
  step(realDt: number): number {
    // Sample before clamping. The clamp exists so a tab-switch or a GC pause
    // cannot teleport the integrators; applying it to the statistics too would
    // silently cap every frame-time reading at 33.3ms and report a stuttering
    // scene as a solid 30fps.
    if (realDt > 0) {
      this.frameMs.push(realDt * 1000);
      if (this.frameMs.length > 240) this.frameMs.shift();
    }
    // eslint-disable-next-line no-param-reassign
    realDt = Math.min(realDt, 1 / 30);
    // Single-stepping: a fixed step for the one frame that was asked for, and no
    // advance at all otherwise. Checked before the seek branch so a step is never
    // interpreted as seek progress.
    if (this.stepping) {
      this.simMs += FIXED_STEP * 1000;
      return FIXED_STEP;
    }
    if (this.seekTo !== null) {
      // Fixed-size steps, never a single big jump: the spring solver and the
      // Bullet sim are integrators, so a 2-second dt does not land where two
      // seconds of 1/60 steps land. Replaying at the real substep size is the
      // only way a seek reproduces normal playback.
      const remaining = this.seekTo - this.simMs;
      if (remaining <= 0.5) {
        this.seekTo = null;
        this.frozen = true;
        const rs = this.seekResolvers;
        this.seekResolvers = [];
        rs.forEach((r) => r());
        return 0;
      }
      this.simMs += FIXED_STEP * 1000;
      return FIXED_STEP;
    }
    if (this.frozen) return 0;
    this.simMs += realDt * 1000;
    return realDt;
  }

  /**
   * What a frame currently COSTS, in real time.
   *
   * Nothing else in the app reports this. The on-screen HUD computes an fps, but
   * it is module-local to space-hud.ts and only ever reaches a DOM string, so
   * judging "did that change make it slower" used to mean showing the HUD and
   * scraping text. `renderer.info` is exposed and is a true whole-pipeline draw
   * count, but draw calls are not time: raising the render resolution costs a
   * lot and changes `calls` by zero.
   *
   * Median rather than mean — one GC pause should not decide the verdict — with
   * p95 alongside so a stutter is still visible. `resetFrameStats()` first, wait
   * out the window, then read.
   */
  frameStats(): {
    samples: number
    fps: number | null
    msMedian: number | null
    msP95: number | null
  } {
    const a = [...this.frameMs].sort((x, y) => x - y);
    if (!a.length) return { samples: 0, fps: null, msMedian: null, msP95: null };
    const at = (p: number) => a[Math.min(a.length - 1, Math.floor(a.length * p))];
    const med = at(0.5);
    return {
      samples: a.length,
      fps: +(1000 / med).toFixed(1),
      msMedian: +med.toFixed(2),
      msP95: +at(0.95).toFixed(2),
    };
  }

  /** Drop the window, so the next read describes only what happens after now. */
  resetFrameStats(): void {
    this.frameMs = [];
  }

  /** Debug-aware wall clock, in ms. Use instead of `performance.now()` for
   *  anything that SCHEDULES itself (blink, gaze) rather than integrating dt. */
  now(): number {
    if (this.driving) return this.simMs;
    return typeof performance !== "undefined" ? performance.now() : Date.now();
  }

  /** Stop the scene dead. Everything holds its current pose and shader phase. */
  freeze() {
    this.driving = true;
    this.frozen = true;
    return this.state();
  }

  /** Hand the clock back to the browser. */
  play() {
    this.frozen = false;
    this.seekTo = null;
    this.driving = false;
    this.paused = false;
    this.pending = 0;
    this.stepping = false;
    const rs = this.frameResolvers;
    this.frameResolvers = [];
    rs.forEach((r) => r());
    return this.state();
  }

  // --- single-stepping ----------------------------------------------------

  /**
   * Render EXACTLY ONE frame, advancing the clock by one fixed step.
   *
   * WHY `setTime()` IS NOT THIS. setTime freezes the clock; it does not stop the
   * render loop, which keeps drawing the frozen pose for as long as the harness
   * takes to come back and photograph it. That was harmless while every pass was
   * stateless — the same pose redrawn is the same pixels — but the temporal
   * resolve carries a history between frames, so those extra redraws let it
   * converge on a pose the live app only ever renders once. A capture taken that
   * way flatters TAA on moving content and cannot be compared against a
   * pre-TAA baseline.
   *
   * Here the loop is suspended and each call lets one frame through, so
   * consecutive captures are consecutive frames in the sense the eye means — and
   * with a fixed dt, so they are also reproducible, which live playback is not.
   */
  stepFrame(): Promise<unknown> {
    this.driving = true;
    this.frozen = true;
    this.paused = true;
    this.pending += 1;
    return new Promise<void>((resolve) => {
      this.frameResolvers.push(resolve);
    }).then(() => this.state());
  }

  /** Leave single-step mode but stay frozen where we are. */
  endStep() {
    this.paused = false;
    this.pending = 0;
    this.stepping = false;
    const rs = this.frameResolvers;
    this.frameResolvers = [];
    rs.forEach((r) => r());
    return this.state();
  }

  /**
   * Gate at the very top of the render loop. Returns false when the loop should
   * do nothing at all this tick. Always true unless `stepFrame()` is driving, so
   * normal operation is one boolean test.
   */
  beginFrame(): boolean {
    if (!this.paused) return true;
    if (this.pending <= 0) return false;
    this.pending -= 1;
    this.stepping = true;
    return true;
  }

  /** Call after the frame has been presented. Releases whoever asked for it. */
  endFrame(): void {
    if (!this.stepping) return;
    this.stepping = false;
    const rs = this.frameResolvers;
    this.frameResolvers = [];
    rs.forEach((r) => r());
  }

  /**
   * Advance to `seconds` on the debug clock and freeze there.
   * Forward-only — see the file header. Resolves once the scene has settled.
   */
  setTime(seconds: number): Promise<unknown> {
    const target = Math.max(0, seconds) * 1000;
    if (target < this.simMs) {
      // Say so loudly rather than freezing at the wrong time and letting a test
      // bake a wrong reference image.
      const msg = `[ai-debug] cannot rewind ${(this.simMs / 1000).toFixed(2)}s -> `
        + `${seconds}s (spring/rigid-body state is path-dependent). `
        + `Reload with ?debug=1&t=${seconds} to replay from zero.`;
      console.warn(msg);
      return Promise.reject(new Error(msg));
    }
    this.driving = true;
    this.frozen = false;
    this.seekTo = target;
    return new Promise<void>((resolve) => {
      this.seekResolvers.push(resolve);
    }).then(() => this.state());
  }

  /** Resolves when no seek is pending — the "safe to screenshot" signal. */
  ready(): Promise<unknown> {
    if (this.seekTo === null) return Promise.resolve(this.state());
    return new Promise<void>((resolve) => {
      this.seekResolvers.push(resolve);
    }).then(() => this.state());
  }

  // --- grain -------------------------------------------------------------

  /**
   * Film grain is a per-pixel randomiser, and at this amplitude it turns a
   * sub-pixel difference in the capture path into a whole-frame one: measured on
   * this scene, two captures of a FROZEN render differ on 9.4% of pixels with
   * grain on and 0.15% with it off. So automated capture always runs without it.
   *
   * Deliberately not routed through LOOK: `__vrmLook()` persists to localStorage,
   * and a debug session must never leave the user's saved look changed.
   */
  private noGrain = false;

  suppressGrain(on = true) {
    this.noGrain = on;
    return this.noGrain;
  }

  /** Wrap the authored grain value wherever it is pushed to the uniform. */
  grainFor(v: number): number {
    return this.noGrain ? 0 : v;
  }

  // --- randomness --------------------------------------------------------

  /**
   * Seeded replacement for `Math.random()` at the few sites that are genuinely
   * nondeterministic (gaze, blink, idle-clip gap). The environment modules do
   * NOT need this — they already build from their own fixed LCG seeds.
   * `setSeed(null)` restores real randomness.
   */
  setSeed(seed: number | null) {
    this.rngState = seed === null ? null : seed >>> 0;
    return this.state();
  }

  random(): number {
    if (this.rngState === null) return Math.random();
    this.rngState = (this.rngState * 1664525 + 1013904223) >>> 0;
    return this.rngState / 4294967296;
  }

  // --- framing -----------------------------------------------------------

  attach(ctx: AttachCtx) {
    this.ctx = ctx;
  }

  // --- viewpoint motion ---------------------------------------------------

  /**
   * Turn the slow camera drift on/off.
   *
   * Its own hook rather than `__space({cameraDrift:0})` because that path calls
   * `applyFraming()`, which re-solves the camera — so the A/B it is meant to
   * support would move the camera for a second reason and the comparison would
   * be worthless. This only flips the flag the drift itself reads.
   */
  setCameraDriftEnabled(on: boolean) {
    SPACE.cameraDriftEnabled = !!on;
    return { cameraDriftEnabled: SPACE.cameraDriftEnabled, amplitudeM: SPACE.cameraDrift };
  }

  /** Exaggerated per-band layer separation on top of the real parallax. */
  setParallaxEnabled(on: boolean) {
    if (on) {
      SPACE.parallaxBoost = this.parallaxLast > 0 ? this.parallaxLast : 1;
    } else {
      if (SPACE.parallaxBoost > 0) this.parallaxLast = SPACE.parallaxBoost;
      SPACE.parallaxBoost = 0;
    }
    return { parallaxBoost: SPACE.parallaxBoost };
  }

  setParallaxStrength(v: number) {
    SPACE.parallaxBoost = Math.max(0, v);
    if (SPACE.parallaxBoost > 0) this.parallaxLast = SPACE.parallaxBoost;
    return { parallaxBoost: SPACE.parallaxBoost };
  }

  private parallaxLast = 1;

  /**
   * Put the camera exactly here. The presets cannot express an arbitrary view,
   * and a mouse-drag orbit is not reproducible — two runs of the same drag
   * landed on [0,0.036,0.413] and [0.515,0.052,0.426], which makes an A/B of two
   * settings meaningless because the views differ. Anything that has to compare
   * two runs frame-by-frame needs this instead.
   */
  setCamera(pos: number[], target: number[]) {
    if (!this.ctx) return { error: "not attached" };
    const { camera, controls } = this.ctx;
    controls.target.set(target[0], target[1], target[2]);
    camera.position.set(pos[0], pos[1], pos[2]);
    camera.lookAt(controls.target);
    controls.update();
    camera.updateMatrixWorld();
    taa.reset(); // a teleport invalidates every pixel of the temporal history
    return {
      position: camera.position.toArray().map((n) => +n.toFixed(4)),
      target: controls.target.toArray().map((n) => +n.toFixed(4)),
    };
  }

  setCameraPreset(name: string) {
    const p = CAMERA_PRESETS[name];
    if (!p) return { error: `unknown preset "${name}"`, available: Object.keys(CAMERA_PRESETS) };
    if (!this.ctx) return { error: "not attached" };
    const { camera, controls } = this.ctx;
    controls.target.set(0, p.ty, 0);
    camera.position.set(0, p.ty + 0.07, p.dist);
    camera.lookAt(controls.target);
    controls.update();
    // A preset TELEPORTS the camera, so every pixel of the temporal history now
    // belongs to a different view. Tests call this AFTER waitUntilStill, so a kept
    // history would carry the previous framing straight into the screenshot.
    taa.reset();
    return { preset: name, ...p };
  }

  // --- readouts ----------------------------------------------------------

  getRendererInfo() {
    const info = this.ctx?.renderer.info;
    if (!info) return null;
    return {
      calls: info.render.calls,
      triangles: info.render.triangles,
      lines: info.render.lines,
      points: info.render.points,
      geometries: info.memory.geometries,
      textures: info.memory.textures,
      programs: info.programs?.length ?? null,
    };
  }

  /**
   * Where objects actually land on screen, in normalised device coordinates
   * (-1..1 on each axis; outside that is off-frame).
   *
   * Exists because arithmetic on the frustum keeps getting this wrong. A panel is
   * rotated about Y, so one of its edges swings TOWARD the camera and is magnified
   * by perspective — the flat `|x| <= half-width(z)` estimate says it fits when it
   * does not. Projecting the real corners through the real camera has no such
   * assumption in it.
   *
   * @param prefix match objects whose own name, or a parent's, starts with this
   */
  ndcBounds(prefix: string) {
    if (!this.ctx) return null;
    const { camera, scene } = this.ctx;
    camera.updateMatrixWorld();
    const out: Record<string, unknown>[] = [];
    const v = new THREE.Vector3();
    scene.traverse((o: THREE.Object3D) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh || !mesh.geometry) return;
      let label = "";
      for (let p: THREE.Object3D | null = o; p; p = p.parent) {
        if (p.name?.startsWith(prefix)) { label = o.name || p.name; break; }
      }
      if (!label) return;
      mesh.geometry.computeBoundingBox();
      const bb = mesh.geometry.boundingBox!;
      let minX = Infinity; let maxX = -Infinity; let minY = Infinity; let maxY = -Infinity;
      for (let i = 0; i < 8; i += 1) {
        v.set(i & 1 ? bb.max.x : bb.min.x, i & 2 ? bb.max.y : bb.min.y, i & 4 ? bb.max.z : bb.min.z);
        mesh.localToWorld(v).project(camera);
        minX = Math.min(minX, v.x); maxX = Math.max(maxX, v.x);
        minY = Math.min(minY, v.y); maxY = Math.max(maxY, v.y);
      }
      const r = (n: number) => +n.toFixed(3);
      out.push({
        label,
        x: [r(minX), r(maxX)],
        y: [r(minY), r(maxY)],
        offscreen: minX < -1 || maxX > 1 || minY < -1 || maxY > 1,
        overflow: {
          left: r(Math.max(0, -1 - minX)),
          right: r(Math.max(0, maxX - 1)),
          bottom: r(Math.max(0, -1 - minY)),
          top: r(Math.max(0, maxY - 1)),
        },
      });
      return;
    });
    return { aspect: +this.ctx.camera.aspect.toFixed(4), fov: this.ctx.camera.fov, items: out };
  }

  getSceneInfo() {
    if (!this.ctx) return null;
    let objects = 0;
    let meshes = 0;
    let visibleMeshes = 0;
    const materials = new Set<string>();
    this.ctx.scene.traverse((o: THREE.Object3D) => {
      objects += 1;
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      meshes += 1;
      if (o.visible) visibleMeshes += 1;
      const mat = Array.isArray(m.material) ? m.material : [m.material];
      mat.forEach((x: THREE.Material | undefined) => x && materials.add(x.type));
    });
    const { camera } = this.ctx;
    return {
      objects,
      meshes,
      visibleMeshes,
      materialTypes: [...materials],
      camera: {
        position: camera.position.toArray().map((n: number) => +n.toFixed(3)),
        target: this.ctx.controls.target.toArray().map((n: number) => +n.toFixed(3)),
        fov: camera.fov,
        aspect: +camera.aspect.toFixed(4),
      },
    };
  }

  state() {
    return {
      frozen: this.frozen,
      driving: this.driving,
      seeking: this.seekTo !== null,
      time: +(this.simMs / 1000).toFixed(4),
      seed: this.rngState,
      grainSuppressed: this.noGrain,
      // A frozen scene is not a still IMAGE until the temporal history latches —
      // the jitter keeps walking while the clock is stopped. Existing assertions
      // only read .frozen / .seeking / .time, so this is additive.
      taa: taa.state(),
    };
  }

  getStats() {
    return { ...this.state(), renderer: this.getRendererInfo(), scene: this.getSceneInfo() };
  }
}

export const aiDebug = new AiDebug();

/**
 * FLASH RECORDER — a black box for the artefact nobody can reproduce on demand.
 *
 * The under-skirt "the front flashes for a frame" report survived every
 * harness-driven hypothesis: TAA one-frame pops (deterministic lockstep:
 * TAA never changes more per frame than the raw render), transparent-sort
 * flips (no one-frame pops in 150 deterministic swing frames), z-fighting
 * (frozen scene + walking jitter: byte-stable), and dt spikes (55 ms stalls
 * injected live: nothing). The one machine it demonstrably happens on is the
 * user's, mid-conversation — so the instrument has to live there.
 *
 * It keeps a rolling ring of the last ~1.2 s of frames (copied GPU-side each
 * rAF, cheap) and, when the user presses "9" at the moment they SEE it, POSTs
 * the ring to the server's /debug/flash-frame drop-box, where the analysis
 * tooling can actually reach the files. Enabled by default while the hunt is
 * on; `__flashRec(0)` turns it off, and the ring only exists while enabled.
 */
class FlashRecorder {
  private ring: HTMLCanvasElement[] = [];

  private meta: { f: number; t: number; taa: boolean }[] = [];

  private head = 0;

  private frame = 0;

  // OFF by default since 2026-08-13: the artefact it was hunting (the necktie
  // flap popping through the hips — see mmd-hair-physics.ts, the group-7 mask
  // exception) was caught and fixed. `__flashRec(1)` re-arms it.
  private enabled = false;

  private uploading = false;

  private raf = 0;

  private readonly SLOTS = 72; // ~1.2 s at 60 fps

  /** One-frame-pop auto-detector state: 96x96 luma-ish downsamples of the last
   *  three frames, and a rolling baseline of frame-to-frame difference. */
  private small: HTMLCanvasElement | null = null;

  private popData: (Uint8ClampedArray | null)[] = [null, null, null];

  private popBase: number[] = [];

  private popCooldown = 0;

  private autoCount = 0;

  start(): void {
    if (this.raf) return;
    const step = () => {
      this.raf = requestAnimationFrame(step);
      if (!this.enabled || this.uploading) return;
      const src = document.querySelector("canvas");
      if (!src || !(src as HTMLCanvasElement).width) return;
      const c = src as HTMLCanvasElement;
      this.frame += 1;
      if (!this.ring[this.head]) {
        this.ring[this.head] = document.createElement("canvas");
        this.ring[this.head].width = 640;
        this.ring[this.head].height = Math.max(1, Math.round((640 * c.height) / c.width));
      }
      const slot = this.ring[this.head];
      const g = slot.getContext("2d");
      if (g) g.drawImage(c, 0, 0, slot.width, slot.height);
      this.meta[this.head] = {
        f: this.frame,
        t: aiDebug.now(),
        taa: (window as any).__AI_DEBUG__?.state?.().taa?.enabled ?? false,
      };
      this.head = (this.head + 1) % this.SLOTS;
      this.detectPop(c);
    };
    this.raf = requestAnimationFrame(step);
    window.addEventListener("keydown", (e) => {
      // Not while typing — a "9" in the chat box must stay a "9".
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      if (e.key === "9" && !e.repeat) this.dump();
    });
  }

  /**
   * The reflex the human cannot have: something that appears for EXACTLY one
   * frame is gone 16ms later, long before a hand reaches the "9" key. The
   * signature is mechanical — prev->cur changes a lot, cur->next changes a
   * lot, prev->next barely changes (real motion fails the third test: its
   * prev->next is the sum, not zero). On a hit, the three full-res ring
   * frames upload themselves, named auto-*.
   */
  private detectPop(src: HTMLCanvasElement): void {
    if (!this.small) {
      this.small = document.createElement("canvas");
      this.small.width = 96;
      this.small.height = 96;
    }
    const g = this.small.getContext("2d", { willReadFrequently: true });
    if (!g) return;
    g.drawImage(src, 0, 0, 96, 96);
    this.popData[0] = this.popData[1];
    this.popData[1] = this.popData[2];
    this.popData[2] = new Uint8ClampedArray(g.getImageData(0, 0, 96, 96).data);
    const [a, b, c2] = this.popData;
    if (!a || !b || !c2) return;
    const diff = (A: Uint8ClampedArray, B: Uint8ClampedArray) => {
      let s = 0;
      for (let i = 0; i < A.length; i += 4) {
        s += Math.abs(A[i] - B[i]) + Math.abs(A[i + 1] - B[i + 1]) + Math.abs(A[i + 2] - B[i + 2]);
      }
      return s / (A.length / 4) / 3;
    };
    const d01 = diff(a, b);
    const d12 = diff(b, c2);
    const d02 = diff(a, c2);
    this.popBase.push(d01);
    if (this.popBase.length > 150) this.popBase.shift();
    if (this.popCooldown > 0) this.popCooldown -= 1;
    const med = [...this.popBase].sort((x, y) => x - y)[Math.floor(this.popBase.length / 2)];
    const hot = Math.max(med * 2.5, med + 3);
    if (this.popBase.length > 60 && this.popCooldown === 0 && this.autoCount < 8
      && d01 > hot && d12 > hot && d02 < 0.5 * Math.min(d01, d12)) {
      this.autoCount += 1;
      this.popCooldown = 90;
      // The pop is the MIDDLE of the three; ring head already advanced once.
      const idx = [
        (this.head + this.SLOTS - 3) % this.SLOTS,
        (this.head + this.SLOTS - 2) % this.SLOTS,
        (this.head + this.SLOTS - 1) % this.SLOTS,
      ];
      // eslint-disable-next-line no-console
      console.log(`[flashRec] one-frame pop auto-detected (d01 ${d01.toFixed(1)} d12 ${d12.toFixed(1)} through ${d02.toFixed(1)} med ${med.toFixed(1)}) — uploading triplet`);
      void this.uploadSlots(idx, `auto-${new Date().toISOString().replace(/[:.]/g, "-")}`);
    }
  }

  private async uploadSlots(indices: number[], stamp: string): Promise<number> {
    let sent = 0;
    for (let i = 0; i < indices.length; i += 1) {
      const slot = this.ring[indices[i]];
      const m = this.meta[indices[i]];
      if (!slot || !m) continue;
      const name = `${stamp}-${String(i).padStart(2, "0")}-f${m.f}-taa${m.taa ? 1 : 0}`;
      // eslint-disable-next-line no-await-in-loop
      await fetch("/debug/flash-frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, png: slot.toDataURL("image/png") }),
      }).catch(() => { /* debug path — losing one frame beats breaking the app */ });
      sent += 1;
    }
    return sent;
  }

  setEnabled(on: boolean): { flashRec: boolean } {
    this.enabled = on;
    if (!on) {
      this.ring = [];
      this.meta = [];
    }
    return { flashRec: this.enabled };
  }

  /** Ship the whole ring, oldest first. Serialisation is the expensive part,
   *  so the ring pauses while it runs — the moment is already captured. */
  async dump(): Promise<{ sent: number } | { error: string }> {
    if (this.uploading) return { error: "already uploading" };
    this.uploading = true;
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    try {
      const indices = Array.from({ length: this.SLOTS }, (_, i) => (this.head + i) % this.SLOTS);
      const sent = await this.uploadSlots(indices, stamp);
      // eslint-disable-next-line no-console
      console.log(`[flashRec] ${sent} frames uploaded as ${stamp}-*`);
      return { sent };
    } catch (e) {
      return { error: String(e) };
    } finally {
      this.uploading = false;
    }
  }
}

export const flashRecorder = new FlashRecorder();

/** `?debug=1` — present so a capture tool can pin the scene from the URL alone. */
export function debugParams(): { debug: boolean; t: number | null; seed: number | null } {
  try {
    const q = new URLSearchParams(window.location.search);
    const debug = q.get("debug") === "1" || q.get("debug") === "true";
    const tRaw = q.get("t");
    const seedRaw = q.get("seed");
    return {
      debug,
      t: tRaw !== null && tRaw !== "" ? Number(tRaw) : null,
      seed: seedRaw !== null && seedRaw !== "" ? Number(seedRaw) : null,
    };
  } catch {
    return { debug: false, t: null, seed: null };
  }
}
