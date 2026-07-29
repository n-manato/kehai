/* eslint-disable no-underscore-dangle */
import * as THREE from "three";
import { SPACE } from "../space/space-config";

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

  // --- clock -------------------------------------------------------------

  /**
   * Convert the real frame delta into the delta the scene should advance by.
   * Call exactly once per frame, as early as possible in the loop.
   */
  step(realDt: number): number {
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
    return this.state();
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

  setCameraPreset(name: string) {
    const p = CAMERA_PRESETS[name];
    if (!p) return { error: `unknown preset "${name}"`, available: Object.keys(CAMERA_PRESETS) };
    if (!this.ctx) return { error: "not attached" };
    const { camera, controls } = this.ctx;
    controls.target.set(0, p.ty, 0);
    camera.position.set(0, p.ty + 0.07, p.dist);
    camera.lookAt(controls.target);
    controls.update();
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
    };
  }

  getStats() {
    return { ...this.state(), renderer: this.getRendererInfo(), scene: this.getSceneInfo() };
  }
}

export const aiDebug = new AiDebug();

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
