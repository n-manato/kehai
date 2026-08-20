import * as THREE from "three";

/**
 * Audition what a loaded model's face can actually do.
 *
 * Two different things live behind the one panel, and they are worth keeping
 * apart because they carry very different amounts of content:
 *
 *  - EMBEDDED CLIPS. A .glb can ship glTF animations. Chancelor has exactly one,
 *    `Female_Animation_Idle` (9.97 s), and while it does contain morph tracks,
 *    measuring them showed the facial part is only `eyeBlinkLeft` /
 *    `eyeBlinkRight` going 0 -> 1. Everything else in that clip is the skeleton.
 *    So "the model has face motion" is true and also nearly empty, and playing
 *    the clip fights the app's own body driver — hence the warning in the panel.
 *
 *  - MORPH TARGETS. This is where the face actually is: 293 of them on this
 *    model, an ARKit-style expression set plus the standard Oculus visemes
 *    (aa/E/ih/oh/ou/PP/FF/TH/DD/kk/CH/SS/nn/RR). Nothing drives them yet, so the
 *    only way to know what is in there is to watch them one at a time.
 *
 * Overrides are applied AFTER vrmExpression and vrmLipSync each frame, so the
 * panel wins while it is open regardless of which model is loaded — on Miku
 * those two are actively writing every frame and would otherwise erase it.
 */

interface Bound {
  meshes: THREE.Mesh[];
  /** morph name -> index, as reported by the meshes themselves. */
  dict: Record<string, number>;
}

/** A one-shot 0 -> 1 -> 0 sweep, so a click shows what a morph does. */
interface Sweep {
  name: string;
  t: number;
  dur: number;
}

export class FaceMotion {
  private bound: Bound = { meshes: [], dict: {} };

  private held = new Map<string, number>();

  private sweeps: Sweep[] = [];

  private mixer: THREE.AnimationMixer | null = null;

  private clips: THREE.AnimationClip[] = [];

  private action: THREE.AnimationAction | null = null;

  /** Names of every morph the loaded model exposes, sorted. */
  get morphNames(): string[] {
    return Object.keys(this.bound.dict).sort();
  }

  get clipNames(): string[] {
    return this.clips.map((c) => c.name);
  }

  get playing(): string | null {
    return this.action && this.action.isRunning() ? this.action.getClip().name : null;
  }

  bind(meshes: THREE.Mesh[], clips: THREE.AnimationClip[], root: THREE.Object3D) {
    this.clear();
    // One dictionary merged across every morph-carrying mesh. A face is usually
    // split by material (head / eyelashes / teeth), each primitive holding its
    // own influences array, and a morph named on one is often named on another —
    // driving all of them together is what makes an expression look whole.
    const dict: Record<string, number> = {};
    const kept: THREE.Mesh[] = [];
    for (const m of meshes) {
      if (!m.morphTargetDictionary || !m.morphTargetInfluences) continue;
      kept.push(m);
      for (const k of Object.keys(m.morphTargetDictionary)) {
        if (dict[k] == null) dict[k] = (m.morphTargetDictionary as Record<string, number>)[k];
      }
    }
    this.bound = { meshes: kept, dict };
    this.clips = clips || [];
    this.mixer = this.clips.length ? new THREE.AnimationMixer(root) : null;
  }

  clear() {
    this.stopClip();
    this.held.clear();
    this.sweeps = [];
    this.bound = { meshes: [], dict: {} };
    this.clips = [];
    this.mixer = null;
  }

  /** Hold a morph at `v` until released. null releases it. */
  hold(name: string, v: number | null) {
    if (v == null) this.held.delete(name);
    else this.held.set(name, v);
  }

  releaseAll() {
    this.held.clear();
    this.sweeps = [];
  }

  sweep(name: string, dur = 1.2) {
    this.sweeps = this.sweeps.filter((s) => s.name !== name);
    this.sweeps.push({ name, t: 0, dur });
  }

  playClip(name: string) {
    if (!this.mixer) return false;
    const clip = this.clips.find((c) => c.name === name);
    if (!clip) return false;
    this.stopClip();
    this.action = this.mixer.clipAction(clip);
    this.action.reset();
    this.action.setLoop(THREE.LoopRepeat, Infinity);
    this.action.play();
    return true;
  }

  stopClip() {
    if (this.action) {
      this.action.stop();
      this.action = null;
    }
  }

  /**
   * Call once per frame, AFTER the expression and lip-sync systems have written
   * their own values — this deliberately overwrites them so that what the panel
   * shows is what you asked for and not a blend with an idle blink.
   */
  update(dt: number) {
    if (this.mixer && this.action) this.mixer.update(dt);

    if (!this.bound.meshes.length) return;
    if (!this.held.size && !this.sweeps.length) return;

    const apply = (name: string, v: number) => {
      for (const m of this.bound.meshes) {
        const idx = (m.morphTargetDictionary as Record<string, number> | undefined)?.[name];
        if (idx == null || !m.morphTargetInfluences) continue;
        m.morphTargetInfluences[idx] = v;
      }
    };

    for (const [name, v] of this.held) apply(name, v);

    for (let i = this.sweeps.length - 1; i >= 0; i -= 1) {
      const s = this.sweeps[i];
      s.t += dt;
      const p = Math.min(1, s.t / s.dur);
      // up then down, eased, so the shape is readable at the top rather than
      // flashing past it
      const tri = p < 0.5 ? p * 2 : (1 - p) * 2;
      apply(s.name, tri * tri * (3 - 2 * tri));
      if (p >= 1) {
        apply(s.name, 0);
        this.sweeps.splice(i, 1);
      }
    }
  }

  /** Group morphs the way a person would look for them. */
  groupOf(name: string): string {
    const n = name.toLowerCase();
    if (/^(aa|e|ee|ih|oh|ou|pp|ff|th|dd|kk|ch|ss|nn|rr|sil)$/.test(n)) return "viseme";
    if (n.startsWith("jaw") || n.startsWith("mouth") || n.startsWith("tongue")) return "mouth";
    if (n.startsWith("eyeb") || n.startsWith("brow")) return "brow";
    if (n.startsWith("eye")) return "eye";
    if (n.startsWith("cheek") || n.startsWith("nose")) return "face";
    if (/[+-]$/.test(name)) return "body-shape";
    return "other";
  }
}

export const faceMotion = new FaceMotion();
