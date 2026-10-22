import * as THREE from "three";

/**
 * "AI" presence for an android avatar: its emissive trim breathes with the
 * conversation, and its silhouette carries a thin cyan rim.
 *
 * Opt-in PER ASSET, not per app. A model asks for it by carrying
 * `extras.kehaiAiFx` on any glTF node (three puts node extras on
 * `object.userData`), e.g. `{ glowMaterial: "glow", rimColor: "#3fe3f5" }`.
 * That keeps the setting with the file it was tuned for: Miku and the photoreal
 * .glb avatars never see it, and no model_dict field has to be plumbed through
 * the React tree to reach the render closure.
 *
 * Why the body and not the room: on the PROTO panel black is transparent, and
 * every drawn environment tried there argued with the real depth behind the
 * glass (see body-depth.ts). Light that comes from HER — trim lines on black —
 * is the one thing the panel shows at full strength.
 *
 * State language, read at a glance from across a room:
 *   idle      — cyan, slow breathing (the unit is on)
 *   listening — cyan, steady and a touch brighter (attending)
 *   thinking  — violet, quicker pulse (processing)
 *   speaking  — cyan towards white, intensity follows the voice
 *   loading   — dim
 *
 * The shell itself is a pearl finish: the albedo is pulled to a neutral,
 * slightly cool white (the authored plates lean green-grey and the lips stay
 * human pink, both of which read "painted person" rather than "machine"), and
 * the light plates carry a thin-film sheen that shifts cyan -> violet ->
 * magenta towards the silhouette. Dark joints are left alone so the structure
 * still reads. Tune live with `__aiFx.pearl.<uniform>.value`.
 */

export interface AiFxConfig {
  glowMaterial: string;
  rimColor: string;
}

/** The rim uniforms vrm-viewer's sublight injection creates per material. */
interface RimUniforms {
  uFresInt: { value: number };
  uFresPow: { value: number };
  uFresColor: { value: THREE.Color };
}

type FxState = "idle" | "listening" | "thinking" | "speaking" | "loading";

const CYAN = new THREE.Color("#3fe3f5");
const VIOLET = new THREE.Color("#8c5cff");
const WHITE = new THREE.Color("#e8fbff");

// Rim strength per state. The fresnel term is additive over the lit surface,
// so these are small: at 0.5 the rim already reads as an edge of light on
// white plates, and above ~0.8 it starts to look like a selection outline.
const RIM: Record<FxState, number> = {
  idle: 0.35, listening: 0.45, thinking: 0.4, speaking: 0.45, loading: 0.15,
};

// Emissive intensity the state multipliers scale from. Blender-authored glow
// arrives at strength ~30, which tone-maps to white whatever its hue — every
// state then looks the same. At 2.5 cyan and violet stay readable, and the
// speaking peak (~6) still blows out to white, which is the point of it.
const GLOW_BASE = 2.5;

// A voice level above this, held recently, means she is audibly speaking —
// `thinking-speaking` alone cannot tell the LLM wait from playback.
const SPEAKING_LEVEL = 0.03;
const SPEAKING_HOLD_S = 0.35;

// Pearl shell. Shared by every shell material, so one console poke retunes all.
//   uPearlSat  — albedo saturation kept (0 = grey, 1 = authored)
//   uPearlTint — multiplier after desaturation; >1 lifts the plates to white,
//                blue/red over green removes the authored green cast
//   uIriInt    — thin-film sheen strength at the silhouette
//   uIriPow    — how far in from the edge the sheen reaches (higher = thinner)
//   uIriPhase  — slow drift of the film colours, advanced in update()
const pearlUniforms = {
  uPearlSat: { value: 0.12 },
  uPearlTint: { value: new THREE.Vector3(1.16, 1.12, 1.2) },
  uIriInt: { value: 0.55 },
  uIriPow: { value: 1.8 },
  uIriPhase: { value: 0 },
};

const PEARL_DECLS =
  "uniform float uPearlSat;\nuniform vec3 uPearlTint;\n" +
  "uniform float uIriInt;\nuniform float uIriPow;\nuniform float uIriPhase;\n";

// After map_fragment: diffuseColor holds the textured albedo (linear).
const PEARL_ALBEDO = [
  "{",
  "  float pl = dot( diffuseColor.rgb, vec3( 0.2126, 0.7152, 0.0722 ) );",
  "  diffuseColor.rgb = mix( vec3( pl ), diffuseColor.rgb, uPearlSat ) * uPearlTint;",
  "}",
].join("\n");

// Before opaque_fragment. Three-stop film palette rather than a full rainbow:
// green and yellow bands read as oil slick, not as a designed object.
const PEARL_SHEEN = [
  "{",
  "  vec3 pN = normalize( normal );",
  "  vec3 pV = normalize( vViewPosition );",
  "  float pEdge = pow( clamp( 1.0 - abs( dot( pN, pV ) ), 0.0, 1.0 ), uIriPow );",
  "  float pT = fract( pEdge * 0.85 + uIriPhase );",
  "  vec3 pC = mix( vec3( 0.25, 0.89, 0.96 ), vec3( 0.55, 0.36, 1.0 ), smoothstep( 0.0, 0.5, pT ) );",
  "  pC = mix( pC, vec3( 1.0, 0.42, 0.85 ), smoothstep( 0.5, 1.0, pT ) );",
  // Only on the light plates: a sheen on black joints reads as a glow outline.
  "  float pPlate = smoothstep( 0.2, 0.45, dot( diffuseColor.rgb, vec3( 0.2126, 0.7152, 0.0722 ) ) );",
  "  outgoingLight += pC * pEdge * uIriInt * pPlate;",
  "}",
].join("\n");

class AiFx {
  enabled = false;

  private cfg: AiFxConfig | null = null;

  private glow: { mat: THREE.MeshStandardMaterial; base: number; color: THREE.Color }[] = [];

  private rims = new Set<RimUniforms>();

  private appState = "idle";

  private t = 0;

  private lastVoiceAt = -1;

  private level = 0;

  private col = new THREE.Color();

  private rimCol = new THREE.Color();

  private tgt = new THREE.Color();

  private intensity = 1;

  private rimInt = 0;

  /** Look for `kehaiAiFx` extras under `root`; returns whether the model opted in. */
  bind(root: THREE.Object3D): boolean {
    this.unbind();
    root.traverse((o: THREE.Object3D) => {
      const c = (o.userData as { kehaiAiFx?: AiFxConfig }).kehaiAiFx;
      if (c && !this.cfg) this.cfg = c;
    });
    if (!this.cfg) return false;
    const seen = new Set<THREE.Material>();
    root.traverse((o: THREE.Object3D) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      for (const m of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
        const sm = m as THREE.MeshStandardMaterial;
        if (seen.has(m)) continue;
        seen.add(m);
        if (sm.name === this.cfg!.glowMaterial && sm.emissive) {
          this.glow.push({ mat: sm, base: sm.emissiveIntensity, color: sm.emissive.clone() });
        } else if (sm.isMeshStandardMaterial) {
          this.pearlShell(sm);
        }
      }
    });
    CYAN.set(this.cfg.rimColor);
    this.col.copy(CYAN);
    this.rimCol.copy(CYAN);
    this.enabled = true;
    console.info(
      `[ai-fx] on: ${this.glow.length} glow, ${this.shells} pearl shell material(s), rim ${this.cfg.rimColor}`,
    );
    return true;
  }

  readonly pearl = pearlUniforms;

  private shells = 0;

  /**
   * Chain the pearl finish into the material's shader. Runs before the
   * viewer's sublight injection, which wraps this as its previous
   * onBeforeCompile, so both edits land. Not undone in unbind(): that only
   * runs when the model is torn down and its materials go with it.
   */
  private pearlShell(m: THREE.MeshStandardMaterial) {
    const prev = m.onBeforeCompile;
    m.onBeforeCompile = (shader: any, renderer: any) => {
      prev.call(m, shader, renderer);
      Object.assign(shader.uniforms, pearlUniforms);
      shader.fragmentShader = shader.fragmentShader
        .replace("#include <common>", `${PEARL_DECLS}#include <common>`)
        .replace("#include <map_fragment>", `#include <map_fragment>\n${PEARL_ALBEDO}`)
        .replace("#include <opaque_fragment>", `${PEARL_SHEEN}\n#include <opaque_fragment>`);
    };
    // The viewer's wrapper stringifies the same for the glow material too;
    // without a distinct key three could hand one of them the other's program.
    m.customProgramCacheKey = () => "kehai-ai-fx-pearl-v1";
    m.needsUpdate = true;
    this.shells++;
  }

  unbind() {
    for (const g of this.glow) {
      g.mat.emissive.copy(g.color);
      g.mat.emissiveIntensity = g.base;
    }
    this.glow = [];
    this.shells = 0;
    this.rims.clear();
    this.cfg = null;
    this.enabled = false;
  }

  /** Called from the viewer's look pass for every material it injects. */
  adoptRim(u: RimUniforms | undefined) {
    if (!this.enabled || !u) return;
    this.rims.add(u);
    u.uFresPow.value = 2.2;
    u.uFresInt.value = this.rimInt;
    u.uFresColor.value.copy(this.rimCol);
  }

  setAiState(s: string) {
    this.appState = s;
  }

  /** Pin a state for tuning / screenshots (`__aiFx.preview("thinking")`); null releases. */
  preview(s: FxState | null) {
    this.pinned = s;
  }

  private pinned: FxState | null = null;

  private state(): FxState {
    if (this.pinned) return this.pinned;
    if (this.appState === "loading") return "loading";
    if (this.lastVoiceAt >= 0 && this.t - this.lastVoiceAt < SPEAKING_HOLD_S) return "speaking";
    if (this.appState === "listening") return "listening";
    if (this.appState === "thinking-speaking") return "thinking";
    return "idle";
  }

  /** Per frame. `voice` is the smoothed speech level (0..1) from vrmLipSync. */
  update(dt: number, voice: number) {
    if (!this.enabled) return;
    this.t += dt;
    // One full cycle of the film colours every ~40s: alive, never busy.
    pearlUniforms.uIriPhase.value = (this.t * 0.025) % 1;
    if (voice > SPEAKING_LEVEL) this.lastVoiceAt = this.t;
    this.level += (voice - this.level) * Math.min(1, dt * 18);

    const s = this.state();
    let k: number;
    const target = this.tgt;
    if (s === "speaking") {
      const v = Math.min(1, this.level * 2.2);
      k = 0.85 + 1.6 * v;
      target.copy(CYAN).lerp(WHITE, v * 0.6);
    } else if (s === "thinking") {
      k = 0.75 + 0.4 * Math.sin(this.t * Math.PI * 2 * 1.1);
      target.copy(VIOLET);
    } else if (s === "listening") {
      k = 1.15;
      target.copy(CYAN);
    } else if (s === "loading") {
      k = 0.25;
      target.copy(CYAN);
    } else {
      k = 0.8 + 0.18 * Math.sin(this.t * Math.PI * 2 * 0.22);
      target.copy(CYAN);
    }

    // Colour eases over ~0.25s so a state change reads as a shift, not a cut;
    // intensity follows faster or the voice coupling turns to mush.
    this.col.lerp(target, Math.min(1, dt * 4));
    this.intensity += (k - this.intensity) * Math.min(1, dt * 14);
    for (const g of this.glow) {
      g.mat.emissive.copy(this.col);
      g.mat.emissiveIntensity = GLOW_BASE * this.intensity;
    }

    const rimTarget = RIM[s] + (s === "speaking" ? 0.25 * Math.min(1, this.level * 2.2) : 0);
    this.rimInt += (rimTarget - this.rimInt) * Math.min(1, dt * 6);
    this.rimCol.copy(this.col).lerp(CYAN, 0.5);
    for (const u of this.rims) {
      u.uFresInt.value = this.rimInt;
      u.uFresColor.value.copy(this.rimCol);
    }
  }
}

export const aiFx = new AiFx();

// Console handle for tuning on the unit itself: `__aiFx.preview("thinking")`.
(window as unknown as { __aiFx: AiFx }).__aiFx = aiFx;
