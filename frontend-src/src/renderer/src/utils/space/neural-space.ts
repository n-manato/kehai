/* eslint-disable no-underscore-dangle */
import * as THREE from "three";
import type { AiState } from "../../context/ai-state-context";
import { BANDS, SPACE_LAYER, type BandName } from "./layers";
import { createBackdrop } from "./backdrop";
import { createShadow } from "./shadow";
import { createWireframe } from "./wireframe";
import { createTraffic } from "./traffic";
import { createParticles } from "./particles";
import { createOrbit } from "./orbit";
import { createNearfield } from "./nearfield";
import { createHudRings } from "./hud-rings";
import { createPanels } from "./panels";
import { createTowers } from "./towers";
import { applyParallax } from "./off-axis";
import type { SpaceModule, SpaceMood } from "./types";
import {
  SPACE,
  getBudget,
  getQuality,
  loadSpaceConfig,
  saveSpaceConfig,
  setQualityValue,
  type Quality,
} from "./space-config";

/**
 * The "Neural Space": the 3D environment that turns the panel from a screen
 * showing a character into a window onto a deep space she is standing in.
 *
 * Owns one `THREE.Group` per depth band (see layers.ts) and a list of modules
 * that fill them (floor, structures, particles). The viewer only ever calls
 * init / update / setAiState / setQuality / dispose — a singleton so the React
 * component and the imperative three.js closure can both reach it without prop
 * drilling, mirroring vrmLipSync / mmdHairPhysics.
 *
 * Everything it adds lives on SPACE_LAYER so the viewer's ink-line pre-pass can
 * exclude it wholesale; nothing here touches the character, its lights, the
 * camera's near/far, or the composer chain.
 */

/** Per-state mood targets. Phase 6 replaces this with real choreography. */
const MOOD_TARGETS: Record<string, SpaceMood> = {
  idle: { pulse: 1, react: 0, stream: 0.08, energy: 0.55 },
  listening: { pulse: 0.5, react: 1, stream: 0.15, energy: 0.8 },
  "thinking-speaking": { pulse: 0.35, react: 0.25, stream: 1, energy: 0.95 },
  interrupted: { pulse: 0.6, react: 0.4, stream: 0.2, energy: 0.6 },
  loading: { pulse: 0.8, react: 0, stream: 0.5, energy: 0.35 },
  waiting: { pulse: 0.9, react: 0.15, stream: 0.1, energy: 0.6 },
};

class NeuralSpace {
  private scene: THREE.Scene | null = null;

  private renderer: THREE.WebGLRenderer | null = null;

  private camera: THREE.PerspectiveCamera | null = null;

  private root: THREE.Group | null = null;

  /**
   * One group per depth band. Phase 3 shifts these by PARALLAX[band] when the
   * viewer's head moves; keeping the split from day one means motion parallax
   * is a transform on five objects rather than a rebuild.
   */
  private bands = new Map<BandName, THREE.Group>();

  private modules: SpaceModule[] = [];

  private elapsed = 0;

  private state: AiState = "idle";

  private mood: SpaceMood = { ...MOOD_TARGETS.idle };

  private moodTarget: SpaceMood = { ...MOOD_TARGETS.idle };

  /** Build the environment and attach it to the scene. Call once, after lights. */
  init(scene: THREE.Scene, renderer: THREE.WebGLRenderer, camera: THREE.PerspectiveCamera) {
    if (this.root) this.dispose();
    loadSpaceConfig();

    this.scene = scene;
    this.renderer = renderer;
    this.camera = camera;

    const root = new THREE.Group();
    root.name = "neural-space";
    // Purely decorative: never let it drive the character's shadow map or AO.
    root.matrixAutoUpdate = false;
    this.root = root;

    (Object.keys(BANDS) as BandName[]).forEach((band) => {
      const g = new THREE.Group();
      g.name = `space-${band}`;
      g.userData.band = band;
      this.bands.set(band, g);
      root.add(g);
    });

    scene.add(root);
    // The environment lives off layer 0; the camera has to be told to see it.
    camera.layers.enable(SPACE_LAYER);

    if (SPACE.enabled) this.build();
    root.visible = SPACE.enabled;
  }

  /** (Re)create the visual modules for the current quality/settings. */
  private build() {
    if (!this.scene || !this.renderer || !this.camera) return;
    this.teardownModules();
    this.attachModules([
      createBackdrop(), createShadow(), createWireframe(), createTraffic(), createParticles(),
      // The two that carry depth now that no room is drawn: rings crossing her
      // silhouette, and marks in front of her. Both are occlusion, which is the
      // only cue a fixed viewpoint on a flat panel cannot flatten.
      createOrbit(), createNearfield(),
      // The holographic console: rings behind her, UI panels flanking her, and a
      // far field of towers. All BEHIND her — she occludes them, never the
      // other way round — and none of it draws a floor, wall or ceiling.
      createHudRings(), createPanels(this.camera, this.renderer, this.scene), createTowers(),
    ]);
  }

  private attachModules(modules: SpaceModule[]) {
    modules.forEach((m) => {
      m.objects.forEach(({ band, object }) => {
        applyLayer(object, SPACE_LAYER);
        this.bands.get(band)?.add(object);
      });
      this.modules.push(m);
    });
  }

  private teardownModules() {
    this.modules.forEach((m) => {
      m.objects.forEach(({ band, object }) => {
        this.bands.get(band)?.remove(object);
      });
      m.dispose();
    });
    this.modules = [];
  }

  /** Advance the environment. Call once per frame, before composer.render(). */
  update(dt: number) {
    if (!this.root || !SPACE.enabled) return;
    this.elapsed += dt;
    // Optional exaggerated layer separation on top of the physically-correct
    // parallax that off-axis projection already produces (Phase 3).
    applyParallax(this.bands);

    // Ease the mood so a state flip never snaps the space. ~0.6s to settle.
    const k = 1 - Math.pow(0.05, dt / 0.6);
    this.mood.pulse += (this.moodTarget.pulse - this.mood.pulse) * k;
    this.mood.react += (this.moodTarget.react - this.mood.react) * k;
    this.mood.stream += (this.moodTarget.stream - this.mood.stream) * k;
    this.mood.energy += (this.moodTarget.energy - this.mood.energy) * k;

    for (let i = 0; i < this.modules.length; i += 1) {
      this.modules[i].update?.(dt, this.elapsed, this.mood);
    }
  }

  /** React to the conversation state (idle / listening / thinking-speaking / …). */
  setAiState(state: AiState) {
    this.state = state;
    this.moodTarget = MOOD_TARGETS[state] ?? MOOD_TARGETS.idle;
  }

  getAiState(): AiState {
    return this.state;
  }

  /** Swap the performance budget and rebuild. Cheap enough to call from a panel. */
  setQuality(q: Quality) {
    if (q === getQuality()) return;
    setQualityValue(q);
    saveSpaceConfig();
    if (this.root && SPACE.enabled) this.build();
  }

  /**
   * Re-apply settings changed through `window.__space({...})`.
   *
   * Deliberately does NOT rebuild: every module re-reads SPACE in its per-frame
   * update, so colours, intensities and haze take effect on the next frame.
   * Only settings that change object COUNTS need `rebuild()`.
   */
  refresh() {
    saveSpaceConfig();
    if (!this.root) return;
    if (SPACE.enabled && this.modules.length === 0) this.build();
    this.root.visible = SPACE.enabled;
  }

  /** Recreate every module — for changes that alter how much geometry exists. */
  rebuild() {
    saveSpaceConfig();
    if (this.root && SPACE.enabled) this.build();
  }

  /** Current resolved budget — handy from the console when tuning. */
  describe() {
    // renderer.info is the honest cost readout: Phase 3's head-coupled
    // perspective needs frame budget, so what matters is how many draw calls and
    // triangles this layer added, not a frame rate measured on some other box.
    const info = this.renderer?.info;
    return {
      enabled: SPACE.enabled,
      quality: getQuality(),
      budget: getBudget(),
      modules: this.modules.length,
      state: this.state,
      mood: { ...this.mood },
      render: info ? { calls: info.render.calls, triangles: info.render.triangles } : null,
    };
  }

  dispose() {
    this.teardownModules();
    if (this.root && this.scene) this.scene.remove(this.root);
    this.bands.clear();
    this.root = null;
    this.scene = null;
    this.renderer = null;
    this.camera = null;
    this.elapsed = 0;
  }
}

/** Put an object and everything under it on a layer (layers don't inherit). */
export function applyLayer(object: THREE.Object3D, layer: number) {
  object.traverse((o: THREE.Object3D) => o.layers.set(layer));
}

export const neuralSpace = new NeuralSpace();
