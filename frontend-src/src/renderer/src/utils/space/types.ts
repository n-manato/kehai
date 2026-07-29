import type * as THREE from "three";
import type { BandName } from "./layers";

/**
 * Shared shapes for the Spatial Presence Layer.
 *
 * These live apart from neural-space.ts on purpose: every visual module imports
 * them and neural-space.ts imports every visual module, so declaring them there
 * would put the whole set in an import cycle the moment one of them needs a
 * value (not just a type) from the coordinator.
 */

/**
 * The smoothed, continuous form of the AI state. Modules read this rather than
 * the raw state so a state flip eases the environment instead of snapping it.
 */
export interface SpaceMood {
  /** Slow ambient breathing of the whole space. 0..1 */
  pulse: number;
  /** Rings/near geometry reacting outward (listening). 0..1 */
  react: number;
  /** Data streaming away toward the vanishing point (thinking/speaking). 0..1 */
  stream: number;
  /** Overall energy; scales emissive on top of SPACE.emissiveGain. 0..1 */
  energy: number;
}

/** What each visual module (floor, structures, particles, …) has to provide. */
export interface SpaceModule {
  /** Objects to parent under the given depth band. */
  readonly objects: { band: BandName; object: THREE.Object3D }[];
  /** Per-frame animation. */
  update?(dt: number, elapsed: number, mood: SpaceMood): void;
  dispose(): void;
}
