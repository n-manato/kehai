/* eslint-disable no-underscore-dangle */
import * as THREE from "three";
import type { VRM } from "@pixiv/three-vrm";

/**
 * LOAD-TIME WASTE REMOVAL FOR THE VRM.
 *
 * Both passes here delete data that provably cannot affect a pixel, so they are
 * verified by a ZERO-pixel render diff rather than by eye. They exist because
 * this model came out of a PMX->VRM conversion, and the converter emits two
 * kinds of bulk that nothing downstream ever reads.
 *
 * Measured on YYB Miku 10th (miku_vrm.vrm) before this file existed:
 *   - 87 THREE.Texture objects over 19 distinct images  -> ~1407 MB of GPU
 *     texture memory where 258 MB of unique pixels exist. three keys its GL
 *     textures by the Texture OBJECT, so a shared image uploaded through 22
 *     different Texture wrappers is uploaded 22 times.
 *   - 2573 morph target slots (83 on every one of the 31 primitives) of which
 *     284 hold a non-zero delta. 42.9 MB of buffers that are 89% zeros, and a
 *     MORPHTARGETS_COUNT of 83 compiled into every vertex shader — including
 *     one 5206-vertex mesh whose 83 targets are ALL zero.
 *
 * That is the difference between needing a 12 GB card and not.
 */

/** Texture-carrying slots that live as plain properties rather than uniforms. */
const DIRECT_TEX_KEYS = [
  "map", "emissiveMap", "normalMap", "aoMap", "lightMap", "alphaMap",
  "matcapTexture", "shadeMultiplyTexture", "shadingShiftTexture",
  "rimMultiplyTexture", "uvAnimationMaskTexture", "outlineWidthMultiplyTexture",
];

type AnyMat = THREE.Material & {
  uniforms?: Record<string, { value: unknown }>;
  transparent?: boolean;
  alphaTest?: number;
};

/**
 * Is this material one whose SHAPE lives in its texture's alpha?
 *
 * Mirrors the test the viewer's own mip pre-pass uses (`alphaShapedTex` in
 * vrm-viewer.tsx). It matters here because that pre-pass decides mipmapping by
 * TEXTURE IDENTITY: merging a texture used by an alpha-shaped material with one
 * used by an opaque material would hand the merged object one classification for
 * both, and the hair would either start dissolving or the trim lines start
 * aliasing. So the classification is part of the dedupe key — textures on
 * opposite sides of it stay separate objects, by design.
 *
 * Measured on this model: 3 of the 19 images (the 590x590 matcaps) really are
 * used by both kinds of material, and they are exactly the ones this keeps apart.
 */
function isAlphaShaped(m: AnyMat): boolean {
  return Boolean(m.transparent) || (m.alphaTest ?? 0) > 0;
}

function texKey(t: THREE.Texture, alphaShaped: boolean): string {
  // `source` is three's shared image wrapper: two textures cloned from one
  // another share it, which is exactly the duplication being collapsed. Falling
  // back to `image` covers textures built by hand without a Source.
  const src = (t as unknown as { source?: { uuid: string } }).source;
  const imgId = src ? `s:${src.uuid}` : `i:${String((t as unknown as { image?: unknown }).image)}`;
  // EVERY sampler property that changes what a fetch returns has to be in here,
  // or two textures that merely LOOK alike get merged and one of them silently
  // renders with the other's filtering / UV transform / colour space.
  return [
    imgId,
    t.colorSpace, t.wrapS, t.wrapT, t.magFilter, t.minFilter,
    t.format, t.type, t.anisotropy, t.flipY, t.generateMipmaps,
    t.premultiplyAlpha, t.unpackAlignment, (t as unknown as { channel?: number }).channel ?? 0,
    t.offset.x, t.offset.y, t.repeat.x, t.repeat.y, t.rotation, t.center.x, t.center.y,
    alphaShaped ? "a" : "o",
  ].join("|");
}

export interface DedupeReport {
  /** Distinct Texture objects before and after. */
  before: number;
  after: number;
  /** Estimated GPU texture bytes before and after (RGBA8 + a mip chain). */
  bytesBefore: number;
  bytesAfter: number;
}

/**
 * Collapse Texture objects that wrap the same image with the same sampler state.
 *
 * Must run BEFORE the viewer's anisotropy / dilation / coverage-mip pass, so
 * that expensive per-texture work runs once per surviving object rather than
 * once per duplicate — and so the mip classification pre-pass sees the final
 * object identities.
 */
export function dedupeTextures(root: THREE.Object3D): DedupeReport {
  const canonical = new Map<string, THREE.Texture>();
  const seen = new Set<string>();
  const replaced = new Set<THREE.Texture>();
  let before = 0;
  let bytesBefore = 0;

  const bytesOf = (t: THREE.Texture) => {
    const img = (t as unknown as { image?: { width?: number; height?: number } }).image;
    const w = img?.width ?? 0;
    const h = img?.height ?? 0;
    return w * h * 4 * (t.generateMipmaps === false ? 1 : 1.33);
  };

  const slots: { get(): THREE.Texture | null; set(t: THREE.Texture): void; alpha: boolean }[] = [];

  root.traverse((obj: THREE.Object3D) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh && !(mesh as unknown as THREE.SkinnedMesh).isSkinnedMesh) return;
    const mats = (Array.isArray(mesh.material) ? mesh.material : [mesh.material]) as AnyMat[];
    for (const m of mats) {
      if (!m) continue;
      const alpha = isAlphaShaped(m);
      const u = m.uniforms;
      if (u) {
        for (const k of Object.keys(u)) {
          const slot = u[k];
          const v = slot?.value as THREE.Texture | undefined;
          if (v && (v as THREE.Texture).isTexture) {
            slots.push({ get: () => slot.value as THREE.Texture, set: (t) => { slot.value = t; }, alpha });
          }
        }
      }
      for (const k of DIRECT_TEX_KEYS) {
        const v = (m as unknown as Record<string, unknown>)[k] as THREE.Texture | undefined;
        // Skip anything already reached through `uniforms`: on MToonMaterial
        // `m.map` is an accessor over `uniforms.map.value`, so it is the same slot.
        if (v && v.isTexture && !(u && k in u)) {
          slots.push({
            get: () => (m as unknown as Record<string, unknown>)[k] as THREE.Texture,
            set: (t) => { (m as unknown as Record<string, unknown>)[k] = t; },
            alpha,
          });
        }
      }
    }
  });

  for (const slot of slots) {
    const tex = slot.get();
    if (!tex) continue;
    if (!seen.has(tex.uuid)) {
      seen.add(tex.uuid);
      before += 1;
      bytesBefore += bytesOf(tex);
    }
    const key = texKey(tex, slot.alpha);
    const keep = canonical.get(key);
    if (!keep) {
      canonical.set(key, tex);
    } else if (keep !== tex) {
      slot.set(keep);
      replaced.add(tex);
    }
  }

  let bytesAfter = 0;
  for (const t of canonical.values()) bytesAfter += bytesOf(t);

  // Safe at load time: nothing has rendered yet, so these have no GL texture to
  // free — the dispose is bookkeeping that stops them being uploaded later.
  for (const t of replaced) if (!Array.from(canonical.values()).includes(t)) t.dispose();

  return { before, after: canonical.size, bytesBefore, bytesAfter };
}

export interface MorphReport {
  meshes: number;
  slotsBefore: number;
  slotsAfter: number;
  bytesBefore: number;
  bytesAfter: number;
  /** Meshes whose morph targets were ALL zero and are now gone entirely. */
  meshesCleared: number;
}

/**
 * Drop morph targets whose deltas are all zero, per mesh.
 *
 * PER MESH, not globally: the union of "non-zero on at least one mesh" is all 83
 * targets, so a global prune saves nothing. Per-mesh it is 2573 -> 284.
 *
 * Every consumer in this app already addresses morphs by NAME through each
 * mesh's own `morphTargetDictionary` (vrm-expression's `idxPerMesh`,
 * vrm-lipsync's `mouthIdxPerMesh`, face-motion's per-mesh `apply`), so a
 * per-mesh renumbering is invisible to them. The one thing that does assume a
 * shared order is three-vrm's own `VRMExpressionMorphTargetBind`, which holds a
 * single `index` for a list of primitives — so those binds are split per
 * primitive here.
 */
export function pruneZeroMorphTargets(vrm: VRM, eps = 1e-5): MorphReport {
  const remaps = new Map<THREE.Object3D, Int32Array>();
  const report: MorphReport = {
    meshes: 0, slotsBefore: 0, slotsAfter: 0, bytesBefore: 0, bytesAfter: 0, meshesCleared: 0,
  };

  vrm.scene.traverse((obj: THREE.Object3D) => {
    const mesh = obj as THREE.Mesh;
    const geo = mesh.geometry as THREE.BufferGeometry | undefined;
    const morphs = geo?.morphAttributes as Record<string, THREE.BufferAttribute[]> | undefined;
    if (!geo || !morphs || !morphs.position || !morphs.position.length) return;

    const kinds = Object.keys(morphs);
    const n = morphs.position.length;
    report.meshes += 1;
    report.slotsBefore += n;
    for (const kind of kinds) {
      for (const a of morphs[kind]) report.bytesBefore += a.count * a.itemSize * 4;
    }

    const keep: number[] = [];
    for (let i = 0; i < n; i += 1) {
      let nonZero = false;
      for (const kind of kinds) {
        const attr = morphs[kind][i];
        if (!attr) continue;
        const arr = attr.array as ArrayLike<number>;
        for (let k = 0; k < arr.length; k += 1) {
          if (Math.abs(arr[k]) > eps) { nonZero = true; break; }
        }
        if (nonZero) break;
      }
      if (nonZero) keep.push(i);
    }

    report.slotsAfter += keep.length;
    if (keep.length === n) return; // nothing to drop on this mesh

    const remap = new Int32Array(n).fill(-1);
    keep.forEach((old, idx) => { remap[old] = idx; });
    remaps.set(mesh, remap);

    if (keep.length === 0) {
      // No morph attribute at all: three then compiles this mesh's shaders with
      // no morph path whatsoever, which is the point.
      for (const kind of kinds) delete morphs[kind];
      report.meshesCleared += 1;
    } else {
      for (const kind of kinds) morphs[kind] = keep.map((i) => morphs[kind][i]);
      for (const kind of kinds) {
        for (const a of morphs[kind]) report.bytesAfter += a.count * a.itemSize * 4;
      }
    }

    const oldDict = mesh.morphTargetDictionary as Record<string, number> | undefined;
    if (oldDict) {
      const nd: Record<string, number> = {};
      for (const name of Object.keys(oldDict)) {
        const ni = remap[oldDict[name]];
        if (ni >= 0) nd[name] = ni;
      }
      mesh.morphTargetDictionary = nd;
    }
    const oldInf = mesh.morphTargetInfluences || [];
    mesh.morphTargetInfluences = keep.map((i) => oldInf[i] ?? 0);

    // `Mesh.updateMorphTargets()` rebuilds the dictionary from this list. Leaving
    // the original 83 names behind would make any later call re-introduce indices
    // that no longer exist.
    const names = (geo.userData as { targetNames?: string[] } | undefined)?.targetNames;
    if (Array.isArray(names)) {
      (geo.userData as { targetNames?: string[] }).targetNames = keep.map((i) => names[i]);
    }
  });

  if (remaps.size) remapExpressionBinds(vrm, remaps);
  return report;
}

/**
 * Re-point three-vrm's expression binds at the renumbered morphs.
 *
 * A `VRMExpressionMorphTargetBind` is `{ primitives: Mesh[], index, weight }` —
 * ONE index shared across every primitive. Once each mesh has its own numbering
 * that no longer holds, so each bind is split into one bind per distinct new
 * index. Primitives whose target was pruned are dropped from the bind: that
 * target was all zeros on that mesh, so driving it moved nothing.
 */
function remapExpressionBinds(vrm: VRM, remaps: Map<THREE.Object3D, Int32Array>): void {
  const em = (vrm as unknown as {
    expressionManager?: { expressions?: unknown[] };
  }).expressionManager;
  const expressions = em?.expressions;
  if (!Array.isArray(expressions)) return;

  for (const expr of expressions) {
    const e = expr as { _binds?: unknown[]; binds?: unknown[] };
    const binds = e._binds ?? e.binds;
    if (!Array.isArray(binds)) continue;
    const out: unknown[] = [];

    for (const bind of binds) {
      const b = bind as { primitives?: THREE.Mesh[]; index?: number; weight?: number };
      if (!Array.isArray(b.primitives) || typeof b.index !== "number") {
        out.push(bind); // not a morph bind (texture transform / material colour)
        continue;
      }
      const groups = new Map<number, THREE.Mesh[]>();
      for (const p of b.primitives) {
        const remap = remaps.get(p);
        const ni = remap ? (remap[b.index] ?? -1) : b.index;
        if (ni < 0) continue;
        const g = groups.get(ni);
        if (g) g.push(p);
        else groups.set(ni, [p]);
      }
      let first = true;
      for (const [ni, prims] of groups) {
        if (first) {
          b.primitives = prims;
          b.index = ni;
          out.push(bind);
          first = false;
        } else {
          const Ctor = (bind as object).constructor as new (a: {
            primitives: THREE.Mesh[]; index: number; weight: number;
          }) => unknown;
          out.push(new Ctor({ primitives: prims, index: ni, weight: b.weight ?? 1 }));
        }
      }
    }

    if (e._binds) e._binds = out;
    else e.binds = out;
  }
}

/**
 * What the model costs on the GPU right now — the number the two passes above
 * exist to move. Exposed as `window.__vrmCost()`; the alternative is reading
 * `renderer.info.memory`, which counts objects and not bytes.
 */
export function describeVrmCost(root: THREE.Object3D) {
  const seenTex = new Set<string>();
  const images = new Set<unknown>();
  let texBytes = 0;
  let morphSlots = 0;
  let morphBytes = 0;
  let tris = 0;
  let drawCalls = 0;

  root.traverse((obj: THREE.Object3D) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh && !(mesh as unknown as THREE.SkinnedMesh).isSkinnedMesh) return;
    const mats = (Array.isArray(mesh.material) ? mesh.material : [mesh.material]) as AnyMat[];
    drawCalls += mats.filter(Boolean).length;
    const geo = mesh.geometry as THREE.BufferGeometry;
    if (geo?.index) tris += geo.index.count / 3;
    else if (geo?.attributes?.position) tris += geo.attributes.position.count / 3;
    const morphs = geo?.morphAttributes as Record<string, THREE.BufferAttribute[]> | undefined;
    if (morphs) {
      for (const kind of Object.keys(morphs)) {
        morphSlots += morphs[kind].length;
        for (const a of morphs[kind]) morphBytes += a.count * a.itemSize * 4;
      }
    }
    for (const m of mats) {
      if (!m) continue;
      const collect = (t: THREE.Texture | undefined) => {
        if (!t?.isTexture || seenTex.has(t.uuid)) return;
        seenTex.add(t.uuid);
        const img = (t as unknown as { image?: { width?: number; height?: number } }).image;
        if (!img) return;
        images.add((t as unknown as { source?: unknown }).source ?? img);
        texBytes += (img.width ?? 0) * (img.height ?? 0) * 4 * (t.generateMipmaps === false ? 1 : 1.33);
      };
      if (m.uniforms) {
        for (const k of Object.keys(m.uniforms)) collect(m.uniforms[k]?.value as THREE.Texture);
      }
      for (const k of DIRECT_TEX_KEYS) {
        collect((m as unknown as Record<string, unknown>)[k] as THREE.Texture);
      }
    }
  });

  const MB = (b: number) => +(b / 1048576).toFixed(1);
  return {
    textureObjects: seenTex.size,
    distinctImages: images.size,
    textureMB: MB(texBytes),
    morphSlots,
    morphMB: MB(morphBytes),
    totalMB: MB(texBytes + morphBytes),
    triangles: Math.round(tris),
    drawCalls,
  };
}
