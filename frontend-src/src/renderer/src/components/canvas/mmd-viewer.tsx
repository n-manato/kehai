/* eslint-disable no-underscore-dangle */
/* eslint-disable @typescript-eslint/ban-ts-comment */
import { memo, useEffect, useRef } from "react";
import * as THREE from "three";
import { MMDLoader } from "three/examples/jsm/loaders/MMDLoader.js";
import { MMDAnimationHelper } from "three/examples/jsm/animation/MMDAnimationHelper.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { OutlineEffect } from "three/examples/jsm/effects/OutlineEffect.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { SMAAPass } from "three/examples/jsm/postprocessing/SMAAPass.js";
import { BokehPass } from "three/examples/jsm/postprocessing/BokehPass.js";
import { Pass } from "three/examples/jsm/postprocessing/Pass.js";
import { mmdLipSync } from "@/utils/mmd-lipsync";
import { mmdExpression } from "@/utils/mmd-expression";
import { mmdMotionPlayer } from "@/utils/mmd-motion-player";

/**
 * Screen-space cel + colour-grade pass. Re-shades the rendered image to match
 * the music-video look: optional cel posterize, then a cinematic grade —
 * chromatic aberration (the cyan/magenta edge fringing all over the video),
 * colour temperature / tint, and a teal-shadow / magenta-highlight split-tone —
 * plus saturation & contrast. Done in ~perceptual (gamma) space.
 */
const ToonShader = {
  uniforms: {
    tDiffuse: { value: null },
    uSteps: { value: 0.0 }, // colour bands per channel (0 = off / smooth)
    uSat: { value: 1.14 },
    uContrast: { value: 1.06 },
    uChroma: { value: 0.0 }, // chromatic aberration (radial RGB split)
    uTemp: { value: 0.0 }, // colour temperature (- cooler, + warmer)
    uTint: { value: 0.0 }, // tint (- green, + magenta)
    uSplit: { value: 0.0 }, // split-tone amount (teal shadows / magenta highlights)
    uVignette: { value: 0.0 }, // edge darkening (FilmGrain "vignette")
    uGrain: { value: 0.0 }, // film grain amount
    uBars: { value: 0.0 }, // letterbox bar height (fraction, top & bottom)
    uTime: { value: 0.0 }, // animates the grain
  },
  vertexShader: `
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float uSteps;
    uniform float uSat;
    uniform float uContrast;
    uniform float uChroma;
    uniform float uTemp;
    uniform float uTint;
    uniform float uSplit;
    uniform float uVignette;
    uniform float uGrain;
    uniform float uBars;
    uniform float uTime;
    varying vec2 vUv;
    void main() {
      vec2 dir = vUv - 0.5;
      // Chromatic aberration: offset R outward / B inward, stronger toward edges.
      vec4 c;
      if (uChroma > 0.001) {
        vec2 off = dir * uChroma * 0.004;
        c.r = texture2D(tDiffuse, vUv + off).r;
        c.g = texture2D(tDiffuse, vUv).g;
        c.b = texture2D(tDiffuse, vUv - off).b;
        c.a = texture2D(tDiffuse, vUv).a;
      } else {
        c = texture2D(tDiffuse, vUv);
      }
      vec3 col = pow(max(c.rgb, 0.0), vec3(1.0 / 2.2)); // -> ~sRGB
      if (uSteps > 0.5) col = floor(col * uSteps + 0.5) / uSteps; // cel banding
      // Temperature (warm/cool) & tint (green/magenta).
      col.r *= 1.0 + uTemp * 0.12;
      col.b *= 1.0 - uTemp * 0.12;
      col.r *= 1.0 + uTint * 0.06;
      col.b *= 1.0 + uTint * 0.06;
      col.g *= 1.0 - uTint * 0.06;
      float l = dot(col, vec3(0.299, 0.587, 0.114));
      // Split-tone: teal in the shadows, magenta in the highlights.
      if (uSplit > 0.001) {
        vec3 shadowTint = vec3(0.10, 0.55, 0.62);
        vec3 hiTint = vec3(0.72, 0.30, 0.78);
        vec3 tintCol = mix(shadowTint, hiTint, smoothstep(0.2, 0.8, l)) * 2.0;
        col = mix(col, col * tintCol, uSplit * 0.5);
      }
      col = mix(vec3(l), col, uSat); // saturation
      col = (col - 0.5) * uContrast + 0.5; // contrast
      // Vignette: darken toward the edges (the video's framing).
      if (uVignette > 0.001) {
        float vd = length(vUv - 0.5);
        col *= 1.0 - uVignette * smoothstep(0.42, 0.95, vd);
      }
      // Film grain: subtle animated noise for a filmic surface.
      if (uGrain > 0.001) {
        float n = fract(sin(dot(vUv * (uTime + 1.0), vec2(12.9898, 78.233))) * 43758.5453);
        col += (n - 0.5) * uGrain;
      }
      col = pow(clamp(col, 0.0, 1.0), vec3(2.2)); // -> linear
      // Letterbox bars (pure black, top & bottom) applied last.
      if (uBars > 0.0001 && (vUv.y < uBars || vUv.y > 1.0 - uBars)) col = vec3(0.0);
      gl_FragColor = vec4(col, c.a);
    }
  `,
};

/**
 * EffectComposer pass that renders the scene + MMD cel outlines (via
 * OutlineEffect) into the composer pipeline, so bloom can be applied on top.
 */
class OutlineRenderPass extends Pass {
  scene: THREE.Scene;

  camera: THREE.Camera;

  outlineEffect: OutlineEffect;

  constructor(scene: THREE.Scene, camera: THREE.Camera, outlineEffect: OutlineEffect) {
    super();
    this.scene = scene;
    this.camera = camera;
    this.outlineEffect = outlineEffect;
    this.needsSwap = false;
  }

  render(renderer: THREE.WebGLRenderer, _writeBuffer: any, readBuffer: any) {
    renderer.setRenderTarget(this.renderToScreen ? null : readBuffer);
    this.outlineEffect.render(this.scene, this.camera);
  }
}

/**
 * MMD (.pmx) viewer — drop-in alternative to the Live2D canvas, used when the
 * selected character's model URL is a `.pmx` (see App.tsx).
 *
 * - Stage 1: load + display textured model, auto-framed camera.
 * - Stage 2: Ammo.js physics (hair/skirt inertia) + procedural idle (breathing,
 *   sway, head micro-movement, blink) + a natural arms-down rest pose (MMD models
 *   load in a "scarecrow" A/T-pose). Camera uses OrbitControls locked to a level
 *   front view (pan + zoom only, no rotate) and the framing + arm pose persist in
 *   localStorage, so you adjust once and it sticks.
 *
 * Lip-sync (Stage 3) and emotion expressions (Stage 4) come next; the mouth
 * (あいうえお) and expression (笑い/にこり/困る/怒り) morphs are already present
 * in this model.
 */
interface MMDViewerProps {
  url: string;
}

const LS_VIEW = "mmdCamView";
const LS_ARMS = "mmdArmPose";
const LS_LOOK = "mmdLook16"; // bumped: user-tuned vivid-toon defaults baked from 🎨 panel
const LS_GFX = "mmdGfx2"; // bumped: SSAA 1.5× default + native (uncapped) pixel ratio

let ammoPromise: Promise<any> | null = null;

// Fetch the .wasm ourselves and hand it to the emscripten factory as
// `wasmBinary`: emscripten's internal fetch() fails under file:// (the Electron
// desktop build) and its error is swallowed into a generic abort. XHR is the
// file:// fallback (it can read local files where fetch cannot).
function fetchWasmBinary(url: string): Promise<ArrayBuffer | undefined> {
  return fetch(url)
    .then((resp) => (resp.ok ? resp.arrayBuffer() : undefined))
    .catch(
      () =>
        new Promise<ArrayBuffer | undefined>((res) => {
          const xhr = new XMLHttpRequest();
          xhr.open("GET", url);
          xhr.responseType = "arraybuffer";
          xhr.onload = () => {
            const ok = xhr.status === 200 || (xhr.status === 0 && xhr.response);
            res(ok ? xhr.response : undefined);
          };
          xhr.onerror = () => res(undefined);
          xhr.send();
        }),
    );
}

function loadAmmo(): Promise<any> {
  const w = window as any;
  if (w.Ammo && typeof w.Ammo.btVector3 === "function") return Promise.resolve(w.Ammo);
  if (!ammoPromise) {
    // Load the emscripten WASM build as a plain <script> from the static assets
    // (public/libs/ammo/). Bundling it (`import("ammo.js")`) breaks: the UMD
    // wrapper's Node-environment detection trips under Vite and the factory
    // never initializes -> physics silently fell back to OFF (stiff hair/tie).
    // Urls resolve against document.baseURI so subpath and file:// (Electron)
    // deployments hit the right location.
    const dirUrl = new URL("libs/ammo/", document.baseURI).href;
    ammoPromise = new Promise<any>((resolvePromise, reject) => {
      const script = document.createElement("script");
      script.src = `${dirUrl}ammo.wasm.js`;
      script.onload = async () => {
        const factory = typeof w.Ammo === "function" ? w.Ammo : null;
        if (!factory) {
          reject(new Error("ammo.wasm.js loaded but no Ammo() factory"));
          return;
        }
        const wasmBinary = await fetchWasmBinary(`${dirUrl}ammo.wasm.wasm`);
        factory({
          locateFile: (file: string) => `${dirUrl}${file}`,
          ...(wasmBinary ? { wasmBinary } : {}),
        }).then((AmmoLib: any) => {
          w.Ammo = AmmoLib; // MMDPhysics reads the initialized module from here
          resolvePromise(AmmoLib);
        }, reject);
      };
      script.onerror = () => reject(new Error(`failed to fetch ${dirUrl}ammo.wasm.js`));
      document.head.appendChild(script);
    }).catch((e) => {
      // A failed attempt must NOT poison future ones: a transient failure (e.g.
      // server restarting during first load) permanently disabled physics.
      ammoPromise = null;
      console.error("[MMD] ammo.js load failed — physics OFF for this attempt:", e);
      throw e;
    });
  }
  return ammoPromise;
}

function findBone(mesh: THREE.SkinnedMesh, names: string[]): THREE.Bone | null {
  for (const n of names) {
    const b = mesh.skeleton.bones.find((bb) => bb.name === n);
    if (b) return b;
  }
  return null;
}

function lsGet<T>(key: string): T | null {
  try {
    const v = localStorage.getItem(key);
    return v ? (JSON.parse(v) as T) : null;
  } catch {
    return null;
  }
}
function lsSet(key: string, val: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(val));
  } catch {
    /* ignore */
  }
}

export const MMDViewer = memo(({ url }: MMDViewerProps): JSX.Element => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;

    let disposed = false;
    let raf = 0;
    let disposePanel: (() => void) | null = null;
    let disposeLookPanel: (() => void) | null = null;
    let disposeGfxPanel: (() => void) | null = null;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      30,
      Math.max(container.clientWidth, 1) / Math.max(container.clientHeight, 1),
      0.1,
      1000,
    );
    camera.position.set(0, 16, 30);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true; // self-shadows (hair on face, body depth)
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping; // re-set by applyLookState
    renderer.toneMappingExposure = 1.1;
    container.appendChild(renderer.domElement);
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.display = "block";

    // MMD-style cel outlines (the "Edge" part of the msToon look). OutlineEffect
    // reads each material's outline params that MMDLoader imported from the PMX,
    // so it draws the outlines the model author intended.
    const outline = new OutlineEffect(renderer, { defaultThickness: 0.003 });

    // --- Lighting -----------------------------------------------------------
    // msToonCoordinator-style flat anime shading: a bright hemisphere + ambient
    // fill lifts the shadows so the MMD toon ramp reads as one clean, flat light
    // band (the cel look), a gentle key gives soft form, and two edge rims (a
    // cool one behind-left, a warm one behind-right) trace the silhouette —
    // msToon's signature outline glow — without blowing the body out. The older
    // moody music-video rig (dark scene, strong cyan/magenta rims) reuses these
    // same lights and is reachable via __mmdMoody(); __mmdMsToon() flips back.
    // Actual intensities/colours are set by applyLookState() below.
    const ambient = new THREE.AmbientLight(0xffffff, 0.45);
    scene.add(ambient);
    const hemi = new THREE.HemisphereLight(0xffffff, 0x9aa0c8, 0.9);
    scene.add(hemi);
    const keyLight = new THREE.DirectionalLight(0xfff4ea, 0.95);
    keyLight.position.set(5, 18, 18);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(1024, 1024);
    keyLight.shadow.camera.near = 0.5;
    keyLight.shadow.camera.far = 100;
    keyLight.shadow.camera.left = -25;
    keyLight.shadow.camera.right = 25;
    keyLight.shadow.camera.top = 32;
    keyLight.shadow.camera.bottom = -6;
    keyLight.shadow.bias = -0.002;
    keyLight.shadow.radius = 4; // soft edges (PCFSoft)
    scene.add(keyLight);
    // True edge rims: both placed BEHIND the model (negative z, away from the
    // camera) so they light only the silhouette edges that wrap toward camera —
    // the body interior stays dark (low-key). Cyan from behind-upper-left,
    // violet-magenta from behind-right. (Front-placed rims flooded the whole
    // body with light, which is what made it "光ってる".)
    const rimCool = new THREE.DirectionalLight(0x5fe0f0, 0.9);
    rimCool.position.set(-11, 13, -16);
    scene.add(rimCool);
    const rimWarm = new THREE.DirectionalLight(0xc25fda, 0.45);
    rimWarm.position.set(12, 9, -15);
    scene.add(rimWarm);

    // Radial-gradient backdrop (lighter at the centre, darker at the edges),
    // redrawn by setBgGrad so the look presets can swap the stage colour. A
    // CanvasTexture is cheap and gives a clean gradient.
    const bgCanvas = document.createElement("canvas");
    bgCanvas.width = 256;
    bgCanvas.height = 256;
    const bgCtx = bgCanvas.getContext("2d")!;
    const bgTex = new THREE.CanvasTexture(bgCanvas);
    bgTex.colorSpace = THREE.SRGBColorSpace;
    const setBgGrad = (inner: string, outer: string) => {
      const g = bgCtx.createRadialGradient(128, 104, 16, 128, 128, 188);
      g.addColorStop(0, inner);
      g.addColorStop(1, outer);
      bgCtx.fillStyle = g;
      bgCtx.fillRect(0, 0, 256, 256);
      bgTex.needsUpdate = true;
    };
    setBgGrad("#bcc2e4", "#6b7099"); // msToon studio (re-set by applyLookState)
    scene.background = bgTex;

    // Post-processing: scene + outline -> bloom (glow) -> output. Bloom is the
    // defining element of the video's stage-lit look.
    // Anti-aliasing: the renderer's own `antialias: true` does NOTHING once we
    // render through EffectComposer — MSAA only applies to the default
    // framebuffer, not the offscreen render targets the composer draws into. So
    // we hand the composer a MULTISAMPLED HDR render target: THIS is what
    // actually anti-aliases the geometry + cel-outline edges (the main source of
    // the jaggies). The final SMAA pass then cleans the post-processed / shaded
    // edges that MSAA can't see. Tunable live with __mmdAA(samples).
    const maxSamples = renderer.capabilities.maxSamples || 8;
    let aaSamples = Math.min(4, maxSamples); // currently-applied MSAA samples (mirrors gfx.aa)
    const makeComposerRT = (samples: number) =>
      new THREE.WebGLRenderTarget(
        Math.max(1, container.clientWidth),
        Math.max(1, container.clientHeight),
        { type: THREE.HalfFloatType, samples },
      );
    const composer = new EffectComposer(renderer, makeComposerRT(aaSamples));
    // Render the post-FX chain at full device resolution (not CSS pixels), so it
    // stays sharp on HiDPI / forwarded displays instead of being upscaled.
    composer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    composer.setSize(container.clientWidth, container.clientHeight);
    composer.addPass(new OutlineRenderPass(scene, camera, outline));
    // Cel re-shade (msToon-spirit posterize) before bloom so the glow blooms the
    // flat-shaded image.
    const toonPass = new ShaderPass(ToonShader);
    composer.addPass(toonPass);
    // Depth of field — blurs the far hair/edges by distance (cinematic bokeh).
    // It blurs the already-composed image (readBuffer) using the scene depth, so
    // it sits after the toon pass. Focus tracks the model each frame (animate).
    const bokehPass = new BokehPass(scene, camera, { focus: 30, aperture: 0, maxblur: 0.01 });
    composer.addPass(bokehPass);
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(container.clientWidth, container.clientHeight),
      0.55, // strength (subtle glow, not blown out)
      0.6, // radius
      0.5, // threshold (the bright cyan/magenta rims glow; the dark body doesn't)
    );
    composer.addPass(bloomPass);
    composer.addPass(new OutputPass());
    // SMAA anti-aliasing on the final image — cleans the outline/edge jaggies the
    // post-processing pipeline introduces (the video has clean, smooth edges).
    composer.addPass(new SMAAPass(container.clientWidth, container.clientHeight));

    // Outline (edge line) thickness. MMDLoader imports each material's PMX edge
    // size; we capture those base values and can scale them — the reference
    // video's contour lines are noticeably bolder than the model's defaults.
    let outlineBaseMap: Map<any, number> | null = null;
    let outlineScale = 1.8;
    const applyOutlineScale = () => {
      if (!outlineBaseMap) return;
      outlineBaseMap.forEach((base, mat) => {
        const op = mat.userData?.outlineParameters;
        if (op) op.thickness = base * outlineScale;
      });
    };

    // Skin materials (body01 / face01 / face02 on this model). We keep them matte
    // (no plastic specular from the strong rims) and add a warm emissive floor so
    // the skin reads as natural warm anime skin with warm shadows, instead of
    // being tinted cold by the cyan rim + cool grade ("肌にサブライトを当てない").
    let skinMaterials: any[] = [];
    const applySkinWarmth = (warmth: number) => {
      // Soft, balanced warm (not orange-red) so the skin reads natural, not flushed.
      skinMaterials.forEach((m) => {
        if (m.emissive) m.emissive.setRGB(0.6 * warmth, 0.47 * warmth, 0.42 * warmth);
      });
    };

    // --- Hard cel shading (the missing piece) -------------------------------
    // MMDToonMaterial shades via a `gradientMap` toon ramp. This model's ramp is
    // soft (smooth, 3D-looking). We swap in a HARD stepped ramp so the lighting
    // quantises into a few flat bands with crisp boundaries — real anime cel
    // shading — affecting only the shading, never the background. NearestFilter
    // on an N-wide texture gives N hard bands (U = N·L mapped across it).
    const rampCache = new Map<number, THREE.DataTexture>();
    const makeToonRamp = (bands: number): THREE.DataTexture => {
      const cached = rampCache.get(bands);
      if (cached) return cached;
      const w = Math.max(2, bands);
      const data = new Uint8Array(w * 4);
      for (let i = 0; i < w; i += 1) {
        const v = 0.38 + 0.62 * (i / (w - 1)); // deeper darkest shadow band .. full light (vivid-toon contrast)
        const c = Math.round(v * 255);
        data[i * 4] = c;
        data[i * 4 + 1] = c;
        data[i * 4 + 2] = c;
        data[i * 4 + 3] = 255;
      }
      const tex = new THREE.DataTexture(data, w, 1, THREE.RGBAFormat);
      tex.magFilter = THREE.NearestFilter;
      tex.minFilter = THREE.NearestFilter;
      tex.generateMipmaps = false;
      tex.needsUpdate = true;
      rampCache.set(bands, tex);
      return tex;
    };
    let celMaterials: any[] = [];
    const celOriginalRamp = new Map<any, any>();
    const applyCel = (bands: number) => {
      celMaterials.forEach((m) => {
        m.gradientMap = bands >= 2 ? makeToonRamp(bands) : (celOriginalRamp.get(m) ?? null);
        m.needsUpdate = true;
      });
    };

    // --- Fresnel rim + non-mixing sublight layer ----------------------------
    // Injected into each MMDToon material's shader at the opaque_fragment (where
    // `normal`, `vViewPosition` and `outgoingLight` are in scope). Two effects:
    //   1) Fresnel rim — an even, view-dependent edge light, bright at grazing
    //      angles (silhouette) regardless of light position.
    //   2) Dual sublight — msToonCoordinator's separate "sublight" layer: a cyan
    //      and a magenta edge light, each gated to the side it faces. They are
    //      composited NON-additively (only the dominant sublight is added per
    //      fragment), so cyan + magenta never sum to white — the reference's
    //      "陰影・ハイライト・サブライトの色が混ざらない". uSubMix lerps back to
    //      the old additive behaviour (1 = additive, 0 = colours don't mix).
    const injectRim = (m: any) => {
      if (!m || m._fresInjected || typeof m.fragmentShader !== "string") return;
      m.uniforms = m.uniforms || {};
      m.uniforms.uFresInt = { value: 0 };
      m.uniforms.uFresPow = { value: 3.5 };
      m.uniforms.uFresColor = { value: new THREE.Color(0xbfe6ff) };
      m.uniforms.uSubInt = { value: 0 };
      m.uniforms.uSubPow = { value: 3.0 };
      m.uniforms.uSubMix = { value: 0 };
      m.uniforms.uSubCoolColor = { value: new THREE.Color(0x40e3f5) };
      m.uniforms.uSubWarmColor = { value: new THREE.Color(0xf24fd0) };
      m.uniforms.uSubCoolDir = { value: new THREE.Vector3(-1, 0, -1).normalize() };
      m.uniforms.uSubWarmDir = { value: new THREE.Vector3(1, 0, -1).normalize() };
      m.fragmentShader = m.fragmentShader
        .replace(
          "#include <common>",
          "uniform float uFresInt;\nuniform float uFresPow;\nuniform vec3 uFresColor;\n" +
            "uniform float uSubInt;\nuniform float uSubPow;\nuniform float uSubMix;\n" +
            "uniform vec3 uSubCoolColor;\nuniform vec3 uSubWarmColor;\n" +
            "uniform vec3 uSubCoolDir;\nuniform vec3 uSubWarmDir;\n#include <common>",
        )
        .replace(
          "#include <opaque_fragment>",
          [
            "{",
            "  vec3 rimN = normalize( normal );",
            "  vec3 rimV = normalize( vViewPosition );",
            "  float rimEdge = clamp( 1.0 - abs( dot( rimN, rimV ) ), 0.0, 1.0 );",
            "  outgoingLight += pow( rimEdge, uFresPow ) * uFresInt * uFresColor;",
            "  if ( uSubInt > 0.001 ) {",
            "    float rimF = pow( rimEdge, uSubPow );",
            "    vec3 coolC = uSubCoolColor * ( rimF * clamp( dot( rimN, uSubCoolDir ), 0.0, 1.0 ) * uSubInt );",
            "    vec3 warmC = uSubWarmColor * ( rimF * clamp( dot( rimN, uSubWarmDir ), 0.0, 1.0 ) * uSubInt );",
            "    float lc = dot( coolC, vec3( 0.299, 0.587, 0.114 ) );",
            "    float lw = dot( warmC, vec3( 0.299, 0.587, 0.114 ) );",
            "    vec3 dom = lc >= lw ? coolC : warmC;", // non-mixing: dominant sublight only
            "    outgoingLight += mix( dom, coolC + warmC, uSubMix );",
            "  }",
            "}",
            "#include <opaque_fragment>",
          ].join("\n"),
        );
      m.needsUpdate = true;
      m._fresInjected = true;
    };
    const applyFresnel = (intensity: number, power: number, colorHex: number) => {
      celMaterials.forEach((m) => {
        injectRim(m);
        if (m.uniforms?.uFresInt) {
          m.uniforms.uFresInt.value = intensity;
          m.uniforms.uFresPow.value = power;
          m.uniforms.uFresColor.value.set(colorHex);
        }
      });
    };
    // Drive the dual non-mixing sublight (cyan + magenta edge accents). Colours
    // come from the shared rim colours so the 🎨 panel's リム寒色/暖色 steer both
    // the directional rims and this sublight layer.
    const applySublight = (
      intensity: number,
      power: number,
      mix: number,
      coolHex: number,
      warmHex: number,
    ) => {
      celMaterials.forEach((m) => {
        injectRim(m);
        if (m.uniforms?.uSubInt) {
          m.uniforms.uSubInt.value = intensity;
          m.uniforms.uSubPow.value = power;
          m.uniforms.uSubMix.value = mix;
          m.uniforms.uSubCoolColor.value.set(coolHex);
          m.uniforms.uSubWarmColor.value.set(warmHex);
        }
      });
    };
    // The shader's `normal` is in VIEW space, so the sublight directions must be
    // re-derived from the rim azimuth/elevation + the current camera each frame.
    // Reuses dirToPos + the shared `look` directions (both defined below).
    const _subCoolW = new THREE.Vector3();
    const _subWarmW = new THREE.Vector3();
    const updateSublightDirs = () => {
      if (!celMaterials.length) return;
      dirToPos(look.rimCoolAzim, look.rimCoolElev, _subCoolW);
      _subCoolW.transformDirection(camera.matrixWorldInverse); // -> unit view-space dir
      dirToPos(look.rimWarmAzim, look.rimWarmElev, _subWarmW);
      _subWarmW.transformDirection(camera.matrixWorldInverse);
      celMaterials.forEach((m) => {
        if (m.uniforms?.uSubCoolDir) {
          m.uniforms.uSubCoolDir.value.copy(_subCoolW);
          m.uniforms.uSubWarmDir.value.copy(_subWarmW);
        }
      });
    };

    // --- Look state + presets ----------------------------------------------
    // The whole visual rig (light intensities + colours, bloom, cel grade,
    // backdrop) lives in one `look` object so it can be driven from the
    // on-screen 🎨 panel and console knobs, and persisted as a unit.
    // "msToon" (default): bright, flat, clean anime cel — even fill so the toon
    // ramp reads as one light band, soft key, gentle cool/warm edge rims,
    // minimal bloom, light studio backdrop. "moody": the dark, cyan/magenta
    // rim-lit music-video look with strong bloom.
    type LookState = {
      ambient: number; hemi: number; key: number; rimCool: number; rimWarm: number;
      bloomStrength: number; bloomRadius: number; bloomThreshold: number;
      toonSteps: number; toonSat: number; toonContrast: number;
      toonChroma: number; toonTemp: number; toonTint: number; toonSplit: number;
      vignette: number; grain: number; bars: number;
      skinWarmth: number; // warm emissive floor on the skin materials
      celBands: number; // hard toon-ramp bands (0/1 = off/original soft ramp)
      shadowStrength: number; // cast-shadow darkness (0 = off)
      toneMode: number; // 0 none, 1 ACES, 2 AgX, 3 Neutral
      toneExposure: number; // tone-mapping exposure
      dofAperture: number; // depth-of-field strength (0 = off)
      dofBlur: number; // depth-of-field max blur
      fresnelInt: number; // fresnel rim intensity (0 = off)
      fresnelPow: number; // fresnel falloff (higher = tighter edge)
      fresnelColor: number; // fresnel rim colour
      subInt: number; // non-mixing sublight intensity (cyan+magenta edge, 0 = off)
      subPow: number; // sublight rim falloff (higher = tighter edge)
      subMix: number; // 0 = colours don't mix (dominant only), 1 = additive
      // Rim light DIRECTIONS as azimuth (left/right angle, deg) + elevation
      // (height, deg). 0 az = front, ±180 = behind, +90 = camera-right.
      rimCoolAzim: number; rimCoolElev: number; rimWarmAzim: number; rimWarmElev: number;
      ambientColor: number; keyColor: number; rimCoolColor: number; rimWarmColor: number;
      bgInner: number; bgOuter: number;
    };
    // "moody" (default) is tuned from the reference music video (ZoaQPuYKclI):
    // dark cool body, a strong cyan front rim + violet-magenta back rim, soft
    // bloom, a cool teal/magenta split-tone grade and chromatic-aberration edge
    // fringing, on an indigo radial backdrop. "msToon" is the bright flat cel.
    const LOOK_PRESETS: Record<"msToon" | "moody", LookState> = {
      msToon: {
        ambient: 0.45, hemi: 0.9, key: 0.95, rimCool: 0.9, rimWarm: 0.45,
        bloomStrength: 0.2, bloomRadius: 0.5, bloomThreshold: 0.86,
        toonSteps: 0, toonSat: 1.14, toonContrast: 1.05,
        toonChroma: 0, toonTemp: 0, toonTint: 0, toonSplit: 0,
        vignette: 0.15, grain: 0, bars: 0,
        skinWarmth: 0, celBands: 0,
        shadowStrength: 0.25, toneMode: 1, toneExposure: 1.05,
        dofAperture: 0, dofBlur: 0.01, fresnelInt: 0, fresnelPow: 3.5, fresnelColor: 0xbfe6ff,
        subInt: 0, subPow: 3.0, subMix: 0,
        rimCoolAzim: -130, rimCoolElev: 30, rimWarmAzim: 130, rimWarmElev: 25,
        ambientColor: 0xffffff, keyColor: 0xfff4ea, rimCoolColor: 0xbfe6ff, rimWarmColor: 0xffd4ec,
        bgInner: 0xbcc2e4, bgOuter: 0x6b7099,
      },
      moody: {
        // User-tuned vivid-toon defaults, baked from the 🎨 panel. No-blur
        // direction: DOF off, bloom near-off, chromatic aberration off — the
        // look is carried by high contrast + saturated non-mixing cyan/magenta
        // sublights. (半球光 / 青光高さ kept at the prior defaults; colours
        // unchanged from the prior tuning.)
        ambient: 0.15, hemi: 0.07, key: 0.42, rimCool: 2.05, rimWarm: 1.75,
        bloomStrength: 0.08, bloomRadius: 0.3, bloomThreshold: 0.84,
        toonSteps: 0, toonSat: 1.54, toonContrast: 1.66,
        toonChroma: 0, toonTemp: -0.25, toonTint: 0, toonSplit: 0.05,
        vignette: 0.75, grain: 0, bars: 0,
        skinWarmth: 0.2, celBands: 2,
        shadowStrength: 0.5, toneMode: 1, toneExposure: 1.08,
        dofAperture: 0, dofBlur: 0.008, fresnelInt: 0.18, fresnelPow: 3.4, fresnelColor: 0x9fe6ff,
        subInt: 1.3, subPow: 3.5, subMix: 0.2,
        rimCoolAzim: -145, rimCoolElev: 20, rimWarmAzim: 145, rimWarmElev: 20,
        ambientColor: 0x1e1940, keyColor: 0xf3e0cf, rimCoolColor: 0x40e3f5, rimWarmColor: 0xf24fd0,
        bgInner: 0x2a2550, bgOuter: 0x060410,
      },
    };
    const DEFAULT_PRESET: "msToon" | "moody" = "moody";
    const hexStr = (n: number) => `#${(n & 0xffffff).toString(16).padStart(6, "0")}`;
    // Convert a light direction (azimuth left/right + elevation height, degrees)
    // into a world position. Directional lights aim at the origin, so only the
    // direction matters; radius is fixed. az 0 = front (+z), +90 = camera-right
    // (+x), ±180 = behind; elev raises the light.
    const dirToPos = (azimDeg: number, elevDeg: number, out: THREE.Vector3) => {
      const a = (azimDeg * Math.PI) / 180;
      const e = (elevDeg * Math.PI) / 180;
      const R = 22;
      out.set(Math.sin(a) * Math.cos(e), Math.sin(e), Math.cos(a) * Math.cos(e)).multiplyScalar(R);
    };
    let look: LookState = { ...LOOK_PRESETS[DEFAULT_PRESET] };
    const applyLookState = () => {
      ambient.color.set(look.ambientColor);
      ambient.intensity = look.ambient;
      hemi.intensity = look.hemi;
      keyLight.color.set(look.keyColor);
      keyLight.intensity = look.key;
      rimCool.color.set(look.rimCoolColor);
      rimCool.intensity = look.rimCool;
      dirToPos(look.rimCoolAzim, look.rimCoolElev, rimCool.position);
      rimWarm.color.set(look.rimWarmColor);
      rimWarm.intensity = look.rimWarm;
      dirToPos(look.rimWarmAzim, look.rimWarmElev, rimWarm.position);
      bloomPass.strength = look.bloomStrength;
      bloomPass.radius = look.bloomRadius;
      bloomPass.threshold = look.bloomThreshold;
      toonPass.uniforms.uSteps.value = look.toonSteps;
      toonPass.uniforms.uSat.value = look.toonSat;
      toonPass.uniforms.uContrast.value = look.toonContrast;
      toonPass.uniforms.uChroma.value = look.toonChroma;
      toonPass.uniforms.uTemp.value = look.toonTemp;
      toonPass.uniforms.uTint.value = look.toonTint;
      toonPass.uniforms.uSplit.value = look.toonSplit;
      toonPass.uniforms.uVignette.value = look.vignette;
      toonPass.uniforms.uGrain.value = look.grain;
      toonPass.uniforms.uBars.value = look.bars;
      applySkinWarmth(look.skinWarmth);
      applyCel(look.celBands);
      applyFresnel(look.fresnelInt, look.fresnelPow, look.fresnelColor);
      applySublight(look.subInt, look.subPow, look.subMix, look.rimCoolColor, look.rimWarmColor);
      updateSublightDirs();
      // Tone mapping (applied by OutputPass from renderer settings).
      const TM = [
        THREE.NoToneMapping,
        THREE.ACESFilmicToneMapping,
        THREE.AgXToneMapping,
        THREE.NeutralToneMapping,
      ];
      renderer.toneMapping = TM[look.toneMode] ?? THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = look.toneExposure;
      // Cast-shadow darkness.
      keyLight.castShadow = look.shadowStrength > 0.001;
      if ((keyLight.shadow as any).intensity !== undefined) {
        (keyLight.shadow as any).intensity = look.shadowStrength;
      }
      // Depth of field (skip the pass entirely when off, to save a depth render).
      bokehPass.uniforms.aperture.value = look.dofAperture;
      bokehPass.uniforms.maxblur.value = look.dofBlur;
      bokehPass.enabled = look.dofAperture > 0.00001;
      setBgGrad(hexStr(look.bgInner), hexStr(look.bgOuter));
    };
    const saveLook = () => lsSet(LS_LOOK, look);
    const setLookPreset = (preset: "msToon" | "moody") => {
      look = { ...LOOK_PRESETS[preset] };
      applyLookState();
      saveLook();
    };
    // Restore any saved tweaks over the msToon defaults (old string-form values
    // from before are ignored, falling back to msToon).
    const savedLook = lsGet<Partial<LookState>>(LS_LOOK);
    if (savedLook && typeof savedLook === "object") {
      look = { ...LOOK_PRESETS[DEFAULT_PRESET], ...savedLook };
    }
    applyLookState();

    // Orbit controls: left-drag rotates the camera angle, right-drag pans,
    // wheel zooms. The resulting view (angle + position + zoom) persists.
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableRotate = true;
    controls.enablePan = true;
    controls.screenSpacePanning = true;
    controls.enableZoom = true;
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN,
    };
    controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
    const saveView = () => {
      lsSet(LS_VIEW, {
        p: [camera.position.x, camera.position.y, camera.position.z],
        t: [controls.target.x, controls.target.y, controls.target.z],
      });
    };
    controls.addEventListener("end", saveView);

    let mesh: THREE.SkinnedMesh | null = null;
    let currentCoverage = 1.0;
    const modelBox = new THREE.Box3();

    /** Level full-body framing: camera & target share a Y, so the view never tilts. */
    const frame = (coverage = 1.0) => {
      if (!mesh) return;
      modelBox.setFromObject(mesh);
      const size = new THREE.Vector3();
      const center = new THREE.Vector3();
      modelBox.getSize(size);
      modelBox.getCenter(center);
      const margin = 1.12;
      const visibleH = size.y * coverage * margin;
      const topY = modelBox.max.y + size.y * 0.03;
      const centerY = topY - visibleH / 2;
      const fovRad = (camera.fov * Math.PI) / 180;
      const dist = visibleH / 2 / Math.tan(fovRad / 2);
      controls.target.set(center.x, centerY, center.z);
      camera.position.set(center.x, centerY, modelBox.max.z + dist);
      camera.updateProjectionMatrix();
      controls.update();
    };

    const applySavedViewOrFrame = () => {
      const v = lsGet<{ p: number[]; t: number[] }>(LS_VIEW);
      if (v?.p && v?.t) {
        camera.position.set(v.p[0], v.p[1], v.p[2]);
        controls.target.set(v.t[0], v.t[1], v.t[2]);
        controls.update();
      } else {
        frame(currentCoverage);
      }
    };

    const tmpEuler = new THREE.Euler();
    const tmpQuat = new THREE.Quaternion();
    const setBoneOffset = (bone: THREE.Bone | null, base: THREE.Quaternion, rx: number, ry: number, rz: number) => {
      if (!bone) return;
      tmpEuler.set(rx, ry, rz, "XYZ");
      tmpQuat.setFromEuler(tmpEuler);
      bone.quaternion.copy(base).multiply(tmpQuat);
    };

    // --- Arms-down rest pose -------------------------------------------------
    // MMD models load in an A/T ("scarecrow") pose. Rather than guess the bone's
    // local rotation axis (which differs per model and made the arms go UP), aim
    // each arm/forearm bone in WORLD space so its child (elbow→wrist) points
    // straight down with a slight outward/forward lean — a natural idle, like
    // Rem's. Robust regardless of the bone's local frame.
    const armRest = new Map<THREE.Bone, THREE.Quaternion>();
    let armBones: Record<string, THREE.Bone | null> | null = null;
    let modelCenterX = 0;
    const savedArm = lsGet<any>(LS_ARMS);
    let armCfg =
      savedArm && typeof savedArm.out === "number"
        ? { out: savedArm.out, fwd: typeof savedArm.fwd === "number" ? savedArm.fwd : 0.05 }
        : { out: 0.18, fwd: 0.05 };

    const aimBoneDown = (bone: THREE.Bone, child: THREE.Bone, outwardSign: number) => {
      bone.updateWorldMatrix(true, false);
      child.updateWorldMatrix(true, false);
      const bp = new THREE.Vector3().setFromMatrixPosition(bone.matrixWorld);
      const cp = new THREE.Vector3().setFromMatrixPosition(child.matrixWorld);
      const cur = cp.sub(bp).normalize();
      const tgt = new THREE.Vector3(outwardSign * armCfg.out, -1, armCfg.fwd).normalize();
      const qWorld = new THREE.Quaternion().setFromUnitVectors(cur, tgt);
      const curWorld = new THREE.Quaternion();
      bone.getWorldQuaternion(curWorld);
      const newWorld = qWorld.multiply(curWorld);
      const parentWorld = new THREE.Quaternion();
      if (bone.parent) bone.parent.getWorldQuaternion(parentWorld);
      bone.quaternion.copy(parentWorld.invert().multiply(newWorld));
      bone.updateWorldMatrix(true, true);
    };

    const applyArmPose = (m: THREE.SkinnedMesh) => {
      if (!armBones) {
        armBones = {
          la: findBone(m, ["左腕"]),
          ra: findBone(m, ["右腕"]),
          le: findBone(m, ["左ひじ"]),
          re: findBone(m, ["右ひじ"]),
          lw: findBone(m, ["左手首"]),
          rw: findBone(m, ["右手首"]),
        };
        [armBones.la, armBones.ra, armBones.le, armBones.re].forEach((b) => {
          if (b) armRest.set(b, b.quaternion.clone());
        });
        const box = new THREE.Box3().setFromObject(m);
        modelCenterX = (box.min.x + box.max.x) / 2;
      }
      // Restore rest, then aim from a clean state (so re-tuning is repeatable).
      armRest.forEach((q, b) => b.quaternion.copy(q));
      m.updateMatrixWorld(true);
      const { la, ra, le, re, lw, rw } = armBones;
      const signOf = (b: THREE.Bone) =>
        new THREE.Vector3().setFromMatrixPosition(b.matrixWorld).x >= modelCenterX ? 1 : -1;
      if (la && le) aimBoneDown(la, le, signOf(la));
      if (le && lw) aimBoneDown(le, lw, signOf(le));
      if (ra && re) aimBoneDown(ra, re, signOf(ra));
      if (re && rw) aimBoneDown(re, rw, signOf(re));
    };

    // --- Physics + procedural idle ------------------------------------------
    let helper: MMDAnimationHelper | null = null;
    let animReady = false;
    const clock = new THREE.Clock();
    let elapsed = 0;

    type IdleBone = { bone: THREE.Bone; base: THREE.Quaternion };
    let bUpper: IdleBone | null = null;
    let bUpper2: IdleBone | null = null;
    let bNeck: IdleBone | null = null;
    let bHead: IdleBone | null = null;
    let blinkIdx = -1;
    const capture = (bone: THREE.Bone | null): IdleBone | null =>
      bone ? { bone, base: bone.quaternion.clone() } : null;

    const setupIdle = (m: THREE.SkinnedMesh) => {
      bUpper = capture(findBone(m, ["上半身"]));
      bUpper2 = capture(findBone(m, ["上半身2"]));
      bNeck = capture(findBone(m, ["首"]));
      bHead = capture(findBone(m, ["頭"]));
      const dict = m.morphTargetDictionary || {};
      const blinkKey = Object.keys(dict).find((k) => /まばたき|またたき|瞬き|blink/i.test(k));
      blinkIdx = blinkKey != null ? dict[blinkKey] : -1;
    };

    const setIdleOffset = (ib: IdleBone | null, rx: number, ry: number, rz: number) => {
      if (ib) setBoneOffset(ib.bone, ib.base, rx, ry, rz);
    };

    // `gated` = a vmd clip is driving the body, so skip the procedural body/head
    // bone writes (the clip owns those bones — single writer per bone). Blink is
    // kept: retargeted HY-Motion clips animate bones only, not the blink morph.
    const updateIdle = (m: THREE.SkinnedMesh, gated = false) => {
      const t = elapsed;
      if (!gated) {
        const breath = Math.sin(t * 1.5) * 0.014;
        const sway = Math.sin(t * 0.6) * 0.012;
        setIdleOffset(bUpper, breath, 0, sway);
        setIdleOffset(bUpper2, breath * 0.5, 0, sway * 0.5);
        const hy = Math.sin(t * 0.45) * 0.06 + Math.sin(t * 0.23) * 0.03;
        const hx = Math.sin(t * 0.37) * 0.025;
        setIdleOffset(bNeck, hx * 0.4, hy * 0.4, sway * 0.3);
        setIdleOffset(bHead, hx, hy, 0);
      }
      if (blinkIdx >= 0 && m.morphTargetInfluences) {
        const ph = t % 4.5;
        m.morphTargetInfluences[blinkIdx] = ph < 0.18 ? Math.sin((ph / 0.18) * Math.PI) : 0;
      }
    };

    // Arm/hand rigid bodies vs the (densely rigged, 108-body) skirt: contact
    // makes the skirt spaz out. Drop ONLY the skirt group (9) from the arm-side
    // collision masks before physics init — skirt still collides with the body
    // and legs, but a passing arm no longer detonates it.
    const patchRigidBodyMasks = (m: THREE.SkinnedMesh) => {
      const mmd = (m.geometry as any)?.userData?.MMD;
      if (!mmd?.rigidBodies) return;
      const armBodies = new Set(["右腕", "右ひじ", "右手首", "左腕", "左ひじ", "左手首"]);
      const SKIRT_GROUP = 9;
      let patched = 0;
      mmd.rigidBodies.forEach((r: any) => {
        if (armBodies.has(r.name)) {
          r.groupTarget &= ~(1 << SKIRT_GROUP);
          patched += 1;
        }
      });
      if (patched) console.log(`[MMD] physics: arm<->skirt collision off (${patched} bodies)`);
    };

    const initPhysics = () => {
      loadAmmo()
        .then(() => {
          if (disposed || !mesh) return;
          try {
            patchRigidBodyMasks(mesh);
            helper = new MMDAnimationHelper();
            helper.add(mesh, { physics: true });
          } catch (e) {
            console.error("[MMD] physics init failed (idle only):", e);
            helper = null;
          }
          setupIdle(mesh);
          animReady = true;
          console.log("[MMD] ready (physics:", !!helper, ")");
        })
        .catch((e) => {
          console.error("[MMD] Ammo load failed (idle only):", e);
          if (mesh) {
            setupIdle(mesh);
            animReady = true;
          }
        });
    };

    // On-screen slider panel to tweak every face morph (eye size, etc.) live.
    // Toggle with the 🎭 button; changes apply to the resting face and persist.
    const buildMorphPanel = (m: THREE.SkinnedMesh): (() => void) => {
      const dict = m.morphTargetDictionary || {};
      const auto = new Set(["あ", "い", "う", "え", "お", "まばたき"]); // lip-sync / blink
      const base = mmdExpression.getBase();

      const panel = document.createElement("div");
      panel.className = "mmd-panel";
      panel.style.cssText =
        "position:absolute;top:40px;right:8px;max-height:78%;overflow-y:auto;width:240px;" +
        "background:rgba(18,18,26,0.9);color:#eee;font:12px sans-serif;padding:8px 10px;" +
        "border-radius:8px;z-index:50;display:none;box-shadow:0 4px 16px rgba(0,0,0,.5)";

      const head = document.createElement("div");
      head.style.cssText =
        "font-weight:bold;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center";
      const heatTitle = document.createElement("span");
      heatTitle.textContent = "表情調整";
      const resetBtn = document.createElement("button");
      resetBtn.textContent = "Reset";
      resetBtn.style.cssText =
        "font-size:11px;cursor:pointer;background:#444;color:#fff;border:none;border-radius:4px;padding:2px 8px";
      head.appendChild(heatTitle);
      head.appendChild(resetBtn);
      panel.appendChild(head);

      const sliders: Record<string, HTMLInputElement> = {};
      Object.keys(dict)
        .sort((a, b) => dict[a] - dict[b])
        .forEach((name) => {
          if (auto.has(name)) return; // skip morphs driven by lip-sync / blink
          const row = document.createElement("label");
          row.style.cssText = "display:flex;align-items:center;gap:6px;margin:2px 0";
          const lab = document.createElement("span");
          lab.textContent = name;
          lab.title = name;
          lab.style.cssText =
            "flex:0 0 84px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
          const sl = document.createElement("input");
          sl.type = "range";
          sl.min = "0";
          sl.max = "1";
          sl.step = "0.05";
          sl.value = String(base[name] ?? 0);
          sl.style.cssText = "flex:1;min-width:0";
          const val = document.createElement("span");
          val.textContent = sl.value;
          val.style.cssText = "flex:0 0 28px;text-align:right;opacity:.7";
          sl.addEventListener("input", () => {
            mmdExpression.setBase(name, parseFloat(sl.value));
            val.textContent = sl.value;
          });
          sliders[name] = sl;
          row.appendChild(lab);
          row.appendChild(sl);
          row.appendChild(val);
          panel.appendChild(row);
        });

      resetBtn.addEventListener("click", () => {
        const nb = mmdExpression.resetBase();
        Object.keys(sliders).forEach((n) => {
          sliders[n].value = String(nb[n] ?? 0);
          // Sync each slider's stored value + label back to the default.
          sliders[n].dispatchEvent(new Event("input"));
        });
      });

      const toggle = document.createElement("button");
      toggle.textContent = "🎭";
      toggle.title = "表情調整パネル";
      toggle.style.cssText =
        "position:absolute;top:8px;right:8px;z-index:51;cursor:pointer;font-size:16px;" +
        "background:rgba(18,18,26,0.75);border:none;border-radius:6px;padding:2px 7px";
      toggle.addEventListener("click", () => {
        const open = panel.style.display === "none";
        container.querySelectorAll(".mmd-panel").forEach((p) => {
          (p as HTMLElement).style.display = "none";
        });
        panel.style.display = open ? "block" : "none";
      });

      container.appendChild(panel);
      container.appendChild(toggle);
      return () => {
        panel.remove();
        toggle.remove();
      };
    };

    // On-screen visual-tuning panel (🎨 button): sliders for the lights / bloom
    // / cel grade and colour pickers for the light & backdrop colours, plus
    // msToon / moody preset buttons. Edits apply live and persist.
    const buildLookPanel = (): (() => void) => {
      const SLIDERS: [keyof LookState, string, number, number, number][] = [
        ["key", "キー光", 0, 2, 0.05],
        ["rimCool", "リム(寒)", 0, 2.5, 0.05],
        ["rimWarm", "リム(暖)", 0, 2.5, 0.05],
        ["ambient", "環境光", 0, 1.5, 0.05],
        ["hemi", "半球光", 0, 1.5, 0.05],
        ["bloomStrength", "Bloom強", 0, 1.5, 0.05],
        ["bloomRadius", "Bloom半径", 0, 1.5, 0.05],
        ["bloomThreshold", "Bloom閾値", 0, 1, 0.02],
        ["celBands", "セル塗り段数", 0, 5, 1],
        ["toonSteps", "粗さ(全体)", 0, 10, 1],
        ["toonSat", "彩度", 0.5, 2, 0.02],
        ["toonContrast", "コントラスト", 0.5, 2, 0.02],
        ["toonChroma", "色収差", 0, 4, 0.1],
        ["toonTemp", "色温度", -1, 1, 0.05],
        ["toonTint", "ティント", -1, 1, 0.05],
        ["toonSplit", "スプリット", 0, 1, 0.05],
        ["vignette", "周辺減光", 0, 1, 0.05],
        ["grain", "フィルム粒子", 0, 0.3, 0.01],
        ["bars", "上下黒帯", 0, 0.2, 0.01],
        ["skinWarmth", "肌の暖かさ", 0, 0.8, 0.02],
        ["rimCoolAzim", "青光 左右", -180, 180, 5],
        ["rimCoolElev", "青光 高さ", -20, 85, 5],
        ["rimWarmAzim", "桃光 左右", -180, 180, 5],
        ["rimWarmElev", "桃光 高さ", -20, 85, 5],
        ["shadowStrength", "落ち影", 0, 1, 0.05],
        ["toneMode", "トーン0-3", 0, 3, 1],
        ["toneExposure", "露出", 0.3, 2, 0.05],
        ["dofAperture", "DOF量", 0, 0.004, 0.0002],
        ["dofBlur", "DOFボケ", 0, 0.02, 0.001],
        ["fresnelInt", "フレネル", 0, 1, 0.02],
        ["fresnelPow", "フレネル幅", 0.5, 8, 0.5],
        ["subInt", "サブ光", 0, 2, 0.05],
        ["subPow", "サブ光幅", 0.5, 8, 0.5],
        ["subMix", "サブ光混合", 0, 1, 0.05],
      ];
      const COLORS: [keyof LookState, string][] = [
        ["fresnelColor", "フレネル色"],
        ["ambientColor", "環境色"],
        ["keyColor", "キー色"],
        ["rimCoolColor", "リム寒色"],
        ["rimWarmColor", "リム暖色"],
        ["bgInner", "背景(内)"],
        ["bgOuter", "背景(外)"],
      ];

      const panel = document.createElement("div");
      panel.className = "mmd-panel";
      panel.style.cssText =
        "position:absolute;top:40px;right:8px;max-height:82%;overflow-y:auto;width:236px;" +
        "background:rgba(18,18,26,0.92);color:#eee;font:12px sans-serif;padding:8px 10px;" +
        "border-radius:8px;z-index:50;display:none;box-shadow:0 4px 16px rgba(0,0,0,.5)";

      const head = document.createElement("div");
      head.style.cssText = "font-weight:bold;margin-bottom:6px";
      head.textContent = "ビジュアル調整";
      panel.appendChild(head);

      const sliderEls: Partial<Record<keyof LookState, HTMLInputElement>> = {};
      const valEls: Partial<Record<keyof LookState, HTMLSpanElement>> = {};
      const colorEls: Partial<Record<keyof LookState, HTMLInputElement>> = {};

      const syncControls = () => {
        SLIDERS.forEach(([k]) => {
          const v = String(look[k]);
          if (sliderEls[k]) sliderEls[k]!.value = v;
          if (valEls[k]) valEls[k]!.textContent = v;
        });
        COLORS.forEach(([k]) => {
          if (colorEls[k]) colorEls[k]!.value = hexStr(look[k]);
        });
      };

      // Preset buttons
      const presetRow = document.createElement("div");
      presetRow.style.cssText = "display:flex;gap:6px;margin-bottom:8px";
      const mkPreset = (label: string, preset: "msToon" | "moody") => {
        const b = document.createElement("button");
        b.textContent = label;
        b.style.cssText =
          "flex:1;cursor:pointer;background:#3a3a5a;color:#fff;border:none;border-radius:4px;padding:4px 0;font-size:11px";
        b.addEventListener("click", () => {
          setLookPreset(preset);
          syncControls();
        });
        return b;
      };
      presetRow.appendChild(mkPreset("msToon", "msToon"));
      presetRow.appendChild(mkPreset("moody", "moody"));
      panel.appendChild(presetRow);

      // Numeric sliders
      SLIDERS.forEach(([k, label, min, max, step]) => {
        const row = document.createElement("label");
        row.style.cssText = "display:flex;align-items:center;gap:6px;margin:2px 0";
        const lab = document.createElement("span");
        lab.textContent = label;
        lab.style.cssText = "flex:0 0 66px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
        const sl = document.createElement("input");
        sl.type = "range";
        sl.min = String(min);
        sl.max = String(max);
        sl.step = String(step);
        sl.value = String(look[k]);
        sl.style.cssText = "flex:1;min-width:0";
        const val = document.createElement("span");
        val.textContent = sl.value;
        val.style.cssText = "flex:0 0 32px;text-align:right;opacity:.7";
        sl.addEventListener("input", () => {
          (look[k] as number) = parseFloat(sl.value);
          val.textContent = sl.value;
          applyLookState();
          saveLook();
        });
        sliderEls[k] = sl;
        valEls[k] = val;
        row.appendChild(lab);
        row.appendChild(sl);
        row.appendChild(val);
        panel.appendChild(row);
      });

      // Colour pickers
      COLORS.forEach(([k, label]) => {
        const row = document.createElement("label");
        row.style.cssText = "display:flex;align-items:center;gap:6px;margin:3px 0";
        const lab = document.createElement("span");
        lab.textContent = label;
        lab.style.cssText = "flex:1";
        const ci = document.createElement("input");
        ci.type = "color";
        ci.value = hexStr(look[k]);
        ci.style.cssText =
          "flex:0 0 36px;height:20px;padding:0;border:none;background:none;cursor:pointer";
        ci.addEventListener("input", () => {
          (look[k] as number) = parseInt(ci.value.slice(1), 16);
          applyLookState();
          saveLook();
        });
        colorEls[k] = ci;
        row.appendChild(lab);
        row.appendChild(ci);
        panel.appendChild(row);
      });

      const toggle = document.createElement("button");
      toggle.textContent = "🎨";
      toggle.title = "ビジュアル調整パネル";
      toggle.style.cssText =
        "position:absolute;top:8px;right:44px;z-index:51;cursor:pointer;font-size:16px;" +
        "background:rgba(18,18,26,0.75);border:none;border-radius:6px;padding:2px 7px";
      toggle.addEventListener("click", () => {
        const open = panel.style.display === "none";
        container.querySelectorAll(".mmd-panel").forEach((p) => {
          (p as HTMLElement).style.display = "none";
        });
        if (open) syncControls(); // reflect any console-driven changes
        panel.style.display = open ? "block" : "none";
      });

      container.appendChild(panel);
      container.appendChild(toggle);
      return () => {
        panel.remove();
        toggle.remove();
      };
    };

    // --- Graphics / quality settings (⚙ panel + console, persisted) ----------
    // Render/perf knobs kept separate from the artistic `look` grade: MSAA,
    // supersampling, anisotropic texture filtering, cel-outline thickness and
    // shadow-map resolution. Driven by the ⚙ panel and __mmdAA/__mmdSS/etc.
    const maxAniso = renderer.capabilities.getMaxAnisotropy();
    type GfxState = {
      aa: number; // MSAA samples (0/2/4/8) on the composer render target
      ss: number; // supersampling factor (1 = off .. 2 = render at 2× then downscale)
      aniso: number; // anisotropic texture filtering (1 = off .. maxAniso)
      outline: number; // cel outline thickness multiplier (× the PMX edge size)
      shadow: number; // shadow-map resolution (px, square)
    };
    const GFX_DEFAULT: GfxState = {
      aa: aaSamples,
      ss: 1.5, // supersample 1.5× by default — renders the character sharper than native
      aniso: Math.min(8, maxAniso),
      outline: outlineScale,
      shadow: 1024,
    };
    let gfx: GfxState = { ...GFX_DEFAULT, ...(lsGet<Partial<GfxState>>(LS_GFX) || {}) };
    const saveGfx = () => lsSet(LS_GFX, gfx);
    // Anisotropic filtering sharpens textures viewed at grazing angles (hair,
    // clothing). Applied to each model material's colour/specular/emissive maps.
    const applyAniso = (v: number) => {
      if (!mesh) return;
      const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      mats.forEach((m: any) => {
        ["map", "specularMap", "emissiveMap"].forEach((slot) => {
          const t = m?.[slot];
          if (t && t.isTexture && t.anisotropy !== v) {
            t.anisotropy = v;
            t.needsUpdate = true;
          }
        });
      });
    };
    const applyGfx = () => {
      // AA: rebuild the multisampled composer target only when the count changes.
      if (aaSamples !== gfx.aa) {
        aaSamples = gfx.aa;
        composer.reset(makeComposerRT(aaSamples));
      }
      // Render at the display's NATIVE pixel ratio (no artificial 2× cap, so
      // HiDPI screens get their full resolution) × the SSAA factor, bounded to
      // 4× so a high-DPI display + supersampling can't melt the GPU.
      const p = Math.min(window.devicePixelRatio * gfx.ss, 4);
      renderer.setPixelRatio(p);
      renderer.setSize(container.clientWidth, container.clientHeight);
      composer.setPixelRatio(p);
      composer.setSize(container.clientWidth, container.clientHeight);
      // Outline thickness (no-ops until the PMX edge sizes are captured on load).
      outlineScale = gfx.outline;
      applyOutlineScale();
      // Anisotropy (no-ops until the model is loaded).
      applyAniso(gfx.aniso);
      // Shadow-map resolution: resize + force a rebuild of the shadow target.
      if (keyLight.shadow.mapSize.width !== gfx.shadow) {
        keyLight.shadow.mapSize.set(gfx.shadow, gfx.shadow);
        keyLight.shadow.map?.dispose?.();
        (keyLight.shadow as any).map = null;
      }
    };
    applyGfx(); // apply saved AA/SSAA/shadow now; aniso/outline re-applied after load

    // On-screen graphics panel (⚙ button): AA / supersampling / anisotropy /
    // outline / shadow resolution. Edits apply live and persist.
    const buildGfxPanel = (): (() => void) => {
      const panel = document.createElement("div");
      panel.className = "mmd-panel";
      panel.style.cssText =
        "position:absolute;top:40px;right:8px;max-height:82%;overflow-y:auto;width:236px;" +
        "background:rgba(18,18,26,0.92);color:#eee;font:12px sans-serif;padding:8px 10px;" +
        "border-radius:8px;z-index:50;display:none;box-shadow:0 4px 16px rgba(0,0,0,.5)";
      const head = document.createElement("div");
      head.style.cssText = "font-weight:bold;margin-bottom:6px";
      head.textContent = "画質設定";
      panel.appendChild(head);

      const mkRow = (labelText: string, control: HTMLElement) => {
        const row = document.createElement("label");
        row.style.cssText = "display:flex;align-items:center;gap:6px;margin:4px 0";
        const lab = document.createElement("span");
        lab.textContent = labelText;
        lab.style.cssText =
          "flex:0 0 84px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
        row.appendChild(lab);
        row.appendChild(control);
        panel.appendChild(row);
      };
      const mkSelect = (
        options: [number, string][],
        current: number,
        onChange: (v: number) => void,
      ): HTMLElement => {
        const sel = document.createElement("select");
        sel.style.cssText =
          "flex:1;min-width:0;background:#2a2a3a;color:#eee;border:1px solid #444;border-radius:4px;padding:2px 4px";
        options.forEach(([v, label]) => {
          const o = document.createElement("option");
          o.value = String(v);
          o.textContent = label;
          if (v === current) o.selected = true;
          sel.appendChild(o);
        });
        sel.addEventListener("change", () => onChange(parseFloat(sel.value)));
        return sel;
      };
      const mkSlider = (
        min: number,
        max: number,
        step: number,
        current: number,
        onChange: (v: number) => void,
      ): HTMLElement => {
        const wrap = document.createElement("div");
        wrap.style.cssText = "flex:1;display:flex;align-items:center;gap:6px;min-width:0";
        const sl = document.createElement("input");
        sl.type = "range";
        sl.min = String(min);
        sl.max = String(max);
        sl.step = String(step);
        sl.value = String(current);
        sl.style.cssText = "flex:1;min-width:0";
        const val = document.createElement("span");
        val.textContent = String(current);
        val.style.cssText = "flex:0 0 30px;text-align:right;opacity:.7";
        sl.addEventListener("input", () => {
          val.textContent = sl.value;
          onChange(parseFloat(sl.value));
        });
        wrap.appendChild(sl);
        wrap.appendChild(val);
        return wrap;
      };

      const aaOpts = ([[0, "オフ"], [2, "2x"], [4, "4x"], [8, "8x"]] as [number, string][]).filter(
        ([v]) => v <= maxSamples,
      );
      mkRow(
        "アンチエイリアス",
        mkSelect(aaOpts, gfx.aa, (v) => {
          gfx.aa = v;
          applyGfx();
          saveGfx();
        }),
      );
      mkRow(
        "スーパーサンプル",
        mkSlider(1, 2, 0.25, gfx.ss, (v) => {
          gfx.ss = v;
          applyGfx();
          saveGfx();
        }),
      );
      const anisoOpts = (
        [[1, "オフ"], [2, "2x"], [4, "4x"], [8, "8x"], [16, "16x"]] as [number, string][]
      ).filter(([v]) => v <= maxAniso);
      mkRow(
        "異方性フィルタ",
        mkSelect(anisoOpts, gfx.aniso, (v) => {
          gfx.aniso = v;
          applyGfx();
          saveGfx();
        }),
      );
      mkRow(
        "輪郭線の太さ",
        mkSlider(0, 4, 0.1, gfx.outline, (v) => {
          gfx.outline = v;
          applyGfx();
          saveGfx();
        }),
      );
      mkRow(
        "影の解像度",
        mkSelect(
          [
            [512, "低"],
            [1024, "中"],
            [2048, "高"],
          ],
          gfx.shadow,
          (v) => {
            gfx.shadow = v;
            applyGfx();
            saveGfx();
          },
        ),
      );

      const toggle = document.createElement("button");
      toggle.textContent = "⚙";
      toggle.title = "画質設定パネル";
      toggle.style.cssText =
        "position:absolute;top:8px;right:80px;z-index:51;cursor:pointer;font-size:16px;" +
        "background:rgba(18,18,26,0.75);border:none;border-radius:6px;padding:2px 7px";
      toggle.addEventListener("click", () => {
        const open = panel.style.display === "none";
        container.querySelectorAll(".mmd-panel").forEach((p) => {
          (p as HTMLElement).style.display = "none";
        });
        panel.style.display = open ? "block" : "none";
      });

      container.appendChild(panel);
      container.appendChild(toggle);
      return () => {
        panel.remove();
        toggle.remove();
      };
    };

    const loader = new MMDLoader();
    loader.load(
      url,
      (loaded) => {
        if (disposed) {
          loaded.geometry?.dispose?.();
          return;
        }
        mesh = loaded;
        loaded.castShadow = true; // self-shadowing (hair on face, body depth)
        loaded.receiveShadow = true;
        scene.add(loaded);
        applyArmPose(loaded); // lower arms before physics captures rest state
        loaded.updateMatrixWorld(true);
        applySavedViewOrFrame();

        // Capture each material's base PMX outline thickness, then thicken to
        // match the reference video's bolder contour lines.
        outlineBaseMap = new Map();
        const outMats = Array.isArray(loaded.material) ? loaded.material : [loaded.material];
        outMats.forEach((m: any) => {
          const op = m?.userData?.outlineParameters;
          if (op && typeof op.thickness === "number") outlineBaseMap!.set(m, op.thickness);
        });
        applyOutlineScale();

        // Identify the skin materials (body01 / face01 / face02) and make them
        // matte so the strong rims don't put a plastic specular hotspot on the
        // skin. Warm emissive is applied by applyLookState (skinWarmth).
        skinMaterials = outMats.filter((m: any) => /^(body|face)0\d/i.test(m?.name || ""));
        skinMaterials.forEach((m: any) => {
          if (m.specular) m.specular.setRGB(0.03, 0.03, 0.03);
          if (typeof m.shininess === "number") m.shininess = 3;
        });
        console.log("[MMD] skin materials:", skinMaterials.map((m: any) => m.name));

        // Hard cel shading: remember each toon material's original ramp so the
        // cel bands can be toggled, then apply the current setting.
        celMaterials = outMats.filter((m: any) => m && m.isMMDToonMaterial);
        celMaterials.forEach((m: any) => celOriginalRamp.set(m, m.gradientMap));

        applyLookState(); // skin warmth + cel bands now that materials exist


        // Lip-sync: drive the あ (mouth-open) morph from speech volume.
        const morphs = loaded.morphTargetDictionary || {};
        const mouthIdx = morphs["あ"] ?? morphs["ア"] ?? morphs["A"] ?? -1;
        if (mouthIdx >= 0) {
          mmdLipSync.setMesh(loaded, mouthIdx);
          console.log(`[MMD] lip-sync morph 'あ' @ index ${mouthIdx}`);
        } else {
          console.warn("[MMD] no 'あ' mouth morph found for lip-sync");
        }
        // Resting face (default expression), e.g. a soft smile.
        mmdExpression.setMesh(loaded, morphs as Record<string, number>);
        // Runtime .vmd clip playback (HY-Motion clips from the PC-B service).
        mmdMotionPlayer.setMesh(loaded);
        // On-screen slider panel (🎭 button) to tweak every morph live.
        disposePanel = buildMorphPanel(loaded);
        // On-screen visual panel (🎨 button): lights / bloom / cel / colours.
        disposeLookPanel = buildLookPanel();
        // On-screen graphics panel (⚙ button): AA / SSAA / aniso / outline / shadow.
        disposeGfxPanel = buildGfxPanel();
        applyGfx(); // re-apply aniso + outline now the model / textures exist

        (window as any).__mmd = { scene, camera, renderer, controls, mesh: loaded };
        // --- Console tuning (all persist to localStorage) ---
        // Camera zoom by coverage: 1.0 full body, 0.6 bust, 0.5 close.
        (window as any).__mmdCam = (coverage = 1.0) => {
          currentCoverage = coverage;
          frame(coverage);
          saveView();
        };
        // Arms hang down; tune the lean: out = how far out to the sides,
        // fwd = how far forward. e.g. __mmdArms(0.1) tucks them closer to the body.
        (window as any).__mmdArms = (out = armCfg.out, fwd = armCfg.fwd) => {
          armCfg = { out, fwd };
          if (mesh) {
            applyArmPose(mesh);
            mesh.updateMatrixWorld(true);
          }
          lsSet(LS_ARMS, armCfg);
          console.log(`[MMD] arms out=${out} fwd=${fwd}`);
        };
        (window as any).__mmdView = () =>
          console.log("camera", camera.position.toArray(), "target", controls.target.toArray());
        // Mouth open amount: __mmdMouth(gain, cap). Lower = smaller opening.
        (window as any).__mmdMouth = (gain = 0.95, cap = 0.72) => {
          mmdLipSync.setGain(gain, cap);
          console.log(`[MMD] mouth gain=${gain} cap=${cap}`);
        };
        // Play a .vmd motion clip on the model (HY-Motion retarget verification +
        // runtime playback). Use the same-origin proxy, e.g.
        //   __mmdPlayVmd("/motion/pool/happy")
        // loop=true by default; the procedural idle is gated while it plays.
        (window as any).__mmdPlayVmd = (url: string, loop = true) => {
          mmdMotionPlayer.play(url, { loop });
          console.log(`[MMD] play vmd ${url} (loop=${loop})`);
        };
        // Fade the current clip out and return to idle: __mmdStopVmd(0.3).
        (window as any).__mmdStopVmd = (fade = 0.3) => {
          mmdMotionPlayer.stop(fade);
          console.log(`[MMD] stop vmd (fade=${fade}s)`);
        };
        // Point live motion at the PC-B HY-Motion service (persists). Empty = off.
        // e.g. __mmdMotionUrl("http://192.168.1.50:8800")
        (window as any).__mmdMotionUrl = (url = "") => {
          mmdMotionPlayer.setBaseUrl(url);
          console.log(`[MMD] motion service = ${url || "(disabled)"}`);
        };
        // Dump the model's REST (bind-pose) bone geometry for the arm/spine
        // chains — gives the retargeter (PC-B smpl_to_mmd.py) the exact A-pose
        // directions to build the SMPL(T-pose)->MMD(A-pose) correction without
        // needing the .pmx transferred. Uses skeleton.boneInverses (bind pose),
        // so it's unaffected by the current idle/clip pose.
        (window as any).__mmdBoneDirs = () => {
          const m = mesh;
          if (!m) {
            console.warn("[MMD] no mesh");
            return;
          }
          const sk = m.skeleton;
          const restPos = (name: string): THREE.Vector3 | null => {
            const i = sk.bones.findIndex((b) => b.name === name);
            if (i < 0) return null;
            return new THREE.Vector3().setFromMatrixPosition(
              sk.boneInverses[i].clone().invert(),
            );
          };
          const chain: [string, string][] = [
            ["左肩", "左腕"], ["左腕", "左ひじ"], ["左ひじ", "左手首"],
            ["右肩", "右腕"], ["右腕", "右ひじ"], ["右ひじ", "右手首"],
            ["上半身", "上半身2"], ["上半身2", "首"], ["首", "頭"],
          ];
          const out: Record<string, unknown> = {};
          for (const [a, b] of chain) {
            const pa = restPos(a);
            const pb = restPos(b);
            if (!pa || !pb) {
              out[`${a}->${b}`] = "missing";
              continue;
            }
            const d = pb.clone().sub(pa);
            const len = d.length();
            const dir = d.clone().normalize();
            const elevationDeg = (Math.asin(Math.max(-1, Math.min(1, dir.y))) * 180) / Math.PI;
            out[`${a}->${b}`] = {
              dir: [+dir.x.toFixed(3), +dir.y.toFixed(3), +dir.z.toFixed(3)],
              elevationDeg: +elevationDeg.toFixed(1),
              len: +len.toFixed(2),
              headPos: [+pa.x.toFixed(2), +pa.y.toFixed(2), +pa.z.toFixed(2)],
            };
          }
          console.log("[MMD] rest bone directions (bind pose):");
          console.log(JSON.stringify(out, null, 2));
          return out;
        };
        // Resting face: __mmdFace("にこり", 0.6) sets a default expression morph
        // (0 removes it). Combine several, e.g. __mmdFace("にこり",0.5)+__mmdFace("ウィンク",0).
        (window as any).__mmdFace = (name: string, value = 0.6) => {
          mmdExpression.setBase(name, value);
          console.log("[MMD] base face:", mmdExpression.getBase());
        };
        // List the current resting face + every available expression morph.
        (window as any).__mmdFaceList = () => {
          console.log("[MMD] base face:", mmdExpression.getBase());
          console.log("[MMD] available morphs:", mmdExpression.morphNames());
          return mmdExpression.getBase();
        };
        // Clear any saved face and restore the default combo.
        (window as any).__mmdFaceReset = () => {
          const b = mmdExpression.resetBase();
          console.log("[MMD] face reset to default:", b);
        };
        // Debug: preview an emotion expression without a conversation, e.g.
        // __mmdEmotion("happy"). __mmdEmotion() (no arg) relaxes to the resting
        // face. Valid: happy/sad/angry/surprised/thinking/shy/neutral.
        (window as any).__mmdEmotion = (emotion?: string) => {
          if (emotion) mmdExpression.playEmotion(emotion);
          else mmdExpression.relax();
          console.log("[MMD] emotion:", emotion || "(relax)");
        };
        // Toggle continuously-generated idle BODY motion (neutral pool clips
        // looping at idle) instead of procedural sway. Off by default.
        (window as any).__mmdIdleMotion = (on = true) => {
          (window as any).__mmdIdleMotionOn = on;
          console.log("[MMD] idle body motion:", on ? "ON" : "OFF");
        };
        // Debug: list every morph this model exposes (name -> index).
        (window as any).__mmdMorphs = () => {
          const dd = mesh?.morphTargetDictionary || {};
          console.log("[MMD] morphs:", Object.keys(dd));
          return dd;
        };
        // Debug: force a morph on (1) for 1.5s to verify it actually deforms.
        (window as any).__mmdTestMorph = (name: string, val = 1) => {
          const dd = mesh?.morphTargetDictionary || {};
          const i = dd[name];
          if (i == null || !mesh?.morphTargetInfluences) {
            console.warn("[MMD] morph not found:", name);
            return;
          }
          mesh.morphTargetInfluences[i] = val;
          console.log(`[MMD] set morph '${name}' (#${i}) = ${val}`);
          setTimeout(() => {
            if (mesh?.morphTargetInfluences) mesh.morphTargetInfluences[i] = 0;
          }, 1500);
        };
        (window as any).__mmdResetView = () => {
          try {
            localStorage.removeItem(LS_VIEW);
          } catch {
            /* ignore */
          }
          currentCoverage = 1.0;
          frame(1.0);
        };
        // All visual knobs below write the shared `look` state, apply it and
        // persist, so they stay in sync with the 🎨 panel and survive reloads.
        // Glow (bloom): __mmdBloom(strength, radius, threshold).
        (window as any).__mmdBloom = (
          strength = look.bloomStrength,
          radius = look.bloomRadius,
          threshold = look.bloomThreshold,
        ) => {
          look.bloomStrength = strength;
          look.bloomRadius = radius;
          look.bloomThreshold = threshold;
          applyLookState();
          saveLook();
          console.log(`[MMD] bloom s=${strength} r=${radius} t=${threshold}`);
        };
        // Background solid colour override: __mmdBg(0x0a0a1a). __mmdBg(null) =
        // transparent. (Not persisted; use the 🎨 panel for the gradient.)
        (window as any).__mmdBg = (hex: number | null = 0x0a0a1a) => {
          scene.background = hex === null ? null : new THREE.Color(hex);
        };
        // Cel toon re-shade: __mmdToon(steps, saturation, contrast). steps 0 =
        // off (rely on the MMD toon material); lower = flatter/cel-er.
        (window as any).__mmdToon = (
          steps = look.toonSteps,
          sat = look.toonSat,
          contrast = look.toonContrast,
        ) => {
          look.toonSteps = steps;
          look.toonSat = sat;
          look.toonContrast = contrast;
          applyLookState();
          saveLook();
          console.log(`[MMD] toon steps=${steps} sat=${sat} contrast=${contrast}`);
        };
        // Cinematic grade: __mmdGrade(chroma, temp, tint, split). chroma=edge
        // RGB fringe, temp -cool/+warm, tint -green/+magenta, split=teal-shadow
        // /magenta-highlight tone.
        (window as any).__mmdGrade = (
          chroma = look.toonChroma,
          temp = look.toonTemp,
          tint = look.toonTint,
          split = look.toonSplit,
        ) => {
          look.toonChroma = chroma;
          look.toonTemp = temp;
          look.toonTint = tint;
          look.toonSplit = split;
          applyLookState();
          saveLook();
          console.log(`[MMD] grade chroma=${chroma} temp=${temp} tint=${tint} split=${split}`);
        };
        // Filmic finish: __mmdFilm(vignette, grain, bars). vignette=edge darken,
        // grain=film noise, bars=letterbox height (top & bottom).
        (window as any).__mmdFilm = (
          vignette = look.vignette,
          grain = look.grain,
          bars = look.bars,
        ) => {
          look.vignette = vignette;
          look.grain = grain;
          look.bars = bars;
          applyLookState();
          saveLook();
          console.log(`[MMD] film vignette=${vignette} grain=${grain} bars=${bars}`);
        };
        // Look presets: flip the whole rig between the bright flat msToon cel
        // look and the dark cyan/magenta music-video look (persists).
        (window as any).__mmdMsToon = () => {
          setLookPreset("msToon");
          console.log("[MMD] look: msToon (bright flat cel)");
        };
        (window as any).__mmdMoody = () => {
          setLookPreset("moody");
          console.log("[MMD] look: moody (music-video rim)");
        };
        // Fine-tune light intensities: __mmdLight(key, rimCool, rimWarm, ambient, hemi).
        (window as any).__mmdLight = (
          key = look.key,
          rimC = look.rimCool,
          rimW = look.rimWarm,
          amb = look.ambient,
          hem = look.hemi,
        ) => {
          look.key = key;
          look.rimCool = rimC;
          look.rimWarm = rimW;
          look.ambient = amb;
          look.hemi = hem;
          applyLookState();
          saveLook();
          console.log(`[MMD] light key=${key} rimCool=${rimC} rimWarm=${rimW} amb=${amb} hemi=${hem}`);
        };
        // Hard cel shading: __mmdCel(3). 0/1 = off (soft original ramp); 2-5 =
        // that many hard flat shading bands (real anime cel look).
        (window as any).__mmdCel = (bands = look.celBands) => {
          look.celBands = bands;
          applyLookState();
          saveLook();
          console.log(`[MMD] cel bands=${bands}`);
        };
        // Cast-shadow darkness: __mmdShadow(0.55). 0 = off.
        (window as any).__mmdShadow = (s = look.shadowStrength) => {
          look.shadowStrength = s;
          applyLookState();
          saveLook();
          console.log(`[MMD] shadow=${s}`);
        };
        // Tone mapping: __mmdTone(mode, exposure). mode 0 none,1 ACES,2 AgX,3 Neutral.
        (window as any).__mmdTone = (mode = look.toneMode, exposure = look.toneExposure) => {
          look.toneMode = mode;
          look.toneExposure = exposure;
          applyLookState();
          saveLook();
          console.log(`[MMD] tone mode=${mode} exposure=${exposure}`);
        };
        // Depth of field: __mmdDof(aperture, maxblur). aperture 0 = off.
        (window as any).__mmdDof = (aperture = look.dofAperture, blur = look.dofBlur) => {
          look.dofAperture = aperture;
          look.dofBlur = blur;
          applyLookState();
          saveLook();
          console.log(`[MMD] dof aperture=${aperture} blur=${blur}`);
        };
        // Fresnel rim: __mmdFresnel(intensity, power, colorHex). intensity 0 = off.
        (window as any).__mmdFresnel = (
          intensity = look.fresnelInt,
          power = look.fresnelPow,
          color = look.fresnelColor,
        ) => {
          look.fresnelInt = intensity;
          look.fresnelPow = power;
          look.fresnelColor = color;
          applyLookState();
          saveLook();
          console.log(`[MMD] fresnel int=${intensity} pow=${power} color=${hexStr(color)}`);
        };
        // Non-mixing sublight: __mmdSub(intensity, power, mix). intensity 0 = off;
        // mix 0 = cyan/magenta colours stay separate (don't mix), 1 = additive.
        // Colours/directions follow the リム寒色/暖色 + 青光/桃光 settings.
        (window as any).__mmdSub = (
          intensity = look.subInt,
          power = look.subPow,
          mix = look.subMix,
        ) => {
          look.subInt = intensity;
          look.subPow = power;
          look.subMix = mix;
          applyLookState();
          saveLook();
          console.log(`[MMD] sublight int=${intensity} pow=${power} mix=${mix}`);
        };
        // Skin warmth: __mmdSkin(0.32). Warm emissive floor on body01/face01/02
        // so the skin reads natural-warm (not blue) under the cool rims.
        (window as any).__mmdSkin = (warmth = look.skinWarmth) => {
          look.skinWarmth = warmth;
          applyLookState();
          saveLook();
          console.log(`[MMD] skin warmth=${warmth}`);
        };
        // Aim the rims: __mmdRimDir(coolAzim, coolElev, warmAzim, warmElev), deg.
        // azim 0=front, ±180=behind, +90=camera-right; elev = height.
        (window as any).__mmdRimDir = (
          coolAzim = look.rimCoolAzim,
          coolElev = look.rimCoolElev,
          warmAzim = look.rimWarmAzim,
          warmElev = look.rimWarmElev,
        ) => {
          look.rimCoolAzim = coolAzim;
          look.rimCoolElev = coolElev;
          look.rimWarmAzim = warmAzim;
          look.rimWarmElev = warmElev;
          applyLookState();
          saveLook();
          console.log(`[MMD] rim dir cool(${coolAzim},${coolElev}) warm(${warmAzim},${warmElev})`);
        };
        // Outline (contour line) thickness multiplier: __mmdOutline(1.8). 1 = the
        // model's own PMX edge size; higher = bolder lines.
        (window as any).__mmdOutline = (scale = gfx.outline) => {
          gfx.outline = scale;
          applyGfx();
          saveGfx();
          console.log(`[MMD] outline scale=${scale}`);
        };
        // Anisotropic texture filtering: __mmdAniso(16). 1 = off.
        (window as any).__mmdAniso = (v = gfx.aniso) => {
          gfx.aniso = Math.max(1, Math.min(Math.round(v), maxAniso));
          applyGfx();
          saveGfx();
          console.log(`[MMD] anisotropy=${gfx.aniso} (max ${maxAniso})`);
        };
        // Shadow-map resolution: __mmdShadowRes(2048). 512/1024/2048.
        (window as any).__mmdShadowRes = (px = gfx.shadow) => {
          gfx.shadow = Math.max(256, Math.round(px));
          applyGfx();
          saveGfx();
          console.log(`[MMD] shadow map=${gfx.shadow}px`);
        };
        // Backdrop gradient: __mmdBgGrad(0xbcc2e4, 0x6b7099) (inner, outer hex).
        (window as any).__mmdBgGrad = (inner = look.bgInner, outer = look.bgOuter) => {
          look.bgInner = inner;
          look.bgOuter = outer;
          applyLookState();
          saveLook();
          console.log(`[MMD] bg gradient inner=${hexStr(inner)} outer=${hexStr(outer)}`);
        };
        // Anti-aliasing (MSAA): __mmdAA(8). 0/1 = off, 2/4/8 = progressively
        // smoother geometry & outline edges (8 = best, more GPU). Recreates the
        // composer's multisampled render target.
        (window as any).__mmdAA = (samples = gfx.aa) => {
          gfx.aa = Math.max(0, Math.min(Math.round(samples), maxSamples));
          applyGfx();
          saveGfx();
          console.log(`[MMD] MSAA samples=${gfx.aa} (max ${maxSamples})`);
        };
        // Supersampling (SSAA): __mmdSS(1.5) renders the whole pipeline at 1.5×
        // then downscales — smooths EVERYTHING including the shaded/rim edges
        // MSAA misses, at a GPU cost (1 = off, 2 = max). Pair with __mmdAA for
        // the cleanest image.
        (window as any).__mmdSS = (factor = gfx.ss) => {
          gfx.ss = Math.max(1, Math.min(factor, 2));
          applyGfx();
          saveGfx();
          console.log(`[MMD] supersample=${gfx.ss}×`);
        };
        console.log(
          "[MMD] loaded — ⚙ 画質パネル + 🎨 panel, or console: __mmdAA(), __mmdSS(), __mmdAniso(), __mmdShadowRes(), __mmdOutline(), __mmdMsToon()/__mmdMoody(), __mmdLight(), __mmdBloom(), __mmdGrade()",
        );
        initPhysics();
      },
      undefined,
      (err) => console.error("[MMD] load error:", err),
    );

    const animate = () => {
      raf = requestAnimationFrame(animate);
      const delta = Math.min(clock.getDelta(), 1 / 30);
      elapsed += delta;
      if (animReady && mesh) {
        // Optional: play continuously-generated neutral clips at idle instead of
        // procedural sway (opt-in via __mmdIdleMotion(true)). Self-throttled and
        // a no-op while a reply clip is active.
        if ((window as any).__mmdIdleMotionOn === true) {
          mmdMotionPlayer.pumpIdle(true);
        }
        // Clip mixer first (poses the bones from the vmd), then gated idle
        // (skips body bones while a clip plays), then physics reacts to the pose.
        mmdMotionPlayer.update(delta);
        updateIdle(mesh, mmdMotionPlayer.isActive());
        if (helper) {
          try {
            helper.update(delta);
          } catch {
            /* ignore physics hiccup */
          }
        }
      }
      mmdLipSync.update(); // mouth morph from speech volume (no-op when silent)
      mmdExpression.update(); // resting-face morphs (e.g. soft smile)
      controls.update();
      toonPass.uniforms.uTime.value = elapsed; // animate the film grain
      bokehPass.uniforms.focus.value = camera.position.distanceTo(controls.target); // keep model sharp
      updateSublightDirs(); // keep the view-space sublight directions current as the camera moves
      composer.render(); // scene + outline + bloom
    };
    animate();

    const resize = () => {
      const w = Math.max(container.clientWidth, 1);
      const h = Math.max(container.clientHeight, 1);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      composer.setSize(w, h);
    };
    const ro = new ResizeObserver(resize);
    ro.observe(container);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.removeEventListener("end", saveView);
      controls.dispose();
      disposePanel?.();
      disposeLookPanel?.();
      disposeGfxPanel?.();
      if (mesh) {
        mmdLipSync.clearMesh(mesh);
        mmdExpression.clearMesh(mesh);
        mmdMotionPlayer.clearMesh(mesh);
      }
      if (helper && mesh) {
        try {
          helper.remove(mesh);
        } catch {
          /* ignore */
        }
      }
      helper = null;
      const w = window as any;
      if (w.__mmd?.renderer === renderer) {
        delete w.__mmd;
        delete w.__mmdCam;
        delete w.__mmdArms;
        delete w.__mmdView;
        delete w.__mmdResetView;
        delete w.__mmdMorphs;
        delete w.__mmdTestMorph;
        delete w.__mmdMouth;
        delete w.__mmdPlayVmd;
        delete w.__mmdStopVmd;
        delete w.__mmdMotionUrl;
        delete w.__mmdBoneDirs;
        delete w.__mmdFace;
        delete w.__mmdFaceList;
        delete w.__mmdFaceReset;
        delete w.__mmdBloom;
        delete w.__mmdBg;
        delete w.__mmdToon;
        delete w.__mmdMsToon;
        delete w.__mmdMoody;
        delete w.__mmdLight;
        delete w.__mmdRimDir;
        delete w.__mmdSkin;
        delete w.__mmdCel;
        delete w.__mmdShadow;
        delete w.__mmdTone;
        delete w.__mmdDof;
        delete w.__mmdFresnel;
        delete w.__mmdSub;
        delete w.__mmdBgGrad;
        delete w.__mmdOutline;
        delete w.__mmdAA;
        delete w.__mmdSS;
        delete w.__mmdAniso;
        delete w.__mmdShadowRes;
      }
      try {
        composer.dispose();
      } catch {
        /* ignore */
      }
      scene.traverse((obj) => {
        const mm = obj as THREE.Mesh;
        if (mm.geometry) mm.geometry.dispose();
        const mat = (mm as any).material;
        if (Array.isArray(mat)) {
          mat.forEach((x) => {
            x.map?.dispose?.();
            x.dispose?.();
          });
        } else if (mat) {
          mat.map?.dispose?.();
          mat.dispose?.();
        }
      });
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [url]);

  return (
    <div
      ref={containerRef}
      id="mmd-viewer-wrapper"
      style={{ width: "100%", height: "100%", overflow: "hidden", position: "relative" }}
    />
  );
});

MMDViewer.displayName = "MMDViewer";
