import * as THREE from "three";
import { SPACE } from "./space-config";
import { driftBase, driftTarget, orbitPose } from "./framing";
import { SPACE_LAYER } from "./layers";
import type { SpaceModule, SpaceMood } from "./types";

/**
 * Frosted glass: the panel replaces what is behind it with a BLURRED copy of it.
 *
 * There is no backdrop-filter in WebGL, so the background has to be rendered and
 * sampled. What gets rendered is only the SPACE layer with the panels themselves
 * hidden — never the whole scene — for two reasons. It is cheap (the layer is
 * lines and a handful of meshes, and the target is a quarter resolution per
 * axis, so a sixteenth of the pixels). And it is CORRECT: the character is in
 * front of the panels at z~0 while they sit at z -1.3..-1.6, so including her
 * would smear her through glass she is standing in front of.
 *
 * The frost is its own mesh rather than a change to the content material. The
 * content is a canvas texture in sRGB and the render target is linear, and
 * mixing the two in one shader means hand-managing a colour-space conversion
 * that `MeshBasicMaterial` already gets right. Separate meshes keep the proven
 * path proven: frost samples linear and outputs linear, content is untouched.
 */
const FROST_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const FROST_FRAG = /* glsl */ `
uniform sampler2D uBack;
uniform vec2 uResolution;
uniform vec2 uTexel;
uniform float uBlur;
uniform float uOpacity;
uniform vec3 uTint;
uniform float uTintAmt;
uniform float uEdge;
uniform float uDetail;
varying vec2 vUv;

void main() {
  // Screen-space lookup from gl_FragCoord, not from an interpolated varying:
  // the panels are rotated 26 degrees about Y, so a varying divided by w in the
  // vertex shader is not perspective-correct across the quad and the blur would
  // visibly slide against the background it is meant to be sampling.
  vec2 su = gl_FragCoord.xy / uResolution;

  /**
   * ELLIPTICAL falloff, on a quad deliberately larger than the panel.
   *
   * Everything the frost does is scaled by this — the blur RADIUS as well as the
   * opacity and the tint — because scaling opacity alone leaves a seam: at the
   * border a half-strength blurred image is composited over a sharp one, and the
   * two differ most exactly where the blur is widest. Taking the radius to zero
   * alongside it removes that by construction: at radius 0 the blurred sample IS
   * the background, so there is nothing left to join.
   *
   * But that was not enough, and the reason is shape rather than strength. The
   * first version multiplied a smoothstep per AXIS, which is soft across each
   * edge yet perfectly straight ALONG it — so the blurred region was still a
   * rectangle, and a rectangle of softened background reads as a panel-shaped
   * patch however gently it fades. Measured by amplifying the panels' own
   * contribution 12x: the outline was unmistakable.
   *
   * A radial falloff has no straight edge anywhere to be seen, and the quad is
   * padded well beyond the panel so the blob reaches zero in open space instead
   * of at the frame the eye is already tracking. length() on the quad's own UV
   * gives an ellipse that follows the panel's proportions for free.
   * (No backticks in here: this is a JS template literal, and one would end it.)
   */
  vec2 d = (vUv - 0.5) * 2.0;
  float r = length(d);
  float e = 1.0 - smoothstep(uEdge, 1.0, r);
  float soft = e * e * (3.0 - 2.0 * e);
  if (soft <= 0.002) discard;

  float radius = uBlur * soft;
  vec3 acc = vec3(0.0);
  float wsum = 0.0;
  for (int j = -2; j <= 2; j++) {
    for (int i = -2; i <= 2; i++) {
      vec2 o = vec2(float(i), float(j)) * uTexel * radius;
      float w = exp(-float(i * i + j * j) * 0.35);
      acc += texture2D(uBack, su + o).rgb * w;
      wsum += w;
    }
  }
  vec3 c = acc / wsum;

  /**
   * Show the frost ONLY where it has something to blur.
   *
   * Without this the glass is visible as a faint veil even over empty backdrop,
   * and that veil is what read as "a shadow behind the middle two windows".
   * Three things stack up there and nowhere else: those two panels sit between
   * two neighbours, so THREE of these soft blobs overlap on them; each blob
   * composites separately, so the veil adds up; and both panels carry only one
   * or three rows of text, leaving a wide empty area for the result to show in.
   * Measured over open backdrop it was +5/255 on every channel.
   *
   * The veil exists because the source is the raw scene render — it never goes
   * through the composer's grade or bloom — so pasting it over the graded frame
   * shifts the colour slightly even where the two should agree. Rather than try
   * to reproduce the grade here (which would then have to track every LOOK
   * change), gate the effect on how much the blur actually CHANGED: on flat
   * backdrop the blurred and sharp samples are equal, the gate is zero, and
   * nothing is drawn at all.
   * No veil, no accumulation where blobs overlap, and no grade mismatch —
   * all three vanish for the same reason. Where there ARE lines behind the glass
   * the gate opens and they soften, which is the whole point of the effect.
   */
  vec3 sharp = texture2D(uBack, su).rgb;
  float detail = length(c - sharp);
  float gate = smoothstep(0.0, max(uDetail, 1e-4), detail);
  float a = uOpacity * soft * gate;
  if (a <= 0.002) discard;

  c = mix(c, uTint, uTintAmt * soft);
  gl_FragColor = vec4(c, a);
}
`;

/**
 * Floating holographic UI panels flanking her — schedule, mail, tasks, system
 * status, identification, notifications.
 *
 * These do the job the drawn room could not. A room states depth by perspective,
 * which a flat image reproduces perfectly, so it never stopped looking like a
 * picture of a room. Panels state it by being OBJECTS at several distinct
 * distances, at angles, partly hidden behind her — and by being the kind of
 * object a viewer already believes floats in mid-air, so there is no expectation
 * of a supporting surface to violate.
 *
 * They also carry the identity of the thing: this is a secretarial AI's console,
 * so the content is her actual work. Text is drawn to a canvas rather than built
 * from geometry — thousands of line segments for glyphs would cost more than the
 * whole rest of the space and still look worse at this size.
 *
 * Fill is kept very low and the structure is carried by thin bright lines: a
 * solid bright rectangle is a lit area, and lit areas make the panel read as a
 * display rather than as something hanging in the air.
 */

/**
 * `x` here is where the panel would like to be, NOT where it ends up: `fitToFrame`
 * pulls each one inward until it is actually on screen (see below).
 *
 * Hard-coded positions were tried twice and failed twice. The frame is far narrower
 * than it looks while authoring — the canvas is 1:3, not 9:16, so at panel depth the
 * visible half-width is about a third of a metre — and two separate things move the
 * goalposts: each panel is rotated about Y, so its inner edge swings toward the
 * camera and perspective magnifies it, and `framing.ts` re-solves the camera every
 * frame from the body fit, overwriting any assumed position. Arithmetic against an
 * assumed camera produced "SCHEDULE" -> "ULE" and then "ODULE".
 */
interface PanelSpec {
  title: string;
  rows: { label: string; value?: string; bar?: number }[];
  /** Metres. x is signed: negative = viewer's left. */
  x: number;
  y: number;
  z: number;
  w: number;
  h: number;
}

const PANELS: PanelSpec[] = [
  {
    title: "SCHEDULE",
    rows: [
      { label: "09:00", value: "Meeting" },
      { label: "10:30", value: "Briefing" },
      { label: "13:00", value: "Analysis" },
      { label: "15:30", value: "Review" },
      { label: "17:00", value: "Sys check" },
    ],
    x: -0.339, y: 1.74, z: -1.3, w: 0.42, h: 0.34,
  },
  {
    title: "MAIL",
    rows: [{ label: "NEW MESSAGE", value: "07" }],
    x: -0.411, y: 1.22, z: -1.6, w: 0.36, h: 0.19,
  },
  {
    title: "TASK LIST",
    rows: [
      { label: "Project Alpha", bar: 0.78 },
      { label: "Database", bar: 0.62 },
      { label: "Client Response", bar: 0.9 },
      { label: "Document Check", bar: 0.45 },
      { label: "Maintenance", bar: 0.3 },
    ],
    x: -0.336, y: 0.66, z: -1.4, w: 0.46, h: 0.34,
  },
  {
    title: "SYSTEM STATUS",
    rows: [
      { label: "ALL SYSTEMS", value: "98%" },
      { label: "NORMAL" },
    ],
    x: 0.351, y: 1.76, z: -1.32, w: 0.4, h: 0.26,
  },
  {
    title: "IDENTIFICATION",
    rows: [
      { label: "USER", value: "AICCORE-MIKU" },
      { label: "ID", value: "39-01-3941" },
      { label: "CLEARANCE", value: "LEVEL 5" },
    ],
    x: 0.387, y: 1.24, z: -1.62, w: 0.42, h: 0.24,
  },
  {
    title: "NOTIFICATION",
    rows: [
      { label: "Server Sync", value: "Complete" },
      { label: "Network", value: "Stable" },
      { label: "Security", value: "Secure" },
      { label: "Backup", value: "OK" },
    ],
    x: 0.353, y: 0.68, z: -1.42, w: 0.43, h: 0.3,
  },
];

const PX_PER_M = 620; // canvas resolution; panels are read at ~1.5 m so this is ample

/**
 * Which elements to draw.
 *
 * Split because the glow layer needs a DIFFERENT blur radius for the big marks
 * than for the small ones, and one radius cannot serve both. Measured: at
 * panelGlow 2.0 with a single 15 px blur, the schedule rows stopped being
 * readable — the row font is about 20 px on the same canvas, so a halo of that
 * size fills the gaps between glyphs and between lines and the text turns to
 * smear. The frame brackets want a wide soft halo; the rows want a tight one.
 */
type Part = "all" | "structure" | "rows";

function drawPanelContent(
  g: CanvasRenderingContext2D, W: number, H: number, spec: PanelSpec,
  part: Part, glow: boolean,
) {
  const cyan = "#7fe6ff";
  const cyanDim = "rgba(127,230,255,0.55)";
  /**
   * Row labels are INK, and deliberately so.
   *
   * They were `rgba(214,245,255,0.92)` — a pale near-white — which is invisible
   * on this field for a reason that is the exact mirror of the one that made the
   * rings invisible: on a LIGHT ground, small text is legible by being DARKER
   * than its surround, not brighter. At 3x magnification the schedule times were
   * washed smears while the title beside them read perfectly, and the title is
   * the same colour it always was. Adding glow to pale text makes it worse still,
   * because the halo lifts the surround along with the glyph and what little
   * local contrast there was goes with it.
   *
   * So the panel is split by JOB, not by uniform styling: the frame, the title
   * and the data values are light and glow; the labels are ink and do not.
   */
  const label = "#2f9fd0";
  const wantStructure = part === "all" || part === "structure";
  const wantRows = part === "all" || part === "rows";
  const fill = !glow;

  // Glass: barely-there fill. Any more and it stops being glass and starts
  // being a screen.
  //
  // Skipped on the glow layer. Blurring a full-rect fill just returns the same
  // rect, so adding it there would put a flat lit rectangle behind each panel —
  // precisely the "lit area" this file exists to avoid. The glow has to come off
  // the EDGES and the TEXT, which is also what the reference shows.
  if (fill) {
    g.fillStyle = "rgba(120,210,255,0.055)";
    g.fillRect(0, 0, W, H);
  }

  const pad = W * 0.06;
  const ruleY = pad * 0.8 + H * 0.155;

  if (wantStructure) {
    // Frame — corner brackets rather than a closed box, so the panel edge does
    // not draw a hard boundary the eye can latch onto as "the edge of the image".
    g.strokeStyle = cyan;
    g.lineWidth = Math.max(2, W * 0.006);
    const k = Math.min(W, H) * 0.16;
    [[0, 0, 1, 1], [W, 0, -1, 1], [0, H, 1, -1], [W, H, -1, -1]].forEach(([x, y, sx, sy]) => {
      g.beginPath();
      g.moveTo(x + sx * k, y);
      g.lineTo(x, y);
      g.lineTo(x, y + sy * k);
      g.stroke();
    });

    // Title + rule. The title is set large enough to survive the wide blur, so
    // it belongs with the structure rather than with the rows.
    g.fillStyle = cyan;
    g.font = `600 ${Math.round(H * 0.115)}px "Segoe UI", system-ui, sans-serif`;
    g.textBaseline = "top";
    g.fillText(spec.title, pad, pad * 0.8);
    g.strokeStyle = cyanDim;
    g.lineWidth = Math.max(1, W * 0.003);
    g.beginPath();
    g.moveTo(pad, ruleY);
    g.lineTo(W - pad, ruleY);
    g.stroke();
  }

  if (!wantRows) return;

  // Rows
  const rowH = (H - ruleY - pad) / Math.max(spec.rows.length, 1);
  const fs = Math.round(Math.min(rowH * 0.5, H * 0.095));
  g.font = `400 ${fs}px "Segoe UI", system-ui, sans-serif`;
  g.textBaseline = "top";
  spec.rows.forEach((r, i) => {
    const y = ruleY + pad * 0.4 + i * rowH;
    // Labels are ink, so they are absent from the glow layer entirely — see the
    // note on `label`. Values and bars are the data highlight and do glow.
    if (!glow) {
      g.fillStyle = label;
      g.fillText(r.label, pad, y);
    }
    if (r.value) {
      g.fillStyle = cyan;
      const tw = g.measureText(r.value).width;
      g.fillText(r.value, W - pad - tw, y);
    }
    if (r.bar !== undefined) {
      const bx = W * 0.56;
      const bw = W - pad - bx;
      const by = y + fs * 0.35;
      const bh = Math.max(2, H * 0.018);
      g.fillStyle = "rgba(127,230,255,0.20)";
      g.fillRect(bx, by, bw, bh);
      g.fillStyle = cyan;
      g.fillRect(bx, by, bw * r.bar, bh);
    }
  });
}

/**
 * Render a panel to a canvas.
 *
 * `padPx`/`blurPx` produce the GLOW layer: the same drawing, blurred, on a
 * canvas with room around it for the blur to spread into. Two layers rather than
 * one, because the two jobs conflict — the base layer has to stay crisp to be
 * legible at this size, and a glow is by definition not crisp. Compositing a
 * blurred copy additively behind the sharp one is the standard way out and keeps
 * the text exactly as readable as it was.
 *
 * The blur is `ctx.filter`, which applies per drawing operation, so each stroke
 * and glyph gets its own halo rather than the whole panel being smeared once.
 * Chromium-only, which is the only engine this ships on (Electron + the PROTO
 * unit's browser).
 *
 * The glow is drawn in TWO passes at different radii — wide for the frame and
 * title, tight for the rows. See `Part`: one radius made the small text
 * unreadable as soon as the glow was strong enough to be worth having.
 */
function drawPanel(spec: PanelSpec, padPx = 0, blurPx = 0): HTMLCanvasElement {
  const c = document.createElement("canvas");
  const W = Math.round(spec.w * PX_PER_M);
  const H = Math.round(spec.h * PX_PER_M);
  c.width = W + padPx * 2;
  c.height = H + padPx * 2;
  const g = c.getContext("2d")!;
  g.clearRect(0, 0, c.width, c.height);
  g.translate(padPx, padPx);
  if (blurPx <= 0) {
    drawPanelContent(g, W, H, spec, "all", false);
    return c;
  }
  g.filter = `blur(${blurPx}px)`;
  drawPanelContent(g, W, H, spec, "structure", true);
  g.filter = `blur(${Math.max(1.5, blurPx * 0.22)}px)`;
  drawPanelContent(g, W, H, spec, "rows", true);
  return c;
}

interface Panel {
  /** Carries the position and the inward rotation; both layers hang off it. */
  group: THREE.Group;
  /** The crisp layer. Also the one whose corners `fitToFrame` projects. */
  mesh: THREE.Mesh;
  mat: THREE.MeshBasicMaterial;
  glowMat: THREE.MeshBasicMaterial;
  frostMat: THREE.ShaderMaterial;
  spec: PanelSpec;
  baseY: number;
  bob: number;
  phase: number;
}

/** Keep this much of the frame clear at each side, in NDC. */
const EDGE_MARGIN = 0.04;
/** Never pull a panel further in than this — see the note in fitToFrame. */
const MIN_FIT_SCALE = 0.5;
/**
 * Fraction of the orbit the fit is solved for.
 *
 * 1 = solve for the very ends of the swing, which is what you want whenever it
 * is achievable. Whether it is depends entirely on the orbit angle: a point
 * `d` behind her moves `d*sin(theta)` sideways, and at the panels' depth the
 * frame is only about +-0.40 m wide. At the shipped 7.5 degrees that is 0.19 m
 * against a 0.19 m panel half-width, so a solution exists. At 15 it was 0.38 m
 * and none did — the search ran to zero and stacked all six panels on the
 * centre line. If the orbit is ever pushed back up, drop this to ~0.5 and
 * expect the ends of the swing to clip.
 */
const FIT_ORBIT_FRAC = 1.0;
/** How often to re-solve. The camera drifts slowly, so this is imperceptible. */
const REFIT_SEC = 0.5;

/**
 * Pull each panel toward the centre of the frame until all four of its corners
 * project inside. Bisects on a scale factor rather than solving in closed form,
 * because the corner that goes off-screen is the one perspective has magnified —
 * the closed form for that is exactly what kept being got wrong, and a dozen
 * projections per panel costs nothing at this count.
 *
 * Scales BOTH axes about the point the frame centre passes through at that panel's
 * depth. An X-only version shipped first and immediately broke the other way: a
 * camera change pushed the schedule rows off the TOP instead, which looked like a
 * different bug and was the same one.
 */
function fitToFrame(panels: Panel[], camera: THREE.PerspectiveCamera, poses: THREE.Vector3[]) {
  const v = new THREE.Vector3();
  const pivot = new THREE.Vector3();
  const dir = new THREE.Vector3();
  const limit = 1 - EDGE_MARGIN;

  const saveP = camera.position.clone();
  const saveQ = camera.quaternion.clone();
  /** Put the camera on one of the sampled orbit positions, aimed at her. */
  const usePose = (i: number) => {
    camera.position.copy(poses[i]);
    camera.lookAt(driftTarget);
    camera.updateMatrixWorld(true);
  };
  const basePoseIdx = Math.floor(poses.length / 2);

  usePose(basePoseIdx);
  for (const p of panels) {
    const { w, h, x, y } = p.spec;
    // Where the centre of the frame lands on this panel's depth plane. Scaling
    // about the world origin instead would drag the panels down as well as in,
    // because the camera is not looking at y=0.
    dir.set(0, 0, 0.5).unproject(camera).sub(camera.position).normalize();
    const t = Math.abs(dir.z) < 1e-6 ? 0 : (p.spec.z - camera.position.z) / dir.z;
    pivot.copy(camera.position).addScaledVector(dir, t);

    /**
     * Worst corner excursion over the WHOLE orbit, not just the resting view.
     *
     * The drift is a +-15 degree swing around her, and a panel solved for the
     * resting pose alone leaves the frame long before the swing ends — measured
     * before this, "SCHEDULE" and "SYSTEM STATUS" were both cut in half at
     * +-12 degrees. Fitting to the extremes instead keeps them in world space,
     * so they still parallax properly, while guaranteeing they stay on screen;
     * the alternative — re-fitting against the live camera — pins them to the
     * frame and makes them slide unnaturally while everything else rotates.
     */
    const worst = (scale: number) => {
      p.group.position.x = pivot.x + (x - pivot.x) * scale;
      p.group.position.y = pivot.y + (y - pivot.y) * scale;
      let m = 0;
      for (let q = 0; q < poses.length; q += 1) {
        usePose(q);
        for (let i = 0; i < 4; i += 1) {
          v.set(i & 1 ? w / 2 : -w / 2, i & 2 ? h / 2 : -h / 2, 0);
          p.mesh.localToWorld(v).project(camera);
          m = Math.max(m, Math.abs(v.x), Math.abs(v.y));
        }
      }
      return m;
    };

    let scale = 1;
    if (worst(1) > limit) {
      let lo = 0;
      let hi = 1;
      for (let i = 0; i < 12; i += 1) {
        const mid = (lo + hi) / 2;
        if (worst(mid) <= limit) lo = mid; else hi = mid;
      }
      /**
       * Floor the result.
       *
       * At a 15-degree orbit there is no scale that keeps these on screen, and
       * that is geometry rather than a tuning failure: the panels sit ~1.4 m
       * BEHIND her, the frame is only about +-0.40 m wide at that depth, and a
       * 15-degree swing moves anything at that depth ~0.34 m sideways. Scaling
       * pulls a panel toward the frame's centre LINE, but the line itself leaves
       * the frame, so the search runs all the way to zero and stacks all six
       * panels on top of each other behind her — far worse than letting them
       * clip. The floor keeps the layout intact and accepts that the outer
       * panels slide partly out of shot at the ends of the swing.
       */
      scale = Math.max(lo, MIN_FIT_SCALE);
    }
    p.group.position.x = pivot.x + (x - pivot.x) * scale;
    // baseY is what the bob animates around, so it has to follow the fit too or
    // the next frame puts the panel straight back where it did not fit.
    p.baseY = pivot.y + (y - pivot.y) * scale;
    p.group.position.y = p.baseY;
    usePose(basePoseIdx); // leave the pivot solve on the resting pose for the next panel
  }

  camera.position.copy(saveP);
  camera.quaternion.copy(saveQ);
  camera.updateMatrixWorld(true);
}

export function createPanels(
  camera?: THREE.PerspectiveCamera,
  renderer?: THREE.WebGLRenderer,
  scene?: THREE.Scene,
): SpaceModule {
  const root = new THREE.Group();
  root.name = "space-panels";
  const panels: Panel[] = [];

  // The backdrop the frost samples. Quarter resolution per axis: the result is
  // blurred anyway, so full resolution would be paying sixteen times the fill
  // rate to throw the detail away. Depth buffer off — nothing in this target is
  // occluded by anything the target does not itself contain.
  const backRT = new THREE.WebGLRenderTarget(4, 4, {
    depthBuffer: true,
    stencilBuffer: false,
    type: THREE.HalfFloatType,
  });
  backRT.texture.minFilter = THREE.LinearFilter;
  backRT.texture.magFilter = THREE.LinearFilter;
  backRT.texture.generateMipmaps = false;
  const rtSize = new THREE.Vector2(4, 4);
  const drawSize = new THREE.Vector2();

  PANELS.forEach((spec, i) => {
    const group = new THREE.Group();
    group.position.set(spec.x, spec.y, spec.z);
    // Angle each one inward toward the viewer's centre line, like a console
    // wrapped around the operator. Flat-on panels look like stickers.
    group.rotation.y = spec.x < 0 ? THREE.MathUtils.degToRad(26) : THREE.MathUtils.degToRad(-26);

    // --- Frost layer, furthest back: the blurred backdrop ---------------------
    const frostMat = new THREE.ShaderMaterial({
      uniforms: {
        uBack: { value: backRT.texture },
        uResolution: { value: new THREE.Vector2(1, 1) },
        uTexel: { value: new THREE.Vector2(1, 1) },
        uBlur: { value: SPACE.panelBlur },
        uOpacity: { value: SPACE.panelFrost },
        uTint: { value: new THREE.Color(SPACE.panelTint) },
        uTintAmt: { value: SPACE.panelTintAmt },
        uEdge: { value: SPACE.panelFrostEdge },
        uDetail: { value: SPACE.panelFrostDetail },
      },
      vertexShader: FROST_VERT,
      fragmentShader: FROST_FRAG,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    // Padded well past the panel: the radial blob has to reach zero out in open
    // space, not on the panel's own outline, or the shape gives itself away.
    const frostPadM = spec.h * SPACE.panelFrostPad;
    const frostMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(spec.w + frostPadM * 2, spec.h + frostPadM * 2),
      frostMat,
    );
    frostMesh.position.z = -0.004; // behind the glow, which is behind the content
    frostMesh.renderOrder = -14;
    // Named per layer so ndcBounds can tell them apart. Unnamed, all three
    // collapsed to the group's label and an overflow report could not say
    // whether the CONTENT had left the frame or merely its halo — and only the
    // first of those is a bug.
    frostMesh.name = `space-panel-${i}-frost`;
    group.add(frostMesh);

    // --- Glow layer, behind: additive, blurred, no glass fill ----------------
    // Measured before this existed: the panels put only +1.4/255 of luminance on
    // the field they sat on (the rings, once fixed, put +13.2), and a quarter of
    // their pixels were DARKER than it. They were flat graphics on a lavender
    // ground, not lit glass. Additive is what the s30 backdrop made possible —
    // see glow-lines.ts for the same argument in full.
    const padPx = Math.round(spec.h * PX_PER_M * SPACE.panelGlowPad);
    const glowTex = new THREE.CanvasTexture(
      drawPanel(spec, padPx, Math.max(2, padPx * 0.45)),
    );
    glowTex.colorSpace = THREE.SRGBColorSpace;
    glowTex.anisotropy = 4;
    const glowMat = new THREE.MeshBasicMaterial({
      map: glowTex,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      opacity: SPACE.panels * SPACE.panelGlow,
      side: THREE.DoubleSide,
      toneMapped: false,
    });
    const padM = padPx / PX_PER_M;
    const glowMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(spec.w + padM * 2, spec.h + padM * 2),
      glowMat,
    );
    glowMesh.position.z = -0.002; // strictly behind the crisp layer
    glowMesh.renderOrder = -13;
    glowMesh.name = `space-panel-${i}-glow`;
    group.add(glowMesh);

    // --- Crisp layer, in front: unchanged, carries legibility ----------------
    const tex = new THREE.CanvasTexture(drawPanel(spec));
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 4;
    const mat = new THREE.MeshBasicMaterial({
      map: tex,
      transparent: true,
      depthWrite: false, // they overlap each other and the rings; no z-fighting
      opacity: SPACE.panels,
      side: THREE.DoubleSide,
      toneMapped: false,
    });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(spec.w, spec.h), mat);
    mesh.renderOrder = -12;
    mesh.name = `space-panel-${i}-content`;
    group.add(mesh);

    root.add(group);
    panels.push({
      group, mesh, mat, glowMat, frostMat, spec,
      baseY: spec.y, bob: 0.09 + (i % 3) * 0.04, phase: i * 1.3,
    });
  });

  /**
   * Fill `backRT` with the space layer, panels excluded.
   *
   * Runs from the module's own update, which the viewer calls once per frame
   * BEFORE `composer.render()` — so the texture the panels sample is this
   * frame's background, not last frame's. Every piece of renderer state it
   * touches (render target, camera layer mask) is restored, because the
   * composer runs immediately afterwards and inherits whatever is left behind.
   */
  const renderBackdrop = () => {
    if (!renderer || !scene || !camera) return false;
    renderer.getDrawingBufferSize(drawSize);
    const w = Math.max(8, Math.floor(drawSize.x / SPACE.panelRtDiv));
    const h = Math.max(8, Math.floor(drawSize.y / SPACE.panelRtDiv));
    if (w !== rtSize.x || h !== rtSize.y) {
      rtSize.set(w, h);
      backRT.setSize(w, h);
    }

    const prevTarget = renderer.getRenderTarget();
    const prevMask = camera.layers.mask;
    root.visible = false;              // no feedback: the glass must not sample itself
    camera.layers.set(SPACE_LAYER);    // and must not sample the character in front of it
    renderer.setRenderTarget(backRT);
    renderer.clear();
    renderer.render(scene, camera);
    renderer.setRenderTarget(prevTarget);
    camera.layers.mask = prevMask;
    root.visible = true;
    return true;
  };

  let sinceFit = REFIT_SEC; // solve on the first frame
  const fitPoses: THREE.Vector3[] = [];

  return {
    objects: [{ band: "near", object: root }],
    update(dt, elapsed, mood: SpaceMood) {
      root.visible = SPACE.panels > 0.001;
      if (!root.visible) return;
      sinceFit += dt;
      if (camera && sinceFit >= REFIT_SEC) {
        sinceFit = 0;
        /**
         * Solve against the UNDRIFTED camera.
         *
         * Fitting to the live camera makes the fit chase the drift, and the
         * panels end up pinned to the frame instead of sitting in the world.
         * Measured over t=6..18 with the camera swinging 0.040 m laterally: the
         * panels moved 0.0046 in NDC while the rings behind them moved 0.0645 —
         * fourteen times more, where the geometry says the panels should move
         * about two thirds as much as the rings. Objects that hold still while
         * everything behind them slides read as decals stuck to the glass, which
         * is the opposite of what the drift is for.
         *
         * Removing the drift for the duration of the solve keeps the guarantee
         * (they fit at the base framing) and lets them parallax around it. The
         * camera's ROTATION still includes the drift's re-aim, but that is under
         * half a degree at these amplitudes against a 0.04 NDC edge margin.
         */
        const deg = SPACE.cameraOrbitDeg * FIT_ORBIT_FRAC;
        fitPoses.length = 0;
        if (deg > 0.001) {
          fitPoses.push(orbitPose(new THREE.Vector3(), -deg));
          fitPoses.push(orbitPose(new THREE.Vector3(), 0));
          fitPoses.push(orbitPose(new THREE.Vector3(), deg));
        } else {
          fitPoses.push(driftBase.clone());
        }
        fitToFrame(panels, camera, fitPoses);
      }
      const frosted = SPACE.panelFrost > 0.001 && renderBackdrop();
      const gain = SPACE.panels * (mood?.energy != null ? 0.85 + mood.energy * 0.3 : 1);
      for (const p of panels) {
        const u = p.frostMat.uniforms;
        u.uResolution.value.copy(drawSize);
        u.uTexel.value.set(1 / Math.max(1, rtSize.x), 1 / Math.max(1, rtSize.y));
        u.uBlur.value = SPACE.panelBlur;
        u.uOpacity.value = frosted ? SPACE.panelFrost * gain : 0;
        u.uTintAmt.value = SPACE.panelTintAmt;
        u.uEdge.value = SPACE.panelFrostEdge;
        u.uDetail.value = SPACE.panelFrostDetail;
        (u.uTint.value as THREE.Color).set(SPACE.panelTint);
        // Slow independent bob. Perfectly static panels read as a printed
        // overlay; a few millimetres of drift each, out of phase, reads as
        // objects held in the air.
        p.group.position.y = p.baseY + Math.sin(elapsed * p.bob + p.phase) * 0.012;
        p.mat.opacity = gain;
        // Re-read every frame so __space({ panelGlow }) tunes live, the contract
        // the rest of this layer honours.
        p.glowMat.opacity = gain * SPACE.panelGlow;
      }
    },
    dispose() {
      panels.forEach((p) => {
        p.mat.map?.dispose();
        p.mat.dispose();
        p.mesh.geometry.dispose();
        p.glowMat.map?.dispose();
        p.glowMat.dispose();
        p.frostMat.dispose();
        p.group.children.forEach((ch: THREE.Object3D) => (ch as THREE.Mesh).geometry?.dispose());
      });
      backRT.dispose();
      root.clear();
    },
  };
}
