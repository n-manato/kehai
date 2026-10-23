import * as THREE from "three";
import { SPACE } from "./space-config";
import type { FlowState } from "./flow-state";
import type { SpaceModule } from "./types";

/**
 * Columns of 0 and 1 falling slowly behind her, layered in depth.
 *
 * The one place text is allowed: binary, as the visual shorthand for "machine".
 * Each glyph is a camera-facing quad sampling a two-cell atlas (0 | 1),
 * so a few thousand of them are one draw call and all motion is in the shader.
 *
 * Depth, for the PROTO panel (transparent LCD on a light box, WHITE see-through):
 * columns are spread from just behind her to ~8 m back, and the far ones are
 * smaller AND paler, so they dissolve into the real box behind the glass instead
 * of sitting on it as a flat pattern. She occludes the columns behind her.
 *
 * Nothing changes abruptly with state — neither the pace (speed changes made the
 * earlier stream read as busy) nor the glyphs (0 <-> 1 flips read as sparkle).
 * State shows only as an eased brightness of the trails:
 *   thinking  — the field brightens a little
 *   speaking  — and a little more with the voice
 */

// Where a glyph is at time t — shared by the colour pass and the TAA velocity pass
// so the two can never disagree about where a glyph is.
const FALL = /* glsl */ `
attribute vec2 aCorner;
attribute float aRow;
attribute float aCol;
uniform float uTop;
uniform float uHeight;
uniform float uSpacing;
uniform float uSpeed;
uniform float uFore;      // 1 = foreground layer: position.x is a SCREEN position
uniform float uCamZ;      // camera z, for the foreground's screen -> world mapping
uniform float uHalfW;     // tan(fov/2) * aspect: half the view width per metre of depth
float hash(float n) { return fract(sin(n) * 43758.5453); }
vec3 fallAt(float t) {
  float speed = uSpeed * (0.6 + 0.8 * hash(aCol * 7.13));
  float y = uTop - mod(aRow * uSpacing + t * speed + hash(aCol) * uHeight, uHeight);
  // Foreground columns hold a fixed place ON SCREEN (the frame edges), re-solved
  // every frame from the live camera — placing them once, from the camera as it
  // was when the scene was built, put them over her once the framing settled.
  float x = mix(position.x, position.x * (uCamZ - position.z) * uHalfW, uFore);
  return vec3(x, y, position.z);
}
`;

const VERT = /* glsl */ `
${FALL}
uniform float uTime;
uniform float uGlyphM;
uniform float uScale;
varying float vRow;
varying float vCol;
varying float vDepth;
varying float vY;
varying vec2 vUv;
varying float vOne;

void main() {
  // 0 or 1, decided ONCE per glyph here and handed down as an exact 0.0/1.0. It
  // used to be hashed per FRAGMENT from interpolated row/column varyings: the
  // interpolation is off by a few ulps, sin() of a ~4000 argument times 43758
  // turns that into a different random number, and glyphs flipped 0 <-> 1 as
  // they moved ("0から1、1から0の切り替わりが速すぎる").
  vOne = step(0.5, hash(aRow * 13.7 + aCol * 71.3));
  vec3 p = fallAt(uTime);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  // A camera-facing quad, not a point sprite: gl_PointSize is rounded to whole
  // pixels, so as the camera drifted a glyph's size stepped by a pixel at a time —
  // the flicker that remained (measured: ~0 with the camera still, 0.13-0.21 with
  // the drift on, TAA or not). A quad scales and moves by sub-pixel amounts.
  mv.xy += aCorner * uGlyphM;
  vDepth = -mv.z;
  vRow = aRow;
  vCol = aCol;
  vY = p.y;
  vUv = aCorner + 0.5;
  gl_Position = projectionMatrix * mv;
}
`;

const FRAG = /* glsl */ `
uniform sampler2D uAtlas;
uniform float uTime;
uniform float uGlow;
uniform float uBand;
uniform float uOpacity;
uniform float uTop;
uniform float uHeight;
uniform vec3 uColor;
uniform vec3 uFarColor;
uniform float uCamDist;
uniform float uFarFade;
uniform float uBlur;
uniform float uBlurBase;
varying float vRow;
varying float vCol;
varying float vDepth;
varying float vY;
varying vec2 vUv;
varying float vOne;

float hash(float n) { return fract(sin(n) * 43758.5453); }

void main() {
  // Which glyph: fixed per glyph. They used to re-roll (0 <-> 1) over time, and
  // with ~2300 glyphs on screen that was dozens of hard changes every second —
  // a constant sparkle ("数字がちらついてうざい"). State shows in brightness only.
  float one = step(0.5, vOne);
  // Depth of field: the farther a glyph, the blurrier (sampled from a coarser
  // mip). Defocus reads as distance even in a still frame.
  float far = clamp((vDepth - uCamDist) / uFarFade, 0.0, 1.0);
  float glyph = texture2D(uAtlas, vec2((vUv.x + one) * 0.5, vUv.y), far * uBlur + uBlurBase).a;

  // A brighter band travelling down each column. Its leading edge used to be a
  // sawtooth: a glyph went from 0.18 to full brightness in ONE frame as the head
  // reached it, and with 60 heads that was a constant twinkle over the field —
  // the flicker that remained after the TAA fix. Now it ramps in over ~6% of the
  // column and fades out over ~40%, so no glyph ever changes faster than the eye
  // reads as motion.
  float headSpeed = 0.35 + 0.3 * hash(vCol * 3.1);
  float head = fract(uTime * headSpeed * 0.25 + hash(vCol * 5.7));
  float pos = (uTop - vY) / uHeight;                 // 0 at the top, 1 at the floor
  float d = fract(head - pos + 0.5) - 0.5;            // + behind the head, - ahead of it
  float band = d < 0.0 ? smoothstep(-0.06, 0.0, d) : 1.0 - smoothstep(0.0, 0.4, d);
  // uBand 0 = every glyph at one steady level: any brightness that changes over
  // time read as flicker on this field ("それでもまだちらついて見える").
  float bright = mix(0.6, mix(0.3, 1.0, band), uBand) * (1.0 + 0.35 * uGlow);

  // Fade in at the top; run nearly down to the floor, so each column visibly
  // STANDS on it — the farther it is, the higher on screen its foot: the plain
  // perspective of things on a ground plane.
  float ends = smoothstep(0.0, 0.08, pos) * (1.0 - smoothstep(0.97, 1.0, pos));
  float a = glyph * bright * ends * uOpacity * (1.0 - 0.65 * far);
  if (a < 0.01) discard;
  gl_FragColor = vec4(mix(uColor, uFarColor, far), min(a, 1.0));
}
`;

/**
 * TAA motion vectors for the glyphs. The fall happens in the vertex shader, which
 * the velocity pre-pass's shared override material cannot see: every glyph read
 * as still, and TAA blended last frame's glyph in a few pixels above the new one —
 * doubled, smeared digits that shimmered ("数字がちらついてうざい"). This draws
 * the real motion (t vs the previous frame's t) in the pre-pass's own format,
 * glyph-shaped. Wired through userData.taaVelocityMaterial (taa.ts).
 */
const VEL_VERT = /* glsl */ `
${FALL}
uniform float uTime;
uniform float uPrevTime;
uniform float uGlyphM;
uniform float uScale;
uniform mat4 uCurrVP;
uniform mat4 uPrevVP;
varying vec4 vCurr;
varying vec4 vPrev;
varying float vGlyphSeed;
varying vec2 vUv;
void main() {
  vec3 p = fallAt(uTime);
  vec3 q = fallAt(uPrevTime);
  // A wrap (floor -> top) is a teleport, not motion: report it as still.
  if (q.y < p.y) q = p;
  vCurr = uCurrVP * modelMatrix * vec4(p, 1.0);
  vPrev = uPrevVP * modelMatrix * vec4(q, 1.0);
  vGlyphSeed = step(0.5, hash(aRow * 13.7 + aCol * 71.3)); // see vOne
  vUv = aCorner + 0.5;
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  mv.xy += aCorner * uGlyphM;
  gl_Position = projectionMatrix * mv;
}
`;

const VEL_FRAG = /* glsl */ `
uniform sampler2D uAtlas;
varying vec4 vCurr;
varying vec4 vPrev;
varying float vGlyphSeed;
varying vec2 vUv;
float hash(float n) { return fract(sin(n) * 43758.5453); }
void main() {
  float one = step(0.5, vGlyphSeed);
  if (texture2D(uAtlas, vec2((vUv.x + one) * 0.5, vUv.y)).a < 0.3) discard;
  vec2 vel = (vCurr.xy / vCurr.w - vPrev.xy / vPrev.w) * 0.5;
  // B = 2: tell the TAA resolve to take no history here at all (taa.ts, noHist).
  gl_FragColor = vec4(vel, 2.0, 1.0);
}
`;

function makeAtlas(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = 128;
  c.height = 64;
  const g = c.getContext("2d")!;
  g.fillStyle = "#fff";
  g.font = "bold 54px ui-monospace, 'DejaVu Sans Mono', Menlo, monospace";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText("0", 32, 34);
  g.fillText("1", 96, 34);
  const t = new THREE.CanvasTexture(c);
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.generateMipmaps = true;
  return t;
}

/**
 * `fore`: the FOREGROUND layer instead — a few columns between the viewer and her,
 * only at the left and right edges of the frame (never over her), large and
 * defocused. Out-of-focus things in front are the strongest depth cue a still
 * frame has: blurred near / sharp her / blurred far reads as three planes.
 */
export function createBinaryRain(
  state: FlowState,
  camera: THREE.PerspectiveCamera,
  renderer: THREE.WebGLRenderer,
  fore = false,
): SpaceModule {
  let s = (fore ? 2027 : 1010) >>> 0;
  const rnd = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  const cols = Math.max(1, Math.round(fore ? SPACE.rainFgColumns : SPACE.rainColumns));
  const rows = Math.max(1, Math.round(SPACE.rainRows));
  const n = cols * rows;
  const position = new Float32Array(n * 3);
  const aRow = new Float32Array(n);
  const aCol = new Float32Array(n);
  // Lanes: each column gets its own slot across the frame (u in -1..1, jittered
  // within the slot), and its depth is random. With x fully random too, columns
  // at different depths landed on the same spot on screen and their glyphs
  // overlapped into "11"/"10" doubles that read as noise.
  const lanes = Array.from({ length: cols }, (_, i) => i);
  for (let i = lanes.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rnd() * (i + 1));
    [lanes[i], lanes[j]] = [lanes[j], lanes[i]];
  }
  for (let c = 0; c < cols; c += 1) {
    // Depth, then a width that grows with it so the far layers still fill the
    // frame. The columns are all behind her.
    // 2-14 m behind her (was 1.2-8): with the camera still, the only depth left
    // is what the field itself states, so it is set further back and stretched
    // deep — near columns about three times the size of the far ones.
    // Biased NEAR (power > 1): the far field only reads as far next to columns
    // that are plainly close and large.
    let z = -(SPACE.rainNear + Math.pow(rnd(), 1.5) * (SPACE.rainFar - SPACE.rainNear));
    const u = ((lanes[c] + 0.5 + (rnd() - 0.5) * 0.4) / cols) * 2 - 1;
    let x = u * (1.4 + -z * 0.45);
    if (fore) {
      // Between her (z 0) and the camera, at the frame edges: x is the SCREEN
      // position (outer 30% of each half), mapped to world in fallAt().
      z = 0.9 + rnd() * 1.1;
      x = Math.sign(u || 1) * (0.7 + 0.3 * rnd());
    }
    for (let r = 0; r < rows; r += 1) {
      const i = c * rows + r;
      position[i * 3] = x;
      position[i * 3 + 2] = z;
      aRow[i] = r;
      aCol[i] = c;
    }
  }
  // Four vertices per glyph, all AT the glyph's column point; the corner offset
  // rides in its own attribute and only this module's shaders apply it. Not an
  // InstancedBufferGeometry: the passes that re-render the scene with a shared
  // override material (GTAO, depth of field) ignore custom instance attributes
  // and drew every glyph as a 1 m quad at the origin — which turned the floor
  // grey. Here an override material sees four coincident points: zero area.
  const vPos = new Float32Array(n * 4 * 3);
  const vCorner = new Float32Array(n * 4 * 2);
  const vRow = new Float32Array(n * 4);
  const vCol = new Float32Array(n * 4);
  const index = new Uint32Array(n * 6);
  const CORNERS = [-0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5];
  for (let i = 0; i < n; i += 1) {
    for (let k = 0; k < 4; k += 1) {
      const v = i * 4 + k;
      vPos[v * 3] = position[i * 3];
      vPos[v * 3 + 2] = position[i * 3 + 2];
      vCorner[v * 2] = CORNERS[k * 2];
      vCorner[v * 2 + 1] = CORNERS[k * 2 + 1];
      vRow[v] = aRow[i];
      vCol[v] = aCol[i];
    }
    index.set([i * 4, i * 4 + 1, i * 4 + 2, i * 4, i * 4 + 2, i * 4 + 3], i * 6);
  }
  const geo = new THREE.BufferGeometry();
  geo.setIndex(new THREE.BufferAttribute(index, 1));
  geo.setAttribute("position", new THREE.BufferAttribute(vPos, 3));
  geo.setAttribute("aCorner", new THREE.BufferAttribute(vCorner, 2));
  geo.setAttribute("aRow", new THREE.BufferAttribute(vRow, 1));
  geo.setAttribute("aCol", new THREE.BufferAttribute(vCol, 1));

  const atlas = makeAtlas();
  const uniforms = {
    uAtlas: { value: atlas },
    uTime: { value: 0 },
    uTop: { value: 3.2 },
    uHeight: { value: 3.2 },
    uSpacing: { value: 3.2 / rows },
    uSpeed: { value: SPACE.rainSpeed },
    uGlyphM: { value: SPACE.rainGlyphM },
    uScale: { value: 1000 },
    uGlow: { value: 0 },
    uBand: { value: SPACE.rainBand },
    uOpacity: { value: SPACE.rain },
    uColor: { value: new THREE.Color(SPACE.rainColor) },
    uFarColor: { value: new THREE.Color(SPACE.flowFarColor) },
    uCamDist: { value: 3 },
    uFarFade: { value: 12 },
    uBlur: { value: SPACE.rainBlur },
    uBlurBase: { value: fore ? SPACE.rainFgBlur : 0 },
    uFore: { value: fore ? 1 : 0 },
    uCamZ: { value: 3.5 },
    uHalfW: { value: 0.15 },
  };
  const mat = new THREE.ShaderMaterial({
    uniforms, vertexShader: VERT, fragmentShader: FRAG,
    transparent: true, depthWrite: false, toneMapped: false, side: THREE.DoubleSide,
  });
  const velMat = new THREE.ShaderMaterial({
    uniforms: {
      // the SAME uniform objects, so both passes always agree on the layout
      uAtlas: uniforms.uAtlas, uTime: uniforms.uTime, uTop: uniforms.uTop,
      uHeight: uniforms.uHeight, uSpacing: uniforms.uSpacing, uSpeed: uniforms.uSpeed,
      uFore: uniforms.uFore, uCamZ: uniforms.uCamZ, uHalfW: uniforms.uHalfW,
      uGlyphM: uniforms.uGlyphM, uScale: uniforms.uScale,
      uPrevTime: { value: 0 },
      uCurrVP: { value: new THREE.Matrix4() },
      uPrevVP: { value: new THREE.Matrix4() },
    },
    vertexShader: VEL_VERT,
    fragmentShader: VEL_FRAG,
    blending: THREE.NoBlending,
  });
  const points = new THREE.Mesh(geo, mat);
  points.frustumCulled = false;
  points.name = fore ? "space-binary-rain-fg" : "space-binary-rain";
  points.renderOrder = -6;
  points.userData.taaVelocityMaterial = velMat; // see taa.ts TaaVelocity.render
  const size = new THREE.Vector2();
  let prevElapsed = 0;

  return {
    objects: [{ band: fore ? "foreground" : "near", object: points }],

    update(dt, elapsed) {
      const w = state.w;
      velMat.uniforms.uPrevTime.value = prevElapsed || elapsed;
      prevElapsed = elapsed;
      uniforms.uTime.value = elapsed;
      // Eased over ~1.5 s: following the voice syllable by syllable pulsed the
      // whole field, which reads as flicker too.
      const glow = Math.min(1, w.voice * 3) * w.speak + 0.5 * w.think;
      uniforms.uGlow.value += (glow - uniforms.uGlow.value) * Math.min(1, dt * 0.7);
      uniforms.uSpeed.value = SPACE.rainSpeed;
      uniforms.uGlyphM.value = SPACE.rainGlyphM;
      uniforms.uOpacity.value = SPACE.rain * (fore ? SPACE.rainFgOpacity : 1);
      uniforms.uBlurBase.value = fore ? SPACE.rainFgBlur : 0;
      uniforms.uBand.value = SPACE.rainBand;
      uniforms.uBlur.value = SPACE.rainBlur;
      uniforms.uFarFade.value = SPACE.rainFar - SPACE.rainNear;
      uniforms.uColor.value.set(SPACE.rainColor);
      uniforms.uFarColor.value.set(SPACE.flowFarColor);
      uniforms.uCamDist.value = camera.position.length();
      uniforms.uCamZ.value = camera.position.z;
      uniforms.uHalfW.value = Math.tan((camera.fov * Math.PI) / 360) * camera.aspect;
      // Pixels per metre at 1 m: point sprites need it to keep glyphs metric.
      renderer.getDrawingBufferSize(size);
      uniforms.uScale.value = size.y / (2 * Math.tan((camera.fov * Math.PI) / 360));
      points.visible = SPACE.rain > 0 && (!fore || SPACE.rainFgOpacity > 0);
    },

    dispose() {
      geo.dispose();
      mat.dispose();
      velMat.dispose();
      atlas.dispose();
    },
  };
}
