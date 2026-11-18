/* eslint-disable no-underscore-dangle */
// JA -> EN DISPLAY names. Only the LABEL is translated: the Japanese strings
// are also PMX bone / morph KEYS (dict["上半身2"], morph lookups), so the
// values themselves must never change.
import UI_LABELS from "@/utils/ui-labels-ja-en.json";
import { memo, useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { BokehPass } from "three/examples/jsm/postprocessing/BokehPass.js";
import { SMAAPass } from "three/examples/jsm/postprocessing/SMAAPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { GTAOPass } from "three/examples/jsm/postprocessing/GTAOPass.js";
import {
  VRMLoaderPlugin, VRMUtils, VRMSpringBoneJoint,
  VRMLookAtBoneApplier, VRMLookAtRangeMap,
  VRMSpringBoneCollider,
  VRMSpringBoneColliderShapeSphere, VRMSpringBoneColliderShapeCapsule, type VRM,
} from "@pixiv/three-vrm";
import { mmdHairPhysics } from "../../utils/mmd-hair-physics";
import { vrmMotionPlayer } from "@/utils/vrm-motion-player";
import { vrmExpression } from "@/utils/vrm-expression";
import { buildVrmFromGltf } from "@/utils/glb-humanoid";
import { dedupeTextures, pruneZeroMorphTargets, describeVrmCost } from "@/utils/vrm-optimize";
import { faceMotion } from "@/utils/face-motion";
import { vrmLipSync } from "@/utils/vrm-lipsync";
import { aiFx } from "@/utils/ai-fx";
import { SmplReferenceFigure } from "@/utils/smpl-reference-figure";
import { SmplMeshFigure } from "@/utils/smpl-mesh-figure";
import { neuralSpace } from "@/utils/space/neural-space";
import { aiDebug, debugParams, flashRecorder } from "@/utils/debug/ai-debug";
import { SPACE_LAYER } from "@/utils/space/layers";
import { isProtoMode, setProtoMode } from "@/utils/space/proto-mode";
import { deviceProfile } from "@/utils/space/device-profile";
import { createFraming, headFollow } from "@/utils/space/framing";
import { createBodyDepth } from "@/utils/space/body-depth";
import { autoHide, refreshAutoHide } from "@/utils/space/auto-hide";
import { SPACE } from "@/utils/space/space-config";
import { taa, TaaResolve, TaaVelocity } from "@/utils/space/taa";
import { describePixelRatio, resolvePixelRatio, watchDevicePixelRatio } from "@/utils/space/pixel-ratio";
import { applyOffAxis, currentPanel, impliedPanelHeightM } from "@/utils/space/off-axis";
import { viewerTransform, viewerWorld } from "@/utils/space/viewer-transform";
import { diagnoseHeadTracking, headTracker, setupHeadTracking, type ServerFaces } from "@/utils/space/head-tracker";
import { calibrateHeadDistance, hidePanelCalibration, showPanelCalibration } from "@/utils/space/calibrate";
import { hideSpaceHud, showSpaceHud } from "@/utils/space/space-hud";
import { hideHeadPreview, showHeadPreview } from "@/utils/space/head-preview";
import { applyViewerLookAt, resetViewerLookAt, undoViewerLookAt } from "@/utils/space/look-at-viewer";
import { isRenderSource } from "@/utils/render-source";
import { useAiState } from "@/context/ai-state-context";

/** Materials hidden on load, by material name. Empty by default.
 *  `q03` was a candidate for the stray navy plate (64 verts weighted 100% to bone D7, whose
 *  D1..D9 ribbon chain lost its MMD physics in the export, leaving it 9.6cm in front of the
 *  body in bind space) — but rendering it ALONE produced nothing on screen, so it is not
 *  what is visible and hiding it would have been a no-op dressed up as a fix.
 *  Live: __vrmHide("name") / __vrmHide("name", false) to test a candidate without a rebuild. */
const HIDE_MATERIALS: string[] = [];

/** Materials that must not CAST shadows — see the castShadow note in the loader.
 *  EMPTY on purpose. ["q01","q02","q202","q302"] (the outer coat / skirt panels) did remove
 *  the rectangular block over the white inner skirt — A/B screenshots confirmed it — but it
 *  also stripped the broad skirt shadow off the thighs, and what was left there was the thin
 *  hem shadow of the still-casting panels. With shadeToony 1 (hard step) and shadeHue 300 /
 *  shadeHueMix 0.3 every shade band on this model is a crisp MAGENTA edge, so that thin hem
 *  shadow read as a red line down the back of the thigh. Chosen trade-off: keep the block,
 *  lose the line. Add names here only if the same A/B is re-run and the thigh stays clean. */
const NO_CAST: string[] = [];

/* ALPHA_BLEND_NOTE
 *  2026-08-13, hair-behind-the-outer-skirt investigation — RECORDED SO IT IS NOT
 *  RE-ATTEMPTED. The 11 alpha-blended materials (Hair01-03, Hairshadow, C01-C07)
 *  ship as transparent + depthWrite:true + alphaTest:0, which LOOKS like the
 *  textbook cause of "the hair is not drawn behind the transparent skirt": a
 *  fully transparent texel still writes depth and rejects whatever is behind it.
 *  Two repairs were tried and BOTH MEASURED AS NO-OPS, so neither is in the code:
 *    - alphaTest 0.1 on every alpha-blended material
 *    - depthWrite:false on the outer layers only (hair sheets keeping theirs)
 *  Method: paint the hair flat opaque magenta, hide every OPAQUE surface, and
 *  count magenta — so only hair lost to TRANSPARENT surfaces is counted, with a
 *  fresh page per variant. Erased: 34.2% with everything untouched, 34.3% with
 *  the depthWrite repair, 35.1% / 35.7% for the alphaTest and hair-depthWrite
 *  variants. All within noise of each other.
 *  Conclusion: depth-write is NOT what removes the hair here, and per-material
 *  depth toggles cannot fix it. Re-confirmed with the render-probe MCP server
 *  (noise floor 0.11%), which is the ONLY trustworthy instrument for this: hair
 *  on screen was 12.740% of frame before the depthWrite repair and 12.750%
 *  after — no effect.
 *  MEASURE HAIR THIS WAY, not with hand-rolled screenshot diffs. `hairColorMix`
 *  tints ONLY the hair materials, so render_compare between hairColorMix 0 and 1
 *  yields the exact set of pixels where hair reaches the screen:
 *    render_compare preset=fullbody
 *      A {look:{hairColorMix:0}}  B {look:{hairColorMix:1,hairColor:0xff0000}}
 *  Hand-rolled Playwright A/Bs CANNOT do this: the hair is physics-animated and
 *  keeps moving between shots, and injecting taaSettle:0 (needed for live
 *  material edits to appear) disables the determinism latch. Two "identical"
 *  renders differed by 134,920 px that way — larger than any effect measured.
 *  What IS fixed and verified: the mipmap exclusion below — alpha-shaped
 *  textures must not be mip-filtered, confirmed at texture level and by GPU A/B. */

/**
 * VRM (.vrm) viewer — a drop-in alternative to the Live2D / MMD canvas, used when
 * the selected character's model URL is a `.vrm` (see App.tsx). Renders a VRM 1.0
 * avatar with @pixiv/three-vrm and drives it with the audio co-speech gestures via
 * vrmMotionPlayer (SMPL-X -> VRM humanoid aim retarget).
 *
 * Look: a port of the MMD viewer's msToonCoordinator vivid-cel stage — a dark navy
 * backdrop, a neutral fill (so Miku keeps her own colours), and a dual "sublight"
 * layer injected straight into every MToon material's fragment shader: a fresnel
 * edge term gated left→cyan / right→magenta, composited NON-additively (only the
 * dominant colour per fragment) so the two never sum to white — the reference's
 * 「陰影・ハイライト・サブライトの色が混ざらない」. Because it's added to the MToon
 * output `col` at the silhouette edge (not fed through the diffuse lighting), it
 * tints only the rim and never floods whole surfaces. UnrealBloom then makes the
 * rims glow. Live-tunable via window.__vrmLook({...}) in the console (no rebuild).
 */
interface VRMViewerProps { url: string }

const LOOK = {
  bgInner: 0x191630, // navy centre
  bgOuter: 0x070510, // near-black edges
  // Low key + front/side fills = flat anime lighting (no part sits in deep MToon
  // shade). The MToon VRM (look13: outline off on hair, purple shade, cool matcap)
  // carries the cel look; these values were tuned against the reference render.
  ambient: 0.62,
  hemiSky: 0xd6dcff,
  hemiGround: 0x33305a,
  hemi: 0.38,
  keyColor: 0xfff4ea,
  key: 0.5,
  keyAzim: 21, // key light direction (deg; 0 = front, + = screen-right) ≈ the old fixed (3,6,8)
  keyElev: 35, // key light elevation (deg)
  fillFront: 0.22, // front fill so the bangs/face aren't in shade
  fillSide: 0.28, // L+R fills
  // Injected dual sublight (non-mixing). Wide + strong so the cyan(L)/magenta(R)
  // rim reads like the reference's dual sublight and blooms.
  subInt: 1.45, // sublight strength
  subPow: 1.5, // fresnel tightness (lower = wider/thicker band)
  subMix: 0.0, // 0 = colours don't mix (dominant only); 1 = additive
  coolColor: 0x3fe3f5, // cyan, left edge
  warmColor: 0xff4ad8, // magenta, right edge (brightened so it isn't out-lumaed by cyan)
  // Faint neutral fresnel edge under the colours (0 to disable).
  fresInt: 0.05,
  fresPow: 3.0,
  fresColor: 0xbfe6ff,
  bloomStrength: 0.5,
  bloomRadius: 0.6,
  bloomThreshold: 0.72,
  // Colour grade (cool cast + purple shadow lift + slight desaturate/contrast).
  gContrast: 1.05,
  gSat: 0.88,
  gWarmR: 1.0,
  gWarmG: 0.9,
  gWarmB: 1.06,
  gLiftR: 0.04,
  gLiftG: 0.02,
  gLiftB: 0.09,
  // --- A: self-shadows + filmic tone + vignette + depth-of-field (MMD parity) ---
  shadowStrength: 0.45, // cast-shadow darkness (0 = off)
  toneMode: 1, // 0 None, 1 ACESFilmic, 2 AgX, 3 Neutral
  toneExposure: 1.06,
  vignette: 0.6, // edge darkening (the video's framing)
  dofAperture: 0, // depth of field OFF by default (tunable); >0 blurs by distance
  dofBlur: 0.008,
  // --- B: chromatic aberration + film grain + letterbox bars (default off) ------
  chroma: 0, // radial RGB split (cyan/magenta edge fringing)
  grain: 0, // film grain amount
  bars: 0, // letterbox bar height (fraction, top & bottom)
  gTemp: 0, // colour temperature (- cooler, + warmer)
  gTint: 0, // tint (- green, + magenta)
  gSplit: 0, // teal-shadow / magenta-highlight split-tone
  // Skin light-response (MMD parity): the skin is kept OUT of the cyan/magenta
  // sublight (so it isn't tinted cold) and gets a warm emissive floor so it reads
  // as natural warm anime skin under the cool rim + grade ("肌にサブライトを当てない").
  skinWarmth: 0.18,
  // How much of the sublight/fresnel reaches the SKIN materials (0 = classic
  // 肌にサブライトを当てない; msTC video look wants some rim on the face).
  skinSub: 0,
  // Light REFLECTION / shine: scales the MToon matcap (sphere-map) — the anime
  // highlight that reads as light reflecting off the hair/eyes. 1 = as-authored.
  matcapInt: 1.0,
  // Separate cyan(cool)/magenta(warm) rim intensities + their view-space directions
  // (azimuth left/right, elevation up/down) — the MMD's リム(寒)/リム(暖) + 方向.
  subIntCool: 1.0,
  subIntWarm: 1.0,
  subCoolAzim: -45, // deg: 0 = front, +90 = screen-right
  subCoolElev: 0,
  subWarmAzim: 45,
  subWarmElev: 0,
  // Global posterize (the MMD's 粗さ/toonSteps): 0 = off, else N flat colour bands.
  toonSteps: 0,
  // Hard cel bands (the MMD's セル塗り段数): quantize the MToon SHADING (light
  // response) into N flat steps — real anime cel shading. 0/1 = MToon's own ramp.
  celBands: 0,
  // --- msToonCoordinator (msTC) reproduction -------------------------------
  // All neutral by default (0 = identical to the pre-msTC look); the msTC preset
  // button turns them on together. Sat/val knobs use the msTC convention:
  // 0.5 = unchanged, 0 = grey/black, 1 = pure/white.
  baseSat: 0.5, // ベース純色化: albedo saturation pivot (0.5 = as-authored) — the msTC「テクスチャ純色化」
  baseVal: 0.5, // ベース純色化: albedo luminance pivot (0.5 = as-authored)
  // Per-part hue rotation (deg) on the albedo — 動画の青い瞳に合わせる用 (虹彩のみ).
  hairHueShift: 0,
  eyeHueShift: 0,
  // 髪色の直接指定: 動画からスポイトした色に髪のalbedoをブレンドする (0 = 元の髪色).
  hairColor: 0x6586be, // 動画実測: 髪 lit #6586BE (H218)
  hairColorMix: 0,
  skinShade: 1, // 肌の影強度: scales the msTC shade hue/sat/val toward neutral on SKIN only
  // (1 = same as everything else; lower = softer pink, keeps the leg/collar cast
  // shadows from reading as neon magenta slabs)
  // 作者「口幅縮小」相当: このVRMには口幅縮小モーフが無い(PMX→VRM欠落)ので、
  // 口周り頂点(=「あ」で動く頂点)をX中心へ寄せて口を小さくする頂点操作で近似。
  // 0 = 元の口幅。作者値0.35に対し見た目一致するよう控えめスケール。
  mouthNarrow: 0,
  faceLit: 0, // 顔マスク: extra +shadingShift on SKIN materials only — the article's ShadeRT R-mask
  // that keeps the face out of the N·L shade (only cast shadows, e.g. bangs, still land)
  shadeCoord: 0, // 影1: 0 = authored shadeColorFactor, 1 = per-pixel HSV shade derived from the albedo (色トレス)
  shadeHue: 250, // 影1色相 (deg): hue the shade is pulled towards (250 = blue-purple)
  shadeHueMix: 0.35, // 影1色相適用: how far the shade hue is pulled to shadeHue
  shadeSat: 0.55, // 影1彩度 (0.5 neutral)
  shadeVal: 0.62, // 影1明度 (0.5 neutral)
  shadeShift: 0, // 影しきい値 offset added to the authored shadingShiftFactor (+ = more shade)
  shadeToony: -1, // 影とがり: -1 = authored shadingToonyFactor, else 0..1 override (1 = hard 2-tone)
  shadowHard: 0, // 落ち影2値化: snap the PCF cast-shadow to a hard toon boundary (msTC: 落ち影と陰が同質)
  shadowRes: 0, // shadow map resolution: 0 = 1024, 1 = 2048, 2 = 4096 (ExcellentShadow parity)
  hiCoord: 0, // ハイ1: 0 = additive matcap (current), 1 = replace-style highlight (keep sat, lift val)
  hiThresh: 0.35, // ハイ1しきい値 on the matcap/rim luminance
  hiHue: 0, // ハイ1色相 (deg; only applies via hiHueMix)
  hiHueMix: 0, // ハイ1色相適用 (fx default 0)
  hiSat: 0.5, // ハイ1彩度 (0.5 neutral)
  hiVal: 0.75, // ハイ1明度 (0.5 neutral; >0.5 = lifted; fx default 0.82)
  subCoord: 0, // サブ光置換: 0 = additive rim (current), 1 = msTC hue-REPLACEMENT sublight
  subThresh: 0.45, // Subしきい値: hard threshold on the rim mask (fx: Sub1 0.5 / Sub2 0.42)
  subSat: 0.5, // サブ彩度 (0.5 neutral) for the replacement colour
  subVal: 0.6, // サブ明度 (0.5 neutral) for the replacement colour
  outlineWidth: 1, // 輪郭太さ scale on the authored outlineWidthFactor (body/clothes only — hair/face are authored "none")
  // (see SCREEN_OUTLINE below — outlineWidth is now a multiplier on a SCREEN-space
  //  width with a sub-pixel floor, not on a width in metres)
  outlineColor: -1, // 輪郭色: -1 = authored outlineColorFactor, else hex override
  outlineMix: -1, // 輪郭ライト混合: -1 = authored outlineLightingMixFactor, else 0..1 (0 = flat colour, msTC-like)
  outlineAll: 0, // 1 = show the runtime-generated hair/face outlines too (MMD標準エッジ相当; the msTC Edge stage outlines everything)
  // Hair (twintail) spring physics. The exported VRM gives the twintails ZERO
  // gravity + low stiffness, so during motion the tips LAG the head and the 11-joint
  // chain amplifies the head's small (~11°) sway into a big tip sweep — the hair
  // "むしの触覚のように動く" (waves like insect antennae). Verified: rigid tip sweep
  // 0.055 vs low-stiffness 0.184 (3.3×). Higher stiffness makes each joint track the
  // head tighter, so the lag doesn't accumulate; gravity gives weight so it hangs.
  // Croquis-style depth line-art (深度エッジ): ink lines where the view depth
  // jumps — hair clump gaps, overlaps, silhouette. The dense internal line work
  // the msTC look gets from Croquis / MMD標準エッジ. 0 = off.
  lineInt: 0,
  lineThresh: 0.05, // depth-relative discontinuity threshold (lower = more lines)
  // ikDiffusion / msPowerDiffusion analogue: threshold-0 wide soft glow over the
  // whole frame — the "anime finish" haze. 0 = off.
  diffusion: 0,
  // SvSSAO/ImSSAO analogue (msTC's remaining shade supplier): GTAO contact
  // shading — darkens creases/contacts (neck under hair, skirt folds). 0 = off.
  ssao: 0,
  // 顔法線: blend the face mesh normals toward a sphere (the ウマ娘/ブルプロ face
  // normal trick, msTC v2.1's NORMAL_RATE_MODE) so face shading falls in clean
  // anime shapes instead of following the 3D nose/cheek geometry. 0 = authored.
  faceNormal: 0,
  // Croquis X/Z: normal-edge lines (garment creases, hair facet breaks) and
  // colour-difference lines. 0 = off.
  lineNInt: 0,
  lineZInt: 0,
  // AutoLuminous analogue: brightness boost on the HL (highlight sparkle)
  // materials so the global bloom makes them glow selectively. 0 = as-authored.
  glowInt: 0,
  // msTC v2.1 SSS: warm red band on the light/shade terminator. 0 = off.
  sssInt: 0,
  sssWidth: 0.18,
  // PostPointLight analogue: a world-space point light that feeds the COOL
  // sublight replacement mask (light pools on chest/face). 0 = off.
  pointInt: 0,
  pointX: 0,
  pointY: 1.25,
  pointZ: 0.4,
  pointRange: 0.9,
  // 落ち影2値化のボケ足 (the ExcellentShadow ボケ足): half-width of the
  // binarize transition around threshold 0.375.
  shadowFt: 0.045,
  // Comma list of material names forced to ハイ1 (Shade_Highlight1_Absolute.fx
  // analogue). Edited via __vrmForceHi("matName") in the console.
  forceHiMats: "",
  // -雾雾雾- 氏配布の YYB Miku 専用貼図 (Live2D Characters/vrm/mstc-masks/):
  // maskUse = hand-painted Shade/SubLight masks, pureTex = 純色化 albedo swap.
  maskUse: 0,
  pureTex: 0,
  // スカート(+腰の半透明パネル)を不透明にする。0 = 元の透明感 (reversible).
  skirtOpaque: 0,
  // 腕の半透明パネル(袖のオーバーレイ)を不透明にする。0 = 元の透明感.
  armOpaque: 0,
  // msGradation/PointGradation analogue: painterly screen-space gradient layers
  // (bright wash from the top, sinking colour from the bottom) — the Chinese
  // guide's "グラデーション図層の重ね合わせ" lighting. 0 = off.
  gradTopInt: 0,
  gradTop: 0x9fd8ff,
  gradBotInt: 0,
  gradBot: 0x2a1550,
  // 眉の露出 (2D感の要): draw the Facial materials (brows/lashes) over the
  // bangs by disabling their depth test. 0 = normal depth.
  browFront: 0,
  hairStiff: 12.0, // ROOT stiffness (HZ1). USER-PICKED (2026-07-13): locks the authored twintail silhouette at the base — the hair POSITION must not drift.
  hairTipSoft: 0.1, // TIP stiffness as a fraction of the root (stiffness falls root->tip). Lower = the ends flow more (physics feel); the soft tip can't vibrate fast, so no プルプル. Live: __vrmSpring({tipSoft}) or the 毛先 柔さ slider.
  hairCenter: 1, // 1 = simulate twintails+skirt in HIPS space (bangs in head space): body translation is cancelled but chest/head motion still DRIVES the hair = MMD-style なびき; 0 = world space (the broken export default). Live: __vrmSpring({center:0|1})
  hairDrag: 0.45, // velocity damping in AUTHORED units (standard VRM ~0.3-0.6; sub-step compensated per frame in the animate loop). Lower = more MMD bounce/follow-through; raise toward 0.9 if the ends oscillate too much.
  hairGrav: 0, // USER-PICKED: this model is AUTHORED with gravity 0 — its rest shape IS the intended
  // twintail silhouette. Any gravity drags the tails down and CHANGES THE HAIR POSITION
  // ("重力をいじくると髪の位置が変わってしまう"). The old 0.35 was an anti-float hack from the
  // undamped era; center-space sim + heavy drag now handle movement, so keep 0.
  // GESTURELSM damping only (hand-animated KAWAII clips always play at full 1.0).
  // The model's raw output shrugs the clavicles, twists the wrist off noisy finger
  // joints and turns the head off-camera, so these stay tuned down; the retarget
  // itself is accurate (measured 0.9-3.9 deg per-bone agreement).
  gestShoulder: 0.55, // clavicle / shoulder shrug
  gestGain: 0.92, // overall upper-body gesture strength (spine/arms)
  gestFinger: 0.12, // finger curl (GestureLSM finger joints jitter)
  ikPelvisRot: 0.5, // pelvis lean/turn
  ikPelvisTrans: 0.5, // weight shift
};

// Build stamp shown in the 🎨 panel title + console — instantly tells whether
// the client is running the latest deployed bundle.
/**
 * Outline width in SCREEN space, and the floor it may not go below.
 *
 * The model authors its outline in `worldCoordinates` — a width in METRES — and
 * that is what made it flicker. Measured on this scene: the character covers
 * 1115 px/m, the authored factor is 0.00048 m, the runtime-generated ones were
 * 0.0012 (body) / 0.0004 (face), and `LOOK.outlineWidth` (0.4 in the shipping
 * preset) scales them:
 *
 *     0.0012 * 1115 * 0.4 = 0.54 px        0.0004 * 1115 * 0.4 = 0.18 px
 *
 * Every outline in the scene was thinner than one pixel. Painting the outline
 * pure red and counting it in isolation: only 71% of its pixels reached full
 * coverage — the other 29% were partial — and the line was ENTIRELY ABSENT from
 * ~325 of the 1559 rows it should occupy. A line missing from a fifth of its own
 * length, half-covering a third of the pixels it does reach, appears and
 * disappears per pixel as she moves. That is the reported ギザギザ / ちらつき.
 *
 * It is NOT the GPU, and not a resolution problem: with the clock frozen the
 * renderer reproduces a frame to 0.000% (scripts/render-metrics.mjs --static).
 * Supersampling hid it; this fixes it.
 *
 * `screenCoordinates` removes the cause: the width stops depending on camera
 * distance, so it cannot thin out when she stands back. Values calibrated
 * against the same measurement — 0.002 matched worldCoordinates at multiplier
 * 1.0 (77% vs 79% full coverage, 1525 vs 1559 rows), i.e. the look this was
 * tuned for.
 *
 * WIDTH IS AN ART DECISION, AND IT IS NOT MINE. Measured, at multiplier 0.4:
 *
 *     factor   full coverage   rows carrying the line   (1544 = none missing)
 *     0.0006        72%              1284               <- the original look
 *     0.0010        76%              1486
 *     0.0015        79%              1544               <- thinnest with none missing
 *     0.0018        81%              1544
 *
 * `body` is set so the shipping multiplier lands on 0.0006: the thin line this
 * was tuned for, restored on request. That width still leaves ~260 rows without
 * a line, so some shimmer remains — it is the same trade the original made, made
 * knowingly this time. For the thinnest width with NO dropout, use 0.00375 here
 * (0.0015 after the 0.4 multiplier).
 *
 * What screenCoordinates buys even at this width: the line no longer thins as
 * the camera pulls back, so it cannot get WORSE with distance the way the
 * metres-based version did.
 *
 * `min` only stops a preset going below what was asked for here.
 */
const SCREEN_OUTLINE = { body: 0.0015, face: 0.0005, min: 0.0006 };

const LOOK_BUILD = "r26"; // r26: video preset cool rim (warmColor 0x9fd8ff, subInt 1.2)

/**
 * One-time migration of individual grade keys onto an already-saved look.
 *
 * Editing LOOK_PRESETS does not reach anyone who already has a look in
 * localStorage: the loader below keeps a saved look verbatim while `_build`
 * matches, which is the normal case. Bumping LOOK_BUILD would deliver it by
 * throwing away the WHOLE saved look — every 🎨 tweak with it. This carries over
 * only the listed keys.
 *
 * Stamped and saved immediately on apply, so it runs once and a later hand-tune
 * of the same key sticks instead of being reverted on the next reload. Bump the
 * rev (and edit the values) only for another correction of the same kind.
 *
 * rev 1: gSplit 0.25 -> 0.15. Reverted at the user's request — trading the
 * preset's cast away for hem accuracy was the wrong axis; rev 2 puts it back.
 * rev 2: gSplit back to 0.25.
 * rev 3: gSat 1.32 -> 1.15 (see the preset note), gSplit 0.25 kept in the fix so
 * a browser that only ever saw rev 1 converges to the same end state.
 */
const GRADE_REV = 3;
const GRADE_FIX: Partial<typeof LOOK> = { gSat: 1.15, gSplit: 0.25 };

/**
 * Alpha dilation ("alpha bleeding") for alpha-TESTED textures.
 *
 * q3.png — the printed circuit pattern on the sleeves and skirt — keeps its shape
 * entirely in the ALPHA channel. Its RGB channel outside that shape was never
 * cleaned: the top half is random static and the bottom half is a flat, fully
 * saturated cyan #16fffd. Byte-identical to the source MMD model's own
 * tex/q3.png (sha1 ac51cbe239de), so this is authoring debris everyone inherits,
 * not something the PMX->VRM conversion did.
 *
 * MMD never shows it because MMD does not mip model textures. We do, and the two
 * do not compose: mip generation box-filters RGB *and* alpha together, so
 * minified texels get an averaged alpha that can cross q301/q302's alphaTest of
 * 0.5 while their averaged RGB has been dragged toward that cyan fill. The
 * survivors are drawn at FULL opacity (alphaMode MASK does not blend), which is
 * the cyan outline around every white speck on the skirt and sleeves — the
 * 「白い斑点をふちどるように水色が入ってる」 report. Simulated against the GPU's own
 * box filter, the surviving texels run #d9dcec (cyan-ness +10.9) at mip 0 to
 * #b3dfeb (+50.7) at mip 3. 1,467,047 of the texture's 1,478,647 cyan texels sit
 * below the cutoff and are never meant to be rasterised at all.
 *
 * The fix is to give the invisible texels the colour of the artwork nearest them,
 * so averaging them in pulls in that artwork rather than the fill. THE ALPHA
 * CHANNEL IS NOT TOUCHED — silhouette, cutoff and antialiasing come out bit for
 * bit identical, and mip-mapped minification stays on, so the trim-line jaggies
 * this replaced stay fixed. It is a repair of data that should never have been
 * visible, not a change to the character's look.
 *
 * Fills by pyramid rather than by iterated 1-texel dilation: a 2048² texture that
 * is 70% transparent needs a radius in the tens of texels, which is hundreds of
 * passes over 4.2M texels. Level 0 is read straight from the ImageData, so only
 * the 1024² level and smaller ever get their own buffer (~21MB, transient).
 *
 * Returns the number of texels filled, or 0 if there was nothing to do.
 */
/**
 * Keyed on THREE.Source, not on the texture and not on the decoded image.
 *
 * The alpha-tested materials expose 42 texture slots over 30 distinct Texture
 * objects — but only SIX distinct Sources: `map` and `shadeMultiplyTexture`, the
 * base material and its generated `(Outline)` twin, and all eleven materials over
 * the face atlas share their pixels. Keying per texture ran the 3.8M-texel pass
 * 26 times and put 3.7s on model load.
 *
 * Source is also what makes one write enough: `Texture.image` is an accessor for
 * `Texture.source.data`, so assigning the dilated canvas once hands it to every
 * texture that shares the Source.
 */
const DILATED_SOURCES = new WeakMap<object, HTMLCanvasElement>();

/**
 * Coverage-preserving mip chain for a texture whose SHAPE lives in its alpha.
 *
 * A box filter conserves the MEAN alpha but not the solid core: measured on
 * q3.png — the sleeve/skirt speck print — the fully opaque fraction runs
 * 3.49% at mip 0, 1.64% at mip 3, 0.82% at mip 4 and 0.00% at mip 5, while the
 * mean sits at 31.7 the whole way down. So at any distance the specks stop being
 * solid and the hair behind the panel starts showing through them, which is what
 * 「髪の毛の透過処理が腕の斑点に残ってる」 is. Nothing about the render is wrong;
 * the prefilter simply cannot keep a sub-texel mark opaque.
 *
 * The standard answer (NVIDIA's alpha-test coverage rescale, Unity's "Preserve
 * Coverage") is to rebuild each level's alpha so the fraction of texels at or
 * above a solidity threshold matches level 0. Same family as the dilation above:
 * repair the DATA so the ordinary mip path is correct, instead of special-casing
 * the material or turning the prefilter off and taking the shimmer back.
 *
 * RGB is carried through the box filter untouched — only alpha is rescaled — so
 * colour, and the dilation already applied to it, are unaffected.
 */
const COVERAGE_MIPPED = new WeakSet<object>();

/** Greyscale shade-multiply copies for the print panels — see _printPanel. */
const GREY_SHADE_TEX = new WeakMap<object, any>();

function buildCoverageMips(tex: any, solid = 250): number {
  const src = tex?.image;
  const w0 = src?.width | 0;
  const h0 = src?.height | 0;
  if (!w0 || !h0 || (w0 & (w0 - 1)) !== 0 || (h0 & (h0 - 1)) !== 0) return 0;
  const key = tex.source ?? src;
  if (COVERAGE_MIPPED.has(key)) return 0;
  try {
    const cv0 = document.createElement("canvas");
    cv0.width = w0; cv0.height = h0;
    const cx0 = cv0.getContext("2d", { willReadFrequently: true });
    if (!cx0) return 0;
    cx0.drawImage(src, 0, 0, w0, h0);
    let cur = cx0.getImageData(0, 0, w0, h0).data;
    let w = w0, h = h0;

    // Solve the gain from an alpha HISTOGRAM instead of bisecting on it: the
    // fraction at or above `solid` after scaling by k is just the fraction at or
    // above solid/k before it, so the needed threshold can be read straight off
    // the suffix sums. Bisection re-scanned 4.2M texels 24 times per level and
    // put 4.1s on model load.
    const fracAtLeast = (data: Uint8ClampedArray, n: number) => {
      const hist = new Uint32Array(256);
      for (let i = 3; i < data.length; i += 4) hist[data[i]]++;
      const suffix = new Float64Array(257);
      for (let a = 255; a >= 0; a--) suffix[a] = suffix[a + 1] + hist[a];
      return (a: number) => suffix[Math.max(0, Math.min(256, a))] / n;
    };
    const target = fracAtLeast(cur, w0 * h0)(solid);
    if (target <= 0) return 0;

    const mips: HTMLCanvasElement[] = [cv0];
    while (w > 1 || h > 1) {
      const nw = Math.max(1, w >> 1);
      const nh = Math.max(1, h >> 1);
      const next = new Uint8ClampedArray(nw * nh * 4);
      for (let y = 0; y < nh; y++) {
        const y0 = y * 2, y1 = Math.min(y * 2 + 1, h - 1);
        for (let x = 0; x < nw; x++) {
          const x0 = x * 2, x1 = Math.min(x * 2 + 1, w - 1);
          const o = (y * nw + x) * 4;
          const a = (y0 * w + x0) * 4, b = (y0 * w + x1) * 4;
          const c = (y1 * w + x0) * 4, d = (y1 * w + x1) * 4;
          for (let ch = 0; ch < 4; ch++) {
            next[o + ch] = (cur[a + ch] + cur[b + ch] + cur[c + ch] + cur[d + ch]) / 4;
          }
        }
      }
      const at = fracAtLeast(next, nw * nh);
      let a0 = 255;
      while (a0 > 1 && at(a0) < target) a0--;
      const k = Math.min(64, Math.max(1, solid / a0));

      // Rescale a COPY. Feeding the boosted alpha into the next level compounds
      // the gain — measured on q3.png, the stored mean ran 31.7 -> 47.4 by mip 4
      // and 69.7 by mip 6 instead of 45.0 / 62.6. The chain has to keep filtering
      // the honest box result; only what is uploaded gets the coverage fix.
      const store = new Uint8ClampedArray(next);
      if (k > 1.0001) for (let i = 3; i < store.length; i += 4) store[i] = Math.min(255, store[i] * k);

      const cv = document.createElement("canvas");
      cv.width = nw; cv.height = nh;
      const cx = cv.getContext("2d");
      if (!cx) break;
      cx.putImageData(new ImageData(store, nw, nh), 0, 0);
      mips.push(cv);
      cur = next; w = nw; h = nh;
    }
    tex.mipmaps = mips;
    tex.generateMipmaps = false;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    tex.needsUpdate = true;
    COVERAGE_MIPPED.add(key);
    return mips.length;
  } catch {
    return 0;
  }
}

function dilateAlphaTexture(tex: any, cutoff: number): number {
  const src = tex?.image;
  const w = src?.width | 0;
  const h = src?.height | 0;
  if (!w || !h) return 0;
  // Every alphaTest material on this model shares one cutoff (0.5), and no
  // alpha-tested Source is also used by a material that ignores alpha, so a
  // single cached result is correct for all of them.
  const key = tex.source ?? src;
  if (DILATED_SOURCES.has(key)) return 0; // the Source already holds the canvas
  try {
    const cv = document.createElement("canvas");
    cv.width = w;
    cv.height = h;
    const cx = cv.getContext("2d", { willReadFrequently: true });
    if (!cx) return 0;
    cx.drawImage(src, 0, 0, w, h);
    const id = cx.getImageData(0, 0, w, h);
    const px = id.data;

    // SEED ON FULL OPACITY, not on the alphaTest cutoff.
    //
    // Seeding at the cutoff repairs only what is discarded, and that is not where
    // the visible fringe lives: the texels at alpha 128..249 are DRAWN (MASK does
    // not blend, so they go down at full opacity) and they were already composited
    // against the cyan fill when the texture was authored. Measured over q3.png's
    // 218,835 such texels — the edge of every white speck — #c5eafc, cyan-ness
    // +46.4, and a cutoff-seeded fill leaves that untouched at +46.4. Seeding on
    // full opacity instead carries the artwork's own colour into them: #e6ecfc,
    // +14.2.
    //
    // Still nothing but colour: the alpha channel is untouched either way, so the
    // silhouette and its antialiasing are unchanged and only the tint under the
    // edge moves.
    const OPAQUE = 250;
    let seeds = 0;
    for (let i = 3; i < px.length; i += 4) if (px[i] >= OPAQUE) seeds++;
    // Fall back to the cutoff if the texture has no fully opaque texel at all,
    // so a soft-edged mask still gets *some* repair rather than none.
    const T = seeds > 0 ? OPAQUE : Math.max(1, Math.min(255, Math.round(cutoff * 255)));

    // Pyramid of alpha-gated colour sums: [r, g, b, count] per texel.
    const levels: { w: number; h: number; d: Float32Array }[] = [];
    let lw = Math.max(1, w >> 1);
    let lh = Math.max(1, h >> 1);
    {
      const d = new Float32Array(lw * lh * 4);
      for (let y = 0; y < h; y++) {
        const oy = Math.min(lh - 1, y >> 1) * lw;
        for (let x = 0; x < w; x++) {
          const s = (y * w + x) * 4;
          if (px[s + 3] < T) continue; // only FULLY OPAQUE texels seed the fill
          const o = (oy + Math.min(lw - 1, x >> 1)) * 4;
          d[o] += px[s]; d[o + 1] += px[s + 1]; d[o + 2] += px[s + 2]; d[o + 3] += 1;
        }
      }
      levels.push({ w: lw, h: lh, d });
    }
    while (lw > 1 || lh > 1) {
      const p = levels[levels.length - 1];
      const nw = Math.max(1, p.w >> 1);
      const nh = Math.max(1, p.h >> 1);
      const d = new Float32Array(nw * nh * 4);
      for (let y = 0; y < p.h; y++) {
        const oy = Math.min(nh - 1, y >> 1) * nw;
        for (let x = 0; x < p.w; x++) {
          const s = (y * p.w + x) * 4;
          if (p.d[s + 3] === 0) continue;
          const o = (oy + Math.min(nw - 1, x >> 1)) * 4;
          d[o] += p.d[s]; d[o + 1] += p.d[s + 1]; d[o + 2] += p.d[s + 2]; d[o + 3] += p.d[s + 3];
        }
      }
      levels.push({ w: nw, h: nh, d });
      lw = nw;
      lh = nh;
    }

    let filled = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const s = (y * w + x) * 4;
        if (px[s + 3] >= T) continue;
        for (let l = 0; l < levels.length; l++) {
          const L = levels[l];
          const o = (Math.min(L.h - 1, y >> (l + 1)) * L.w + Math.min(L.w - 1, x >> (l + 1))) * 4;
          const n = L.d[o + 3];
          if (n === 0) continue; // nothing visible at this scale — go coarser
          px[s] = L.d[o] / n;
          px[s + 1] = L.d[o + 1] / n;
          px[s + 2] = L.d[o + 2] / n;
          filled += 1;
          break;
        }
      }
    }
    if (!filled) return 0;
    cx.putImageData(id, 0, 0);
    DILATED_SOURCES.set(key, cv);
    tex.image = cv; // === tex.source.data, so every sharer gets it too
    return filled;
  } catch {
    return 0; // a tainted or unreadable image must not break model loading
  }
}

// Skin materials on this model — kept off the sublight + given a warm floor.
const SKIN_MATS = /^(body01|face01|face02)$/i;
// Eye materials (白目・瞳・ハイライト板): excluded from the msTC recolor entirely
// — the classic 白目シャドウ問題. They also stop receiving cast shadows.
const EYE_MATS = /^(eyes?\b|eyeball|HL)/i;
// Mouth interior — same protection. NAMES ARE ENGLISH on this VRM (oral cavity /
// tooth / tongue); the old 日本語 regex never matched, so every mouth-interior
// fix silently no-op'd (that's why the thick mouth ring survived every attempt).
const MOUTH_MATS = /^(口腔|歯|舌|oral cavity|tooth|tongue)/i;
// Name match for the スカート不透明 toggle, alongside the skin-weight detection
// in applyLook. q203 is rigged off the スカート bones, so the name list must
// cover the q2xx panels explicitly (verified with __vrmMats()).
const SKIRT_MATS = /^(q02\b|q2\d|q3\d)/i;

// Pristine defaults (captured before any saved-look load) — for the 🎨 panel Reset.
const LOOK_DEFAULTS = { ...LOOK };

// The msTC pipeline starts from a FLAT base — the note article's 「ベース色」
// stage is SimplePaint: texture colour with no lighting gradient, no matcap,
// no fresnel, no colour grade, and under MMD's SINGLE main light. Every legacy
// "moody" cosmetic must be neutralised or it pollutes all later stages.
const FLAT_BASE: Partial<typeof LOOK> = {
  toneMode: 0, // no filmic tone map — flat colours stay flat
  toneExposure: 1,
  gContrast: 1,
  gSat: 1,
  gWarmR: 1, gWarmG: 1, gWarmB: 1,
  gLiftR: 0, gLiftG: 0, gLiftB: 0,
  gSplit: 0, gTemp: 0, gTint: 0,
  vignette: 0.25,
  grain: 0, chroma: 0, bars: 0,
  bloomStrength: 0.35,
  bloomThreshold: 0.8,
  matcapInt: 0, // SimplePaint has no sphere map
  fresInt: 0,
  celBands: 0, // banding comes from the msTC threshold, not the ramp
  toonSteps: 0,
  skinWarmth: 0, // base = pure texture colour
  // MMD's single main light. Ambient MUST be 0 for msTC-exact shade colours:
  // indirect light adds albedo on top of the replaced shade colour and muddies
  // it (shade = ambient·albedo + key·mstcShade unless ambient = 0).
  key: 0.95,
  keyColor: 0xffffff,
  ambient: 0,
  hemi: 0,
  fillFront: 0,
  fillSide: 0,
};

// Look presets (applied as LOOK_DEFAULTS + overrides): "base" = the tuned
// pre-msTC reference look; "mstc" = FLAT base + the EXACT defaults shipped in
// msTC v1.1 default_v1.1.vmd; "video" = FLAT base + start values aimed at the
// demo video's vivid magenta-shade look (oJjM3Y-xj0k) — tune from there.
const LOOK_PRESETS: Record<string, Partial<typeof LOOK>> = {
  base: {},
  // default_v1.1.vmd: 影1しきい値0.38 色相適用0 彩度0.59 明度0.33 / ハイ1しきい値0.69
  // 彩度0.5 明度0.82 / Sub1色相0.11(40°暖) 彩度0.31 明度0.69 / Sub2色相0.53(191°寒)
  // 彩度0.66 明度0.47 / エッジ = HSY(0,0,0.1) の絶対色 ≒ #191919
  mstc: {
    ...FLAT_BASE,
    shadeCoord: 1,
    shadeHue: 0,
    shadeHueMix: 0,
    shadeSat: 0.59,
    shadeVal: 0.33,
    shadeShift: 0.1,
    shadeToony: 1, // fx = pure hard step
    shadowHard: 1,
    shadowFt: 0.09, // 広めのボケ足 — ネクタイ等の細物の落ち影が自然に
    shadowRes: 2,
    hiCoord: 1,
    hiThresh: 0.5,
    hiSat: 0.5,
    hiVal: 0.82,
    subCoord: 1,
    subThresh: 0.45,
    subSat: 0.5,
    subVal: 0.58,
    coolColor: 0x2fd6e8, // Sub2 hue ≈ 191° cyan
    warmColor: 0xffb347, // Sub1 hue ≈ 40° warm orange
    outlineAll: 1, // MMD標準エッジ相当: 髪・顔にも輪郭
    outlineWidth: 1.2,
    outlineColor: 0x191919, // エッジ = HSY(0,0,0.1)
    outlineMix: 0,
  },
  // -雾雾雾- 氏の bilibili 4部作 (opus cv25257013) に載っていた YYB Miku 10th
  // 専用レシピの移植: 影1彩度0.36 影1明度0.38 ハイ1彩度0.17 ハイ1明度0.85
  // sss彩度0.35 + Croquis X1/Z1 + Diffusion/msPowerDiffusion/SoftFocus の
  // 重ねグロー + AutoLuminous弱 + 顔球面化。専用マスク配布: bowlroll file/304270.
  recipe: {
    ...FLAT_BASE,
    keyAzim: -25,
    keyElev: 30,
    baseSat: 0.45,
    baseVal: 0.58,
    eyeHueShift: 45,
    faceLit: 0.5,
    faceNormal: 0.7, // 顔の球面化 (チュートリアルで「重要」)
    shadeCoord: 1,
    shadeHue: 300,
    shadeHueMix: 0.2,
    shadeSat: 0.36, // 影1彩度 (レシピは彩度を下げる)
    shadeVal: 0.38, // 影1明度
    shadeShift: 0.16,
    skinShade: 0.6,
    skinSub: 0.5,
    shadeToony: 1,
    shadowHard: 1,
    shadowRes: 2,
    hiCoord: 1,
    hiThresh: 0.5,
    hiSat: 0.17, // ハイ1彩度 (白く飛ばす)
    hiVal: 0.85, // ハイ1明度
    sssInt: 0.15, // sss彩度 (0.35だと脚にオレンジの帯が出る)
    sssWidth: 0.12,
    subCoord: 1,
    subThresh: 0.4,
    subInt: 1.5,
    subPow: 1.3,
    subSat: 0.6,
    subVal: 0.58,
    mouthNarrow: 0.2, // 作者「口幅縮小」相当
    maskUse: 1, // 配布の手描きShade/SubLightマスク (bowlroll file/304270)
    pureTex: 1, // 配布の純色化テクスチャ
    lineInt: 0.85,
    lineThresh: 0.035,
    lineNInt: 0.3, // ComicalEdge補助相当 (球面化で消える顎線を深度/法線で拾う)
    lineZInt: 0, // 色差エッジは布の柄でちらつくのでOFF (純色化前提の機能)
    diffusion: 0.08, // 白ぽわぽわ対策
    bloomStrength: 0.22,
    bloomThreshold: 0.85,
    glowInt: 0.3, // AutoLuminous Si0.05 (弱め)
    ssao: 0.4,
    gContrast: 1.18, // msPowerOverlay/ikOverray のオーバーレイ相当
    gSat: 1.15,
    gLiftR: 0.04,
    gLiftG: 0.02,
    gLiftB: 0.1,
    gradTopInt: 0.06,
    gradBotInt: 0.16,
    browFront: 0, // Facial材質は頬の赤み等も含み髪を貫通して「顔が透けて」見える
    outlineAll: 1,
    outlineWidth: 2.0, // 動画の輪郭線は太め
    outlineColor: 0x151030,
    outlineMix: 0,
    bgInner: 0x120e2e,
    bgOuter: 0x050310,
  },
  video: {
    ...FLAT_BASE,
    key: 1.05, // whites bright but not blown
    keyAzim: -25, // the article/video lights from upper-LEFT (bangs shadow falls lower-right)
    keyElev: 30,
    baseSat: 0.55,
    baseVal: 0.52,
    mouthNarrow: 0.2, // 作者「口幅縮小」相当 (口を小さく)
    hairColorMix: 0, // 髪はデフォルト (元テクスチャ #276A91)。変えたい時は「髪色 適用」を上げる
    eyeHueShift: 45, // ティール緑 → 動画の青い瞳 (虹彩のみ)
    faceLit: 0.45, // 動画は頬・首にしっかり影が入る
    // Shade: mild hue pull + STRONG sat boost + dark. Chromatic materials snap
    // vivid (skin -> magenta, hair -> deep blue); achromatic whites just darken
    // and get their blue-violet from the grade's shadow lift below — that split
    // (pink skin shade / blue cloth shade) is how the reference frame reads.
    shadeCoord: 1,
    // 実測: 髪 #6586BE(H218)→影 #4348A6(H237) = 色相+19°・彩度ほぼ維持・明度×0.8。
    // hue300 へ 0.3 の RGB ブレンドがちょうど H243 に落ちる (肌は H318 マゼンタへ ✓)
    shadeHue: 300,
    shadeHueMix: 0.3,
    shadeSat: 0.55, // 彩度は上げない (実測 S0.42→0.42)
    shadeVal: 0.42, // 影は思ったより明るい (実測 B0.55→0.45)
    shadeShift: 0.2, // widen the shade band past the authored threshold
    skinShade: 0.75, // 動画の肌影はビビッドなマゼンタ (実測 #A76292 H318)
    skinSub: 0.6, // let the face take the rim replacement like the video
    skinWarmth: 0.12, // 肌に暖色エミッシブ床を軽く焼く (ユーザー指定)
    maskUse: 1, // hand-painted Shade/SubLight masks (bowlroll file/304270)
    pureTex: 1, // purified albedo swap
    shadeToony: 1,
    shadowHard: 1,
    shadowFt: 0.09, // 広めのボケ足 — ネクタイ等の細物の落ち影が自然に
    shadowRes: 2,
    hiCoord: 0.7,
    hiThresh: 0.5,
    hiSat: 0.5,
    hiVal: 0.72, // 実測: ハイライトは B0.55→0.68 の控えめリフト
    subCoord: 1,
    subThresh: 0.45, // ↑: 襟パイピング等の内側曲面に淡い帯が出ない高さ
    subInt: 1.2, // was 2.0: at full strength the two rim colours read as chromatic aberration
    subPow: 2.2, // 帯を細く — 真のシルエットエッジのみ (襟の「ずれた線」対策)
    subSat: 0.78,
    subVal: 0.5,
    coolColor: 0x37b8dc, // 実測: シアンのアクセント #4CAFCD/#84ADC6 系
    // Was 0xe060b8 (pink). Two opposed rim colours, one cyan and one magenta, put a
    // coloured fringe on every silhouette edge — indistinguishable from lens
    // chromatic aberration, and nothing like the holographic reference. Cooling the
    // warm side keeps the rim separation without the fringe.
    warmColor: 0x9fd8ff,
    // Grade: the reference's vivid look — contrast + saturation up, blue-violet
    // shadow lift (tints the achromatic cloth shadows), split-tone, vignette.
    gContrast: 1.25,
    // Was 1.32. At 1.32 the grade amplifies any faint hue the shading leaves on
    // near-white surfaces — the skirt hem's mint crushed to R=0 (hue error 46%)
    // and the sleeve prints flipped cyan in shade — and it does so on top of the
    // uWarm cast, so no single other knob could pull it back. 1.15 keeps the
    // vivid reading (measured: white cloth hue error 16.3% -> 13.8%, hem
    // 46.1% -> 27.5%) while the blue-violet convergence stays.
    gSat: 1.15,
    // 実測: 服の「白」は #B1A8D3 のラベンダー — 画面全体を青紫に収束させる
    // グレード (白→薄紫、G を沈めて B を持ち上げ)
    gWarmR: 0.84,
    gWarmG: 0.78,
    gWarmB: 1.06,
    gLiftR: 0.05,
    gLiftG: 0.03,
    gLiftB: 0.12,
    gSplit: 0.25,
    vignette: 0.6,
    bloomStrength: 0.22, // 「bloomで白みがかってぽわぽわ」対策
    bloomThreshold: 0.85,
    // 深度線画はOFF: 反転ハル輪郭と同じ境界に微妙にズレた線を二重に引いて
    // 「線が浮く/襟がずれる」の原因になる (ハルが全材質に付いた今は冗長)。
    lineInt: 0,
    lineThresh: 0.035,
    lineNInt: 0.3, // Croquis X: garment creases / hair facet lines
    lineZInt: 0, // 色差エッジは布の柄でちらつくのでOFF
    diffusion: 0.07, // 白ぽわぽわの正体 — ほぼ切る
    ssao: 0.45, // SvSSAO analogue: contact shading (neck/hair/skirt creases)
    faceNormal: 0.65, // anime face shading shapes
    glowInt: 0.5, // HL sparkles glow (AutoLuminous)
    sssInt: 0.1, // warm terminator band (高いと脚にオレンジの帯)
    sssWidth: 0.12,
    gradTopInt: 0.04, // painterly light wash (控えめ)
    gradBotInt: 0.2, // sinking violet from below
    browFront: 0, // Facial材質は頬の赤みも含み髪を貫通するためOFF (スライダーで任意ON)
    dofAperture: 0.0004, // subtle depth bokeh like the video
    dofBlur: 0.01,
    toneExposure: 0.98,
    outlineAll: 1,
    // 0.4 焼き込み (2026-07-29 ユーザー指定). 1.8 は PROTO の実機で線が
    // 主張しすぎ、白地・線主体の画づくりだと輪郭だけが浮いて「絵」に見えた。
    outlineWidth: 0.4,
    outlineColor: 0x1c1440, // 実測: 暗部は #2C245A 系の菫色に収束
    outlineMix: 0,
    bgInner: 0x1e163b, // 実測 #1E163B
    bgOuter: 0x110b22,
  },
};

// Directions are in view space (camera is fixed), so cyan sits on screen-left and
// magenta on screen-right regardless of the avatar's pose.
const COOL_DIR = new THREE.Vector3(-1, 0, -1).normalize();
const WARM_DIR = new THREE.Vector3(1, 0, -1).normalize();

// Post colour grade: cool cast (green down, blue up -> teal hair reads blue-grey),
// slight desaturate, a purple lift in the shadows, and mild contrast — the
// reference video's cool/lavender grade.
const GradeShader = {
  uniforms: {
    tDiffuse: { value: null },
    uContrast: { value: LOOK.gContrast },
    uSat: { value: LOOK.gSat },
    uWarm: { value: new THREE.Color(LOOK.gWarmR, LOOK.gWarmG, LOOK.gWarmB) },
    uShadowLift: { value: new THREE.Color(LOOK.gLiftR, LOOK.gLiftG, LOOK.gLiftB) },
    // ported from the MMD ToonShader for effect parity
    uChroma: { value: LOOK.chroma },
    uVignette: { value: LOOK.vignette },
    uGrain: { value: LOOK.grain },
    uBars: { value: LOOK.bars },
    uTemp: { value: LOOK.gTemp },
    uTint: { value: LOOK.gTint },
    uSplit: { value: LOOK.gSplit },
    uSteps: { value: LOOK.toonSteps },
    uTime: { value: 0.0 },
    uGradTopC: { value: new THREE.Color(LOOK.gradTop) },
    uGradTopI: { value: LOOK.gradTopInt },
    uGradBotC: { value: new THREE.Color(LOOK.gradBot) },
    uGradBotI: { value: LOOK.gradBotInt },
  },
  vertexShader:
    "varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 ); }",
  fragmentShader:
    "uniform sampler2D tDiffuse; uniform float uContrast, uSat, uChroma, uVignette, uGrain, uBars, uTemp, uTint, uSplit, uSteps, uTime, uGradTopI, uGradBotI; uniform vec3 uWarm, uShadowLift, uGradTopC, uGradBotC; varying vec2 vUv;\n" +
    "void main(){\n" +
    "  vec2 dir = vUv - 0.5;\n" +
    "  vec4 t;\n" +
    "  if ( uChroma > 0.001 ) {\n" + // chromatic aberration: R out / B in, stronger at edges
    "    vec2 off = dir * uChroma * 0.004;\n" +
    "    t.r = texture2D( tDiffuse, vUv + off ).r;\n" +
    "    t.g = texture2D( tDiffuse, vUv ).g;\n" +
    "    t.b = texture2D( tDiffuse, vUv - off ).b;\n" +
    "    t.a = texture2D( tDiffuse, vUv ).a;\n" +
    "  } else { t = texture2D( tDiffuse, vUv ); }\n" +
    "  vec3 c = t.rgb;\n" +
    "  c *= uWarm;\n" +
    "  c.r *= 1.0 + uTemp * 0.12; c.b *= 1.0 - uTemp * 0.12;\n" + // temperature
    "  c.r *= 1.0 + uTint * 0.06; c.b *= 1.0 + uTint * 0.06; c.g *= 1.0 - uTint * 0.06;\n" + // tint
    "  float l = dot( c, vec3( 0.299, 0.587, 0.114 ) );\n" +
    "  if ( uSplit > 0.001 ) {\n" + // split-tone: teal shadows / magenta highlights
    "    vec3 st = vec3( 0.10, 0.55, 0.62 ); vec3 hi = vec3( 0.72, 0.30, 0.78 );\n" +
    "    vec3 tc = mix( st, hi, smoothstep( 0.2, 0.8, l ) ) * 2.0;\n" +
    "    c = mix( c, c * tc, uSplit * 0.5 );\n" +
    "  }\n" +
    "  c = mix( vec3( l ), c, uSat );\n" +
    "  c += uShadowLift * ( 1.0 - smoothstep( 0.0, 0.5, l ) );\n" +
    "  c = ( c - 0.5 ) * uContrast + 0.5;\n" +
    "  if ( uSteps > 0.5 ) { float pl = dot( clamp( c, 0.0, 1.0 ), vec3( 0.299, 0.587, 0.114 ) ); float pq = floor( pl * uSteps + 0.5 ) / uSteps; c *= pq / max( pl, 0.001 ); }\n" + // value posterize (粗さ, hue-preserving)
    // msGradation-style painterly light layers: SCREEN-blend a bright wash from
    // the top, MULTIPLY a sinking colour from the bottom.
    "  if ( uGradTopI > 0.001 ) c = mix( c, 1.0 - ( 1.0 - c ) * ( 1.0 - uGradTopC ), uGradTopI * smoothstep( 0.35, 1.0, vUv.y ) );\n" +
    "  if ( uGradBotI > 0.001 ) c = mix( c, c * uGradBotC, uGradBotI * smoothstep( 0.55, 0.0, vUv.y ) );\n" +
    "  if ( uVignette > 0.001 ) c *= 1.0 - uVignette * smoothstep( 0.42, 0.95, length( vUv - 0.5 ) );\n" + // vignette
    "  if ( uGrain > 0.001 ) {\n" + // animated film grain
    "    float n = fract( sin( dot( vUv * ( uTime + 1.0 ), vec2( 12.9898, 78.233 ) ) ) * 43758.5453 );\n" +
    "    c += ( n - 0.5 ) * uGrain;\n" +
    "  }\n" +
    "  c = clamp( c, 0.0, 1.0 );\n" +

    "  if ( uBars > 0.0001 && ( vUv.y < uBars || vUv.y > 1.0 - uBars ) ) c = vec3( 0.0 );\n" + // letterbox
    "  gl_FragColor = vec4( c, t.a );\n" +
    "}",
};

// Croquis-style line pass (msTC's Edge supply): Y = depth edges (scene depth
// buffer), X = normal edges (dedicated normal pre-pass), Z = colour-difference
// edges (beauty luma). Runs right after the RenderPass, before the grade.
const LineShader = {
  uniforms: {
    tDiffuse: { value: null },
    tDepth: { value: null },
    tNormal: { value: null },
    uLineInt: { value: LOOK.lineInt },
    uLineN: { value: LOOK.lineNInt },
    uLineZ: { value: LOOK.lineZInt },
    uLineThresh: { value: LOOK.lineThresh },
    uLineColor: { value: new THREE.Color(0x151030) },
    uTexel: { value: new THREE.Vector2(1 / 1024, 1 / 1024) },
    uCamNear: { value: 0.1 },
    uCamFar: { value: 100 },
  },
  vertexShader:
    "varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 ); }",
  fragmentShader:
    "uniform sampler2D tDiffuse; uniform sampler2D tDepth; uniform sampler2D tNormal;\n" +
    "uniform float uLineInt, uLineN, uLineZ, uLineThresh, uCamNear, uCamFar; uniform vec3 uLineColor; uniform vec2 uTexel; varying vec2 vUv;\n" +
    "float ld( vec2 uv ){ float z = texture2D( tDepth, uv ).x * 2.0 - 1.0; return ( 2.0 * uCamNear * uCamFar ) / ( uCamFar + uCamNear - z * ( uCamFar - uCamNear ) ); }\n" +
    "float luma( vec2 uv ){ return dot( texture2D( tDiffuse, uv ).rgb, vec3( 0.299, 0.587, 0.114 ) ); }\n" +
    "vec3 nrm( vec2 uv ){ return texture2D( tNormal, uv ).xyz * 2.0 - 1.0; }\n" +
    "void main(){\n" +
    "  vec4 c = texture2D( tDiffuse, vUv );\n" +
    "  float ink = 0.0;\n" +
    "  float d0 = ld( vUv );\n" +
    "  if ( uLineInt > 0.001 ) {\n" + // Y: depth edges
    // SINGLE-SIDED: ink only the FAR (background) side of the discontinuity, so
    // the line lands ON/OUTSIDE the silhouette where the inverted-hull outline
    // sits — two-sided ink drew a floating line INSIDE the visible hull edge.
    "    float dmin = min( min( ld( vUv + vec2( uTexel.x, 0.0 ) ), ld( vUv - vec2( uTexel.x, 0.0 ) ) ),\n" +
    "                      min( ld( vUv + vec2( 0.0, uTexel.y ) ), ld( vUv - vec2( 0.0, uTexel.y ) ) ) );\n" +
    "    float dth = uLineThresh * max( dmin, 1.0 );\n" +
    "    ink = smoothstep( dth * 0.75, dth * 1.35, d0 - dmin ) * uLineInt;\n" +
    "  }\n" +
    "  if ( uLineN > 0.001 && d0 < uCamFar * 0.5 ) {\n" + // X: normal edges (surface creases)
    "    vec3 n0 = nrm( vUv );\n" +
    "    float nd = 1.0 - min( min( dot( n0, nrm( vUv + vec2( uTexel.x, 0.0 ) ) ), dot( n0, nrm( vUv - vec2( uTexel.x, 0.0 ) ) ) ),\n" +
    "                          min( dot( n0, nrm( vUv + vec2( 0.0, uTexel.y ) ) ), dot( n0, nrm( vUv - vec2( 0.0, uTexel.y ) ) ) ) );\n" +
    "    ink = max( ink, smoothstep( 0.32, 0.52, nd ) * uLineN );\n" +
    "  }\n" +
    "  if ( uLineZ > 0.001 && d0 < uCamFar * 0.5 ) {\n" + // Z: colour-difference edges
    "    float l0 = luma( vUv );\n" +
    "    float zd = max( max( abs( luma( vUv + vec2( uTexel.x, 0.0 ) ) - l0 ), abs( luma( vUv - vec2( uTexel.x, 0.0 ) ) - l0 ) ),\n" +
    "                    max( abs( luma( vUv + vec2( 0.0, uTexel.y ) ) - l0 ), abs( luma( vUv - vec2( 0.0, uTexel.y ) ) - l0 ) ) );\n" +
    "    ink = max( ink, smoothstep( 0.14, 0.24, zd ) * uLineZ );\n" +
    "  }\n" +
    "  c.rgb = mix( c.rgb, uLineColor, clamp( ink, 0.0, 1.0 ) );\n" +
    "  gl_FragColor = c;\n" +
    "}",
};

// GLSL for the fresnel + non-mixing dual sublight (cyan left / magenta right)
// and the msTC (msToonCoordinator) colour-coordination uniforms + HSV helpers.
const SUB_DECLS =
  "uniform float uIdMode;\nuniform vec3 uMatId;\n" +
  "uniform float uFresInt;\nuniform float uFresPow;\nuniform vec3 uFresColor;\n" +
  "uniform float uSubInt;\nuniform float uSubPow;\nuniform float uSubMix;\n" +
  "uniform float uSubIntCool;\nuniform float uSubIntWarm;\n" +
  "uniform float uCelBands;\n" +
  "uniform vec3 uSubCoolColor;\nuniform vec3 uSubWarmColor;\n" +
  "uniform vec3 uSubCoolDir;\nuniform vec3 uSubWarmDir;\n" +
  "uniform float uBaseSat;\nuniform float uBaseVal;\nuniform float uBaseHue;\n" +
  "uniform vec3 uBaseTint;\nuniform float uBaseTintMix;\n" +
  "uniform float uShadeCoord;\nuniform float uShadeHue;\nuniform float uShadeHueMix;\n" +
  "uniform float uShadeSat;\nuniform float uShadeVal;\n" +
  "uniform float uShadowHard;\n" +
  "uniform float uHiCoord;\nuniform float uHiThresh;\nuniform float uHiHue;\nuniform float uHiHueMix;\n" +
  "uniform float uHiSat;\nuniform float uHiVal;\n" +
  "uniform float uSubCoord;\nuniform float uSubThresh;\nuniform float uSubSat;\nuniform float uSubVal;\n" +
  "uniform float uShadowFt;\n" +
  "uniform float uSssInt;\nuniform float uSssWidth;\n" +
  "uniform float uForceHi;\n" +
  "uniform sampler2D uMaskMap;\nuniform float uMaskOn;\n" +
  "uniform sampler2D uMaskSubMap;\nuniform float uMaskSubOn;\n" +
  "uniform vec3 uPointPos;\nuniform float uPointInt;\nuniform float uPointRange;\n" +
  // Cross-function globals: the SHADOW_RATE_TEXTURE mask channels (R = keep
  // base/lit, G = force highlight, B = force shade) sampled in main, applied
  // inside getShading; and the terminator band for the SSS red edge.
  "float mstcMaskLit = 0.0;\nfloat mstcMaskHi = 0.0;\nfloat mstcMaskShade = 0.0;\n" +
  "vec3 mstcMaskSub = vec3( 0.0 );\n" + // SUBLIGHT_RATE: R=sub1(warm) G=sub2(cool) B=hi&sub cancel
  "float mstcBorder = 0.0;\n" +
  // ==== Exact ports from msToonCoordinator v1.1 Shade.fx (ましまし) ====
  // Colour model is HSY: hue / saturation / Rec.709 LUMINANCE (not HSV value),
  // and the fx operates on non-linear sRGB screen colours — so we convert
  // gamma<->linear around the ops (LINEARIZE was commented out in the fx).
  "vec3 mstcToGamma( vec3 c ) { return pow( max( c, 0.0 ), vec3( 1.0 / 2.2 ) ); }\n" +
  "vec3 mstcToLinear( vec3 c ) { return pow( max( c, 0.0 ), vec3( 2.2 ) ); }\n" +
  "vec3 mstcHsy2rgb( vec3 hsy ) {\n" + // fx hsy2rgb (verbatim, vec3)
  "  float h = fract( abs( hsy.x ) );\n" +
  "  float regionH = h * 6.0;\n" +
  "  float mid = 1.0 - abs( mod( regionH, 2.0 ) - 1.0 );\n" +
  "  vec3 rgb = vec3( 1.0, mid, 0.0 );\n" +
  "  rgb = mix( rgb, vec3( mid, 1.0, 0.0 ), step( 1.0, regionH ) );\n" +
  "  rgb = mix( rgb, vec3( 0.0, 1.0, mid ), step( 2.0, regionH ) );\n" +
  "  rgb = mix( rgb, vec3( 0.0, mid, 1.0 ), step( 3.0, regionH ) );\n" +
  "  rgb = mix( rgb, vec3( mid, 0.0, 1.0 ), step( 4.0, regionH ) );\n" +
  "  rgb = mix( rgb, vec3( 1.0, 0.0, mid ), step( 5.0, regionH ) );\n" +
  "  float pureY = dot( rgb, vec3( 0.2126, 0.7152, 0.0722 ) );\n" +
  "  vec3 lt = rgb * ( hsy.z / max( pureY, 1.0e-5 ) );\n" +
  "  vec3 gt = mix( rgb, vec3( 1.0 ), ( hsy.z - pureY ) / max( 1.0 - pureY, 1.0e-5 ) );\n" +
  "  rgb = mix( lt, gt, step( pureY, hsy.z ) );\n" +
  "  return mix( vec3( hsy.z ), rgb, hsy.y );\n" +
  "}\n" +
  "vec3 mstcRgb2hsy( vec3 c ) {\n" + // fx rgb2hsy (div guards added for GLSL NaN)
  "  vec3 s = clamp( c, 0.0, 1.0 );\n" +
  "  float minC = min( min( s.r, s.g ), s.b );\n" +
  "  float maxC = max( max( s.r, s.g ), s.b );\n" +
  "  float d = max( maxC - minC, 1.0e-5 );\n" +
  "  float h = ( s.g - s.r ) / d + 1.0;\n" +
  "  h = mix( h, ( s.r - s.b ) / d + 5.0, step( s.g, minC ) );\n" +
  "  h = mix( h, ( s.b - s.g ) / d + 3.0, step( s.r, minC ) );\n" +
  "  h = mix( h, 0.0, step( maxC, minC ) );\n" + // achromatic -> hue 0 (as fx)
  "  h /= 6.0;\n" +
  "  float y = dot( s, vec3( 0.2126, 0.7152, 0.0722 ) );\n" +
  "  vec3 pureC = mstcHsy2rgb( vec3( h, 1.0, y ) );\n" +
  "  float den = pureC.g - y;\n" +
  "  den = abs( den ) < 1.0e-5 ? 1.0e-5 : den;\n" +
  "  float sat = ( s.g - y ) / den;\n" +
  "  sat = mix( sat, 1.0, step( y, 0.0 ) + step( 1.0, y ) );\n" +
  "  return vec3( h, clamp( sat, 0.0, 1.0 ), y );\n" +
  "}\n" +
  // fx adjustHSY1: sat/lum pivots (s, y in -1..1; sat only on chromatic pixels),
  // then hue applied as an RGB-space blend toward the fully hue-REPLACED colour
  // (NOT a hue-wheel lerp) by hAlpha (色相適用). Returns HSY.
  "vec3 mstcAdjust1( vec3 baseHSY, float h, float s, float y, float hAlpha ) {\n" +
  "  vec3 t = baseHSY;\n" +
  "  t.y = baseHSY.y > 0.0 ? baseHSY.y + ( 0.5 - baseHSY.y ) * abs( s ) + 0.5 * s : baseHSY.y;\n" +
  "  t.z = baseHSY.z + ( 0.5 - baseHSY.z ) * abs( y ) + 0.5 * y;\n" +
  "  return mstcRgb2hsy( mix( mstcHsy2rgb( t ), mstcHsy2rgb( vec3( h, t.yz ) ), hAlpha ) );\n" +
  "}\n" +
  // fx adjustHSY2 (sublights): hue is REPLACED outright, sat/lum pivots as above.
  "vec3 mstcAdjust2( vec3 baseHSY, float h, float s, float y ) {\n" +
  "  return vec3( h,\n" +
  "    baseHSY.y + ( 0.5 - baseHSY.y ) * abs( s ) + 0.5 * s,\n" +
  "    baseHSY.z + ( 0.5 - baseHSY.z ) * abs( y ) + 0.5 * y );\n" +
  "}\n" +
  // 影1: shade colour derived from the pixel's own base colour (色トレス).
  // Knobs stay in morph space (0..1, 0.5 neutral) like the msTC controller.
  "vec3 mstcShadeOf( vec3 base ) {\n" +
  "  vec3 hsy = mstcRgb2hsy( mstcToGamma( base ) );\n" +
  "  hsy = mstcAdjust1( hsy, uShadeHue / 360.0, uShadeSat * 2.0 - 1.0, uShadeVal * 2.0 - 1.0, uShadeHueMix );\n" +
  "  return mstcToLinear( mstcHsy2rgb( hsy ) );\n" +
  "}\n";
// `target` is whatever the material calls the accumulated lit colour before the
// final write — `outgoingLight` for standard three materials, `col` for MToon.
const subEffect = (target: string) =>
  [
    "{",
    "  vec3 subN = normalize( normal );",
    "  vec3 subV = normalize( vViewPosition );",
    "  float rimEdge = clamp( 1.0 - abs( dot( subN, subV ) ), 0.0, 1.0 );",
    `  ${target} += pow( rimEdge, uFresPow ) * uFresInt * uFresColor;`,
    "  if ( uSubInt > 0.001 || uPointInt > 0.001 || uMaskSubOn > 0.5 ) {",
    "    float rimF = pow( rimEdge, uSubPow );",
    "    float coolM = rimF * clamp( dot( subN, uSubCoolDir ), 0.0, 1.0 ) * uSubInt * uSubIntCool;",
    "    float warmM = rimF * clamp( dot( subN, uSubWarmDir ), 0.0, 1.0 ) * uSubInt * uSubIntWarm;",
    // SUBLIGHT_RATE mask: R forces sub1(warm), G forces sub2(cool), B cancels.
    "    warmM += smoothstep( 0.2, 0.5, mstcMaskSub.r );",
    "    coolM += smoothstep( 0.2, 0.5, mstcMaskSub.g );",
    "    float subCancel = 1.0 - clamp( mstcMaskSub.b, 0.0, 1.0 );",
    "    warmM *= subCancel;",
    "    coolM *= subCancel;",
    // PostPointLight analogue: a view-space point light adds to the COOL
    // sublight mask (light pools where the surface faces the point).
    "    if ( uPointInt > 0.001 ) {",
    "      vec3 subPl = uPointPos + vViewPosition;", // vViewPosition = -viewPos
    "      coolM += clamp( 1.0 - length( subPl ) / max( uPointRange, 0.001 ), 0.0, 1.0 )",
    "             * clamp( dot( subN, normalize( subPl ) ), 0.0, 1.0 ) * uPointInt;",
    "    }",
    "    vec3 coolC = uSubCoolColor * coolM;",
    "    vec3 warmC = uSubWarmColor * warmM;",
    "    float lc = dot( coolC, vec3( 0.299, 0.587, 0.114 ) );",
    "    float lw = dot( warmC, vec3( 0.299, 0.587, 0.114 ) );",
    "    vec3 dom = lc >= lw ? coolC : warmC;", // non-mixing: dominant sublight only
    "    vec3 subAdd = mix( dom, coolC + warmC, uSubMix );",
    "    if ( uSubCoord > 0.001 ) {",
    // msTC (Shade.fx) sublight semantics: a hard-thresholded hue-REPLACEMENT
    // (「サブ1色相 … この色相になります」), NOT added light. The replacement is
    // adjustHSY2 of the pixel's BASE colour (ColorOrgHSY in the fx); Sub1 is
    // applied after Sub2 in the fx, so cool wins where both masks hit.
    "      float coolHit = smoothstep( uSubThresh, uSubThresh + 0.03, coolM );",
    "      float warmHit = smoothstep( uSubThresh, uSubThresh + 0.03, warmM );",
    "      vec3 subBase = mstcRgb2hsy( mstcToGamma( material.diffuseColor ) );",
    "      float subS = uSubSat * 2.0 - 1.0;",
    "      float subY = uSubVal * 2.0 - 1.0;",
    "      vec3 repCool = mstcToLinear( mstcHsy2rgb( mstcAdjust2( subBase, mstcRgb2hsy( clamp( uSubCoolColor, 0.0, 1.0 ) ).x, subS, subY ) ) );",
    "      vec3 repWarm = mstcToLinear( mstcHsy2rgb( mstcAdjust2( subBase, mstcRgb2hsy( clamp( uSubWarmColor, 0.0, 1.0 ) ).x, subS, subY ) ) );",
    `      vec3 subRep = mix( ${target}, repWarm, warmHit );`,
    "      subRep = mix( subRep, repCool, coolHit );",
    `      ${target} = mix( ${target} + subAdd, subRep, uSubCoord );`,
    "    } else {",
    `      ${target} += subAdd;`,
    "    }",
    "  }",
    "}",
  ].join("\n");

// Shared 1×1 black texture = "no mask" default for uMaskMap.
const BLACK_TEX = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
BLACK_TEX.needsUpdate = true;

const makeSubUniforms = () => ({
  // MATERIAL ID PASS. uIdMode 1 makes the fragment emit uMatId flat, bypassing
  // every shading and msTC stage. It exists because identifying which material
  // draws a given pixel by hiding meshes one at a time does not survive this
  // scene: spring-bone physics and the TAA history drift between captures, and
  // recolouring materials from the JS side is overwritten by the msTC injection
  // further down. Driven by __vrmIdMode(); also the hook a per-material grade
  // mask would render through.
  uIdMode: { value: 0 },
  uMatId: { value: new THREE.Color(0, 0, 0) },
  uFresInt: { value: LOOK.fresInt },
  uFresPow: { value: LOOK.fresPow },
  uFresColor: { value: new THREE.Color(LOOK.fresColor) },
  uSubInt: { value: LOOK.subInt },
  uSubPow: { value: LOOK.subPow },
  uSubMix: { value: LOOK.subMix },
  uSubIntCool: { value: LOOK.subIntCool },
  uSubIntWarm: { value: LOOK.subIntWarm },
  uCelBands: { value: LOOK.celBands },
  uSubCoolColor: { value: new THREE.Color(LOOK.coolColor) },
  uSubWarmColor: { value: new THREE.Color(LOOK.warmColor) },
  uSubCoolDir: { value: COOL_DIR.clone() },
  uSubWarmDir: { value: WARM_DIR.clone() },
  // msTC (msToonCoordinator) coordination — all neutral at the defaults.
  uBaseSat: { value: LOOK.baseSat },
  uBaseVal: { value: LOOK.baseVal },
  uBaseHue: { value: 0 },
  uBaseTint: { value: new THREE.Color(LOOK.hairColor) },
  uBaseTintMix: { value: 0 },
  uShadeCoord: { value: LOOK.shadeCoord },
  uShadeHue: { value: LOOK.shadeHue },
  uShadeHueMix: { value: LOOK.shadeHueMix },
  uShadeSat: { value: LOOK.shadeSat },
  uShadeVal: { value: LOOK.shadeVal },
  uShadowHard: { value: LOOK.shadowHard },
  uHiCoord: { value: LOOK.hiCoord },
  uHiThresh: { value: LOOK.hiThresh },
  uHiHue: { value: LOOK.hiHue },
  uHiHueMix: { value: LOOK.hiHueMix },
  uHiSat: { value: LOOK.hiSat },
  uHiVal: { value: LOOK.hiVal },
  uSubCoord: { value: LOOK.subCoord },
  uSubThresh: { value: LOOK.subThresh },
  uSubSat: { value: LOOK.subSat },
  uSubVal: { value: LOOK.subVal },
  uShadowFt: { value: LOOK.shadowFt },
  uSssInt: { value: LOOK.sssInt },
  uSssWidth: { value: LOOK.sssWidth },
  uForceHi: { value: 0 },
  uMaskMap: { value: BLACK_TEX },
  uMaskOn: { value: 0 },
  uMaskSubMap: { value: BLACK_TEX },
  uMaskSubOn: { value: 0 },
  uPointPos: { value: new THREE.Vector3(0, 1.25, 2) },
  uPointInt: { value: LOOK.pointInt },
  uPointRange: { value: LOOK.pointRange },
});

// Inject the sublight into a material's shader. The exported Miku VRM's materials
// are standard glTF PBR (MeshPhysicalMaterial) — NOT MToon (the Blender export
// dropped VRMC_materials_mtoon) — so the real path is onBeforeCompile on the
// standard `outgoingLight` before <opaque_fragment>, the same technique the MMD
// viewer uses. The MToon branch is kept in case the model is later re-exported
// with proper toon materials. The live uniform refs live on material.userData.
const injectSublight = (m: any) => {
  if (!m || m._subInjected) return;

  // Real MToon (edit its ShaderMaterial source directly).
  if (
    m.isMToonMaterial &&
    typeof m.fragmentShader === "string" &&
    m.fragmentShader.includes("col += totalEmissiveRadiance;")
  ) {
    const u = makeSubUniforms();
    m.uniforms = m.uniforms || {};
    Object.assign(m.uniforms, u);
    m.userData._subU = u;
    m.fragmentShader = m.fragmentShader
      .replace("#include <common>", SUB_DECLS + "#include <common>")
      // Hard cel bands (セル塗り段数): quantize the SHADING factor (how much light
      // hits the surface) into N steps — real cel shading, follows the light, does
      // NOT band the texture. The shade colour is darkened in applyLook when active
      // so the bands stay visible even under this look's flat fills.
      .replace(
        "shading = linearstep( -1.0 + shadingToonyFactor, 1.0 - shadingToonyFactor, shading );",
        // (a) the SHADOW_RATE mask (R keeps lit, B forces shade) pushes the RAW
        // shading BEFORE the toony threshold — like the RT path in the fx, so
        // gradient masks move the hard boundary instead of showing as gradients;
        // (b) capture the terminator band for the SSS edge; (c) cel quantize.
        "shading += ( mstcMaskLit - mstcMaskShade ) * 2.0;\n"
          + "  mstcBorder = max( mstcBorder, 1.0 - smoothstep( 0.0, max( uSssWidth, 0.001 ), abs( shading ) ) );\n"
          + "  shading = linearstep( -1.0 + shadingToonyFactor, 1.0 - shadingToonyFactor, shading );\n"
          + "  if ( uCelBands > 1.5 ) shading = floor( shading * ( uCelBands - 1.0 ) + 0.5 ) / ( uCelBands - 1.0 );",
      )
      // msTC 落ち影2値化: MToon multiplies the PCF cast-shadow into the shading
      // AFTER the toony step, so a soft shadow edge reads as a gradient. Snapping
      // it to a hard boundary makes the cast shadow the SAME flat tone as the
      // shade (msTC: 落ち影と陰が同質). Needs shadow.intensity = 1 (applyLookState
      // forces it when uShadowHard > 0, else the shadow value never crosses 0.5).
      // Threshold below 0.5 so half-occluded PCF noise (shadow acne) resolves to
      // LIT instead of leaving hard binarized patches — the ES「ボケ足」placement.
      .replace(
        "shading *= shadow;",
        "shading *= mix( shadow, smoothstep( 0.375 - uShadowFt, 0.375 + uShadowFt, shadow ), uShadowHard );",
      )
      // msTC テクスチャ純色化: sat/lum pivots on the albedo itself (0.5 = off).
      // Everything downstream (shade derivation, highlight, sublight) reads
      // material.diffuseColor, so the whole pipeline sees the pale base.
      .replace(
        "material.diffuseColor = diffuseColor.rgb;",
        "vec3 mstcBase = diffuseColor.rgb;\n"
          + "  if ( abs( uBaseSat - 0.5 ) + abs( uBaseVal - 0.5 ) + abs( uBaseHue ) > 0.002 ) {\n"
          + "    vec3 bHSY = mstcRgb2hsy( mstcToGamma( mstcBase ) );\n"
          + "    bHSY.x = fract( bHSY.x + uBaseHue + 1.0 );\n" // per-part hue rotation (hair/eyes)
          + "    bHSY = mstcAdjust1( bHSY, 0.0, uBaseSat * 2.0 - 1.0, uBaseVal * 2.0 - 1.0, 0.0 );\n"
          + "    mstcBase = mstcToLinear( mstcHsy2rgb( bHSY ) );\n"
          + "  }\n"
          + "  if ( uBaseTintMix > 0.001 ) mstcBase = mix( mstcBase, uBaseTint, uBaseTintMix );\n" // 髪色の直接指定
          // SHADOW_RATE_TEXTURE analogue: R = keep base/lit, G = force
          // highlight, B = force shade — painted in UV space (__vrmMask).
          + "  #if ( defined( MTOON_USE_UV ) && !defined( MTOON_UVS_VERTEX_ONLY ) )\n"
          + "  if ( uMaskOn > 0.5 ) {\n"
          + "    vec3 mstcMk = texture2D( uMaskMap, uv ).rgb;\n"
          + "    mstcMaskLit = mstcMk.r; mstcMaskHi = mstcMk.g; mstcMaskShade = mstcMk.b;\n"
          + "  }\n"
          + "  if ( uMaskSubOn > 0.5 ) mstcMaskSub = texture2D( uMaskSubMap, uv ).rgb;\n"
          + "  #endif\n"
          + "  material.diffuseColor = mstcBase;",
      )
      // msTC 影1: derive the shade colour per-pixel from the albedo (色トレス —
      // hue pulled toward 影1色相, sat/val through the 0.5 pivots) instead of the
      // authored flat shadeColorFactor. uShadeCoord blends authored -> derived.
      .replace(
        "vec3 col = lightColor * BRDF_Lambert( mix( material.shadeColor, material.diffuseColor, shading ) );",
        "vec3 col = lightColor * BRDF_Lambert( mix( mix( material.shadeColor, mstcShadeOf( material.diffuseColor ), uShadeCoord ), material.diffuseColor, shading ) );",
      )
      // msTC ハイ1: threshold the matcap/rim luminance into a hard mask (the RT
      // G channel analogue) and REPLACE the pixel with adjustHSY1 of its base
      // colour — sat-preserving, luminance-lifted (fx defaults: 彩度0.5 明度0.82).
      // uHiCoord = 0 keeps the plain additive matcap.
      .replace(
        "col += rimMix * rim;",
        "{\n"
          + "  vec3 hiC = rimMix * rim;\n"
          + "  float hiM = clamp( smoothstep( uHiThresh, uHiThresh + 0.04, dot( hiC, vec3( 0.299, 0.587, 0.114 ) ) ) + smoothstep( 0.2, 0.45, mstcMaskHi ) + uForceHi, 0.0, 1.0 ) * ( 1.0 - clamp( mstcMaskSub.b, 0.0, 1.0 ) ) * uHiCoord;\n"
          + "  vec3 hiHsy = mstcRgb2hsy( mstcToGamma( material.diffuseColor ) );\n"
          + "  hiHsy = mstcAdjust1( hiHsy, uHiHue / 360.0, uHiSat * 2.0 - 1.0, uHiVal * 2.0 - 1.0, uHiHueMix );\n"
          + "  col = mix( col + hiC, mstcToLinear( mstcHsy2rgb( hiHsy ) ), hiM );\n"
          + "}",
      )
      .replace(
        "col += totalEmissiveRadiance;",
        "#ifndef OUTLINE\n"
          // msTC v2.1 SSS: warm red band on the light/shade terminator.
          + "if ( uSssInt > 0.001 ) col = mix( col, col * vec3( 1.32, 0.72, 0.68 ), clamp( mstcBorder, 0.0, 1.0 ) * uSssInt );\n"
          + `${subEffect("col")}\n#endif\ncol += totalEmissiveRadiance;\nif ( uIdMode > 0.5 ) col = uMatId;`,
      );
    m.needsUpdate = true;
    m._subInjected = true;
    return;
  }

  // Standard glTF PBR (the actual Miku VRM case).
  if (m.isMeshStandardMaterial || m.isMeshPhysicalMaterial) {
    m._subInjected = true;
    // Flatten the glossy PBR so it reads flatter / more anime — the coloured rim
    // + bloom carry the style instead of realistic speculars/clearcoat.
    if ("metalness" in m) m.metalness = 0.0;
    if ("roughness" in m) m.roughness = Math.max(m.roughness ?? 1, 0.85);
    if ("clearcoat" in m) m.clearcoat = 0.0;
    if ("envMapIntensity" in m) m.envMapIntensity = 0.0;
    const u = makeSubUniforms();
    m.userData._subU = u;
    const prevOBC = m.onBeforeCompile;
    m.onBeforeCompile = (shader: any, renderer: any) => {
      if (prevOBC) prevOBC.call(m, shader, renderer);
      Object.assign(shader.uniforms, u); // same refs -> live tuning uploads
      shader.fragmentShader = shader.fragmentShader
        .replace("#include <common>", SUB_DECLS + "#include <common>")
        .replace(
          "#include <opaque_fragment>",
          `${subEffect("outgoingLight")}\nif ( uIdMode > 0.5 ) { outgoingLight = uMatId; diffuseColor.a = 1.0; }\n#include <opaque_fragment>`,
        );
    };
    m.needsUpdate = true;
  }
};

export const VRMViewer = memo(({ url }: VRMViewerProps): JSX.Element => {
  // These panels are built as raw DOM, outside React, so react-i18next never sees
  // them and a language switch cannot re-render them. `L` picks the string at
  // build time from the same language the app is set to; the panels are rebuilt
  // when the model binds, which is when a switch takes effect.
  const uiLang = (() => {
    try {
      return (localStorage.getItem('i18nextLng') || navigator.language || 'ja').toLowerCase();
    } catch { return 'ja'; }
  })();
  // Read the language at CALL time, not once at mount. The panels are built on
  // mount, so a value captured here would freeze whatever language was active
  // then and never follow the app's language selector.
  const isJa = (): boolean => {
    try {
      return (localStorage.getItem('i18nextLng') || navigator.language || uiLang)
        .toLowerCase().startsWith('ja');
    } catch { return uiLang.startsWith('ja'); }
  };
  const L = (ja: string, en: string): string => (isJa() ? ja : en);
  /**
   * Set a label AND remember both wordings on the element, so a later language
   * change can re-render it in place — the panels themselves are never rebuilt.
   */
  const SET = (el: HTMLElement, ja: string, en?: string): void => {
    el.dataset.ja = ja;
    if (en !== undefined) el.dataset.en = en;
    el.textContent = en !== undefined ? L(ja, en) : TX(ja);
  };
  /** Re-render every remembered label. Runs whenever a panel is opened. */
  const retranslate = (root: HTMLElement | null): void => {
    root?.querySelectorAll<HTMLElement>('[data-ja]').forEach((el) => {
      const ja = el.dataset.ja as string;
      el.textContent = el.dataset.en !== undefined ? L(ja, el.dataset.en) : TX(ja);
    });
  };
  /** Label text only — never a lookup key. Unknown strings pass through. */
  const TX = (ja: string): string =>
    (isJa() ? ja : ((UI_LABELS as Record<string, string>)[ja] ?? ja));
  // Mixer for a model's OWN embedded idle clip (Chancelor ships
  // Female_Animation_Idle). Only created for models that both have such a clip
  // and are not driven by the KAWAII idle library.
  const idleMixerRef = useRef<THREE.AnimationMixer | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const { aiState } = useAiState();

  // Let the environment breathe with the conversation (idle pulse / listening
  // ripple / thinking data-stream). Kept as its own effect rather than a value
  // read inside the render closure below: that closure is built once per `url`,
  // so anything captured there would go stale on the first state change.
  useEffect(() => {
    neuralSpace.setAiState(aiState);
    aiFx.setAiState(aiState);
  }, [aiState]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;
    let disposed = false;
    let raf = 0;
    let vrm: VRM | null = null;
    let disposeFacePanel: (() => void) | null = null;
    let disposeLookPanel: (() => void) | null = null;
    let disposeFaceMotionPanel: (() => void) | null = null;

    /**
     * Drop the stylised look for models it was not designed for.
     *
     * Everything in LOOK — the cyan/magenta rim light, the line art, the toon
     * banding, the saturated purple-lifted grade — was dialled in against Miku,
     * a toon character. Applied unchanged to a photoreal avatar it reads as a
     * fault rather than a style: the first .glb loaded came out violet with a
     * cyan edge glow.
     *
     * Implemented as a READ-TIME override rather than by writing LOOK, and that
     * distinction matters: LOOK is persisted to localStorage and carries a lot
     * of hand-tuning, so mutating it for one model would quietly destroy the
     * other's look the moment anything called saveLook().
     */
    let plainStyle = false;
    const fx = <T,>(styled: T, neutral: T): T => (plainStyle ? neutral : styled);

    const scene = new THREE.Scene();
    /**
     * The world-matrix walk is driven BY HAND, once a frame, not by every render.
     *
     * `renderer.render()` begins with `scene.updateMatrixWorld()`, and this frame
     * makes six render calls (panel backdrop, ink-line normals, TAA velocity, the
     * composer's own pass, GTAO's depth/normal pre-pass, DOF's depth pass). That
     * walk is not conditional in any useful sense: `updateMatrix()` runs
     * unconditionally wherever `matrixAutoUpdate` is on, and it re-arms
     * `matrixWorldNeedsUpdate`, so each of the six recomposed and re-multiplied
     * every one of the ~1150 nodes in this scene. Profiling put matrix work at
     * 34% of all CPU time — the single largest item in the frame, ahead of the
     * physics.
     *
     * Nothing writes a transform between the second walk and the last render, so
     * two explicit walks reproduce the six exactly: one inside the panels module
     * before its backdrop render, one here after the environment has moved.
     */
    scene.matrixWorldAutoUpdate = false;
    const camera = new THREE.PerspectiveCamera(
      30,
      Math.max(container.clientWidth, 1) / Math.max(container.clientHeight, 1),
      0.01,
      100,
    );
    camera.position.set(0, 1.05, 2.4);
    camera.lookAt(0, 0.98, 0);

    // antialias:false is NOT a quality decision — the default framebuffer never
    // receives geometry. Every pass renders into a composer target (the scene
    // side carries its own MSAA), and the only thing ever drawn to the screen
    // buffer is SMAAPass's final fullscreen quad, which has no interior edges
    // for MSAA to find. Asking for it allocated a multisampled backbuffer and
    // paid a resolve blit every frame for an effect that cannot exist.
    // alpha stays: the canvas is composited over the container's CSS gradient.
    const renderer = new THREE.WebGLRenderer({
      antialias: false,
      alpha: true,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5)); // cap for FPS on hi-DPI
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    // Self-shadows (hair on face, body depth) + filmic tone mapping — parity with
    // the MMD viewer's effect stack. OutputPass applies the tone map at the end.
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    // The shadow map is re-rendered by EVERY renderer.render() whose camera can
    // see the key light — and this frame makes several: the TAA velocity pass,
    // the ink-line normal pre-pass, and the composer's own RenderPass. That is
    // two or three full re-renders of a 4096x4096 PCF-soft map per frame, all
    // producing the identical result because the pose does not change between
    // them. Driven by hand instead: `needsUpdate` is set once per frame,
    // immediately before the pass whose output is actually seen.
    renderer.shadowMap.autoUpdate = false;
    renderer.shadowMap.needsUpdate = true;
    const TONE_MODES = [
      THREE.NoToneMapping,
      THREE.ACESFilmicToneMapping,
      THREE.AgXToneMapping,
      THREE.NeutralToneMapping,
    ];
    renderer.toneMapping = TONE_MODES[LOOK.toneMode] ?? THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = LOOK.toneExposure;
    renderer.domElement.style.cssText = "width:100%;height:100%;display:block";
    container.appendChild(renderer.domElement);

    // Mouse controls: left-drag orbits (rotate), right-drag pans (move), wheel
    // zooms. Damped for a smooth feel. Target sits at the upper body.
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.98, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enablePan = true;
    controls.screenSpacePanning = true; // pan moves in screen plane (feels like "move the character")
    controls.minDistance = 0.4;
    controls.maxDistance = 8;
    controls.update();

    // --- Backdrop: radial gradient (navy centre -> near-black edges) ----------
    const bgCanvas = document.createElement("canvas");
    bgCanvas.width = 256;
    bgCanvas.height = 256;
    const bgCtx = bgCanvas.getContext("2d")!;
    const bgTex = new THREE.CanvasTexture(bgCanvas);
    bgTex.colorSpace = THREE.SRGBColorSpace;
    const hex = (n: number) => `#${(n & 0xffffff).toString(16).padStart(6, "0")}`;
    const setBgGrad = (inner: number, outer: number) => {
      const g = bgCtx.createRadialGradient(128, 110, 12, 128, 128, 190);
      g.addColorStop(0, hex(inner));
      g.addColorStop(1, hex(outer));
      bgCtx.fillStyle = g;
      bgCtx.fillRect(0, 0, 256, 256);
      bgTex.needsUpdate = true;
      // The backdrop is drawn by CSS, BEHIND the (transparent) canvas, not by the scene.
      // Reason: the GTAO pass is screen-space, so it shaded the backdrop too — it
      // reconstructs a position from depth, and for pixels that are nothing but background
      // that reconstruction breaks, leaving a dead-straight horizontal colour step right
      // across the frame (the "背景に色の境 / 板みたいなもの"; measured: the step vanished the
      // moment ssao went to 0, and neither bloom, DOF, nor GTAO's own setSceneClipBox
      // touched it — a clip box that contains the avatar also contains the broken
      // reconstruction). Out of the composer, nothing can paint it. Same gradient geometry
      // as the canvas版: centre (128,110)/256, stops at r=12 and r=190.
      // The spatial layer owns the aperture colour when it wants to. Checked
      // HERE rather than set once at startup because applyLookState() and the
      // look panel both call back into setBgGrad, and either would repaint the
      // navy gradient over it. LOOK's own values are never modified — clearing
      // SPACE.backdrop hands the backdrop straight back.
      container.style.background = SPACE.backdrop
        ? SPACE.backdrop
        : `radial-gradient(circle at 50% 43%, ${hex(inner)} 5%, ${hex(outer)} 74%)`;
    };
    setBgGrad(LOOK.bgInner, LOOK.bgOuter);
    scene.background = null;
    renderer.setClearColor(0x000000, 0); // let the CSS backdrop through

    // --- Lighting: low key + front/side fills = flat anime lighting -----------
    const ambient = new THREE.AmbientLight(0xffffff, LOOK.ambient);
    scene.add(ambient);
    const hemi = new THREE.HemisphereLight(LOOK.hemiSky, LOOK.hemiGround, LOOK.hemi);
    scene.add(hemi);
    const keyLight = new THREE.DirectionalLight(LOOK.keyColor, LOOK.key);
    keyLight.position.set(3, 6, 8);
    keyLight.castShadow = LOOK.shadowStrength > 0.001;
    keyLight.shadow.mapSize.set(1024, 1024); // default; 影解像度 slider rebuilds to 2048/4096
    keyLight.shadow.camera.near = 0.5;
    keyLight.shadow.camera.far = 20;
    // Tight frustum around the (origin-planted) character — ~1.7× the effective
    // texel density of the old ±2/3/-1.5 box, the cheap half of ExcellentShadow.
    keyLight.shadow.camera.left = -1.2;
    keyLight.shadow.camera.right = 1.2;
    keyLight.shadow.camera.top = 2.2;
    keyLight.shadow.camera.bottom = -0.1;
    keyLight.shadow.bias = -0.0005;
    keyLight.shadow.normalBias = 0.02; // keep thin-occluder shadows (tie) attached; acne is absorbed by the binarize window

    if ((keyLight.shadow as any).intensity !== undefined) {
      (keyLight.shadow as any).intensity = LOOK.shadowStrength;
    }
    scene.add(keyLight);
    // Fills so no part of the (thin, outline-free) hair sits in deep MToon shade.
    const fillFront = new THREE.DirectionalLight(0xffffff, LOOK.fillFront);
    fillFront.position.set(0, 1.5, 10);
    scene.add(fillFront);
    const fillLeft = new THREE.DirectionalLight(0xdfe6ff, LOOK.fillSide);
    fillLeft.position.set(-8, 3, 4);
    scene.add(fillLeft);
    const fillRight = new THREE.DirectionalLight(0xdfe6ff, LOOK.fillSide);
    fillRight.position.set(8, 3, 4);
    scene.add(fillRight);

    // --- Spatial Presence Layer: the 3D space Miku stands IN ------------------
    // Adds a floor, neon structures and depth haze on their own layer (SPACE_LAYER),
    // so the panel reads as a window into a deep space rather than a screen with a
    // character on it. Adds NO lights — neon glow comes from emissive materials
    // through the UnrealBloomPass already in the chain below, which keeps the
    // carefully tuned character lighting untouched and costs nothing.
    renderer.info.autoReset = false; // see the reset() in animate()
    neuralSpace.init(scene, renderer, camera);
    // On a PROTO unit, frame the whole body + floor instead of the desktop
    // bust-up (her feet are off-screen there, and the contact shadow / grid /
    // reflection all need the ground in shot). No-op in a normal browser, so the
    // desktop framing is untouched. Runs AFTER init so the saved SPACE values
    // (loaded there) are the ones applied.
    // (auto-hide is installed once by App.tsx — the arrows it fades exist in
    // every model mode, and two install sites raced on model switch.)
    const framing = createFraming(camera, controls);
    const bodyDepth = createBodyDepth();
    /**
     * Take the framing range from the CHARACTER, not from a constant.
     *
     * The solver frames `fitBottomY..fitTopY`, and that top was 1.65 — Miku's
     * height in this scene, hard-coded while she was the only 3D model. The
     * first 1.80 m avatar was therefore framed as if it were 1.65 m and lost its
     * head off the top edge. Measuring the loaded model makes the crop correct
     * for whatever is standing there.
     *
     * Uses the world bounding box rather than the head bone: hair, ears and
     * headgear all sit outside the skeleton, and it is the silhouette that gets
     * cropped, not the bone.
     */
    const measureFitRange = () => {
      if (!SPACE.fitAutoHeight || !vrm?.scene) return;
      vrm.scene.updateWorldMatrix(true, true);
      const box = new THREE.Box3().setFromObject(vrm.scene);
      if (!box.isEmpty()) {
        const h = box.max.y - box.min.y;
        if (h > 0.2 && h < 5) {
          SPACE.fitBottomY = Math.min(0, box.min.y);
          SPACE.fitTopY = box.max.y + h * SPACE.fitHeadroom;
        }
      }
    };
    const applyFraming = () => {
      measureFitRange();
      framing.apply();
      setBgGrad(LOOK.bgInner, LOOK.bgOuter); // re-evaluates SPACE.backdrop
    };
    applyFraming();

    // Live tuning from the console, same flavour as __vrmLook: the space has to
    // be dialled in against the real PROTO unit, and a rebuild per tweak makes
    // that impractical. __space patches settings (applied next frame),
    // __spaceRebuild re-creates geometry when a COUNT changed, __spaceQuality
    // swaps the performance budget, __protoMode toggles the framing.
    (window as any).__space = (patch?: Record<string, unknown>) => {
      if (patch) {
        Object.assign(SPACE, patch);
        neuralSpace.refresh();
        applyFraming();
        refreshAutoHide();
        resize(); // pixelRatioCap may have moved; re-derive the buffer sizes
      }
      return { ...neuralSpace.describe(), ...framingReport() };
    };
    /** Where the solved framing actually put things — the horizon is an output now. */
    const framingReport = () => {
      const half = (camera.fov * Math.PI) / 360;
      const d = camera.position.distanceTo(controls.target);
      // Frame fraction where the level line lands, then in panel metres.
      const pitch = Math.atan2(camera.position.y - controls.target.y, d);
      const frac = 0.5 + (pitch * 180) / Math.PI / camera.fov;
      return {
        eyeY: +camera.position.y.toFixed(2),
        horizonPctUpFrame: +(frac * 100).toFixed(1),
        horizonAtM: +(frac * SPACE.panelHeightM).toFixed(2),
        fovDeg: camera.fov,
        halfFovRad: +half.toFixed(3),
      };
    };
    // Sharpness diagnosis on the real panel: says whether the drawing buffer
    // actually covers the display's physical pixels or is being upscaled.
    (window as any).__spaceDpr = () =>
      describePixelRatio(container, renderer.domElement as HTMLCanvasElement);
    let sharpSaved: { dof: number; diff: number; ssao: number; line: number } | null = null;
    (window as any).__sharpTest = (on = true) => {
      if (on && !sharpSaved) {
        sharpSaved = {
          dof: LOOK.dofAperture, diff: LOOK.diffusion, ssao: LOOK.ssao, line: LOOK.lineInt,
        };
        LOOK.dofAperture = 0;
        LOOK.diffusion = 0;
        applyLookState();
        return "post FX off — if it looks sharp now the softness is DOF/diffusion, not resolution. __sharpTest(false) to restore.";
      }
      if (!on && sharpSaved) {
        LOOK.dofAperture = sharpSaved.dof;
        LOOK.diffusion = sharpSaved.diff;
        LOOK.ssao = sharpSaved.ssao;
        LOOK.lineInt = sharpSaved.line;
        sharpSaved = null;
        applyLookState();
        return "restored";
      }
      return sharpSaved ? "already off" : "already on";
    };

    // --- Phase 2/3 console surface -----------------------------------------
    (window as any).__headTracker = (on?: boolean) => {
      if (on === true) {
        // start() is async (camera permission + ~15MB of wasm/model), and
        // returning status() synchronously reported `running: false` before it
        // had a chance to be true — which reads exactly like a failure.
        headTracker.start().then((ok) => {
          console.log(
            ok ? "[space] head tracking STARTED" : "[space] head tracking FAILED",
            headTracker.status(),
          );
        });
        return "starting... (result will be logged; watch __spaceHud / __headPreview)";
      }
      if (on === false) {
        headTracker.stop();
        return "stopped";
      }
      return headTracker.status();
    };
    // Faces the server found in the PROTO's camera (render-source.ts calls this).
    (window as any).__headIngest = (m: ServerFaces) => headTracker.ingest(m);
    (window as any).__headCameras = () => {
      headTracker.listCameras().then((cams) => {
        if (!cams.length) console.warn("[space] no video inputs visible to this browser");
        // Labels stay blank until camera permission has been granted once, so
        // start the tracker first if the list comes back unnamed.
        else console.table(cams);
      });
      return "listing cameras... (start the tracker first if the labels are blank)";
    };
    // The resolution/frame-rate dial, in megapixels. Watch the HUD's fps line
    // while turning it — that trade cannot be guessed from here.
    (window as any).__spacePixels = (megapixels?: number) => {
      if (typeof megapixels === "number") {
        SPACE.maxDrawPixels = Math.max(0.5e6, megapixels * 1e6);
        neuralSpace.refresh();
        resize();
      }
      return describePixelRatio(container, renderer.domElement as HTMLCanvasElement);
    };
    // See what the camera sees, with the tracker's own landmarks drawn on it.
    // Four video inputs on this machine and getUserMedia picks the default,
    // which need not be the one pointed at the viewer.
    // One command for the whole bring-up: restart, wait, diagnose.
    (window as any).__headSetup = () => {
      showHeadPreview(container);
      setupHeadTracking().then((r) => console.log("[headSetup] verdict:", r.verdict, r));
      return "setting up head tracking... (steps log as they complete, ~6s)";
    };
    (window as any).__headDiag = () => {
      diagnoseHeadTracking().then((r) => console.log("[headDiag] verdict:", r.verdict, r));
      return "running diagnostics... (each step logs as it completes)";
    };
    (window as any).__headCamera = (which: number | string) => {
      headTracker.useCamera(which).then((ok) => {
        console.log(ok ? "[space] camera switched" : "[space] camera switch FAILED", headTracker.status());
      });
      return "switching... (result will be logged)";
    };
    (window as any).__lookAt = (patch?: Record<string, number>) => {
      if (patch) {
        Object.assign(SPACE, patch);
        resetViewerLookAt();
        neuralSpace.refresh();
      }
      return {
        lookAt: SPACE.lookAt,
        maxYaw: SPACE.lookAtMaxYaw,
        maxPitch: SPACE.lookAtMaxPitch,
        speed: SPACE.lookAtSpeed,
      };
    };
    // The camera following the attended person, and who that is. Patch keys to
    // tune live, e.g. __follow({ headOrbitMaxDeg: 15, headOrbitGain: -1 }).
    (window as any).__follow = (patch?: Record<string, number>) => {
      if (patch) Object.assign(SPACE, patch);
      const st = headTracker.status();
      return {
        faces: st.faces,
        tracking: st.tracking,
        pose: st.pose,
        follow: { ...headFollow },
        gain: SPACE.headOrbitGain,
        maxDeg: SPACE.headOrbitMaxDeg,
        lookAt: SPACE.lookAt,
      };
    };
    (window as any).__headPreview = (show = true) =>
      (show ? showHeadPreview(container) : (hideHeadPreview(), "hidden"));
    // Hand the camera to the user for the duration of a drag. Without this the
    // drift overwrote OrbitControls' result every frame and the controls fought
    // back against a spring.
    controls.addEventListener("start", () => framing.setInteracting(true));
    controls.addEventListener("end", () => framing.setInteracting(false));
    (window as any).__spaceHud = (show = true) =>
      (show
        ? showSpaceHud(container, renderer.domElement as HTMLCanvasElement)
        : (hideSpaceHud(), "hidden"));
    // Pin the eye somewhere to see (and screenshot) the off-axis view from that
    // position without anyone standing there. null releases it back to tracking.
    (window as any).__viewerPose = (p?: { x?: number; y?: number; z?: number } | null) => {
      if (p === null) viewerTransform.unpin();
      else if (p) viewerTransform.pin(p.x ?? 0, p.y ?? 0, p.z ?? 1.2);
      return { ...viewerTransform.get(), pinned: viewerTransform.isPinned() };
    };
    (window as any).__headCalibrate = (trueMetres?: number) => {
      if (typeof trueMetres !== "number") {
        return "usage: __headCalibrate(1.2)  <- your ACTUAL distance from the camera, in metres";
      }
      return calibrateHeadDistance(trueMetres, viewerTransform.get().z);
    };
    (window as any).__offAxis = () => {
      const p = currentPanel(container);
      const implied = impliedPanelHeightM();
      return {
        panel: p,
        pose: { ...viewerTransform.get(), pinned: viewerTransform.isPinned() },
        // These two must agree, or the browser window is not filling the panel
        // and every distance in the projection is off by that ratio.
        measuredPanelHeightM: SPACE.panelHeightM,
        impliedPanelHeightM: +implied.toFixed(3),
        heightAgreement: +(implied / SPACE.panelHeightM).toFixed(3),
      };
    };
    (window as any).__protoCalibrate = (show = true) =>
      (show
        ? showPanelCalibration(container, renderer.domElement as HTMLCanvasElement)
        : (hidePanelCalibration(), "closed"));

    // Head tracking runs on the PROTO unit only. Auto-starting a camera on an
    // ordinary desktop load would be an unpleasant surprise, and the desktop
    // view has no window to be coupled to anyway.
    if (isProtoMode() && SPACE.headTracking) {
      headTracker.start().then((ok) => {
        if (!ok) {
          // Not fatal: without a tracked head the viewer pose eases to its
          // resting default and the scene falls back to a plain centred view.
          console.info("[space] head tracking unavailable; using the resting viewpoint");
        }
      });
    }
    // The automated-capture surface. Everything here already existed as separate
    // `__space*` / `__vrm*` handles; what this adds is the CLOCK, without which a
    // screenshot cannot be compared to another screenshot. See utils/debug/ai-debug.ts.
    aiDebug.attach({ camera, controls, renderer, scene });
    (window as any).__AI_DEBUG__ = {
      freeze: () => aiDebug.freeze(),
      play: () => aiDebug.play(),
      setTime: (sec: number) => aiDebug.setTime(sec),
      // Advance by exactly one rendered frame. Use this, not setTime, whenever
      // what is being measured has state that survives a frame — the temporal
      // resolve does. See the note on stepFrame() in ai-debug.ts.
      stepFrame: () => aiDebug.stepFrame(),
      endStep: () => aiDebug.endStep(),
      ready: () => aiDebug.ready(),
      setSeed: (seed: number | null) => aiDebug.setSeed(seed),
      suppressGrain: (on = true) => aiDebug.suppressGrain(on),
      /** "rest" = no idle clips (deterministic); "idle" = normal clip rotation. */
      setPosePreset: (name: "rest" | "idle") => {
        if (name === "rest") {
          vrmMotionPlayer.idleClips = false;
          vrmMotionPlayer.cancelGesture();
          vrmMotionPlayer.stop();
        } else {
          vrmMotionPlayer.idleClips = true;
        }
        return { pose: name, idleClips: vrmMotionPlayer.idleClips };
      },
      setCameraPreset: (name: string) => aiDebug.setCameraPreset(name),
      // Viewpoint motion. Separate from __space({...}) on purpose: that path
      // re-solves the framing, which would move the camera for a second reason
      // and ruin the very A/B these are for.
      setCameraDriftEnabled: (on: boolean) => aiDebug.setCameraDriftEnabled(on),
      setParallaxEnabled: (on: boolean) => aiDebug.setParallaxEnabled(on),
      setParallaxStrength: (v: number) => aiDebug.setParallaxStrength(v),
      // The environment already has a live config channel; a HUD "preset" is just
      // a named patch through it, so this stays a thin alias rather than a
      // second source of truth for what the console looks like.
      setHudPreset: (name: string) => {
        const presets: Record<string, Record<string, unknown>> = {
          secretary: { hud: 1.0, panels: 0.92, towers: 0.85, particles: 1 },
          minimal: { hud: 0.5, panels: 0, towers: 0.4, particles: 0.5 },
          debug: { hud: 1, panels: 1, towers: 1, particles: 0 },
          off: { hud: 0, panels: 0, towers: 0, particles: 0 },
        };
        const p = presets[name];
        if (!p) return { error: `unknown hud preset "${name}"`, available: Object.keys(presets) };
        Object.assign(SPACE, p);
        neuralSpace.rebuild();
        // rebuild() replaces the environment wholesale, so the temporal history
        // describes geometry that no longer exists. Same reasoning as
        // setCameraPreset: tests call this after the scene has already settled.
        taa.reset();
        return { preset: name, ...neuralSpace.describe() };
      },
      // Reproducible arbitrary view — a drag-orbit is not, so any A/B that
      // compares two runs frame-by-frame has to place the camera this way.
      setCamera: (pos: number[], target: number[]) => aiDebug.setCamera(pos, target),
      // The live camera object itself, for probes that must project a world
      // point (a bone) to screen pixels in the same frame they measure — a
      // per-frame round trip through getSceneInfo() cannot do that.
      camera: () => camera,
      // WHICH BONE OWNS THIS PIXEL — the instrument that separates "the shading
      // changed" from "a different surface won the depth test". Paints every
      // fragment by the class of its dominant skinning bone: red = hand/fingers,
      // green = skirt panels, blue = everything else. Two surfaces that
      // interpenetrate produce SPECKLE (isolated red inside green and vice
      // versa) which no amount of legitimate motion can create — a clean moving
      // silhouette has a clean boundary. Counting that speckle is a direct,
      // motion-immune measure of cloth-vs-body punch-through.
      classPass: (on: boolean = true) => {
        if (!on) {
          scene.overrideMaterial = null;
          return { classPass: false };
        }
        let skel: THREE.Skeleton | null = null;
        scene.traverse((o) => {
          const m = o as THREE.SkinnedMesh;
          if (!skel && m.isSkinnedMesh && m.skeleton) skel = m.skeleton;
        });
        if (!skel) return { error: "no skeleton" };
        const bones = (skel as THREE.Skeleton).bones;
        const data = new Uint8Array(bones.length * 4);
        let nHand = 0; let nSkirt = 0;
        bones.forEach((b, i) => {
          let v = 255; // other
          if (/手首|指/.test(b.name)) { v = 0; nHand += 1; } // hand + fingers
          else if (/^スカート/.test(b.name)) { v = 128; nSkirt += 1; }
          data[i * 4] = v;
          data[i * 4 + 3] = 255;
        });
        const tex = new THREE.DataTexture(data, bones.length, 1, THREE.RGBAFormat);
        tex.minFilter = THREE.NearestFilter;
        tex.magFilter = THREE.NearestFilter;
        tex.needsUpdate = true;
        scene.overrideMaterial = new THREE.ShaderMaterial({
          uniforms: { uClass: { value: tex }, uCount: { value: bones.length } },
          vertexShader:
            "#include <common>\n#include <skinning_pars_vertex>\n"
            + "uniform sampler2D uClass;\nuniform float uCount;\nvarying float vClass;\n"
            + "void main() {\n#include <begin_vertex>\n#include <skinbase_vertex>\n"
            + "  vClass = 1.0;\n"
            + "  #ifdef USE_SKINNING\n"
            + "    float bi = skinIndex.x; float bw = skinWeight.x;\n"
            + "    if (skinWeight.y > bw) { bw = skinWeight.y; bi = skinIndex.y; }\n"
            + "    if (skinWeight.z > bw) { bw = skinWeight.z; bi = skinIndex.z; }\n"
            + "    if (skinWeight.w > bw) { bw = skinWeight.w; bi = skinIndex.w; }\n"
            + "    vClass = texture2D(uClass, vec2((bi + 0.5) / uCount, 0.5)).r;\n"
            + "  #endif\n"
            + "#include <skinning_vertex>\n"
            + "  gl_Position = projectionMatrix * modelViewMatrix * vec4(transformed, 1.0);\n"
            + "  gl_PointSize = 1.0;\n}",
          fragmentShader:
            "varying float vClass;\nvoid main() {\n"
            + "  vec3 c = vec3(0.0, 0.0, 1.0);\n"
            + "  if (vClass < 0.25) c = vec3(1.0, 0.0, 0.0);\n"
            + "  else if (vClass < 0.75) c = vec3(0.0, 1.0, 0.0);\n"
            + "  gl_FragColor = vec4(c, 1.0);\n}",
        });
        return { classPass: true, bones: bones.length, handBones: nHand, skirtBones: nSkirt };
      },
      // The flash black-box: press "9" the moment the artefact is visible and
      // the last ~1.2s of frames upload to the server's cache/flash/ drop-box.
      flashDump: () => flashRecorder.dump(),
      ndcBounds: (prefix: string) => aiDebug.ndcBounds(prefix),
      getSceneInfo: () => aiDebug.getSceneInfo(),
      getRendererInfo: () => aiDebug.getRendererInfo(),
      // What a frame COSTS. getRendererInfo() reports draw calls, which do not
      // move when the render resolution does — so it cannot price the very
      // changes this surface exists to evaluate.
      getFrameStats: () => aiDebug.frameStats(),
      resetFrameStats: () => aiDebug.resetFrameStats(),
      getStats: () => aiDebug.getStats(),
      state: () => aiDebug.state(),
    };
    // `?debug=1&t=2&seed=1` — a capture tool can pin the scene from the URL alone,
    // which is the only form that is reproducible across a page reload.
    {
      const dp = debugParams();
      if (dp.debug) {
        aiDebug.suppressGrain(true);
        // Pin the POSE as well as the clock. Which idle clip is playing at a given
        // time depends on when its fetch completed, and that is network timing —
        // no seed controls it. Two runs of the identical build were measured 42%
        // different for this reason alone (arms out vs arms down), which would
        // make every look change indistinguishable from clip roulette.
        vrmMotionPlayer.idleClips = false;
        vrmMotionPlayer.cancelGesture();
        vrmMotionPlayer.stop();
        if (dp.seed !== null) aiDebug.setSeed(dp.seed);
        if (dp.t !== null) aiDebug.setTime(dp.t).catch(() => {});
        else aiDebug.freeze();
      }
    }
    (window as any).__spaceRebuild = () => {
      neuralSpace.rebuild();
      return neuralSpace.describe();
    };
    (window as any).__spaceQuality = (q: "low" | "medium" | "high") => {
      neuralSpace.setQuality(q);
      return neuralSpace.describe();
    };
    // Force the environment's conversation state without holding a conversation —
    // the listening ripple and the thinking data-stream are otherwise only
    // reachable mid-turn, which makes them impossible to tune.
    (window as any).__spaceState = (s: string) => {
      neuralSpace.setAiState(s as never);
      return neuralSpace.describe();
    };
    (window as any).__protoMode = (on: boolean) => {
      setProtoMode(on);
      if (!on) {
        // Back to the desktop bust-up framing the viewer ships with.
        camera.position.set(0, 1.05, 2.4);
        controls.target.set(0, 0.98, 0);
        controls.enabled = true;
        controls.update();
        setBgGrad(LOOK.bgInner, LOOK.bgOuter); // restore the desktop backdrop
      }
      applyFraming();
      resize(); // PROTO mode uses a different pixel-ratio cap
      return { proto: isProtoMode(), dpr: (window as any).__spaceDpr() };
    };

    // --- Post: scene -> line -> grade -> dof -> bloom -> diffusion -> output -> SMAA
    // The composer RT is MULTISAMPLED (geometry edges shimmer badly without it);
    // the line pass reads depth+normals from a dedicated NON-multisampled
    // pre-pass target rendered in the animate loop.
    // NOT a constant: the window can be dragged to a display with a different
    // scale factor, and a stale ratio means the compositor upscales a too-small
    // buffer (see utils/space/pixel-ratio.ts). Re-resolved in resize().
    let pixelRatio = resolvePixelRatio(container.clientWidth, container.clientHeight);
    const drawW = Math.max(1, Math.floor(container.clientWidth * pixelRatio));
    const drawH = Math.max(1, Math.floor(container.clientHeight * pixelRatio));
    const maxSamples = renderer.capabilities.maxSamples || 8;
    // Was a hard 4. Most desktop GPUs report 8 or 16, and the outline is extruded
    // GEOMETRY — the one thing MSAA is genuinely good at. Clamped to the driver's
    // own ceiling, so this is a no-op where 4 really is the maximum.
    const aaSamples = Math.min(SPACE.msaaSamples || 4, maxSamples);
    // Internal render scale (TAAU). The SCENE side renders at pixelRatio *
    // this; the TAA resolve reconstructs full resolution from the jittered
    // low-res frames, and the post side runs at full res. Only meaningful with
    // the accumulator running — without it a low-res render is just a stretch,
    // so taa:0 forces 1.
    const upscaleNow = () => (taa.enabled()
      ? Math.min(1, Math.max(0.5, SPACE.taaUpscale ?? 1)) : 1);
    // THE CHAIN IS SPLIT IN TWO at exactly the TAA seam, because an
    // EffectComposer ping-pongs two same-sized buffers and can therefore never
    // change resolution mid-chain. Scene side (this one, possibly low-res):
    // Render -> linePass -> GTAO. Post side (full-res): TAA bridge -> grade ->
    // DOF -> bloom -> diffusion -> Output -> SMAA. The TaaResolve sits between
    // them as a standalone step.
    const sceneComposer = new EffectComposer(
      renderer,
      new THREE.WebGLRenderTarget(
        Math.max(1, container.clientWidth),
        Math.max(1, container.clientHeight),
        { type: THREE.HalfFloatType, samples: aaSamples },
      ),
    );
    sceneComposer.renderToScreen = false;
    sceneComposer.setPixelRatio(pixelRatio * upscaleNow());
    sceneComposer.setSize(container.clientWidth, container.clientHeight);
    sceneComposer.addPass(new RenderPass(scene, camera));
    // SSAO supply (msTC folds SvSSAO/ImSSAO into the shade): GTAO contact
    // shading, composited before the grade so its darks pick up the same
    // blue-violet shadow lift as everything else.
    const gtaoPass = new GTAOPass(scene, camera, container.clientWidth, container.clientHeight);
    gtaoPass.updateGtaoMaterial({ radius: 0.25, distanceExponent: 1, thickness: 1, scale: 1, samples: 16 });
    gtaoPass.blendIntensity = LOOK.ssao;
    gtaoPass.enabled = LOOK.ssao > 0.001;
    // HALF-RESOLUTION AO on the render host (2026-11-05, 60 fps on the PROTO). At the full
    // 2160x3840 GTAO (its own scene pass for normals + 16-sample AO + denoise) was the
    // biggest GPU cost of a frame — the render used ~60% of GPU0, ~25% with AO off — and with
    // the TTS on the same GPU the frame no longer fit: 38-46 fps while she spoke, 54-58 with AO
    // off. AO is low-frequency and denoised, so half size reads the same at a quarter of the
    // cost; it is blended back at full size. Desktop viewers keep full size.
    // localStorage.gtaoScale overrides (1 = full size).
    const gtaoScale = (() => {
      try {
        const v = parseFloat(localStorage.getItem("gtaoScale") || "");
        if (v > 0 && v <= 1) return v;
      } catch { /* storage blocked */ }
      return isRenderSource ? 0.5 : 1;
    })();
    if (gtaoScale !== 1) {
      const setSizeFull = gtaoPass.setSize.bind(gtaoPass);
      gtaoPass.setSize = (w: number, h: number) =>
        setSizeFull(Math.max(1, Math.round(w * gtaoScale)), Math.max(1, Math.round(h * gtaoScale)));
      gtaoPass.setSize(container.clientWidth, container.clientHeight);
    }
    // Objects flagged userData.hideFromOverrides sit out GTAO's depth/normal render
    // (an override material, so it draws everything): the floor mirror there was
    // solid floor to the AO and darkened the air behind her shins — the "shadow
    // behind her" (space/floor-reflection.ts).
    {
      const gtaoRender = gtaoPass.render.bind(gtaoPass);
      gtaoPass.render = ((...args: Parameters<typeof gtaoPass.render>) => {
        const hidden: THREE.Object3D[] = [];
        scene.traverse((o: THREE.Object3D) => {
          if (o.userData.hideFromOverrides && o.visible) { o.visible = false; hidden.push(o); }
        });
        try {
          gtaoRender(...args);
        } finally {
          hidden.forEach((o) => { o.visible = true; });
        }
      }) as typeof gtaoPass.render;
    }
    // Confine AO to the avatar's own volume — a cheap belt-and-braces limit now that the
    // backdrop lives outside the composer (see setBgGrad). It is NOT what fixed the backdrop
    // step: measured with the model posed and frozen, the straight step at y=397 survived
    // both with and without this box and only disappeared when ssao went to 0, because a box
    // big enough to hold her also holds the broken far-plane reconstruction. Kept loose
    // (±1.5m laterally, 3m up, 1m below the floor) so hair, skirt and a raised arm stay in.
    gtaoPass.setSceneClipBox(new THREE.Box3(
      new THREE.Vector3(-1.5, -1.0, -1.5),
      new THREE.Vector3(1.5, 3.0, 1.5),
    ));
    // __vrmAO() reports the pass state; __vrmAO(min, max) re-aims the clip box, __vrmAO(null)
    // removes it. Kept because the AO pass is the one thing here that paints OUTSIDE the
    // avatar, and that is invisible in code but obvious as a straight step in the backdrop.
    (window as any).__vrmAO = (min?: number[] | null, max?: number[]) => {
      if (min === null) gtaoPass.setSceneClipBox(null as never);
      else if (min && max) {
        gtaoPass.setSceneClipBox(new THREE.Box3(
          new THREE.Vector3(min[0], min[1], min[2]),
          new THREE.Vector3(max[0], max[1], max[2]),
        ));
      }
      const m = (gtaoPass as any).gtaoMaterial;
      return {
        enabled: gtaoPass.enabled,
        blendIntensity: gtaoPass.blendIntensity,
        clipBox: m?.defines?.SCENE_CLIP_BOX,
        boxMin: m?.uniforms?.sceneBoxMin?.value?.toArray?.(),
        boxMax: m?.uniforms?.sceneBoxMax?.value?.toArray?.(),
        radius: m?.uniforms?.radius?.value,
      };
    };
    // Line pre-pass: normals in the colour attachment, scene depth in the
    // attached DepthTexture — one extra render, feeds both edge detectors.
    const normalRT = new THREE.WebGLRenderTarget(drawW, drawH);
    normalRT.depthTexture = new THREE.DepthTexture(drawW, drawH);
    const normalMat = new THREE.MeshNormalMaterial();
    const linePass = new ShaderPass(LineShader);
    linePass.uniforms.tDepth.value = normalRT.depthTexture;
    linePass.uniforms.tNormal.value = normalRT.texture;
    linePass.uniforms.uCamNear.value = camera.near;
    linePass.uniforms.uCamFar.value = camera.far;
    linePass.uniforms.uTexel.value.set(1 / drawW, 1 / drawH);
    linePass.enabled = LOOK.lineInt + LOOK.lineNInt + LOOK.lineZInt > 0.001;
    // linePass MUST be the first needsSwap pass: it samples the depth texture
    // attached to renderTarget2, so it must always WRITE renderTarget1. GTAO
    // (which renders its own depth/normals) composites after it.
    sceneComposer.addPass(linePass);
    sceneComposer.addPass(gtaoPass);
    // Temporal resolve — the bridge between the two composers. AFTER GTAO (so
    // it cannot displace linePass from the head of the scene chain) and BEFORE
    // the grade (its neighbourhood clamp needs a sharp frame, and grain must
    // never enter the history). History at OUTPUT res, input at scene res.
    const inW = Math.max(1, Math.floor(container.clientWidth * pixelRatio * upscaleNow()));
    const inH = Math.max(1, Math.floor(container.clientHeight * pixelRatio * upscaleNow()));
    const taaResolve = new TaaResolve(drawW, drawH, inW, inH);
    let taaWasOn = taa.enabled(); // edge-detects __space({taa}) toggles in the loop
    // Motion vectors for the resolve — a scene pre-pass like normalRT, not a
    // composer pass, because it renders geometry rather than a fullscreen quad.
    const taaVelocity = new TaaVelocity(inW, inH);
    taaResolve.setVelocity(taaVelocity.texture, taaVelocity.rt.depthTexture);
    const postComposer = new EffectComposer(
      renderer,
      new THREE.WebGLRenderTarget(
        Math.max(1, container.clientWidth),
        Math.max(1, container.clientHeight),
        { type: THREE.HalfFloatType }, // no MSAA: this side only composites quads
      ),
    );
    postComposer.setPixelRatio(pixelRatio);
    postComposer.setSize(container.clientWidth, container.clientHeight);
    // The bridge pass: copy/sharpen/debug-view, fed each frame with either the
    // TAA output (full res) or, with taa:0, the scene composer's own result.
    // The second ctor arg names a texture that does not exist ON PURPOSE —
    // ShaderPass would otherwise overwrite tDiffuse with its readBuffer.
    postComposer.addPass(new ShaderPass(taaResolve.copyMat, "tTaaBridge"));
    const gradePass = new ShaderPass(GradeShader);
    postComposer.addPass(gradePass); // cool/purple colour grade before bloom
    // Depth of field — blurs the far hair/edges by distance (focus tracks the model).
    const bokehPass = new BokehPass(scene, camera, {
      focus: 3.0,
      aperture: LOOK.dofAperture,
      maxblur: LOOK.dofBlur,
    });
    bokehPass.enabled = LOOK.dofAperture > 0.00001;
    postComposer.addPass(bokehPass);
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(container.clientWidth, container.clientHeight),
      LOOK.bloomStrength,
      LOOK.bloomRadius,
      LOOK.bloomThreshold,
    );
    postComposer.addPass(bloomPass);
    // ikDiffusion analogue: threshold-0 wide soft glow over the whole frame.
    const diffPass = new UnrealBloomPass(
      new THREE.Vector2(container.clientWidth, container.clientHeight),
      LOOK.diffusion,
      1.0,
      0.0,
    );
    diffPass.enabled = LOOK.diffusion > 0.001;
    postComposer.addPass(diffPass);
    postComposer.addPass(new OutputPass());
    const smaaPass = new SMAAPass(container.clientWidth, container.clientHeight);
    postComposer.addPass(smaaPass);
    // SMAA re-decides its edge blends from scratch every frame, which on a
    // dense moving pattern can flip per frame — a shimmer source that sits
    // AFTER the temporal resolve and thus outside its reach. It was never
    // testable in isolation before (every other stage has a LOOK/space knob).
    (window as any).__smaa = (on: boolean = true) => {
      smaaPass.enabled = !!on;
      return { smaa: smaaPass.enabled };
    };

    // View-space rim direction from azimuth (0 = front, +90 = screen-right) +
    // elevation (up/down), in degrees. Default cool -45 / warm +45 reproduce the
    // original fixed (-1,0,-1)/(1,0,-1) screen-left / screen-right rims.
    const dirFrom = (azDeg: number, elDeg: number) => {
      const az = (azDeg * Math.PI) / 180;
      const el = (elDeg * Math.PI) / 180;
      return new THREE.Vector3(
        Math.sin(az) * Math.cos(el),
        Math.sin(el),
        -Math.cos(az) * Math.cos(el),
      ).normalize();
    };

    // msTC Edge stage: the VRM was exported with outlineWidthMode "none" on the
    // hair/face materials, so three-vrm generated no outline meshes for them —
    // but the msTC pipeline (MMD標準エッジ → Edge.x) outlines everything.
    // Replicate the loader plugin's _generateOutline at runtime for those
    // materials; visibility is driven by LOOK.outlineAll (輪郭 全材質).
    // Must run BEFORE applyLook so the clone happens pre-shader-injection.
    // Skip: alpha overlays / eyes (a hull would z-fight) AND the face itself —
    // a hull outline on the face draws a thick black ring around the mouth when
    // it opens (唇の線が太い/口を開くと黒い). The model authored the face with
    // outlineWidthMode "none" on purpose; the lip/jaw lines live in the texture.
    const OUTLINE_SKIP = /HL|Hairshadow|eye|Facial|^face0/i;
    const generateMissingOutlines = (v: VRM) => {
      v.scene.traverse((obj: THREE.Object3D) => {
        const mesh = obj as THREE.Mesh;
        if (!mesh.isMesh || Array.isArray(mesh.material)) return;
        const m = mesh.material as any;
        if (!m?.isMToonMaterial || m.isOutline) return;
        if (m.outlineWidthMode !== "none") return; // the plugin already made one
        if (m.transparent || OUTLINE_SKIP.test(m.name || "")) return;
        const geometry = mesh.geometry as THREE.BufferGeometry;
        if (geometry.groups.length > 0) return;
        const outline = m.clone();
        outline.name = `${m.name} (Outline)`;
        outline.isOutline = true;
        outline.side = THREE.BackSide;
        // screenCoordinates, NOT worldCoordinates — see SCREEN_OUTLINE below.
        outline.outlineWidthMode = "screenCoordinates";
        // The face outline must be MUCH thinner — at body thickness it drew a
        // heavy line down the jaw with skin bleeding past it (あごの線が肌色).
        const isFaceOutline = /^face0\d/i.test(m.name);
        outline.outlineWidthFactor = isFaceOutline
          ? SCREEN_OUTLINE.face
          : SCREEN_OUTLINE.body;
        outline.outlineColorFactor.setRGB(0.03, 0.02, 0.05);
        outline.userData._genOutline = true;
        outline.userData._genFace = isFaceOutline;
        outline.visible = LOOK.outlineAll >= 0.5;
        mesh.material = [m, outline];
        // addGroup's `count` is a VERTEX/INDEX count, not a triangle count — the
        // "/ 3" here drew only a third of the triangles for BOTH the base and
        // the outline material, i.e. two thirds of the mesh silently vanished.
        // It never fired on this model (every geometry is indexed, verified via
        // __vrmMats), but VRMUtils.removeUnnecessaryVertices above can change
        // indexing and any non-indexed mesh would have hit it. Matches
        // three-vrm's own _generateOutline.
        const primitiveVertices = geometry.index
          ? geometry.index.count
          : geometry.attributes.position.count;
        geometry.addGroup(0, primitiveVertices, 0);
        geometry.addGroup(0, primitiveVertices, 1);
      });
    };

    // Live sub-uniform sets (one per injected material) — the animate loop
    // pushes the per-frame view-space point-light position into these.
    const subUList: any[] = [];

    // 顔法線: blend the face geometry normals toward a sphere centred behind the
    // face — face shading falls in clean anime shapes instead of tracing the
    // nose/cheek geometry (msTC v2.1 NORMAL_RATE_MODE / ウマ娘方式).
    const spherizeFaceNormals = (mesh: THREE.Mesh) => {
      const g = mesh.geometry as THREE.BufferGeometry;
      const nor = g.attributes.normal as THREE.BufferAttribute;
      const pos = g.attributes.position as THREE.BufferAttribute;
      if (!nor || !pos) return;
      if (!g.userData._nBase) g.userData._nBase = (nor.array as Float32Array).slice();
      const t = LOOK.faceNormal;
      if (g.userData._nApplied === t) return;
      g.userData._nApplied = t;
      const base = g.userData._nBase as Float32Array;
      if (!g.boundingBox) g.computeBoundingBox();
      const c = g.boundingBox!.getCenter(new THREE.Vector3());
      c.z -= (g.boundingBox!.max.z - g.boundingBox!.min.z) * 0.6; // centre behind the face
      const sp = new THREE.Vector3();
      for (let i = 0; i < pos.count; i += 1) {
        sp.set(pos.getX(i) - c.x, pos.getY(i) - c.y, pos.getZ(i) - c.z).normalize();
        sp.set(
          base[i * 3] * (1 - t) + sp.x * t,
          base[i * 3 + 1] * (1 - t) + sp.y * t,
          base[i * 3 + 2] * (1 - t) + sp.z * t,
        ).normalize();
        nor.setXYZ(i, sp.x, sp.y, sp.z);
      }
      nor.needsUpdate = true;
    };

    // Auto-load the -雾雾雾- YYB Miku mask set (bowlroll file/304270, deployed
    // under Live2D Characters/vrm/mstc-masks/): per-material SHADOW_RATE masks,
    // SUBLIGHT_RATE masks, and 純色化 (purified) albedo replacements. Missing
    // files just resolve to null — the table is a best-effort name match.
    const MASK_BASE = "/Live2D%20Characters/vrm/mstc-masks";
    // NOTE: the q/C series (sleeve & skirt translucent overlay panels) keep
    // their ORIGINAL textures — the distributed pure versions lose the pattern
    // detail and alpha (「スカートや腕の模様が消える」), and the q3 mask's
    // material mapping was a bad guess (purple patches on the skirt).
    // body01 keeps only the PURE swap — its painted Shade/SubLight masks were
    // authored against the PMX outfit and land wrong on the VRM (ネクタイ周りに
    // 変な影). The hair masks line up correctly and stay.
    const MASK_TABLE: [RegExp, string | null, string | null, string | null][] = [
      [/^body01$/i, "Pure/body.png", null, null],
      // face-lip.png: auto-generated from tex/body.png (lip-pink pixel detect) —
      // G channel forces the lip crescent to ハイ1 = the video's glossy lips.
      [/^face0\d/i, null, "Shade/face-lip.png", null],
      // Hair keeps its ORIGINAL texture (#276A91 steel blue = the model's real
      // default). The distributed Pure/Hair.png is #6ABFB0 mint-green — that was
      // the "hair colour looks wrong" culprit, so no pure swap for hair.
      // q301/q302 — the printed circuit panels. VERIFIED against the UV rather
      // than guessed this time: 100% of the non-zero texels in Shade/q3.png and
      // SubLight/q3.png fall inside tex/q3.png's own alpha, so the masks are
      // authored in exactly this material's UV space. Only the PURE swap is
      // wrong for it (28.8% overlap), and that is what produced the purple
      // patches the note above records — so the pure slot stays null.
      //
      // What the author painted, by channel: SubLight uses B alone (8.57% of the
      // sheet) = CANCEL the sublight, and Shade uses G alone (7.62%) = force
      // highlight. In other words the print is meant to be lit, NOT rimmed — and
      // running it without the mask is what put a cyan edge on every white speck.
      [/^q30\d/i, null, "Shade/q3.png", "SubLight/q3.png"],
      [/^Hair01/i, null, "Shade/Hair01.png", "SubLight/Hair01.png"],
      [/^Hair02/i, null, "Shade/Hair02.png", "SubLight/Hair02.png"],
      [/^Hair03/i, null, "Shade/Hair01.png", "SubLight/Hair01.png"],
    ];
    const mstcTexCache = new Map<string, Promise<THREE.Texture | null>>();
    const loadMstcTex = (rel: string, srgb: boolean) => {
      if (!mstcTexCache.has(rel)) {
        mstcTexCache.set(
          rel,
          new Promise((res) => {
            new THREE.TextureLoader().load(
              `${MASK_BASE}/${rel}`,
              (t) => {
                t.flipY = false; // glTF-style UVs
                if (srgb) t.colorSpace = THREE.SRGBColorSpace;
                t.wrapS = THREE.RepeatWrapping;
                t.wrapT = THREE.RepeatWrapping;
                t.anisotropy = renderer.capabilities.getMaxAnisotropy();
                res(t);
              },
              undefined,
              () => res(null),
            );
          }),
        );
      }
      return mstcTexCache.get(rel)!;
    };
    const loadMstcMasks = (v: VRM) => {
      const jobs: Promise<void>[] = [];
      v.scene.traverse((obj: THREE.Object3D) => {
        const mesh = obj as THREE.Mesh;
        if (!mesh.isMesh) return;
        const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        mats.forEach((mat: THREE.Material) => {
          const m = mat as any;
          if (!m || m.isOutline) return; // outlines reuse the surface look anyway
          const nm = (m.name || "").replace(" (Outline)", "");
          const row = MASK_TABLE.find(([re]) => re.test(nm));
          if (!row) return;
          const [, pure, shade, sub] = row;
          if (pure) jobs.push(loadMstcTex(pure, true).then((t) => { if (t) m.userData._pureTex = t; }));
          if (shade) jobs.push(loadMstcTex(shade, false).then((t) => { if (t) m.userData._maskTex = t; }));
          if (sub) jobs.push(loadMstcTex(sub, false).then((t) => { if (t) m.userData._maskSubTex = t; }));
        });
      });
      Promise.all(jobs).then(() => {
        if (!disposed && vrm) applyLook(vrm); // push the loaded textures/flags
      });
    };

    // 口幅縮小 (作者の口幅縮小モーフ相当): pull the mouth-ring vertices (those the
    // あ morph moves) toward the mouth's X centre. Static & reversible — caches
    // the base X once; mouthNarrow 0 restores. Runs on every mesh carrying あ
    // (face01 lips + oral cavity + tooth + tongue) so they narrow together.
    const narrowMouth = (mesh: THREE.Mesh) => {
      const g = mesh.geometry as THREE.BufferGeometry;
      const aIdx = (mesh.morphTargetDictionary as any)?.["あ"];
      if (aIdx == null) return;
      const aDelta = g.morphAttributes.position?.[aIdx] as THREE.BufferAttribute | undefined;
      const pos = g.attributes.position as THREE.BufferAttribute;
      if (!aDelta || !pos) return;
      if (!g.userData._mouthMaskV2) {
        // ONLY THE LIPS. The あ morph also moves the cheeks/jaw a little; masking
        // every moved vertex dragged the whole face inward (顔の形がおかしくなる).
        // Keep only verts with a STRONG あ delta AND inside a tight Y band around
        // the mouth — cheeks (weak delta) and jaw (off the band) are excluded.
        const baseX = new Float32Array(pos.count);
        const mm = new Float32Array(pos.count);
        let maxM = 0;
        for (let i = 0; i < pos.count; i += 1) {
          baseX[i] = pos.getX(i);
          const m = Math.abs(aDelta.getX(i)) + Math.abs(aDelta.getY(i)) + Math.abs(aDelta.getZ(i));
          mm[i] = m;
          if (m > maxM) maxM = m;
        }
        const thresh = Math.max(0.006, maxM * 0.4); // strong movers only
        let sy = 0, sw = 0;
        for (let i = 0; i < pos.count; i += 1) {
          if (mm[i] > thresh) { sy += pos.getY(i) * mm[i]; sw += mm[i]; }
        }
        const mouthY = sw > 0 ? sy / sw : 0;
        const mask = new Uint8Array(pos.count);
        for (let i = 0; i < pos.count; i += 1) {
          if (mm[i] > thresh && Math.abs(pos.getY(i) - mouthY) < 0.05) mask[i] = 1;
        }
        g.userData._mouthBaseX = baseX;
        g.userData._mouthMask = mask;
        g.userData._mouthMaskV2 = true; // X-symmetric model → mouth centre is 0
      }
      if (g.userData._narrowApplied === LOOK.mouthNarrow) return;
      g.userData._narrowApplied = LOOK.mouthNarrow;
      const baseX = g.userData._mouthBaseX as Float32Array;
      const mask = g.userData._mouthMask as Uint8Array;
      for (let i = 0; i < pos.count; i += 1) {
        if (mask[i]) pos.setX(i, baseX[i] * (1 - LOOK.mouthNarrow));
      }
      pos.needsUpdate = true;
    };

    // Inject the sublight into every material and push the current uniform values.
    const applyLook = (v: VRM) => {
      const coolDir = dirFrom(LOOK.subCoolAzim, LOOK.subCoolElev);
      const warmDir = dirFrom(LOOK.subWarmAzim, LOOK.subWarmElev);
      const forceHiSet = new Set(
        LOOK.forceHiMats.split(",").map((s) => s.trim()).filter(Boolean),
      );
      subUList.length = 0;
      v.scene.traverse((obj: THREE.Object3D) => {
        const mesh = obj as THREE.Mesh;
        if (!mesh.isMesh) return;
        const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        if (
          mats.length > 0 &&
          mats.every((mt: any) => /^face0\d/i.test((mt?.name || "").replace(" (Outline)", "")))
        ) {
          spherizeFaceNormals(mesh);
        }
        if ((mesh.morphTargetDictionary as any)?.["あ"] != null) narrowMouth(mesh);
        // Detect (once) which body region this mesh is skinned to — the reliable
        // way to find the skirt/arm translucent panels, whatever their names.
        if (mesh.userData._skirtSkinned === undefined) {
          mesh.userData._skirtSkinned = false;
          mesh.userData._armSkinned = false;
          const sm = mesh as THREE.SkinnedMesh;
          if ((sm as any).isSkinnedMesh && sm.skeleton) {
            const g = mesh.geometry as THREE.BufferGeometry;
            const si = g.attributes.skinIndex as any;
            const sw = g.attributes.skinWeight as any;
            if (si && sw) {
              const getters = ["getX", "getY", "getZ", "getW"];
              outer: for (let i = 0; i < si.count; i += 7) { // stride sample
                for (let k = 0; k < 4; k += 1) {
                  if (sw[getters[k]](i) > 0.3) {
                    const bn = sm.skeleton.bones[si[getters[k]](i)]?.name || "";
                    if (bn.startsWith("スカート")) mesh.userData._skirtSkinned = true;
                    // this model suffixes sides: 腕R/腕L, ひじR, 手首L, 袖...
                    if (/^(腕|ひじ|手首|袖)/.test(bn)) mesh.userData._armSkinned = true;
                    if (mesh.userData._skirtSkinned && mesh.userData._armSkinned) break outer;
                  }
                }
              }
            }
          }
        }
        const skirtMesh = mesh.userData._skirtSkinned === true;
        const armMesh = mesh.userData._armSkinned === true;
        // Eye / mouth-interior meshes stop receiving cast shadows (bangs shadow
        // across the sclera, head shadow filling the mouth cavity).
        if (
          mats.length > 0 &&
          mats.every((mt: any) => {
            const nm = (mt?.name || "").replace(" (Outline)", "");
            return EYE_MATS.test(nm) || MOUTH_MATS.test(nm);
          })
        ) {
          mesh.receiveShadow = false;
        }
        mats.forEach((mat: THREE.Material) => {
          const m = mat as any;
          if (!m) return;
          const matName = (m.name || "").replace(" (Outline)", "");
          const skinM = SKIN_MATS.test(matName);
          const eyeM = EYE_MATS.test(matName);
          const mouthM = MOUTH_MATS.test(matName);
          // 口内は両面描画: 笑い口の隙間からは口腔メッシュの裏面が見えるため、
          // 片面のままだとカリングされて背景(=真っ黒)が抜けて見える。
          // ※face01/02 は両面化NG — face02(顔影オーバーレイ)の裏面が頭全体に
          // 被さって顔が真っ暗になる (実験済み)。
          if (mouthM && m.side !== THREE.DoubleSide) {
            m.side = THREE.DoubleSide;
            m.shadowSide = THREE.FrontSide;
            m.needsUpdate = true;
          }
          // スカート/腕 不透明: kill the translucent panels (reversible — the
          // authored transparent/depthWrite flags are cached and restored). The
          // arm toggle only touches AUTHORED-transparent materials so the (opaque)
          // body/jacket sharing the same skinning stay untouched.
          {
            const skirtTarget = skirtMesh || SKIRT_MATS.test(matName);
            const armTarget = !skirtTarget && armMesh;
            if ((skirtTarget || armTarget) && "transparent" in m && !eyeM && !/^Hair/i.test(matName)) {
              if (m.userData._trBase === undefined) {
                m.userData._trBase = m.transparent;
                m.userData._dwBase = m.depthWrite;
              }
              const opaque = skirtTarget
                ? LOOK.skirtOpaque >= 0.5
                : LOOK.armOpaque >= 0.5 && m.userData._trBase === true;
              const wantT = opaque ? false : m.userData._trBase;
              const wantD = opaque ? true : m.userData._dwBase;
              if (m.transparent !== wantT || m.depthWrite !== wantD) {
                m.transparent = wantT;
                m.depthWrite = wantD;
                m.needsUpdate = true;
              }
            }
          }
          // DEPTH WRITE STAYS ON for the translucent layers — see the source
          // model. Every one of the PMX's 31 materials is authored at diffuse
          // alpha 1.000; the see-through look comes ENTIRELY from the textures'
          // alpha channels (q2.png, the sleeve overlay, is 80% partial alpha;
          // C.png 35%). So each panel is meant to be composited ONCE, exactly as
          // its texture says. Turning depthWrite off here was tried and made the
          // coat wash out: with nothing rejecting the layers behind, every
          // overlapping panel blended again on top of the last and the compound
          // transparency ate through to the shoulder (「透過しすぎてる」「肩の
          // ところまで透過している」). Depth write is what keeps it to one blend.
          //
          // ...and the outer layers draw AFTER the hair. Both are alpha-blended,
          // so three.js orders them by mesh-centroid distance — decided by where
          // a whole twintail averages out, not by what is actually in front at a
          // given pixel. When that sort put a coat panel first, the panel wrote
          // depth across its quad and every hair fragment behind it was rejected
          // and never rasterised: the hair was not see-through, it was ABSENT.
          // The arm never showed the fault because it is opaque and therefore
          // already in the framebuffer before any of this
          // (「腕の描画はうまくいってるんだけど髪の描画だけがうまくいってない」).
          //
          // Drawing HAIR FIRST and the panels after — both still writing depth —
          // is correct in both directions, which is why it beats forcing either
          // one on top:
          //   panel nearer -> passes the depth test, blends ONCE over the hair
          //                   already in the buffer = the hair shows through it,
          //                   at exactly the texture's alpha (MMD's look)
          //   hair nearer  -> the panel is depth-rejected there, hair survives
          // Ordering was the bug; occlusion and blending were both fine.
          if (m.transparent && !/^Hair0\d/i.test(matName) && mesh.renderOrder < 1) {
            mesh.renderOrder = 1;
          }
          // 眉の露出 (前髪透け): Facial (brows/lashes) skips the depth test so it
          // draws over the bangs — the guide's "前髪の影と眉の露出が2D感の要".
          if (/^Facial/i.test(matName)) {
            if (m.userData._dtBase === undefined) m.userData._dtBase = m.depthTest;
            m.depthTest = LOOK.browFront >= 0.5 ? false : m.userData._dtBase;
            if (mesh.renderOrder !== undefined) {
              if (mesh.userData._roBase === undefined) mesh.userData._roBase = mesh.renderOrder;
              mesh.renderOrder = LOOK.browFront >= 0.5 ? 999 : mesh.userData._roBase;
            }
          }
          // AutoLuminous analogue: brighten the HL sparkle materials past the
          // bloom threshold so they glow selectively.
          if (/^HL/i.test(matName) && m.color?.copy) {
            if (!m.userData._colBase) m.userData._colBase = m.color.clone();
            m.color.copy(m.userData._colBase).multiplyScalar(1 + LOOK.glowInt * 2);
          }
          // Light reflection / shine: scale the MToon matcap (sphere-map highlight).
          if (m.matcapFactor?.copy) {
            if (!m.userData._mcBase) m.userData._mcBase = m.matcapFactor.clone();
            m.matcapFactor.copy(m.userData._mcBase).multiplyScalar(LOOK.matcapInt);
          }
          // Cel bands: darken the shade colour + widen the shade area so the
          // quantized shading reads as real cel shadows under the flat fills.
          if (m.shadeColorFactor?.copy) {
            if (!m.userData._shadeBase) m.userData._shadeBase = m.shadeColorFactor.clone();
            const sk = LOOK.celBands >= 2 ? 0.42 : 1.0;
            m.shadeColorFactor.copy(m.userData._shadeBase).multiplyScalar(sk);
          }
          if ("shadingShiftFactor" in m) {
            if (m.userData._shiftBase === undefined) m.userData._shiftBase = m.shadingShiftFactor;
            // MToon: HIGHER shadingShift = more lit, so subtract — the 影しきい値
            // slider follows msTC's 影1しきい値 (+ = wider shade area). faceLit
            // is the article's ShadeRT R-mask: skin is pushed toward lit so only
            // cast shadows (bangs) land on the face.
            m.shadingShiftFactor =
              (LOOK.celBands >= 2 ? m.userData._shiftBase - 0.22 : m.userData._shiftBase)
              - LOOK.shadeShift
              + (skinM ? LOOK.faceLit : 0)
              + (eyeM || mouthM ? 1 : 0); // eyes / mouth interior: always fully lit
          }
          // msTC 影とがり: override the authored shadingToonyFactor (-1 keeps it).
          if ("shadingToonyFactor" in m) {
            if (m.userData._toonyBase === undefined) m.userData._toonyBase = m.shadingToonyFactor;
            m.shadingToonyFactor = LOOK.shadeToony >= 0 ? LOOK.shadeToony : m.userData._toonyBase;
          }
          // msTC エッジ: scale / recolour the authored MToon outline. BEFORE the
          // skin early-return — body01's outline clone is skin-named. Hair/face
          // were exported with outlineWidthMode "none" (no outline mesh exists),
          // so this only drives the body/clothes edges.
          if ("outlineWidthFactor" in m) {
            if (m.userData._genOutline) m.visible = LOOK.outlineAll >= 0.5;
            // 口内 (口腔/歯/舌) の輪郭は完全にオフ: 狭い口では太さを絞っても口を
            // 一周する太い黒紫のリングに見える (口を開くと口枠が黒い). Hide the
            // outline material AND zero its width — belt and braces.
            if (mouthM && m.isOutline) m.visible = false;
            if (m.userData._olwBase === undefined) {
              // Convert the AUTHORED outlines too, once. The runtime-generated
              // ones are already screenCoordinates (generateMissingOutlines);
              // these came from the VRM in metres and are the ones measured at
              // 0.53 px. Rebasing here rather than at load keeps the conversion
              // in one place with the clamp that gives it its point.
              if (m.outlineWidthMode === "worldCoordinates") {
                m.outlineWidthMode = "screenCoordinates";
                m.userData._olwBase = SCREEN_OUTLINE.body;
                m.needsUpdate = true;
              } else {
                m.userData._olwBase = m.outlineWidthFactor;
              }
            }
            const olw =
              m.userData._olwBase * (mouthM ? 0 : m.userData._genFace ? 1 : LOOK.outlineWidth);
            // Floor, not scale: 0 still means "no outline" (the mouth interior
            // relies on that), but anything asked to be visible is at least a
            // pixel wide. A preset is free to want a thin line; it is not free
            // to want one that only exists on some of its own rows.
            m.outlineWidthFactor = olw > 0 ? Math.max(olw, SCREEN_OUTLINE.min) : 0;
            if (m.outlineColorFactor?.copy) {
              if (!m.userData._olcBase) m.userData._olcBase = m.outlineColorFactor.clone();
              if (LOOK.outlineColor >= 0) m.outlineColorFactor.set(LOOK.outlineColor);
              else m.outlineColorFactor.copy(m.userData._olcBase);
            }
            if ("outlineLightingMixFactor" in m) {
              if (m.userData._olmBase === undefined) m.userData._olmBase = m.outlineLightingMixFactor;
              m.outlineLightingMixFactor = LOOK.outlineMix >= 0 ? LOOK.outlineMix : m.userData._olmBase;
            }
          }
          // Skin: warm emissive floor, and the additive cyan/magenta sublight is
          // scaled by skinSub (0 = classic 肌にサブライトを当てない). The msTC
          // shade/highlight recolor DOES apply to skin — the face shade is where
          // the demo video's pink 色トレス reads most.
          const isSkin = skinM;
          if (isSkin) {
            const w = LOOK.skinWarmth;
            if (m.emissive?.setRGB) {
              m.emissive.setRGB(0.6 * w, 0.47 * w, 0.42 * w);
              if ("emissiveIntensity" in m) m.emissiveIntensity = 1;
            }
          }
          injectSublight(m);
          const u = m.userData?._subU;
          if (!u) return;
          const skinScale = isSkin ? LOOK.skinSub : 1;
          u.uFresInt.value = fx(LOOK.fresInt * skinScale, 0);
          u.uFresPow.value = LOOK.fresPow;
          u.uFresColor.value.set(LOOK.fresColor);
          aiFx.adoptRim(u); // no-op unless the model opted in (overrides the fresnel above)
          u.uSubInt.value = fx(LOOK.subInt * skinScale, 0); // the coloured rim
          u.uSubPow.value = LOOK.subPow;
          u.uSubMix.value = LOOK.subMix;
          u.uSubIntCool.value = LOOK.subIntCool;
          u.uSubIntWarm.value = LOOK.subIntWarm;
          u.uCelBands.value = fx(LOOK.celBands, 0); // toon banding
          u.uSubCoolColor.value.set(LOOK.coolColor);
          u.uSubWarmColor.value.set(LOOK.warmColor);
          u.uSubCoolDir.value.copy(coolDir);
          u.uSubWarmDir.value.copy(warmDir);
          {
            // Per-part albedo colouring: hair = optional direct colour blend
            // (動画スポイト色) + hue shift; IRIS only = eye hue shift; the white
            // eye parts (eyeball/HL) get NO base colour ops at all — the sat
            // pivot was tinting the near-achromatic sclera yellow.
            const hairM = /^Hair/i.test(matName);
            const irisM = /^eyes\b/i.test(matName);
            const whiteEyeM = eyeM && !irisM;
            u.uBaseSat.value = whiteEyeM ? 0.5 : LOOK.baseSat;
            u.uBaseVal.value = whiteEyeM ? 0.5 : LOOK.baseVal;
            u.uBaseHue.value = (hairM ? LOOK.hairHueShift : irisM ? LOOK.eyeHueShift : 0) / 360;
            u.uBaseTint.value.set(LOOK.hairColor);
            u.uBaseTintMix.value = hairM ? LOOK.hairColorMix : 0;
          }
          u.uShadeCoord.value = LOOK.shadeCoord;
          u.uShadeHue.value = LOOK.shadeHue;
          {
            // Skin: pull the shade transform toward neutral by skinShade so cast
            // shadows on skin (skirt->thigh, collar->chest) stay soft pink
            // instead of saturated magenta slabs.
            const st = isSkin ? LOOK.skinShade : 1;
            u.uShadeHueMix.value = LOOK.shadeHueMix * st;
            u.uShadeSat.value = 0.5 + (LOOK.shadeSat - 0.5) * st;
            u.uShadeVal.value = 0.5 + (LOOK.shadeVal - 0.5) * st;
          }
          u.uShadowHard.value = LOOK.shadowHard;
          u.uHiCoord.value = LOOK.hiCoord;
          u.uHiThresh.value = LOOK.hiThresh;
          u.uHiHue.value = LOOK.hiHue;
          u.uHiHueMix.value = LOOK.hiHueMix;
          u.uHiSat.value = LOOK.hiSat;
          u.uHiVal.value = LOOK.hiVal;
          u.uSubCoord.value = LOOK.subCoord;
          u.uSubThresh.value = LOOK.subThresh;
          u.uSubSat.value = LOOK.subSat;
          u.uSubVal.value = LOOK.subVal;
          u.uShadowFt.value = LOOK.shadowFt;
          u.uSssInt.value = LOOK.sssInt * (skinM ? 0.7 : 0.5); // 脚に帯が出ない程度に
          u.uSssWidth.value = LOOK.sssWidth;
          u.uForceHi.value = forceHiSet.has(matName) ? 1 : 0;
          u.uPointInt.value = LOOK.pointInt;
          u.uPointRange.value = LOOK.pointRange;
          // Hand-painted mask/pure textures (loaded into userData at model load).
          const maskOn = LOOK.maskUse >= 0.5;
          u.uMaskOn.value = maskOn && m.userData._maskTex ? 1 : 0;
          if (m.userData._maskTex) u.uMaskMap.value = m.userData._maskTex;
          u.uMaskSubOn.value = maskOn && m.userData._maskSubTex ? 1 : 0;
          if (m.userData._maskSubTex) u.uMaskSubMap.value = m.userData._maskSubTex;
          if (m.userData._pureTex && "map" in m) {
            if (!m.userData._mapBase) m.userData._mapBase = m.map;
            const want = LOOK.pureTex >= 0.5 ? m.userData._pureTex : m.userData._mapBase;
            if (m.map !== want) m.map = want;
          }
          subUList.push(u);
          if (m.userData._printPanel) {
            // Print panels: shade stays the authored-luma grey set at load —
            // 色トレス would re-derive a hue from the albedo and hand the grade
            // its amplification seed back. Sublight/highlight/fresnel keep
            // their normal behaviour (measured non-contributors).
            u.uShadeCoord.value = 0;
            u.uShadeHueMix.value = 0;
          }
          if (eyeM || mouthM) {
            // 白目/瞳/HL/口内: keep them completely out of the msTC recolor — no
            // shade derivation, no sublight replacement, no highlight, no fresnel.
            u.uShadeCoord.value = 0;
            u.uShadeHueMix.value = 0;
            u.uShadeSat.value = 0.5;
            u.uShadeVal.value = 0.5;
            u.uSubCoord.value = 0;
            u.uSubInt.value = 0;
            u.uFresInt.value = 0;
            u.uHiCoord.value = 0;
          }
        });
      });
    };

    // REVIVE THE DEAD STRANDS. The original PMX runs the twintails as FOUR parallel
    // physics chains per side — HZ (11 heavy spine bodies) + HA/HB/HC (~20-22 light
    // strand bodies each, linDamp 0.01 = very lively in MMD). The PMX->VRM conversion
    // only exported springbones for HZ; the HA/HB/HC bones still exist in the scene and
    // ~2/3 OF THE TWINTAIL MESH IS SKINNED TO THEM, but nothing ever moved them — the
    // single biggest reason the hair looked rigid ("固すぎる"). Create springbone joints
    // for those chains at runtime; applyHairSprings styles them like the HZ chain.
    const reviveStrandSprings = (v: VRM) => {
      const sbm = (v as unknown as { springBoneManager?: any })?.springBoneManager;
      if (!sbm?.joints || typeof sbm.addJoint !== "function") return;
      const existing = new Set<string>();
      for (const j of sbm.joints) existing.add(j.bone?.name || "");
      if (existing.has("HA1R")) return; // already revived (character re-init)
      const chains = new Map<string, { idx: number; bone: THREE.Object3D }[]>();
      v.scene.traverse((o: THREE.Object3D) => {
        const m = (o.name || "").match(/^H([ABC])(\d+)([RL])$/);
        if (!m) return;
        const key = m[1] + m[3];
        if (!chains.has(key)) chains.set(key, []);
        chains.get(key)!.push({ idx: parseInt(m[2], 10), bone: o });
      });
      v.scene.updateMatrixWorld(true);
      let added = 0;
      for (const arr of chains.values()) {
        arr.sort((a, b) => a.idx - b.idx);
        for (let i = 0; i < arr.length; i += 1) {
          if (existing.has(arr[i].bone.name)) continue;
          const joint = new VRMSpringBoneJoint(arr[i].bone, i + 1 < arr.length ? arr[i + 1].bone : null, {
            hitRadius: 0.02,
            stiffness: 1,
            gravityPower: 0,
            gravityDir: new THREE.Vector3(0, -1, 0),
            dragForce: 0.4,
          });
          sbm.addJoint(joint);
          joint.setInitState();
          added += 1;
        }
      }
      if (added) console.info(`[vrm] revived ${added} twintail strand spring joints (HA/HB/HC)`);
    };

    // --- Spring-bone colliders --------------------------------------------------------
    // The export dropped MMD's Bullet rigid bodies along with the springbone `center`, so
    // this VRM shipped with ZERO colliders: hair and skirt pass straight through the body.
    // That got worse once the legs actually started articulating (the append-chain fix) —
    // thighs now swing into the skirt. These 22 shapes are the STATIC rigid bodies read
    // straight out of the source PMX (YYB Hatsune Miku_10th_v1.02), so they are the
    // author's own collision volumes, not a guess. PMX units; scaled at runtime.
    // b = bone (PMX 右X/左X renamed to the VRM's XR/XL), s = shape, r = radius (for a box:
    // half-width), h = capsule cylinder length (box: half-height), p = world centre,
    // e = euler XYZ, t = box half-thickness.
    // MMD is LEFT-handed and the glTF export negates Z — verified against the rig, e.g.
    // 上半身 sits at z=-0.054 in the PMX and +0.054 in the VRM. Positions AND capsule axes
    // are mirrored below; getting this wrong put every volume the wrong side of the body.
    const PMX_COLLIDERS: { b: string; s: "c" | "s" | "b"; r: number; h: number; p: number[]; e: number[]; t?: number; g?: string }[] = [
      { b: "上半身", s: "c", r: 0.9, h: 1.091, p: [0, 13.8528, -0.652], e: [0.044, 0, 0] },
      { b: "上半身2", s: "c", r: 1.2, h: 1, p: [0, 15.6185, -0.5301], e: [0.2091, 0, 0] },
      { b: "下半身", s: "c", r: 1.3, h: 0.9768, p: [0, 12.7608, -0.3231], e: [-0.009, -0, 1.5708] },
      { b: "首", s: "c", r: 0.3715, h: 0.8576, p: [-0, 17.2681, -0.1142], e: [-0.0099, 0, 0] },
      { b: "頭", s: "s", r: 0.8, h: 0, p: [-0, 18.8379, -0.1184], e: [-0, 0, 0] },
      { b: "肩R", s: "s", r: 0.6, h: 0, p: [-0.6642, 16.4495, -0.1414], e: [-0.0034, 0.0133, -1.3202] },
      { b: "腕R", s: "c", r: 0.5486, h: 2.7428, p: [-2.1827, 15.5063, -0.1188], e: [-0.0073, 0.0096, -0.9177] },
      { b: "ひじR", s: "c", r: 0.4845, h: 2.4225, p: [-4.2455, 13.9529, -0.1241], e: [0.0107, -0.0145, -0.9342] },
      { b: "手首R", s: "s", r: 0.8, h: 0, p: [-5.8824, 12.7726, -0.1341], e: [-0.0314, 0.0351, -0.8408] },
      { b: "手首L", s: "s", r: 0.8, h: 0, p: [5.8824, 12.7726, -0.1341], e: [-0.0314, -0.0351, 0.8408] },
      { b: "肩L", s: "s", r: 0.6, h: 0, p: [0.6642, 16.4495, -0.1414], e: [-0.0034, -0.0133, 1.3202] },
      { b: "腕L", s: "c", r: 0.5486, h: 2.7428, p: [2.1827, 15.5063, -0.1188], e: [-0.0073, -0.0096, 0.9177] },
      { b: "ひじL", s: "c", r: 0.4845, h: 2.4225, p: [4.2455, 13.9529, -0.1241], e: [0.0107, 0.0145, 0.9342] },
      { b: "足R", s: "c", r: 0.86, h: 2.8, p: [-0.7802, 9.3533, -0.2312], e: [-0.0312, -0.0015, 0.0485] },
      { b: "ひざR", s: "c", r: 0.85, h: 5.4804, p: [-0.6476, 4.2212, 0.1281], e: [-0.0511, -0.0003, 0.006] },
      { b: "足首R", s: "c", r: 0.5614, h: 1.8071, p: [-0.6313, 1.0947, -0.6915], e: [1.1245, 0, 0] },
      { b: "足L", s: "c", r: 0.86, h: 2.8, p: [0.7802, 9.3533, -0.2312], e: [-0.0312, 0.0015, -0.0485] },
      { b: "ひざL", s: "c", r: 0.85, h: 5.4804, p: [0.6476, 4.2212, 0.1281], e: [-0.0511, 0.0003, -0.006] },
      { b: "足首L", s: "c", r: 0.5614, h: 1.8071, p: [0.6313, 1.0947, -0.6915], e: [1.1245, 0, 0] },
      { b: "HZ1R", s: "c", r: 0.5763, h: 0.3, p: [-1.0022, 18.8894, 0.6495], e: [0.0647, 0.3161, 1.3757] },
      { b: "HZ1L", s: "c", r: 0.5763, h: 0.3, p: [1.0022, 18.8894, 0.6495], e: [0.0647, -0.3161, -1.3757] },
      { b: "下半身", s: "s", r: 2.6, h: 0, p: [0, 9.6617, -0.0654], e: [-0.009, 0, 0] }, // 下半身辅: the hip/skirt volume
      // Dx: the necktie's backing plate — the ONE box in the model, and the reason the tie
      // doesn't swing into the chest. VRM has no box shape, so it is tiled below with rows
      // of thin capsules (exact thickness, no gaps) instead of being dropped.
      { b: "上半身2", s: "b", r: 0.5, h: 0.5, t: 0.1, p: [0, 15.9883, -1.2198], e: [0.2213, 0, 0], g: "Dx" },
    ];
    // Which colliders each cloth family should hit. Deliberately NOT everything-hits-
    // everything: a chain whose ROOT sits inside a collider gets ejected by it (bangs
    // would lift off the forehead), so each family only gets the volumes it actually
    // sinks into. PMX group masks encode the same idea; this is the readable version.
    // DECODED FROM THE PMX, not guessed. Each rigid body carries a 16-bit 非衝突グループ
    // mask where a CLEAR bit means "collides with that group". Cross-checked on the necktie:
    // group 7 / mask 0xBF7E clears bits 0, 7, 14 = body, itself, wrists — exactly what a tie
    // should hit. Static shapes live in group 0 (body), 4/6 (hair roots) and 14 (wrists).
    //
    // The headline: スカート is group 9 / mask 0xFDAF -> groups 4, 6, 9 only, so the author
    // gave the skirt NO body collision whatsoever. An earlier guess here attached 下半身
    // (including the 20.8cm 下半身辅 sphere, which CONTAINS the skirt's upper rings), both
    // thighs and both shins — five volumes the author excluded — so the skirt was being
    // shoved outward every frame. That was the "outer skirt physics is bugged" report.
    const BODY0 = ["上半身", "上半身2", "下半身", "首", "頭", "肩R", "肩L", "腕R", "腕L",
      "ひじR", "ひじL", "足R", "足L", "ひざR", "ひざL", "足首R", "足首L", "Dx"];
    const HANDS = ["手首R", "手首L"];
    const COLLIDE_FOR: Record<string, string[]> = {
      skirt: ["HZ1R", "HZ1L"],
      strandR: ["HZ1R"],                            // HA/HB/HC right: own hair root only
      strandL: ["HZ1L"],
      hzR: ["HZ1R", ...HANDS],                      // twintail spine: hair root + both hands
      hzL: ["HZ1L", ...HANDS],
      bangs: [...BODY0, "HZ1R", "HZ1L", ...HANDS],  // 前髪 0xBFAE -> 0, 4, 6, 14
      tieKnot: [...BODY0, ...HANDS],                // D1  0xBF7E -> 0, 7, 14
      tieUpper: [...HANDS],                         // D2  0xBF7F -> 7, 14
      tieRest: [],                                  // D3..D8 0xFF7F -> itself only
      breast: [...BODY0, ...HANDS],                 // 胸 mask 0x0000 -> everything
    };
    const applyBodyColliders = (v: VRM) => {
      const sbm = (v as unknown as { springBoneManager?: any })?.springBoneManager;
      if (!sbm?.joints) return;
      const byName: Record<string, THREE.Object3D> = {};
      v.scene.traverse((o: THREE.Object3D) => { if (o.name) byName[o.name] = o; });
      // Re-entry guard, same reason reviveStrandSprings has one: a character re-init would
      // otherwise stack a second full set of 22 collider volumes onto the bones every time.
      if (byName["col_頭_s0.8"]) return;
      v.scene.updateMatrixWorld(true);
      // PMX -> metres, measured off the model itself rather than assuming the usual 1/12.5
      const head = byName["頭"]; const lower = byName["下半身"];
      if (!head || !lower) return;
      const a = new THREE.Vector3(); const b2 = new THREE.Vector3();
      head.getWorldPosition(a); lower.getWorldPosition(b2);
      const scale = a.distanceTo(b2) / 4.4825; // same pair measured in the PMX
      if (!isFinite(scale) || scale <= 0) return;
      const groups = new Map<string, any>();
      const inv = new THREE.Matrix4(); const wp = new THREE.Vector3();
      const axis = new THREE.Vector3(); const eul = new THREE.Euler();
      const rotInv = new THREE.Matrix4();
      const up = new THREE.Vector3(); const side = new THREE.Vector3();
      for (const c of PMX_COLLIDERS) {
        const bone = byName[c.b];
        if (!bone) continue;
        const node = new THREE.Object3D();
        node.name = `col_${c.b}_${c.s}${c.r}`;
        bone.add(node);
        bone.updateWorldMatrix(true, false);
        inv.copy(bone.matrixWorld).invert();
        rotInv.extractRotation(inv);
        // mirrorZ: MMD -> glTF. Applies to the centre and to every direction below.
        wp.set(c.p[0], c.p[1], -c.p[2]).multiplyScalar(scale).applyMatrix4(inv);
        const radius = c.r * scale;
        eul.set(c.e[0], c.e[1], c.e[2], "XYZ");
        const dir = (x: number, y: number, z: number, out: THREE.Vector3) => {
          out.set(x, y, z).applyEuler(eul);
          out.z = -out.z; // same mirror as the position
          return out.applyMatrix4(rotInv);
        };
        const mk = (shape: any) => {
          const collider = new VRMSpringBoneCollider(shape);
          node.add(collider);
          const key = c.g ?? c.b;
          const prev = groups.get(key);
          if (prev) prev.colliders.push(collider);
          else groups.set(key, { name: key, colliders: [collider] });
        };
        if (c.s === "s") {
          mk(new VRMSpringBoneColliderShapeSphere({ offset: wp.clone(), radius }));
        } else if (c.s === "c") {
          // MMD capsule: centre + cylinder length h along the shape's local Y. VRM wants
          // the two end SPHERE CENTRES, which are exactly centre +/- axis * h/2.
          dir(0, 1, 0, axis).multiplyScalar((c.h * scale) / 2);
          mk(new VRMSpringBoneColliderShapeCapsule({
            offset: wp.clone().sub(axis), tail: wp.clone().add(axis), radius,
          }));
        } else {
          // BOX -> rows of thin capsules. VRM has no box shape, and a sphere grid would
          // have to be as thick as its spacing (a 1.6cm plate becoming ~4cm). Rows of
          // capsules whose radius IS the box's half-thickness tile the face exactly:
          // n = halfHeight / halfThickness rows, each spanning the full width.
          const t = (c.t ?? 0.05) * scale;      // half thickness -> capsule radius
          const halfW = c.r * scale;            // half width  (along the plate's local X)
          const halfH = c.h * scale;            // half height (along the plate's local Y)
          const rows = Math.max(1, Math.round(halfH / t));
          dir(1, 0, 0, side).multiplyScalar(halfW);
          dir(0, 1, 0, up);
          for (let i = 0; i < rows; i += 1) {
            const off = wp.clone().addScaledVector(up, -halfH + t + 2 * t * i);
            mk(new VRMSpringBoneColliderShapeCapsule({
              offset: off.clone().sub(side), tail: off.clone().add(side), radius: t,
            }));
          }
        }
      }
      const pick = (names: string[]) => names.map((n) => groups.get(n)).filter(Boolean);
      const sets: Record<string, any[]> = {};
      for (const k of Object.keys(COLLIDE_FOR)) sets[k] = pick(COLLIDE_FOR[k]);
      // Which family does this spring joint belong to? A joint must never receive the
      // collider that sits at its OWN origin (it would be ejected by it), which is why the
      // HZ chain root takes the hands only.
      const familyOf = (nm: string): string | null => {
        if (nm.startsWith("スカート")) return "skirt";
        if (nm.startsWith("前髪")) return "bangs";
        if (/^胸[RL]$/.test(nm)) return "breast";
        if (nm === "D1") return "tieKnot";
        if (nm === "D2") return "tieUpper";
        if (/^D[3-8]$/.test(nm)) return "tieRest";
        const hz = nm.match(/^HZ(\d+)([RL])$/);
        if (hz) return parseInt(hz[1], 10) === 1 ? null : (hz[2] === "R" ? "hzR" : "hzL");
        const st = nm.match(/^H[ABC]\d+([RL])$/);
        if (st) return st[1] === "R" ? "strandR" : "strandL";
        return null;
      };
      const HAND_ONLY = pick(HANDS);
      let wired = 0;
      for (const j of sbm.joints) {
        const nm: string = j.bone?.name || "";
        // HZ1 is the chain root and its own capsule is bound to it — hands only.
        const set = /^HZ1[RL]$/.test(nm) ? HAND_ONLY : (() => {
          const f = familyOf(nm);
          return f ? sets[f] : null;
        })();
        if (!set || !set.length) continue;
        j.colliderGroups = set;
        wired += 1;
      }
      console.info(`[vrm] ported ${groups.size} PMX collider volumes onto ${wired} spring joints`);
      (window as any).__vrmColliders = (on: boolean) => {
        for (const j of sbm.joints) {
          const nm: string = j.bone?.name || "";
          const set = /^HZ1[RL]$/.test(nm) ? HAND_ONLY : (() => {
            const f = familyOf(nm);
            return f ? sets[f] : null;
          })();
          if (!set) continue;
          j.colliderGroups = on ? set : [];
        }
        return on;
      };
    };

    // --- Gaze -----------------------------------------------------------------------
    // 目L/目R carry 400 skin weight each and ARE mapped to the VRM humanoid, yet nothing
    // ever moved them: the export handed lookAt a VRMLookAtExpressionApplier (blendshape
    // driven) and the model has no lookAt blendshapes, so her eyes were locked dead ahead
    // for the whole session. Swap in the BONE applier and give it a target that mostly
    // holds eye contact with the camera, with human-timed glances away and back.
    // Live: __vrmGaze({enabled:false}) or ({wander:0.3, hold:0.6}).
    const GAZE = {
      enabled: true,
      // Held eye contact with whoever is looking at the screen: no glances away,
      // and only micro-saccades around the camera so the eyes never look frozen.
      // (The camera follows the viewer's head when off-axis tracking is live,
      // so this is the viewer's actual eye position.)
      hold: 1.1,      // s: minimum time on one fixation
      holdVar: 2.6,   // s: extra random dwell
      wander: 0.03,   // m of target offset for a small shift of attention
      away: 0.55,     // m for a real glance away
      awayChance: 0,
      speed: 0.4,     // per 60Hz frame — saccades are FAST, a slow drift reads as dazed
      // How hard the eyes hold the camera while the HEAD is moving. lookAt is a stabiliser:
      // it keeps the eyes on the target, so a gesture that swings the head 30 deg swings the
      // eyes ~14 deg the other way (measured: eye rotation mean 9.2 deg while speaking vs
      // 5.0 silent). That over-compensation is what reads as the eyes doing something odd
      // mid-sentence. Blending the target toward wherever the head is already pointing lets
      // the eyes travel WITH the head. 1 = lock on the camera, 0 = always centred in the head.
      // Raised from 0.55 for held eye contact: the eyes stay on the viewer
      // through gestures, at the cost of some counter-rotation mid-sentence.
      stabilise: 0.9,
    };
    const gazeTarget = new THREE.Object3D();

    // --- Attention: noticing someone, and looking away to think ----------------
    // What people do when they notice someone: the eyes get there first, the head
    // follows, then a small nod (and, for someone who just walked up, a light
    // smile). And while working out a reply they look away, up and to one side,
    // and come back as they start to speak, rather than staring through the whole
    // turn. Live: __attend({ ... }); __notice("new" | "switch") to see one.
    const ATTEND = {
      nodDeg: 8,         // for someone new; a switch between people gets 0.6x
      nodDelay: 0.35,    // s after noticing — the head is mostly round by then
      nodDur: 0.7,       // s, down and back up
      smileS: 1.6,       // s of a light smile for someone new (never mid-turn)
      cooldown: 5,       // s between reactions, so detection flicker cannot spam nods
      maxDistM: 3.5,     // further away is passing by, not arriving
      avertYawDeg: 25,   // where her eyes go while she thinks
      avertPitchDeg: 8,
      avertHead: 0.4,    // share of that the head joins in
      avertBackMin: 0.5, // s into her speech before she looks back
      avertBackMax: 1.1,
      avertMax: 4,       // s: a long think still comes back
      // A listener's nod when the visitor pauses or finishes a sentence (server
      // "listening-nod"): the first thing they see after they stop is that she
      // took it in, ~0.3-0.5 s before her voice. 0 turns it off.
      listenDeg: 4,
      listenDur: 0.45,   // s, down and back up
    };
    let focusSeen = headTracker.focusEvent().seq;
    let noticeAt = -1e9;
    let noticeNew = false;
    let smiled = false;
    let turnLive = false; // a reply is under way (kehai-turn events)
    let speakSeen = false;
    let avertUntil = 0;
    let avertSide = 1;
    let listenAt = -1e9;
    const onListen = () => { listenAt = aiDebug.now() / 1000; };
    window.addEventListener("kehai-listen", onListen);
    const onTurn = (e: Event) => {
      const kind = (e as CustomEvent).detail;
      const t = aiDebug.now() / 1000;
      if (kind === "think") {
        turnLive = true;
        speakSeen = false;
        avertSide = aiDebug.random() < 0.5 ? -1 : 1;
        avertUntil = t + ATTEND.avertMax;
      } else if (kind === "speak") {
        turnLive = true;
        if (!speakSeen) { // her first sentence: look back shortly after it starts
          speakSeen = true;
          avertUntil = Math.min(
            avertUntil,
            t + ATTEND.avertBackMin + aiDebug.random() * (ATTEND.avertBackMax - ATTEND.avertBackMin),
          );
        }
      } else if (kind === "end") {
        turnLive = false;
        avertUntil = 0;
      }
    };
    window.addEventListener("kehai-turn", onTurn);
    const notice = (isNew: boolean, t: number) => {
      noticeAt = t;
      noticeNew = isNew;
      if (isNew && !turnLive) {
        vrmExpression.playEmotion("joy_smile");
        smiled = true;
      }
    };
    (window as any).__attend = (patch?: Partial<typeof ATTEND>) => {
      if (patch) Object.assign(ATTEND, patch);
      return { ...ATTEND, turnLive, averting: aiDebug.now() / 1000 < avertUntil };
    };
    (window as any).__notice = (kind: "new" | "switch" = "new") => {
      notice(kind === "new", aiDebug.now() / 1000);
      return `noticed (${kind})`;
    };

    // Where the EYES aim (attendPoint) and where the HEAD turns (headPoint): the
    // camera, or — while the PROTO's camera tracks someone — that person's head,
    // blended by the camera follow's hand-over; plus the thinking look-away.
    const attendPoint = new THREE.Vector3();
    const headPoint = new THREE.Vector3();
    const attendViewer = new THREE.Vector3();
    const attendHead = new THREE.Vector3();
    const avertOff = new THREE.Vector3();
    /** Per frame, before the head turn and the gaze read the points. Returns the nod (rad). */
    const updateAttend = (v: VRM): number => {
      const t = aiDebug.now() / 1000;
      const fe = headTracker.focusEvent();
      if (fe.seq !== focusSeen) {
        focusSeen = fe.seq;
        if (t - noticeAt > ATTEND.cooldown && fe.z <= ATTEND.maxDistM) notice(fe.kind === "new", t);
      }
      if (smiled && (turnLive || t - noticeAt > ATTEND.smileS)) {
        if (!turnLive) vrmExpression.relax(); // a reply's own emotion is left alone
        smiled = false;
      }
      // Eyes jump to the person the moment she notices them; the head only
      // comes round with the follow hand-over (~0.8 s) and the turn's easing.
      const eyesW = t - noticeAt < 1 && viewerTransform.isTracking() ? 1 : headFollow.weight;
      attendPoint.copy(camera.position);
      if (eyesW > 0) attendPoint.lerp(viewerWorld(attendViewer), eyesW);
      headPoint.copy(camera.position);
      if (headFollow.weight > 0) headPoint.lerp(viewerWorld(attendViewer), headFollow.weight);
      if (t < avertUntil) {
        const head = v.humanoid?.getNormalizedBoneNode("head");
        const d = head ? head.getWorldPosition(attendHead).distanceTo(attendPoint) : 2;
        avertOff.set(
          avertSide * Math.tan(THREE.MathUtils.degToRad(ATTEND.avertYawDeg)) * d,
          Math.tan(THREE.MathUtils.degToRad(ATTEND.avertPitchDeg)) * d,
          0,
        );
        attendPoint.add(avertOff);
        headPoint.addScaledVector(avertOff, ATTEND.avertHead);
      }
      // The nods: one smooth dip over its window, and not while she is mid-reply.
      if (turnLive) return 0;
      const nt = (t - noticeAt - ATTEND.nodDelay) / ATTEND.nodDur;
      const noticeNod = nt <= 0 || nt >= 1 ? 0
        : THREE.MathUtils.degToRad(ATTEND.nodDeg * (noticeNew ? 1 : 0.6)) * Math.sin(Math.PI * nt);
      const lt = (t - listenAt) / ATTEND.listenDur;
      const listenNod = lt <= 0 || lt >= 1 ? 0
        : THREE.MathUtils.degToRad(ATTEND.listenDeg) * Math.sin(Math.PI * lt);
      return Math.max(noticeNod, listenNod);
    };
    const gazeWant = new THREE.Vector3();
    const gazeTmp = new THREE.Vector3();
    const gazeHead = new THREE.Vector3();
    const gazeFwd = new THREE.Vector3();
    const gazeQ = new THREE.Quaternion();
    let gazeNextAt = 0;
    const setupGaze = (v: VRM) => {
      if (!v.lookAt || !v.humanoid) return;
      // A glb rig arrives with placeholder ranges (see glb-humanoid), so it is
      // swapped too.
      const glbRig = (v.meta as { name?: string } | undefined)?.name === "glb-humanoid";
      if (glbRig || !(v.lookAt.applier instanceof VRMLookAtBoneApplier)) {
        // outputScale is in DEGREES of eye rotation at inputMaxValue. Kept modest: real
        // eyes only travel ~15 deg before the head takes over, and more looks googly.
        v.lookAt.applier = new VRMLookAtBoneApplier(
          v.humanoid,
          new VRMLookAtRangeMap(45, 14), // horizontal, toward the nose
          new VRMLookAtRangeMap(45, 14), // horizontal, toward the temple
          new VRMLookAtRangeMap(35, 9),  // vertical down
          new VRMLookAtRangeMap(35, 9),  // vertical up
        );
        console.info("[vrm] gaze: swapped the lookAt expression applier for the bone applier");
      }
      if (!gazeTarget.parent) scene.add(gazeTarget);
      gazeTarget.position.copy(camera.position);
      v.lookAt.target = gazeTarget;
      v.lookAt.autoUpdate = true;
      (window as any).__vrmGaze = (o: Partial<typeof GAZE>) => Object.assign(GAZE, o);
      // Show/hide one material by name, so a hidden part can be checked against the model
      // (and any OTHER stray part found the same way). Returns what it matched.
      (window as any).__vrmHide = (name: string, hidden = true) => {
        const hit: string[] = [];
        v.scene.traverse((o: THREE.Object3D) => {
          const m = o as THREE.Mesh;
          if (!m.isMesh) return;
          const m0 = (Array.isArray(m.material) ? m.material[0] : m.material) as THREE.Material | undefined;
          if (m0?.name === name) { m.visible = !hidden; hit.push(m0.name); }
        });
        return hit;
      };
    };
    const updateGaze = (dt: number, v: VRM | null) => {
      if (!v?.lookAt) return;
      if (!GAZE.enabled) { v.lookAt.target = null; return; }
      if (v.lookAt.target !== gazeTarget) v.lookAt.target = gazeTarget;
      const now = aiDebug.now();
      if (now >= gazeNextAt) {
        gazeNextAt = now + (GAZE.hold + aiDebug.random() * GAZE.holdVar) * 1000;
        const r = aiDebug.random() < GAZE.awayChance ? GAZE.away : GAZE.wander;
        // vertical range is narrower: people scan sideways far more than up/down
        gazeWant.set((aiDebug.random() * 2 - 1) * r, (aiDebug.random() * 2 - 1) * r * 0.55, 0);
      }
      gazeTmp.copy(attendPoint).add(gazeWant);
      if (GAZE.stabilise < 0.999) {
        const head = v.humanoid?.getNormalizedBoneNode("head");
        if (head) {
          head.getWorldPosition(gazeHead);
          gazeFwd.set(0, 0, 1).applyQuaternion(head.getWorldQuaternion(gazeQ))
            .multiplyScalar(gazeHead.distanceTo(attendPoint))
            .add(gazeHead);
          gazeTmp.lerp(gazeFwd, 1 - GAZE.stabilise); // pull the target toward the head's own aim
        }
      }
      gazeTarget.position.lerp(gazeTmp, 1 - Math.pow(1 - GAZE.speed, Math.max(1, dt * 60)));
    };

    // Cloth springs the export left behind. Same story as the HA/HB/HC strands above:
    // the PMX drives these with Bullet rigid bodies, the glTF kept the bones and weights but
    // not the physics. Cross-checked against the PMX body list, so only chains the author
    // ACTUALLY simulated get springs — the breast bones (胸L/R) and the necktie (D1..D8).
    // Live retune: __vrmCloth({breastStiff: ...}).
    const CLOTH = {
      // stiffness 1.6 was so tight the chest bones still only moved 0.8 deg through a whole
      // dance — a spring that can't be seen isn't worth the joint. 0.8 gives visible
      // follow-through without wobble (drag keeps it settling).
      breastStiff: 0.8, breastDrag: 0.5, breastGrav: 0.04,
      // Necktie (D1..D8, 258 skin weight): the last MMD-dynamic chain with no VRM spring at
      // all, so the tie was a rigid board glued to the chest. PMX gives it decreasing mass
      // down the chain (1.10 -> 0.10), so soften the stiffness toward the tip to match.
    };
    const reviveClothSprings = (v: VRM) => {
      const sbm = (v as unknown as { springBoneManager?: any })?.springBoneManager;
      if (!sbm?.joints || typeof sbm.addJoint !== "function") return;
      const existing = new Set<string>();
      for (const j of sbm.joints) existing.add(j.bone?.name || "");
      if (existing.has("胸L")) return; // already revived
      const rawHips = (v as any).humanoid?.getRawBoneNode?.("hips") ?? null;
      const byName: Record<string, THREE.Object3D> = {};
      v.scene.traverse((o: THREE.Object3D) => { if (o.name) byName[o.name] = o; });
      v.scene.updateMatrixWorld(true);
      // MMD marks a chain's visual end with a "…先" bone that carries no weight — prefer a
      // real (non-先) child as the spring tail, and fall back to the 先 marker.
      const pickChild = (b?: THREE.Object3D) => {
        if (!b) return null;
        const kids = b.children.filter((c) => (c as unknown as { isBone?: boolean }).isBone || c.type === "Bone");
        return kids.find((c) => !c.name.endsWith("先")) ?? kids[0] ?? null;
      };
      let added = 0;
      const add = (b: THREE.Object3D | undefined, stiffness: number, dragForce: number, gravityPower: number) => {
        if (!b || existing.has(b.name)) return;
        const joint = new VRMSpringBoneJoint(b, pickChild(b), {
          hitRadius: 0.03, stiffness, dragForce, gravityPower,
          gravityDir: new THREE.Vector3(0, -1, 0),
        });
        (joint as any).center = rawHips; // cancel body translation, still driven by the torso
        sbm.addJoint(joint);
        joint.setInitState();
        // the animate loop converts __dragTarget to a per-sub-step value; without it the
        // authored drag would be applied subN times over and the cloth could never move.
        (joint as any).__dragTarget = dragForce;
        existing.add(b.name);
        added += 1;
      };
      // NO sleeve springs. x/xR/x2 looked like a missed cloth chain (382 + 264 skin weight,
      // never rotating), but the PMX has NO rigid body for them at all — they are static
      // geometry in MMD too, riding the elbow. Adding springs there was inventing physics
      // the author never had, and it read as odd motion around the hands.
      for (const s of ["L", "R"]) {
        add(byName[`胸${s}`], CLOTH.breastStiff, CLOTH.breastDrag, CLOTH.breastGrav);
      }
      // NO necktie springs — REVERTED. The PMX really does simulate D1..D8, but it does so
      // with rigid-body joints that have ANGULAR LIMITS, and a VRM spring has only stiffness.
      // Measured: the tie's wide flat bottom drifted 4-8cm outward and 10cm below its
      // authored rest, swung between the key light and the thighs, and its binarized toon
      // shadow landed there as a moving hard-edged RECTANGLE inside the skirt — the "四角い
      // 謎の影" report. Proven by A/B: freezing the tie or disabling castShadow both remove
      // it. Even stiffness 2.6 still drifted enough, and stiff enough to stop it is rigid
      // anyway, so the tie goes back to riding the chest as the export left it.
      // Live experiment: __vrmCloth({tieStiff: n}) only affects joints that exist.
      if (added) console.info(`[vrm] revived ${added} cloth spring joints (breast 胸)`);
      (window as any).__vrmCloth = (o: Partial<typeof CLOTH>) => {
        Object.assign(CLOTH, o);
        for (const j of sbm.joints) {
          const nm: string = j.bone?.name || "";
          if (!/^胸[LR]$/.test(nm)) continue;
          j.settings.stiffness = CLOTH.breastStiff;
          j.settings.gravityPower = CLOTH.breastGrav;
          (j as any).__dragTarget = CLOTH.breastDrag;
        }
        return CLOTH;
      };
    };

    // Hair (twintail) springs: push the 🎨-panel hair physics onto the long-hair
    // (HZ*) spring joints. j.settings.* is read live by three-vrm each update(), so
    // the sliders retune the hair without a rebuild or a jarring reset.
    const applyHairSprings = (v: VRM) => {
      const sbm = (v as unknown as { springBoneManager?: any })?.springBoneManager;
      if (!sbm?.joints) return;
      // CENTER-SPACE SIMULATION. This VRM's export dropped the springbone `center` on ALL
      // 139 joints (same as the colliders). center picks the sim space = WHAT DRIVES the
      // hair: world (null) = every body translation is a gust (the old constant agitation);
      // head = NOTHING drives it (hair rides the head rigidly; with grav 0 there is zero
      // force left -> "全然物理っぽくない"). The right space is the HIPS (VRoid convention):
      // whole-body translation is cancelled, but chest/neck/head motion RELATIVE to the
      // hips remains a driver, so the soft lower chain lags ~0.5s behind = the "なびき".
      // Bangs keep center=head (forehead hair must track the head tightly, driven only by
      // gravity-vs-tilt). Toggle: __vrmSpring({center:0|1}).
      const rawHead = (v as any).humanoid?.getRawBoneNode?.("head") ?? null;
      const rawHips = (v as any).humanoid?.getRawBoneNode?.("hips") ?? null;
      const useCenter = LOOK.hairCenter !== 0;
      // per-chain max index (HZ=11, HA~20, HB/HC~22) so depth normalizes correctly
      const maxIdx = new Map<string, number>();
      for (const j of sbm.joints) {
        const m = (j.bone?.name || "").match(/^H([ZABC])(\d+)([RL])$/);
        if (m) maxIdx.set(m[1] + m[3], Math.max(maxIdx.get(m[1] + m[3]) || 1, parseInt(m[2], 10)));
      }
      for (const j of sbm.joints) {
        const nm: string = j.bone?.name || "";
        const s = j.settings;
        const hz = nm.match(/^H([ZABC])(\d+)([RL])$/);
        const wantCenter = hz || nm.startsWith("スカート")
          ? (useCenter ? rawHips : null)
          : nm.startsWith("前髪")
            ? (useCenter ? rawHead : null)
            : undefined;
        if (wantCenter !== undefined && j.center !== wantCenter) j.center = wantCenter;
        if (hz) {
          // twintails (long hair). Lag accumulates DOWN the chain, so the ENDS jitter
          // most ("下側がびくびく"). Graduate the drag UP toward the tip (higher joint
          // index) to damp the lower half without freezing the lively roots.
          // In the source PMX the twintail is FOUR chains per side: HZ = the heavy spine
          // (mass 50, linDamp 0.1), HA/HB/HC = light strands (linDamp 0.01 = very lively).
          // Mirror that: strands get lower stiffness + lighter tip drag than the spine.
          const strand = hz[1] !== "Z"; // HA/HB/HC = the revived strand chains
          const idx = parseInt(hz[2], 10); // 1 = root (at the head), max = tip (bottom)
          const last = maxIdx.get(hz[1] + hz[3]) || 11;
          const depth = Math.min(1, (idx - 1) / Math.max(1, last - 1)); // 0 root -> 1 tip
          // STIFF ROOT -> SOFT TIP (user request: "根元の位置は変えたくない、毛先は若干変わって
          // いい"). The root holds the authored silhouette rigidly (hair POSITION never
          // drifts), and stiffness falls toward the tip so the ends FLOW = the physics feel.
          // This also structurally prevents the fast プルプル: fast vibration needs a HIGH
          // natural frequency (stiff spring) — a soft tip can only sway slowly. Heavy drag
          // still ramps toward the tip so the flow settles instead of oscillating.
          const ts = LOOK.hairTipSoft;
          s.stiffness = LOOK.hairStiff * (1 - (1 - ts) * depth) * (strand ? 0.55 : 1);
          // __dragTarget is the AUTHORED-UNITS drag (per single 60Hz step, the standard VRM
          // range ~0.3-0.6). The animate loop converts it per-frame to a per-SUB-STEP value
          // (1-(1-t)^(1/subN)) — without that, N sub-steps apply the drag N times and even
          // 0.4 becomes near-total damping = the hair can never bounce like MMD.
          // Damping must MATCH stiffness (critical damping ∝ √k): a stiff joint with low
          // damping vibrates FAST (the "減衰バーを下げるとびくびく"), a soft joint with low
          // damping just bounces slowly (=なびき). So the slider only frees the SOFT tip
          // side; the stiff root stays heavily damped and physically cannot buzz.
          const dragRoot = Math.max(0.9, LOOK.hairDrag);
          const dragTip = LOOK.hairDrag * (strand ? 0.8 : 1); // strands livelier (PMX linD 0.01)
          const dragTarget = dragRoot * (1 - depth) + dragTip * depth;
          (j as any).__dragTarget = dragTarget;
          s.dragForce = dragTarget; // fallback (overwritten per-frame by the compensation)
          s.gravityPower = LOOK.hairGrav;
        } else if (nm.startsWith("前髪")) {
          // BANGS had gravityPower=0 -> they floated up / antenna'd and twitched (びくびく)
          // whenever the head moved. Give them downward gravity so they hang, plus more
          // stiffness/drag so they settle instead of jittering.
          s.stiffness = 3;
          (j as any).__dragTarget = 0.9; // heavy damping so the bangs settle smoothly (no びくびく)
          s.dragForce = 0.9;
          s.gravityPower = 0.35;
        } else if (nm.startsWith("スカート")) {
          // SKIRT: gravity 0 like the hair — the model is AUTHORED at grav 0, its rest pose
          // IS the intended flared skirt shape; any gravity drags it inward/down and it
          // "しぼむ" (deflates). The old 0.45 was a float-era hack; hips-center sim + heavy
          // drag keep it stable now. Heavy drag (no colliders — a lively skirt clips legs).
          (j as any).__dragTarget = 0.9;
          s.dragForce = 0.9;
          s.gravityPower = 0;
        } else {
          continue;
        }
        if (s.gravityDir?.set) s.gravityDir.set(0, -1, 0);
      }
    };

    // Live tuning from the devtools console — no rebuild needed.
    // Load any saved 🎨 look tuning, then define a single apply-all that both the
    // console __vrmLook and the on-screen 🎨 panel drive; persist on every change.
    // bumped to 8: the gest*/ikPelvis* sliders now tune the GESTURELSM set only, and a
    // browser that saved the brief "everything at 1.0" experiment would otherwise keep
    // feeding 1.0 back in and re-break GestureLSM.
    // PER MODEL, not global. One shared key meant tuning Miku's look also changed
    // Chancelor's — the two are different meshes, materials and proportions, so a
    // value that flatters one is wrong on the other. Keyed on the model URL's
    // basename, which is stable and already unique per preset.
    const modelKey = (url || "unknown")
      .split("?")[0]
      .split("/")
      .pop()!
      .replace(/\.[^.]+$/, "")
      .replace(/[^A-Za-z0-9_-]/g, "_");
    const LS_LOOK = `vrmLook8:${modelKey}`;
    const LS_LOOK_LEGACY = "vrmLook8";
    let migrated = false;
    try {
      // Fall back to the pre-split key ONCE, so an existing tuning is inherited by
      // whichever model is loaded first rather than silently reset to defaults.
      const s = localStorage.getItem(LS_LOOK) ?? localStorage.getItem(LS_LOOK_LEGACY);
      const saved = s ? JSON.parse(s) : null;
      if (saved && saved._build === LOOK_BUILD) {
        delete saved._build;
        // See GRADE_FIX — applied once per saved look, then stamped.
        if (saved._grade !== GRADE_REV) {
          Object.assign(saved, GRADE_FIX);
          migrated = true;
        }
        delete saved._grade;
        Object.assign(LOOK, saved);
      } else {
        // New deploy (or fresh storage): saved tweaks belong to an older build —
        // restart on the CURRENT 映像 preset so what ships is what you see.
        // (「変わってない」の再発防止: 保存値が新プリセットを永遠に上書きする罠)
        Object.assign(LOOK, LOOK_PRESETS.video);
      }
    } catch {
      /* ignore */
    }
    const saveLook = () => {
      try {
        localStorage.setItem(
          LS_LOOK,
          JSON.stringify({ ...LOOK, _build: LOOK_BUILD, _grade: GRADE_REV }),
        );
      } catch {
        /* ignore */
      }
    };
    // Persist the migration NOW rather than waiting for the next 🎨 edit, so it
    // cannot re-apply on every load and undo a deliberate later hand-tune.
    if (migrated) saveLook();
    const applyLookState = () => {
      setBgGrad(LOOK.bgInner, LOOK.bgOuter);
      // The lights were dialled in for Miku's toon shading too: dim, with a violet
      // ground bounce. On a PBR model that has no rim/subsurface pass to lift it,
      // that rig turns a white android a dull violet-grey, so plainStyle gets a
      // neutral white studio instead.
      ambient.intensity = fx(LOOK.ambient, 0.9);
      hemi.intensity = fx(LOOK.hemi, 0.6);
      hemi.color.set(fx(LOOK.hemiSky, 0xffffff));
      hemi.groundColor.set(fx(LOOK.hemiGround, 0x8a8a8a));
      keyLight.intensity = fx(LOOK.key, 1.3);
      keyLight.color.set(fx(LOOK.keyColor, 0xffffff));
      {
        // Key light direction (MMD's 照明方向): world-space azimuth/elevation.
        const kaz = (LOOK.keyAzim * Math.PI) / 180;
        const kel = (LOOK.keyElev * Math.PI) / 180;
        keyLight.position
          .set(Math.sin(kaz) * Math.cos(kel), Math.sin(kel), Math.cos(kaz) * Math.cos(kel))
          .multiplyScalar(10);
      }
      fillFront.intensity = LOOK.fillFront;
      fillLeft.intensity = LOOK.fillSide;
      fillRight.intensity = LOOK.fillSide;
      // Off for PBR models (plainStyle), like the diffusion below: on the android's
      // white shell every edge sits over the threshold, and the bloom wrapped her
      // in a white haze ("アバターの周りのもやはいらない").
      bloomPass.strength = fx(LOOK.bloomStrength, 0);
      bloomPass.radius = LOOK.bloomRadius;
      bloomPass.threshold = LOOK.bloomThreshold;
      diffPass.strength = fx(LOOK.diffusion, 0);
      diffPass.enabled = diffPass.strength > 0.001;
      gtaoPass.blendIntensity = LOOK.ssao;
      gtaoPass.enabled = LOOK.ssao > 0.001;
      linePass.uniforms.uLineInt.value = fx(LOOK.lineInt, 0);
      linePass.uniforms.uLineN.value = fx(LOOK.lineNInt, 0);
      linePass.uniforms.uLineZ.value = fx(LOOK.lineZInt, 0);
      linePass.uniforms.uLineThresh.value = LOOK.lineThresh;
      linePass.uniforms.uLineColor.value.set(LOOK.outlineColor >= 0 ? LOOK.outlineColor : 0x151030);
      linePass.enabled = linePass.uniforms.uLineInt.value
        + linePass.uniforms.uLineN.value + linePass.uniforms.uLineZ.value > 0.001;
      // Grade: neutral means "leave the colours as authored" — contrast and
      // saturation at 1, every lift/split/temperature at 0.
      gradePass.uniforms.uContrast.value = fx(LOOK.gContrast, 1);
      gradePass.uniforms.uSat.value = fx(LOOK.gSat, 1);
      gradePass.uniforms.uWarm.value.setRGB(
        fx(LOOK.gWarmR, 1), fx(LOOK.gWarmG, 1), fx(LOOK.gWarmB, 1),
      );
      gradePass.uniforms.uShadowLift.value.setRGB(
        fx(LOOK.gLiftR, 0), fx(LOOK.gLiftG, 0), fx(LOOK.gLiftB, 0),
      );
      gradePass.uniforms.uChroma.value = fx(LOOK.chroma, 0);
      gradePass.uniforms.uVignette.value = fx(LOOK.vignette, 0);
      gradePass.uniforms.uGrain.value = aiDebug.grainFor(fx(LOOK.grain, 0));
      gradePass.uniforms.uBars.value = fx(LOOK.bars, 0);
      gradePass.uniforms.uTemp.value = fx(LOOK.gTemp, 0);
      gradePass.uniforms.uTint.value = fx(LOOK.gTint, 0);
      gradePass.uniforms.uSplit.value = fx(LOOK.gSplit, 0);
      gradePass.uniforms.uSteps.value = fx(LOOK.toonSteps, 0);
      gradePass.uniforms.uGradTopC.value.set(LOOK.gradTop);
      gradePass.uniforms.uGradTopI.value = LOOK.gradTopInt;
      gradePass.uniforms.uGradBotC.value.set(LOOK.gradBot);
      gradePass.uniforms.uGradBotI.value = LOOK.gradBotInt;
      renderer.toneMapping = TONE_MODES[Math.round(LOOK.toneMode)] ?? THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = LOOK.toneExposure;
      keyLight.castShadow = LOOK.shadowStrength > 0.001;
      // msTC 落ち影2値化 needs the raw shadow to reach 0 (with intensity 0.45 the
      // shadowed value bottoms out at 0.55 and never crosses the 0.5 threshold):
      // force full intensity and let 影明度/彩度 decide the darkness instead.
      if ((keyLight.shadow as any).intensity !== undefined) {
        (keyLight.shadow as any).intensity = LOOK.shadowHard > 0.001 ? 1.0 : LOOK.shadowStrength;
      }
      // 影解像度 (0=1024, 1=2048, 2=4096 — the ExcellentShadow half msTC delegates
      // to): resizing requires dropping the allocated map so three rebuilds it.
      const shadowRes = [1024, 2048, 4096][Math.round(LOOK.shadowRes)] ?? 1024;
      if (keyLight.shadow.mapSize.x !== shadowRes) {
        keyLight.shadow.mapSize.set(shadowRes, shadowRes);
        if (keyLight.shadow.map) {
          keyLight.shadow.map.dispose();
          (keyLight.shadow as any).map = null;
        }
      }
      // The space layer may ask for a STRONGER depth-of-field than her look
      // wants, because a visibly out-of-focus background is the single clearest
      // statement that a depth axis exists at all — with everything sharp the
      // frame reads as one flat plane no matter what is drawn on it. Focus
      // tracks her (below), so raising this blurs the space behind her and
      // leaves her sharp. Her LOOK values are not edited; this only overrides
      // the live pass while the space is on.
      const spaceDof = SPACE.enabled ? SPACE.dofBackground : 0;
      const ap = Math.max(LOOK.dofAperture, spaceDof);
      bokehPass.uniforms.aperture.value = ap;
      bokehPass.uniforms.maxblur.value = Math.max(
        LOOK.dofBlur,
        spaceDof > 0 ? SPACE.dofBlur : 0,
      );
      bokehPass.enabled = ap > 0.00001;
      if (vrm) applyLook(vrm);
      if (vrm) applyHairSprings(vrm);
      // These sliders tune the GESTURELSM set only — that is the noisy source they were
      // measured against. Hand-animated KAWAII clips always play at 1.0 (player.authored),
      // so tuning here can't flatten the authored animation.
      vrmMotionPlayer.lsm.shoulder = LOOK.gestShoulder;
      vrmMotionPlayer.lsm.gesture = LOOK.gestGain;
      vrmMotionPlayer.lsm.finger = LOOK.gestFinger;
      vrmMotionPlayer.lsm.pelvisRot = LOOK.ikPelvisRot;
      vrmMotionPlayer.lsm.pelvisTrans = LOOK.ikPelvisTrans;
      (vrmMotionPlayer as any)._gainsFor = null; // re-apply on the next frame
    };
    (window as any).__vrmLook = (patch: Partial<typeof LOOK> = {}) => {
      Object.assign(LOOK, patch);
      applyLookState();
      // Only PERSIST when something was actually set. `__vrmLook()` with no args
      // is the read accessor — every tool and every console poke uses it — and it
      // used to write the saved look back out on each call. That turns any
      // transient patch into a permanent one: a measurement run that set
      // hairColorMix:1 to identify hair pixels left the character's hair stored
      // as bright red, and the same path is how a temporary 「スカート不透明」
      // A/B became the persisted state that killed the coat's transparency.
      if (Object.keys(patch).length) saveLook();
      return LOOK;
    };
    // __vrmPick() — arm a one-shot picker: the NEXT click on the canvas raycasts and logs
    // which mesh/material is under the cursor (nearest first). Naming a stray costume part
    // from the file alone is guesswork — bounding boxes, texture averages and pixel diffs all
    // fail on this model (31 materials, every colour in the texture, an animated navy
    // background, and MToon shading that survives any attempt at an ID render). One click
    // answers it. Pair with __vrmHide(name) to confirm, then add it to HIDE_MATERIALS.
    (window as any).__vrmPick = () => {
      const rc = new THREE.Raycaster();
      const onClick = (ev: MouseEvent) => {
        renderer.domElement.removeEventListener("click", onClick);
        const r = renderer.domElement.getBoundingClientRect();
        rc.setFromCamera(new THREE.Vector2(
          ((ev.clientX - r.left) / r.width) * 2 - 1,
          -((ev.clientY - r.top) / r.height) * 2 + 1,
        ), camera);
        const root = vrm?.scene;
        if (!root) { console.warn("[vrm-pick] no model"); return; }
        const out = (rc.intersectObject(root, true) as any[])
          .filter((h: any) => (h.object as THREE.Mesh).isMesh && h.object.visible)
          .slice(0, 5)
          .map((h: any) => {
            const mm = h.object as THREE.Mesh;
            const m0 = (Array.isArray(mm.material) ? mm.material[0] : mm.material) as THREE.Material | undefined;
            return `${m0?.name ?? "?"}  (mesh ${mm.name}, ${h.distance.toFixed(3)}m)`;
          });
        console.info("[vrm-pick] under the cursor, nearest first:\n  " + (out.join("\n  ") || "(nothing)"));
      };
      renderer.domElement.addEventListener("click", onClick);
      return "now click the part you want identified";
    };
    // __vrmView(targetY, distance) — snap the camera to a framed shot (debug/QA).
    // Face close-up ≈ __vrmView(1.4, 0.5). Restore with __vrmView() (defaults).
    (window as any).__vrmView = (ty = 1.0, dist = 2.4) => {
      controls.target.set(0, ty, 0);
      camera.position.set(0, ty, dist);
      camera.lookAt(0, ty, 0);
      controls.update();
      return { ty, dist };
    };
    // __vrmPreset("mstc") / __vrmPreset("base") — same as the 🎨 panel buttons.
    (window as any).__vrmPreset = (name: string = "mstc") => {
      const p = LOOK_PRESETS[name];
      if (!p) return Object.keys(LOOK_PRESETS);
      const keepSkirt = LOOK.skirtOpaque; // user choices, survive preset swaps
      const keepArm = LOOK.armOpaque;
      Object.assign(LOOK, LOOK_DEFAULTS, p);
      LOOK.skirtOpaque = keepSkirt;
      LOOK.armOpaque = keepArm;
      applyLookState();
      saveLook();
      return LOOK;
    };
    // __vrmMask("face01", "/mstc-masks/face01.png") — attach a SHADOW_RATE
    // texture to a material: R = keep base/lit, G = force highlight, B = force
    // shade, painted in the material's UV space (鼻影の三角・唇下ハイライト等).
    // __vrmMask("face01", null) removes it. Runtime only (not persisted).
    (window as any).__vrmMask = (matName: string, texUrl: string | null) => {
      if (!vrm) return "no vrm";
      const apply = (tex: THREE.Texture | null) => {
        let hit = 0;
        vrm!.scene.traverse((o: THREE.Object3D) => {
          const ms = (o as THREE.Mesh).material;
          (Array.isArray(ms) ? ms : [ms]).forEach((mt: any) => {
            if (!mt || (mt.name || "").replace(" (Outline)", "") !== matName) return;
            if (!mt.userData?._subU) return;
            mt.userData._maskTex = tex ?? undefined;
            hit += 1;
          });
        });
        if (tex) LOOK.maskUse = 1;
        applyLookState();
        console.info(`[vrm] mask ${texUrl ? "set" : "cleared"} on ${hit} material(s) named ${matName}`);
      };
      if (!texUrl) { apply(null); return "cleared"; }
      new THREE.TextureLoader().load(texUrl, (tex) => {
        tex.flipY = false; // glTF-style UVs
        apply(tex);
      });
      return "loading...";
    };
    // __vrmForceHi("Hair01") — toggle a material onto the forced-highlight list
    // (Shade_Highlight1_Absolute.fx analogue). Persisted via the saved look.
    (window as any).__vrmForceHi = (matName: string) => {
      const list = new Set(LOOK.forceHiMats.split(",").map((s) => s.trim()).filter(Boolean));
      if (list.has(matName)) list.delete(matName);
      else list.add(matName);
      LOOK.forceHiMats = Array.from(list).join(",");
      applyLookState();
      saveLook();
      return LOOK.forceHiMats || "(none)";
    };
    // __vrmObj() — raw VRM handle (debug only).
    // Mip-mapped minification on/off, live. The trade-off is real and only the
    // eye can settle it — mips kill the stair-stepping on the outfit's thin trim
    // lines, and cost a little crispness in the skirt interior at grazing
    // angles. Both states are one call away rather than a rebuild apart.
    /**
     * __vrmIdMode(1) — paint every material a unique flat colour and return the
     * name->colour table, so a single deterministic capture says which material
     * owns which pixel. __vrmIdMode(0) restores. See uIdMode in makeSubUniforms.
     *
     * IDs are spaced on a coarse lattice (6 levels per channel, black reserved)
     * so nearest-match classification survives SMAA/TAA edge blending.
     */
    (window as any).__vrmIdMode = (on: boolean | number = 1) => {
      const want = !!on;
      const table: { name: string; rgb: number[] }[] = [];
      let i = 0;
      vrm?.scene.traverse((obj: THREE.Object3D) => {
        const mesh = obj as THREE.Mesh;
        if (!mesh.isMesh) return;
        const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        mats.forEach((mat: THREE.Material) => {
          const m = mat as any;
          const u = m?.userData?._subU;
          if (!u?.uIdMode) return;
          i += 1;
          const r = ((i % 6) + 1) / 7;
          const g = ((Math.floor(i / 6) % 6) + 1) / 7;
          const b = ((Math.floor(i / 36) % 6) + 1) / 7;
          u.uIdMode.value = want ? 1 : 0;
          if (want) {
            u.uMatId.value.setRGB(r, g, b);
            table.push({
              name: m.name,
              rgb: [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)],
            });
          }
          m.uniformsNeedUpdate = true;
        });
      });
      return { idMode: want, materials: table.length, table };
    };
    (window as any).__vrmMips = (on: boolean | number = 1) => {
      const want = !!on;
      let n = 0;
      // Alpha-shaped textures are excluded here for the same reason the loader
      // excludes them — mipping them dissolves the hair. Recomputed rather than
      // shared so this toggle is honest on its own: turning mips back ON must
      // not be a way to reintroduce the bug.
      const keys = ["map", "emissiveTexture", "shadeMultiplyTexture", "matcapTexture"];
      const skip = new Set<unknown>();
      vrm?.scene.traverse((obj: THREE.Object3D) => {
        const mesh = obj as THREE.Mesh;
        if (!mesh.isMesh) return;
        const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        mats.forEach((mat: THREE.Material) => {
          const m = mat as any;
          if (!m?.transparent) return;
          keys.forEach((k) => {
            const tex = m[k] || m.uniforms?.[k]?.value;
            if (tex) skip.add(tex);
          });
        });
      });
      vrm?.scene.traverse((obj: THREE.Object3D) => {
        const mesh = obj as THREE.Mesh;
        if (!mesh.isMesh) return;
        const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        mats.forEach((mat: THREE.Material) => {
          const m = mat as any;
          keys.forEach((k) => {
            const tex = m[k] || m.uniforms?.[k]?.value;
            if (!tex || !("minFilter" in tex) || skip.has(tex)) return;
            const next = want ? THREE.LinearMipmapLinearFilter : THREE.LinearFilter;
            if (tex.minFilter === next) return;
            tex.minFilter = next;
            tex.generateMipmaps = want;
            tex.needsUpdate = true; // re-uploads and (re)builds the mip chain
            n += 1;
          });
        });
      });
      return { mips: want, texturesChanged: n };
    };
    (window as any).__vrmObj = () => vrm;
    // __vrmBones("腕") — list skeleton bone names (debug).
    (window as any).__vrmBones = (pat: string = "") => {
      if (!vrm) return "no vrm";
      const names = new Set<string>();
      vrm.scene.traverse((o: any) => {
        if (o.isSkinnedMesh && o.skeleton) {
          o.skeleton.bones.forEach((b: any) => names.add(b.name));
        }
      });
      return Array.from(names).filter(
        (n) => !pat || n.toLowerCase().includes(pat.toLowerCase()),
      );
    };
    // __vrmMats() — dump every material: transparency flags + skirt skinning.
    (window as any).__vrmMats = () => {
      if (!vrm) return "no vrm";
      const rows: any[] = [];
      vrm.scene.traverse((o: THREE.Object3D) => {
        const mesh = o as THREE.Mesh;
        if (!mesh.isMesh) return;
        const ms = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        ms.forEach((mt: any) => {
          if (!mt) return;
          rows.push({
            mat: mt.name,
            transparent: !!mt.transparent,
            opacity: mt.opacity,
            depthWrite: !!mt.depthWrite,
            skirtSkinned: mesh.userData._skirtSkinned === true,
            mesh: mesh.name,
          });
        });
      });
      console.table(rows);
      return rows;
    };
    // __vrmDebug(0..3) — MToon debug view: 0 none, 1 litShadeRate (the ShadeRT
    // 的な可視化), 2 normal, 3 uv.
    (window as any).__vrmDebug = (n: number = 1) => {
      if (!vrm) return "no vrm";
      const mode = ["none", "litShadeRate", "normal", "uv"][Math.max(0, Math.min(3, Math.round(n)))];
      vrm.scene.traverse((o: THREE.Object3D) => {
        const ms = (o as THREE.Mesh).material;
        (Array.isArray(ms) ? ms : [ms]).forEach((mt: any) => {
          if (mt?.isMToonMaterial) mt.debugMode = mode;
        });
      });
      return mode;
    };
    // __vrmStage(1..4): the note article's process, stage by stage, on the video
    // preset — 1 = ベース色 (flat SimplePaint-like base, no shade/edge),
    // 2 = +Shade.x (threshold shade/highlight/sublight), 3 = +Edge.x (outlines),
    // 4 = コントローラー (the full preset). For A/B-ing against the article images.
    (window as any).__vrmStage = (n: number = 4, preset: string = "video") => {
      const stage = Math.max(1, Math.min(4, Math.round(n)));
      Object.assign(LOOK, LOOK_DEFAULTS, LOOK_PRESETS[preset] ?? LOOK_PRESETS.video);
      if (stage < 3) {
        LOOK.outlineAll = 0;
        LOOK.outlineWidth = 0;
      }
      if (stage < 2) {
        LOOK.shadeShift = -1; // force everything to the lit side = pure flat base
        LOOK.faceLit = 1;
        LOOK.shadowStrength = 0; // no cast shadows either
        LOOK.subInt = 0;
        LOOK.hiCoord = 0;
        LOOK.subCoord = 0;
      }
      applyLookState();
      saveLook();
      return `stage ${stage}: ${["", "ベース色", "+Shade", "+Edge", "コントローラー"][stage]}`;
    };
    console.info(`[vrm] look build ${LOOK_BUILD}`);
    applyLookState(); // sync everything to any loaded saved look

    // On-screen slider panel (🎭 button) to tune the resting FACE — one slider per
    // face morph (eye shape, brows, mouth corners, effects). Edits drive
    // vrmExpression.setBase and persist (shared with the MMD viewer's saved face).
    /**
     * 😀 FACE / CLIP AUDITION — the same idea as the 🎬 idle picker, for the face.
     *
     * Built because "this model has face motion" needed checking and could not be
     * checked: the .glb's one embedded clip turned out to move exactly two morphs
     * (eyeBlinkLeft/Right), while the model itself carries 293 that nothing drives.
     * A list you can click through is the only way to find out what is in there,
     * and it reads the morphs off the loaded meshes so it works for any model —
     * Miku's Japanese PMX names and the Avaturn ARKit/Oculus ones alike.
     */
    const buildFaceMotionPanel = (): (() => void) => {
      const btn = document.createElement("button");
      btn.textContent = "😀";
      btn.title = L("表情モーフ / 内蔵アニメを試聴", "Audition face morphs / embedded clips");
      btn.style.cssText = "position:absolute;top:8px;right:84px;z-index:20;width:32px;height:32px;"
        + "border-radius:8px;border:1px solid rgba(255,255,255,.25);background:rgba(20,24,40,.6);"
        + "color:#fff;cursor:pointer;font-size:15px;line-height:1";
      container.appendChild(autoHide(btn));

      const panel = document.createElement("div");
      // Second class: the 🎭 toggle opens this alongside the resting-face panel,
      // so the two surfaces read as one "face" panel. Offset downward so they
      // stack instead of covering each other.
      panel.className = "vrm-tune-panel vrm-face-motion-panel";
      panel.style.cssText = "position:absolute;top:48px;right:8px;z-index:21;width:320px;max-height:76%;"
        + "display:none;flex-direction:column;border-radius:10px;border:1px solid rgba(255,255,255,.18);"
        + "background:rgba(14,17,28,.94);color:#e8ecf5;font:12px/1.45 system-ui,sans-serif;"
        + "box-shadow:0 8px 30px rgba(0,0,0,.5);overflow:hidden";
      container.appendChild(panel);

      const head = document.createElement("div");
      head.style.cssText = "padding:8px 10px;border-bottom:1px solid rgba(255,255,255,.12);"
        + "display:flex;gap:6px;align-items:center;flex:0 0 auto";
      const info = document.createElement("span");
      info.style.cssText = "flex:1;opacity:.85";
      const relaxBtn = document.createElement("button");
      SET(relaxBtn, "全解除", "Clear all");
      relaxBtn.style.cssText = "padding:3px 8px;border-radius:6px;border:1px solid rgba(255,255,255,.25);"
        + "background:rgba(60,64,80,.9);color:#fff;cursor:pointer";
      head.append(info, relaxBtn);

      const clipBox = document.createElement("div");
      clipBox.style.cssText = "padding:6px 10px;border-bottom:1px solid rgba(255,255,255,.12);flex:0 0 auto";

      const filter = document.createElement("input");
      filter.placeholder = L("モーフ名で絞り込み (例: jaw, blink, aa)", "Filter by morph name (e.g. jaw, blink, aa)");
      filter.style.cssText = "margin:8px 10px 4px;padding:5px 8px;border-radius:6px;"
        + "border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.06);"
        + "color:#e8ecf5;flex:0 0 auto";

      const list = document.createElement("div");
      list.style.cssText = "overflow-y:auto;padding:4px 6px 10px;flex:1 1 auto";
      panel.append(head, clipBox, filter, list);

      const renderClips = () => {
        clipBox.textContent = "";
        const names = faceMotion.clipNames;
        const label = document.createElement("div");
        label.style.cssText = "opacity:.75;margin-bottom:4px";
        if (!names.length) {
          SET(label, "内蔵アニメーション: なし", "Embedded animations: none");
          clipBox.append(label);
          return;
        }
        label.textContent = L(`内蔵アニメーション (${names.length})`, `Built-in animations (${names.length})`);
        clipBox.append(label);
        for (const n of names) {
          const row = document.createElement("div");
          row.style.cssText = "display:flex;gap:6px;align-items:center;margin:3px 0";
          const nm = document.createElement("span");
          nm.textContent = n;
          nm.style.cssText = "flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
          const play = document.createElement("button");
          play.textContent = faceMotion.playing === n ? L("停止", "Stop") : L("再生", "Play");
          play.style.cssText = "padding:2px 10px;border-radius:6px;border:1px solid rgba(255,255,255,.25);"
            + "background:rgba(80,140,220,.85);color:#fff;cursor:pointer";
          play.addEventListener("click", () => {
            if (faceMotion.playing === n) faceMotion.stopClip();
            else faceMotion.playClip(n);
            renderClips();
          });
          row.append(nm, play);
          clipBox.append(row);
        }
        const warn = document.createElement("div");
        warn.style.cssText = "opacity:.6;margin-top:4px;font-size:11px";
        SET(warn, "※ 内蔵クリップは全身も動かすため、通常のモーションと競合します", "Note: an embedded clip also drives the body, so it fights the normal motion");
        clipBox.append(warn);
      };

      const renderList = () => {
        list.textContent = "";
        const q = filter.value.trim().toLowerCase();
        const all = faceMotion.morphNames;
        const matched = q ? all.filter((n) => n.toLowerCase().includes(q)) : all;
        // Sort by GROUP first. Sorting by name alone put the headings back every
        // few rows (viseme, other, viseme, brow...) because the names interleave
        // alphabetically, which made the grouping useless as a way to find things.
        const ORDER = ["viseme", "mouth", "eye", "brow", "face", "body-shape", "other"];
        const shown = matched.slice().sort((a, b) => {
          const ga = ORDER.indexOf(faceMotion.groupOf(a));
          const gb = ORDER.indexOf(faceMotion.groupOf(b));
          return ga !== gb ? ga - gb : a.localeCompare(b);
        });
        info.textContent = L(`モーフ ${shown.length}/${all.length}`, `Morphs ${shown.length}/${all.length}`);
        let group = "";
        for (const name of shown) {
          const g = faceMotion.groupOf(name);
          if (g !== group) {
            group = g;
            const h = document.createElement("div");
            SET(h, g);
            h.style.cssText = "margin:8px 4px 2px;opacity:.55;text-transform:uppercase;font-size:10px;letter-spacing:.05em";
            list.append(h);
          }
          const row = document.createElement("div");
          row.style.cssText = "display:flex;gap:6px;align-items:center;padding:1px 4px";
          const nm = document.createElement("span");
          SET(nm, name);
          nm.title = name; // key stays Japanese
          nm.style.cssText = "flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;cursor:pointer";
          // Click the NAME to sweep it 0->1->0: the fastest way to see what a
          // morph is without deciding a value first.
          nm.addEventListener("click", () => faceMotion.sweep(name));
          const sl = document.createElement("input");
          sl.type = "range";
          sl.min = "0"; sl.max = "1"; sl.step = "0.02"; sl.value = "0";
          sl.style.cssText = "width:96px";
          sl.addEventListener("input", () => {
            const v = parseFloat(sl.value);
            faceMotion.hold(name, v > 0.001 ? v : null);
          });
          // Ticked = keep this morph in the RESTING face (persisted). Unticked,
          // the slider is a preview only — which is the whole difference the two
          // separate lists used to encode, now expressed per row.
          row.append(nm, sl);
          list.append(row);
        }
        if (!shown.length) {
          const e = document.createElement("div");
          e.style.cssText = "opacity:.6;padding:10px 4px";
          e.textContent = all.length ? L("一致するモーフがありません", "No matching morphs") : L("このモデルにモーフはありません", "This model has no morphs");
          list.append(e);
        }
      };

      relaxBtn.addEventListener("click", () => {
        faceMotion.releaseAll();
        renderList();
      });
      filter.addEventListener("input", renderList);
      btn.addEventListener("click", () => {
        retranslate(container);
        container.querySelectorAll(".vrm-tune-panel").forEach((p) => {
          if (p !== panel) (p as HTMLElement).style.display = "none";
        });
        const open = panel.style.display === "none";
        panel.style.display = open ? "flex" : "none";
        if (open) { renderClips(); renderList(); }
        else faceMotion.releaseAll();
      });

      return () => {
        faceMotion.releaseAll();
        faceMotion.stopClip();
        btn.remove();
        panel.remove();
      };
    };

    const buildFacePanel = (dict: Record<string, number>): (() => void) => {
      // まばたき IS adjustable now (it's the resting "hood"); the blink animates on
      // top of it. Only skip the lip-sync mouth vowels.
      const skip = new Set(["あ", "い", "う", "え", "お"]);
      const base = vrmExpression.getBase();
      const panel = document.createElement("div");
      panel.className = "vrm-face-panel vrm-tune-panel";
      panel.style.cssText =
        "position:absolute;top:40px;right:8px;max-height:78%;overflow-y:auto;width:230px;" +
        "background:rgba(18,18,26,0.9);color:#eee;font:12px sans-serif;padding:8px 10px;" +
        "border-radius:8px;z-index:50;display:none;box-shadow:0 4px 16px rgba(0,0,0,.5)";
      const head = document.createElement("div");
      head.style.cssText = "font-weight:bold;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center";
      const title = document.createElement("span");
      SET(title, "表情調整", "Face");
      const resetBtn = document.createElement("button");
      resetBtn.textContent = "Reset";
      resetBtn.style.cssText = "font-size:11px;cursor:pointer;background:#444;color:#fff;border:none;border-radius:4px;padding:2px 8px";
      head.appendChild(title);
      head.appendChild(resetBtn);
      panel.appendChild(head);
      const sliders: Record<string, HTMLInputElement> = {};
      // ONE list, not two. This panel used to render a slider per morph and the
      // section below renders the same set again — same morphs, different order,
      // one with search. Keeping both meant the same "笑い" appeared twice with two
      // independent sliders. The list below is the better of the two (grouped and
      // filterable), so it is the only one now; what this panel keeps is the part
      // the list has no notion of — committing the current pose as the RESTING face.
      // RESTING FACE sliders — the persisted base pose, one row per morph. This is
      // the panel that existed before the merge experiment; the morph AUDITION list
      // lives behind its own button again.
      Object.keys(dict)
        .sort((a, b) => dict[a] - dict[b])
        .forEach((name) => {
          if (skip.has(name)) return;
          const row = document.createElement("label");
          row.style.cssText = "display:flex;align-items:center;gap:6px;margin:2px 0";
          const lab = document.createElement("span");
          SET(lab, name); lab.title = name; // key stays Japanese
          lab.style.cssText = "flex:0 0 84px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
          const sl = document.createElement("input");
          sl.type = "range"; sl.min = "0"; sl.max = "1"; sl.step = "0.01";
          sl.value = String((vrmExpression.getBase() as Record<string, number>)[name] ?? 0);
          sl.style.cssText = "flex:1;min-width:0";
          const val = document.createElement("span");
          val.style.cssText = "flex:0 0 34px;text-align:right;opacity:.7";
          val.textContent = sl.value;
          sl.addEventListener("input", () => {
            val.textContent = sl.value;
            vrmExpression.setBase(name, parseFloat(sl.value));
          });
          sliders[name] = sl;
          row.append(lab, sl, val);
          panel.appendChild(row);
        });
      resetBtn.addEventListener("click", () => {
        const nb = vrmExpression.resetBase();
        Object.keys(sliders).forEach((n) => {
          sliders[n].value = String(nb[n] ?? 0);
          sliders[n].dispatchEvent(new Event("input"));
        });
      });
      const toggle = document.createElement("button");
      toggle.textContent = "🎭";
      toggle.title = L("顔の調整（安静時の表情 + モーフ確認）", "Face (resting expression + morph audit)");
      toggle.style.cssText =
        "position:absolute;top:8px;right:8px;z-index:51;width:32px;height:32px;"
        + "border-radius:8px;border:1px solid rgba(255,255,255,.25);"
        + "background:rgba(20,24,40,.6);color:#fff;cursor:pointer;"
        + "font-size:15px;line-height:1;display:flex;align-items:center;"
        + "justify-content:center;padding:0";
      toggle.addEventListener("click", () => {
        const open = panel.style.display === "none";
        retranslate(container);
        container.querySelectorAll(".vrm-tune-panel").forEach((p) => {
          (p as HTMLElement).style.display = "none";
        });
        panel.style.display = open ? "block" : "none";
        // ONE button for the whole face. The two surfaces stayed separate
        // builders (one tunes the persisted resting face, the other auditions
        // every morph and is deliberately not saved), but there was no reason
        // for the USER to know that — they are both "the face", and the panels
        // looked identical. Opened together, stacked.
        // the morph section now lives inside `panel`, so it opens with it
      });
      container.appendChild(panel);
      container.appendChild(autoHide(toggle));
      return () => { panel.remove(); toggle.remove(); };
    };

    // On-screen 🎨 panel — the VRM equivalent of the MMD viewer's visual-tuning
    // panel: sliders for lights / bloom / grade / effects + colour pickers, applied
    // live and persisted (localStorage vrmLook1). Mirrors the MMD look-panel controls.
    const buildLookPanel = (): (() => void) => {
      type LK = keyof typeof LOOK;
      const SLIDERS: [LK, string, number, number, number][] = [
        ["hairStiff", "髪 硬さ", 1.3, 16, 0.1], // 根元の硬さ = 髪の位置の固定度
        ["hairTipSoft", "毛先 柔さ", 0.05, 1, 0.05], // ↓ = 毛先が自由になびく（物理感）; ↑1 = 全体が硬い棒
        ["hairDrag", "髪 減衰", 0.3, 1, 0.02],
        ["hairGrav", "髪 重力", 0, 1, 0.02], // ↑ = 下に垂れる（重さ）※このモデルは0が本来の髪位置
        ["gestShoulder", "肩の上下", 0, 1, 0.05], // ↓ = 肩が上がらない（すくみ防止）
        ["gestGain", "上半身の強さ", 0.3, 1, 0.05], // 上半身ジェスチャーの大きさ
        ["gestFinger", "指の曲げ", 0, 1, 0.05], // ↓ = 手を開く（握り防止）
        ["ikPelvisRot", "骨盤の傾き", 0, 1, 0.05], // 足はIK接地・骨盤の傾き量
        ["ikPelvisTrans", "重心移動", 0, 1, 0.05], // 重心移動（膝が曲がる）量
        ["key", "キー光", 0, 2, 0.05],
        ["keyAzim", "キー 左右", -180, 180, 5],
        ["keyElev", "キー 高さ", 0, 85, 5],
        ["ambient", "環境光", 0, 1.5, 0.05],
        ["hemi", "半球光", 0, 1.5, 0.05],
        ["fillFront", "前フィル", 0, 1, 0.02],
        ["fillSide", "横フィル", 0, 1, 0.02],
        ["subInt", "サブ光", 0, 2.5, 0.05],
        ["subPow", "サブ光幅", 0.5, 8, 0.5],
        ["subMix", "サブ光混合", 0, 1, 0.05],
        ["subIntCool", "リム寒 強", 0, 2.5, 0.05],
        ["subIntWarm", "リム暖 強", 0, 2.5, 0.05],
        ["subCoolAzim", "寒 左右", -180, 180, 5],
        ["subCoolElev", "寒 高さ", -85, 85, 5],
        ["subWarmAzim", "暖 左右", -180, 180, 5],
        ["subWarmElev", "暖 高さ", -85, 85, 5],
        ["matcapInt", "反射(艶)", 0, 2, 0.05],
        ["fresInt", "フレネル", 0, 1, 0.02],
        ["fresPow", "フレネル幅", 0.5, 8, 0.5],
        ["bloomStrength", "Bloom強", 0, 1.5, 0.05],
        ["bloomRadius", "Bloom半径", 0, 1.5, 0.05],
        ["bloomThreshold", "Bloom閾値", 0, 1, 0.02],
        ["lineInt", "線画 強さ", 0, 1, 0.05], // Croquis Y: 深度エッジ線画
        ["lineThresh", "線画 閾値", 0.01, 0.2, 0.005], // ↓ = 線が増える
        ["lineNInt", "線画(法線)", 0, 1, 0.05], // Croquis X: シワ・面の折れ目
        ["lineZInt", "線画(色差)", 0, 1, 0.05], // Croquis Z: 色の境界
        ["diffusion", "ソフトグロー", 0, 0.5, 0.02], // ikDiffusion相当の全面グロー
        ["ssao", "SSAO接触影", 0, 1, 0.05], // SvSSAO/ImSSAO相当 (GTAO)
        ["faceNormal", "顔法線", 0, 1, 0.05], // 顔の影をアニメ的な形に (ウマ娘方式)
        ["glowInt", "発光(AL)", 0, 2, 0.05], // AutoLuminous相当: HL材質の発光
        ["sssInt", "SSS境界赤み", 0, 1, 0.05], // 明暗境界の赤み (v2.1 SSS)
        ["sssWidth", "SSS幅", 0.02, 0.5, 0.02],
        ["shadowFt", "影ボケ足", 0.005, 0.3, 0.005], // 落ち影2値化の境界幅
        ["pointInt", "点光サブ", 0, 2, 0.05], // PostPointLight相当 (寒色サブに合流)
        ["pointX", "点光 左右", -1.5, 1.5, 0.05],
        ["pointY", "点光 高さ", 0, 2, 0.05],
        ["pointZ", "点光 前後", -1, 1.5, 0.05],
        ["pointRange", "点光 範囲", 0.1, 2, 0.05],
        ["gradTopInt", "グラデ上", 0, 1, 0.02], // msGradation相当: 上からの明るい色
        ["gradBotInt", "グラデ下", 0, 1, 0.02], // 下からの沈み色 (乗算)
        ["browFront", "眉の露出", 0, 1, 1], // 眉・まつげを前髪の上に描く
        ["maskUse", "貼図マスク", 0, 1, 1], // 配布Shade/SubLightマスク (手描き影・ハイ)
        ["pureTex", "純色貼図", 0, 1, 1], // 配布の純色化テクスチャに差し替え
        ["skirtOpaque", "スカート不透明", 0, 1, 1], // 0 = 元の透明感に戻す
        ["armOpaque", "腕 不透明", 0, 1, 1], // 袖の半透明パネル (0 = 元の透明感)
        ["gContrast", "コントラスト", 0.5, 2, 0.02],
        ["gSat", "彩度", 0.3, 2, 0.02],
        ["chroma", "色収差", 0, 4, 0.1],
        ["gTemp", "色温度", -1, 1, 0.05],
        ["gTint", "ティント", -1, 1, 0.05],
        ["gSplit", "スプリット", 0, 1, 0.05],
        ["celBands", "セル塗り段数", 0, 5, 1],
        ["baseSat", "ベース彩度", 0, 1, 0.02], // テクスチャ純色化 (0.5 = 無変化)
        ["baseVal", "ベース明度", 0, 1, 0.02],
        ["hairColorMix", "髪色 適用", 0, 1, 0.05], // 「髪色」ピッカーの色へのブレンド量
        ["hairHueShift", "髪 色相", -180, 180, 5], // 補助: 元の髪色の色相回転
        ["eyeHueShift", "目 色相", -180, 180, 5], // + = 青寄り (虹彩のみ・白目は不変)
        ["faceLit", "顔を守る", 0, 1, 0.05], // ShadeRT Rマスク相当: 顔をN·L影から保護
        ["mouthNarrow", "口幅縮小", 0, 0.4, 0.02], // 作者の口幅縮小相当 (口を小さく)
        ["skinShade", "肌の影", 0, 1, 0.05], // 肌に乗る影色の強さ (↓ = 淡いピンク)
        ["shadeCoord", "msTC影 適用", 0, 1, 0.05], // 0 = モデルの影色, 1 = ベース色からHSV導出(色トレス)
        ["shadeHue", "影 色相", 0, 360, 5],
        ["shadeHueMix", "影 色相適用", 0, 1, 0.05],
        ["shadeSat", "影 彩度", 0, 1, 0.02], // 0.5 = 無変化
        ["shadeVal", "影 明度", 0, 1, 0.02], // 0.5 = 無変化
        ["shadeShift", "影 しきい値", -0.5, 0.5, 0.02], // + = 影が広がる
        ["shadeToony", "影 とがり", -1, 1, 0.05], // -1 = モデル値, 1 = 完全2値
        ["shadowHard", "落ち影2値化", 0, 1, 0.05],
        ["shadowRes", "影解像度0-2", 0, 2, 1], // 1024/2048/4096
        ["hiCoord", "ハイ 適用", 0, 1, 0.05], // 0 = 加算matcap, 1 = 置換ハイライト
        ["hiThresh", "ハイ 閾値", 0, 1, 0.02],
        ["hiHue", "ハイ 色相", 0, 360, 5],
        ["hiHueMix", "ハイ 色相適用", 0, 1, 0.05],
        ["hiSat", "ハイ 彩度", 0, 1, 0.02],
        ["hiVal", "ハイ 明度", 0, 1, 0.02],
        ["subCoord", "サブ 置換", 0, 1, 0.05], // 0 = 加算リム, 1 = msTC色相置換
        ["subThresh", "サブ 閾値", 0, 1, 0.02],
        ["subSat", "サブ 彩度", 0, 1, 0.02],
        ["subVal", "サブ 明度", 0, 1, 0.02],
        ["outlineWidth", "輪郭 太さ", 0, 3, 0.1],
        ["outlineMix", "輪郭 ライト", -1, 1, 0.05], // -1 = モデル値, 0 = ベタ輪郭色
        ["outlineAll", "輪郭 全材質", 0, 1, 1], // 髪・顔のランタイム生成輪郭 (MMD標準エッジ相当)
        ["toonSteps", "粗さ(段数)", 0, 10, 1],
        ["vignette", "周辺減光", 0, 1, 0.05],
        ["grain", "フィルム粒子", 0, 0.3, 0.01],
        ["bars", "上下黒帯", 0, 0.2, 0.01],
        ["skinWarmth", "肌の暖かさ", 0, 0.8, 0.02],
        ["skinSub", "肌サブ光", 0, 1, 0.05], // 0 = 肌にサブライトを当てない(従来)
        ["shadowStrength", "落ち影", 0, 1, 0.05],
        ["toneMode", "トーン0-3", 0, 3, 1],
        ["toneExposure", "露出", 0.3, 2, 0.05],
        ["dofAperture", "DOF量", 0, 0.004, 0.0002],
        ["dofBlur", "DOFボケ", 0, 0.02, 0.001],
      ];
      const HEXCOLORS: [LK, string][] = [
        ["keyColor", "キー色"],
        ["coolColor", "リム寒色"],
        ["warmColor", "リム暖色"],
        ["fresColor", "フレネル色"],
        // -1 (モデル値) renders as white in the picker; picking any colour
        // overrides — go back to the authored colour via Reset / __vrmLook.
        ["outlineColor", "輪郭色"],
        ["hairColor", "髪色"],
        ["gradTop", "グラデ上色"],
        ["gradBot", "グラデ下色"],
        ["bgInner", "背景(内)"],
        ["bgOuter", "背景(外)"],
      ];
      const RGBCOLORS: [string, [LK, LK, LK]][] = [
        ["階調暖色", ["gWarmR", "gWarmG", "gWarmB"]],
        ["影の色", ["gLiftR", "gLiftG", "gLiftB"]],
      ];
      const panel = document.createElement("div");
      panel.className = "vrm-look-panel vrm-tune-panel";
      panel.style.cssText =
        "position:absolute;top:40px;right:8px;max-height:82%;overflow-y:auto;width:236px;" +
        "background:rgba(18,18,26,0.92);color:#eee;font:12px sans-serif;padding:8px 10px;" +
        "border-radius:8px;z-index:50;display:none;box-shadow:0 4px 16px rgba(0,0,0,.5)";
      const head = document.createElement("div");
      head.style.cssText = "font-weight:bold;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center";
      const title = document.createElement("span"); title.textContent = L("ビジュアル調整", "Visual") + ` ${LOOK_BUILD}`;
      const resetBtn = document.createElement("button"); resetBtn.textContent = "Reset";
      resetBtn.style.cssText = "font-size:11px;cursor:pointer;background:#444;color:#fff;border:none;border-radius:4px;padding:2px 8px";
      head.appendChild(title); head.appendChild(resetBtn); panel.appendChild(head);

      const rows: Array<() => void> = [];
      // Preset buttons: 既定 (the tuned reference look) / msTC (msToonCoordinator
      // coordination on). Applied as defaults + overrides so they fully reset.
      const presetRow = document.createElement("div");
      presetRow.style.cssText = "display:flex;gap:6px;margin:2px 0 6px";
      // Highlight whichever preset is active, and restore it on load.
      const presetBtns: Record<string, HTMLButtonElement> = {};
      const markPreset = (active: string | null) => {
        Object.entries(presetBtns).forEach(([k, el]) => {
          el.style.background = k === active ? "#5570b0" : "#3a3a55";
        });
      };
      (["base", "mstc", "video", "recipe"] as const).forEach((nm) => {
        const b = document.createElement("button");
        b.textContent =
          nm === "mstc" ? "msTC" : nm === "video" ? L("映像", "Video") : nm === "recipe" ? L("配方", "Recipe") : L("既定", "Default");
        b.style.cssText =
          "flex:1;font-size:11px;cursor:pointer;background:#3a3a55;color:#fff;border:none;border-radius:4px;padding:3px 0";
        b.addEventListener("click", () => {
          const keepSkirt = LOOK.skirtOpaque; // user choices, survive preset swaps
          const keepArm = LOOK.armOpaque;
          Object.assign(LOOK, LOOK_DEFAULTS, LOOK_PRESETS[nm]);
          LOOK.skirtOpaque = keepSkirt;
          LOOK.armOpaque = keepArm;
          applyLookState(); saveLook();
          // Remember WHICH preset, not just the resulting slider values. The
          // values were already persisted, but a preset is also a statement of
          // intent — reopening the panel showed no sign of what had been picked,
          // and a later default change could not be inherited.
          try { localStorage.setItem(`${LS_LOOK}:preset`, nm); } catch { /* private mode */ }
          markPreset(nm);
          rows.forEach((r) => r());
        });
        presetBtns[nm] = b;
        presetRow.appendChild(b);
      });
      // Restore the remembered choice. The slider VALUES are already restored
      // from LS_LOOK, so this only re-marks the button — re-applying the preset
      // here would discard any tweaks made after picking it.
      try { markPreset(localStorage.getItem(`${LS_LOOK}:preset`)); } catch { markPreset(null); }
      panel.appendChild(presetRow);
      const hex6 = (n: number) => `#${(n & 0xffffff).toString(16).padStart(6, "0")}`;
      const b2 = (x: number) => Math.max(0, Math.min(255, Math.round(x))).toString(16).padStart(2, "0");
      SLIDERS.forEach(([key, label, min, max, step]) => {
        const row = document.createElement("label");
        row.style.cssText = "display:flex;align-items:center;gap:6px;margin:2px 0";
        const lab = document.createElement("span"); SET(lab, label); lab.title = String(key);
        lab.style.cssText = "flex:0 0 72px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
        const sl = document.createElement("input");
        sl.type = "range"; sl.min = String(min); sl.max = String(max); sl.step = String(step);
        sl.style.cssText = "flex:1;min-width:0";
        const val = document.createElement("span"); val.style.cssText = "flex:0 0 38px;text-align:right;opacity:.7";
        const sync = () => { sl.value = String((LOOK as any)[key]); val.textContent = String((LOOK as any)[key]); };
        sync();
        sl.addEventListener("input", () => {
          (LOOK as any)[key] = parseFloat(sl.value); val.textContent = sl.value;
          applyLookState(); saveLook();
        });
        rows.push(sync);
        row.appendChild(lab); row.appendChild(sl); row.appendChild(val); panel.appendChild(row);
      });
      HEXCOLORS.forEach(([key, label]) => {
        const row = document.createElement("label");
        row.style.cssText = "display:flex;align-items:center;gap:6px;margin:2px 0";
        const lab = document.createElement("span"); SET(lab, label);
        lab.style.cssText = "flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
        const ci = document.createElement("input"); ci.type = "color";
        ci.style.cssText = "flex:0 0 34px;height:20px;padding:0;border:none;background:none;cursor:pointer";
        const sync = () => { ci.value = hex6((LOOK as any)[key]); };
        sync();
        ci.addEventListener("input", () => {
          (LOOK as any)[key] = parseInt(ci.value.slice(1), 16);
          applyLookState(); saveLook();
        });
        rows.push(sync);
        row.appendChild(lab); row.appendChild(ci); panel.appendChild(row);
      });
      RGBCOLORS.forEach(([label, keys]) => {
        const row = document.createElement("label");
        row.style.cssText = "display:flex;align-items:center;gap:6px;margin:2px 0";
        const lab = document.createElement("span"); SET(lab, label);
        lab.style.cssText = "flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
        const ci = document.createElement("input"); ci.type = "color";
        ci.style.cssText = "flex:0 0 34px;height:20px;padding:0;border:none;background:none;cursor:pointer";
        const sync = () => {
          ci.value = `#${b2((LOOK as any)[keys[0]] * 255)}${b2((LOOK as any)[keys[1]] * 255)}${b2((LOOK as any)[keys[2]] * 255)}`;
        };
        sync();
        ci.addEventListener("input", () => {
          (LOOK as any)[keys[0]] = parseInt(ci.value.slice(1, 3), 16) / 255;
          (LOOK as any)[keys[1]] = parseInt(ci.value.slice(3, 5), 16) / 255;
          (LOOK as any)[keys[2]] = parseInt(ci.value.slice(5, 7), 16) / 255;
          applyLookState(); saveLook();
        });
        rows.push(sync);
        row.appendChild(lab); row.appendChild(ci); panel.appendChild(row);
      });
      resetBtn.addEventListener("click", () => {
        Object.assign(LOOK, LOOK_DEFAULTS);
        applyLookState(); saveLook();
        rows.forEach((r) => r());
      });

      const toggle = document.createElement("button");
      toggle.textContent = "🎨"; toggle.title = L("ビジュアル調整パネル", "Visual tuning");
      toggle.style.cssText =
        "position:absolute;top:8px;right:46px;z-index:51;width:32px;height:32px;"
        + "border-radius:8px;border:1px solid rgba(255,255,255,.25);"
        + "background:rgba(20,24,40,.6);color:#fff;cursor:pointer;"
        + "font-size:15px;line-height:1;display:flex;align-items:center;"
        + "justify-content:center;padding:0";
      toggle.addEventListener("click", () => {
        const open = panel.style.display === "none";
        retranslate(container);
        container.querySelectorAll(".vrm-tune-panel").forEach((p) => {
          (p as HTMLElement).style.display = "none";
        });
        panel.style.display = open ? "block" : "none";
      });
      container.appendChild(panel); container.appendChild(autoHide(toggle));
      return () => { panel.remove(); toggle.remove(); };
    };

    const clock = new THREE.Clock();
    const pointViewPos = new THREE.Vector3();

    const loader = new GLTFLoader();
    loader.register((parser: ConstructorParameters<typeof VRMLoaderPlugin>[0]) => new VRMLoaderPlugin(parser));
    // Cache-bust the .vrm: the file is re-deployed under the SAME name while the
    // look is tuned, so a cached copy would show a stale (dark/old) model. A
    // per-load query forces a fresh fetch. (Local file, so the refetch is cheap.)
    const bustUrl = url.includes(".vrm") || url.includes(".glb")
      ? `${url}${url.includes("?") ? "&" : "?"}cb=${Date.now()}`
      : url;
    loader.load(
      bustUrl,
      (gltf: { userData: { vrm: VRM }; scene: THREE.Object3D; animations?: THREE.AnimationClip[] }) => {
        if (disposed) return;
        vrm = gltf.userData.vrm as VRM;
        // Plain .glb: no VRM extension, so the loader plugin leaves userData.vrm
        // undefined. Build a humanoid over the glTF's own bones instead — see
        // glb-humanoid.ts for why that is a name table and not a retarget.
        if (!vrm) {
          const built = buildVrmFromGltf(gltf);
          if (!built) {
            console.error("[vrm-viewer] .glb has no humanoid rig this app can drive");
            return;
          }
          vrm = built.vrm;
          // A plain .glb is a PBR asset from some other pipeline, not a toon
          // model — so it opts out of the stylised pass by default. Override
          // per-session with __vrmPlainStyle(false) if a particular .glb does
          // suit the look.
          plainStyle = true;
          console.info(
            `[glb-humanoid] rig mapped from ${built.boneNames.length} nodes`
            + (built.missing.length ? `; unmapped optional: ${built.missing.join(", ")}` : ""),
          );
        } else {
          // A VRM can be a PBR asset too. The test above — "is this a plain
          // .glb" — is a proxy for "did this come from a toon pipeline", and it
          // held only while every .vrm loaded here was a VRoid/MMD model. It is
          // the MToon material that actually marks a toon asset, so ask for that
          // instead: Miku (31/31 MToon) and the VRoid sample (13/13) keep the
          // stylised pass, and a photoreal export that happens to carry a VRM
          // extension no longer comes out violet with a cyan edge glow.
          let mtoon = false;
          vrm.scene.traverse((o: THREE.Object3D) => {
            const mat = (o as THREE.Mesh).material;
            for (const m of Array.isArray(mat) ? mat : [mat]) {
              if ((m as { isMToonMaterial?: boolean } | undefined)?.isMToonMaterial) mtoon = true;
            }
          });
          if (!mtoon) {
            plainStyle = true;
            console.info("[vrm-viewer] no MToon material — PBR asset, stylised pass off");
          }
        }
        // Before the look pass below, so the rim uniforms it injects are adopted.
        // An asset that opts into the AI look is the android: it also gets the robot
        // standby stance instead of Rem's/Miku's (vrm-motion-player robotIdle).
        vrmMotionPlayer.robotIdle = aiFx.bind(gltf.scene);
        // plainStyle is read by applyLookState, which last ran before this model
        // was known — re-run it or the lights and grade stay on the toon values.
        if (plainStyle) applyLookState();
        VRMUtils.removeUnnecessaryVertices(gltf.scene);
        // VRM 1.0 faces +Z; VRM 0.x faces -Z -> flip to face the +Z camera.
        if (vrm.meta && (vrm.meta as { metaVersion?: string }).metaVersion === "0") {
          vrm.scene.rotation.y = Math.PI;
        }
        scene.add(vrm.scene);
        // WASTE FROM THE PMX->VRM CONVERSION, removed before anything touches the
        // model. Both passes delete data that cannot reach a pixel, so the render
        // diff across them must be exactly zero — see utils/vrm-optimize.ts.
        //
        // Order matters twice over. The texture dedupe runs BEFORE the mip /
        // dilation work below so that work runs once per surviving texture rather
        // than once per duplicate, and so the `alphaShapedTex` pre-pass classifies
        // the objects that will actually be uploaded. The morph prune runs before
        // the face systems bind (further down), so they resolve names against the
        // renumbered dictionaries rather than being re-bound afterwards.
        {
          const tex = dedupeTextures(vrm.scene);
          const morph = pruneZeroMorphTargets(vrm);
          const MB = (b: number) => Math.round(b / 1048576);
          console.info(
            `[vrm-optimize] textures ${tex.before} -> ${tex.after} objects `
            + `(${MB(tex.bytesBefore)} -> ${MB(tex.bytesAfter)} MB); `
            + `morphs ${morph.slotsBefore} -> ${morph.slotsAfter} slots `
            + `(${MB(morph.bytesBefore)} -> ${MB(morph.bytesAfter)} MB, `
            + `${morph.meshesCleared}/${morph.meshes} meshes now morph-free)`,
          );
        }
        (window as any).__vrmCost = () => describeVrmCost(vrm!.scene);
        // Anisotropic filtering on every texture: the detailed sleeve / armband
        // patterns aliased into grainy noise when seen at grazing angles.
        const maxAniso = renderer.capabilities.getMaxAnisotropy();
        // ALPHA-SHAPED TEXTURES — never mip these (see the mip block below).
        // Collected in a PRE-PASS because textures are SHARED between materials:
        // Hair01, Hair02 and Hairshadow are three materials over one 2048x1200
        // image, so a per-material test would still mip the hair the moment any
        // sibling that happens to be opaque reached the same texture object.
        // Membership is by texture identity, decided before anything is changed.
        // Captured at LOAD time on purpose: applyLook later flips the coat
        // overlays q201-q203 to transparent:false, but their silhouette is the
        // printed circuit pattern in their alpha exactly like the hair's, so the
        // load-time reading is the one that describes the texture. Verified on
        // this model: 32 alpha-shaped textures, 48 opaque, and ZERO shared
        // between the two groups — so nothing opaque loses its prefilter here.
        const TEX_KEYS = ["map", "emissiveTexture", "shadeMultiplyTexture", "matcapTexture"];
        const alphaShapedTex = new Set<unknown>();
        vrm.scene.traverse((obj: THREE.Object3D) => {
          const mesh = obj as THREE.Mesh;
          if (!mesh.isMesh) return;
          const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
          mats.forEach((mat: THREE.Material) => {
            const m = mat as any;
            // ALPHA-TESTED counts too. The test used to be `transparent` alone,
            // which split two textures that are the same kind of thing: the coat
            // overlays q201-q203 came out of the converter as BLEND and were
            // excluded, while q301/q302 — the SAME printed circuit pattern, shape
            // in the alpha — came out as MASK, read as transparent:false here, and
            // got mipped. One print stayed crisp and its twin dissolved, purely
            // from an alphaMode the conversion picked. Shape-in-alpha is the
            // property that matters, and a cutoff is just another way to declare it.
            if (!m?.transparent && !(m?.alphaTest > 0)) return;
            TEX_KEYS.forEach((k) => {
              const tex = m[k] || m.uniforms?.[k]?.value;
              if (tex) alphaShapedTex.add(tex);
            });
          });
        });
        vrm.scene.traverse((obj: THREE.Object3D) => {
          const mesh = obj as THREE.Mesh;
          if (!mesh.isMesh) return;
          // Per-material kill switch (see HIDE_MATERIALS) for stray costume geometry.
          if (HIDE_MATERIALS.length) {
            const m0 = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material) as THREE.Material | undefined;
            if (m0?.name && HIDE_MATERIALS.includes(m0.name)) {
              mesh.visible = false;
              return;
            }
          }
          mesh.castShadow = true; // still casts: the floor catcher gives her a contact shadow
          // ...but NOTHING on the character RECEIVES shadows. Self-shadowing is what produced
          // the rectangular dark block on the white inner skirt: the coat's sleeve and front
          // panel hang ~0.2m in FRONT of the body, the single shadow-casting key light comes
          // from the upper LEFT, so their shadow lands straight on the white front — and with
          // shadeToony 1 (hard step) it has no soft edge, so it reads as a square that pops in
          // and out as the skirt swings. Chased through GTAO, alphaTest, renderOrder,
          // polygonOffset, z-fighting, physics collision masks, light angle and per-material
          // cast/receive toggles; killing self-shadowing outright is what actually removed it,
          // and on a toon character the MToon shade band already carries the form. Verified on
          // screen by the user before this was made the default.
          mesh.receiveShadow = false;
          // ...except the OUTER coat / skirt panels. Their silhouette lives entirely in the
          // texture's alpha (the printed circuit pattern), so the shadow pass — even with the
          // alpha-aware depth material below — drops chunks of the QUAD onto the white inner
          // skirt: a rectangular dark block that comes and goes as the skirt swings and the
          // pattern's alpha crosses the test. Confirmed by A/B screenshots: with these four
          // excluded the block over the white front disappears completely while every other
          // shadow (arms, hair, collar) stays. They still RECEIVE shadows; they only stop
          // casting, which costs nothing because they hang on the outside of the silhouette.
          {
            const mc = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material) as THREE.Material | undefined;
            if (mc?.name && NO_CAST.includes(mc.name)) mesh.castShadow = false;
          }
          // (depthWrite / alphaTest on the alpha-blended materials are left as
          //  authored — see ALPHA_BLEND_NOTE at the top for what was tried.)
          const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
          mats.forEach((mat: THREE.Material) => {
            const m = mat as any;
            TEX_KEYS.forEach((k) => {
              const tex = m[k] || m.uniforms?.[k]?.value;
              if (tex && "anisotropy" in tex) {
                tex.anisotropy = maxAniso;
                // Repair the invisible RGB BEFORE the mip chain is built below —
                // once the driver has box-filtered the fill colour into the chain
                // there is nothing left to fix. Alpha-TESTED materials only: they
                // are the ones where a mip-averaged alpha can cross the cutoff and
                // paint that fill at full opacity. See dilateAlphaTexture.
                if (m.alphaTest > 0 && !tex.userData?._dilated) {
                  const t0 = performance.now();
                  const n = dilateAlphaTexture(tex, m.alphaTest);
                  // ...then rebuild the chain so the print stays solid when it is
                  // minified. Must come AFTER the dilation: it consumes the
                  // repaired RGB, and it turns generateMipmaps off, so the driver
                  // never sees the texture again.
                  // Only for textures that still get a mip chain. An alpha-shaped
                  // one is not mipped at all now, so there is no chain to correct
                  // and rescaling its alpha would only make it denser than authored.
                  const lv = alphaShapedTex.has(tex) ? 0 : buildCoverageMips(tex);
                  tex.userData = { ...(tex.userData || {}), _dilated: true };
                  if (n || lv) {
                    console.log(
                      `[vrm] ${m.name}: dilated ${n} texels, ${lv} coverage mips`
                      + ` in ${Math.round(performance.now() - t0)}ms`,
                    );
                  }
                }
                // MIP-MAPPED MINIFICATION. The VRM's glTF samplers ask for
                // minFilter LINEAR, and the loader honours it — so every 2048px
                // character texture was minified with a raw bilinear tap and NO
                // mipmaps. Sampling a high-frequency print past Nyquist is
                // textbook aliasing: the thin bright trim lines that edge the
                // outfit came out stair-stepped and crawled with every movement.
                // Mip-mapped minification is the prefilter that fixes it at the
                // source (measured: shimmer spikes 4.8-5.4% -> 2.0-2.6%, i.e.
                // the strobe collapses to the noise floor); the anisotropy set
                // just above is what keeps those mips sharp at grazing angles.
                // magFilter is untouched, so close-ups are unchanged.
                //
                // This was briefly reverted (2026-08-13) and then restored on
                // the user's report that it had visibly fixed the trim-line
                // jaggies. Known trade-off: the skirt INTERIOR print reads a
                // little softer at grazing angles, which is the same prefilter
                // doing its job. `__vrmMips(0)` flips it live for an A/B.
                //
                // NOT for alpha-shaped textures. Eleven of this model's
                // materials are alpha-BLENDED with alphaTest 0, so their shape
                // is the texture's alpha channel and nothing else: Hair01-03,
                // Hairshadow, and the hip panels C01-C07. Box-filtering that
                // alpha down a mip chain averages the opaque strands with the
                // transparent gaps between them, so every mip level is more
                // see-through than the last and the hair dissolves into the
                // background objects behind it — reproduced and confirmed by
                // A/B on the real GPU (logs/render-probe/mipsAB-front-mips*.png:
                // solid twintails at mips=0, washed out at mips=1). That was the
                // 「髪の毛が映らない / 髪の毛を通り越して奥の背景オブジェクトが
                // 映ってる」 report. The trim lines that motivated mips live on
                // the OPAQUE materials (body01, q01-q04, q201-q203), so they
                // keep their prefilter and the jaggies fix is unaffected.
                if (alphaShapedTex.has(tex)) {
                  tex.needsUpdate = true;
                  return;
                }
                if (tex.minFilter === THREE.LinearFilter) {
                  tex.minFilter = THREE.LinearMipmapLinearFilter;
                  tex.generateMipmaps = true;
                }
                tex.needsUpdate = true;
              }
            });
            // SOFT GLOW, NOT A HARD CUT. q301/q302 are the only outfit materials
            // the PMX->VRM conversion gave alphaMode MASK; every sibling panel
            // (q201-q203, C01-C07) came out BLEND. PMX has no alpha modes at all —
            // MMD always blends on the texture's alpha — so MASK is the converter
            // guessing, and guessing against the artwork: these motifs are a crisp
            // shape wrapped in a WIDE soft alpha falloff, drawn to read as a glow.
            // alphaTest 0.5 turns that falloff into a hard-edged slab at whatever
            // radius the alpha happens to cross 0.5, which is what made every speck
            // look outlined rather than lit.
            //
            // Matched to the sibling convention exactly — transparent + alphaTest 0
            // + depthWrite ON — so ALPHA_BLEND_NOTE still holds: each panel is
            // composited once and cannot eat through to the layers behind it. The
            // `m.transparent` rule further down then gives these meshes renderOrder
            // 1 on its own, so they draw after the hair like every other panel.
            //
            // The (Outline) twins are deliberately left alone: they are alpha-tested
            // hulls contributing 0.043% of the frame (measured with outlineWidth 0),
            // and making them transparent would drop them into the same renderOrder
            // as the surface they outline, where the sort between them is arbitrary.
            if (/^q30[12]$/i.test(m.name || "") && m.alphaTest > 0) {
              m.alphaTest = 0;
              m.transparent = true;
              m.depthWrite = true;
              m.needsUpdate = true;
              // NEUTRAL SHADE for the print panels — the "moving cyan" fix.
              //
              // The印 marks are authored near-white (#f2fefd) but carry a faint
              // cyan hue, and the pipeline's shade path amplifies hue three ways:
              // shadeColorFactor is blue-violet [0.88,0.85,0.97], the shade
              // re-multiplies the same cyan-tinted texture (shadeMultiplyTexture
              // = q3.png), and 色トレス (shadeCoord 1) saturates whatever hue the
              // albedo has. Individually mild — but the video grade doubles the
              // result: measured at a frozen mid-swing pose, mark pixels sat at
              // cyan-ness +63.8 pre-grade and +137.7 with the grade, and ONLY a
              // full grade-neutral killed it (sub/mask/hi/cel toggles left
              // 84-98% of it standing). As the arm swings, marks cross the
              // terminator and flip white -> saturated cyan (「腕をいろんな角度に
              // してみて青く映る」).
              //
              // The grade is the look and stays. What gets fixed is the SEED:
              // these two materials render their shade in NEUTRAL GREY — same
              // darkening, no hue for the grade to amplify. Material-scoped on
              // purpose; the rest of the outfit keeps its blue-violet shade.
              m.userData._printPanel = true;
              if (m.shadeColorFactor?.clone) {
                const sc = m.shadeColorFactor;
                const l = 0.299 * sc.r + 0.587 * sc.g + 0.114 * sc.b;
                sc.setRGB(l, l, l);
                // applyLook re-derives from _shadeBase every pass (celBands
                // scales it); seed it grey so later passes stay grey.
                m.userData._shadeBase = sc.clone();
              }
              const sm = m.shadeMultiplyTexture || m.uniforms?.shadeMultiplyTexture?.value;
              const smImg = sm?.image;
              if (smImg?.width) {
                let grey = GREY_SHADE_TEX.get(sm.source ?? smImg);
                if (!grey) {
                  const cv = document.createElement("canvas");
                  cv.width = smImg.width;
                  cv.height = smImg.height;
                  const cx = cv.getContext("2d", { willReadFrequently: true });
                  if (cx) {
                    cx.drawImage(smImg, 0, 0);
                    const id = cx.getImageData(0, 0, cv.width, cv.height);
                    const px = id.data;
                    for (let i = 0; i < px.length; i += 4) {
                      const l = 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
                      px[i] = px[i + 1] = px[i + 2] = l;
                    }
                    cx.putImageData(id, 0, 0);
                    grey = new THREE.Texture(cv);
                    grey.flipY = sm.flipY;
                    grey.colorSpace = sm.colorSpace;
                    grey.wrapS = sm.wrapS;
                    grey.wrapT = sm.wrapT;
                    grey.minFilter = sm.minFilter;
                    grey.magFilter = sm.magFilter;
                    grey.anisotropy = sm.anisotropy;
                    grey.generateMipmaps = sm.generateMipmaps;
                    grey.needsUpdate = true;
                    GREY_SHADE_TEX.set(sm.source ?? smImg, grey);
                  }
                }
                if (grey) {
                  if ("shadeMultiplyTexture" in m && m.shadeMultiplyTexture) m.shadeMultiplyTexture = grey;
                  else if (m.uniforms?.shadeMultiplyTexture) m.uniforms.shadeMultiplyTexture.value = grey;
                }
              }
            }
          });
        });
        // DOUBLE-SIDE the swinging cloth (skirt). MMD renders every face two-sided; the
        // PMX->VRM conversion made materials single-sided (FrontSide), so once the Bullet
        // skirt physics swings a panel far enough to face away, its back face is CULLED and
        // you see straight through it to the dark interior ("スカートの中のものが貫通して見える").
        // Only meshes actually skinned to スカート bones get flipped (skin/face stay FrontSide,
        // else the whole model renders its dark MToon backfaces). The skirt interior reading
        // slightly shadowed is correct — it's the inside of a skirt.
        vrm.scene.traverse((obj: THREE.Object3D) => {
          const mesh = obj as THREE.SkinnedMesh;
          const si = (mesh as any).isSkinnedMesh && mesh.geometry?.attributes?.skinIndex;
          if (!si || !mesh.skeleton) return;
          const bones = mesh.skeleton.bones;
          let skirt = 0; let n = 0;
          for (let i = 0; i < si.count; i += 13) { n += 1; if (/^スカート/.test(bones[si.getX(i)]?.name || "")) skirt += 1; }
          if (n > 0 && skirt / n > 0.3) {
            const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
            mats.forEach((mat: THREE.Material) => {
              // ONLY the plain OPAQUE white skirt cloth (q02) — that's whose culled back
              // faces caused the "black shows through the skirt". Every sheer/patterned
              // overlay must keep the AUTHORED FrontSide rendering: q302 (dark pattern
              // panels) is SHARED between the skirt rim and the arm bell-sleeves, so
              // double-siding it washed the pattern colour out on both ("透明の布の模様の
              // 色がなくなった、腕のほうも").
              const m = mat as any;
              if (m.transparent || !/^q02\b/.test(m.name || "")) return;
              mat.side = THREE.DoubleSide;
              mat.needsUpdate = true;
            });
          }
        });
        generateMissingOutlines(vrm); // msTC Edge stage: hair/face outlines (pre-injection)
        applyLook(vrm);
        loadMstcMasks(vrm); // 配布マスク/純色貼図 (async; re-applies the look when done)
        reviveStrandSprings(vrm); // give the HA/HB/HC strand chains their MMD physics back
        reviveClothSprings(vrm); // ...and the sleeves + breast, which had none at all
        applyBodyColliders(vrm); // PMX rigid bodies -> spring colliders (hair/skirt/sleeves)
        setupGaze(vrm); // eyes: bone applier + a camera-tracking target with saccades
        applyHairSprings(vrm); // twintail spring physics from the 🎨 panel (硬さ/減衰/重力)
        // Phase 2: REAL MMD physics (Bullet/ammo, params baked from the source PMX) for
        // EVERY part the PMX simulated — twintails, bangs, skirt, accessory ties. When it
        // comes up, hand those bones over: remove their springbone joints (double-driving
        // fights). If ammo or the baked asset is unavailable, init() returns false and the
        // springbone setup above stays as the fallback.
        // NOT ON A PHONE. 296 bodies and 381 constraints cost 6.3 ms a frame on a
        // desktop CPU — a third of the 60 fps budget before an A16 is asked for it,
        // and this frame is CPU-bound rather than fill-bound, so no amount of
        // resolution reduction pays for it. The spring bones below are left intact
        // in that case, so the hair still swings; see device-profile.ts.
        const wantHairPhysics = deviceProfile().hairPhysics;
        if (!wantHairPhysics) {
          console.info("[vrm] mobile profile: Bullet hair physics off, spring bones retained");
        }
        (wantHairPhysics ? mmdHairPhysics.init(vrm) : Promise.resolve(false)).then((ok) => {
          if (!ok || disposed) return;
          const sbm = (vrm as unknown as { springBoneManager?: any }).springBoneManager;
          if (sbm?.joints) {
            const owned = mmdHairPhysics.hairBoneNames;
            const doomed = Array.from(sbm.joints).filter((j: any) => owned.has(j.bone?.name || ""));
            for (const j of doomed) {
              if (typeof sbm.deleteJoint === "function") sbm.deleteJoint(j);
              else sbm.joints.delete?.(j);
            }
            console.info(`[vrm] ${doomed.length} spring joints handed to Bullet (${owned.size} physics bones)`);
          }
          (window as any).__vrmHairPhys = mmdHairPhysics;
        });
        vrmMotionPlayer.setVRM(vrm);
        // Console convenience (the 🎨 panel's 髪 sliders do the same, persisted):
        //   __vrmSpring({stiff, drag, grav})  ->  returns the current hair physics.
        (window as any).__vrmSpring = (cfg: { stiff?: number; drag?: number; grav?: number; tipSoft?: number; center?: number } = {}) => {
          if (cfg.stiff != null) LOOK.hairStiff = cfg.stiff;
          if (cfg.drag != null) LOOK.hairDrag = cfg.drag;
          if (cfg.grav != null) LOOK.hairGrav = cfg.grav;
          if (cfg.tipSoft != null) LOOK.hairTipSoft = cfg.tipSoft; // tip stiffness fraction (0=free tips .. 1=rigid rod)
          if (cfg.center != null) LOOK.hairCenter = cfg.center; // 1=head-space sim, 0=world-space (broken export)
          applyLookState();
          saveLook();
          return { stiff: LOOK.hairStiff, drag: LOOK.hairDrag, grav: LOOK.hairGrav, tipSoft: LOOK.hairTipSoft, center: LOOK.hairCenter };
        };
        // Facial expression: bind every face mesh primitive that carries the morph
        // targets (the face is split by material) so the MMD expression system can
        // drive blink + emotion eye-shapes on all of them.
        const faceMeshes: THREE.Mesh[] = [];
        let faceDict: Record<string, number> | null = null;
        vrm.scene.traverse((obj: THREE.Object3D) => {
          const mesh = obj as THREE.Mesh;
          if (mesh.isMesh && mesh.morphTargetDictionary && mesh.morphTargetInfluences) {
            faceMeshes.push(mesh);
            if (!faceDict && (mesh.morphTargetDictionary as any)["まばたき"] != null) {
              faceDict = mesh.morphTargetDictionary;
            }
          }
        });
        // Audition surface for whatever THIS model's face can do. Bound from the
        // meshes themselves rather than from a preset list, because the naming
        // is per-pipeline: Miku carries Japanese PMX morphs, the Avaturn .glb
        // carries ARKit/Oculus ones, and neither knows about the other.
        faceMotion.bind(faceMeshes, gltf.animations || [], vrm.scene);
        disposeFaceMotionPanel = buildFaceMotionPanel();
        // LIP SYNC IS BOUND FOR ANY MODEL WITH A FACE.
        //
        // It used to be bound only inside the `faceDict` branch below, and
        // faceDict is only set when the model has まばたき — a Miku/PMX morph. So
        // a model with a perfectly good mouth got no lip sync at all and simply
        // never opened it while speaking, with nothing logged. Resolution of
        // WHICH morph to drive now happens inside setMeshes against the names
        // each model actually has (あ / jawOpen / aa / ...).
        if (faceMeshes.length) {
          vrmLipSync.setMeshes(faceMeshes, faceDict || {});
          if (vrmLipSync.hasMouth()) {
            console.info(`[VRM] lip-sync morph: ${vrmLipSync.mouthMorph()}`);
          } else {
            console.warn("[VRM] no mouth-open morph found for lip-sync");
          }
        }
        // EXPRESSIONS ARE BOUND FOR ANY MODEL WITH A FACE, for the same reason
        // lip sync above is: this was gated on `faceDict`, which is only set for
        // a model carrying まばたき, so the Avaturn .glb got no expression system
        // at all — and therefore not even a blink (measured: peak eyeBlinkLeft
        // 0.000 over five seconds). Which preset table applies is decided inside
        // setMeshes from the morph names the model actually has, so a model with
        // neither vocabulary still binds and simply finds nothing to drive.
        if (faceMeshes.length) {
          vrmExpression.setMeshes(faceMeshes, faceDict || {});
          // The picker is useful for ANY model now that the rotation is per
          // character — it is how a character's own set gets built in the first
          // place. "" restores the element default; "flex" would left-align the
          // glyph because this button has no flex centering in its own cssText.
          pickBtn.style.display = "";
          // The idle SOURCE is decided by the server, not by the face vocabulary:
          // a character plays the library only if it has a rotation of its own.
          // Start off so the embedded-clip fallback below is set up either way,
          // then switch over asynchronously if the server does have clips for it.
          // (This used to key off `profile === "pmx"`, which hard-wired "ARKit =>
          // no library" and made a Chancelor idle set unreachable by construction.)
          vrmMotionPlayer.idleEnabled = false;
          // The model's own idle takes over from the KAWAII library. Named match
          // first, then the single embedded clip: Chancelor has exactly one, and a
          // future model that ships several should not get an arbitrary pick.
          // Rebuilt only when the MODEL changes. This block runs on every bind,
          // and creating a fresh AnimationMixer each time stacked one per turn
          // (each still holding the skeleton) instead of replacing it.
          if (idleMixerRef.current) {
            idleMixerRef.current.stopAllAction();
            idleMixerRef.current.uncacheRoot(idleMixerRef.current.getRoot() as THREE.Object3D);
          }
          idleMixerRef.current = null;
          if (!vrmMotionPlayer.idleEnabled) {
            const clips = (gltf.animations || []) as THREE.AnimationClip[];
            const own = clips.find((c) => /idle/i.test(c.name))
              || (clips.length === 1 ? clips[0] : undefined);
            if (own) {
              // RETARGET onto the NORMALIZED humanoid bones. The clip animates the
              // raw skeleton, but three-vrm's humanoid.update() copies
              // normalized -> raw every frame, so anything written straight to the
              // raw nodes is erased before it is ever drawn — the clip played and
              // nothing moved.
              const rawToNorm = new Map<string, string>();
              for (const b of Object.keys((vrm.humanoid as any).humanBones || {})) {
                const raw = vrm.humanoid.getRawBoneNode(b as any);
                const norm = vrm.humanoid.getNormalizedBoneNode(b as any);
                if (raw && norm) rawToNorm.set(raw.name, norm.name);
              }
              const retargeted = own.clone();
              let hit = 0;
              retargeted.tracks = retargeted.tracks.filter((t: THREE.KeyframeTrack) => {
                const dot = t.name.indexOf(".");
                const node = t.name.slice(0, dot);
                const prop = t.name.slice(dot);
                const mapped = rawToNorm.get(node);
                if (!mapped) return false;      // non-humanoid track: leave the rig alone
                if (prop !== ".quaternion") return false; // normalized bones take rotation only
                t.name = `${mapped}${prop}`;
                hit += 1;
                return true;
              });
              if (!hit) {
                console.warn("[VRM] embedded idle: no humanoid tracks matched — not played");
              }
              const mixer = new THREE.AnimationMixer(vrm.scene);
              mixer.clipAction(retargeted).play();
              idleMixerRef.current = hit ? mixer : null;
              console.info("[VRM] embedded idle clip:", own.name, own.duration.toFixed(2), "tracks:", hit);
            } else {
              console.info("[VRM] no embedded idle clip — the body holds its last pose");
            }
          }
          // Does the SERVER have an idle rotation for this character? Asked after the
          // embedded fallback is already running, so a slow or failed answer simply
          // leaves the model on its own clip rather than freezing it. A non-empty
          // rotation wins: stop the embedded mixer and let the library drive.
          void (async () => {
            const conf = vrmMotionPlayer.confUid;
            // NO conf, NO library. Without a character we cannot tell whose rotation
            // the default set is, and guessing costs far more in one direction than
            // the other: Chancelor running Miku's anime idles misrepresents a real
            // public figure, while a Miku that waits one model-bind for her clips is
            // merely dull. conf_uid is set from set-model-and-conf, which is the same
            // message that carries the model URL, so it is already there in practice.
            if (!conf) return;
            try {
              const r = await fetch(`/motion/catalog?conf=${encodeURIComponent(conf)}`);
              if (!r.ok) return;
              const d = await r.json();
              // The model may have been swapped while this was in flight.
              if (vrmMotionPlayer.confUid !== conf) return;
              if (!(d.active || []).length) return;
              vrmMotionPlayer.idleEnabled = true;
              if (idleMixerRef.current) {
                idleMixerRef.current.stopAllAction();
                idleMixerRef.current.uncacheRoot(idleMixerRef.current.getRoot() as THREE.Object3D);
                idleMixerRef.current = null;
              }
              console.info(`[VRM] idle library: ${d.active.length} clips for ${d.conf}`
                + (d.inherited ? " (shared default)" : ""));
            } catch { /* offline / no motion routes: keep the embedded clip */ }
          })();
          disposeFacePanel = buildFacePanel(
            faceDict || (faceMeshes[0].morphTargetDictionary as Record<string, number>),
          ); // 🎭 resting-face tuning panel
        }
        if (faceMeshes.length) {
          disposeLookPanel = buildLookPanel(); // 🎨 visual (light/grade/effect) panel
        }
        (window as unknown as { __vrm?: VRM }).__vrm = vrm;
        // Re-frame now that there is a body to measure. Until this existed the
        // framing was solved once at startup against a hard-coded 1.65 m, so a
        // taller model was cropped at the head and a shorter one floated.
        applyFraming();
        // Debug handles for the retarget-accuracy overlay (ground-truth SMPL skeleton).
        (window as any).__three = THREE;
        (window as any).__vrmCam = camera;
        (window as any).__vrmGtao = gtaoPass; // AO tuning from the console (radius etc.)
        // __vrmFace(true) brings the emotion presets back (default face only for now).
        (window as any).__vrmFace = (on = true) => { vrmExpression.emotionEnabled = on; if (!on) vrmExpression.relax(); return on; };
        (window as any).__vrmScene = scene;
        // Toggle the stylised pass (rim light / line art / toon grade) for the
        // loaded model. Reads LOOK through the same override, so nothing is
        // written and Miku's saved look cannot be affected either way.
        (window as any).__vrmPlainStyle = (on?: boolean) => {
          if (typeof on === "boolean") { plainStyle = on; applyLookState(); }
          return { plainStyle };
        };
        // Console helpers: set a resting-face morph (persists), test an emotion,
        // list morphs, reset the resting face.
        (window as any).__vrmFace = (name: string, value = 0.6) => {
          vrmExpression.setBase(name, value);
          return vrmExpression.getBase();
        };
        (window as any).__vrmEmotion = (e: string) => vrmExpression.playEmotion(e);
        (window as any).__vrmFaceList = () => vrmExpression.morphNames();
        (window as any).__vrmFaceReset = () => vrmExpression.resetBase();
        // How much of the resting face survives at full emotion (see emotionBaseKeep):
        // 0 reproduces the old behaviour where an emotion wiped the narrowed-eye base and
        // the pupils blew up; 1 keeps the resting face and lets the emotion only add.
        (window as any).__vrmFaceKeep = (v: number) => {
          (vrmExpression as unknown as { emotionBaseKeep: number }).emotionBaseKeep = v;
          return v;
        };
        // Tune the mouth-open amount live: __vrmMouth(gain, cap).
        (window as any).__vrmMouth = (gain: number, cap?: number) => vrmLipSync.setGain(gain, cap);
        // Closed-mouth あ bias (msTC あ２ approximation): __vrmLipBias(-0.02..-0.2).
        (window as any).__vrmLipBias = (v: number) => { (vrmLipSync as any).idleBias = v; return v; };
        // msTC mouth narrowing (author's value 0.35). 口横縮小 was one of the 31 morphs the
        // PMX->VRM export dropped and has been re-injected, so this drives the real morph.
        (window as any).__vrmLipNarrow = (v: number) => {
          localStorage.setItem("vrmLipNarrow", String(v)); // NOT per-model: the reader in
          // vrm-expression.ts uses the bare key, and this value only applies to the
          // PMX lip morphs, so it is already effectively one model's setting.
          const ex = vrmExpression as unknown as { base: Record<string, number>; dict: Record<string, number> };
          if (ex.dict?.["口横縮小"] != null) ex.base["口横縮小"] = v; // applied on the next update()
          return v;
        };
        (window as any).__vrmLipSync = vrmLipSync; // debug handle (tuning / tests)
        (window as any).__vrmMotion = vrmMotionPlayer; // debug handle (motion tests)
      },
      undefined,
      (e: unknown) => console.error("[VRM] load error:", e),
    );

    // --- Side-by-side REFERENCE figure (the motion model's own default body) ----
    // Draws the raw SMPL joints that ALSO drive Miku (vrmMotionPlayer.getCurrentFrame)
    // as a plain articulated body standing beside her, so the motion model's intended
    // movement can be compared against Miku's retarget. Off by default; toggle with the
    // 🔬 button or window.__vrmCompare(true). Persisted in localStorage.motionCompare.
    // The reference is the ActionPlan paper's SMPL body MESH (SmplMeshFigure, GPU
    // LBS). If its asset fails to load, fall back to the lightweight joint stick
    // figure (SmplReferenceFigure). Both share {group, offsetX, lockLegs, update, dispose}.
    let refFig: SmplMeshFigure | SmplReferenceFigure | null = null;
    let compareOn = false;
    const DEF_TARGET = new THREE.Vector3(0, 0.98, 0);
    const DEF_CAMPOS = new THREE.Vector3(0, 1.05, 2.4);
    const frameBoth = () => {
      const ox = refFig ? refFig.offsetX : 0.9;
      controls.target.set(ox / 2, 1.0, 0);
      camera.position.set(ox / 2, 1.15, 3.9); // pulled back so both bodies + labels fit
      controls.update();
    };
    const cmpBtn = document.createElement("button");
    cmpBtn.textContent = "🔬";
    cmpBtn.title = L("論文のSMPL素体と並べて比較", "Compare against the paper SMPL body");
    cmpBtn.style.cssText = "position:absolute;top:8px;right:80px;z-index:20;width:32px;height:32px;"
      + "border-radius:8px;border:1px solid rgba(255,255,255,.25);background:rgba(20,24,40,.6);"
      + "color:#fff;cursor:pointer;font-size:15px;line-height:1";
    // 論文比較ボタンは非表示（不要との判断）。生成と配線はそのまま残して
    // あるので、DOM に足す一行を戻せば復活する。
    // container.appendChild(autoHide(cmpBtn));
    const setCompare = async (on: boolean) => {
      compareOn = on;
      try { localStorage.setItem("motionCompare", on ? "1" : "0"); } catch { /* ignore */ }
      cmpBtn.style.background = on ? "rgba(80,140,220,.75)" : "rgba(20,24,40,.6)";
      if (on) {
        if (!refFig) {
          const mesh = new SmplMeshFigure();
          const ok = await mesh.load();
          if (disposed || !compareOn) { mesh.dispose(); return; } // toggled off while loading
          refFig = ok ? mesh : new SmplReferenceFigure();
          if (!ok) { mesh.dispose(); console.warn("[compare] SMPL mesh asset missing -> stick figure"); }
          scene.add(refFig.group);
          (window as any).__vrmRef = refFig;
        }
        frameBoth();
      } else {
        if (refFig) { scene.remove(refFig.group); refFig.dispose(); refFig = null; delete (window as any).__vrmRef; }
        controls.target.copy(DEF_TARGET);
        camera.position.copy(DEF_CAMPOS);
        controls.update();
      }
    };
    cmpBtn.addEventListener("click", () => { setCompare(!compareOn); });
    (window as any).__vrmCompare = (on = true) => { setCompare(!!on); };
    if (localStorage.getItem("motionCompare") === "1") setCompare(true);

    // 🎬 IDLE PICKER: audition every standing clip in the pack on the actual avatar and
    // tick the ones that go into the between-turns rotation. Whether a clip survives the
    // planted-foot IK can only really be judged by WATCHING it, so this previews on the
    // real model; the numbers next to each row (foot lift / knee) are just a warning.
    const pickBtn = document.createElement("button");
    pickBtn.textContent = "🎬";
    pickBtn.title = L("アイドルモーションを試聴して選ぶ", "Audition and pick idle motions");
    pickBtn.style.cssText = "position:absolute;top:8px;right:122px;z-index:20;width:32px;height:32px;"
      + "border-radius:8px;border:1px solid rgba(255,255,255,.25);background:rgba(20,24,40,.6);"
      + "color:#fff;cursor:pointer;font-size:15px;line-height:1";
    // The idle picker belongs to the KAWAII library, so it is shown only for the
    // models that actually play those clips. Decided from the MODEL (its morph
    // vocabulary) rather than a character name, so a new PMX model gets it too.
    pickBtn.style.display = "none";
    container.appendChild(autoHide(pickBtn));
    const panel = document.createElement("div");
    // Same class as the others: that is what the "close every panel" pass in each
    // toggle looks for. Without it this one stayed open behind whatever was opened next.
    panel.className = "vrm-tune-panel";
    panel.style.cssText = "position:absolute;top:48px;right:8px;z-index:21;width:330px;max-height:74%;"
      + "display:none;flex-direction:column;border-radius:10px;border:1px solid rgba(255,255,255,.18);"
      + "background:rgba(14,17,28,.94);color:#e8ecf5;font:12px/1.45 system-ui,sans-serif;"
      + "box-shadow:0 8px 30px rgba(0,0,0,.5);overflow:hidden";
    container.appendChild(panel);
    let picker: { load: () => void } | null = null;
    const buildPicker = () => {
      const head = document.createElement("div");
      head.style.cssText = "padding:8px 10px;border-bottom:1px solid rgba(255,255,255,.12);"
        + "display:flex;gap:6px;align-items:center;flex:0 0 auto";
      const info = document.createElement("span");
      info.style.cssText = "flex:1;opacity:.85";
      const saveBtn = document.createElement("button");
      SET(saveBtn, "保存", "Save");
      saveBtn.style.cssText = "padding:3px 10px;border-radius:6px;border:1px solid rgba(255,255,255,.25);"
        + "background:rgba(80,140,220,.85);color:#fff;cursor:pointer";
      const stopBtn = document.createElement("button");
      SET(stopBtn, "停止", "Stop");
      stopBtn.style.cssText = "padding:3px 8px;border-radius:6px;border:1px solid rgba(255,255,255,.25);"
        + "background:rgba(60,64,80,.9);color:#fff;cursor:pointer";
      head.append(info, stopBtn, saveBtn);
      const filter = document.createElement("input");
      filter.placeholder = L("絞り込み (例: look, dance, pose)", "Filter (e.g. look, dance, pose)");
      filter.style.cssText = "margin:8px 10px 4px;padding:5px 8px;border-radius:6px;"
        + "border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.06);color:#e8ecf5;flex:0 0 auto";
      const list = document.createElement("div");
      list.style.cssText = "overflow-y:auto;padding:4px 6px 10px;flex:1 1 auto";
      panel.append(head, filter, list);

      let clips: any[] = [];
      const chosen = new Set<string>();
      const render = () => {
        const q = filter.value.trim().toLowerCase();
        list.innerHTML = "";
        let shown = 0;
        for (const c of clips) {
          if (q && !c.key.includes(q)) continue;
          shown += 1;
          const row = document.createElement("div");
          row.style.cssText = "display:flex;gap:6px;align-items:center;padding:3px 4px;border-radius:6px";
          const cb = document.createElement("input");
          cb.type = "checkbox"; cb.checked = chosen.has(c.key); cb.style.cssText = "cursor:pointer";
          cb.addEventListener("change", () => {
            if (cb.checked) chosen.add(c.key); else chosen.delete(c.key);
            info.textContent = L(`${chosen.size} 個を採用 / 全 ${clips.length}`, `${chosen.size} in use / ${clips.length} total`);
          });
          const play = document.createElement("button");
          play.textContent = "▶";
          play.title = L("このモーションを今すぐ再生", "Play this motion now");
          play.style.cssText = "width:24px;padding:1px 0;border-radius:5px;cursor:pointer;"
            + "border:1px solid rgba(255,255,255,.22);background:rgba(255,255,255,.08);color:#fff";
          play.addEventListener("click", async () => {
            try {
              const r = await fetch(`/motion/idle?key=${encodeURIComponent(c.key)}`);
              if (r.status !== 200) return;
              const d = await r.json();
              vrmMotionPlayer.idleClips = false;   // hold the rotation while auditioning
              vrmMotionPlayer.setClip(d, null);
              row.style.background = "rgba(80,140,220,.28)";
              setTimeout(() => { row.style.background = ""; }, Math.min(12000, c.secs * 1000));
            } catch { /* ignore */ }
          });
          const name = document.createElement("span");
          name.textContent = c.key.replace(/^idle\d+_/, "");
          name.style.cssText = "flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis";
          if (c.verified) name.style.color = "#8fe3a6";      // checked on the avatar
          else if (c.warn) name.style.color = "#e6b768";      // moves the feet a lot
          const meta = document.createElement("span");
          meta.textContent = `${c.secs}s`;
          meta.title = `foot lift ${c.footLift ?? "?"}m / knee ${c.kneeMax ?? "?"}°`
            + (c.warn ? L(" — 脚が大きく動くので崩れやすい", " — large leg travel, prone to breaking") : "");
          meta.style.cssText = "opacity:.6;font-variant-numeric:tabular-nums";
          row.append(cb, play, name, meta);
          list.appendChild(row);
        }
        if (!shown) SET(list, "該当なし", "No matches");
      };
      filter.addEventListener("input", render);
      stopBtn.addEventListener("click", () => {
        vrmMotionPlayer.stop();
        vrmMotionPlayer.idleClips = true; // resume the normal rotation
      });
      saveBtn.addEventListener("click", async () => {
        saveBtn.disabled = true; SET(saveBtn, "保存中", "Saving");
        try {
          // conf: saves THIS character's rotation. Without it the save would edit
          // the shared default and hand Miku's set to Chancelor as well.
          const r = await fetch("/motion/idle-set", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ active: [...chosen], conf: vrmMotionPlayer.confUid || undefined }),
          });
          const d = await r.json();
          info.textContent = L(`保存しました (${(d.active || []).length} 個)`, `Saved (${(d.active || []).length})`);
          vrmMotionPlayer.idleClips = true;
          // A character that had no set now has one, so the library takes over from
          // whatever it was doing (embedded clip / held pose) without a reload —
          // and un-ticking everything hands it straight back.
          vrmMotionPlayer.idleEnabled = (d.active || []).length > 0;
        } catch { SET(info, "保存に失敗", "Save failed"); }
        saveBtn.disabled = false; SET(saveBtn, "保存", "Save");
      });
      return {
        load: async () => {
          SET(info, "読み込み中...", "Loading...");
          try {
            const conf = vrmMotionPlayer.confUid;
            const r = await fetch(`/motion/catalog${conf ? `?conf=${encodeURIComponent(conf)}` : ""}`);
            const d = await r.json();
            clips = d.clips || [];
            chosen.clear();
            for (const k of d.active || []) chosen.add(k);
            // `inherited` = these ticks are the shared default, not this character's
            // own set. Saying so matters: the first save FORKS it, and the default
            // stops following this character from then on.
            const scope = d.inherited
              ? L(" — 共有の既定値", " — shared default")
              : L(` — ${d.conf} 専用`, ` — ${d.conf} only`);
            info.textContent = L(`${chosen.size} 個を採用 / 全 ${clips.length}`, `${chosen.size} in use / ${clips.length} total`) + scope;
            render();
          } catch { SET(info, "カタログを取得できません", "Could not load the catalogue"); }
        },
      };
    };
    pickBtn.addEventListener("click", () => {
      const open = panel.style.display === "none";
      // Close every other tuning panel first, exactly as the 🎭 and 🎨
      // toggles do. This one used to just toggle itself, so it appeared
      // on top of an already-open panel instead of replacing it.
      retranslate(container);
        container.querySelectorAll(".vrm-tune-panel").forEach((p) => {
        (p as HTMLElement).style.display = "none";
      });
      panel.style.display = open ? "flex" : "none";
      if (open) {
        if (!picker) picker = buildPicker();
        picker.load();
      } else {
        vrmMotionPlayer.idleClips = true; // leaving the picker resumes the rotation
      }
    });

    const animate = () => {
      raf = requestAnimationFrame(animate);
      // Single-step gate for measurement. A no-op (one boolean) unless
      // __AI_DEBUG__.stepFrame() is driving, in which case the loop must not draw
      // an unrequested frame — the temporal resolve accumulates, so a spare
      // redraw of the same pose is not free the way it used to be.
      if (!aiDebug.beginFrame()) return;
      // Per-FRAME draw stats. three resets info on every render() call, so with
      // a multi-pass composer the numbers otherwise only describe the final
      // fullscreen quad — read back via __space().render.
      renderer.info.reset();
      // Clamp the timestep: a frame drop (large dt) would blow up the spring-bone
      // integration (hair/skirt exploding on a big gesture). Cap at ~1/30 s.
      // Single point of control for the whole frame: freeze/seek for automated
      // capture happens here, so no system needs its own debug flag.
      // RAW delta: the long-stall clamp moved inside step(), so that frameStats()
      // can see the true frame time instead of a value already flattened to 1/30.
      const dt = aiDebug.step(clock.getDelta());
      controls.update(); // orbit damping
      // Head-coupled perspective. MUST run after controls.update(), which writes
      // the camera transform from its own orbit state and would otherwise undo
      // this. When it returns false (disabled, or the eye is behind the glass)
      // the stock symmetric projection controls left in place is still correct.
      viewerTransform.update(dt);
      // Slow parallax. Without head tracking this is the only thing making the
      // near and far structure separate, and separation is what "3D" is.
      framing.update(dt);
      applyOffAxis(camera, currentPanel(container));
      // TAA jitter goes HERE and nowhere else. off-axis.ts has just overwritten
      // projectionMatrix via makePerspective(); anything applied earlier is gone.
      // Everything downstream — the normal pre-pass, GTAO, the composer — reads
      // the camera after this point, so they all share one jitter and their
      // buffers stay mutually consistent for free.
      // Jitter in units of the INTERNAL raster — with taaUpscale active the
      // scene renders fewer pixels, and the sub-pixel offsets must walk that
      // grid, not the output one.
      taa.apply(
        camera,
        Math.floor(container.clientWidth * pixelRatio * upscaleNow()),
        Math.floor(container.clientHeight * pixelRatio * upscaleNow()),
        dt,
      );
      // __space({ taa: 0 }) has to be a real escape hatch: the resolve is
      // skipped and the bridge feeds the scene composer's output straight into
      // the post chain, untouched. The resize() puts the internal scale back to
      // 1 in the same breath — a low-res render with no accumulator to
      // reconstruct it would just be a blur.
      if (taaWasOn !== taa.enabled()) {
        taaWasOn = taa.enabled();
        taa.reset();
        resize();
      }
      if (vrm) {
        // EMBEDDED IDLE (Chancelor's Female_Animation_Idle). It is a glTF clip of
        // bone ROTATIONS, so it cannot go through vrmMotionPlayer (which consumes
        // SMPL joint POSITIONS) — it needs its own mixer. The two must never run
        // together: they write the same bones, so whichever runs last wins and the
        // result is a fight, not a blend. The gesture player has priority; the
        // mixer fills the gaps.
        // Last frame's turn toward the viewer comes off first, so every motion layer
        // below starts from its own pose and the turn is re-applied once, at the
        // same strength whatever is driving the body (look-at-viewer _undo).
        undoViewerLookAt();
        // Someone is being attended to: quieter idles, and less of their head motion.
        vrmMotionPlayer.attention = headFollow.weight;
        vrmMotionPlayer.update(dt); // pose from the current gesture frame
        // The mixer runs AFTER the player when the player is not driving, so the
        // held pose it would otherwise keep writing does not overwrite the idle.
        if (idleMixerRef.current && !vrmMotionPlayer.bodyBusy) {
          idleMixerRef.current.update(dt);
        }
        if (refFig) refFig.update(vrmMotionPlayer.getCurrentFrame()); // sync the reference body
        vrmMotionPlayer.updateIdle(dt); // breathing/sway when no gesture is playing
        // Re-drive the MMD append (付与親) chains the VRM export dropped — this rig skins
        // its legs to 足D/ひざD/足首D, not to the bones the humanoid maps. Must run before
        // the spring pass so the skirt simulates against the legs' real pose.
        vrmMotionPlayer.syncAppendBones();
        // Her own slow travel in depth. Applied to the ROOT, after the pose is
        // written, so it composes with whatever the gesture is doing instead of
        // being overwritten by it. With the drawn room gone this is the depth
        // cue a flat image cannot reproduce (see body-depth.ts).
        bodyDepth.update(dt, vrm.scene);
        // Head/upper-body turn toward whoever is watching. Additive on top of
        // the gesture pose, and BEFORE vrm.update() so humanoid.update() pushes
        // it to the raw bones and the eye look-at is computed against the head's
        // new orientation rather than a stale one. The target is the person the
        // PROTO's camera picked (the closest one); the turn is weighted by the
        // follow hand-over, so with nobody tracked she faces front as before.
        const nod = updateAttend(vrm);
        applyViewerLookAt(vrm, headPoint, dt, headFollow.weight, nod);
        updateGaze(dt, vrm); // move the gaze target; vrm.update() applies it to the eyes
        // Sub-step the spring/constraint update at a FIXED ~240Hz. With a variable/coarse
        // frame timestep (e.g. 60fps -> 16ms) the spring integration oscillates = the hair
        // "プルプル横揺れ". three-vrm applies dragForce PER sub-step, so finer sub-steps give
        // BOTH more numerical stability AND more effective damping per frame -> this is what
        // actually kills the sideways vibration (stiffness only trades ring-freq for sweep;
        // it can't lower the vibration amplitude — damping/sub-steps can). Was 120Hz (subN=2
        // at 60fps); 240Hz -> subN=4, which roughly doubles the per-frame damping.
        const subN = Math.min(6, Math.max(1, Math.round(dt * 240)));
        // three-vrm applies dragForce PER update() call, so N sub-steps multiply the
        // damping N times — hair could never carry momentum ("MMDのように弾まない").
        // Convert each joint's authored-units target to a per-sub-step equivalent.
        const sbm = (vrm as unknown as { springBoneManager?: any }).springBoneManager;
        if (sbm?.joints) {
          for (const j of sbm.joints) {
            const t = (j as any).__dragTarget;
            if (t != null) j.settings.dragForce = 1 - Math.pow(1 - t, 1 / subN);
          }
        }
        // DECOMPOSED vrm.update(). Only the spring integrator needs sub-stepping;
        // everything else vrm.update() calls is a direct solve that produces the
        // same answer however many times it is asked, so running it 4-6x a frame
        // was pure repetition — humanoid.update() over 55 bones, the look-at
        // solve, expressionManager.update() (which allocates a weight object per
        // expression on every call), and a loop over all 41 materials.
        //
        // Same call order as three-vrm's own update(): humanoid -> lookAt ->
        // expressions -> constraints -> springs -> materials. Sub-steps see a
        // pose that does not change between them, so the moved calls would have
        // recomputed identical values; the material UV animation advances by
        // delta, so one call at dt totals exactly what N calls at dt/N did.
        const vc = vrm as unknown as {
          humanoid?: { update(): void };
          lookAt?: { update(d: number): void };
          expressionManager?: { update(): void };
          nodeConstraintManager?: { update(): void };
          materials?: { update?(d: number): void }[];
        };
        vc.humanoid?.update();
        vc.lookAt?.update(dt);
        vc.expressionManager?.update();
        vc.nodeConstraintManager?.update();
        if (sbm) for (let i = 0; i < subN; i += 1) sbm.update(dt / subN);
        if (vc.materials) for (const m of vc.materials) m.update?.(dt);
        // Real MMD hair physics (Bullet). Needs fresh world matrices: the pose above only
        // set local transforms. Steps the sim in PMX space and writes the twintail bones.
        if (mmdHairPhysics.enabled) {
          vrm.scene.updateMatrixWorld(true);
          mmdHairPhysics.step(dt);
        }
        vrmExpression.update(); // AFTER vrm.update so it overrides the expressionManager
        vrmLipSync.update(); // mouth-open morph from speech volume (no-op when silent)
        aiFx.update(dt, vrmLipSync.voice); // glow/rim follow state + voice (opt-in models only)
        // LAST, so the 😀 audition panel wins while it is open: on Miku the two
        // above write every frame and would otherwise erase whatever is being
        // previewed. No-op when nothing is held or sweeping.
        faceMotion.update(dt);
      }
      neuralSpace.update(dt); // environment: floor pulse, drifting geometry, particles
      // THE frame's world-matrix walk (see scene.matrixWorldAutoUpdate above).
      // Everything that writes a transform — the pose, body depth, the camera
      // drift, every environment module — has run by now, and nothing below
      // moves anything, so every render after this point sees the same matrices
      // the six separate walks used to produce.
      scene.updateMatrixWorld();
      gradePass.uniforms.uTime.value += dt; // animate film grain
      if (bokehPass.enabled) {
        (bokehPass.uniforms as any).focus.value = camera.position.distanceTo(controls.target);
      }
      // Line pre-pass: scene normals + depth for the ink-line detectors.
      if (LOOK.lineInt + LOOK.lineNInt + LOOK.lineZInt > 0.001) {
        const bg = scene.background;
        scene.background = null;
        scene.overrideMaterial = normalMat;
        // Keep the environment OUT of the normal/depth buffer the edge detectors
        // read: the ink line is a character look, and left in, it draws a toon
        // outline around every neon ring and grid line. The shadow-catcher plane
        // deliberately stays on layer 0 so it DOES take part — that's what gives
        // her feet a contact line against the floor.
        camera.layers.disable(SPACE_LAYER);
        renderer.setRenderTarget(normalRT);
        renderer.render(scene, camera);
        renderer.setRenderTarget(null);
        camera.layers.enable(SPACE_LAYER);
        scene.overrideMaterial = null;
        scene.background = bg;
      }
      // PostPointLight analogue: keep the point-light position in view space.
      if (LOOK.pointInt > 0.001 && subUList.length) {
        pointViewPos
          .set(LOOK.pointX, LOOK.pointY, LOOK.pointZ)
          .applyMatrix4(camera.matrixWorldInverse);
        for (let i = 0; i < subUList.length; i += 1) {
          subUList[i].uPointPos.value.copy(pointViewPos);
        }
      }
      // Motion vectors for this frame, rendered just before the resolve
      // consumes them. Skipped while the history is latched (the hold path
      // reads nothing) and while TAA is off (the resolve that samples them is
      // skipped too, so the buffer would be dead weight).
      if (taa.enabled() && !taa.holding()) taaVelocity.render(renderer, scene, camera);
      // Shadow map: rendered by the NEXT render() call and only that one (see
      // shadowMap.autoUpdate at setup). Armed here — after the velocity and
      // normal pre-passes, which do not sample it — so the one map that gets
      // built is the one the visible pass uses, at this frame's pose.
      renderer.shadowMap.needsUpdate = true;
      sceneComposer.render();
      // The seam: scene side -> temporal resolve -> post side. With TAA off the
      // resolve stands down and the bridge carries the scene output through
      // untouched. readBuffer is re-read every frame because the composer's
      // internal swaps change which target holds the final image.
      const sceneOut = (sceneComposer as unknown as {
        readBuffer: THREE.WebGLRenderTarget
      }).readBuffer.texture;
      if (taa.enabled()) taaResolve.render(renderer, camera, sceneOut);
      taaResolve.updateCopy(taa.enabled() ? taaResolve.output : sceneOut, taa.enabled());
      postComposer.render();
      // Bone snapshot MUST come after the scene render: Skeleton.update() runs
      // 2-5x inside it with identical values, and "previous frame" has to mean
      // the pose this frame was drawn with, captured before the next
      // vrm.update() moves the bones.
      if (taa.enabled()) taaVelocity.snapshot();
      aiDebug.endFrame(); // release a single-step waiter, once the frame exists
    };
    animate();
    // Black box for the unreproducible under-skirt flash: rolling frame ring,
    // "9" uploads it. Its own rAF, registered after animate()'s, so it copies
    // each frame AFTER it is drawn. __flashRec(0) disables.
    flashRecorder.start();
    (window as any).__flashRec = (on: number | boolean = 1) => flashRecorder.setEnabled(!!on);

    // Resize on CONTAINER changes (not just window) so toggling the sidebar —
    // which resizes the canvas without a window resize event — doesn't stretch
    // the avatar horizontally.
    const resize = () => {
      const w = Math.max(container.clientWidth, 1);
      const h = Math.max(container.clientHeight, 1);
      // Re-resolve every time: dragging the window to the PROTO panel changes
      // the scale factor, and keeping the old one is what made it look soft.
      pixelRatio = resolvePixelRatio(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setPixelRatio(pixelRatio);
      renderer.setSize(w, h);
      // Two composers, two resolutions: the scene side carries the TAAU scale,
      // the post side is always full output resolution.
      const s = upscaleNow();
      sceneComposer.setPixelRatio(pixelRatio * s);
      sceneComposer.setSize(w, h);
      postComposer.setPixelRatio(pixelRatio);
      postComposer.setSize(w, h);
      bloomPass.setSize(w, h);
      diffPass.setSize(w, h);
      // WebGLRenderTarget.setSize doesn't resize an attached DepthTexture —
      // update it (drawing-buffer size) or the line pass samples a stale size.
      const dw = Math.max(1, Math.floor(w * pixelRatio));
      const dh = Math.max(1, Math.floor(h * pixelRatio));
      // Everything the SCENE side reads or writes lives at the internal scale.
      const iw = Math.max(1, Math.floor(w * pixelRatio * s));
      const ih = Math.max(1, Math.floor(h * pixelRatio * s));
      normalRT.setSize(iw, ih);
      if (normalRT.depthTexture) {
        (normalRT.depthTexture as any).image.width = iw;
        (normalRT.depthTexture as any).image.height = ih;
      }
      linePass.uniforms.uTexel.value.set(1 / iw, 1 / ih);
      taaVelocity.setSize(iw, ih);
      taaResolve.setSize(dw, dh, iw, ih); // history out, current-frame taps in
    };
    const ro = new ResizeObserver(resize);
    ro.observe(container);
    // A ResizeObserver alone misses the move that matters most: 1080p@100% ->
    // 4K@200% is the same size in CSS pixels, so nothing fires while the panel
    // gains 4x the physical pixels.
    const unwatchDpr = watchDevicePixelRatio(resize);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      unwatchDpr();
      controls.dispose();
      disposeFacePanel?.();
      disposeLookPanel?.();
      disposeFaceMotionPanel?.();
      faceMotion.clear();
      mmdHairPhysics.dispose();
      delete (window as any).__vrmHairPhys;
      delete (window as any).__vrmMask;
      delete (window as any).__vrmCost;
      delete (window as any).__vrmForceHi;
      delete (window as any).__vrmDebug;
      delete (window as any).__vrmPreset;
      delete (window as any).__vrmStage;
      delete (window as any).__vrmLook;
      delete (window as any).__vrmFace;
      delete (window as any).__vrmEmotion;
      delete (window as any).__vrmFaceList;
      delete (window as any).__vrmFaceReset;
      delete (window as any).__vrmMouth;
      delete (window as any).__vrmLipBias;
      delete (window as any).__vrmLipNarrow;
      delete (window as any).__vrmLipSync;
      delete (window as any).__vrmMotion;
      delete (window as any).__vrmSpring;
      delete (window as any).__vrmCompare;
      delete (window as any).__vrmRef;
      if (refFig) { scene.remove(refFig.group); refFig.dispose(); refFig = null; }
      cmpBtn.remove();
      if (vrm) {
        vrmMotionPlayer.clearVRM(vrm);
        vrmExpression.clearMeshes();
        vrmLipSync.clearMeshes();
        aiFx.unbind();
        if ((window as unknown as { __vrm?: VRM }).__vrm === vrm) {
          delete (window as unknown as { __vrm?: VRM }).__vrm;
        }
        VRMUtils.deepDispose(vrm.scene);
      }
      neuralSpace.dispose();
      delete (window as any).__space;
      delete (window as any).__spaceRebuild;
      delete (window as any).__spaceQuality;
      delete (window as any).__spaceState;
      delete (window as any).__spaceDpr;
      delete (window as any).__sharpTest;
      delete (window as any).__protoMode;
      headTracker.stop();
      hidePanelCalibration();
      delete (window as any).__headTracker;
      hideSpaceHud();
      hideHeadPreview();
      delete (window as any).__headPreview;
      delete (window as any).__lookAt;
      delete (window as any).__headCamera;
      delete (window as any).__headDiag;
      delete (window as any).__headSetup;
      delete (window as any).__headCameras;
      delete (window as any).__spaceHud;
      delete (window as any).__spacePixels;
      delete (window as any).__viewerPose;
      delete (window as any).__offAxis;
      delete (window as any).__follow;
      window.removeEventListener("kehai-turn", onTurn);
      window.removeEventListener("kehai-listen", onListen);
      delete (window as any).__attend;
      delete (window as any).__notice;
      delete (window as any).__headIngest;
      delete (window as any).__headCalibrate;
      delete (window as any).__protoCalibrate;
      sceneComposer.dispose();
      postComposer.dispose();
      // EffectComposer.dispose() frees its own two targets and nothing else — it
      // never calls dispose() on the passes. The history buffers are full-size
      // RGBA16F, so leaving them behind leaks tens of MB every time this effect
      // re-runs, which it does on every model switch.
      taaResolve.dispose();
      taaVelocity.dispose();
      bgTex.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [url]);

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "100%", overflow: "hidden", position: "relative" }}
    />
  );
});

VRMViewer.displayName = "VRMViewer";
