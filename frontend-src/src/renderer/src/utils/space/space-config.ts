/**
 * Tunables for the Spatial Presence Layer.
 *
 * Deliberately a SEPARATE namespace from the viewer's `LOOK` object and its
 * `vrmLook8` / `LOOK_BUILD` storage. Adding fields to `LOOK` would force a build
 * bump, and a build bump throws away every look tweak the user has dialled in
 * (the saved blob is discarded wholesale when `_build` mismatches). Keeping the
 * space settings in their own key means the two can never reset each other.
 */

import { deviceProfile } from "./device-profile";

export type Quality = "low" | "medium" | "high";

export const SPACE = {
  /** Master switch. */
  enabled: true,

  // --- Scene ------------------------------------------------------------------
  /**
   * Which environment to build. "dataflow" (2026-10-22): no text and no drawn
   * structure — a projector base at her feet and a cloud of data bits that turns
   * around her, streams into her while she thinks and out of her while she
   * talks (data-flow.ts / pedestal.ts). "console": the earlier holographic
   * console (HUD rings, text panels, towers, traffic). Rebuilt on change.
   */
  scene: "dataflow" as "dataflow" | "console",
  /** Data bits: opacity scale (0 = off), count (rebuild), size, colours.
   *  Off by default (2026-10-22, "チリいらない"): beside the 0/1 rain the square
   *  bits read as dust. The 0/1 flips and the base carry the state instead. */
  flow: 0,
  flowCount: 1600,
  flowSize: 22,
  flowColor: 0x0577b8,
  /** What a bit fades toward behind her. Near the field = see-through on PROTO. */
  flowFarColor: 0x7cc4e8,
  /** The point the bits stream into / out of: the emblem on her chest, just in
   *  FRONT of it — a core inside the body hid every stream behind her torso. */
  flowCoreY: 1.36,
  flowCoreZ: 0.2,
  /** 0/1 columns falling behind her (binary-rain.ts): opacity (0 = off), layout
   *  (columns x rows: rebuild), fall speed m/s, glyph height m, colour. */
  rain: 0.85,
  rainColumns: 20, // 60 -> 30 -> 20 (2026-10-22, "数字の数が多すぎる" / "まだ少しおおい")
  // 18 rows over 3.2 m (0.18 m apart) at 0.07 m/s: a given spot on screen sees a
  // new digit arrive every 2-5 s. At 26 rows / 0.18 m/s it was 1-2 a second, which
  // read as the digits "switching 0/1 too fast" even with no switching at all.
  rainRows: 9,
  rainSpeed: 0.07,
  rainGlyphM: 0.09,
  /** Depth band of the columns behind her (m, rebuild) and how much the far ones
   *  blur (mip bias at the far end). */
  rainNear: 2,
  rainFar: 14,
  rainBlur: 2.5,
  /** Foreground 0/1 (binary-rain.ts `fore`): columns at the frame edges in front
   *  of her, defocused. Count (rebuild), opacity share (0 = off), blur (mip bias). */
  rainFgColumns: 2,
  rainFgOpacity: 0.35,
  rainFgBlur: 3.5,
  rainColor: 0x0568a8,
  /** Bright band travelling down each column (0 = off: one steady level). */
  rainBand: 0,
  /** Cyan veil just behind her that separates her from the far field
   *  (backlight.ts): opacity at the centre (0 = off) and colour. */
  backlight: 0, // off (2026-10-22, "もやはいらない"); 0.22 was the tuned look
  backlightColor: 0x4fc3ea,
  /** Her reflection in the floor, fading away from her feet (floor-reflection.ts).
   *  Opacity (0 = off). Costs one extra scene render at 1/3 resolution. */
  // Her reflection stays; it is the DRAWN floor (rings, shadow patch) that flattened
  // the scene — on PROTO the real box floor is right there, and a reflection alone
  // reads as lying on it.
  reflection: 0.22,
  /** Projector base under her feet (0 = off) and its colour. */
  pedestal: 0, // off (2026-10-22): a drawn floor ring flattened the depth
  pedestalColor: 0x0796d0,
  /**
   * Field colour for the dataflow scene. On the PROTO panel white is see-through,
   * so this is as close to white as the grade allows without tinting the box.
   */
  flowBackdrop: 0xf1f2f6,

  /**
   * ONE knob over how much light the environment puts on the panel.
   *
   * On this display the field must stay black. Black is not a colour here — it
   * is the state where the panel reads as "nothing is there" and the character
   * floats. The moment a broad area is lifted off black, the viewer sees the
   * DISPLAY instead of a hologram, and no amount of perspective work recovers
   * that. So the environment's whole budget is: thin luminous marks on black.
   */
  emissiveGain: 1.0,

  /**
   * Backdrop colour for the whole aperture.
   *
   * White, because on this panel a coloured field stops reading as transparent.
   * Applied in EVERY mode, not just on the unit: the viewer otherwise paints
   * LOOK's navy radial gradient across the container, which is what "背景紺色
   * 主体になってる" was. Set from here rather than by editing LOOK so the
   * character's tuned look is untouched; set to "" to hand the backdrop back.
   */
  backdrop: "#ffffff",
  /**
   * Colour of the in-scene backdrop sphere.
   *
   * Leans very slightly green because the character's grade pass pushes the
   * frame toward magenta, so a pure-white sphere comes out faintly pink. The
   * compensation belongs here and not in the grade: the grade is part of her
   * look. Adjust with __space({ backdropColor: 0xffffff }) against the real
   * panel — this is a guess at the correction, not a measurement.
   */
  // A near-white field cannot make a thin cyan line look lit: glow is perceived as a
  // LUMINANCE DIFFERENCE against its surround, and against white there is no headroom
  // left to be brighter. A softly tinted field gives the HUD somewhere to glow from.
  // Note the grade amplifies this hard (gSat 1.32, gContrast 1.25) — feed it something
  // much greyer than the colour you actually want. 0x8d89a8 came out a purple box.
  backdropColor: 0xc4c2cc,

  // --- Palette --------------------------------------------------------------
  /**
   * The field is white and the LINES carry the colour — the opposite of the
   * usual dark-scene-with-neon instinct, and the way round this panel wants.
   *
   * Cyan throughout, on white. Note the value has to be reasonably DARK to read
   * against a white field — a pale cyan disappears completely, which is what the
   * first attempt at this did. The accents are simply a darker and a lighter cyan
   * so the structure has depth of tone without introducing a second hue.
   */
  lineColor: 0x1fa8d8,
  accentA: 0x0d7fae,
  accentB: 0x63d4f0,
  /** Fraction of lines that take an accent colour instead of the base. */
  accentMix: 0.22,

  // --- Ground ---------------------------------------------------------------
  /**
   * Grid brightness. Drawn as GL LINES, never as a filled plane: a filled
   * surface — even a dim one — is a lit area and gives the panel away.
   */
  // 0 (2026-07-29): the physical boxes sitting behind the panel show through
   // exactly where the ground would be, so a drawn floor lands on top of real
   // objects and the two read as one confused surface. Removed entirely rather
   // than dimmed — depth now comes from cues that do not need a ground plane:
   // occlusion (traffic crossing her silhouette), camera drift parallax, the
   // contact shadow for grounding, and haze/DoF for distance. Restore with
   // __vrmSpace({ grid: 1 }) if the unit is ever moved somewhere clear.
  grid: 0,
  /**
   * Grid cell size, metres.
   *
   * 0.5 rather than 1.0 because the frame is portrait and very narrow: at her
   * depth the visible width is only about ±0.5 m, so a 1 m grid put every
   * longitudinal line off-screen and the floor showed as rows with nothing
   * running away from the viewer.
   */
  gridSpacing: 0.5,
  /** Rows of the floor grid running across; they carry most of the depth read. */
  gridRows: 26,
  /** Half-width of the floor grid in metres. */
  gridHalfWidth: 3,
  /** Depth the grid geometry runs back to, metres. */
  gridDepth: 34,
  /**
   * Distance from the CAMERA at which the floor has completely faded out — the
   * visible ground boundary.
   *
   * This is a different thing from the horizon and is the one that can actually
   * be moved. The horizon is where the eye's level line falls, so with her whole
   * body framed it is pinned by the camera height (~75% of frame at 1.25 m eye)
   * and can only be lowered by shrinking her. Where the FLOOR stops is free: end
   * it short of the horizon and the boundary the eye reads drops with it, while
   * the camera and her size stay put.
   *
   * Measured from the CAMERA, which sits ~3.2 m back — so 6 m leaves roughly
   * 2.8 m of floor behind her and puts the boundary near a third of the frame.
   */
  floorFadeM: 6,

  // --- Structure ------------------------------------------------------------
  /**
   * ALL room geometry is off (2026-07-29). Floor, side/ceiling rails and the
   * receding portal rings were each an attempt to DRAW a space around her, and
   * every one of them made the panel read as a picture of a room rather than a
   * window into one: a drawn wall is a flat pattern that contradicts the real
   * depth behind the glass, and it fights the physical objects the viewer can
   * actually see back there.
   *
   * The space is no longer drawn at all. What remains are only cues that a flat
   * still image physically cannot fake — occlusion across her silhouette, and
   * parallax from camera drift — plus her own movement in depth. Anything that
   * builds a virtual room belongs off.
   */
  portals: 0,
  portalCount: 9,
  /** Ceiling and side rails — the L shapes down each side. Off, see above. */
  rails: 0,

  // --- Motion (the depth cues that work without a camera) -------------------
  /**
   * Wireframe shapes travelling from far to near, PAST her and out through the
   * front. Their value is occlusion: a line crossing her body is the single
   * strongest depth cue available on a static flat panel, far stronger than any
   * amount of perspective in a still scene.
   */
  traffic: 1.0,
  // 7 -> 11 (2026-07-29): with the floor gone, occlusion is carrying the depth
  // read almost alone, and how often something crosses her silhouette is what
  // sets how continuously the space reads as deep. More shapes, not brighter
  // ones — brightness would flatten the panel, crossings do not.
  trafficCount: 7,
  /** Metres per second the traffic approaches. */
  trafficSpeed: 1.6,
  /**
   * Height the shapes fly through, metres — set to her shoulders. That is the
   * band where a passing shape actually crosses her silhouette, which is the
   * whole reason they exist; higher and they sail over her head, lower and they
   * are lost against the floor.
   */
  trafficHeightM: 1.35,
  /** Spread above and below that height. */
  trafficSpreadM: 0.30,

  /**
   * Slow camera drift, metres. Real parallax between the layers, with no camera
   * hardware. Small enough not to read as a moving shot, large enough that the
   * near and far structure visibly separate. This is what stops the scene
   * reading as a flat picture.
   */
  // 0.05 -> 0.09 (2026-07-29): the floor used to supply a static perspective
  // scaffold; without it, parallax between near and far structure is the only
  // thing left that a fixed viewpoint cannot explain away as a flat picture.
  // Still well under the threshold where it reads as a moving camera shot.
  /**
   * Drift as an ORBIT AROUND HER, in degrees to each side. This is the main
   * viewpoint-motion control; `cameraDrift` below is only used when this is 0.
   *
   * An orbit rather than a slide because at this size the two stop being
   * interchangeable. Fifteen degrees of translation is 0.83 m sideways at her
   * distance, which throws the framing away completely; fifteen degrees of orbit
   * holds the distance, so her size and her place in the frame are fixed by
   * construction and the entire movement goes into seeing her — and the space
   * behind her — from a different angle.
   *
   * 7.5 = a 15-degree TOTAL sweep. The angle is bounded by the frame, not by
   * taste: at the panels' depth the frame is only about +-0.40 m wide, and a
   * point 1.7 m behind her (the HUD ring) moves 1.7*sin(theta) sideways, so
   * theta = 15 per side swings the whole console 0.44 m — further than the frame
   * is wide. Measured at +-12 degrees: the ring left the frame entirely on one
   * side and all six panels clipped, worst content overflow 0.70 NDC. No fitting
   * can recover that, because scaling a panel pulls it toward the frame's centre
   * LINE and that line is itself what leaves the frame. At 7.5 the swing is
   * 0.22 m and everything stays in shot.
   */
  // Back to 7.5 (2026-10-22). It was cut to 2.5 against the 0/1 field's shimmer, but
  // that shimmer was fixed at its source since (camera-facing quads, glyph choice in
  // the vertex shader); measured at 7.5 the flicker stays proportional to the motion
  // (flicker/motion 1.2-1.5, same as at 2.5) — and at 2.5 the sway was too small to see
  // ("カメラが左右に揺れなくなった").
  cameraOrbitDeg: 7.5,
  /**
   * LATERAL drift amplitude, metres — the pre-orbit behaviour, used only when
   * cameraOrbitDeg is 0. Sideways movement is what slides the background against
   * her and separates the depth layers.
   */
  cameraDrift: 0.01, // 1/3 of the original 0.03 (2026-10-22): parallax for depth behind her, without the shimmer the full drift put on the 0/1 field
  /**
   * Vertical and depth amplitudes, metres — deliberately much smaller, and now
   * independent knobs rather than fixed fractions of the lateral one.
   *
   * They were 0.45x and 0.70x of the lateral amplitude, which on the retuned
   * 0.03 would be 0.0135 and 0.021. Z is the one that matters: moving the camera
   * along its own axis changes her APPARENT SIZE, and a subject that breathes
   * in and out of the frame is the most legible kind of camera wobble there is —
   * the exact impression this effect must avoid. It buys almost no parallax in
   * return, because motion along the view axis barely shifts anything laterally.
   * So Z is cut to 0.004 and Y to 0.010, and the budget goes to X.
   */
  cameraDriftY: 0.0035,
  cameraDriftZ: 0.0015,
  /** Seconds for one full drift cycle. Each axis also mixes a second, slower
   *  incommensurate component, so nothing in the path repeats on this period. */
  cameraDriftPeriod: 16,
  /** Master switch for the drift — `__AI_DEBUG__.setCameraDriftEnabled()`. */
  // On, at a third of the original amplitude (see cameraDrift). At full strength it
  // was the largest measured source of flicker on the 0/1 field — every column
  // shifts by its own parallax — and was switched off; the glyphs have since moved
  // from point sprites to quads, which take sub-pixel motion without stepping.
  cameraDriftEnabled: true,

  /**
   * How far SHE travels in depth, metres (see body-depth.ts). With no drawn
   * room left, this is the cue that a flat image cannot imitate: a still can
   * fake perspective and haze, but not a subject changing angular size. Period
   * is deliberately prime-ish against cameraDriftPeriod so subject motion and
   * viewpoint motion never sync into a predictable loop.
   * 0 disables. Past ~0.2 she reads as walking toward the viewer.
   */
  bodyDepth: 0, // reverted: not visible enough on its own to justify the motion
  bodyDepthPeriod: 23,

  // --- Depth by occlusion (replaces the drawn room) -------------------------
  /**
   * Gyroscope rings orbiting HER: they cross in front of her body, then pass
   * behind it. Half of every revolution asserts a depth order that a flat image
   * cannot assert at all — which is why this replaced the floor/rails/portals
   * rather than joining them.
   */
  orbit: 0,     // reverted 2026-07-29: too thin/far to actually cross her — read as extra scratches, not depth
  orbitCount: 3,
  /** Radius of the innermost ring, metres — set to clear her shoulders. */
  orbitRadius: 0.62,
  /** Centre height, metres. Her chest, so the crossing lands on her torso. */
  orbitHeightM: 1.15,

  /**
   * A few instrument-like marks in front of her, drifting sideways. States that
   * the frame has volume on the VIEWER's side of her too. Nearer ones move
   * faster, which is parallax on top of the occlusion. Keep the count low: dense
   * near-field marks stop reading as objects and become a dirty screen.
   */
  nearfield: 0, // reverted with orbit, same reason

  // --- Holographic console (the background, 2026-07-29) ---------------------
  /**
   * Concentric HUD rings + hex frames centred behind her. She occludes them,
   * which is the occlusion running the RIGHT way round: it states that she
   * stands in front of a space, where the earlier orbiting rings at her own
   * depth only ever read as marks on the picture.
   */
  hud: 1.0,
  /**
   * Radius of the dominant ring, metres.
   *
   * 0.70 -> 0.42, MEASURED not calculated. The old value carried the note "fits
   * the portrait frame: half-width is ~0.85 m at hudZ", and that was wrong twice
   * over: the frame is aspect 1/3, so the half-width at hudZ is about half that,
   * and off-axis projection moves the mapping again so no closed form predicts
   * it. `__AI_DEBUG__.ndcBounds("space-hud")` put the dominant ring at NDC
   * x -1.98..2.01 — literally twice the width of the frame, with each side
   * running off it. Only the top and bottom arcs were ever visible, which is why
   * it read as a bright band across the top rather than as a ring, however
   * brightly it was drawn.
   *
   * At 0.42 the same measurement gives x -1.057..1.057 for the dominant ring and
   * the two inner bands fully inside the frame, so the system reads as concentric
   * circles. Re-measure with ndcBounds after changing this; the arithmetic will
   * not tell you.
   */
  hudRadius: 0.42,
  /** Centre height and depth of the ring system, metres. */
  hudHeightM: 1.15,
  hudZ: -1.7,
  hudHexCount: 14,

  /**
   * The ring system is drawn as ADDITIVE ribbons with a core and a falloff
   * (glow-lines.ts), not as the GL lines the rest of the layer uses. Measured on
   * the frozen t=6 frame, the GL-line version put 82.5% of its pixels BELOW the
   * field in luminance and peaked at L=157 against a bloom threshold of 217 — it
   * was ink, and could not have glowed at any intensity. These knobs describe a
   * mark made of light instead.
   *
   * Colours are bright because they are ADDED to the field rather than
   * composited over it: the dark 0x1fa8d8 the ink lines needed would barely
   * register. Where core and halo overlap the sum clips toward white, which is
   * what gives the filament-in-a-glow look rather than a flat coloured band.
   */
  hudColor: 0x7fe0ff,
  hudAccent: 0x4fc8f5,
  /** Gain of the bright filament. Above ~1.2 the core clips white and blooms. */
  hudCore: 1.6,
  /** Gain of the surrounding glow. This is the knob that lifts the whole field,
   *  so it is the one to pull back if the panel stops reading as a window. */
  hudHalo: 0.30,
  /** Half-width of the filament, metres. ~0.0035 is about 2.6 px at hudZ. */
  hudCoreW: 0.0035,
  /** Half-width of the glow, metres. Also sets how wide the ribbon is baked, so
   *  raising it far past the built value needs __spaceRebuild(). */
  hudHaloW: 0.028,

  /**
   * Floating glass UI panels flanking her (schedule / mail / tasks / status /
   * identification / notifications). Objects a viewer already believes float,
   * so they need no supporting surface — which is what lets them build depth
   * without a floor, wall or ceiling.
   */
  panels: 0.92,
  /**
   * Strength of the additive glow layer behind each panel.
   *
   * Measured before it existed: the panels lifted the field they sat on by only
   * +1.4/255 on average and a quarter of their pixels were DARKER than it — flat
   * graphics on a lavender ground rather than lit glass. 0 restores that.
   *
   * 1.8 is where the sweep put it: measured 0 / 0.9 / 1.4 / 2.0 / 2.6 gave mean
   * field lifts of +0.7 / +4.6 / +6.4 / +8.3 / +10.0 with no saturated pixels
   * anywhere, so the ceiling is legibility, not clipping. It sits deliberately
   * below the rings (+12) — they are the hero, the panels are the chorus.
   */
  panelGlow: 1.8,
  /** Room around the panel for the glow to spread into, as a fraction of its
   *  height. Also sets the blur radius (0.45x of it), so raising it makes the
   *  halo both wider and softer. Needs __spaceRebuild() — it is baked. */
  panelGlowPad: 0.16,

  /**
   * Frosted glass: how much of the panel's own area is replaced by a BLURRED
   * copy of whatever is behind it. 0 turns the effect off entirely (and skips
   * the extra render), 1 makes the glass fully opaque to the sharp background.
   *
   * This is what makes them read as physical panes rather than as decals: a
   * decal lets the ring behind it stay pin-sharp right up to its edge, and the
   * eye reads "printed on top of the image" from exactly that.
   */
  panelFrost: 0.88,
  /** Blur radius at the CENTRE of the panel, in texels of the (downscaled)
   *  backdrop target. It ramps to zero at the border — see panelFrostEdge. */
  panelBlur: 2.2,
  /** Resolution divisor for that target. 4 = a sixteenth of the pixels; the
   *  result is blurred anyway, so detail here is paid for and thrown away. */
  panelRtDiv: 4,
  /** Slight tint mixed into the frost, so the glass has a colour of its own.
   *  0.16 -> 0.05: at 0.16 it was a visible veil over empty backdrop, and where
   *  three of these blobs overlapped (the two middle panels) it stacked into
   *  something that read as a shadow behind them. */
  panelTint: 0xdCEBff,
  panelTintAmt: 0.05,
  /**
   * How much the blur must CHANGE the background before the glass shows at all.
   *
   * The frost is drawn only where it has something to soften. Over flat backdrop
   * the blurred and unblurred samples are identical, this gate is zero, and
   * nothing is drawn — which removes the faint veil, its build-up where blobs
   * overlap, and the colour mismatch from the source never passing through the
   * composer's grade, all at once. Raise it to make the glass shyer, lower it
   * toward 0 to go back to a full-area pane.
   */
  panelFrostDetail: 0.035,
  /**
   * Radius at which the frost's radial falloff STARTS, as a fraction of the
   * padded quad's half-size. It reaches zero at 1.0, so 0.18 means the blob is
   * fading over more than 80% of its extent.
   *
   * The falloff is radial rather than per-axis for a reason worth keeping: a
   * per-axis ramp is soft across each edge but perfectly straight along it, so
   * the blurred area stayed a visible rectangle no matter how gentle the fade.
   */
  panelFrostEdge: 0.18,
  /** How far past the panel the frost quad extends, as a fraction of the panel
   *  height. Generous, so the blob ends in open space rather than on the panel's
   *  own outline — that outline is exactly what the eye was picking up. */
  panelFrostPad: 0.55,

  /**
   * Far field: thin data towers and floating hex fragments at many depths.
   * The same kind of object repeated at many apparent sizes is what actually
   * carries "deep" — and unlike a floor grid it draws no surface for the real
   * objects behind the glass to collide with.
   */
  towers: 0.85,
  towerCount: 44,
  hexCount: 16,
  /**
   * Far-field colours, separate from the shared `lineColor`/`accentB`.
   *
   * Separate so the far field can be tuned to sit just above the backdrop
   * without dragging the traffic, wireframe and near-field marks with it — those
   * are read against her silhouette and want to stay ink. Set by MEASUREMENT
   * against the diff probe, not by eye: the target is a mean luminance delta of
   * roughly 0..+2 per 255 against the surrounding field, which is "visible as
   * structure" without the far field becoming a light source. See towers.ts.
   *
   * Swept against the probe, mean delta / fraction of pixels brighter than the
   * field:  0x1fa8d8 (the old shared colour) -4.05 / 3.5% — ink.  0xa8c9d8
   * +0.11 / 72%.  0xb0cfdd +1.20 / 97.5%.  0xb6d4e1 +2.03 / 99.1%.  0xbcd9e6
   * +2.85 / 99.3%.  0xdceef5 +5.44.  Picked the low end of the acceptable band
   * rather than the high one, because the note this file carries is a warning
   * about brightening and the failure it describes is not visible from here.
   */
  towerColor: 0xb0cfdd,
  towerAccent: 0xc6dee9,

  // --- Depth of field (the depth AXIS) --------------------------------------
  /**
   * Aperture the space layer asks the shared BokehPass for, taking the larger of
   * this and LOOK.dofAperture so her tuned look is never weakened.
   *
   * This is the first thing to get right and the cheapest. A frame in which
   * everything is equally sharp has no depth axis — the eye has no evidence
   * that any part of the image is further away than any other, so drawn
   * structure just becomes pattern on a plane (which is exactly what the thin
   * rings did). Once the background is visibly defocused and she is not, every
   * other cue back there starts to read.
   *
   * Focus tracks the orbit target (her), so raising this only softens the space.
   */
  dofBackground: 0, // OFF by user request 2026-07-29 (was 0.0022)
  /** Blur ceiling for the above. */
  dofBlur: 0.016,
  nearfieldCount: 5,
  /** How far in front of her they sit, metres (positive = toward the viewer). */
  nearfieldZ: 1.1,

  // --- Atmosphere -----------------------------------------------------------
  /** Sparse point field. Points only — no sprites, no glow quads. */
  particles: 0.35,

  // --- Overlay chrome -------------------------------------------------------
  /**
   * Fade the connection badge and the corner buttons out when idle, like a video
   * player. They are the only obviously-an-application thing on the panel, and a
   * permanent installation should not be advertising that.
   */
  uiAutoHide: true,
  /** Milliseconds of no touch before the chrome fades. */
  uiIdleMs: 3000,

  // --- Grounding ------------------------------------------------------------
  /**
   * Soft contact shadow under her feet. Only possible at all with the white
   * field: on the previous black one, dark ink was invisible by construction.
   *
   * The key light's real cast shadow was tried alongside it and removed — two
   * shadows from different directions read as two light sources and made the
   * grounding less believable, not more.
   */
  contactShadow: 0, // off (2026-10-22, dataflow): the shadow patch read as a drawn floor; was 0.40
  /** Lateral half-width of the shadow, metres. */
  shadowRadius: 1.05,
  /** How far it stretches toward the viewer — long enough to leave the frame. */
  shadowReach: 5.5,
  /** Falloff exponent. Higher = softer, with no readable edge. */
  shadowSoft: 3.0,

  // --- Where the viewer's eye is assumed to be ------------------------------
  /**
   * People look at a PROTO standing up, so the virtual eye has to be at standing
   * height too. With the camera at chest height the horizon lands low on the
   * panel, and the floor consequently reads as tilted up toward the viewer —
   * "地面が上に着すぎている".
   *
   * `horizonFrac` is where the horizon sits in the frame, 0 = bottom, 1 = top.
   * Physically it should be (eyeHeight - panelBottom) / panelHeight, which for a
   * standing adult in front of this cabinet is around 0.75. It is left tunable
   * because the panel has not been measured and because how "right" it looks
   * also depends on the FOV (see `fovDeg`).
   */
  /**
   * Height of the virtual eye, metres. THE camera control.
   *
   * Cannot be set independently of where the horizon lands: with her whole body
   * framed the pitch is already determined, so the horizon is wherever the level
   * line falls. Driving it from the horizon instead (an earlier version) put the
   * eye at 0.5 m and had the camera looking up at her. 1.25 m sits between her
   * chest and chin and reads as level.
   */
  viewerEyeY: 1.45, // 1.25 -> 1.45 (2026-10-22): a touch higher, looking slightly down
  /**
   * Frame her by solving, not by eye. With this on, the camera distance and
   * pitch are derived so that fitBottomY..fitTopY is always inside the frame —
   * so her feet cannot end up cropped by a settings change. Turn off to drive
   * the framing with horizonFrac instead.
   */
  fitBody: true,
  /**
   * World Y range that must stay in frame. Set tight around her actual extent
   * (soles ~0, hair ~1.63) — every centimetre of slack here is margin the frame
   * has to spend, and the reference framing leaves almost none.
   */
  fitBottomY: 0.0,
  /**
   * Top of the range that must stay in frame, metres.
   *
   * MEASURED FROM THE MODEL when `fitAutoHeight` is on, not assumed. 1.65 is
   * Hatsune Miku's height in this scene and was hard-coded while she was the
   * only 3D character; the first plain-.glb avatar stands 1.80 m and had its
   * head cropped off the top of the frame, because the solver was still framing
   * a 1.65 m subject. Anything model-specific has to come from the model.
   */
  fitTopY: 1.65,
  /** Re-derive fitTopY from the loaded character's own bounding box. Off pins
   *  the value above, which is what you want when hand-tuning a crop. */
  fitAutoHeight: true,
  /** Headroom above the measured top, as a fraction of the model's height.
   *  Hair and antennae usually sit outside the mesh bounds people think of. */
  fitHeadroom: 0.012,
  /**
   * How far above the bottom edge her soles sit, as a fraction of the frame
   * height. Small on purpose — the brief is "just short of being cut off" — and
   * it is a hard floor, not a hope: the camera pitch is solved from it, so the
   * feet stay exactly this far in whatever else changes.
   */
  // 0.035 (2026-10-22): a little room in front of her feet for her reflection
  // (floor-reflection.ts); 0.12 left too much empty floor once the chat bar
  // started collapsed.
  bottomMargin: 0.035,
  /**
   * Fraction of the frame height that range fills. Measured: 0.965 puts her body
   * at ~92% of the frame, head ~4% down and toes ~96% down, which is the
   * reference crop. The earlier 0.86 left her filling only ~72% and reading as a
   * figure across the room rather than someone standing in front of you.
   */
  // Must sum with bottomMargin to under 1, or her head leaves the top of the frame.
  fillFrac: 0.935,
  /** Only used when fitBody is false. */
  horizonFrac: 0.52,
  /**
   * Vertical field of view.
   *
   * 30 is the viewer's original value and it is a TELEPHOTO lens — it compresses
   * depth, which works directly against everything else here. A real window
   * 1.9 m tall seen from 1.2 m away subtends about 76 degrees, and a wide angle
   * is what gives strong perspective and a genuine sense of depth. Widening this
   * changes how she is drawn, so it is opt-in rather than assumed.
   */
  fovDeg: 30,
  /** Distance fade. Lines must reach zero before the far plane, never band. */
  haze: 1.0,

  // --- Framing --------------------------------------------------------------
  /**
   * Life-size: the camera distance is DERIVED so one world metre maps to one
   * metre of panel at her depth. Measure the panel with __protoCalibrate() and
   * the scale becomes physically exact.
   */
  lifeSize: true,
  /** Her eye-level height for framing, metres. */
  protoTargetY: 0.95,
  /** Camera height, metres. */
  protoCamY: 1.05,
  /** Used only when lifeSize is false. */
  protoCamZ: 3.6,
  protoLock: false,

  // --- Phase 2: head tracking (parked) --------------------------------------
  headTracking: false,
  headCamDeviceId: "",
  headCamWidth: 1280,
  headCamFovDeg: 60,
  /** Outer eye corners, metres — the depth ruler for this page's own tracker. */
  eyeSpanM: 0.098,
  /**
   * Eye CENTRES, metres (average adult pupil distance) — the depth ruler for
   * faces from the server, whose detector marks the middle of each eye.
   */
  pupilSpanM: 0.063,
  headCamMirror: true,
  headCamOffsetX: 0,
  /**
   * Camera height above the panel centre, metres. The PROTO's camera sits on top
   * of the cabinet, i.e. about half a panel (panelHeightM / 2) above centre.
   * Left at 0 every face read ~1 m too low and her gaze dropped to the viewer's
   * knees — the eyes are the one place where a height error is obvious.
   */
  headCamOffsetY: 0.95,
  headCamOffsetZ: 0,
  /** Faces this page's own tracker detects per frame; the CLOSEST is attended. */
  headCamMaxFaces: 6,
  /**
   * Hysteresis for changing who she attends to. Another face has to be this much
   * closer (fraction of the current one's distance) for `focusSwitchS` without
   * a break before she turns to them. Without it two people at nearly the same
   * distance swap places with every frame of depth noise, and her head ping-pongs.
   */
  focusSwitchMargin: 0.15,
  focusSwitchS: 0.6,

  // --- Following the attended person with the world camera -----------------
  /**
   * While someone is tracked, the drift's ORBIT is driven by where they stand
   * instead of by the sine: stand to the right of the PROTO and the camera swings
   * round to her right-hand side of the screen, so you see her from your side.
   * Degrees of orbit per degree of the viewer's bearing from her — 1 means the
   * camera sits on the line from her to you, which is also what makes her look
   * into the lens when she looks at you. Negative mirrors the direction.
   */
  headOrbitGain: 1.0,
  /**
   * Limit of the follow orbit to each side. Bounded by the frame, like
   * cameraOrbitDeg: past ~12 the HUD ring behind her starts leaving the frame.
   * Raised a little above the idle sway's 7.5 so a viewer off to the side
   * visibly sees round her.
   */
  headOrbitMaxDeg: 12,
  /** Camera metres up/down per metre the viewer's head is above/below the virtual eye (viewerEyeY). */
  headOrbitY: 0,
  /** 1/s. How fast the camera settles on a new bearing (a focus change is a jump). */
  headOrbitSpeed: 2.0,
  /** Seconds to hand the camera over between the idle sway and following. */
  headOrbitBlendS: 0.8,

  // --- Phase 4: looking at the viewer ---------------------------------------
  /**
   * Strength of the head/upper-body turn toward the attended person. Only acts
   * while someone is tracked (scaled by the follow hand-over), so the untracked
   * view keeps her facing front exactly as before.
   */
  // 0.8 (was 0.55): the head carries most of the turn and the eyes the last few
  // degrees, as people do when they attend to someone. 0.55 was tuned when the
  // effective strength still depended on the motion (see look-at-viewer _undo).
  lookAt: 0.8,
  lookAtMaxYaw: 40,
  lookAtMaxPitch: 22,
  lookAtSpeed: 4.0,

  // --- Phase 3: off-axis projection (inert without a tracked head) ----------
  /**
   * Off by default: it takes the whole camera over the moment a face is tracked,
   * and the follow orbit above is what the PROTO installation uses instead. It
   * also needs the panel measured (__protoCalibrate) before it is right at all.
   */
  offAxis: false,
  offAxisStrength: 1.0,
  parallaxBoost: 0.0,
  /** PHYSICAL PANEL GEOMETRY — measure with __protoCalibrate(). */
  panelWidthM: 0.5,
  panelHeightM: 1.9,
  panelCenterY: 1.0,

  // --- Resolution -----------------------------------------------------------
  pixelRatioCap: 3.0,
  maxDrawPixels: 8_300_000,
  /**
   * Render ABOVE the display's own pixel ratio, then let the downscale average.
   * 1 = native (previous behaviour). 1.5 costs 2.25x the pixels, 2 costs 4x.
   *
   * This is the lever for outline shimmer. On a dpr-1 monitor the character is
   * drawn at exactly 1.0, where her ~1px outline hull re-lands on a different
   * pixel every frame she moves and reads as a crawling, flickering edge.
   * Supersampling is the only thing that actually removes it — MSAA antialiases
   * geometry coverage within a frame, and the final SMAA pass is a spatial
   * filter on an already-flickering image; neither addresses motion.
   *
   * Live-tunable, no rebuild:  __space({ superSample: 1.5 })
   * The total-pixel budget above still caps it, so a big window quietly gets
   * less of it rather than dropping frames.
   *
   * Back to 1.0 (2026-08-10). It was briefly 1.25, which measurably reduced the
   * outline flicker — but that was treating a symptom. The line was sub-pixel
   * because it was specified in metres; supersampling only meant more pixels for
   * a line that should never have been that thin. The cause is fixed at
   * SCREEN_OUTLINE in vrm-viewer.tsx, so there is no reason to charge every
   * machine 1.56x the pixels for a workaround that is no longer doing anything.
   *
   * Still here, still live-tunable, if a display ever wants it.
   */
  superSample: 1.0,
  /**
   * Temporal antialiasing. 0 = off (the pre-TAA pipeline, exactly: the pass is
   * skipped by the composer, so the chain is bit-for-bit what it was).
   *
   * This is the fix for the crawling edges, where supersampling was only ever a
   * mitigation. Nothing in the pipeline looked at more than one frame at a time,
   * so every edge recomputed itself from scratch each frame; see utils/space/taa.ts.
   *
   * ON: reprojection (bone-exact motion vectors + closest-depth dilation) is in,
   * and with it the edge instability drops on every axis that was measured when
   * this was the complaint. Per-frame changed pixels, taa=0 -> taa=1, one render
   * per frame:
   *
   *          outline        internal edges   flat
   *   t=6    2.04 -> 0.34   1.74 -> 0.30     0.097 -> 0.006
   *   t=12   0.79 -> 0.72   0.52 -> 0.23     0.018 -> 0.003
   *   t=20   0.97 -> 0.64   0.77 -> 0.27     0.035 -> 0.004
   *
   * Checked for the failure mode that would have vetoed it: consecutive frames
   * of the swinging twintails show no trailing and no double image.
   */
  taa: 1,
  /**
   * How much of the reprojected history to keep, 0..1. Higher is steadier and
   * ghostier. Was a conservative 0.85 while there was no reprojection — then the
   * history was ALWAYS slightly wrong under motion, and keeping less of it was
   * the only defence. With motion vectors the aligned history is right, and at
   * 0.85 the 15%-per-frame ingest of the still-walking jitter kept thin ink
   * lines oscillating (measured: raising this to 0.95 cut outline-pixel change
   * by 69-76% at the low-motion times where that oscillation dominates). Ghost
   * risk is carried by the clamp + reject + dilation instead, and the swinging
   * twintails were checked frame-by-frame at this value: no tails.
   */
  taaFeedback: 0.95,
  /**
   * Neighbourhood clamp width, in standard deviations of the 3x3 colour
   * distribution. Lower rejects history sooner — that is the anti-ghost knob.
   */
  taaClamp: 1.0,
  /**
   * How far the clamp may drag the history before it is thrown away outright,
   * as a fraction of the local neighbourhood's own width. A history that has to
   * travel most of the way across the box is not a slightly-stale version of this
   * pixel — it is a different surface, uncovered since last frame. Clipping it
   * would leave a plausible-looking wrong colour; binning it costs one aliased
   * frame. Lower = more ghost-averse.
   */
  taaReject: 0.5,
  /**
   * Reprojection sanity ceiling, in pixels of motion per frame. Past it the
   * history fetch is a guess, not a track — a gesture snap, a traffic mover
   * wrapping from END_Z back to START_Z — so the history is discarded outright
   * rather than fetched from wherever the vector happens to land.
   */
  taaMaxVel: 80,
  /**
   * Feedback ceiling for pixels that are moving fast (faded in between 2 and
   * 20 px/frame). Reprojection puts old colour in the right PLACE but cannot
   * give a surface its new silhouette, so a long history on something swinging
   * hard keeps drawing the old shape — measured on her tie under a body swing,
   * TAA's frame differed from the true frame on 9.1% of the region with the tie
   * visibly the wrong width. Slow and still pixels keep the full `taaFeedback`,
   * which is where the crawling-edge win actually comes from.
   */
  taaFastFeedback: 0.6,
  /**
   * Post-resolve contrast-adaptive sharpen, 0 = off. What CAS is to a game's
   * TAA: the resolve trades a little sharpness for stability even with a
   * bicubic history fetch, and this buys it back after the accumulator so it
   * can never compound. Clamped to the local extremes, so no halos. 0.001 steps
   * matter less than they look — the useful range is roughly 0.2..0.7.
   */
  taaSharpen: 0.4,
  /**
   * Thin-feature lock strength (FSR2's "locks", simplified), 0 disables. A
   * sub-pixel ink line or hair strand only rasterises on the frames the jitter
   * catches it; the clamp then erases on the misses what the hits accumulated,
   * and the feature blinks. A pixel whose luma pokes past all four neighbours
   * is locked for ~8 frames: it keeps more history and a wider clamp box.
   * Ghost bound: a lock dies on motion (>4px/frame), on depth disocclusion,
   * and by expiry — so the worst case is a wrong thin feature for ~133 ms.
   */
  taaLock: 1,
  /**
   * How much a depth disocclusion kills the history, 0..1. The clamp rejects
   * by colour, so a newly-uncovered surface that resembles its old occluder
   * ghosts; comparing this frame's depth against last frame's (stored by the
   * resolve) catches that geometrically. Also the safety net that lets
   * taaLock hold history as hard as it does.
   */
  taaDisocc: 1,
  /**
   * Internal render scale while TAA is on (TAAU), 0.5..1. The scene side —
   * RenderPass, ink lines, GTAO, velocity — renders at this scale; the TAA
   * history and everything after it (grade, DOF, bloom, SMAA) stay at full
   * resolution, and the jittered low-res samples land on different output
   * pixels each frame, so the accumulator reconstructs the difference. This is
   * how current games buy their frame budget (DLSS/FSR quality mode = 0.67).
   * Ignored (forced to 1) while taa is 0 — without the accumulator there is
   * nothing to reconstruct with, only a bilinear stretch.
   */
  taaUpscale: 1.0,
  /**
   * Frames of a completely still scene after which the history LATCHES and stops
   * accumulating. Determinism, not performance: `freeze()` stops the clock but not
   * the renderer, and the jitter keeps walking, so a frozen scene never produces
   * two identical frames — see the `converged` note in utils/space/taa.ts.
   *
   * Set to 0 to disable the latch. A measurement run MUST do that: latched frames
   * are byte-identical by construction, so they report a perfect stability score
   * whether or not the accumulation underneath works at all.
   */
  taaSettle: 24,
  /**
   * Diagnostic views, painted instead of the image.
   *   1 = the resolve's decision: red = how far the history had to be clipped,
   *       green = how much of it survived. Shows WHERE history is rejected.
   *   2 = the velocity buffer: 0.5 grey = still, R/G offset = screen motion
   *       (saturating at ±5 px/frame), blue marks skinned — her — pixels.
   * Turn the blurring passes off with `__vrmLook({ bloomStrength: 0,
   * diffusion: 0, dofAperture: 0 })` first, or bloom and DOF smear the readout.
   */
  taaDebug: 0,
  /**
   * MSAA samples on the main composer target. Was hard-coded to 4; most desktop
   * GPUs report 8 or 16 for MAX_SAMPLES, and geometry edges (the outline hull is
   * geometry) are exactly what MSAA is good at. Clamped to what the driver
   * actually supports. Changing this needs a viewer rebuild, not just __space.
   */
  msaaSamples: 8,
};

export type SpaceConfig = typeof SPACE;

/**
 * Resolved per-quality budgets. Everything that costs frame time scales here.
 * Real-time latency is the priority for Phase 3 (head-coupled perspective), so
 * Phase 1 must leave headroom rather than spend it.
 */
export const BUDGETS: Record<Quality, {
  ringCount: number;
  railSegments: number;
  pillarCount: number;
  particlesNear: number;
  particlesFar: number;
  particlesFore: number;
  distantLights: number;
  reflection: boolean;
  shadowCatcher: boolean;
}> = {
  low: {
    ringCount: 6,
    railSegments: 12,
    pillarCount: 6,
    particlesNear: 420,
    particlesFar: 700,
    particlesFore: 0,
    distantLights: 120,
    reflection: false,
    shadowCatcher: true,
  },
  medium: {
    ringCount: 10,
    railSegments: 18,
    pillarCount: 12,
    particlesNear: 800,
    particlesFar: 1300,
    particlesFore: 0,
    distantLights: 220,
    reflection: true,
    shadowCatcher: true,
  },
  high: {
    ringCount: 13,
    railSegments: 26,
    pillarCount: 18,
    particlesNear: 1400,
    particlesFar: 2400,
    particlesFore: 0,
    distantLights: 340,
    reflection: true,
    shadowCatcher: true,
  },
};

/**
 * Bump when the shipped defaults change meaningfully. A saved blob from an older
 * build is discarded so what ships is what you see — same guard the viewer's
 * LOOK uses, for the same reason ("directly why 変わってない keeps happening").
 */
// Bumped to discard TAA experiment state left in localStorage. A sweep writes its
// last value back, and a stale entry is silently load-bearing: a persisted
// taaSettle of 0 keeps the determinism latch off for every later run, and a
// persisted taaDebug of 1 leaves the app showing the green/red diagnostic instead
// of the character until someone thinks to clear it.
export const SPACE_BUILD = "s42"; // s42: TAA locks + depth disocclusion + TAAU split
const LS_SPACE = "vrmSpace1";

/** Pick a starting quality from the hardware, so a weak PROTO box isn't crushed. */
export function detectQuality(): Quality {
  if (typeof navigator === "undefined") return "medium";
  const cores = navigator.hardwareConcurrency || 4;
  const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
  if (cores <= 4) return "low";
  if (cores <= 8 || dpr > 2) return "medium";
  return "high";
}

let quality: Quality = detectQuality();

export function getQuality(): Quality {
  return quality;
}

export function getBudget() {
  return BUDGETS[quality];
}

export function setQualityValue(q: Quality) {
  quality = q;
}

/**
 * One-time migration of individual keys onto an already-saved space config —
 * the space twin of the look's GRADE_FIX. A saved blob with a matching _build
 * overrides the shipped defaults verbatim, so editing a default here never
 * reaches a browser that already persisted the old value; bumping SPACE_BUILD
 * would deliver it only by discarding every other saved knob with it.
 *
 * Stamped (_rev) and re-saved immediately, so it applies once and a later
 * deliberate hand-set of the same key sticks.
 *
 * rev 1: taa 0 -> 1. A stored taa:0 (from the TAA experiments) pins the
 * temporal pass off and brings back exactly the crawling edges it was built to
 * fix; shipped default is 1 and nothing in the sweeps depends on 0 anymore.
 *
 * rev 2: camera-driven attention. lookAt 0 -> 0.55 (now gated on tracking, so
 * it no longer turns her toward the drifting camera), offAxis -> false (the
 * follow orbit replaces it), headCamOffsetY 0 -> 0.95 (PROTO camera on top).
 *
 * rev 3: lookAt 0.55 -> 0.8, once the correction stopped compounding on some
 * motions and not others (look-at-viewer undoViewerLookAt).
 */
const SPACE_REV = 3;
const SPACE_FIX: Partial<SpaceConfig> = { taa: 1, lookAt: 0.8, offAxis: false, headCamOffsetY: 0.95 };

export function loadSpaceConfig() {
  // The device budget goes on FIRST, so a saved blob can still override it: the
  // tuning panel has to stay authoritative on whatever box it is being used on.
  // Desktop values are identical to the shipped defaults, so this is a no-op
  // anywhere that is not a phone.
  const dev = deviceProfile();
  if (dev.mobile) {
    SPACE.maxDrawPixels = dev.maxDrawPixels;
    SPACE.pixelRatioCap = dev.pixelRatioCap;
    SPACE.msaaSamples = dev.msaaSamples;
    // The particle fields are the one thing the quality tier actually scales.
    quality = "low";
  }
  try {
    const raw = localStorage.getItem(LS_SPACE);
    const saved = raw ? JSON.parse(raw) : null;
    if (saved && saved._build === SPACE_BUILD) {
      const { _build, _quality, _rev, ...rest } = saved;
      const migrate = _rev !== SPACE_REV;
      if (migrate) Object.assign(rest, SPACE_FIX);
      Object.assign(SPACE, rest);
      if (_quality === "low" || _quality === "medium" || _quality === "high") {
        quality = _quality;
      }
      if (migrate) saveSpaceConfig();
    }
  } catch {
    /* ignore — fall back to the shipped defaults */
  }
}

export function saveSpaceConfig() {
  try {
    localStorage.setItem(
      LS_SPACE,
      JSON.stringify({ ...SPACE, _build: SPACE_BUILD, _quality: quality, _rev: SPACE_REV }),
    );
  } catch {
    /* ignore — private mode / quota */
  }
}
