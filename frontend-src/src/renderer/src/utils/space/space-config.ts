/**
 * Tunables for the Spatial Presence Layer.
 *
 * Deliberately a SEPARATE namespace from the viewer's `LOOK` object and its
 * `vrmLook8` / `LOOK_BUILD` storage. Adding fields to `LOOK` would force a build
 * bump, and a build bump throws away every look tweak the user has dialled in
 * (the saved blob is discarded wholesale when `_build` mismatches). Keeping the
 * space settings in their own key means the two can never reset each other.
 */

export type Quality = "low" | "medium" | "high";

export const SPACE = {
  /** Master switch. */
  enabled: true,

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
  cameraOrbitDeg: 7.5,
  /**
   * LATERAL drift amplitude, metres — the pre-orbit behaviour, used only when
   * cameraOrbitDeg is 0. Sideways movement is what slides the background against
   * her and separates the depth layers.
   */
  cameraDrift: 0.03,
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
  cameraDriftY: 0.010,
  cameraDriftZ: 0.004,
  /** Seconds for one full drift cycle. Each axis also mixes a second, slower
   *  incommensurate component, so nothing in the path repeats on this period. */
  cameraDriftPeriod: 16,
  /** Master switch for the drift — `__AI_DEBUG__.setCameraDriftEnabled()`. */
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
  contactShadow: 0.40,
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
  viewerEyeY: 1.25,
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
  fitTopY: 1.65,
  /**
   * How far above the bottom edge her soles sit, as a fraction of the frame
   * height. Small on purpose — the brief is "just short of being cut off" — and
   * it is a hard floor, not a hope: the camera pitch is solved from it, so the
   * feet stay exactly this far in whatever else changes.
   */
  bottomMargin: 0.02,
  /**
   * Fraction of the frame height that range fills. Measured: 0.965 puts her body
   * at ~92% of the frame, head ~4% down and toes ~96% down, which is the
   * reference crop. The earlier 0.86 left her filling only ~72% and reading as a
   * figure across the room rather than someone standing in front of you.
   */
  fillFrac: 0.96,
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
  eyeSpanM: 0.098,
  headCamMirror: true,
  headCamOffsetX: 0,
  headCamOffsetY: 0,
  headCamOffsetZ: 0,

  // --- Phase 4: looking at the viewer (parked with the camera) --------------
  /** 0 while head tracking is off; 0.55 is the tuned value to restore with it. */
  lookAt: 0,
  lookAtMaxYaw: 40,
  lookAtMaxPitch: 22,
  lookAtSpeed: 4.0,

  // --- Phase 3: off-axis projection (inert without a tracked head) ----------
  offAxis: true,
  offAxisStrength: 1.0,
  parallaxBoost: 0.0,
  /** PHYSICAL PANEL GEOMETRY — measure with __protoCalibrate(). */
  panelWidthM: 0.5,
  panelHeightM: 1.9,
  panelCenterY: 1.0,

  // --- Resolution -----------------------------------------------------------
  pixelRatioCap: 3.0,
  maxDrawPixels: 8_300_000,
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
export const SPACE_BUILD = "s38"; // s38: frost only where it has something to blur
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

export function loadSpaceConfig() {
  try {
    const raw = localStorage.getItem(LS_SPACE);
    const saved = raw ? JSON.parse(raw) : null;
    if (saved && saved._build === SPACE_BUILD) {
      const { _build, _quality, ...rest } = saved;
      Object.assign(SPACE, rest);
      if (_quality === "low" || _quality === "medium" || _quality === "high") {
        quality = _quality;
      }
    }
  } catch {
    /* ignore — fall back to the shipped defaults */
  }
}

export function saveSpaceConfig() {
  try {
    localStorage.setItem(
      LS_SPACE,
      JSON.stringify({ ...SPACE, _build: SPACE_BUILD, _quality: quality }),
    );
  } catch {
    /* ignore — private mode / quota */
  }
}
