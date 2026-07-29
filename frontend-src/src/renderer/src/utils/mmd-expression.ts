/* eslint-disable no-underscore-dangle */
import type * as THREE from "three";

const LS_FACE = "mmdBaseFace2"; // bumped: discard any stale saved resting face
// (a persisted base with a mouth morph could leave the idle mouth parted)

// Emotion -> facial morph preset. Morph names verified against the yyb_miku PMX
// (note full-width chars: ｷﾘｯ, 困る２). ONLY 眉(brow) + 目(eye) + mouth-CORNER +
// manga-effect morphs — never the あいうえお open-mouth morphs, which the
// independent lip-sync layer owns (it writes one mouth morph every frame). The
// whole expressive face crossfades from the resting base to the active preset,
// so e.g. a happy face doesn't keep the resting worried brow.
// yyb_miku's bangs largely HIDE the eyebrows, so brow morphs read weakly. The
// presets therefore lean on the visible channels — eye SHAPE, mouth corners, and
// the manga-effect morphs (涙 tears, 照れ/赤面 blush, 怒りマーク veins, ！marks) —
// with brows only as a supporting accent. Never the あいうえお open-mouth morphs
// (owned by lip-sync). Weights are strong so the emotion reads at a glance.
const EMOTION_PRESETS: Record<string, Record<string, number>> = {
  neutral: {}, // = the resting base (crossfade target 0)
  // --- the 7 basic families ---
  happy: { 笑い: 0.9, 口角上げ: 0.7, にやり: 0.15, 照れ: 0.2 },
  sad: { たれ目: 0.7, 困る２: 0.5, 口角下げ: 0.5, 涙: 0.35, 下: 0.3 },
  angry: { ｷﾘｯ: 0.7, 怒り: 0.6, "怒り+左": 0.6, "怒り+右": 0.6, 口角下げ: 0.4 },
  surprised: { びっくり: 0.9, 上: 0.6, 眉頭: 0.3, "！左": 0.4, "！右": 0.4 },
  thinking: { じと目: 0.6, 真面目: 0.4, 困る: 0.3, "上 左": 0.3 },
  shy: { 照れ: 0.85, 赤面: 0.45, はぅ: 0.5, 困る: 0.3, 口角上げ: 0.3 },
  // --- fine-grained variants the LLM emits (B) ---
  joy_grin: { 笑い: 0.7, にやり: 0.35, 口角上げ: 0.85, 照れ: 0.25 }, // big open grin
  joy_smile: { 笑い: 0.55, 口角上げ: 0.45, にこり: 0.35 }, // soft gentle smile
  anger_pout: { じと目: 0.5, 怒り: 0.4, 口角下げ: 0.35, 赤面: 0.3 }, // sulky pout
  anger_glare: { ｷﾘｯ: 0.9, 瞳小: 0.55, 怒り: 0.5 }, // cold sharp glare, no veins
  sadness_tearful: { たれ目: 0.75, 困る２: 0.6, 口角下げ: 0.55, 涙: 0.75 }, // crying
  sadness_worry: { 困る２: 0.65, たれ目: 0.4, 口角下げ: 0.2 }, // worried, no tears
  surprise_amazed: { びっくり: 0.75, 星目: 0.7, 上: 0.5, 照れ: 0.15 }, // sparkly wow
  smug: { にやり: 0.4, ｷﾘｯ: 0.4, じと目: 0.3, "上 左": 0.3 }, // smirk, one brow up
  fear: { びっくり: 0.6, 困る２: 0.55, "汗 左": 0.55, "汗 右": 0.55, たれ目: 0.3 }, // scared, sweat
  love: { はぁと: 0.8, 照れ: 0.6, 笑い: 0.4 }, // heart eyes
};

// Resolve a raw emotion keyword (backend actions.emotions) to a preset name:
// exact match first (joy_grin, anger_pout, ...), else the family by prefix, else
// neutral. Keeps the face working for both fine and coarse emotionMaps.
function resolveFacePreset(keyword: string): string {
  const k = keyword.toLowerCase();
  if (EMOTION_PRESETS[k]) return k;
  if (k.startsWith("joy")) {
    if (k.includes("grin")) return "joy_grin";
    if (k.includes("smile")) return "joy_smile";
    return "happy";
  }
  if (k.startsWith("anger") || k.startsWith("angry")) {
    if (k.includes("pout") || k.includes("sulk")) return "anger_pout";
    if (k.includes("glare") || k.includes("serious")) return "anger_glare";
    return "angry";
  }
  if (k === "disgust") return "anger_pout";
  if (k.startsWith("sad")) {
    if (k.includes("tear")) return "sadness_tearful";
    if (k.includes("worry")) return "sadness_worry";
    return "sad";
  }
  if (k.startsWith("surprise")) {
    if (k.includes("amaze")) return "surprise_amazed";
    return "surprised";
  }
  if (k === "fear") return "fear";
  if (k === "smug" || k === "smirk") return "smug";
  if (k.startsWith("love") || k.includes("heart")) return "love";
  if (k.startsWith("think")) return "thinking";
  if (k.startsWith("shy")) return "shy";
  return "neutral";
}

// Eye-shape morphs that a blink must close OVER (scaled down while the lids
// shut). Everything else (brows, mouth corners, blush) is untouched by blinking.
const EYE_MORPHS = new Set([
  "笑い", "笑い２", "笑い目", "じと目", "たれ目", "びっくり", "はぅ",
  "ｷﾘｯ", "ｷﾘｯ２", "瞳小", "なごみ", "はちゅ目", "はちゅ目２", "星目", "はぁと",
]);

// Structural morphs kept regardless of emotion (eye-highlight placement); losing
// them under an emotion crossfade makes the eyes read as dead.
const ALWAYS_ON: Record<string, number> = { HL移: 1.0 };

const BLINK_MS = 130; // one blink close+open
const BLINK_MIN_MS = 2500; // min gap between blinks
const BLINK_RND_MS = 3800; // + up to this much random

// Prosody: the active emotion's strength is scaled between PROSODY_FLOOR (quiet)
// and 1.0 (loud) by the current speech energy, so the face pulses with emphasis
// instead of holding a flat pose. Between sentences (no audio) it holds at
// NEUTRAL_EMPHASIS so the expression doesn't visibly dim in the gaps.
const PROSODY_FLOOR = 0.72;
const PROSODY_GAIN = 1.6; // maps raw RMS volume -> 0..1 emphasis
const NEUTRAL_EMPHASIS = 0.8;

/**
 * Drives the MMD model's FACE: a persistent "resting face" plus per-sentence
 * EMOTION expressions (crossfaded on top) and a procedural blink. Mirrors the
 * mmd-motion-player design (singleton, crossfade, per-frame update); the mouth
 * (あ) is owned by mmd-lipsync and left untouched here.
 *
 *  - `playEmotion(e)` crossfades the expressive face to that emotion's preset.
 *  - `relax()` eases it back to the resting face (call when speech ends).
 *  - blink runs on its own timer over whatever expression is showing.
 *
 * Resting face + panel tuning (window.__mmdFace) are unchanged from before.
 */
class MMDExpression {
  private mesh: THREE.SkinnedMesh | null = null;

  private dict: Record<string, number> = {};

  private base: Record<string, number> = {}; // resting-face morph -> value

  private cur: Record<string, number> = {}; // eased applied values (pre-blink)

  // Active emotion + its eased crossfade level (0 = resting, 1 = full preset).
  // `emotion` holds the LAST non-neutral preset so relaxing fades out FROM it.
  private emotion = "neutral";

  private emoLevel = 0;

  private emoTarget = 0;

  // Blink scheduler (performance.now()-based; delta-free so the viewer's
  // update() call site is unchanged).
  private blinking = false;

  private blinkT0 = 0;

  private nextBlinkAt = 0;

  // Prosody source (same volume array lip-sync gets) + eased speech energy.
  private audio: HTMLAudioElement | null = null;

  private volumes: number[] = [];

  private sliceMs = 20;

  private emphasis = NEUTRAL_EMPHASIS;

  setMesh(mesh: THREE.SkinnedMesh, dict: Record<string, number>) {
    this.mesh = mesh;
    this.dict = dict || {};
    this.cur = {};
    this.emotion = "neutral";
    this.emoLevel = 0;
    this.emoTarget = 0;
    this.blinking = false;
    this.nextBlinkAt = this.now() + BLINK_MIN_MS;
    try {
      const s = localStorage.getItem(LS_FACE);
      if (s) this.base = JSON.parse(s);
    } catch {
      /* ignore */
    }
    if (Object.keys(this.base).length === 0) this.base = this.defaultBase();
  }

  private now(): number {
    return typeof performance !== "undefined" ? performance.now() : Date.now();
  }

  /**
   * Default resting face: narrowed eyes (じと目) for a cool look, plus a little
   * 笑い to lift the lower lids (less round, gentle "smiling eyes") and にこり for
   * a faint mouth smile — narrowed but kind. Only includes morphs the model has.
   */
  private defaultBase(): Record<string, number> {
    const b: Record<string, number> = {};
    // Tuned via the 🎭 face panel — Miku's resting "cool but kind" look.
    // Exact morph keys verified against the PMX (note full-width ２ in ｷﾘｯ２/困る２).
    const want: Record<string, number> = {
      "笑い": 0.4,
      "ｷﾘｯ２": 0.6,
      "瞳小": 0.4,
      "口角上げ": 0.3,
      "真面目": 0.15,
      "困る２": 0.6,
      "下": 0.45,
      "HL移": 1.0,
    };
    for (const name of Object.keys(want)) {
      if (this.dict[name] != null) b[name] = want[name];
    }
    return b;
  }

  /**
   * Crossfade the expressive face to `emotion`'s preset (live-conversation path,
   * called per sentence). Accepts either a preset name or a raw emotion keyword
   * (joy_grin, anger_pout, ...) — resolved to the closest preset. Unknown/
   * "neutral" relaxes toward the resting face; the last real emotion's preset is
   * kept so the fade-out looks right.
   */
  playEmotion(emotion: string | null | undefined) {
    const key = emotion ? resolveFacePreset(emotion) : "neutral";
    if (key === "neutral") {
      this.emoTarget = 0; // relax to resting; keep `emotion` for the fade-out
    } else {
      this.emotion = key;
      this.emoTarget = 1;
    }
  }

  /** Ease the face back to the resting expression (call when speech ends). */
  relax() {
    this.emoTarget = 0;
  }

  /** Feed the per-sentence speech volume array (same one lip-sync gets) so the
   *  expression can pulse with prosody. */
  onAudioStart(audio: HTMLAudioElement, volumes: number[], sliceMs: number) {
    this.audio = audio;
    this.volumes = volumes || [];
    this.sliceMs = sliceMs || 20;
  }

  onAudioEnd() {
    this.audio = null;
    this.volumes = [];
  }

  /** Current speech energy 0..1 from the playback position (0 when silent). */
  private speechEnergy(): number {
    const a = this.audio;
    if (a && !a.paused && !a.ended && this.volumes.length) {
      const idx = Math.floor((a.currentTime * 1000) / this.sliceMs);
      const v = this.volumes[Math.min(idx, this.volumes.length - 1)] || 0;
      return Math.min(1, v * PROSODY_GAIN);
    }
    return -1; // no audio -> caller holds at NEUTRAL_EMPHASIS
  }

  /** Clear any saved face and restore the default combo. */
  resetBase() {
    this.base = this.defaultBase();
    try {
      localStorage.setItem(LS_FACE, JSON.stringify(this.base));
    } catch {
      /* ignore */
    }
    return { ...this.base };
  }

  clearMesh(mesh: THREE.SkinnedMesh) {
    if (this.mesh === mesh) {
      this.mesh = null;
      this.dict = {};
      this.cur = {};
      this.emotion = "neutral";
      this.emoLevel = 0;
      this.emoTarget = 0;
      this.audio = null;
      this.volumes = [];
    }
  }

  /** Set a resting-face morph (0 removes it). Persists. */
  setBase(name: string, value: number) {
    if (value <= 0) delete this.base[name];
    else this.base[name] = Math.min(1, value);
    try {
      localStorage.setItem(LS_FACE, JSON.stringify(this.base));
    } catch {
      /* ignore */
    }
  }

  getBase() {
    return { ...this.base };
  }

  morphNames() {
    return Object.keys(this.dict);
  }

  /** Compute the current blink amount (0 open .. 1 shut) and reschedule. */
  private blinkAmount(now: number): number {
    if (!this.blinking && now >= this.nextBlinkAt) {
      this.blinking = true;
      this.blinkT0 = now;
    }
    if (!this.blinking) return 0;
    const t = (now - this.blinkT0) / BLINK_MS;
    if (t >= 1) {
      this.blinking = false;
      // Math.random is fine in the browser runtime (not a workflow script).
      this.nextBlinkAt = now + BLINK_MIN_MS + Math.random() * BLINK_RND_MS;
      return 0;
    }
    return t < 0.5 ? t * 2 : (1 - t) * 2; // triangle: open -> shut -> open
  }

  /** Ease the face morphs toward their targets + blink. Call once per frame. */
  update() {
    const inf = this.mesh?.morphTargetInfluences;
    if (!this.mesh || !inf) return;
    const now = this.now();
    // Ease the emotion crossfade level (~0.4s to full).
    this.emoLevel += (this.emoTarget - this.emoLevel) * 0.08;
    // Prosody: ease speech energy, then scale the emotion strength by it so the
    // face pulses on emphasis (quiet -> PROSODY_FLOOR, loud -> full).
    const raw = this.speechEnergy();
    const energyTarget = raw < 0 ? NEUTRAL_EMPHASIS : raw;
    this.emphasis += (energyTarget - this.emphasis) * 0.18;
    const lvl =
      this.emoLevel * (PROSODY_FLOOR + (1 - PROSODY_FLOOR) * this.emphasis);
    const preset = EMOTION_PRESETS[this.emotion] || {};
    const names = new Set<string>([
      ...Object.keys(this.base),
      ...Object.keys(this.cur),
      ...Object.keys(preset),
      ...Object.keys(ALWAYS_ON),
    ]);
    // 1) smoothed expression (resting base crossfaded to the emotion preset).
    names.forEach((name) => {
      const idx = this.dict[name];
      if (idx == null) return;
      const restVal = this.base[name] || 0;
      const emoVal = preset[name] || 0;
      let target = restVal * (1 - lvl) + emoVal * lvl;
      if (ALWAYS_ON[name] != null) target = Math.max(target, ALWAYS_ON[name]);
      const c = this.cur[name] || 0;
      this.cur[name] = c + (target - c) * 0.25;
      inf[idx] = this.cur[name];
    });
    // 2) blink overlay: shut the lids over whatever eye shape is showing.
    const blink = this.blinkAmount(now);
    if (blink > 0) {
      names.forEach((name) => {
        if (!EYE_MORPHS.has(name)) return;
        const idx = this.dict[name];
        if (idx != null) inf[idx] = (this.cur[name] || 0) * (1 - blink);
      });
    }
    const bidx = this.dict["まばたき"];
    if (bidx != null) inf[bidx] = blink;
  }
}

export const mmdExpression = new MMDExpression();
