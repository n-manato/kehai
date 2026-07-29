/* eslint-disable no-underscore-dangle */
import type * as THREE from "three";
import { aiDebug } from "./debug/ai-debug";

// Reproduces the MMD facial-expression system (see utils/mmd-expression.ts) on the
// VRM: the same resting "cool but kind" narrowed-eye base, per-sentence EMOTION
// presets crossfaded on top, a procedural blink, and prosody pulsing — driven by
// the SAME Japanese morph names (kept in the VRM at export). The one difference vs
// MMD: the VRM's face is split by material into MANY mesh primitives, each with its
// own morphTargetInfluences array (but a shared morph order), so every value is
// written to ALL face meshes.
// VRM's OWN saved resting face. v10: the user's OWN 🎭-panel tuning, baked as the
// default ("だいぶ近い"). Sharp cool eye via ｷﾘｯ２ + じと目 + 笑い２ (lower-lid lift is
// safe now that the eye/face MToon outline is off — no dark lower ring).
const LS_FACE = "vrmBaseFace10";

// Morph CHANNELS. Inside a channel the resting face and a preset are MUTUALLY EXCLUSIVE.
// The base sets one eye shape (ｷﾘｯ２ + 笑い２ + じと目) and one pupil look (瞳小); if a preset
// names any lid morph, stacking a narrow resting lid under a wide preset lid is exactly the
// "目がバグる" the user warned about. So naming ANY member of a channel takes over the WHOLE
// channel — the base members of it fade out — while channels the preset never mentions keep
// their resting value (emotionBaseKeep). Effects (照れ/赤面/汗/涙/！) are overlays and stay
// additive, which is why they are in no channel.
const CH_LID = new Set([
  "笑い", "笑い２", "笑い目", "じと目", "たれ目", "びっくり", "はぅ", "ｷﾘｯ", "ｷﾘｯ２",
  "なごみ", "はちゅ目", "はちゅ目２", "ウィンク", "ウィンク右", "ウィンク２", "ｳｨﾝｸ２右",
  "なぬ！", "恐ろしい子！", "カメラ目",
]);
// 星目 and はぁと are alternative PUPIL GRAPHICS and must replace each other.
// 瞳小 is not a graphic, it is how big the character's pupil is — structural. Leaving it in
// the takeover set meant that crossfading INTO 星目/はぁと faded 瞳小 out first, so the pupil
// visibly ballooned for the ~0.4s before the star/heart arrived. It now only moves when a
// preset names it (anger_rage / fear shrink it deliberately).
const CH_PUPIL = new Set(["星目", "はぁと"]);
const STRUCTURAL = new Set(["瞳小"]);

// Keyed to the ACTUAL emotion vocabulary this character emits (model_dict.json
// emotionMap for yyb_miku_10th): anger anger_rage disgust fear joy joy_big joy_smile
// look neutral sadness sadness_worry shy surprise surprise_shock thinking.
// Presets that no keyword can reach are marked — they exist for other characters.
const EMOTION_PRESETS: Record<string, Record<string, number>> = {
  neutral: {},
  happy: { 笑い: 0.85, 口角上げ: 0.7, にやり: 0.15, 照れ: 0.25, 赤面: 0.2 },
  joy_big: { はちゅ目: 0.6, 星目: 0.45, 口角上げ: 0.9, 照れ: 0.45, 赤面: 0.35 },
  joy_smile: { 笑い: 0.55, 口角上げ: 0.45, にこり: 0.35, 照れ: 0.15 },
  sad: { たれ目: 0.7, 困る２: 0.5, 口角下げ: 0.5, 涙: 0.4, 下: 0.3, 眉頭: 0.4 },
  sadness_worry: { 困る２: 0.65, たれ目: 0.4, 口角下げ: 0.2, 眉頭: 0.4, "汗 左": 0.25, "汗 右": 0.25 },
  angry: { ｷﾘｯ: 0.7, 怒り: 0.6, "怒り+左": 0.6, "怒り+右": 0.6, 口角下げ: 0.4, 眉頭: 0.2 },
  anger_rage: { ｷﾘｯ: 0.95, 瞳小: 0.7, "HL消１": 0.7, 怒り: 0.7, "怒り+左": 0.75, "怒り+右": 0.75, 口角下げ: 0.5, "赤面２": 0.25 },
  anger_pout: { じと目: 0.55, 怒り: 0.35, 口角下げ: 0.4, 赤面: 0.25, 眉頭: 0.3 },
  surprised: { びっくり: 0.85, 上: 0.6, 眉頭: 0.35, "！左": 0.45, "！右": 0.45 },
  // なぬ！ is already the blank-eyed shock face, so pairing it with a strong 瞳小 emptied
  // the eyes completely (white with pinprick dots — the "目がバグる" case). Keep it as an
  // ACCENT over びっくり and leave the pupil channel alone so the resting 瞳小 applies.
  surprise_shock: { びっくり: 0.7, "なぬ！": 0.35, 上: 0.4, "！左": 0.6, "！右": 0.6, "汗 左": 0.45, "汗 右": 0.45, 眉頭: 0.4 },
  fear: { びっくり: 0.6, 瞳小: 0.75, 困る２: 0.55, "汗 左": 0.6, "汗 右": 0.6, "！左": 0.3, "！右": 0.3 },
  shy: { はぅ: 0.55, 照れ: 0.85, 赤面: 0.5, "赤面２": 0.35, "赤面３": 0.25, 困る: 0.3, 口角上げ: 0.3 },
  thinking: { じと目: 0.6, 真面目: 0.4, 困る: 0.3, "上 左": 0.3, 平行: 0.35 },
  look: { じと目: 0.25, "上 左": 0.45, 平行: 0.3, にこり: 0.2 },
  // reachable only on characters whose emotionMap has these keywords
  joy_grin: { 笑い: 0.7, にやり: 0.35, 口角上げ: 0.85, 照れ: 0.25, 赤面: 0.25 },
  anger_glare: { ｷﾘｯ: 0.9, 瞳小: 0.6, 怒り: 0.5, "怒り+左": 0.5, "怒り+右": 0.5 },
  sadness_tearful: { たれ目: 0.75, 困る２: 0.6, 口角下げ: 0.55, 涙: 0.8, 眉頭: 0.5 },
  surprise_amazed: { はちゅ目２: 0.7, 星目: 0.7, 上: 0.5, 照れ: 0.15, 口角上げ: 0.4 },
  smug: { ウィンク２: 0.85, にやり: 0.5, "上 右": 0.45, 照れ: 0.2 },
  love: { はぁと: 0.8, 照れ: 0.6, 赤面: 0.5, "赤面３": 0.3, 笑い: 0.4 },
};

function resolveFacePreset(keyword: string): string {
  const k = keyword.toLowerCase();
  if (EMOTION_PRESETS[k]) return k;
  if (k.startsWith("joy")) {
    if (k.includes("grin")) return "joy_grin";
    if (k.includes("smile")) return "joy_smile";
    if (k.includes("big")) return "joy_big";
    return "happy";
  }
  if (k.startsWith("anger") || k.startsWith("angry")) {
    if (k.includes("pout") || k.includes("sulk")) return "anger_pout";
    if (k.includes("glare") || k.includes("serious")) return "anger_glare";
    if (k.includes("rage") || k.includes("fury")) return "anger_rage";
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
    if (k.includes("shock")) return "surprise_shock";
    return "surprised";
  }
  if (k === "fear") return "fear";
  if (k === "smug" || k === "smirk") return "smug";
  if (k.startsWith("love") || k.includes("heart")) return "love";
  if (k.startsWith("wink")) return "smug";
  if (k.startsWith("think")) return "thinking";
  if (k.startsWith("shy")) return "shy";
  if (k === "look") return "look";
  return "neutral";
}

const EYE_MORPHS = new Set([
  "笑い", "笑い２", "笑い目", "じと目", "たれ目", "びっくり", "はぅ",
  "ｷﾘｯ", "ｷﾘｯ２", "瞳小", "なごみ", "はちゅ目", "はちゅ目２", "星目", "はぁと",
]);

// HL移 (eye highlight) is structural — without it the eyes read as dead, so it is forced on.
// HL消１ is its counterpart (highlight OFF) and is the classic MMD rage face, so a preset that
// asks for it has to be able to WIN: the always-on floor is lifted while HL消１ is active.
const ALWAYS_ON: Record<string, number> = { HL移: 1.0 };
const ALWAYS_ON_SUPPRESS: Record<string, string> = { HL移: "HL消１" };

const BLINK_MS = 130;
const BLINK_MIN_MS = 2500;
const BLINK_RND_MS = 3800;

const PROSODY_FLOOR = 0.72;
const PROSODY_GAIN = 1.6;
const NEUTRAL_EMPHASIS = 0.8;

class VRMExpression {
  private meshes: THREE.Mesh[] = []; // all face primitives that share the morphs

  private dict: Record<string, number> = {};

  private base: Record<string, number> = {};

  private cur: Record<string, number> = {};

  private emotion = "neutral";

  private emoLevel = 0;

  private emoTarget = 0;

  private blinking = false;

  private blinkT0 = 0;

  private nextBlinkAt = 0;

  private audio: HTMLAudioElement | null = null;

  private volumes: number[] = [];

  private sliceMs = 20;

  /** How much of the RESTING face survives at full emotion, for morphs the preset does
   *  not name. 0 = the old behaviour (emotion wipes the resting face), 1 = the resting
   *  face is fully preserved and the emotion only adds what it names. */
  emotionBaseKeep = 0.8;

  private emphasis = NEUTRAL_EMPHASIS;

  /** Bind all VRM face meshes (each has its own morphTargetInfluences, shared order). */
  setMeshes(meshes: THREE.Mesh[], dict: Record<string, number>) {
    this.meshes = meshes || [];
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
    // msTC lip settings — the effect AUTHOR's own values (マシュマロ回答, r16): he said only
    // two morphs were added to the demo, the rest is the stock motion.
    //   ・あ２:-0.02  sinks the mouth centre a hair when closed so the seam line/エッジ
    //                disappears (口を閉じたときに口の真ん中をめり込ませてエッジを消す)
    //   ・口横縮小:0.35 narrows the mouth
    // These are now the REAL morphs. Both used to be unusable — the PMX→VRM export kept
    // only 52 of the model's 83 vertex morphs and dropped these two, and the old code also
    // spelled the second one 口幅縮小, which is not a morph name anywhere. With the
    // `dict[n] != null` guard that meant the whole block silently did nothing. The 31
    // missing morphs were re-injected into the .vrm from the PMX (scratchpad/
    // morph_inject.py), so the author's own numbers apply as written.
    // Amounts stay overridable: localStorage.vrmLipNarrow / __vrmLipNarrow(v).
    const num = (k: string, d: number) => {
      const v = parseFloat(localStorage.getItem(k) || "");
      return Number.isFinite(v) ? v : d;
    };
    const MSTC_LIP: Record<string, number> = { "あ２": -0.02, 口横縮小: num("vrmLipNarrow", 0.35) };
    for (const n of Object.keys(MSTC_LIP)) {
      if (this.dict[n] != null) this.base[n] = MSTC_LIP[n];
      else console.warn(`[vrm-face] morph ${n} missing — msTC lip setting not applied`);
    }
  }

  private now(): number {
    // Debug-aware: freezing the render dt does not stop a wall-clock schedule.
    return aiDebug.now();
  }

  /** Resting COOL HOODED-DROOPY eyes — the reference look: a lowered upper lid
   *  (まばたき ≈ half-close = the "hood") + a たれ目 outer downturn (the "slant").
   *  The blink animation adds ON TOP of the resting まばたき (see update()). Tuned
   *  against the target render; the user can adjust further via the 🎭 panel. */
  private defaultBase(): Record<string, number> {
    const b: Record<string, number> = {};
    // Copied VERBATIM from mmd-expression.ts defaultBase — the SHARP (ｷﾘｯ２) cool
    // resting eye the user tuned on the MMD model to match the target.
    const want: Record<string, number> = {
      "笑い２": 0.35, //   lower-lid lift (narrows; safe now the outline ring is gone)
      "ｷﾘｯ２": 0.35, //    KEEN/SHARP upper lid
      "じと目": 0.2, //    flat half-lid (cool narrowing)
      "瞳小": 0.35, //     smaller pupil
      "笑い目": 0.05,
      "口角上げ": 0.3,
      "真面目": 0.3,
      "困る２": 0.65,
      "HL移": 1.0, //     eye highlight
    };
    for (const name of Object.keys(want)) {
      if (this.dict[name] != null) b[name] = want[name];
    }
    return b;
  }

  /** Emotion presets OFF: hold the model's default face. Requested while the emotion ->
   *  face mapping is reworked; blink and lip-sync are untouched, so she still looks alive.
   *  Re-enable live with __vrmFace(true), or flip this default when the rework lands. */
  emotionEnabled = false;

  playEmotion(emotion: string | null | undefined) {
    if (!this.emotionEnabled) { this.emoTarget = 0; return; }
    const key = emotion ? resolveFacePreset(emotion) : "neutral";
    if (key === "neutral") {
      this.emoTarget = 0;
    } else {
      this.emotion = key;
      this.emoTarget = 1;
    }
  }

  relax() {
    this.emoTarget = 0;
  }

  onAudioStart(audio: HTMLAudioElement, volumes: number[], sliceMs: number) {
    this.audio = audio;
    this.volumes = volumes || [];
    this.sliceMs = sliceMs || 20;
  }

  onAudioEnd() {
    this.audio = null;
    this.volumes = [];
  }

  private speechEnergy(): number {
    const a = this.audio;
    if (a && !a.paused && !a.ended && this.volumes.length) {
      const idx = Math.floor((a.currentTime * 1000) / this.sliceMs);
      const v = this.volumes[Math.min(idx, this.volumes.length - 1)] || 0;
      return Math.min(1, v * PROSODY_GAIN);
    }
    return -1;
  }

  resetBase() {
    this.base = this.defaultBase();
    try {
      localStorage.setItem(LS_FACE, JSON.stringify(this.base));
    } catch {
      /* ignore */
    }
    return { ...this.base };
  }

  clearMeshes() {
    this.meshes = [];
    this.dict = {};
    this.cur = {};
    this.emotion = "neutral";
    this.emoLevel = 0;
    this.emoTarget = 0;
    this.audio = null;
    this.volumes = [];
  }

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

  private blinkAmount(now: number): number {
    if (!this.blinking && now >= this.nextBlinkAt) {
      this.blinking = true;
      this.blinkT0 = now;
    }
    if (!this.blinking) return 0;
    const t = (now - this.blinkT0) / BLINK_MS;
    if (t >= 1) {
      this.blinking = false;
      this.nextBlinkAt = now + BLINK_MIN_MS + aiDebug.random() * BLINK_RND_MS;
      return 0;
    }
    return t < 0.5 ? t * 2 : (1 - t) * 2;
  }

  /** Write a morph value to ALL face meshes (shared morph order). */
  private write(name: string, value: number) {
    const idx = this.dict[name];
    if (idx == null) return;
    for (let i = 0; i < this.meshes.length; i += 1) {
      const inf = this.meshes[i].morphTargetInfluences;
      if (inf && idx < inf.length) inf[idx] = value;
    }
  }

  /** Ease the face morphs toward their targets + blink. Call once per frame,
   *  AFTER vrm.update() so it overrides the VRM expressionManager. */
  update() {
    if (this.meshes.length === 0) return;
    const now = this.now();
    this.emoLevel += (this.emoTarget - this.emoLevel) * 0.08;
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
    // does this preset define its own eye shape / pupil? then it owns that whole channel
    let lidTaken = false; let pupilTaken = false;
    for (const k of Object.keys(preset)) {
      if (CH_LID.has(k)) lidTaken = true;
      if (CH_PUPIL.has(k)) pupilTaken = true;
    }
    const blink = this.blinkAmount(now);
    names.forEach((name) => {
      if (this.dict[name] == null) return;
      const restVal = this.base[name] || 0;
      const emoVal = preset[name] || 0;
      // An emotion should MODULATE the resting face, not erase it. The old blend
      // (rest*(1-lvl) + emo*lvl) applied to EVERY morph, so anything the preset did not
      // mention decayed to zero as the emotion came in — including the morphs that make
      // up her resting eye: measured 瞳小 0.349 -> 0.02, ｷﾘｯ２ 0.349 -> 0.02, じと目 0.20
      // -> 0.012 under any emotion. Losing 瞳小 in particular blows the pupils up, which
      // is the "目が大きくなりすぎる" report. Morphs the preset OWNS still crossfade
      // normally (so `angry` can override `笑い` etc.); morphs it says nothing about keep
      // `emotionBaseKeep` of their resting value. Live: __vrmFaceKeep(v).
      const owned = Object.prototype.hasOwnProperty.call(preset, name)
        || (lidTaken && CH_LID.has(name))
        || (pupilTaken && CH_PUPIL.has(name));
      // The PUPIL is structural, like the eye highlight: it is the character's eye, not a
      // mood. Emotions fire on nearly every sentence, so even the 20% decay left by
      // emotionBaseKeep pulled 瞳小 0.349 -> ~0.28 the moment she started talking, and the
      // pupils visibly grew ("しゃべると瞳が大きくなる"). Keep it at 100% unless a preset
      // OWNS the pupil channel — 星目 / はぁと / a deliberate 瞳小 still replace it outright.
      const keep = (CH_PUPIL.has(name) || STRUCTURAL.has(name)) ? 1 : this.emotionBaseKeep;
      let target = owned
        ? restVal * (1 - lvl) + emoVal * lvl
        : restVal * (1 - lvl * (1 - keep));
      if (ALWAYS_ON[name] != null) {
        const off = ALWAYS_ON_SUPPRESS[name];
        const suppressed = off ? (preset[off] || 0) * lvl : 0;
        target = Math.max(target, ALWAYS_ON[name] * (1 - suppressed));
      }
      const c = this.cur[name] || 0;
      this.cur[name] = c + (target - c) * 0.25;
      // まばたき is the resting "hood" — handled below so the blink adds on top.
      if (name === "まばたき") return;
      // blink shuts the lids over whatever eye shape is showing
      const out = blink > 0 && EYE_MORPHS.has(name)
        ? this.cur[name] * (1 - blink)
        : this.cur[name];
      this.write(name, out);
    });
    // Resting hood (まばたき in the base) + blink on top: rest .. 1.0 during a blink.
    if (this.dict["まばたき"] != null) {
      const rb = this.cur["まばたき"] || 0;
      this.write("まばたき", rb + blink * (1 - rb));
    }
  }
}

export const vrmExpression = new VRMExpression();
