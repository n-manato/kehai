/* eslint-disable no-underscore-dangle */
import type * as THREE from "three";

/**
 * Drives the VRM's mouth-open morph (あ) from the per-clip volume array the backend
 * already sends (one RMS sample per `sliceMs`), so Miku's mouth opens and closes in
 * time with her speech — the VRM equivalent of mmd-lipsync.ts / the Live2D wav lip
 * sync. The one difference vs MMD: the VRM face is split by material into MANY mesh
 * primitives that SHARE a morph order, so the mouth value is written to EVERY face
 * primitive (same gotcha as vrm-expression.ts) or it's invisible.
 *
 * The VRMViewer binds the meshes on load and calls `update()` once per frame (AFTER
 * vrm.update, like vrm-expression); the audio task calls onAudioStart/onAudioEnd
 * around playback. A singleton so the always-mounted audio task and the viewer reach
 * the same instance without prop drilling.
 */
/**
 * Mouth-open morph, best first.
 *
 * Japanese PMX names come first because Miku is the tuned case. `jawOpen` and
 * `aa` are what an ARKit/Oculus rig (Ready Player Me, Avaturn, Character
 * Creator) calls it; `jawOpen` is preferred over the `aa` viseme because this is
 * driven by RMS LOUDNESS, not by phonemes — it wants a generic opening, and a
 * viseme held at a fixed shape for a whole sentence reads as a grimace.
 *
 * Note `a` and `aa` are both listed and are NOT the same thing: `a` is a
 * Japanese-export spelling of あ, `aa` is the Oculus viseme.
 */
const MOUTH_CANDIDATES = [
  "あ", "ア", "A", "a",
  "jawOpen", "JawOpen", "jaw_open",
  "aa", "viseme_aa", "mouthOpen", "Mouth_Open",
];

/**
 * A scheduled mouth shape: viseme `v` at `t` seconds (on the page clock) for
 * `d` seconds.
 */
export interface VisemeCue {
  v: string;
  t: number;
  d: number;
}

/**
 * Fallbacks for viseme names a model might not carry. Ready Player Me and
 * Avaturn ship the bare Oculus names; Character Creator prefixes them; some
 * exports only have a jaw. Resolved once at bind time so the per-frame path
 * stays a lookup.
 */
const VISEME_ALIASES: Record<string, string[]> = {
  aa: ["aa", "viseme_aa", "AA", "jawOpen"],
  E: ["E", "viseme_E", "EE", "jawOpen"],
  ih: ["ih", "viseme_I", "IH", "jawOpen"],
  oh: ["oh", "viseme_O", "OH", "jawOpen"],
  ou: ["ou", "viseme_U", "OU", "jawOpen"],
  PP: ["PP", "viseme_PP", "mouthClose"],
  FF: ["FF", "viseme_FF"],
  TH: ["TH", "viseme_TH"],
  DD: ["DD", "viseme_DD"],
  kk: ["kk", "viseme_kk"],
  CH: ["CH", "viseme_CH"],
  SS: ["SS", "viseme_SS"],
  nn: ["nn", "viseme_nn"],
  RR: ["RR", "viseme_RR"],
  sil: ["sil", "viseme_sil"],
};

class VRMLipSync {
  private meshes: THREE.Mesh[] = [];

  /** viseme name -> per-mesh morph index (-1 where a mesh lacks it). */
  private visemeIdx: Record<string, number[]> = {};

  private cues: VisemeCue[] = [];

  /** Current blend per viseme, so shapes cross-fade instead of snapping. */
  private vLevel: Record<string, number> = {};

  /**
   * Morph index PER MESH, not one shared index.
   *
   * The original code resolved a single index from a merged dictionary, on the
   * stated assumption that a VRM face is split by material but shares one morph
   * ORDER. That holds for the Miku export and is false in general: the Avaturn
   * .glb's primitives carry 149 / 67 / 20 / 3 morphs respectively, so a single
   * index writes to a different morph on every mesh — the head would open its
   * mouth while the eyelashes ran whatever their morph 42 happened to be.
   */
  private mouthIdxPerMesh: number[] = [];

  private mouthIdx = -1;

  private mouthName = "";

  private audio: HTMLAudioElement | null = null;

  private volumes: number[] = [];

  private sliceMs = 20;

  // Streaming (true-TTS-streaming) drive: a live mouth-openness provider (0..1)
  // read from the streamed PCM's RMS instead of `audio.currentTime`+volumes[].
  // When set it takes precedence in update(). Cleared by onLiveEnd().
  private liveSource: (() => number) | null = null;

  private smoothed = 0;

  // How wide the mouth opens. `gain` scales the raw volume; `cap` is the max
  // opening (1 = fully open). Tunable live via window.__vrmMouth.
  private gain = 0.95;

  private cap = 0.55; // max mouth-open (lower = fewer teeth while talking)

  // Closed-mouth あ bias — the msTC author's あ２:-0.02 approximated on あ.
  idleBias = 0; // see below: あ２ is a real morph again, so あ needs no stand-in

  /**
   * Bind the face primitives and resolve the mouth morph on EACH of them.
   *
   * `dict` is accepted for compatibility with the old call and is only used to
   * choose WHICH name to drive; the indices themselves come from each mesh, for
   * the reason in `mouthIdxPerMesh`.
   */
  setMeshes(meshes: THREE.Mesh[], dict?: Record<string, number>) {
    this.meshes = meshes || [];
    this.smoothed = 0;
    this.mouthIdxPerMesh = [];
    this.mouthIdx = -1;
    this.mouthName = "";

    // Pick the name once, from everything the model offers, so all primitives
    // drive the SAME morph even where their indices differ.
    const offered = new Set<string>(Object.keys(dict || {}));
    for (const m of this.meshes) {
      const d = m.morphTargetDictionary as Record<string, number> | undefined;
      if (d) for (const k of Object.keys(d)) offered.add(k);
    }
    this.mouthName = MOUTH_CANDIDATES.find((n) => offered.has(n)) || "";
    if (!this.mouthName) return;

    this.mouthIdxPerMesh = this.meshes.map((m) => {
      const d = m.morphTargetDictionary as Record<string, number> | undefined;
      const i = d?.[this.mouthName];
      return i == null ? -1 : i;
    });
    this.mouthIdx = this.mouthIdxPerMesh.findIndex((i) => i >= 0) >= 0 ? 1 : -1;

    // Resolve the viseme set once. A model that has none simply keeps the
    // volume-driven jaw, which is why this is a separate, optional layer rather
    // than a replacement.
    this.visemeIdx = {};
    this.vLevel = {};
    for (const [name, aliases] of Object.entries(VISEME_ALIASES)) {
      const perMesh = this.meshes.map((m) => {
        const d = m.morphTargetDictionary as Record<string, number> | undefined;
        if (!d) return -1;
        for (const a of aliases) if (d[a] != null) return d[a];
        return -1;
      });
      if (perMesh.some((i) => i >= 0)) {
        this.visemeIdx[name] = perMesh;
        this.vLevel[name] = 0;
      }
    }
    this.cues = [];
  }

  /** Does this model have real mouth shapes, or only a jaw? */
  hasVisemes(): boolean {
    // `sil` alone is not a mouth; require actual articulation.
    return Object.keys(this.visemeIdx).filter((v) => v !== "sil").length >= 4;
  }

  visemeNames(): string[] {
    return Object.keys(this.visemeIdx);
  }

  /**
   * Schedule one sentence's mouth shapes.
   *
   * `startAt` is when that sentence's audio begins on the page clock — the
   * caller owns it because only it knows the playback backlog. Cues already in
   * the past are kept: a late-arriving sentence should catch up mid-shape rather
   * than skip the mouth entirely.
   */
  setVisemes(cues: VisemeCue[], startAt: number) {
    if (!cues || !cues.length) return;
    const now = performance.now() / 1000;
    const base = startAt > 0 ? startAt : now;
    const mapped = cues.map((c) => ({ v: c.v, t: base + c.t, d: c.d }));
    // Drop anything that has already fully passed, then keep the queue ordered.
    this.cues = this.cues
      .filter((c) => c.t + c.d > now)
      .concat(mapped.filter((c) => c.t + c.d > now))
      .sort((a, b) => a.t - b.t);
  }

  clearVisemes() {
    this.cues = [];
    for (const k of Object.keys(this.vLevel)) this.vLevel[k] = 0;
  }

  hasMouth(): boolean {
    return this.mouthIdx >= 0;
  }

  /** Which morph is being driven — for diagnosing a silent mouth. */
  mouthMorph(): string {
    return this.mouthName;
  }

  /** Adjust mouth-open amount: gain scales volume, cap limits the maximum. */
  setGain(gain: number, cap = this.cap) {
    this.gain = gain;
    this.cap = cap;
  }

  /**
   * Drive the viseme morphs from the scheduled cues.
   *
   * Every known viseme is eased toward its target each frame rather than set
   * outright. Snapping between shapes at 20-30 changes a second reads as a
   * stutter; a short ease reads as a mouth. The rise is faster than the fall so
   * shapes still land crisply on their consonant.
   */
  private updateVisemes() {
    const now = performance.now() / 1000;
    // Retire cues that have finished, but keep the array small rather than
    // rebuilding it every frame.
    while (this.cues.length && this.cues[0].t + this.cues[0].d < now - 0.25) {
      this.cues.shift();
    }
    const active: Record<string, number> = {};
    for (const c of this.cues) {
      if (c.t > now) break; // ordered, so nothing later has started
      if (now <= c.t + c.d) {
        active[c.v] = Math.max(active[c.v] || 0, 1);
      }
    }
    const RISE = 0.55;
    const FALL = 0.28;
    for (const name of Object.keys(this.visemeIdx)) {
      const want = name === "sil" ? 0 : (active[name] || 0) * this.cap * 1.8;
      const cur = this.vLevel[name] || 0;
      const k = want > cur ? RISE : FALL;
      const next = cur + (want - cur) * k;
      this.vLevel[name] = next < 0.001 ? 0 : next;
      const idxs = this.visemeIdx[name];
      for (let i = 0; i < this.meshes.length; i += 1) {
        const idx = idxs[i];
        if (idx < 0) continue;
        const inf = this.meshes[i].morphTargetInfluences;
        if (inf && idx < inf.length) inf[idx] = this.vLevel[name];
      }
    }
  }

  clearMeshes() {
    this.meshes = [];
    this.cues = [];
    this.visemeIdx = {};
    this.vLevel = {};
    this.mouthIdxPerMesh = [];
    this.mouthName = "";
    this.mouthIdx = -1;
    this.audio = null;
    this.volumes = [];
    this.smoothed = 0;
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

  /** Streaming drive: `mouthFn` returns 0..1 mouth openness read from the live
   *  streamed-PCM RMS (StreamingAudioPlayer.mouthValue). Takes precedence over the
   *  audio+volumes path in update(); cleared by onLiveEnd() at turn end/interrupt. */
  onLiveStart(mouthFn: () => number) {
    this.liveSource = mouthFn;
  }

  onLiveEnd() {
    this.liveSource = null;
    this.smoothed = 0;
  }

  /** Set the mouth-open morph from the current playback position. Call per frame,
   *  AFTER vrm.update() so it overrides the expressionManager. */
  update() {
    // Real mouth shapes win whenever a sentence has supplied them. They are not
    // BLENDED with the volume-driven jaw: the jaw morph is one of the shapes the
    // viseme set already drives, so running both would fight over the same
    // number and wash the articulation out.
    if (this.cues.length && this.hasVisemes()) {
      this.updateVisemes();
      return;
    }
    if (this.mouthIdx < 0 || this.meshes.length === 0) return;
    // A model WITH visemes is only on the volume-driven jaw until this sentence's
    // cues arrive — they ride the sentence's END frame, because the character
    // timings only complete with the last audio chunk. So the opening of every
    // reply runs on volume alone, and volume moves ONE morph: the jaw. A jaw
    // hanging open with no lip shaping is what "the mouth opens too wide at the
    // start, then it's fine" is describing. Half the travel while it is a
    // stopgap; the cues take over with the full range the moment they land.
    const stopgap = this.hasVisemes() ? 0.5 : 1;
    const cap = this.cap * stopgap;
    let target = 0;
    if (this.liveSource) {
      // Streaming: mouth openness straight from the live PCM RMS (already 0..1).
      target = Math.min(cap, this.liveSource() * this.gain);
    } else {
      const a = this.audio;
      if (a && !a.paused && !a.ended && this.volumes.length) {
        const idx = Math.floor((a.currentTime * 1000) / this.sliceMs);
        const v = this.volumes[Math.min(idx, this.volumes.length - 1)] || 0;
        target = Math.min(this.cap, v * this.gain);
      }
    }
    // Smooth so the mouth doesn't chatter between 20ms samples.
    this.smoothed += (target - this.smoothed) * 0.4;
    // The author's 「あ２:-0.02」 mouth-centre sink used to be approximated here as a
    // negative bias on あ, because あ２ had been dropped by the PMX→VRM export. It has
    // since been re-injected into the .vrm, so vrm-expression applies the real morph in
    // the resting base and this stand-in defaults to 0 — leaving it on would apply the
    // sink twice. Still available for taste: __vrmLipBias(v).
    const bias = this.idleBias * Math.max(0, 1 - this.smoothed / 0.15);
    for (let i = 0; i < this.meshes.length; i += 1) {
      const idx = this.mouthIdxPerMesh[i];
      if (idx < 0) continue; // this primitive simply has no mouth morph
      const inf = this.meshes[i].morphTargetInfluences;
      if (inf && idx < inf.length) inf[idx] = this.smoothed + bias;
    }
  }
}

export const vrmLipSync = new VRMLipSync();
