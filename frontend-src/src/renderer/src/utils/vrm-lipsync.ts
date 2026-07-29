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
class VRMLipSync {
  private meshes: THREE.Mesh[] = [];

  private mouthIdx = -1;

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

  /** Bind all face primitives and resolve the あ (mouth-open) morph in the shared dict. */
  setMeshes(meshes: THREE.Mesh[], dict: Record<string, number>) {
    this.meshes = meshes || [];
    const d = dict || {};
    this.mouthIdx = d["あ"] ?? d["ア"] ?? d["A"] ?? d["a"] ?? -1;
    this.smoothed = 0;
  }

  hasMouth(): boolean {
    return this.mouthIdx >= 0;
  }

  /** Adjust mouth-open amount: gain scales volume, cap limits the maximum. */
  setGain(gain: number, cap = this.cap) {
    this.gain = gain;
    this.cap = cap;
  }

  clearMeshes() {
    this.meshes = [];
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
    if (this.mouthIdx < 0 || this.meshes.length === 0) return;
    let target = 0;
    if (this.liveSource) {
      // Streaming: mouth openness straight from the live PCM RMS (already 0..1).
      target = Math.min(this.cap, this.liveSource() * this.gain);
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
      const inf = this.meshes[i].morphTargetInfluences;
      if (inf && this.mouthIdx < inf.length) inf[this.mouthIdx] = this.smoothed + bias;
    }
  }
}

export const vrmLipSync = new VRMLipSync();
