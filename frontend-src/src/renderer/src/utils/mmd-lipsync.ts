/* eslint-disable no-underscore-dangle */
import type * as THREE from "three";

/**
 * Drives an MMD model's mouth morph from the per-clip volume array the backend
 * already sends (one RMS sample per `sliceMs`), so Miku's mouth opens and closes
 * in time with her speech — the MMD equivalent of the Live2D wav lip-sync.
 *
 * The MMDViewer registers its mesh + mouth-morph index and calls `update()` once
 * per frame; the audio task calls `onAudioStart` / `onAudioEnd` around playback.
 * A singleton so the (always-mounted) audio task and the MMD viewer can reach the
 * same instance without prop drilling — mirrors audioManager / talkingMotion.
 */
class MMDLipSync {
  private mesh: THREE.SkinnedMesh | null = null;

  private mouthIdx = -1;

  private audio: HTMLAudioElement | null = null;

  private volumes: number[] = [];

  private sliceMs = 20;

  private smoothed = 0;

  // How wide the mouth opens. `gain` scales the raw volume; `cap` is the max
  // opening (1 = fully open). Tunable live via window.__mmdMouth.
  private gain = 0.95;

  private cap = 0.55; // max mouth-open (lower = less open, fewer teeth while talking)

  setMesh(mesh: THREE.SkinnedMesh, mouthIdx: number) {
    this.mesh = mesh;
    this.mouthIdx = mouthIdx;
  }

  /** Adjust mouth-open amount: gain scales volume, cap limits the maximum. */
  setGain(gain: number, cap = this.cap) {
    this.gain = gain;
    this.cap = cap;
  }

  clearMesh(mesh: THREE.SkinnedMesh) {
    if (this.mesh === mesh) {
      this.mesh = null;
      this.mouthIdx = -1;
      this.audio = null;
      this.volumes = [];
      this.smoothed = 0;
    }
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

  /** Set the mouth-open morph from the current playback position. Call per frame. */
  update() {
    if (!this.mesh || this.mouthIdx < 0 || !this.mesh.morphTargetInfluences) return;
    let target = 0;
    const a = this.audio;
    if (a && !a.paused && !a.ended && this.volumes.length) {
      const idx = Math.floor((a.currentTime * 1000) / this.sliceMs);
      const v = this.volumes[Math.min(idx, this.volumes.length - 1)] || 0;
      target = Math.min(this.cap, v * this.gain);
    }
    // Smooth so the mouth doesn't chatter between 20ms samples.
    this.smoothed += (target - this.smoothed) * 0.4;
    this.mesh.morphTargetInfluences[this.mouthIdx] = this.smoothed;
  }
}

export const mmdLipSync = new MMDLipSync();
