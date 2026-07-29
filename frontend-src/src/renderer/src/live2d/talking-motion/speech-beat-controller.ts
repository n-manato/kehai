import type { GestureType, MotionBehaviorConfig } from './types';

export interface ScheduledBeat {
  atSec: number; // absolute time (controller clock) to fire
  gesture: GestureType;
}

/**
 * SpeechBeatController — turns a sentence's text + spoken duration into a small
 * timeline of motion beats (nods, tilts, micro-pauses) placed at the points
 * where their punctuation roughly occurs. We have no word-level alignment, so
 * a punctuation mark at character index i fires at (i / length) * duration —
 * good enough to feel like the body is reacting to the speech rhythm.
 */
export class SpeechBeatController {
  constructor(private cfg: MotionBehaviorConfig) {}

  schedule(text: string, durationSec: number, nowSec: number): ScheduledBeat[] {
    const beats: ScheduledBeat[] = [];
    if (!text || durationSec <= 0) return beats;

    const len = Math.max(text.length, 1);
    const gestures = this.cfg.punctuationGestures || {};
    let lastAt = -Infinity;
    const minGapSec = this.cfg.minReactionIntervalMs / 1000;

    for (let i = 0; i < text.length; i += 1) {
      const gesture = gestures[text[i]];
      if (!gesture || gesture === 'none') continue;

      const at = nowSec + (i / len) * durationSec;
      // Don't overreact: keep beats spaced out (terminal nod/tilt still lands
      // because it's the last punctuation in the sentence).
      if (at - lastAt < minGapSec) continue;
      lastAt = at;
      beats.push({ atSec: at, gesture });
    }
    return beats;
  }
}
