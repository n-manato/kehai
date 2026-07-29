/* eslint-disable no-underscore-dangle */
import { CubismFramework } from '@framework/live2dcubismframework';
import type { Emotion, EmotionMotionState, GestureType } from './types';
import { motionConfig, strengthFor } from './config';
import { SpeechBeatController, type ScheduledBeat } from './speech-beat-controller';
import { KeywordReactionController } from './keyword-reaction-controller';
import { MotionTransitionController } from './motion-transition-controller';

// Parameters this layer may nudge (head / body / gaze / brows + a few emotional
// "wetness" params). NEVER mouth lip-sync (ParamMouthOpenY), eye blink
// (ParamEyeLOpen/ROpen), or breath (ParamBreath).
const PARAM_NAMES = {
  angleX: 'ParamAngleX',
  angleY: 'ParamAngleY',
  angleZ: 'ParamAngleZ',
  bodyX: 'ParamBodyAngleX',
  bodyY: 'ParamBodyAngleY',
  bodyZ: 'ParamBodyAngleZ',
  eyeX: 'ParamEyeBallX',
  eyeY: 'ParamEyeBallY',
  eyeLOpen: 'ParamEyeLOpen', // only nudged for the nod-synced blink (see below)
  eyeROpen: 'ParamEyeROpen',
  browL: 'ParamBrowLY',
  browR: 'ParamBrowRY',
  shoulder: 'ParamShoulder',
  cheek: 'ParamCheek',
  tear: 'ParamBxNamida',
  eyeHi: 'ParamEyeHi',
} as const;
type ParamKey = keyof typeof PARAM_NAMES;
// Cached once — avoids allocating a keys array every frame (GC stutter).
const PARAM_KEYS = Object.keys(PARAM_NAMES) as ParamKey[];

type Mode = 'speaking' | 'thinking' | 'listening' | 'idle';

interface Impulse {
  param: ParamKey;
  peak: number;
  startSec: number;
  durSec: number;
}

export interface ReactionSinks {
  setExpression?: (name: string) => void;
  playMotion?: (group: string) => void;
}

const rnd = (a: number, b: number): number => a + Math.random() * (b - a);
const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, v));
const num = (v: number | undefined, d: number): number => (typeof v === 'number' ? v : d);

function emotionFromMotionGroup(group?: string | null): Emotion {
  if (!group) return 'neutral';
  const g = group.toLowerCase();
  if (g.startsWith('joy')) return 'happy';
  if (g.startsWith('sadness')) return 'sad';
  if (g.startsWith('surprise')) return 'surprised';
  if (g.startsWith('thinking') || g === 'look') return 'thinking';
  if (g.startsWith('anger')) return 'nervous';
  if (g.startsWith('idle')) return 'calm';
  return 'neutral';
}

/** Two-sine pseudo-noise in [-1, 1]. `seed` varies it per session. */
function wobble(t: number, f1: number, f2: number, phase: number): number {
  return (Math.sin(t * f1 + phase) + 0.5 * Math.sin(t * f2 + phase * 1.7)) / 1.5;
}

/** Eased beat envelope: fast attack, settle to 0 with a small recoil. */
function beatShape(p: number): number {
  const attack = 0.3;
  if (p < attack) {
    const k = p / attack;
    return 1 - (1 - k) * (1 - k) * (1 - k);
  }
  const r = (p - attack) / (1 - attack);
  return Math.cos((r * Math.PI) / 2) - 0.15 * Math.sin(r * Math.PI);
}

class TalkingMotionController {
  private clockSec = 0;
  private talkingLevel = 0;
  private listeningLevel = 0;
  private thinkingLevel = 0;
  private swayStrength = 0; // smoothed movement intensity (avoids mode-change steps)
  private speakingUntilSec = -Infinity;
  private listening = false;
  private thinking = false;
  private emotion: Emotion = 'neutral';
  private energy = motionConfig.defaultEnergy;
  private lastGesture: GestureType = 'none';
  private replyActive = false; // true between first sentence and speech end

  private pendingBeats: ScheduledBeat[] = [];
  private impulses: Impulse[] = [];
  private lastNodSec = -Infinity;
  private nextListenNodSec = 0;
  private nextFidgetSec = 0;
  private nodBlinkWanted = false; // a nod fired -> blink with it (if eyes are free)

  // Prosody (A).
  private envelope: number[] | null = null;
  private sliceSec = 0.02;
  private sentenceStartSec = 0;
  private loud = 0;

  // Gaze + posture state machines.
  private gaze = { x: 0, y: 0, tx: 0, ty: 0, nextSec: 0 };
  private posture = { x: 0, tx: 0, nextSec: 0 };

  // Smoothed emotion bias + emotional "wetness" state (cheek/tear/eyeHi).
  private bias: Record<ParamKey, number> = {} as Record<ParamKey, number>;
  private biasTarget: Record<ParamKey, number> = {} as Record<ParamKey, number>;
  private wet = { cheek: 0, tear: 0, eyeHi: 0 };
  private expression = '';

  // Per-session "personality": small fixed phase offset so no two runs move identically.
  private readonly seed = Math.random() * 1000;

  private ids: Partial<Record<ParamKey, unknown>> = {};
  private idsResolved = false;
  private _loggedActive = false;
  private forceDisabled = false; // runtime A/B toggle (window.__talkingMotion.setEnabled)

  private readonly beat = new SpeechBeatController(motionConfig);
  private readonly keyword = new KeywordReactionController(motionConfig);
  private readonly transition = new MotionTransitionController();
  private sinks: ReactionSinks = {};

  registerSinks(sinks: ReactionSinks): void {
    this.sinks = sinks;
  }

  get state(): EmotionMotionState {
    return {
      speaking: this.clockSec < this.speakingUntilSec,
      emotion: this.emotion,
      energy: this.energy,
      gesture: this.lastGesture,
    };
  }

  private log(msg: string): void {
    if (motionConfig.debugLog) console.log(`[TalkingMotion] ${msg}`);
  }

  /** Runtime on/off (for A/B testing whether this layer causes stutter). */
  setEnabled(on: boolean): void {
    this.forceDisabled = !on;
    console.info(`[TalkingMotion] ${on ? 'enabled' : 'DISABLED'} at runtime`);
  }

  setListening(on: boolean): void {
    if (on !== this.listening) {
      this.listening = on;
      this.log(`listening = ${on}`);
    }
  }

  /** Called when a reply starts generating (before any audio): "thinking" pose. */
  onThinkingStart(): void {
    this.thinking = true;
    this.replyActive = false;
    this.log('thinking start');
  }

  onSentenceStart(
    text: string,
    durationSec: number,
    motionGroup?: string | null,
    volumes?: number[] | null,
    sliceLengthMs?: number,
    expression?: string | null,
  ): void {
    if (!motionConfig.enableTalkingMotion) return;

    this.thinking = false; // we're talking now, not thinking
    this.emotion = emotionFromMotionGroup(motionGroup);
    this.expression = (expression ?? '').toLowerCase();
    const dur = durationSec > 0 && Number.isFinite(durationSec) ? durationSec : 2.0;

    // Energy from sentence length/pace (4): longer/faster -> livelier.
    if (motionConfig.energyFromLength) {
      const cps = text.length / Math.max(dur, 0.3);
      this.energy = clamp(0.25 + cps * 0.05 + Math.min(text.length, 60) / 130, 0.2, 1);
    } else {
      this.energy = motionConfig.defaultEnergy;
    }

    // Anticipation (2): a small "gear up" when a reply first begins. Head only —
    // no body turn here, so it doesn't fight the emotion motion's body pose at
    // the very start of speaking.
    if (!this.replyActive) {
      this.impulses.push({ param: 'angleY', peak: 4, startSec: this.clockSec, durSec: 0.5 });
      this.replyActive = true;
    }

    this.speakingUntilSec = this.clockSec + dur + 0.35;
    this.envelope = volumes && volumes.length > 0 ? volumes : null;
    this.sliceSec = sliceLengthMs && sliceLengthMs > 0 ? sliceLengthMs / 1000 : 0.02;
    this.sentenceStartSec = this.clockSec;

    this.log(`speech start | emotion=${this.emotion} energy=${this.energy.toFixed(2)} dur=${dur.toFixed(2)}s`);

    this.pendingBeats.push(...this.beat.schedule(text, dur, this.clockSec));

    const result = this.keyword.scan(text, this.clockSec);
    if (result && 'skipped' in result && result.skipped) {
      this.log(`keyword skipped: ${result.reason}`);
    } else if (result && 'reaction' in result) {
      const r = result.reaction;
      const priority = r.priority ?? 1;
      if (this.transition.request(priority, 0.9, this.clockSec)) {
        this.log(`keyword reaction: ${r.reaction} (priority ${priority})`);
        if (r.gesture) this.pendingBeats.push({ atSec: this.clockSec + 0.05, gesture: r.gesture });
        if (r.expression && this.sinks.setExpression) this.sinks.setExpression(r.expression);
        if (r.motionGroup && this.sinks.playMotion) this.sinks.playMotion(r.motionGroup);
      } else {
        this.log(`keyword reaction ${r.reaction} skipped: lower priority`);
      }
    }
  }

  /** Called when the whole reply finishes (or is interrupted): settle + fade. */
  onSpeechEnd(): void {
    if (this.speakingUntilSec > this.clockSec) this.log('speech end -> settle + idle');
    this.speakingUntilSec = this.clockSec;
    this.envelope = null;
    this.thinking = false;
    // Return to a neutral emotional state so blush / tears / eye-highlight and
    // the head bias ramp back to idle (otherwise the last sentence's expression
    // lingered on the face after speech ended).
    this.emotion = 'neutral';
    this.expression = '';
    if (this.replyActive) {
      // Post-speech "settle": relax shoulders + a small breath-out.
      this.impulses.push({ param: 'shoulder', peak: -1.5, startSec: this.clockSec, durSec: 0.9 });
      this.impulses.push({ param: 'angleY', peak: 2.5, startSec: this.clockSec, durSec: 0.8 });
      this.replyActive = false;
    }
  }

  private resolveIds(): void {
    const mgr = CubismFramework.getIdManager?.();
    if (!mgr) return;
    PARAM_KEYS.forEach((key) => {
      this.ids[key] = mgr.getId(PARAM_NAMES[key]);
    });
    this.idsResolved = true;
  }

  private gestureToImpulses(gesture: GestureType, strength: number): Impulse[] {
    const now = this.clockSec;
    const s = Math.max(strength, 0.3);
    const brow = num(motionConfig.browStrength, 0.6);
    const dir = wobble(now, 11, 7, this.seed) >= 0 ? 1 : -1;
    switch (gesture) {
      case 'small_nod':
        return [{ param: 'angleY', peak: -11 * s, startSec: now, durSec: 0.5 }];
      case 'head_tilt':
        return [
          { param: 'angleZ', peak: 9 * s * dir, startSec: now, durSec: 0.9 },
          { param: 'angleX', peak: 4 * s * dir, startSec: now, durSec: 0.9 },
          { param: 'browL', peak: 0.3 * brow, startSec: now, durSec: 0.8 },
          { param: 'browR', peak: 0.3 * brow, startSec: now, durSec: 0.8 },
        ];
      case 'emphasis':
        return [
          { param: 'angleY', peak: -13 * s, startSec: now, durSec: 0.35 },
          { param: 'bodyX', peak: 4 * s, startSec: now, durSec: 0.5 },
          { param: 'browL', peak: 0.4 * brow, startSec: now, durSec: 0.4 },
          { param: 'browR', peak: 0.4 * brow, startSec: now, durSec: 0.4 },
        ];
      case 'micro_pause':
        return [{ param: 'angleX', peak: 3 * s * dir, startSec: now, durSec: 0.4 }];
      case 'surprise_pop':
        return [
          { param: 'angleY', peak: 11 * s, startSec: now, durSec: 0.3 },
          { param: 'eyeY', peak: 0.25 * s, startSec: now, durSec: 0.4 },
          { param: 'browL', peak: 0.6 * brow, startSec: now, durSec: 0.5 },
          { param: 'browR', peak: 0.6 * brow, startSec: now, durSec: 0.5 },
        ];
      default:
        return [];
    }
  }

  private fireDueBeats(): void {
    if (this.pendingBeats.length === 0) return;
    const nodCooldown = motionConfig.nodCooldownMs / 1000;
    const strength = strengthFor(this.emotion);
    const remaining: ScheduledBeat[] = [];
    for (const b of this.pendingBeats) {
      if (b.atSec > this.clockSec) {
        remaining.push(b);
        continue;
      }
      const isNod = b.gesture === 'small_nod' || b.gesture === 'emphasis';
      if (isNod && this.clockSec - this.lastNodSec < nodCooldown) {
        this.log(`gesture ${b.gesture} skipped: nod cooldown`);
        continue;
      }
      if (isNod) {
        this.lastNodSec = this.clockSec;
        this.nodBlinkWanted = true; // blink along with this nod
      }
      this.lastGesture = b.gesture;
      this.impulses.push(...this.gestureToImpulses(b.gesture, strength));
      this.log(`gesture: ${b.gesture}`);
    }
    this.pendingBeats = remaining;
  }

  /** Random unconscious micro-fidgets (brow flash, tiny gaze, small tilt). */
  private maybeFidget(): void {
    if (this.clockSec < this.nextFidgetSec) return;
    const fMin = num(motionConfig.fidgetMinMs, 4000);
    const fMax = num(motionConfig.fidgetMaxMs, 12000);
    this.nextFidgetSec = this.clockSec + rnd(fMin, fMax) / 1000;
    const brow = num(motionConfig.browStrength, 0.6);
    const r = Math.random();
    const now = this.clockSec;
    if (r < 0.4) {
      this.impulses.push({ param: 'browL', peak: 0.25 * brow, startSec: now, durSec: 0.4 });
      this.impulses.push({ param: 'browR', peak: 0.25 * brow, startSec: now, durSec: 0.4 });
    } else if (r < 0.7) {
      this.impulses.push({ param: 'angleZ', peak: rnd(-3, 3), startSec: now, durSec: 0.8 });
    } else {
      // nudge gaze to a fresh target soon
      this.gaze.nextSec = now;
    }
    this.log('fidget');
  }

  private updateGaze(dt: number, mode: Mode): void {
    const gMin = num(motionConfig.gazeMinIntervalMs, 700);
    const gMax = num(motionConfig.gazeMaxIntervalMs, 2600);
    if (this.clockSec >= this.gaze.nextSec) {
      const r = Math.random();
      const avert = mode === 'thinking' || (mode === 'speaking' && this.emotion === 'thinking');
      if (avert && r < 0.6) {
        this.gaze.tx = (Math.random() < 0.5 ? -1 : 1) * rnd(0.4, 0.8);
        this.gaze.ty = -rnd(0.3, 0.6);
        this.gaze.nextSec = this.clockSec + rnd(0.9, 1.9);
      } else if (r < (mode === 'listening' ? 0.55 : 0.25)) {
        this.gaze.tx = rnd(-0.08, 0.08);
        this.gaze.ty = rnd(-0.05, 0.05);
        this.gaze.nextSec = this.clockSec + rnd(gMin, gMax) / 1000;
      } else {
        const scale = mode === 'idle' ? 0.18 : mode === 'listening' ? 0.22 : 0.45;
        this.gaze.tx = rnd(-scale, scale);
        this.gaze.ty = rnd(-scale * 0.6, scale * 0.6);
        const slow = mode === 'idle' ? 1.6 : 1;
        this.gaze.nextSec = this.clockSec + (rnd(gMin, gMax) / 1000) * slow;
      }
    }
    const k = Math.min(1, dt / 0.12);
    this.gaze.x += (this.gaze.tx - this.gaze.x) * k;
    this.gaze.y += (this.gaze.ty - this.gaze.y) * k;
  }

  private updatePosture(dt: number): void {
    const pMin = num(motionConfig.postureShiftMinMs, 5000);
    const pMax = num(motionConfig.postureShiftMaxMs, 11000);
    if (this.clockSec >= this.posture.nextSec) {
      this.posture.tx = rnd(-1, 1) * 4;
      this.posture.nextSec = this.clockSec + rnd(pMin, pMax) / 1000;
    }
    this.posture.x += (this.posture.tx - this.posture.x) * Math.min(1, dt / 0.6);
  }

  private sampleProsody(dt: number, speaking: boolean): number {
    let target = 0;
    if (speaking && this.envelope && this.sliceSec > 0) {
      const idx = Math.floor((this.clockSec - this.sentenceStartSec) / this.sliceSec);
      if (idx >= 0 && idx < this.envelope.length) target = this.envelope[idx] ?? 0;
    }
    // Per-chunk RMS is spiky (syllable level); smooth heavily (~250ms) so it
    // modulates motion gently instead of making the head jitter frame-to-frame.
    this.loud += (target - this.loud) * Math.min(1, dt / 0.25);
    return this.loud;
  }

  /** Target emotion-bias (snapped); smoothed in apply via this.bias. */
  private emotionBiasTarget(level: number): Record<ParamKey, number> {
    const brow = num(motionConfig.browStrength, 0.6);
    const out = this.biasTarget; // reused object (no per-frame allocation)
    PARAM_KEYS.forEach((k) => { out[k] = 0; });
    switch (this.emotion) {
      case 'sad':
        out.angleY = -5 * level;
        out.browL = -0.2 * brow * level;
        out.browR = -0.2 * brow * level;
        break;
      case 'thinking':
        out.angleZ = 4 * level;
        out.browL = -0.12 * brow * level;
        out.browR = -0.12 * brow * level;
        break;
      case 'happy':
        out.angleY = 2 * level;
        out.browL = 0.1 * brow * level;
        out.browR = 0.1 * brow * level;
        break;
      default:
        break;
    }
    return out;
  }

  /** Emotional "wetness" targets (blush / tears / eye sparkle) — smoothed. */
  private updateWet(dt: number): void {
    const k = Math.min(1, (dt * 1000) / num(motionConfig.emotionSmoothMs, 400));
    const shy = this.expression.includes('cheek');
    const teary = this.expression.includes('uru') || this.expression.includes('kanashimu');
    const cheekT = (shy ? 0.7 : this.emotion === 'happy' ? 0.15 : 0) * num(motionConfig.cheekStrength, 0.6);
    const tearT = ((teary ? 0.6 : 0) + (this.emotion === 'sad' ? 0.2 : 0)) * num(motionConfig.tearStrength, 0.5);
    const hiT = (this.emotion === 'happy' || this.emotion === 'surprised' ? 0.4 : 0) * num(motionConfig.eyeHiStrength, 0.4);
    this.wet.cheek += (cheekT - this.wet.cheek) * k;
    this.wet.tear += (tearT - this.wet.tear) * k;
    this.wet.eyeHi += (hiT - this.wet.eyeHi) * k;
  }

  apply(
    model: {
      addParameterValueById: (id: unknown, v: number) => void;
      getParameterValueById: (id: unknown) => number;
    },
    dt: number,
  ): void {
    if (this.forceDisabled || !motionConfig.enableTalkingMotion) return;
    this.clockSec += dt;
    const t = this.clockSec;

    const speaking = t < this.speakingUntilSec;
    const fade = (lvl: number, target: number): number => {
      const fadeMs = target > lvl ? motionConfig.talkingFadeInMs : motionConfig.talkingFadeOutMs;
      const step = fadeMs > 0 ? (dt * 1000) / fadeMs : 1;
      return target > lvl ? Math.min(1, lvl + step) : Math.max(0, lvl - step);
    };
    this.talkingLevel = fade(this.talkingLevel, speaking ? 1 : 0);
    this.listeningLevel = fade(this.listeningLevel, this.listening && !speaking ? 1 : 0);
    this.thinkingLevel = fade(this.thinkingLevel, this.thinking && !speaking ? 1 : 0);

    this.fireDueBeats();
    this.maybeFidget();

    // Attentive nods while listening.
    if (this.listening && !speaking && t >= this.nextListenNodSec) {
      this.impulses.push({ param: 'angleY', peak: -6, startSec: t, durSec: 0.55 });
      if (Math.random() < 0.4) this.impulses.push({ param: 'angleZ', peak: rnd(-5, 5), startSec: t, durSec: 0.9 });
      this.nodBlinkWanted = true; // blink with the listening nod too
      this.nextListenNodSec = t + rnd(num(motionConfig.listeningNodMinMs, 2200), num(motionConfig.listeningNodMaxMs, 4800)) / 1000;
      this.log('listening nod');
    }

    if (!this.idsResolved) this.resolveIds();
    if (!this.idsResolved) return;
    if (!this._loggedActive) {
      this._loggedActive = true;
      console.info('[TalkingMotion] active — alive idle + speaking/thinking/listening motion');
    }

    // Blink together with a nod — but only if the eyes are currently FREE. If a
    // blink is already in progress, or a motion is driving the eyes (squint,
    // sleepy...), the current eye-open value is low, so we skip to avoid
    // doubling up ("if the eyes already have a motion, don't overlap"). Blinks
    // are only ever triggered BY a nod, never the reverse.
    if (this.nodBlinkWanted) {
      this.nodBlinkWanted = false;
      const eyeOpen = model.getParameterValueById(this.ids.eyeLOpen);
      if (eyeOpen > 0.7) {
        this.impulses.push({ param: 'eyeLOpen', peak: -1, startSec: t, durSec: 0.16 });
        this.impulses.push({ param: 'eyeROpen', peak: -1, startSec: t, durSec: 0.16 });
        this.log('nod blink');
      } else {
        this.log('nod blink skipped: eyes already busy');
      }
    }

    // Determine mode + level (idle baseline keeps her from being a statue).
    const idleBase = num(motionConfig.idleLifeStrength, 0.12);
    const stateLevel = Math.max(this.talkingLevel, this.listeningLevel, this.thinkingLevel);
    const level = Math.max(stateLevel, idleBase);
    let mode: Mode = 'idle';
    if (speaking) mode = 'speaking';
    else if (this.thinkingLevel >= this.listeningLevel && this.thinkingLevel > 0.01) mode = 'thinking';
    else if (this.listeningLevel > 0.01) mode = 'listening';

    const baseStrength = mode === 'speaking'
      ? strengthFor(this.emotion)
      : mode === 'thinking'
        ? num(motionConfig.thinkingStrength, 0.4)
        : mode === 'listening'
          ? num(motionConfig.listeningStrength, 0.4)
          : idleBase;
    const energyScale = mode === 'speaking' ? 0.7 + this.energy * 0.6 : 1;
    // Smooth the intensity toward its target so it never STEPS at a mode change
    // (idle->speaking etc.), which previously caused an abrupt jerk at the start
    // and end of speech.
    const targetStrength = Math.max(baseStrength, 0) * energyScale;
    this.swayStrength += (targetStrength - this.swayStrength) * Math.min(1, dt / 0.3);
    const sway = level * this.swayStrength;

    const prosody = this.sampleProsody(dt, speaking) * num(motionConfig.prosodyStrength, 0.6);
    // Prosody only gently scales the (smooth) sway amplitude + lifts brows — it
    // does NOT push head position directly (that caused audio-synced jitter).
    const energyGain = 1 + prosody * 0.4;

    this.updateGaze(dt, mode);
    this.updatePosture(dt);
    this.updateWet(dt);

    const sd = this.seed;
    const breath = num(motionConfig.breathStrength, 0.6) * Math.sin(t * 2 * Math.PI * num(motionConfig.breathRateHz, 0.25));

    // Body ORIENTATION (left/right turn + tilt + posture lean) fades out while
    // speaking, because the emotion motions already drive the body then —
    // running both at once made the body lurch at the start of a reply. Idle /
    // listening / thinking keep the lively body shift. Head, gaze and breath
    // are unaffected (head movement stays on during speech).
    const bodyFactor = 1 - this.talkingLevel;

    const out: Record<ParamKey, number> = {
      angleX: 7.0 * sway * energyGain * wobble(t, 1.3, 2.7, sd),
      angleY: 4.0 * sway * energyGain * wobble(t, 1.7, 3.1, sd + 1.1),
      angleZ: 5.0 * sway * wobble(t, 0.9, 1.9, sd + 2.0),
      bodyX: (4.0 * sway * wobble(t, 0.8, 1.6, sd + 0.7) + this.posture.x * level) * bodyFactor,
      bodyY: 0.6 * breath, // subtle breath-coupled body rise (orientation-neutral)
      bodyZ: 1.5 * sway * wobble(t, 0.6, 1.3, sd + 1.5) * bodyFactor,
      eyeX: clamp(this.gaze.x * level, -1, 1),
      eyeY: clamp(this.gaze.y * level, -1, 1),
      eyeLOpen: 0, // 0 = no change; nod-blink impulse dips this negative briefly
      eyeROpen: 0,
      browL: 0,
      browR: 0,
      shoulder: 1.2 * breath, // subtle breath-coupled shoulder rise
      cheek: this.wet.cheek,
      tear: this.wet.tear,
      eyeHi: this.wet.eyeHi,
    };

    // Prosody brow raise only (no direct head-position drive — see energyGain).
    const browRaise = 0.3 * prosody * num(motionConfig.browStrength, 0.6) * level;
    out.browL += browRaise;
    out.browR += browRaise;

    // Thinking pose: gentle tilt + slight furrow while generating a reply.
    if (mode === 'thinking') {
      out.angleZ += 5 * this.thinkingLevel;
      out.browL -= 0.1 * num(motionConfig.browStrength, 0.6) * this.thinkingLevel;
      out.browR -= 0.1 * num(motionConfig.browStrength, 0.6) * this.thinkingLevel;
    }

    // Smoothed emotion bias (no snapping between emotions).
    const target = this.emotionBiasTarget(level);
    const bk = Math.min(1, (dt * 1000) / num(motionConfig.emotionSmoothMs, 400));
    PARAM_KEYS.forEach((key) => {
      const cur = this.bias[key] ?? 0;
      this.bias[key] = cur + (target[key] - cur) * bk;
      out[key] += this.bias[key];
    });

    // Transient impulses (beats / fidgets / anticipation / settle / blink).
    if (this.impulses.length > 0) {
      const alive: Impulse[] = [];
      for (const imp of this.impulses) {
        const p = (t - imp.startSec) / imp.durSec;
        if (p >= 1) continue;
        if (p >= 0) {
          // Eye-open (blink) uses a smooth down-up; everything else uses the
          // eased gesture envelope (with a little recoil).
          const isBlink = imp.param === 'eyeLOpen' || imp.param === 'eyeROpen';
          out[imp.param] += imp.peak * (isBlink ? Math.sin(Math.PI * p) : beatShape(p));
        }
        alive.push(imp);
      }
      this.impulses = alive;
    }

    PARAM_KEYS.forEach((key) => {
      const v = out[key];
      if (v !== 0) model.addParameterValueById(this.ids[key], v);
    });
  }
}

export const talkingMotionController = new TalkingMotionController();

// Live debug/tuning in the browser console, e.g.:
//   window.__talkingMotion.state
//   window.__talkingMotion.onThinkingStart()
//   window.__talkingMotion.onSentenceStart("test。", 3, "Joy_big")
if (typeof window !== 'undefined') {
  (window as any).__talkingMotion = talkingMotionController;
}
