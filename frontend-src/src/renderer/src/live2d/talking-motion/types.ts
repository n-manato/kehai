// Shared types for the Talking Motion System — a lightweight, rule-based layer
// that adds subtle continuous movement (head sway, body sway, sentence-end
// nods, question tilts, keyword reactions) on TOP of the existing emotion
// motions / idle loop. It only ADDS to head/body/gaze parameters and never
// touches mouth lip-sync, eye blink, or breath.

export type Emotion =
  | 'happy'
  | 'calm'
  | 'thinking'
  | 'surprised'
  | 'sad'
  | 'nervous'
  | 'neutral';

export type GestureType =
  | 'small_nod'
  | 'head_tilt'
  | 'emphasis'
  | 'micro_pause'
  | 'surprise_pop'
  | 'none';

/** Live state used to drive movement strength (see spec EmotionMotionState). */
export interface EmotionMotionState {
  speaking: boolean;
  emotion: Emotion;
  energy: number; // 0.0 .. 1.0
  gesture: GestureType;
}

export interface KeywordReactionDef {
  keywords: string[];
  reaction: string; // semantic label, used only for logging
  gesture?: GestureType; // subtle parameter impulse to play
  expression?: string; // optional EXISTING expression name to set
  motionGroup?: string; // optional EXISTING motion group to play
  priority?: number;
}

export interface MotionBehaviorConfig {
  enableTalkingMotion: boolean;
  talkingFadeInMs: number;
  talkingFadeOutMs: number;
  minReactionIntervalMs: number;
  nodCooldownMs: number;
  keywordReactionCooldownMs: number;
  defaultEnergy: number;
  /** Per-emotion movement-strength multiplier (0..1-ish). */
  emotionStrength: Record<string, number>;
  keywordReactions: KeywordReactionDef[];
  /** Single punctuation char -> gesture played around that point in speech. */
  punctuationGestures: Record<string, GestureType>;
  /** Emit verbose per-event console logs. */
  debugLog?: boolean;

  // --- "Human-ness" layers (all optional; sensible defaults applied) ---
  /** How strongly the live speech loudness (volumes) drives head/brow. 0..1. */
  prosodyStrength?: number;
  /** How strongly brows move on emphasis/question/emotion. 0..1. */
  browStrength?: number;
  /** Random gaze-shift (saccade) interval range, ms. */
  gazeMinIntervalMs?: number;
  gazeMaxIntervalMs?: number;
  /** Random posture (weight-shift) interval range, ms. */
  postureShiftMinMs?: number;
  postureShiftMaxMs?: number;
  /** Movement strength while listening to the user (attentive nods). 0..1. */
  listeningStrength?: number;
  /** Attentive-nod interval range while listening, ms. */
  listeningNodMinMs?: number;
  listeningNodMaxMs?: number;

  // --- Extra "alive" layers ---
  /** Always-on micro-motion strength when fully idle (never a statue). 0..1. */
  idleLifeStrength?: number;
  /** Subtle breath-coupled shoulder/body bob strength + rate. */
  breathStrength?: number;
  breathRateHz?: number;
  /** Strength of the "thinking" pose while generating a reply. 0..1. */
  thinkingStrength?: number;
  /** Time to ease emotion bias between emotions, ms (smooths snapping). */
  emotionSmoothMs?: number;
  /** Strength of blush / tears / eye-highlight responses. 0..1 each. */
  cheekStrength?: number;
  tearStrength?: number;
  eyeHiStrength?: number;
  /** Random micro-fidget interval range, ms. */
  fidgetMinMs?: number;
  fidgetMaxMs?: number;
  /** Scale movement by sentence length/pace (longer/faster = livelier). */
  energyFromLength?: boolean;
}
