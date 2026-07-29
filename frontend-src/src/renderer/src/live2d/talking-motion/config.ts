import type { MotionBehaviorConfig } from './types';
import rawConfig from './motion-behavior-config.json';

// Safe defaults so a partial/edited JSON never crashes the system.
const DEFAULT_CONFIG: MotionBehaviorConfig = {
  enableTalkingMotion: true,
  debugLog: false,
  talkingFadeInMs: 300,
  talkingFadeOutMs: 800,
  minReactionIntervalMs: 1200,
  nodCooldownMs: 1500,
  keywordReactionCooldownMs: 2500,
  defaultEnergy: 0.4,
  emotionStrength: {
    neutral: 0.35,
    calm: 0.3,
    happy: 0.6,
    thinking: 0.45,
    surprised: 0.85,
    sad: 0.25,
    nervous: 0.5,
  },
  keywordReactions: [],
  punctuationGestures: {},
  prosodyStrength: 0.6,
  browStrength: 0.6,
  gazeMinIntervalMs: 700,
  gazeMaxIntervalMs: 2600,
  postureShiftMinMs: 5000,
  postureShiftMaxMs: 11000,
  listeningStrength: 0.4,
  listeningNodMinMs: 2200,
  listeningNodMaxMs: 4800,
  idleLifeStrength: 0.12,
  breathStrength: 0.6,
  breathRateHz: 0.25,
  thinkingStrength: 0.4,
  emotionSmoothMs: 400,
  cheekStrength: 0.6,
  tearStrength: 0.5,
  eyeHiStrength: 0.4,
  fidgetMinMs: 4000,
  fidgetMaxMs: 12000,
  energyFromLength: true,
};

// Merge once at module load. Editing motion-behavior-config.json requires a
// frontend rebuild (pwsh scripts/build-frontend.ps1) to take effect.
export const motionConfig: MotionBehaviorConfig = {
  ...DEFAULT_CONFIG,
  ...(rawConfig as Partial<MotionBehaviorConfig>),
  emotionStrength: {
    ...DEFAULT_CONFIG.emotionStrength,
    ...((rawConfig as Partial<MotionBehaviorConfig>).emotionStrength ?? {}),
  },
};

export function strengthFor(emotion: string): number {
  return (
    motionConfig.emotionStrength[emotion] ??
    motionConfig.emotionStrength.neutral ??
    0.35
  );
}
