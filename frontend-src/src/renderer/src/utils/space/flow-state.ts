/**
 * Fine-grained AI state for the data-flow scene, eased.
 *
 * The space's own mood (types.ts SpaceMood) cannot tell THINKING from SPEAKING —
 * the app reports both as "thinking-speaking". The flow needs the difference: it
 * pulls in while she thinks and pushes out while she talks. Speech is detected
 * from the voice level, the same way ai-fx does it, and a pause shorter than
 * SENTENCE_GAP_S inside a reply counts as speaking so the flow does not reverse
 * at every sentence boundary.
 */

export interface FlowWeights {
  idle: number;
  listen: number;
  think: number;
  speak: number;
  /** Smoothed voice level 0..1 (for pulsing with speech). */
  voice: number;
}

const SPEAKING_LEVEL = 0.03;
const SENTENCE_GAP_S = 1.5;

export class FlowState {
  readonly w: FlowWeights = { idle: 1, listen: 0, think: 0, speak: 0, voice: 0 };

  private t = 0;

  private lastVoiceAt = -1;

  constructor(
    private readonly getState: () => string,
    private readonly getVoice: () => number,
  ) {}

  update(dt: number) {
    this.t += dt;
    const voice = this.getVoice();
    if (voice > SPEAKING_LEVEL) this.lastVoiceAt = this.t;
    const app = this.getState();
    const recent = this.lastVoiceAt >= 0 && this.t - this.lastVoiceAt < SENTENCE_GAP_S;
    let target: keyof Omit<FlowWeights, "voice"> = "idle";
    if (app === "listening") target = "listen";
    else if (app === "thinking-speaking") target = recent ? "speak" : "think";
    else if (recent) target = "speak";

    // ~0.8s to settle: a state change reads as the flow turning, not switching.
    const k = 1 - Math.pow(0.05, dt / 0.8);
    for (const key of ["idle", "listen", "think", "speak"] as const) {
      this.w[key] += ((key === target ? 1 : 0) - this.w[key]) * k;
    }
    this.w.voice += (voice - this.w.voice) * Math.min(1, dt * 12);
  }
}
