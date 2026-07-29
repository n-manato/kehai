/**
 * StreamingAudioPlayer — gapless Web Audio playback of raw int16 PCM chunks that
 * arrive over the WebSocket (backend `audio-stream-start/chunk/end`), used for the
 * true-TTS-streaming path. The classic per-sentence HTMLAudioElement path
 * (use-audio-task.ts) is untouched and still handles non-streaming `audio`
 * messages; this only runs when the backend streams.
 *
 * Design:
 *  - One shared AudioContext (created lazily on the first user-gesture-driven
 *    playback; the mic button already unlocks audio).
 *  - Each PCM chunk becomes an AudioBuffer scheduled back-to-back on a running
 *    clock (`nextStartTime`) so there are no gaps between chunks OR sentences.
 *  - An AnalyserNode taps the graph so lip-sync can read a LIVE RMS every frame
 *    (no server-side `volumes[]` needed) — `mouthValue()` returns 0..1.
 *  - `currentTime()` exposes the AudioContext-relative playback clock for any
 *    consumer that was previously reading `HTMLAudioElement.currentTime`.
 */

// Shared flag so the classic playback-complete effect (use-audio-task) knows a
// streamed turn is in flight and must NOT report "playback complete" off its
// (empty) task queue — the streaming path reports it when the player drains, so
// the backend's per-turn cleanup (which cancels in-flight co-speech gestures)
// fires AFTER the audio actually finished, not ~2.5x too early.
export const streamingPlaybackState = { active: false };

// The most recent (single) live player, so UI layers without a ref to it (the
// VAD context) can duck the output while the USER is speaking. Cleared on
// stop/drain.
let activePlayer: StreamingAudioPlayer | null = null;

/** Duck/restore the streaming output (user started/stopped speaking). Safe
 *  no-op when nothing is playing. */
export function duckStreamingAudio(on: boolean): void {
  activePlayer?.duck(on);
}

export interface StreamingAudioPlayerOpts {
  sampleRate: number; // GSV raw PCM is 32000 Hz
  onEnded?: () => void; // fired once playback has fully drained after end()
  // Delay the START of playback by this many seconds (buffer up front). This buys
  // co-speech gestures LEAD: GestureLSM makes ~1 clip / 2.5s (serial) while audio
  // plays ~1 sentence / 3s, so playback out-paces gesture generation — but only if
  // playback starts a couple seconds behind, else the first sentences play before
  // their clip is ready. Trades a little first-audio latency for working motion.
  prerollSec?: number;
}

export class StreamingAudioPlayer {
  private ctx: AudioContext;

  private analyser: AnalyserNode;

  private gain: GainNode;

  private readonly sampleRate: number;

  /** AudioContext time at which the next enqueued buffer should start. */
  private nextStartTime = 0;

  /** true once the backend signalled `audio-stream-end` (no more chunks). */
  private ended = false;

  private stopped = false;

  private onEnded?: () => void;

  private drainTimer: ReturnType<typeof setTimeout> | null = null;

  private analyserBuf: Float32Array;

  private liveSources = new Set<AudioBufferSourceNode>();

  private preroll = 0;

  /** Trailing byte of a chunk when its length is odd — an int16 sample split
   *  across the (TCP) chunk boundary. Carried into the next chunk so samples stay
   *  aligned; dropping it byte-shifts everything after = continuous noise. */
  private leftover: Uint8Array | null = null;

  constructor(opts: StreamingAudioPlayerOpts) {
    this.sampleRate = opts.sampleRate;
    this.onEnded = opts.onEnded;
    this.preroll = Math.max(0, opts.prerollSec ?? 0);
    const Ctx =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext })
        .webkitAudioContext;
    // Run the context AT the PCM rate (32000) so our buffers play natively — a
    // context at 48000 resamples every small per-chunk buffer independently, which
    // adds edge artifacts at each boundary (audible as clipped sentence tails).
    try {
      this.ctx = new Ctx({ sampleRate: this.sampleRate });
    } catch {
      this.ctx = new Ctx();
    }
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;
    this.analyserBuf = new Float32Array(this.analyser.fftSize);
    this.gain = this.ctx.createGain();
    this.analyser.connect(this.gain);
    this.gain.connect(this.ctx.destination);
    // Resume if the context started suspended (autoplay policy). The mic gesture
    // usually unlocks it already; resume() is a no-op then.
    if (this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
    // Start the clock prerollSec in the future so a buffer lead builds before
    // playback — gives co-speech gestures time to generate ahead of their sentence.
    this.nextStartTime = this.ctx.currentTime + this.preroll;
    activePlayer = this;
  }

  /** Duck (attenuate) the output while the USER is speaking so speaker echo
   *  doesn't feed the VAD; restore on misfire/speech end. The lip-sync taps the
   *  analyser BEFORE this gain, so the mouth keeps tracking the ducked speech. */
  duck(on: boolean): void {
    const g = this.gain.gain;
    const t = this.ctx.currentTime;
    g.cancelScheduledValues(t);
    g.setTargetAtTime(on ? 0.18 : 1.0, t, 0.06);
  }

  /** Feed one raw little-endian int16 mono PCM chunk (arbitrary TCP boundary). */
  enqueuePCM(bytes: ArrayBuffer): void {
    if (this.stopped) return;
    let u8 = new Uint8Array(bytes);
    // Prepend any odd trailing byte carried from the previous chunk so an int16
    // sample straddling the boundary is reassembled instead of corrupting the
    // alignment of everything after it.
    if (this.leftover && this.leftover.length) {
      const merged = new Uint8Array(this.leftover.length + u8.length);
      merged.set(this.leftover, 0);
      merged.set(u8, this.leftover.length);
      u8 = merged;
      this.leftover = null;
    }
    const evenLen = u8.length - (u8.length % 2);
    if (u8.length % 2 === 1) this.leftover = u8.slice(evenLen); // carry the odd tail
    if (evenLen <= 0) return;

    // Decode via DataView (little-endian, alignment-safe — u8 may be byte-offset
    // into its buffer after slicing/merging, which Int16Array cannot view).
    const dv = new DataView(u8.buffer, u8.byteOffset, evenLen);
    const nSamples = evenLen / 2;
    const float32 = new Float32Array(nSamples);
    for (let i = 0; i < nSamples; i += 1) float32[i] = dv.getInt16(i * 2, true) / 32768;

    const buffer = this.ctx.createBuffer(1, float32.length, this.sampleRate);
    buffer.copyToChannel(float32, 0);
    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(this.analyser);
    // Never schedule in the past — if playback fell behind (underrun), restart
    // the clock from now so we don't drop audio into an already-elapsed slot.
    const startAt = Math.max(this.ctx.currentTime, this.nextStartTime);
    src.start(startAt);
    this.nextStartTime = startAt + buffer.duration;
    this.liveSources.add(src);
    src.onended = () => {
      this.liveSources.delete(src);
    };
  }

  /** Backend signalled end-of-stream: schedule the onEnded callback for when the
   *  already-queued audio finishes playing. */
  end(): void {
    this.ended = true;
    this.scheduleDrainCheck();
  }

  private scheduleDrainCheck(): void {
    if (!this.ended || this.stopped) return;
    if (this.drainTimer !== null) clearTimeout(this.drainTimer);
    const remainingMs = Math.max(0, (this.nextStartTime - this.ctx.currentTime) * 1000);
    this.drainTimer = setTimeout(() => {
      this.drainTimer = null;
      if (this.stopped) return;
      if (this.ctx.currentTime >= this.nextStartTime - 0.02) {
        if (activePlayer === this) activePlayer = null;
        this.onEnded?.();
      } else {
        this.scheduleDrainCheck(); // more arrived after end() — re-arm
      }
    }, remainingMs + 30);
  }

  /** 0..1 mouth openness from the live signal RMS (call each animation frame). */
  mouthValue(scale = 1.6): number {
    this.analyser.getFloatTimeDomainData(this.analyserBuf);
    let sum = 0;
    for (let i = 0; i < this.analyserBuf.length; i += 1) {
      const v = this.analyserBuf[i];
      sum += v * v;
    }
    const rms = Math.sqrt(sum / this.analyserBuf.length);
    return Math.min(1, rms * scale);
  }

  /** AudioContext playback clock (seconds). */
  currentTime(): number {
    return this.ctx.currentTime;
  }

  /** AudioContext time at which currently-queued audio ends = the scheduled start
   *  of the NEXT sentence. Captured at a sentence's audio-stream-start to give its
   *  co-speech gesture a per-sentence clock (see websocket-handler). */
  scheduledEnd(): number {
    return this.nextStartTime;
  }

  /** Seconds of audio still queued ahead of the playhead. */
  bufferedAhead(): number {
    return Math.max(0, this.nextStartTime - this.ctx.currentTime);
  }

  isPlaying(): boolean {
    return !this.stopped && this.bufferedAhead() > 0.01;
  }

  /** Hard stop (interrupt / new turn). */
  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    if (activePlayer === this) activePlayer = null;
    if (this.drainTimer !== null) {
      clearTimeout(this.drainTimer);
      this.drainTimer = null;
    }
    this.liveSources.forEach((s) => {
      try {
        s.stop();
      } catch {
        /* already stopped */
      }
    });
    this.liveSources.clear();
    this.ctx.close().catch(() => {});
  }
}
