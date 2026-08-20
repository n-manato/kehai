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

/** One pending autoplay unlock at a time — see armAutoplayUnlock. A player is
 *  built per turn, so re-arming blindly would pile up listeners for the whole
 *  session on a page that never gets clicked. */
let unlockArmed = false;

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
  // True while audio for this turn is still upstream of `enqueue*` — packets
  // sitting in the Opus decoder. An empty playback queue means "the turn
  // finished" only when this is false; otherwise it means the decoder has not
  // caught up yet, and calling the turn there throws the rest of the reply
  // away. That was the whole of "she stops talking mid-sentence": see
  // ws-opus-decoder.worker.ts for the starvation that made it routine.
  morePending?: () => boolean;
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

  private morePending?: () => boolean;

  /** performance.now() past which a stalled decoder stops holding the turn
   *  open. Without a stop the turn could never end if the upstream wedged. */
  private drainDeadline = 0;

  /** Trailing byte of a chunk when its length is odd — an int16 sample split
   *  across the (TCP) chunk boundary. Carried into the next chunk so samples stay
   *  aligned; dropping it byte-shifts everything after = continuous noise. */
  private leftover: Uint8Array | null = null;

  constructor(opts: StreamingAudioPlayerOpts) {
    this.sampleRate = opts.sampleRate;
    this.onEnded = opts.onEnded;
    this.morePending = opts.morePending;
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
    if (this.ctx.state === 'suspended') this.armAutoplayUnlock();
    // Start the clock prerollSec in the future so a buffer lead builds before
    // playback — gives co-speech gestures time to generate ahead of their sentence.
    this.nextStartTime = this.ctx.currentTime + this.preroll;
    activePlayer = this;
  }

  /**
   * A suspended AudioContext is the one failure on this path that leaves
   * NOTHING to look at. Chunks decode, buffers schedule, gestures and subtitles
   * run off their own clocks — the only symptom is silence, and the bare
   * `resume().catch(() => {})` this replaced swallowed the reason.
   *
   * resume() alone is not enough: the browser wants a user GESTURE, and a
   * voice-only session never produces one — speaking into the microphone does
   * not count, and neither does a mic that was switched on in an earlier page
   * load. So try now, retry on the next click/keypress, and if the context is
   * still held after that, say so instead of playing to a stopped clock.
   */
  private armAutoplayUnlock(): void {
    this.ctx.resume().catch(() => {});
    if (unlockArmed) return;
    unlockArmed = true;
    const events = ['pointerdown', 'keydown', 'touchstart'];
    const unlock = () => {
      for (const e of events) window.removeEventListener(e, unlock);
      unlockArmed = false;
      this.ctx.resume().catch(() => {});
    };
    for (const e of events) window.addEventListener(e, unlock);
    window.setTimeout(() => {
      if (this.ctx.state !== 'suspended') return;
      console.warn(
        '[audio] the browser is holding the AI voice — this page has had no '
        + `user gesture, so its AudioContext is suspended (${location.origin}). `
        + 'Click anywhere on the page once; talking into the mic does not count.',
      );
      // ...and SAY it on screen. A console warning is invisible to someone who
      // is just talking to her, and this failure has no other symptom: she
      // gestures, the subtitles run, the server logs a healthy turn, and the
      // only thing missing is the sound. Autoplay permission is per ORIGIN, so
      // moving between 127.0.0.1 / localhost / a forwarded port lands on a
      // fresh, blocked one even though the previous URL worked for months.
      window.dispatchEvent(new CustomEvent('kehai-audio-blocked'));
    }, 500);
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
    this.schedule(float32);
  }

  /** Feed already-decoded mono float samples — the Opus path (WebCodecs hands
   *  back f32), which carries the same speech at ~1/8 the bytes of raw PCM. */
  enqueueFloat32(samples: Float32Array): void {
    if (this.stopped || samples.length === 0) return;
    this.schedule(samples);
  }

  private scheduledBuffers = 0;

  private scheduledSeconds = 0;

  /**
   * What this player actually DID with the turn, for the server log.
   *
   * "No sound" is the one complaint that cannot be answered from the server
   * side alone: it can prove it sent the audio, but not that any browser
   * decoded, scheduled or played it — and asking someone to open a console
   * mid-conversation is how a five-minute fault becomes an hour. `ctx` is the
   * whole answer in one field: 'running' means the samples reached the output
   * and the fault is past this app (device, tab mute, OS mixer), 'suspended'
   * means the browser is holding them for a user gesture.
   */
  health(): Record<string, unknown> {
    return {
      ctx: this.ctx.state,
      buffers: this.scheduledBuffers,
      seconds: +this.scheduledSeconds.toFixed(2),
      rate: this.ctx.sampleRate,
      gain: this.gain.gain.value,
      origin: location.origin,
    };
  }

  /** Append one mono buffer to the gapless schedule. */
  private schedule(float32: Float32Array): void {
    const buffer = this.ctx.createBuffer(1, float32.length, this.sampleRate);
    buffer.copyToChannel(float32, 0);
    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(this.analyser);
    // Never schedule in the past — if playback fell behind (underrun), restart
    // the clock from now so we don't drop audio into an already-elapsed slot.
    const startAt = Math.max(this.ctx.currentTime, this.nextStartTime);
    src.start(startAt);
    this.scheduledBuffers += 1;
    this.scheduledSeconds += buffer.duration;
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
    // Cap how long a wedged upstream may hold the turn open. Generous: the
    // stream is complete by now, so this only has to cover a decoder finishing
    // what it already holds, and ending the turn early is the failure being
    // fixed here — a late idle pose costs nothing next to a truncated reply.
    this.drainDeadline = performance.now() + 30000;
    this.scheduleDrainCheck();
  }

  private scheduleDrainCheck(delayMs?: number): void {
    if (!this.ended || this.stopped) return;
    if (this.drainTimer !== null) clearTimeout(this.drainTimer);
    const wait = delayMs
      ?? Math.max(0, (this.nextStartTime - this.ctx.currentTime) * 1000) + 30;
    this.drainTimer = setTimeout(() => {
      this.drainTimer = null;
      if (this.stopped) return;
      if (this.ctx.currentTime < this.nextStartTime - 0.02) {
        this.scheduleDrainCheck(); // more arrived after end() — re-arm
        return;
      }
      // The queue is empty — but empty because it was PLAYED, or because the
      // samples have not arrived yet? Only the first is the end of the turn.
      if (this.morePending?.() && performance.now() < this.drainDeadline) {
        this.scheduleDrainCheck(50);
        return;
      }
      if (activePlayer === this) activePlayer = null;
      this.onEnded?.();
    }, wait);
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
