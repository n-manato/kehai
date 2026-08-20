/**
 * Opus decoding for the WEBSOCKET audio path.
 *
 * The WebRTC path gets Opus for free — the browser decodes the track. The
 * WebSocket fallback did not, and it is the path that actually runs whenever
 * ICE cannot complete (always, through an SSH port forward). It was shipping
 * raw 32 kHz s16 PCM in base64: 683 kbps for speech that costs ~86 kbps
 * encoded, on the one link that is genuinely constrained.
 *
 * WebCodecs does the decode, IN A WORKER — see ws-opus-decoder.worker.ts for
 * why that is not a detail. On the main thread `AudioDecoder` hands back about
 * one packet per render frame, which at the server's 20 ms packet size is 0.6x
 * realtime while the avatar is drawing: the decoder falls permanently behind
 * the speech, the player starves between buffers, and the turn is torn down
 * mid-reply with most of the audio still undelivered. The worker decodes the
 * same stream at ~219x realtime.
 *
 * WebCodecs is not universal, so the client tells the server what it can take
 * (`audio-caps`) and a browser without `AudioDecoder` simply keeps receiving
 * PCM. A browser that has `AudioDecoder` but cannot start the worker falls back
 * to decoding on the main thread — degraded, but the previous behaviour.
 */

const OPUS_RATE = 48000;
// 20 ms per packet, in microseconds. WebCodecs wants strictly increasing
// timestamps; the server emits fixed 20 ms frames, so counting them is exact
// and needs nothing from the wire.
const FRAME_US = 20000;

type AudioDecoderCtor = new (init: {
  output: (data: unknown) => void
  error: (e: unknown) => void
}) => {
  configure: (cfg: Record<string, unknown>) => void
  decode: (chunk: unknown) => void
  flush: () => Promise<void>
  close: () => void
  state: string
};

function ctor(): AudioDecoderCtor | null {
  const C = (window as unknown as { AudioDecoder?: AudioDecoderCtor }).AudioDecoder;
  return typeof C === 'function' ? C : null;
}

/** Cheap synchronous probe: is the API present at all? */
export function opusDecodeAvailable(): boolean {
  return ctor() !== null;
}

/** Authoritative check — the API can exist without libopus behind it. */
export async function opusDecodeSupported(): Promise<boolean> {
  const C = ctor() as unknown as {
    isConfigSupported?: (c: Record<string, unknown>) => Promise<{ supported: boolean }>
  } | null;
  if (!C || typeof C.isConfigSupported !== 'function') return false;
  try {
    const r = await C.isConfigSupported({
      codec: 'opus',
      sampleRate: OPUS_RATE,
      numberOfChannels: 1,
    });
    return !!r?.supported;
  } catch {
    return false;
  }
}

export class WsOpusDecoder {
  /** Main-thread decoder — only built if the worker could not be. */
  private dec: InstanceType<AudioDecoderCtor> | null = null;

  private worker: Worker | null = null;

  private ts = 0;

  private failed = false;

  /** Packets handed to the decoder that have not come back yet. The player
   *  asks for this: an empty playback queue means "the turn is over" ONLY when
   *  nothing is still in flight, otherwise it just means the decoder has not
   *  caught up and concluding there discards the rest of the reply. */
  private pending = 0;

  /**
   * @param onSamples receives decoded mono f32 in arrival order. Decoding is
   *   async, so ONE decoder must serve the whole turn: a second one running
   *   alongside would interleave its output and scramble the speech.
   */
  constructor(private onSamples: (samples: Float32Array) => void) {
    if (!ctor()) { this.failed = true; return; }
    if (this.startWorker()) return;
    this.startMainThread();
  }

  /** Preferred path. Returns false if a worker cannot be created at all. */
  private startWorker(): boolean {
    try {
      const w = new Worker(new URL('./ws-opus-decoder.worker.ts', import.meta.url), {
        type: 'module',
      });
      w.onmessage = (ev: MessageEvent) => {
        const m = ev.data as { t: string, buf?: ArrayBuffer, msg?: string };
        if (m.t === 'samples' && m.buf) {
          this.pending = Math.max(0, this.pending - 1);
          this.onSamples(new Float32Array(m.buf));
        } else if (m.t === 'dropped') {
          this.pending = Math.max(0, this.pending - 1);
        } else if (m.t === 'error') {
          console.warn('[ws-opus] decoder error', m.msg);
          this.failed = true;
          this.pending = 0;
        } else if (m.t === 'closed') {
          w.terminate();
        }
      };
      // A worker that fails to LOAD reports here rather than throwing above.
      // Everything already posted to it is gone, so drop the in-flight count
      // (those packets will never come back and must not hold the turn open)
      // and finish the turn on the main thread.
      w.onerror = (e) => {
        console.warn('[ws-opus] decode worker failed, falling back to the main '
          + 'thread — the voice may stutter on this turn', e.message);
        this.worker = null;
        this.pending = 0;
        try { w.terminate(); } catch { /* already gone */ }
        this.startMainThread();
      };
      this.worker = w;
      return true;
    } catch (e) {
      console.warn('[ws-opus] decode worker unavailable', e);
      return false;
    }
  }

  private startMainThread(): void {
    const C = ctor();
    if (!C) { this.failed = true; return; }
    try {
      this.dec = new C({
        output: (data) => this.emit(data),
        error: (e) => {
          console.warn('[ws-opus] decoder error', e);
          this.failed = true;
          this.pending = 0;
        },
      });
      this.dec.configure({
        codec: 'opus',
        sampleRate: OPUS_RATE,
        numberOfChannels: 1,
      });
    } catch (e) {
      console.warn('[ws-opus] decoder unavailable', e);
      this.dec = null;
      this.failed = true;
    }
  }

  get ok(): boolean {
    return (this.worker !== null || this.dec !== null) && !this.failed;
  }

  /** Packets handed over that have not been decoded back yet. */
  get inFlight(): number {
    return this.pending;
  }

  decode(packet: ArrayBuffer): void {
    if (this.failed) return;
    const ts = this.ts;
    this.ts += FRAME_US;
    if (this.worker) {
      this.pending += 1;
      // Transfer, don't copy: the buffer is freshly decoded from base64 for
      // this packet alone and nothing on this side reads it again.
      this.worker.postMessage({ t: 'decode', ts, data: packet }, [packet]);
      return;
    }
    if (!this.dec) return;
    const Chunk = (window as unknown as {
      EncodedAudioChunk?: new (i: Record<string, unknown>) => unknown
    }).EncodedAudioChunk;
    if (!Chunk) { this.failed = true; return; }
    try {
      // Every Opus packet is independently decodable, so they are all 'key'.
      this.pending += 1;
      this.dec.decode(new Chunk({ type: 'key', timestamp: ts, data: packet }));
    } catch (e) {
      console.warn('[ws-opus] decode failed', e);
      this.pending = Math.max(0, this.pending - 1);
      this.failed = true;
    }
  }

  private emit(data: unknown): void {
    const d = data as {
      numberOfFrames: number
      allocationSize: (o: Record<string, unknown>) => number
      copyTo: (dst: ArrayBufferView, o: Record<string, unknown>) => void
      close: () => void
    };
    this.pending = Math.max(0, this.pending - 1);
    try {
      // Ask for f32-planar explicitly: the decoder may hand back s16 otherwise,
      // and the player wants floats either way.
      const opts = { planeIndex: 0, format: 'f32-planar' };
      const out = new Float32Array(d.allocationSize(opts) / 4);
      d.copyTo(out, opts);
      this.onSamples(out);
    } catch (e) {
      console.warn('[ws-opus] copyTo failed', e);
      this.failed = true;
    } finally {
      d.close();
    }
  }

  /** Drain in-flight packets, then release. Safe to call more than once. */
  close(): void {
    this.pending = 0;
    const w = this.worker;
    if (w) {
      this.worker = null;
      // The worker flushes its decoder and answers 'closed'; terminating on the
      // spot would cut the last packet of the turn.
      try { w.postMessage({ t: 'close' }); } catch { w.terminate(); }
    }
    const d = this.dec;
    if (!d) return;
    this.dec = null;
    const done = () => { try { if (d.state !== 'closed') d.close(); } catch { /* already gone */ } };
    try {
      d.flush().then(done, done);
    } catch {
      done();
    }
  }
}
