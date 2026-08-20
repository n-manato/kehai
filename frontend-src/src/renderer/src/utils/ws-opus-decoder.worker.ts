/**
 * Opus decoding for the WEBSOCKET audio path, OFF the main thread.
 *
 * Why this is a worker
 * --------------------
 * `AudioDecoder` delivers its `output` callbacks on the thread that owns it,
 * and Chrome dispatches roughly ONE of them per event-loop tick. On the main
 * thread that tick is the render loop, so the ceiling is `fps x packet_length`
 * of audio per second of wall clock. The server emits 20 ms packets, which
 * makes the ceiling 1.2x realtime at a perfect 60 fps and 0.6x at the ~32 fps
 * this app actually runs while it is drawing the avatar — i.e. the decoder
 * cannot keep up with the speech it is decoding.
 *
 * Measured on this box (400 packets = 8.0 s of speech, same page, same build):
 *
 *     main thread   13226 ms   30 packets/s   0.60x realtime
 *     this worker      37 ms   10959 packets/s  219x realtime
 *
 * The 0.6x was the whole fault behind "her voice stutters, or she just stops
 * talking": the player starved between every 20 ms buffer (audible as a rasp),
 * and once it starved with the stream marked complete, the drain check read the
 * empty queue as end-of-turn and threw away everything still in the decoder —
 * up to 80% of the reply. Nothing in the logs showed it, because every packet
 * did arrive and every packet did decode; they simply arrived too late to be
 * played.
 *
 * The worker has no render loop, so its tick is free and the decode runs at
 * memory speed. Output order is preserved: packets are posted back in the order
 * `AudioDecoder` emits them, which is the order they were fed.
 */

// WebCodecs is not in this project's TS lib set (see ws-opus-decoder.ts, which
// reaches for the same globals through casts). Keep the surface minimal and
// local rather than pulling in a dom-lib bump for four call sites.
type Decoder = {
  configure: (cfg: Record<string, unknown>) => void
  decode: (chunk: unknown) => void
  flush: () => Promise<void>
  close: () => void
  state: string
};

type InMsg =
  | { t: 'decode', ts: number, data: ArrayBuffer }
  | { t: 'close' };

const ctx = self as unknown as {
  AudioDecoder?: new (init: {
    output: (d: unknown) => void
    error: (e: unknown) => void
  }) => Decoder
  EncodedAudioChunk?: new (i: Record<string, unknown>) => unknown
  postMessage: (m: unknown, transfer?: Transferable[]) => void
  onmessage: ((e: MessageEvent) => void) | null
};

const OPUS_RATE = 48000;

let dec: Decoder | null = null;
let broken = false;

/** Hand one decoded packet back as mono f32, transferring the buffer. */
function emit(data: unknown): void {
  const d = data as {
    allocationSize: (o: Record<string, unknown>) => number
    copyTo: (dst: ArrayBufferView, o: Record<string, unknown>) => void
    close: () => void
  };
  try {
    // f32-planar explicitly: the decoder may hand back s16 otherwise, and the
    // player wants floats either way.
    const opts = { planeIndex: 0, format: 'f32-planar' };
    const out = new Float32Array(d.allocationSize(opts) / 4);
    d.copyTo(out, opts);
    ctx.postMessage({ t: 'samples', buf: out.buffer }, [out.buffer]);
  } catch (e) {
    broken = true;
    ctx.postMessage({ t: 'error', msg: `copyTo failed: ${String(e)}` });
  } finally {
    d.close();
  }
}

function ensure(): Decoder | null {
  if (dec || broken) return dec;
  const C = ctx.AudioDecoder;
  if (!C) { broken = true; return null; }
  try {
    dec = new C({
      output: emit,
      error: (e) => {
        broken = true;
        ctx.postMessage({ t: 'error', msg: String(e) });
      },
    });
    dec.configure({ codec: 'opus', sampleRate: OPUS_RATE, numberOfChannels: 1 });
  } catch (e) {
    dec = null;
    broken = true;
    ctx.postMessage({ t: 'error', msg: `decoder unavailable: ${String(e)}` });
  }
  return dec;
}

ctx.onmessage = (ev: MessageEvent) => {
  const msg = ev.data as InMsg;
  if (msg.t === 'decode') {
    const d = ensure();
    const Chunk = ctx.EncodedAudioChunk;
    if (!d || !Chunk) {
      // Tell the host the packet is not coming back, so its in-flight count
      // cannot stall the end of the turn.
      ctx.postMessage({ t: 'dropped' });
      return;
    }
    try {
      // Every Opus packet is independently decodable, so they are all 'key'.
      d.decode(new Chunk({ type: 'key', timestamp: msg.ts, data: msg.data }));
    } catch (e) {
      broken = true;
      ctx.postMessage({ t: 'error', msg: `decode failed: ${String(e)}` });
      ctx.postMessage({ t: 'dropped' });
    }
    return;
  }
  if (msg.t === 'close') {
    const d = dec;
    dec = null;
    if (!d) { ctx.postMessage({ t: 'closed' }); return; }
    const done = () => {
      try { if (d.state !== 'closed') d.close(); } catch { /* already gone */ }
      ctx.postMessage({ t: 'closed' });
    };
    try { d.flush().then(done, done); } catch { done(); }
  }
};
