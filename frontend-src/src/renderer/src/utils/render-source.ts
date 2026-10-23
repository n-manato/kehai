/**
 * Render-host mode (`?render=1`): this tab is the app, drawn on the server's
 * GPU, and someone else is watching it.
 *
 * The server streams this tab's pixels to the viewer itself (CDP screencast,
 * see src/kehai/render_stream.py) and injects the viewer's clicks and keys as
 * real input. What it cannot reach from outside the page is SOUND, in either
 * direction, so this module carries it over `/render-stream/source`:
 *
 *   out — everything the app plays (streamed TTS through an AudioContext, clip
 *         playback through <audio>) is tapped at the point it leaves for the
 *         speakers, mixed to 48 kHz mono, and sent as Int16 PCM.
 *   in  — the viewer's microphone arrives the same way and is handed to the app
 *         as the result of getUserMedia, so VAD and the WebSocket mic path run
 *         unchanged. The server box has no sound card; this IS the microphone.
 *
 * Imported first in main.tsx: the patches must be in place before any audio
 * node is wired or any mic is opened. Outside render mode it does nothing.
 */

const RATE = 48000;
const KIND_AUDIO = 1;
const KIND_MIC = 3;
const KIND_CAM = 5;

export const isRenderSource =
  typeof window !== "undefined" && new URLSearchParams(location.search).get("render") === "1";

// Mixes every tapped context into 20 ms Int16 chunks.
const TAP_WORKLET = `
class Tap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Int16Array(960); this.n = 0; }
  process(inputs) {
    const ch = inputs[0];
    if (!ch.length) return true;
    const len = ch[0].length;
    for (let i = 0; i < len; i++) {
      let s = 0;
      for (let c = 0; c < ch.length; c++) s += ch[c][i];
      s /= ch.length;
      this.buf[this.n++] = Math.max(-1, Math.min(1, s)) * 0x7fff;
      if (this.n === this.buf.length) { this.port.postMessage(this.buf.buffer, [this.buf.buffer]); this.buf = new Int16Array(960); this.n = 0; }
    }
    return true;
  }
}
registerProcessor("kehai-tap", Tap);
`;

// Plays the viewer's mic from a queue; silence on underrun.
//
// The queue must not be allowed to sit full. Chunks arrive in bursts over the
// TCP tunnel, and a queue that only drops past a hard cap stays at the cap once
// a burst has filled it: every word then reaches VAD that much later, and the
// end-of-turn with it. So past 80 ms it drops back to 40 ms — a few ms of
// skipped mic audio at a burst, instead of permanent lag.
const MIC_WORKLET = `
class Mic extends AudioWorkletProcessor {
  constructor() {
    super(); this.q = []; this.cur = null; this.pos = 0; this.queued = 0; this.n = 0;
    this.port.onmessage = (e) => {
      const f = new Float32Array(e.data.length);
      for (let i = 0; i < f.length; i++) f[i] = e.data[i] / 0x7fff;
      this.q.push(f); this.queued += f.length;
      if (this.queued > 48000 * 0.08) {
        while (this.q.length > 1 && this.queued > 48000 * 0.04) this.queued -= this.q.shift().length;
      }
      if (++this.n % 50 === 0) this.port.postMessage(this.queued / 48);
    };
  }
  process(_, outputs) {
    const out = outputs[0][0];
    for (let i = 0; i < out.length; i++) {
      if (!this.cur || this.pos >= this.cur.length) { this.cur = this.q.shift() || null; this.pos = 0; if (this.cur) this.queued -= this.cur.length; }
      out[i] = this.cur ? this.cur[this.pos++] : 0;
    }
    return true;
  }
}
registerProcessor("kehai-mic", Mic);
`;

function moduleUrl(src: string): string {
  return URL.createObjectURL(new Blob([src], { type: "application/javascript" }));
}

function start() {
  // The server-side tab has no route to WebRTC peers; force the WS audio path.
  try { localStorage.setItem("webrtc", "0"); } catch { /* storage blocked */ }

  const mix = new AudioContext({ sampleRate: RATE });
  let ws: WebSocket | null = null;
  let tapNode: AudioWorkletNode | null = null;
  let micNode: AudioWorkletNode | null = null;
  const micDest = mix.createMediaStreamDestination();
  const ready = Promise.all([
    mix.audioWorklet.addModule(moduleUrl(TAP_WORKLET)),
    mix.audioWorklet.addModule(moduleUrl(MIC_WORKLET)),
  ]).then(() => {
    tapNode = new AudioWorkletNode(mix, "kehai-tap", { numberOfOutputs: 0 });
    tapNode.port.onmessage = (e) => {
      if (ws?.readyState !== WebSocket.OPEN) return;
      const pcm = new Uint8Array(e.data as ArrayBuffer);
      const msg = new Uint8Array(pcm.length + 1);
      msg[0] = KIND_AUDIO;
      msg.set(pcm, 1);
      ws.send(msg);
    };
    micNode = new AudioWorkletNode(mix, "kehai-mic", { numberOfInputs: 0, outputChannelCount: [1] });
    micNode.connect(micDest);
    // Queue depth (ms), roughly once a second, for the server's latency line.
    micNode.port.onmessage = (e) => {
      if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ t: "micq", ms: Math.round(e.data as number) }));
    };
  });

  // ---- out: tap every connection to a destination ---------------------------
  const tapped = new WeakMap<BaseAudioContext, MediaStreamAudioDestinationNode>();
  const tapFor = (ctx: BaseAudioContext) => {
    let d = tapped.get(ctx);
    if (!d && ctx instanceof AudioContext && ctx !== mix) {
      d = ctx.createMediaStreamDestination();
      tapped.set(ctx, d);
      const src = mix.createMediaStreamSource(d.stream);
      void ready.then(() => tapNode && src.connect(tapNode));
    }
    return d;
  };
  const connect = AudioNode.prototype.connect as (...a: unknown[]) => unknown;
  AudioNode.prototype.connect = function patched(this: AudioNode, ...args: unknown[]) {
    const ret = connect.apply(this, args);
    const dest = args[0];
    if (dest instanceof AudioDestinationNode) {
      const d = tapFor(this.context);
      if (d) connect.call(this, d);
    }
    return ret;
  } as typeof AudioNode.prototype.connect;

  // <audio> clips: route through the mix context once, keep them audible.
  const routed = new WeakSet<HTMLMediaElement>();
  const play = HTMLMediaElement.prototype.play;
  HTMLMediaElement.prototype.play = function patchedPlay(this: HTMLMediaElement) {
    if (!routed.has(this)) {
      routed.add(this);
      try {
        const src = mix.createMediaElementSource(this);
        src.connect(mix.destination);
        void ready.then(() => tapNode && src.connect(tapNode));
      } catch { /* already owned by another context */ }
    }
    void mix.resume();
    return play.call(this);
  };

  // ---- in: the viewer's camera ---------------------------------------------
  // Normally the SERVER finds the faces in the viewer's camera frames and only
  // the result arrives here ({"t": "faces"}, see src/kehai/face_detect.py),
  // handed to the head tracker through __headIngest.
  //
  // Fallback, when the server cannot run its detector: the JPEG frames come
  // through instead and are painted onto a canvas whose captureStream() is
  // handed out as the webcam. The head tracker then runs here unchanged and is
  // started by the first frame that arrives.
  const camCanvas = document.createElement("canvas");
  camCanvas.width = 640;
  camCanvas.height = 360;
  const camCtx = camCanvas.getContext("2d");
  const camStream = camCanvas.captureStream(30);
  let camBusy = false;
  let camStarted: unknown = null; // the __headTracker we started (one per viewer mount)
  const onCamFrame = (jpeg: Uint8Array) => {
    if (camBusy || !camCtx) return; // drop, never queue: a stale pose is worse
    camBusy = true;
    createImageBitmap(new Blob([jpeg], { type: "image/jpeg" }))
      .then((img) => {
        if (camCanvas.width !== img.width || camCanvas.height !== img.height) {
          camCanvas.width = img.width;
          camCanvas.height = img.height;
        }
        camCtx.drawImage(img, 0, 0);
        img.close();
        // Start once per VIEWER, not once per page. Frames start arriving before the
        // viewer mounts and registers __headTracker (the stream is already running when
        // this tab reloads), and a viewer remount stops the tracker on unmount — a
        // plain once-flag caught neither, and head tracking stayed off for the session.
        const start = (window as unknown as { __headTracker?: (on?: boolean) => unknown }).__headTracker;
        if (start && start !== camStarted) {
          camStarted = start;
          start(true);
          console.info("[render-source] viewer camera arrived: head tracking started");
        }
      })
      .catch(() => { /* a bad frame: skip it */ })
      .finally(() => { camBusy = false; });
  };

  // ---- in: the viewer's mic is this page's mic ------------------------------
  const gum = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (c?: MediaStreamConstraints) => {
    if (c?.audio && !c.video) {
      await ready;
      void mix.resume();
      return new MediaStream(micDest.stream.getAudioTracks().map((t) => t.clone()));
    }
    if (c?.video && !c.audio) {
      return new MediaStream(camStream.getVideoTracks().map((t) => t.clone()));
    }
    return gum(c);
  };

  // Lost the server and got it back = the server restarted. Reload instead of
  // carrying on: while it was down the app's own socket retried and rebuilt the
  // scene over and over, and the tab came back with the 0/1 background gone
  // (WebGL objects from a dead context). A reload also picks up a new build.
  let lostServer = false;
  const open = () => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/render-stream/source`);
    ws.onopen = () => {
      if (lostServer) location.reload();
    };
    ws.binaryType = "arraybuffer";
    ws.onmessage = (e) => {
      if (typeof e.data === "string") {
        try {
          const m = JSON.parse(e.data);
          // Dropped until the viewer mounts and registers the hook; the next
          // camera frame is a fraction of a second behind.
          if (m.t === "faces") (window as unknown as { __headIngest?: (m: unknown) => void }).__headIngest?.(m);
        } catch { /* not ours */ }
        return;
      }
      if (!(e.data instanceof ArrayBuffer)) return;
      const b = new Uint8Array(e.data);
      if (b[0] === KIND_MIC && micNode) {
        micNode.port.postMessage(new Int16Array(b.buffer.slice(1)));
      } else if (b[0] === KIND_CAM) {
        onCamFrame(b.subarray(1));
      }
    };
    ws.onclose = () => { lostServer = true; setTimeout(open, 1000); };
  };
  open();
  void mix.resume();
  console.info("[render-source] render-host mode: audio tapped, mic from viewer");
}

if (isRenderSource) start();
