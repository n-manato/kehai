/* Phase G: WebRTC audio transport. Control stays on the WebSocket; only the
 * two AUDIO paths move onto an RTCPeerConnection:
 *
 *  - Mic goes UP as a real Opus track. Because the AI's voice now PLAYS from a
 *    remote WebRTC track, the browser's AEC finally has a render reference and
 *    cancels speaker echo properly (ducking becomes a belt-and-braces extra).
 *  - AI speech comes DOWN as a track: the browser's jitter buffer replaces the
 *    hand-rolled preroll and the base64 PCM frames disappear.
 *
 * Enabled by default; opt out with localStorage.webrtc = '0' (falls back to
 * the WS streaming player path). Signaling: webrtc-offer / webrtc-answer over
 * the existing WS. LAN host candidates only (client + server share a network).
 */

class WebRTCAudio {
  private pc: RTCPeerConnection | null = null;

  private micStream: MediaStream | null = null;

  private audioEl: HTMLAudioElement | null = null;

  private ctx: AudioContext | null = null;

  private analyser: AnalyserNode | null = null;

  private analyserBuf: Float32Array | null = null;

  private starting = false;

  // Reconnect state. WebRTC 'disconnected' is usually a TRANSIENT ICE blip that
  // self-heals in a second or two; the old code tore the peer down on the FIRST
  // 'disconnected' and never rebuilt it, so a momentary network/CPU hiccup killed
  // the mic and the user's speech stopped reaching the server ("しゃべっても反応
  // しない"). We now wait out 'disconnected' and auto-reconnect on 'failed'.
  private sendSignal: ((msg: object) => void) | null = null;

  private disconnectTimer: number | null = null;

  private reconnectTimer: number | null = null;

  private reconnectAttempts = 0;

  /** True once the remote (AI voice) track is live — the signal for the
   *  websocket handler to use webrtc branches instead of the WS player. */
  active = false;

  enabled(): boolean {
    return localStorage.getItem('webrtc') !== '0';
  }

  /** Create the peer + mic capture and send the offer over the WS. Idempotent. */
  async start(sendSignal: (msg: object) => void): Promise<void> {
    this.sendSignal = sendSignal; // saved so we can rebuild the peer on drop
    if (this.pc || this.starting || !this.enabled()) return;
    this.starting = true;
    try {
      const pc = new RTCPeerConnection();
      this.pc = pc;
      try {
        this.micStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
      } catch (e) {
        console.error('[webrtc] mic capture failed', e);
        this.stop();
        return;
      }
      this.micStream.getAudioTracks().forEach((t) => pc.addTrack(t, this.micStream!));

      pc.ontrack = (ev) => {
        const stream = ev.streams[0] ?? new MediaStream([ev.track]);
        // Playback element — REQUIRED for the AEC render reference.
        const el = document.createElement('audio');
        el.autoplay = true;
        (el as HTMLAudioElement).srcObject = stream;
        el.style.display = 'none';
        document.body.appendChild(el);
        el.play().catch(() => {}); // mic gesture already unlocked autoplay
        this.audioEl = el;
        // Parallel analyser tap (NOT routed to destination — the element plays)
        // for the live-RMS lip-sync + procedural talking body.
        const Ctx = window.AudioContext
          || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        this.ctx = new Ctx();
        const src = this.ctx.createMediaStreamSource(stream);
        this.analyser = this.ctx.createAnalyser();
        this.analyser.fftSize = 1024;
        this.analyserBuf = new Float32Array(this.analyser.fftSize);
        src.connect(this.analyser);
        this.active = true;
        console.log('[webrtc] remote AI-voice track live');
      };
      pc.onconnectionstatechange = () => {
        if (!this.pc) return;
        const st = this.pc.connectionState;
        if (st === 'connected') {
          // recovered — cancel any pending teardown and reset the backoff.
          this.reconnectAttempts = 0;
          if (this.disconnectTimer !== null) {
            clearTimeout(this.disconnectTimer);
            this.disconnectTimer = null;
          }
          return;
        }
        if (st === 'disconnected') {
          // TRANSIENT: ICE usually restores itself in a couple of seconds. Wait
          // instead of killing the mic; only reconnect if it's STILL down after.
          console.warn('[webrtc] disconnected — waiting for ICE recovery');
          if (this.disconnectTimer === null) {
            this.disconnectTimer = window.setTimeout(() => {
              this.disconnectTimer = null;
              if (this.pc && this.pc.connectionState === 'disconnected') {
                this.reconnect();
              }
            }, 4000);
          }
          return;
        }
        if (st === 'failed') {
          console.warn('[webrtc] failed — reconnecting');
          this.reconnect();
        }
        // 'closed' = intentional teardown; do nothing.
      };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await this.waitIceComplete(pc);
      sendSignal({ type: 'webrtc-offer', sdp: pc.localDescription!.sdp });
    } finally {
      this.starting = false;
    }
  }

  private waitIceComplete(pc: RTCPeerConnection): Promise<void> {
    if (pc.iceGatheringState === 'complete') return Promise.resolve();
    return new Promise((resolve) => {
      const done = () => {
        pc.removeEventListener('icegatheringstatechange', check);
        resolve();
      };
      const check = () => {
        if (pc.iceGatheringState === 'complete') done();
      };
      pc.addEventListener('icegatheringstatechange', check);
      setTimeout(done, 1500); // LAN host candidates arrive fast; never hang
    });
  }

  async onAnswer(sdp: string): Promise<void> {
    if (!this.pc) return;
    try {
      await this.pc.setRemoteDescription({ type: 'answer', sdp });
    } catch (e) {
      console.error('[webrtc] setRemoteDescription failed', e);
      this.stop();
    }
  }

  /** 0..1 live RMS of the AI voice (lip-sync / talking body), like the WS
   *  player's mouthValue. */
  mouthValue(scale = 1.6): number {
    if (!this.analyser || !this.analyserBuf) return 0;
    this.analyser.getFloatTimeDomainData(this.analyserBuf);
    let sum = 0;
    for (let i = 0; i < this.analyserBuf.length; i += 1) {
      const v = this.analyserBuf[i];
      sum += v * v;
    }
    return Math.min(1, Math.sqrt(sum / this.analyserBuf.length) * scale);
  }

  /** Attenuate while the user speaks (AEC already suppresses echo; this keeps
   *  the confirmation window extra clean). */
  duck(on: boolean): void {
    if (this.audioEl) this.audioEl.volume = on ? 0.18 : 1.0;
  }

  /** Enable/disable the outgoing mic track (mic toggle) WITHOUT tearing the
   *  peer down — the AI's voice keeps playing. */
  setMicEnabled(on: boolean): void {
    this.micStream?.getAudioTracks().forEach((t) => {
      t.enabled = on;
    });
  }

  /** Rebuild mic + peer + offer after a dropped connection, with exponential
   *  backoff. Gives up after a few tries (the WS audio path stays as fallback).
   *  Reuses the saved sendSignal so the new offer goes over the live WebSocket. */
  private reconnect(): void {
    const send = this.sendSignal;
    this.stop(); // tears down pc/mic/audio and clears timers; keeps sendSignal
    if (!send || this.reconnectAttempts >= 5) {
      if (this.reconnectAttempts >= 5) {
        console.warn('[webrtc] gave up reconnecting — staying on WS audio');
      }
      return;
    }
    const delay = Math.min(8000, 500 * 2 ** this.reconnectAttempts);
    this.reconnectAttempts += 1;
    console.warn(`[webrtc] reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      this.start(send);
    }, delay);
  }

  stop(): void {
    if (this.disconnectTimer !== null) {
      clearTimeout(this.disconnectTimer);
      this.disconnectTimer = null;
    }
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.active = false;
    this.micStream?.getTracks().forEach((t) => t.stop());
    this.micStream = null;
    if (this.audioEl) {
      (this.audioEl as HTMLAudioElement).srcObject = null;
      this.audioEl.remove();
      this.audioEl = null;
    }
    this.ctx?.close().catch(() => {});
    this.ctx = null;
    this.analyser = null;
    this.analyserBuf = null;
    try {
      this.pc?.close();
    } catch {
      /* already closed */
    }
    this.pc = null;
  }
}

export const webrtcAudio = new WebRTCAudio();
