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

// Server-driven mic restarts tolerated before WebRTC is declared unusable for
// this session. Each one costs a mic dropout, so this is deliberately small.
const MAX_FORCED_RESTARTS = 3;
// Two stalls further apart than this are separate incidents, not a loop. Only a
// RUN of them means the transport itself cannot carry audio.
const FORCED_WINDOW_MS = 60_000;

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

  // Forced (server-observed) restarts, counted separately from reconnectAttempts
  // because forceReconnect deliberately clears that one — see forceReconnect.
  private forcedRestarts = 0;

  private lastForcedAt = 0;

  /** Has ICE completed even once on this origin? A peer that has NEVER
   *  connected is not a flaky link — it is a transport that cannot carry media
   *  at all (a tunnelled or reverse-proxied origin reaches none of the server's
   *  host candidates), and every retry costs a microphone. See reconnect(). */
  private everConnected = false;

  /** True once the remote (AI voice) track is live — the signal for the
   *  websocket handler to use webrtc branches instead of the WS player. */
  active = false;

  /**
   * The remote track has arrived, but ICE may still be negotiating. Kept apart
   * from `active` because the server only sends over a peer whose
   * connectionState is 'connected'; treating "track received" as "audio is
   * coming over WebRTC" is what made a failed ICE swallow the WebSocket audio.
   */
  private trackReady = false;

  /** True when the browser refused to start playback and is waiting for a user
   *  gesture. Everything else keeps working in this state — subtitles, mouth,
   *  gestures — so without this flag the only symptom is silence. */
  autoplayBlocked = false;

  private gestureUnlock: (() => void) | null = null;

  /** This session gave up on WebRTC. Deliberately NOT persisted — see
   *  fallbackToWebSocket for why a remembered fallback was the wrong shape. */
  private sessionFallback = false;

  enabled(): boolean {
    // A stale '0' written by an earlier build still disables it, so it is
    // honoured here — but nothing writes that key any more.
    return !this.sessionFallback && localStorage.getItem('webrtc') !== '0';
  }

  /**
   * Start playback, and if the browser refuses, arm a one-shot retry on the
   * next user gesture anywhere in the page.
   *
   * A rejected play() is not recoverable by retrying on a timer — the browser
   * wants a gesture — so the only fix that works without the user knowing why
   * is to wait for the next click or keypress and try again then. The
   * AudioContext is resumed on the same gesture: it is created suspended on a
   * fresh origin, and while it only feeds the analyser tap, a suspended one
   * silently stops the live-RMS mouth as well.
   */
  private playWithGestureFallback(el: HTMLAudioElement): void {
    const tryPlay = () => el.play().then(
      () => {
        this.autoplayBlocked = false;
        if (this.ctx && this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
        return true;
      },
      (err: DOMException) => {
        // Report it. The whole point is that this failure used to be invisible.
        console.warn(
          `[webrtc] the browser refused to play the AI voice (${err?.name || err}). `
          + 'Audio will start on your next click. Autoplay permission is per ORIGIN, '
          + `so ${location.origin} needs its own.`,
        );
        return false;
      },
    );

    tryPlay().then((ok) => {
      if (ok || this.gestureUnlock) return;
      this.autoplayBlocked = true;
      const unlock = () => {
        tryPlay().then((played) => {
          if (played) {
            console.info('[webrtc] AI voice unlocked by user gesture');
            this.removeGestureUnlock();
          }
        });
      };
      this.gestureUnlock = unlock;
      for (const ev of ['pointerdown', 'keydown', 'touchstart']) {
        window.addEventListener(ev, unlock, { passive: true });
      }
    });
  }

  private removeGestureUnlock(): void {
    if (!this.gestureUnlock) return;
    for (const ev of ['pointerdown', 'keydown', 'touchstart']) {
      window.removeEventListener(ev, this.gestureUnlock);
    }
    this.gestureUnlock = null;
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
        this.audioEl = el;
        // Autoplay can be refused, and when it is, THIS IS THE ONLY SYMPTOM:
        // the reply still arrives, the subtitle still appears, the mouth still
        // moves — there is simply no sound. The old code was
        // `el.play().catch(() => {})` on the assumption that granting the mic
        // had already unlocked audio for the origin. That assumption is per
        // ORIGIN, so opening the app on a different address (a LAN IP instead
        // of 127.0.0.1) starts with no autoplay permission and no mic grant
        // yet — and the swallowed rejection left nothing to find.
        this.playWithGestureFallback(el);
        // Parallel analyser tap (NOT routed to destination — the element plays)
        // for the live-RMS lip-sync + procedural talking body.
        const Ctx = window.AudioContext
          || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        this.ctx = new Ctx();
        // Created SUSPENDED on an origin with no prior gesture. It only feeds
        // the analyser tap, so a suspended context does not mute the voice —
        // it stops the live-RMS mouth, which reads as "she talks without moving
        // her lips". Resumed here when allowed, and again on the unlock gesture.
        if (this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
        const src = this.ctx.createMediaStreamSource(stream);
        this.analyser = this.ctx.createAnalyser();
        this.analyser.fftSize = 1024;
        this.analyserBuf = new Float32Array(this.analyser.fftSize);
        src.connect(this.analyser);
        // NOT `active = true` here. ontrack fires at setRemoteDescription — the
        // answer has been parsed, but ICE is still `connecting` and may never
        // succeed (it never does over an SSH port-forward, where aiortc offers
        // only the server's own host candidates). Meanwhile the SERVER refuses
        // to hand PCM to the track until connectionState === 'connected'
        // (realtime/webrtc_audio.py: track_for_send) and correctly falls back to
        // audio-stream-chunk over the WebSocket. If `active` were true here, the
        // handler would take the WebRTC branch, never build streamPlayerRef, and
        // silently drop every one of those chunks — the whole reply lost. Agree
        // with the server instead: active only once the peer is really connected.
        this.trackReady = true;
        console.log('[webrtc] remote AI-voice track received (waiting for ICE)');
      };
      pc.onconnectionstatechange = () => {
        if (!this.pc) return;
        const st = this.pc.connectionState;
        if (st === 'connected') {
          // Now — and only now — is the server willing to send over this track.
          if (this.trackReady && !this.active) {
            this.active = true;
            console.log('[webrtc] remote AI-voice track live');
          }
          // recovered — cancel any pending teardown and reset the backoff.
          this.everConnected = true;
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
  /** Rebuild the peer NOW, ignoring connectionState. The state machine only reacts to
   *  'disconnected'/'failed'; a peer can also sit at 'connected' while the mic media
   *  quietly stops flowing — the server then hears nothing, the avatar stops answering,
   *  and nothing on this side ever notices because every local signal still looks fine.
   *  The server watches for that (no mic chunk for several seconds) and calls this. */
  forceReconnect(): void {
    // A stall that KEEPS repeating is not a stall — it is a transport that
    // cannot carry audio at all (an SSH-forwarded origin reaches none of the
    // server's host candidates, so ICE never completes). Clearing
    // reconnectAttempts unconditionally made reconnect()'s give-up check
    // unreachable: the server's mic-watchdog re-armed the budget every ~6s and
    // the peer rebuilt itself forever, dropping the mic on every cycle. Measured
    // on a tunnelled session: 990 rebuilds, one every ~7s, and 41 of 50 turns
    // cancelled because the fragmented mic stream kept superseding itself.
    // So: keep the fresh-budget behaviour for a ONE-OFF stall, but count the
    // forced restarts too and fall back to the WS audio path once they run on.
    const now = Date.now();
    if (now - this.lastForcedAt > FORCED_WINDOW_MS) this.forcedRestarts = 0;
    this.lastForcedAt = now;
    this.forcedRestarts += 1;
    if (this.forcedRestarts > MAX_FORCED_RESTARTS) {
      this.fallbackToWebSocket(
        `${MAX_FORCED_RESTARTS} forced mic restarts without recovery`,
      );
      return;
    }
    console.warn('[webrtc] forced mic restart — server saw no audio');
    this.reconnectAttempts = 0; // an operator-visible stall is not a flaky-network retry
    this.reconnect();
  }

  private reconnect(): void {
    const send = this.sendSignal;
    this.stop(); // tears down pc/mic/audio and clears timers; keeps sendSignal
    // Retrying is only meaningful for a link that HAS worked. When ICE has
    // never completed once, the five backoff attempts cost ~3.5 minutes during
    // which stop() drops the mic every ~23s: utterances are cut in half so the
    // server VAD never closes a turn, and the client-VAD's speech-end never
    // fires, wedging aiState at 'listening' — which makes the WS audio branch
    // discard the reply as well. Symptom: she neither hears nor speaks for the
    // first few minutes after every load, then silently fixes herself. Give up
    // on the first failure instead; the WS path carries both directions.
    if (!this.everConnected) {
      this.fallbackToWebSocket('ICE never completed on this origin');
      return;
    }
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
    this.trackReady = false;
    // The peer is rebuilt on every reconnect, so a listener left behind here
    // would accumulate one set per reconnect and keep poking a dead element.
    this.removeGestureUnlock();
    this.autoplayBlocked = false;
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

  /**
   * Is the AI voice actually ARRIVING, and is it actually PLAYING?
   *
   * Two separate questions with two separate answers. The analyser taps the
   * received MediaStream, so it reads > 0 whenever packets arrive — even if the
   * element never plays a sample. The element's clock is what says a speaker
   * heard it. Distinguishing them is the difference between "the network broke"
   * and "the browser refused".
   */
  playbackHealth(): { arriving: boolean; playing: boolean } {
    const el = this.audioEl as HTMLAudioElement | null;
    const t = el ? el.currentTime : 0;
    const advanced = el ? t > this.lastElapsed + 0.05 : false;
    if (el) this.lastElapsed = t;
    return {
      arriving: this.mouthValue(1) > 0.002,
      playing: !!el && !el.paused && !el.muted && el.volume > 0 && advanced,
    };
  }

  private lastElapsed = 0;

  /**
   * Give up on WebRTC for this browser and let the server go back to sending
   * audio over the WebSocket.
   *
   * The server picks the audio path from whether a peer EXISTS, not from
   * whether that peer can play — the send is an `if rtc_track ... elif
   * websocket_send(...)`, so a browser that cannot play WebRTC audio gets no
   * audio at all rather than the old path. Closing the peer is what tells the
   * server to switch. The microphone keeps working: with `active` false the VAD
   * sends `raw-audio-data` over the same WebSocket.
   */
  fallbackToWebSocket(reason: string): void {
    if (!this.enabled()) return;
    console.warn(
      `[webrtc] falling back to WebSocket audio — ${reason}. `
      + "The voice will now stream over the WebSocket instead. "
      + "Undo with: localStorage.removeItem('webrtc'); location.reload()",
    );
    // NOT persisted. Writing localStorage here disabled WebRTC for that browser
    // FOREVER — one misfire and every later session silently ran on the
    // WebSocket path, with a different mouth source and no peer at all. A
    // fallback is a decision about THIS session; the next load should get to
    // try the better path again.
    this.sessionFallback = true;
    this.stop();
  }

  /**
   * Everything needed to place "no sound" on one side of the wire or the other.
   *
   * The server can prove it SENT audio; nothing until now could show whether a
   * given browser RECEIVED or PLAYED it, which left the question unanswerable
   * whenever the page was open on another machine. `inboundPackets` settles the
   * network half and the element fields settle the playback half.
   */
  async diagnose(): Promise<Record<string, unknown>> {
    const el = this.audioEl as HTMLAudioElement | null;
    const out: Record<string, unknown> = {
      origin: location.origin,
      webrtcEnabled: this.enabled(),
      trackLive: this.active,
      peerState: this.pc?.connectionState ?? 'no peer',
      iceState: this.pc?.iceConnectionState ?? 'no peer',
      autoplayBlocked: this.autoplayBlocked,
      audioContext: this.ctx?.state ?? 'none',
      element: el
        ? {
          paused: el.paused,
          muted: el.muted,
          volume: el.volume,
          readyState: el.readyState, // 4 = HAVE_ENOUGH_DATA
          sinkId: (el as HTMLAudioElement & { sinkId?: string }).sinkId || 'default',
        }
        : 'no audio element',
      liveRms: Number(this.mouthValue(1).toFixed(4)), // >0 => audio IS arriving
    };
    try {
      const stats = await this.pc?.getStats();
      stats?.forEach((r: Record<string, unknown>) => {
        if (r.type === 'inbound-rtp' && r.kind === 'audio') {
          out.inboundPackets = r.packetsReceived;
          out.inboundBytes = r.bytesReceived;
          out.concealedSamples = r.concealedSamples;
        }
        if (r.type === 'outbound-rtp' && r.kind === 'audio') {
          out.micPacketsSent = r.packetsSent;
        }
      });
    } catch {
      out.inboundPackets = 'getStats failed';
    }
    return out;
  }
}

export const webrtcAudio = new WebRTCAudio();

// Console handle: `await __audioDiag()`. Deliberately global — the machine with
// the problem is usually not the machine with the dev tools open.
(window as unknown as { __audioDiag: () => Promise<unknown> }).__audioDiag = () => webrtcAudio.diagnose();
