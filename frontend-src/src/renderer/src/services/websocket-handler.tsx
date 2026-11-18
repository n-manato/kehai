/* eslint-disable no-sparse-arrays */
/* eslint-disable react-hooks/exhaustive-deps */
// eslint-disable-next-line object-curly-newline
import { useEffect, useState, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { wsService, MessageEvent } from '@/services/websocket-service';
import {
  WebSocketContext, HistoryInfo, defaultWsUrl, defaultBaseUrl,
} from '@/context/websocket-context';
import { ModelInfo, useLive2DConfig } from '@/context/live2d-config-context';
import { useSubtitle } from '@/context/subtitle-context';
import { audioTaskQueue } from '@/utils/task-queue';
import { useAudioTask } from '@/components/canvas/live2d';
import { useBgUrl } from '@/context/bgurl-context';
import { useConfig } from '@/context/character-config-context';
import { useChatHistory } from '@/context/chat-history-context';
import { toaster } from '@/components/ui/toaster';
import { useVAD } from '@/context/vad-context';
import { AiState, useAiState } from "@/context/ai-state-context";
import { useLocalStorage } from '@/hooks/utils/use-local-storage';
import { useGroup } from '@/context/group-context';
import { useInterrupt } from '@/hooks/utils/use-interrupt';
import { talkingMotionController } from '@/live2d/talking-motion';
import { mmdMotionPlayer } from '@/utils/mmd-motion-player';
import { vrmMotionPlayer } from '@/utils/vrm-motion-player';
import { webrtcAudio } from '@/utils/webrtc-audio';
import { mmdExpression } from '@/utils/mmd-expression';
import { vrmExpression } from '@/utils/vrm-expression';
import { vrmLipSync } from '@/utils/vrm-lipsync';
import { StreamingAudioPlayer, streamingPlaybackState, streamingControl } from '@/utils/streaming-audio-player';
import { WsOpusDecoder } from '@/utils/ws-opus-decoder';
import { playBackchannel } from '@/utils/backchannel-player';
import { isRenderSource } from '@/utils/render-source';
import { useBrowser } from '@/context/browser-context';

// Watches the first seconds of a reply and, if no sound actually happens, moves
// the audio off WebRTC. Armed once per turn; the check is cheap and stops as
// soon as it sees audio.
//
// It exists because "she speaks but nothing comes out" is invisible to the app:
// every other part of a turn (subtitle, mouth, gesture) is driven by control
// messages that arrive perfectly well while the audio path is dead.
let silenceWatchdog: number | null = null;

function startSilenceWatchdog(): void {
  if (silenceWatchdog !== null || !webrtcAudio.active) return;
  const t0 = performance.now();
  let sawArriving = false;
  const tick = () => {
    if (!webrtcAudio.active) { silenceWatchdog = null; return; }
    sawArriving = sawArriving || webrtcAudio.playbackHealth().arriving;
    // ARRIVAL is the only trigger, deliberately.
    //
    // `playing` cannot be trusted either way: an element fed a stream carrying
    // nothing still reports itself as playing, since its clock advances through
    // the silence — so it does not prove sound. And its absence does not prove
    // failure either; when audio arrives but the browser will not start it, the
    // cause is a missing user gesture, which the WebSocket player needs just as
    // much (its AudioContext starts suspended). Falling back then would swap one
    // blocked path for another while discarding a working one.
    //
    // So this fires only for the unambiguous case: nothing arrived at all. A
    // session that can hear is never disturbed by it.
    if (sawArriving) { silenceWatchdog = null; return; }
    if (performance.now() - t0 < 6000) {
      silenceWatchdog = window.setTimeout(tick, 300);
      return;
    }
    silenceWatchdog = null;
    webrtcAudio.fallbackToWebSocket('no audio arrived over WebRTC for 6s of speech');
  };
  silenceWatchdog = window.setTimeout(tick, 250);
}

// Decode a base64 audio-stream-chunk payload to an ArrayBuffer of raw int16 PCM.
function base64ToArrayBuffer(b64: string): ArrayBuffer {
  const bin = atob(b64);
  const len = bin.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i += 1) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

// A minimal HTMLAudioElement-like clock for driving vrmMotionPlayer's co-speech
// gesture from a STREAMED sentence — it reads only currentTime/paused/ended. The
// sentence's audio starts at `origin` (AudioContext time) and lasts `dur` s: while
// the audio is still buffered ahead (currentTime < origin) the clock reports
// paused so the gesture waits, then syncs to sentence-relative time once it plays.
function makeStreamClock(
  player: StreamingAudioPlayer,
  origin: number,
  dur: number,
): HTMLAudioElement {
  return {
    get currentTime() {
      return Math.max(0, Math.min(dur, player.currentTime() - origin));
    },
    get paused() {
      return player.currentTime() < origin;
    },
    get ended() {
      return player.currentTime() >= origin + dur - 0.02;
    },
    get duration() {
      return dur;
    },
    play() {},
    pause() {},
    addEventListener() {},
    removeEventListener() {},
  } as unknown as HTMLAudioElement;
}

// WebRTC variant (Phase G): there is no local audio buffer to clock against —
// the track is jitter-buffered by the browser — so a sentence's co-speech
// gesture is clocked by WALL TIME from its backend-announced play origin
// (audio-stream-start's play_in = the track backlog when it was queued).
function makeWallClock(startAtMs: number, dur: number): HTMLAudioElement {
  return {
    get currentTime() {
      return Math.max(0, Math.min(dur, (performance.now() - startAtMs) / 1000));
    },
    get paused() {
      return performance.now() < startAtMs;
    },
    get ended() {
      return (performance.now() - startAtMs) / 1000 >= dur - 0.02;
    },
    get duration() {
      return dur;
    },
    play() {},
    pause() {},
  } as unknown as HTMLAudioElement;
}

function WebSocketHandler({ children }: { children: React.ReactNode }) {
  const { t } = useTranslation();
  const [wsState, setWsState] = useState<string>('CLOSED');
  const [wsUrl, setWsUrl] = useLocalStorage<string>('wsUrl', defaultWsUrl);
  const [baseUrl, setBaseUrl] = useLocalStorage<string>('baseUrl', defaultBaseUrl);
  const {
    aiState, getAiState, setAiState, backendSynthComplete, setBackendSynthComplete,
  } = useAiState();
  const { setModelInfo } = useLive2DConfig();
  const { setSubtitleText } = useSubtitle();
  const {
    clearResponse, setForceNewMessage, appendHumanMessage, appendAIMessage,
    appendOrUpdateToolCallMessage, appendResponse,
  } = useChatHistory();
  const { addAudioTask } = useAudioTask();
  const bgUrlContext = useBgUrl();
  const {
    confName, confUid, configFiles, getFilenameByName, setConfName, setConfUid, setConfigFiles,
  } = useConfig();
  const [pendingModelInfo, setPendingModelInfo] = useState<ModelInfo | undefined>(undefined);
  const { setSelfUid, setGroupMembers, setIsOwner } = useGroup();
  const { startMic, stopMic, autoStartMicOnConvEnd } = useVAD();
  const autoStartMicOnConvEndRef = useRef(autoStartMicOnConvEnd);
  const { interrupt } = useInterrupt();
  const interruptRef = useRef(interrupt);
  interruptRef.current = interrupt;
  const { setBrowserViewData } = useBrowser();

  // --- true-TTS-streaming playback (audio-stream-* messages) ----------------
  // One StreamingAudioPlayer per turn plays the streamed raw-PCM gaplessly and
  // its live RMS drives the VRM mouth (vrmLipSync.onLiveStart). The classic
  // per-sentence HTMLAudioElement path (case 'audio') is untouched; only ONE of
  // the two runs, depending on whether the backend streams (GSV_STREAM=1).
  const streamPlayerRef = useRef<StreamingAudioPlayer | null>(null);
  // ONE Opus decoder per turn, not per sentence. Decoding is asynchronous, so a
  // second decoder running alongside the first would interleave its output into
  // the player and scramble the speech. The server restarts its encoder each
  // sentence, which this side simply reads as a packet-loss-shaped seam — Opus
  // is designed to survive that, and it costs one 20 ms frame of transient.
  const opusDecoderRef = useRef<WsOpusDecoder | null>(null);
  const closeOpusDecoder = useCallback(() => {
    opusDecoderRef.current?.close();
    opusDecoderRef.current = null;
  }, []);
  // AudioContext time at which the CURRENT streamed sentence's audio starts —
  // captured at its audio-stream-start (= player.scheduledEnd, the tail of the
  // buffered audio) so its co-speech gesture gets a per-sentence clock.
  const streamSentenceOriginRef = useRef(0);
  // Phase G (WebRTC): wall-clock (performance.now ms) at which the CURRENT
  // sentence starts playing on the remote track.
  const webrtcOriginRef = useRef(0);
  // Diagnostics: how many audio-stream-end frames carried a gesture id this turn.
  const gestureFramesRef = useRef(0);
  // Gesture ids already applied from their audio-stream-START frame (text-streaming
  // motion rides the start frame now), so the end frame doesn't re-apply them.
  const appliedGestureRef = useRef<Set<string>>(new Set());
  // Per-sentence event timers (subtitle/emotion/gesture). Because GSV synthesises
  // ~2.5x faster than realtime, whole sentences arrive buffered several seconds
  // AHEAD of playback; firing their subtitle/gesture on RECEIPT desyncs them from
  // speech. Instead each is scheduled to fire when its audio actually plays
  // (delay = origin - now). Cleared on interrupt / turn end.
  const streamTimersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const clearStreamTimers = useCallback(() => {
    streamTimersRef.current.forEach((tid) => clearTimeout(tid));
    streamTimersRef.current = [];
  }, []);

  const finishStreamingTurn = useCallback(() => {
    // The streamed audio has fully drained. NOW tell the backend playback is
    // complete (its finalize step waits for this before the per-turn cleanup that
    // cancels in-flight co-speech gestures) — reporting it here, not off the empty
    // task queue, is what keeps this turn's gestures alive long enough to play.
    streamingPlaybackState.active = false;
    // Motion-usage ledger: report which layer actually drove the body this turn
    // (logged server-side as [motion-usage] for measurement).
    wsService.sendMessage({
      type: 'motion-usage',
      ...vrmMotionPlayer.statsSnapshot(),
      gframes: gestureFramesRef.current,
      vrmOk: !!(window as any).__vrm,
    });
    wsService.sendMessage({ type: 'frontend-playback-complete' });
    // Tell the server what this browser did with the audio it was sent, so
    // "she has no voice" can be placed on one side of the wire from the log
    // alone. See StreamingAudioPlayer.health().
    if (streamPlayerRef.current) {
      wsService.sendMessage({
        type: 'audio-health',
        ...streamPlayerRef.current.health(),
      });
    }
    setBackendSynthComplete(false);
    clearStreamTimers();
    vrmLipSync.onLiveEnd();
    // Scheduled mouth shapes outlive the audio otherwise: on an interrupt the
    // queue would keep articulating a sentence that is no longer being spoken.
    vrmLipSync.clearVisemes();
    vrmMotionPlayer.cancelGesture(); // invalidate in-flight primary/upgrade polls at turn end
    // endTurn (not stop): keep the pose she finished on and let the first idle clip
    // CROSSFADE out of it — clearing the clip snapped the body toward rest (the ugly
    // talk->idle seam). stop() is still used for interrupts, where cutting is correct.
    vrmMotionPlayer.endTurn();
    vrmMotionPlayer.stopTalking();
    vrmExpression.relax();
    mmdExpression.relax();
    talkingMotionController.onSpeechEnd();
    // Drained, so stop() only closes its AudioContext. Dropping the ref alone
    // leaked one running context per reply — and in render mode each stays
    // wired into the tap that carries her voice to the PROTO.
    streamPlayerRef.current?.stop();
    streamPlayerRef.current = null;
    closeOpusDecoder();
    setAiState((s: AiState) => (s === 'thinking-speaking' ? 'idle' : s));
    if (autoStartMicOnConvEndRef.current) startMic();
  }, [setAiState, startMic, clearStreamTimers, setBackendSynthComplete, closeOpusDecoder]);

  const stopStreamingNow = useCallback(() => {
    streamingPlaybackState.active = false;
    clearStreamTimers();
    if (streamPlayerRef.current) {
      streamPlayerRef.current.stop();
      streamPlayerRef.current = null;
    }
    closeOpusDecoder();
    vrmLipSync.onLiveEnd();
    // Scheduled mouth shapes outlive the audio otherwise: on an interrupt the
    // queue would keep articulating a sentence that is no longer being spoken.
    vrmLipSync.clearVisemes();
    vrmMotionPlayer.cancelGesture(); // invalidate in-flight primary/upgrade polls
    vrmMotionPlayer.interrupt(); // hold the pose across the handover (see interrupt())
    vrmMotionPlayer.stopTalking();
  }, [clearStreamTimers, closeOpusDecoder]);

  // Interrupt (barge-in / user spoke): kill the stream immediately. interrupt()
  // calls it directly (streamingControl); the effect stays for other paths that
  // set 'interrupted'.
  useEffect(() => {
    if (aiState === 'interrupted') stopStreamingNow();
  }, [aiState, stopStreamingNow]);
  useEffect(() => {
    streamingControl.stopNow = stopStreamingNow;
    return () => {
      if (streamingControl.stopNow === stopStreamingNow) streamingControl.stopNow = null;
    };
  }, [stopStreamingNow]);

  // The browser is refusing to play her voice until this page gets a click.
  // Surfaced as a toast because nothing else about it is visible: the reply
  // streams, the body moves, the subtitles run, and only the sound is missing —
  // which reads as "the server broke" and sends you debugging the wrong end.
  useEffect(() => {
    const onBlocked = () => toaster.create({
      title: t('error.audioBlocked', 'Click anywhere to enable sound'),
      type: 'warning',
      duration: 6000,
    });
    window.addEventListener('kehai-audio-blocked', onBlocked);
    return () => window.removeEventListener('kehai-audio-blocked', onBlocked);
  }, [t]);

  useEffect(() => {
    autoStartMicOnConvEndRef.current = autoStartMicOnConvEnd;
  }, [autoStartMicOnConvEnd]);

  // HOLD the first model load while a saved character is waiting to be restored.
  // conf.yaml's default is Miku, so a session whose last character was Chancelor
  // loaded Miku's model, then switched and loaded Chancelor's — visible as Miku
  // flashing up first. Seeded from localStorage at mount (before any message can
  // arrive) and cleared the moment we know no switch is coming, or the switch has
  // landed.
  //
  // The timeout is only a backstop for a switch that never completes; it must NOT
  // be tight enough to fire during a normal startup or it re-creates the very bug
  // it guards. Measured on this machine: config-files -> switch-config ->
  // config-switched lands ~12s after navigation on a cold load (a 6s backstop fired
  // first, Miku loaded at 8.9s, and the flash was still there). 25s is well clear of
  // that while still bounding a blank stage if the backend never answers.
  const [restorePending, setRestorePending] = useState(() => {
    try {
      return !!localStorage.getItem('lastConfigFile');
    } catch {
      return false;
    }
  });
  useEffect(() => {
    if (!restorePending) return undefined;
    const t2 = setTimeout(() => {
      console.warn('[restore] character switch did not land — showing the default model');
      setRestorePending(false);
    }, 25000);
    return () => clearTimeout(t2);
  }, [restorePending]);

  useEffect(() => {
    if (pendingModelInfo && confUid && !restorePending) {
      setModelInfo(pendingModelInfo);
      setPendingModelInfo(undefined);
    }
  }, [pendingModelInfo, setModelInfo, confUid, restorePending]);

  const {
    setCurrentHistoryUid, setMessages, setHistoryList,
  } = useChatHistory();

  const handleControlMessage = useCallback((controlText: string) => {
    switch (controlText) {
      case 'start-mic':
        console.log('Starting microphone...');
        startMic();
        break;
      case 'stop-mic':
        console.log('Stopping microphone...');
        stopMic();
        break;
      case 'restart-mic':
        // The server has not received a mic frame for several seconds while the peer
        // still reports 'connected' — the media stalled without any state change, so
        // only the far end can tell. Rebuild the peer; the mic track comes back with it.
        console.warn('Server reports a stalled mic — restarting the WebRTC peer');
        webrtcAudio.forceReconnect();
        break;
      case 'user-speech-start':
        // The server's VAD opened a turn — past its proximity gate, so this is
        // speech it is going to answer. In server-VAD mode this, not the local
        // VAD, is the barge-in: the local one has no level gate and stopped her
        // for voices the server then ignored, leaving no reply at all.
        if (getAiState() === 'thinking-speaking' || streamPlayerRef.current) {
          interruptRef.current();
        }
        setAiState((s: AiState) => (s === 'idle' || s === 'interrupted' ? 'listening' : s));
        break;
      case 'listening-nod':
        // The visitor in front of her paused or finished a sentence (server VAD,
        // websocket_handler): a small nod, the "mm-hm" a listener gives
        // (vrm-viewer ATTEND.listenDeg).
        window.dispatchEvent(new CustomEvent('kehai-listen'));
        break;
      case 'conversation-chain-start':
        // Turn events for her gaze (vrm-viewer ATTEND): she looks away to think.
        window.dispatchEvent(new CustomEvent('kehai-turn', { detail: 'think' }));
        // A reply still playing here was superseded server-side (a new turn
        // replaced it). Its player must not carry over: the new reply's audio
        // would queue behind the old one's leftovers and skip the turn setup.
        if (streamPlayerRef.current) stopStreamingNow();
        setAiState('thinking-speaking');
        talkingMotionController.onThinkingStart(); // "thinking" pose until audio starts
        audioTaskQueue.clearQueue();
        clearResponse();
        break;
      case 'conversation-chain-end':
        window.dispatchEvent(new CustomEvent('kehai-turn', { detail: 'end' }));
        if (streamPlayerRef.current) {
          // Streaming path: the audioTaskQueue isn't used. Signal end-of-stream;
          // when the buffered audio drains the player's onEnded -> finishStreamingTurn
          // settles the avatar back to idle (mirrors the queue cleanup below).
          streamPlayerRef.current.end();
          break;
        }
        audioTaskQueue.addTask(() => new Promise<void>((resolve) => {
          // Reset the face and body to neutral when the reply finishes, so
          // the last sentence's expression (a frown, wide eyes...) doesn't
          // stay frozen on the model until the next conversation. A finished
          // motion's final pose persists on the model, so after stopping all
          // motions, immediately force-start an Idle beat — its rest-pose
          // channels ease any leftover gesture (raised arm etc.) back to
          // neutral instead of waiting for the next idle-cycle tick.
          // Fade the talking-motion layer out promptly now the reply is done.
          talkingMotionController.onSpeechEnd();
          // MMD: fade the HY-Motion body clip out, back to procedural idle,
          // and ease the face back to its resting expression.
          mmdMotionPlayer.stop();
          // VRM body: settle the co-speech motion to idle ONLY at the reply end (this
          // conversation-end control msg). Per-sentence clips no longer stop themselves
          // (they loop/crossfade so the whole reply is one continuous talking motion),
          // so this is the single place the body returns to idle when the AI finishes.
          // endTurn keeps her final pose and lets the first idle clip crossfade from it.
          vrmMotionPlayer.endTurn();
          vrmMotionPlayer.stopTalking();
          mmdExpression.relax();
          vrmExpression.relax();
          try {
            // Reset the FACE expression via the SAME path the per-sentence
            // expressions use (lappAdapter.setExpression) — model.setExpression
            // directly did not reliably clear the last emotion face.
            const lappAdapter = (window as any).getLAppAdapter?.();
            lappAdapter?.setExpression?.('face_normal');

            const live2dManager = (window as any).getLive2DManager?.();
            const model = live2dManager?.getModel?.(0);
            if (model) {
              // Cross-fade to an idle motion (FadeIn ~1s) instead of
              // stopAllMotions(), which abruptly cut the current pose.
              model.startRandomMotion?.('Idle', 3); // 3 = LAppDefine.PriorityForce
              if (typeof model._userTimeSeconds === 'number') {
                model._lastIdleMotionStartSeconds = model._userTimeSeconds;
              }
            }
            console.log('[reset] face+motion -> idle (conversation end)');
          } catch (e) {
            console.warn('resetOnIdle', e);
          }
          setAiState((currentState: AiState) => {
            if (currentState === 'thinking-speaking' || currentState === 'listening') {
              // Auto start mic if enabled
              if (autoStartMicOnConvEndRef.current) {
                startMic();
              }
              return 'idle';
            }
            return currentState;
          });
          resolve();
        }));
        break;
      default:
        console.warn('Unknown control command:', controlText);
    }
  }, [setAiState, getAiState, clearResponse, setForceNewMessage, startMic, stopMic, stopStreamingNow]);

  const handleWebSocketMessage = useCallback((message: MessageEvent) => {
    console.log('Received message from server:', message);
    switch (message.type) {
      case 'control':
        if (message.text) {
          handleControlMessage(message.text);
        }
        break;
      case 'set-model-and-conf':
        setAiState('loading');
        if (message.conf_name) {
          setConfName(message.conf_name);
        }
        if (message.conf_uid) {
          setConfUid(message.conf_uid);
          console.log('confUid', message.conf_uid);
          // The idle rotation is per character, and the motion player is a plain
          // class outside React, so it cannot read the config context. Hand it the
          // uid here — this is the one place the server announces which character
          // the session is now running.
          vrmMotionPlayer.confUid = message.conf_uid;
        }
        if (message.client_uid) {
          setSelfUid(message.client_uid);
        }
        setPendingModelInfo(message.model_info);
        // setModelInfo(message.model_info);
        // We don't know when the confRef in live2d-config-context will be updated, so we set a delay here for convenience
        if (message.model_info && !message.model_info.url.startsWith("http")) {
          const modelUrl = baseUrl + message.model_info.url;
          // eslint-disable-next-line no-param-reassign
          message.model_info.url = modelUrl;
        }

        setAiState('idle');
        break;
      case 'full-text':
        if (message.text) {
          setSubtitleText(message.text);
        }
        break;
      case 'config-files':
        if (message.configs) {
          setConfigFiles(message.configs);
        }
        break;
      case 'config-switched':
        setAiState('idle');
        setRestorePending(false); // the restore landed — this model may now load
        setSubtitleText(t('notification.characterLoaded'));

        toaster.create({
          title: t('notification.characterSwitched'),
          type: 'success',
          duration: 2000,
        });

        // setModelInfo(undefined);

        // Character switching stops the mic (use-switch-character) so the swap
        // isn't picked up as speech — bring it back automatically once the new
        // character is live, so EDI⇄レム switching stays hands-free.
        startMic();

        wsService.sendMessage({ type: 'fetch-history-list' });
        // NOTE: no 'create-new-history' here. The backend already attached this
        // character's most recent conversation when the config switched, and
        // forcing a new one wiped that continuity on every switch (Miku <-> Rem
        // could not remember anything from before the swap). Start a fresh
        // thread explicitly from the history sidebar instead.
        break;
      case 'background-files':
        if (message.files) {
          bgUrlContext?.setBackgroundFiles(message.files);
        }
        break;
      case 'audio':
        if (aiState === 'interrupted' || aiState === 'listening') {
          console.log('Audio playback intercepted. Sentence:', message.display_text?.text);
        } else {
          console.log("actions", message.actions);
          addAudioTask({
            audioBase64: message.audio || '',
            volumes: message.volumes || [],
            sliceLength: message.slice_length || 0,
            displayText: message.display_text || null,
            expressions: message.actions?.expressions || null,
            motions: message.actions?.motions || null,
            emotions: message.actions?.emotions || null,
            motion: message.actions?.motion || null,
            forwarded: message.forwarded || false,
          });
          // Pre-fetch THIS sentence's co-speech gesture NOW — this message arrives
          // while the PRIOR sentence is still playing (parallel/per-sentence TTS),
          // so by the time this one plays the clip is in hand and starts aligned.
          const gm = message.actions?.motion;
          if (gm?.id && gm.key === 'gesture') {
            if ((window as any).__vrm) {
              // VRM: pre-fetch the SMPL clip so it's in hand when this line plays
              // (else the 500ms fetch cap misses the ~1.8s clip -> the body loops).
              vrmMotionPlayer.prefetch(`/motion/clip-smpl/${gm.id}`, gm.id);
            } else {
              mmdMotionPlayer.prefetchGesture(
                `${mmdMotionPlayer.getBaseUrl()}/motion/clip/${gm.id}`,
                gm.id,
              );
            }
          }
        }
        break;
      case 'audio-stream-start': {
        // Her voice is starting (every sentence; the gaze only uses the first).
        window.dispatchEvent(new CustomEvent('kehai-turn', { detail: 'speak' }));
        // True-TTS-streaming: first sentence of the turn creates the gapless
        // player and wires the live-RMS mouth once; every sentence sets its own
        // subtitle + facial emotion. (Co-speech gesture in streaming mode is a
        // follow-up — the clip is generated ~2s after the audio, so it needs the
        // prefetch-while-playing timing the file path uses; the body idles here.)
        if (webrtcAudio.active) {
          // Phase G: audio rides the WebRTC track (no local player), so it
          // plays REGARDLESS of what this handler does. Drop frames only on a
          // real barge-in ('interrupted') — dropping on 'listening' (a client-
          // VAD race, or its end-of-speech wedging) silenced the mouth, the
          // talking body, subtitles and gestures while the voice kept playing
          // (the "voice but frozen body" bug). The wiring below is idempotent,
          // so re-running it every sentence also self-heals a missed turn start.
          if (aiState === 'interrupted') break;
          const openingSentenceRtc = !streamingPlaybackState.active; // see the WS branch
          if (!streamingPlaybackState.active) { vrmMotionPlayer.statsReset(); gestureFramesRef.current = 0; appliedGestureRef.current.clear(); }
          streamingPlaybackState.active = true;
          // SILENCE WATCHDOG. The server has just said "I am speaking now", so
          // audio must start arriving AND playing within a couple of seconds.
          // If it does neither, the turn is silent and nothing else in the app
          // notices — subtitles, mouth and gestures all run off these control
          // messages, so the only symptom is no sound. Rather than leave that
          // for someone to discover, drop to the WebSocket audio path, which
          // the server uses whenever no WebRTC peer exists.
          startSilenceWatchdog();
          vrmLipSync.onLiveStart(() => webrtcAudio.mouthValue(2.4));
          vrmMotionPlayer.suspendIdle(); // a turn is starting — no idle clips over speech
          vrmMotionPlayer.startTalkingLive(() => webrtcAudio.mouthValue(2.4));
          setAiState((s: AiState) => (s === 'thinking-speaking' ? s : 'thinking-speaking'));
          const playInMs = Math.max(0, ((message as any).play_in ?? 0) * 1000) + 120; // + jitter margin
          webrtcOriginRef.current = performance.now() + playInMs;
          {
            const dtxt = message.display_text?.text;
            const dname = message.display_text?.name;
            const davatar = message.display_text?.avatar;
            const emo = message.actions?.emotions?.[0];
            const tid = setTimeout(() => {
              if (dtxt) setSubtitleText(dtxt);
              // ...and into the chat panel. Only the per-file 'audio' path fed the
              // history (via addAudioTask -> appendAIMessage), so with TRUE TTS
              // STREAMING on, every reply was spoken but never written down.
              // Appended on the same timer as the subtitle so the panel follows the
              // voice, and so a barge-in that clears these timers also drops the
              // lines she never got to say. appendAIMessage merges consecutive AI
              // text, so the sentences of one turn land as one message.
              if (dtxt) appendAIMessage(dtxt, dname, davatar);
              if (dtxt) appendResponse(dtxt); // the heard part (see the WS branch)
              if (emo) vrmExpression.playEmotion(emo);
            }, playInMs);
            streamTimersRef.current.push(tid);
          }
          // TEXT-streaming motion: the clip id now rides the START frame (it was
          // fired from the sentence text, before synthesis) — prefetch NOW and
          // apply exactly at this sentence's play origin, so the body moves from
          // the first spoken frame instead of ~1s in (end-frame-only apply).
          {
            const gm2 = message.actions?.motion;
            if (gm2?.id && gm2.key === 'gesture' && (window as any).__vrm) {
              const url = `/motion/clip-smpl/${gm2.id}`;
              vrmMotionPlayer.prefetch(url, gm2.id, 80, 250);
              appliedGestureRef.current.add(gm2.id);
              const startAt = webrtcOriginRef.current;
              // Estimated duration (+margin): the real audio_secs isn't known yet;
              // generated-only mode holds the last frame past the estimate anyway.
              const dur = Math.max(0.5, (gm2.duration ?? 3)) + 1.5;
              const tid2 = setTimeout(() => {
                if (!streamingPlaybackState.active) return;
                // Keep GestureLSM CONTINUOUS. This start-frame path had no such guard
                // (only the two audio-stream-end handlers did), so EVERY sentence start
                // slammed its KAWAII talk-loop over a running GestureLSM clip — measured
                // 8 source flips in one turn, i.e. the visible KAWAII/GestureLSM seam.
                // Emote sentences (kind !== 'talk') still play: the pose is the point.
                if (!openingSentenceRtc && gm2.kind === 'talk' && vrmMotionPlayer.gestureLSMActive()) return;
                const clock = makeWallClock(startAt, dur);
                const clip = vrmMotionPlayer.takePrefetched(gm2.id);
                if (clip) vrmMotionPlayer.setClip(clip, clock);
                else vrmMotionPlayer.playGesture(url, { audio: clock, tries: 80, intervalMs: 250 });
              }, playInMs);
              streamTimersRef.current.push(tid2);
            }
          }
          break;
        }
        // Same race the WebRTC branch above already gave up on, but here it
        // costs the VOICE, not just the visuals: with no player built, every
        // audio-stream-chunk of the turn is dropped on the floor while motion
        // and subtitles (which ride audio-stream-end) keep running — "she moves
        // but makes no sound". The two VADs end a turn independently, and the
        // server's is FASTER: measured TTFA 600-800ms against a client VAD that
        // still needs its redemption frames, so audio-stream-start routinely
        // lands while aiState is 'listening' and the whole reply is lost. Only
        // a REAL barge-in ('interrupted', set by use-interrupt) may drop audio;
        // the server has already stopped sending for that turn anyway.
        if (aiState === 'interrupted') break;
        // The turn's FIRST sentence: whatever GestureLSM clip is on the body belongs to an
        // earlier (possibly interrupted) reply, so its continuity guard must not skip this
        // turn's opener — it did, and the body held 3.7 s at a reply start (2026-11-05).
        const openingSentence = !streamPlayerRef.current;
        if (!streamPlayerRef.current) {
          const player = new StreamingAudioPlayer({
            sampleRate: message.sample_rate || 32000,
            // Buffer lead before playback starts. Lower = faster first audio
            // (TTFA); too low = underrun (gaps) if the stream stalls. GSV streams
            // at ~2x realtime so the buffer refills fast — 0.5s was the safe
            // default (down from 1.5s, which was sized for the first co-speech
            // clip to land; gesture is being decoupled to a procedural base
            // layer). 0.25s now that the audio is Opus: the same speech is ~1/8
            // the bytes, so a chunk clears the link far faster and the jitter
            // this lead exists to absorb is correspondingly smaller. Straight
            // 250ms off the time to first word. Tune live WITHOUT a rebuild:
            // localStorage.streamPreroll = '0.5'; reload.
            //
            // Render-host mode (?render=1) runs on the server box itself, so the
            // chunks arrive over loopback with next to no jitter: 0.05s there.
            // Measured 2026-10-22 over ~35s replies: every turn shows exactly ONE
            // gap at 0.05, 0.1 and 0.25 alike (48 / 32-136 / 160-208ms), so that
            // gap is not jitter and a longer lead does not cover it.
            // The viewer's own playback worklet still absorbs the network leg.
            // health() reports `gaps` per turn if this ever proves too tight.
            prerollSec: (() => {
              const v = parseFloat(localStorage.getItem('streamPreroll') || '');
              if (Number.isFinite(v) && v >= 0) return v;
              return isRenderSource ? 0.05 : 0.25;
            })(),
            onEnded: () => finishStreamingTurn(),
            // Opus decoding is asynchronous, so "nothing left to play" and
            // "nothing has arrived yet" look identical to the drain check. The
            // decoder is built just below and lives in a ref, so read it lazily.
            morePending: () => (opusDecoderRef.current?.inFlight ?? 0) > 0,
          });
          streamPlayerRef.current = player;
          // The server picks the codec per sentence and says so here; it only
          // sends Opus to a client that declared it can decode (see audio-caps
          // in websocket-service). Decoder outlives the sentence — see the ref.
          if (message.codec === 'opus' && !opusDecoderRef.current) {
            const dec = new WsOpusDecoder((samples) => {
              streamPlayerRef.current?.enqueueFloat32(samples);
            });
            opusDecoderRef.current = dec.ok ? dec : null;
            if (!dec.ok) {
              console.warn('[ws-opus] server sent Opus but this client cannot decode it');
            }
          }
          vrmMotionPlayer.statsReset();
          gestureFramesRef.current = 0;
          appliedGestureRef.current.clear();
          streamingPlaybackState.active = true;
          // scale 2.4: live PCM RMS runs lower than the old normalized volumes[],
          // so the mouth needs more gain to open convincingly.
          vrmLipSync.onLiveStart(() => player.mouthValue(2.4));
          // D1 (procedural talking body): the SAME live RMS drives the volume-
          // scaled talking motion (torso/head/arm sway), so the body moves from
          // the FIRST streamed chunk — the GestureLSM clip crossfades over it
          // when it arrives ~1-2s later.
          vrmMotionPlayer.suspendIdle(); // a turn is starting — no idle clips over speech
          vrmMotionPlayer.startTalkingLive(() => player.mouthValue(2.4));
          setAiState((s: AiState) => (s === 'thinking-speaking' ? s : 'thinking-speaking'));
        }
        {
          const sp = streamPlayerRef.current;
          // This sentence's audio plays starting at the tail of what's buffered.
          const origin = sp.scheduledEnd();
          streamSentenceOriginRef.current = origin;
          // Defer subtitle + facial emotion to when this sentence actually PLAYS
          // (it's buffered several seconds ahead), so they track the speech instead
          // of all flashing up front.
          const delayMs = Math.max(0, (origin - sp.currentTime()) * 1000);
          const dtxt = message.display_text?.text;
          const dname = message.display_text?.name;
          const davatar = message.display_text?.avatar;
          const emo = message.actions?.emotions?.[0];
          const tid = setTimeout(() => {
            if (dtxt) setSubtitleText(dtxt);
            if (dtxt) appendAIMessage(dtxt, dname, davatar); // see the WebRTC branch
            // What she has actually begun to say: a barge-in reports this as the
            // heard part (use-interrupt), so the history keeps her reply only up
            // to where she was cut off. Streaming never filled it before, so
            // every interrupt reported "" and her reply vanished from the file.
            if (dtxt) appendResponse(dtxt);
            if (emo) vrmExpression.playEmotion(emo);
          }, delayMs);
          streamTimersRef.current.push(tid);
          // MOTION FROM THE START FRAME, as the WebRTC branch above does. The clip id
          // rides this frame (fired from the sentence text), but this path only applied
          // it on audio-stream-end — i.e. once the whole sentence was synthesized, which
          // for a reply's FIRST sentence is after it has begun playing: measured 0.46 s
          // and 1.75 s of no motion at the start of a reply (2026-11-05). Prefetch now,
          // apply at this sentence's play origin; the end frame then only adds the upgrade.
          const gm0 = message.actions?.motion;
          if (gm0?.id && gm0.key === 'gesture' && (window as any).__vrm) {
            const url0 = `/motion/clip-smpl/${gm0.id}`;
            vrmMotionPlayer.prefetch(url0, gm0.id, 80, 250);
            appliedGestureRef.current.add(gm0.id);
            // The real length is not known until the end frame; the estimate + margin
            // only bounds the clock (a clip that runs out holds or wraps while speaking).
            const dur0 = Math.max(0.5, gm0.duration ?? 3) + 1.5;
            const tid0 = setTimeout(() => {
              if (!streamPlayerRef.current) return;
              if (!openingSentence && gm0.kind === 'talk' && vrmMotionPlayer.gestureLSMActive()) return; // keep GestureLSM continuous
              const clock = makeStreamClock(sp, origin, dur0);
              const clip = vrmMotionPlayer.takePrefetched(gm0.id);
              if (clip) vrmMotionPlayer.setClip(clip, clock);
              else vrmMotionPlayer.playGesture(url0, { audio: clock, tries: 80, intervalMs: 250 });
            }, delayMs);
            streamTimersRef.current.push(tid0);
          }
        }
        break;
      }
      case 'audio-stream-chunk': {
        const p = streamPlayerRef.current;
        if (!p || !message.data) break;
        // Codec is read off EACH chunk, not off the start frame: if the encoder
        // throws mid-sentence the server drops back to PCM rather than going
        // silent, and this follows it without needing to be told.
        if (message.codec === 'opus') {
          opusDecoderRef.current?.decode(base64ToArrayBuffer(message.data));
        } else {
          p.enqueuePCM(base64ToArrayBuffer(message.data));
        }
        break;
      }
      case 'audio-stream-end': {
        // MOUTH SHAPES for this sentence, if the TTS backend produced them.
        // They ride the END frame because the character timings only complete
        // with the last audio chunk — but they are timed from the sentence's
        // START, and webrtcOriginRef is exactly when that start plays. Same
        // reference the co-speech gesture below uses, for the same reason.
        // Absent (GPT-SoVITS has no timings) leaves the volume-driven jaw alone.
        {
          const vis = (message as any).visemes;
          if (Array.isArray(vis) && vis.length) {
            const startAtS = (webrtcOriginRef.current || performance.now()) / 1000;
            vrmLipSync.setVisemes(vis, startAtS);
          }
        }
        // All of this sentence's PCM is queued now. Its co-speech gesture id rode
        // the end frame. PREFETCH the clip immediately (max lead while the sentence
        // is still buffered ahead — clips take ~2s to generate), then APPLY it
        // synced when the sentence reaches its play position.
        if (webrtcAudio.active) {
          const gmw = message.actions?.motion;
          if (gmw?.id && gmw.key === 'gesture') gestureFramesRef.current += 1;
          if (gmw?.id && gmw.key === 'gesture' && (window as any).__vrm) {
            const url = `/motion/clip-smpl/${gmw.id}`;
            const startAt = webrtcOriginRef.current || performance.now();
            const dur = Math.max(0.1, (message as any).audio_secs || 3);
            // The KAWAII primary already rode the START frame (text-streaming) — don't
            // re-apply it, BUT still schedule the GestureLSM upgrade below (the old
            // early `break` here skipped the upgrade entirely in WebRTC mode).
            if (!appliedGestureRef.current.has(gmw.id)) {
              vrmMotionPlayer.prefetch(url, gmw.id, 80, 250);
              const tid = setTimeout(() => {
                if (!streamingPlaybackState.active) return;
                if (gmw.kind === 'talk' && vrmMotionPlayer.gestureLSMActive()) return; // keep GestureLSM continuous
                const clock = makeWallClock(startAt, dur);
                const clip = vrmMotionPlayer.takePrefetched(gmw.id);
                if (clip) vrmMotionPlayer.setClip(clip, clock);
                else vrmMotionPlayer.playGesture(url, { audio: clock, tries: 80, intervalMs: 250 });
              }, Math.max(0, startAt - performance.now()));
              streamTimersRef.current.push(tid);
            }
            // HYBRID: GestureLSM co-speech upgrade — crossfade OVER the KAWAII loop the
            // moment it lands (upgradeGesture never cancels the primary's poll). Nearly
            // immediate schedule + fast poll = minimal opener catch-up latency.
            if (gmw.upgrade_id) {
              const uurl = `/motion/clip-smpl/${gmw.upgrade_id}`;
              const tid2 = setTimeout(() => {
                if (!streamingPlaybackState.active) return;
                // +3s slack: see the streaming branch — lets the opener clip apply late
                // in a short clause and bridge into the next sentence.
                const clock = makeWallClock(startAt, dur + 3);
                vrmMotionPlayer.upgradeGesture(uurl, { audio: clock, tries: 160, intervalMs: 150 });
              }, Math.max(0, startAt - performance.now()) + 60);
              streamTimersRef.current.push(tid2);
            }
          }
          break;
        }
        const pe = streamPlayerRef.current;
        const gm = message.actions?.motion;
        // Count the gesture-carrying end frames on THIS path too. The increment
        // used to exist only in the WebRTC branch above, so on any box where ICE
        // fails and the frontend falls back to WS audio — which is every run on
        // PC-B, see docs/PCB-HANDOFF.md — [motion-usage] reported gesture-frames 0
        // for three days while the gestures were in fact playing. A metric that
        // reads 0 on the transport it is measured on is worse than no metric: it
        // was read as "co-speech motion is dead" and sent us looking for a fault
        // that was not there.
        if (gm?.id && gm.key === 'gesture') gestureFramesRef.current += 1;
        if (pe && gm?.id && gm.key === 'gesture' && (window as any).__vrm) {
          const origin = streamSentenceOriginRef.current;
          const dur = Math.max(0.1, pe.scheduledEnd() - origin);
          const url = `/motion/clip-smpl/${gm.id}`;
          // ~20s poll budget: streamed sentences synthesise in a burst so their
          // gestures queue behind each other on GestureLSM — poll far longer than
          // the file path's default (each clip still lands before its late sentence).
          const delayMs = Math.max(0, (origin - pe.currentTime()) * 1000);
          // Already applied from the start frame (above) — only the upgrade is new here.
          if (!appliedGestureRef.current.has(gm.id)) {
            vrmMotionPlayer.prefetch(url, gm.id, 80, 250);
            const tid = setTimeout(() => {
              if (!streamPlayerRef.current) return;
              // Keep GestureLSM CONTINUOUS: skip this sentence's KAWAII talk-loop while a
              // GestureLSM clip is still driving (its own upgrade below chains on). Emote
              // sentences (kind !== 'talk') always play their cute pose.
              if (gm.kind === 'talk' && vrmMotionPlayer.gestureLSMActive()) return;
              const clock = makeStreamClock(pe, origin, dur);
              const clip = vrmMotionPlayer.takePrefetched(gm.id);
              if (clip) vrmMotionPlayer.setClip(clip, clock);
              else vrmMotionPlayer.playGesture(url, { audio: clock, tries: 80, intervalMs: 250 });
            }, delayMs);
            streamTimersRef.current.push(tid);
          }
          // HYBRID: GestureLSM co-speech upgrade — crossfade OVER the instant KAWAII
          // loop the moment it lands. upgradeGesture never claims gestureSeq, so it
          // cannot cancel the primary's in-flight poll; near-immediate schedule + fast
          // poll minimizes the opener catch-up (backend fires it at the 1st PCM chunk).
          // If GestureLSM never arrives (short/failed) the loop just keeps playing.
          if (gm.upgrade_id) {
            const uurl = `/motion/clip-smpl/${gm.upgrade_id}`;
            const tid2 = setTimeout(() => {
              if (!streamPlayerRef.current) return;
              // +3s SLACK: the opener clip is fired from a SHORT comma-split clause and
              // lands near/after that clause's end — without slack the stale-drop killed
              // it and GestureLSM only started at sentence 2. The slack lets it apply and
              // BRIDGE across the boundary; the next sentence's (chained) upgrade replaces
              // it within ~0.1-1s, so the bridge stays continuous and voice-synced.
              const clock = makeStreamClock(pe, origin, dur + 3);
              vrmMotionPlayer.upgradeGesture(uurl, { audio: clock, tries: 160, intervalMs: 150 });
            }, delayMs + 60);
            streamTimersRef.current.push(tid2);
          }
        }
        break;
      }
      case 'history-data':
        if (message.messages) {
          setMessages(message.messages);
        }
        toaster.create({
          title: t('notification.historyLoaded'),
          type: 'success',
          duration: 2000,
        });
        break;
      case 'new-history-created':
        setAiState('idle');
        setSubtitleText(t('notification.newConversation'));
        // No need to open mic here
        if (message.history_uid) {
          setCurrentHistoryUid(message.history_uid);
          setMessages([]);
          const newHistory: HistoryInfo = {
            uid: message.history_uid,
            latest_message: null,
            timestamp: new Date().toISOString(),
          };
          setHistoryList((prev: HistoryInfo[]) => [newHistory, ...prev]);
          toaster.create({
            title: t('notification.newChatHistory'),
            type: 'success',
            duration: 2000,
          });
        }
        break;
      case 'history-deleted':
        toaster.create({
          title: message.success
            ? t('notification.historyDeleteSuccess')
            : t('notification.historyDeleteFail'),
          type: message.success ? 'success' : 'error',
          duration: 2000,
        });
        break;
      case 'history-list':
        if (message.histories) {
          setHistoryList(message.histories);
          if (message.histories.length > 0) {
            setCurrentHistoryUid(message.histories[0].uid);
          }
        }
        break;
      case 'user-input-transcription':
        console.log('user-input-transcription: ', message.text);
        if (message.text) {
          appendHumanMessage(message.text);
        }
        break;
      case 'backchannel':
        if (message.audio) void playBackchannel(message.audio);
        break;
      case 'error':
        toaster.create({
          title: message.message,
          type: 'error',
          duration: 2000,
        });
        break;
      case 'group-update':
        console.log('Received group-update:', message.members);
        if (message.members) {
          setGroupMembers(message.members);
        }
        if (message.is_owner !== undefined) {
          setIsOwner(message.is_owner);
        }
        break;
      case 'group-operation-result':
        toaster.create({
          title: message.message,
          type: message.success ? 'success' : 'error',
          duration: 2000,
        });
        break;
      case 'playback-drained':
        // Server-authoritative end of the turn's audio (WebRTC track drained).
        if (webrtcAudio.active && streamingPlaybackState.active && !streamPlayerRef.current) {
          finishStreamingTurn();
        }
        break;
      case 'webrtc-answer':
        if ((message as any).sdp) webrtcAudio.onAnswer((message as any).sdp);
        break;
      case 'backend-synth-complete':
        setBackendSynthComplete(true);
        if (webrtcAudio.active && streamingPlaybackState.active && !streamPlayerRef.current) {
          // Phase G: the BACKEND watches its track buffer and sends
          // 'playback-drained' at the true end of playback (the old analyser
          // silence-watch misfired on quiet passages / pre-playback silence and
          // killed gestures mid-turn). Keep only a long failsafe here.
          const failsafe = setTimeout(() => {
            if (streamingPlaybackState.active && !streamPlayerRef.current) finishStreamingTurn();
          }, 90_000);
          streamTimersRef.current.push(failsafe);
          break;
        }
        if (streamPlayerRef.current) {
          // Streaming: all chunks received. Signal end-of-stream so the player
          // drains the buffered audio and then reports playback-complete
          // (finishStreamingTurn) — the real "this turn finished playing" moment.
          streamPlayerRef.current.end();
        } else {
          // No more audio is coming this turn — stop holding back for the
          // look-ahead buffer and play whatever is queued straight through.
          audioTaskQueue.setStreamComplete(true);
        }
        break;
      case 'conversation-chain-end':
        if (!audioTaskQueue.hasTask()) {
          setAiState((currentState: AiState) => {
            if (currentState === 'thinking-speaking') {
              return 'idle';
            }
            return currentState;
          });
        }
        break;
      case 'force-new-message':
        setForceNewMessage(true);
        break;
      case 'interrupt-signal':
        // Handle forwarded interrupt
        interrupt(false); // do not send interrupt signal to server
        break;
      case 'tool_call_status':
        if (message.tool_id && message.tool_name && message.status) {
          // If there's browser view data included, store it in the browser context
          if (message.browser_view) {
            console.log('Browser view data received:', message.browser_view);
            setBrowserViewData(message.browser_view);
          }

          appendOrUpdateToolCallMessage({
            id: message.tool_id,
            type: 'tool_call_status',
            role: 'ai',
            tool_id: message.tool_id,
            tool_name: message.tool_name,
            name: message.name,
            status: message.status as ('running' | 'completed' | 'error'),
            content: message.content || '',
            timestamp: message.timestamp || new Date().toISOString(),
          });
        } else {
          console.warn('Received incomplete tool_call_status message:', message);
        }
        break;
      default:
        console.warn('Unknown message type:', message.type);
    }
  }, [aiState, addAudioTask, appendHumanMessage, appendAIMessage, setRestorePending, baseUrl, bgUrlContext, setAiState, setConfName, setConfUid, setConfigFiles, setCurrentHistoryUid, setHistoryList, setMessages, setModelInfo, setSubtitleText, startMic, stopMic, setSelfUid, setGroupMembers, setIsOwner, backendSynthComplete, setBackendSynthComplete, clearResponse, handleControlMessage, appendOrUpdateToolCallMessage, appendResponse, interrupt, setBrowserViewData, t]);

  useEffect(() => {
    wsService.connect(wsUrl);
  }, [wsUrl]);

  // Restore the user's last-selected character across page reloads. Once the
  // backend's default has loaded (confName set) and the config list is in,
  // switch to the saved file if it differs. Fires once per mount.
  const hasRestoredCharacterRef = useRef(false);
  useEffect(() => {
    if (hasRestoredCharacterRef.current) return;
    if (!confName || configFiles.length === 0) return;
    hasRestoredCharacterRef.current = true;
    let saved: string | null = null;
    try {
      saved = localStorage.getItem('lastConfigFile');
    } catch {
      saved = null;
    }
    // Every path that does NOT switch has to release the model-load hold above,
    // or a stale/removed lastConfigFile would leave the stage empty for 6s.
    if (!saved) { setRestorePending(false); return; }
    if (!configFiles.some((c) => c.filename === saved)) { setRestorePending(false); return; }
    if (saved !== getFilenameByName(confName)) {
      wsService.sendMessage({ type: 'switch-config', file: saved });
    } else {
      setRestorePending(false); // already the right character — show it now
    }
  }, [confName, configFiles, getFilenameByName]);

  useEffect(() => {
    const stateSubscription = wsService.onStateChange(setWsState);
    const messageSubscription = wsService.onMessage(handleWebSocketMessage);
    return () => {
      stateSubscription.unsubscribe();
      messageSubscription.unsubscribe();
    };
  }, [wsUrl, handleWebSocketMessage]);

  const webSocketContextValue = useMemo(() => ({
    sendMessage: wsService.sendMessage.bind(wsService),
    wsState,
    reconnect: () => wsService.connect(wsUrl),
    wsUrl,
    setWsUrl,
    baseUrl,
    setBaseUrl,
  }), [wsState, wsUrl, baseUrl]);

  return (
    <WebSocketContext.Provider value={webSocketContextValue}>
      {children}
    </WebSocketContext.Provider>
  );
}

export default WebSocketHandler;
