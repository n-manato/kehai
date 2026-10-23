/* eslint-disable no-use-before-define */
import {
  createContext, useContext, useRef, useCallback, useEffect, useReducer, useMemo,
} from 'react';
import { useTranslation } from 'react-i18next';
import { MicVAD } from '@ricky0123/vad-web';
import { useInterrupt } from '@/components/canvas/live2d';
import { audioTaskQueue } from '@/utils/task-queue';
import { useSendAudio } from '@/hooks/utils/use-send-audio';
import { useWebSocket } from '@/context/websocket-context';
import { duckStreamingAudio } from '@/utils/streaming-audio-player';
import { webrtcAudio } from '@/utils/webrtc-audio';
import { SubtitleContext } from './subtitle-context';
import { AiStateContext, AiState } from './ai-state-context';
import { useLocalStorage } from '@/hooks/utils/use-local-storage';
import { toaster } from '@/components/ui/toaster';
import { headTracker } from '@/utils/space/head-tracker';

/**
 * VAD settings configuration interface
 * @interface VADSettings
 */
export interface VADSettings {
  /** Threshold for positive speech detection (0-100) */
  positiveSpeechThreshold: number;

  /** Threshold for negative speech detection (0-100) */
  negativeSpeechThreshold: number;

  /** Number of frames for speech redemption */
  redemptionFrames: number;
}

/**
 * VAD context state interface
 * @interface VADState
 */
interface VADState {
  /** Auto stop mic feature state */
  autoStopMic: boolean;

  /** True when the mic must stay open for the session (server VAD / WebRTC track).
   *  Check before any side-effect call to stopMic(). */
  isContinuousMic: () => boolean;

  /** Microphone active state */
  micOn: boolean;

  /** Set microphone state */
  setMicOn: (value: boolean) => void;

  /** Set Auto stop mic state */
  setAutoStopMic: (value: boolean) => void;

  /** Start microphone and VAD */
  startMic: () => Promise<void>;

  /** Stop microphone and VAD */
  stopMic: () => void;

  /** Previous speech probability value */
  previousTriggeredProbability: number;

  /** Set previous speech probability */
  setPreviousTriggeredProbability: (value: number) => void;

  /** VAD settings configuration */
  settings: VADSettings;

  /** Update VAD settings */
  updateSettings: (newSettings: VADSettings) => void;

  /** Auto start microphone when AI starts speaking */
  autoStartMicOn: boolean;

  /** Set auto start microphone state */
  setAutoStartMicOn: (value: boolean) => void;

  /** Auto start microphone when conversation ends */
  autoStartMicOnConvEnd: boolean;

  /** Set auto start microphone when conversation ends state */
  setAutoStartMicOnConvEnd: (value: boolean) => void;
}

/**
 * Default values and constants
 */
const DEFAULT_VAD_SETTINGS: VADSettings = {
  positiveSpeechThreshold: 50,
  negativeSpeechThreshold: 35,
  // 2026-07-21: 35->18. redemptionFrames = consecutive silence frames (Silero v5
  // = 32ms/frame) before end-of-speech fires. 35 => ~1.1s of "wait after you
  // stop talking" before the turn even starts (a big chunk of perceived lag).
  // 18 => ~0.58s: still tolerates brief mid-utterance pauses but responds much
  // sooner. Live-tunable in Settings > ASR > "Redemption Frames" (persisted to
  // localStorage, which OVERRIDES this default for existing users). Raise if it
  // cuts off mid-sentence.
  redemptionFrames: 18,
};

const DEFAULT_VAD_STATE = {
  micOn: false,
  autoStopMic: false,
  autoStartMicOn: false,
  autoStartMicOnConvEnd: false,
};

/**
 * Create the VAD context
 */
export const VADContext = createContext<VADState | null>(null);

/**
 * VAD Provider Component
 * Manages voice activity detection and microphone state
 *
 * @param {Object} props - Provider props
 * @param {React.ReactNode} props.children - Child components
 */
export function VADProvider({ children }: { children: React.ReactNode }) {
  const { t } = useTranslation();
  // Refs for VAD instance and state
  const vadRef = useRef<MicVAD | null>(null);
  const previousTriggeredProbabilityRef = useRef(0);
  const previousAiStateRef = useRef<AiState>('idle');

  // Persistent state management
  const [micOn, setMicOn] = useLocalStorage('micOn', DEFAULT_VAD_STATE.micOn);
  const autoStopMicRef = useRef(true);
  const [autoStopMic, setAutoStopMicState] = useLocalStorage(
    'autoStopMic',
    DEFAULT_VAD_STATE.autoStopMic,
  );
  const [settings, setSettings] = useLocalStorage<VADSettings>(
    'vadSettings',
    DEFAULT_VAD_SETTINGS,
  );
  const [autoStartMicOn, setAutoStartMicOnState] = useLocalStorage(
    'autoStartMicOn',
    DEFAULT_VAD_STATE.autoStartMicOn,
  );
  const autoStartMicRef = useRef(false);
  const [autoStartMicOnConvEnd, setAutoStartMicOnConvEndState] = useLocalStorage(
    'autoStartMicOnConvEnd',
    DEFAULT_VAD_STATE.autoStartMicOnConvEnd,
  );
  const autoStartMicOnConvEndRef = useRef(false);

  // Force update mechanism for ref updates
  const [, forceUpdate] = useReducer((x) => x + 1, 0);

  // External hooks and contexts
  const { interrupt } = useInterrupt();
  const { sendAudioPartition } = useSendAudio();
  const { sendMessage } = useWebSocket();
  const { setSubtitleText } = useContext(SubtitleContext)!;
  const { aiState, setAiState } = useContext(AiStateContext)!;

  // Refs for callback stability
  const interruptRef = useRef(interrupt);
  const sendAudioPartitionRef = useRef(sendAudioPartition);
  const sendMessageRef = useRef(sendMessage);
  // Server-VAD continuous upstream (default ON): every 32ms mic frame streams
  // to the backend, whose Silero VAD segments + transcribes DURING speech so
  // turn-end only waits for the last short segment. Opt out (legacy client
  // batch path): localStorage.serverVad = '0' + conf.yaml vad_model: null.
  const serverVadRef = useRef(localStorage.getItem('serverVad') !== '0');

  /** True when the mic is meant to stay OPEN for the whole session: the server's VAD
   *  (or the WebRTC track) decides where turns end, and feedback is handled by AEC +
   *  ducking rather than by closing the mic. Anything that would call stopMic() as a
   *  side effect must check this first — exported so there is ONE definition; having
   *  it inline here meant the text-input path could (and did) miss it and leave the
   *  microphone dead after a single typed message. */
  const isContinuousMic = useCallback(
    () => serverVadRef.current || webrtcAudio.active,
    [],
  );
  const aiStateRef = useRef<AiState>(aiState);
  const setSubtitleTextRef = useRef(setSubtitleText);
  const setAiStateRef = useRef(setAiState);

  const isProcessingRef = useRef(false);
  /** performance.now() when the current utterance started, for the addressee check. */
  const speechStartRef = useRef(0);
  /**
   * The current utterance is not to her: visitors talking among themselves
   * near the PROTO (nobody facing it — headTracker.addressed). It must not
   * interrupt her or move the conversation state; the server ignores the
   * turn on the same evidence.
   */
  const asideRef = useRef(false);

  // Update refs when dependencies change
  useEffect(() => {
    aiStateRef.current = aiState;
  }, [aiState]);

  useEffect(() => {
    interruptRef.current = interrupt;
  }, [interrupt]);

  useEffect(() => {
    sendAudioPartitionRef.current = sendAudioPartition;
  }, [sendAudioPartition]);

  useEffect(() => {
    sendMessageRef.current = sendMessage;
  }, [sendMessage]);

  useEffect(() => {
    setSubtitleTextRef.current = setSubtitleText;
  }, [setSubtitleText]);

  useEffect(() => {
    setAiStateRef.current = setAiState;
  }, [setAiState]);

  useEffect(() => {
    autoStopMicRef.current = autoStopMic;
  }, []);

  useEffect(() => {
    autoStartMicRef.current = autoStartMicOn;
  }, []);

  useEffect(() => {
    autoStartMicOnConvEndRef.current = autoStartMicOnConvEnd;
  }, []);

  /**
   * Update previous triggered probability and force re-render
   */
  const setPreviousTriggeredProbability = useCallback((value: number) => {
    previousTriggeredProbabilityRef.current = value;
    forceUpdate();
  }, []);

  /**
   * Handle speech start event (initial detection)
   */
  const handleSpeechStart = useCallback(() => {
    console.log('Speech started - saving current state');
    // Save current AI state but DON'T change to listening yet
    previousAiStateRef.current = aiStateRef.current;
    isProcessingRef.current = true;
    speechStartRef.current = performance.now();
    asideRef.current = false;
    // Duck the AI's streaming voice IMMEDIATELY (before speech is even
    // confirmed) so speaker echo doesn't feed back into the VAD during the
    // confirmation window; restored on misfire. A confirmed barge-in stops the
    // stream entirely anyway.
    duckStreamingAudio(true);
    webrtcAudio.duck(true);
    // Don't change state here - wait for onSpeechRealStart
  }, []);

  /**
   * Handle real speech start event (confirmed speech)
   */
  const handleSpeechRealStart = useCallback(() => {
    if (headTracker.addressed(speechStartRef.current) === false) {
      // Speech nearby, but nobody facing her: not a barge-in. Give her voice
      // back its volume too — the duck was only for the confirmation window.
      console.log('Real speech confirmed, but not to her (nobody facing) - not interrupting');
      asideRef.current = true;
      duckStreamingAudio(false);
      webrtcAudio.duck(false);
      return;
    }
    console.log('Real speech confirmed - checking if need to interrupt');
    // Check if we need to interrupt based on the PREVIOUS state (before speech started)
    if (previousAiStateRef.current === 'thinking-speaking') {
      console.log('Interrupting AI speech due to user speaking');
      interruptRef.current();
    }
    // Now change to listening state
    setAiStateRef.current('listening');
  }, []);

  /**
   * Handle frame processing event
   */
  const handleFrameProcessed = useCallback(
    (probs: { isSpeech: number }, frame: Float32Array) => {
      if (probs.isSpeech > previousTriggeredProbabilityRef.current) {
        setPreviousTriggeredProbability(probs.isSpeech);
      }
      // Continuous upstream for the server-side VAD: stream EVERY 32ms frame so
      // the backend segments + transcribes while the user is still speaking
      // (the overlap that hides ASR latency). When the WebRTC peer is live the
      // mic already rides its track, so the JSON frames are skipped.
      if (serverVadRef.current && frame && !webrtcAudio.active) {
        sendMessageRef.current({
          type: 'raw-audio-data',
          audio: Array.from(frame),
        });
      }
    },
    [],
  );

  /**
   * Handle speech end event
   */
  const handleSpeechEnd = useCallback((audio: Float32Array) => {
    if (!isProcessingRef.current) return;
    console.log('Speech ended');
    duckStreamingAudio(false); // restore volume (no-op if the stream was interrupted)
    webrtcAudio.duck(false);
    // Re-judged over the whole utterance: someone may have turned to her
    // mid-sentence, and the server decides on the full span too.
    const aside = asideRef.current && headTracker.addressed(speechStartRef.current) === false;
    // Became addressed after all: the server is about to answer it, so this is
    // the barge-in that was held back at the start.
    if (asideRef.current && !aside && previousAiStateRef.current === 'thinking-speaking') {
      interruptRef.current();
    }
    asideRef.current = false;
    // An aside leaves her reply alone: clearing the queue here would drop the
    // rest of what she is saying.
    if (!aside) audioTaskQueue.clearQueue();

    // CONTINUOUS-mic modes (server-side VAD or the WebRTC track) must NEVER
    // stop the mic on speech end — the whole design is an always-open mic with
    // AEC + ducking handling feedback, and the backend deciding turn ends.
    // autoStopMic (a legacy anti-feedback setting) only applies to the old
    // client-VAD batch path; honoring it here made the mic die after every
    // utterance ("話し終えたらマイクがOFFになる").
    const continuousMic = isContinuousMic();
    if (autoStopMicRef.current && !continuousMic) {
      stopMic();
    } else {
      console.log('Keeping mic active (continuous mode or auto-stop off)');
    }

    setPreviousTriggeredProbability(0);
    // Server-VAD mode: frames were already streamed; the backend's VAD decides
    // the turn end and triggers the conversation itself. Only the legacy batch
    // path sends the whole utterance here.
    if (!serverVadRef.current && !aside) {
      sendAudioPartitionRef.current(audio);
    }
    isProcessingRef.current = false;
    // Not a turn: whatever state the conversation is in stays as it is.
    if (!aside) setAiStateRef.current("thinking-speaking");
  }, []);

  /**
   * Handle VAD misfire event
   */
  const handleVADMisfire = useCallback(() => {
    if (!isProcessingRef.current) return;
    console.log('VAD misfire detected');
    duckStreamingAudio(false); // not real speech — restore the AI's volume
    webrtcAudio.duck(false);
    setPreviousTriggeredProbability(0);
    isProcessingRef.current = false;

    // Restore previous AI state and show helpful misfire message
    setAiStateRef.current(previousAiStateRef.current);
    setSubtitleTextRef.current(t('error.vadMisfire'));
  }, [t]);

  /**
   * Update VAD settings and restart if active
   */
  const updateSettings = useCallback((newSettings: VADSettings) => {
    setSettings(newSettings);
    if (vadRef.current) {
      stopMic();
      setTimeout(() => {
        startMic();
      }, 100);
    }
  }, []);

  /**
   * Initialize new VAD instance
   */
  const initVAD = async () => {
    const newVAD = await MicVAD.new({
      model: "v5",
      preSpeechPadFrames: 20,
      positiveSpeechThreshold: settings.positiveSpeechThreshold / 100,
      negativeSpeechThreshold: settings.negativeSpeechThreshold / 100,
      redemptionFrames: settings.redemptionFrames,
      baseAssetPath: './libs/',
      onnxWASMBasePath: './libs/',
      onSpeechStart: handleSpeechStart,
      onSpeechRealStart: handleSpeechRealStart,
      onFrameProcessed: handleFrameProcessed,
      onSpeechEnd: handleSpeechEnd,
      onVADMisfire: handleVADMisfire,
    });

    vadRef.current = newVAD;
    newVAD.start();
  };

  /**
   * Start microphone and VAD processing
   */
  const startMic = useCallback(async () => {
    try {
      if (!vadRef.current) {
        console.log('Initializing VAD');
        await initVAD();
      } else {
        console.log('Starting VAD');
        vadRef.current.start();
      }
      // Phase G: bring the WebRTC audio peer up with the mic (idempotent; the
      // mic gesture also satisfies autoplay for the AI-voice track). The local
      // VAD keeps running for ducking/interrupt UI; the SERVER's VAD hears the
      // WebRTC mic track instead of JSON frames.
      webrtcAudio.start(sendMessageRef.current);
      webrtcAudio.setMicEnabled(true);
      setMicOn(true);
    } catch (error) {
      console.error('Failed to start VAD:', error);
      toaster.create({
        title: `${t('error.failedStartVAD')}: ${error}`,
        type: 'error',
        duration: 2000,
      });
    }
  }, [t]);

  /**
   * Stop microphone and VAD processing
   */
  const stopMic = useCallback(() => {
    console.log('Stopping VAD');
    if (vadRef.current) {
      vadRef.current.pause();
      vadRef.current.destroy();
      vadRef.current = null;
      console.log('VAD stopped and destroyed successfully');
      setPreviousTriggeredProbability(0);
    } else {
      console.log('VAD instance not found');
    }
    // Keep the peer alive (the AI may still be speaking on its track); just
    // mute the outgoing mic so the server VAD hears silence.
    webrtcAudio.setMicEnabled(false);
    setMicOn(false);
    isProcessingRef.current = false;
  }, []);

  /**
   * Set Auto stop mic state
   */
  const setAutoStopMic = useCallback((value: boolean) => {
    autoStopMicRef.current = value;
    setAutoStopMicState(value);
    forceUpdate();
  }, []);

  const setAutoStartMicOn = useCallback((value: boolean) => {
    autoStartMicRef.current = value;
    setAutoStartMicOnState(value);
    forceUpdate();
  }, []);

  const setAutoStartMicOnConvEnd = useCallback((value: boolean) => {
    autoStartMicOnConvEndRef.current = value;
    setAutoStartMicOnConvEndState(value);
    forceUpdate();
  }, []);

  // Memoized context value
  const contextValue = useMemo(
    () => ({
      autoStopMic: autoStopMicRef.current,
      isContinuousMic,
      micOn,
      setMicOn,
      setAutoStopMic,
      startMic,
      stopMic,
      previousTriggeredProbability: previousTriggeredProbabilityRef.current,
      setPreviousTriggeredProbability,
      settings,
      updateSettings,
      autoStartMicOn: autoStartMicRef.current,
      setAutoStartMicOn,
      autoStartMicOnConvEnd: autoStartMicOnConvEndRef.current,
      setAutoStartMicOnConvEnd,
    }),
    [
      micOn,
      startMic,
      stopMic,
      settings,
      updateSettings,
    ],
  );

  return (
    <VADContext.Provider value={contextValue}>
      {children}
    </VADContext.Provider>
  );
}

/**
 * Custom hook to use the VAD context
 * @throws {Error} If used outside of VADProvider
 */
export function useVAD() {
  const context = useContext(VADContext);

  if (!context) {
    throw new Error('useVAD must be used within a VADProvider');
  }

  return context;
}
