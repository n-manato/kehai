/* eslint-disable func-names */
/* eslint-disable no-underscore-dangle */
/* eslint-disable @typescript-eslint/ban-ts-comment */
import { useRef, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useAiState } from '@/context/ai-state-context';
import { useSubtitle } from '@/context/subtitle-context';
import { useChatHistory } from '@/context/chat-history-context';
import { audioTaskQueue } from '@/utils/task-queue';
import { streamingPlaybackState } from '@/utils/streaming-audio-player';
import { audioManager } from '@/utils/audio-manager';
import { mmdLipSync } from '@/utils/mmd-lipsync';
import { mmdMotionPlayer, mmdEmotionFromGroup, mmdEmotionFromKeyword } from '@/utils/mmd-motion-player';
import { vrmMotionPlayer } from '@/utils/vrm-motion-player';
import { mmdExpression } from '@/utils/mmd-expression';
import { vrmExpression } from '@/utils/vrm-expression';
import { vrmLipSync } from '@/utils/vrm-lipsync';
import { toaster } from '@/components/ui/toaster';
import { useWebSocket } from '@/context/websocket-context';
import { DisplayText } from '@/services/websocket-service';
import { useLive2DExpression } from '@/hooks/canvas/use-live2d-expression';
import { talkingMotionController } from '@/live2d/talking-motion';
import * as LAppDefine from '../../../WebSDK/src/lappdefine';

// Simple type alias for Live2D model
type Live2DModel = any;

interface AudioTaskOptions {
  audioBase64: string
  volumes: number[]
  sliceLength: number
  displayText?: DisplayText | null
  expressions?: string[] | number[] | null
  motions?: string[] | null
  emotions?: string[] | null
  motion?: { id: string; key: string; duration: number } | null
  speaker_uid?: string
  forwarded?: boolean
}

// Neutral/filler motion groups that should NOT trigger a full Cubism motion per
// sentence (that cross-fades the body/neck to a new base pose and looks like an
// abrupt reset between sentences). The smooth additive Talking Motion layer
// handles per-sentence liveliness for these; only real emotion groups play a
// Cubism motion.
const FILLER_MOTION_GROUPS = new Set(['Idle', 'Idle_nod', 'Idle_sigh', 'Look', 'Option']);

// Recently-played clip indices per motion group, most-recent last. Used to
// pick randomly while avoiding the last few clips, so small groups (e.g.
// Joy_smile with 2 clips) don't visibly alternate A-B-A-B and large groups
// feel clearly varied. The backend names a motion GROUP (e.g. "Joy_big" /
// "Idle"); the clip within it is chosen here. Random from the very first call.
const recentMotionIndices: Record<string, number[]> = {};
const AVOID_RECENT = 2; // try to avoid replaying the last N clips of a group

// VRM gesture buffering (full sync). The co-speech clip is generated ~2.2s after its
// audio exists on a single-slot gateway, so it can't always be ready in time. We wait
// for it before starting the audio, but the wait budget differs by position:
//  - reply OPENING: wait longer — the pause reads as the AI "thinking" before it speaks.
//  - MID-reply: wait only briefly — the previous line's audio already ended, so a long
//    wait would be an audible SILENCE GAP. If the gateway is behind (rapid short lines),
//    fall back to catch-up (audio-locked) rather than gap the speech.
// A line is the opening when the previous VRM line ended a while ago (LLM think time
// between replies) — mid-reply lines play back-to-back so their gap is tiny.
let vrmPrevSentenceEndMs = 0;
const VRM_BUF_OPENING_MS = 3000;
const VRM_BUF_MIDREPLY_MS = 500;
const VRM_REPLY_GAP_MS = 1500; // gap since last line that marks a new reply's opening

/**
 * Play a random motion from the named group, avoiding (where the group is big
 * enough) the last few clips played from it. Returns true if a motion started.
 */
const startGroupMotion = (model: Live2DModel, group: string, priority: number): boolean => {
  const count = model._modelSetting?.getMotionCount?.(group) ?? 0;
  if (count <= 0) {
    console.warn(`Motion group "${group}" not found or empty`);
    return false;
  }
  const recent = recentMotionIndices[group] ?? [];
  // Never avoid so many that no clip is left to pick.
  const avoid = new Set(recent.slice(-Math.min(recent.length, count - 1)));
  let idx = Math.floor(Math.random() * count);
  for (let tries = 0; avoid.has(idx) && tries < 20; tries += 1) {
    idx = Math.floor(Math.random() * count);
  }
  recent.push(idx);
  while (recent.length > AVOID_RECENT) recent.shift();
  recentMotionIndices[group] = recent;
  console.log(`Starting motion '${group}' #${idx}/${count}`);
  model.startMotion(group, idx, priority);
  return true;
};

/**
 * Custom hook for handling audio playback tasks with Live2D lip sync
 */
export const useAudioTask = () => {
  const { t } = useTranslation();
  const { aiState, backendSynthComplete, setBackendSynthComplete } = useAiState();
  const { setSubtitleText } = useSubtitle();
  const { appendResponse, appendAIMessage } = useChatHistory();
  const { sendMessage } = useWebSocket();
  const { setExpression } = useLive2DExpression();

  // State refs to avoid stale closures
  const stateRef = useRef({
    aiState,
    setSubtitleText,
    appendResponse,
    appendAIMessage,
  });

  // Note: currentAudioRef and currentModelRef are now managed by the global audioManager

  stateRef.current = {
    aiState,
    setSubtitleText,
    appendResponse,
    appendAIMessage,
  };

  /**
   * Stop current audio playback and lip sync (delegates to global audioManager)
   */
  const stopCurrentAudioAndLipSync = useCallback(() => {
    audioManager.stopCurrentAudioAndLipSync();
  }, []);

  // Let the Talking Motion System's keyword reactions optionally trigger an
  // EXISTING expression / motion group (falls back gracefully if absent).
  useEffect(() => {
    talkingMotionController.registerSinks({
      setExpression: (name: string) => {
        const adapter = (window as any).getLAppAdapter?.();
        if (adapter) setExpression(name, adapter, `keyword reaction expression: ${name}`);
      },
      playMotion: (group: string) => {
        const model = (window as any).getLive2DManager?.()?.getModel?.(0);
        if (model && LAppDefine?.PriorityNormal) {
          startGroupMotion(model, group, LAppDefine.PriorityNormal);
        }
      },
    });
  }, [setExpression]);

  // Attentive "listening" motion while the user is speaking (aiState === 'listening').
  useEffect(() => {
    talkingMotionController.setListening(aiState === 'listening');
  }, [aiState]);

  /**
   * Handle audio playback with Live2D lip sync
   */
  const handleAudioPlayback = (options: AudioTaskOptions): Promise<void> => new Promise((resolve) => {
    const {
      aiState: currentAiState,
      setSubtitleText: updateSubtitle,
      appendResponse: appendText,
      appendAIMessage: appendAI,
    } = stateRef.current;

    // Skip if already interrupted
    if (currentAiState === 'interrupted') {
      console.warn('Audio playback blocked by interruption state.');
      resolve();
      return;
    }

    const {
      audioBase64, displayText, expressions, motions, emotions, motion, volumes, sliceLength, forwarded,
    } = options;

    // Update display text
    if (displayText) {
      appendText(displayText.text);
      appendAI(displayText.text, displayText.name, displayText.avatar);
      if (audioBase64) {
        updateSubtitle(displayText.text);
      }
      if (!forwarded) {
        sendMessage({
          type: "audio-play-start",
          display_text: displayText,
          forwarded: true,
        });
      }
    }

    try {
      // Process audio if available
      if (audioBase64) {
        const audioDataUrl = `data:audio/wav;base64,${audioBase64}`;

        // Raw SMPL-X BODY viewer active: play audio + drive the ground-truth SMPL
        // joints figure (zero retarget). Checked first — window.__smpl only set
        // while the SMPL body viewer is mounted.
        if ((window as any).__smpl) {
          const audio = new Audio(audioDataUrl);
          audioManager.setCurrentAudio(audio, null);
          let done = false;
          const finish = () => { audioManager.clearCurrentAudio(audio); if (!done) { done = true; resolve(); } };
          audio.addEventListener('canplaythrough', () => {
            if (stateRef.current.aiState === 'interrupted' || !audioManager.hasCurrentAudio()) { finish(); return; }
            audio.play().catch((err) => { console.error('SMPL audio play error:', err); finish(); });
            // the SMPL body viewer loops a baked co-speech clip; nothing to fetch.
          });
          audio.addEventListener('ended', finish);
          audio.addEventListener('error', finish);
          audio.load();
          return;
        }

        // VRM renderer active (a .vrm character is selected): play the audio and
        // drive the VRM co-speech gesture (vrmMotionPlayer retargets SMPL joints).
        // Checked FIRST — window.__vrm is only set while the VRM viewer is mounted.
        const vrm = (window as any).__vrm;
        if (vrm) {
          const audio = new Audio(audioDataUrl);
          audioManager.setCurrentAudio(audio, null);
          let vrmDone = false;
          const finishVRM = () => {
            audioManager.clearCurrentAudio(audio);
            vrmLipSync.onAudioEnd();
            vrmExpression.onAudioEnd();
            vrmMotionPlayer.stopTalking();
            // Do NOT stop the gesture here: cutting each clip at its sentence end is what
            // made the motion choppy ("一つ一つ途切れる"). The clip now keeps moving (loops
            // + crossfades the next line's clip in), so the whole reply is ONE continuous
            // talking motion. It settles to idle only at the reply end (conversation-end
            // control msg in websocket-handler) or on interrupt.
            vrmPrevSentenceEndMs = performance.now(); // mark line end (reply-opening detection)
            if (!vrmDone) { vrmDone = true; resolve(); }
          };
          audio.addEventListener('canplaythrough', async () => {
            if (stateRef.current.aiState === 'interrupted' || !audioManager.hasCurrentAudio()) {
              finishVRM();
              return;
            }
            // FULL SYNC via buffering: wait for THIS line's gesture to be ready BEFORE
            // starting the audio, so it plays synced from frame 0. The wait budget is
            // large for the reply's opening (reads as "thinking") but small mid-reply
            // (a long wait there = an audible silence gap); on miss we play and the
            // gesture catches up (audio-locked) instead of stalling the speech.
            const isOpening = vrmPrevSentenceEndMs === 0
              || (performance.now() - vrmPrevSentenceEndMs) > VRM_REPLY_GAP_MS;
            const bufMs = isOpening ? VRM_BUF_OPENING_MS : VRM_BUF_MIDREPLY_MS;
            // Prefetched-while-the-prior-line-played clip is ready NOW -> use it (synced,
            // no wait, no loop). Else fall back to the short buffered fetch, then catch-up.
            let clipData: any = motion?.id ? vrmMotionPlayer.takePrefetched(motion.id) : null;
            if (!clipData && motion?.id) {
              clipData = await vrmMotionPlayer.fetchClip(`/motion/clip-smpl/${motion.id}`, bufMs);
            }
            if (stateRef.current.aiState === 'interrupted' || !audioManager.hasCurrentAudio()) {
              finishVRM();
              return;
            }
            audio.play().catch((err) => { console.error('VRM audio play error:', err); finishVRM(); });
            if (clipData) {
              vrmMotionPlayer.setClip(clipData, audio); // synced from frame 0
            } else if (motion?.id) {
              // Buffer timed out — play now and let the gesture catch up to audio time.
              vrmMotionPlayer.playGesture(`/motion/clip-smpl/${motion.id}`, { audio });
            }
            // Procedural talking = fallback only (when no clip yet). Harmless while a
            // synced clip plays (updateIdle is skipped whenever a clip is active).
            vrmMotionPlayer.startTalking(audio, volumes || [], sliceLength || 20);
            // Facial expression: same per-sentence emotion keyword as the MMD viewer
            // (backend actions.emotions) drives the eyes; prosody pulses the face.
            const emoKeyword = emotions && emotions.length > 0 ? emotions[0] : null;
            vrmLipSync.onAudioStart(audio, volumes || [], sliceLength || 20);
            vrmExpression.onAudioStart(audio, volumes || [], sliceLength || 20);
            vrmExpression.playEmotion(emoKeyword);
          });
          audio.addEventListener('ended', finishVRM);
          audio.addEventListener('error', finishVRM);
          audio.load();
          return;
        }

        // MMD renderer active (a .pmx character is selected): play the audio and
        // drive the 3D model's mouth morph. Checked FIRST because
        // window.getLive2DManager lingers after a Live2D character was shown.
        const mmd = (window as any).__mmd;
        const live2dManager = (window as any).getLive2DManager?.();
        if (mmd?.mesh) {
          const audio = new Audio(audioDataUrl);
          audioManager.setCurrentAudio(audio, null);
          let mmdDone = false;
          const finishMMD = () => {
            audioManager.clearCurrentAudio(audio);
            mmdLipSync.onAudioEnd();
            mmdExpression.onAudioEnd();
            if (!mmdDone) {
              mmdDone = true;
              resolve();
            }
          };
          audio.addEventListener('canplaythrough', () => {
            if (stateRef.current.aiState === 'interrupted' || !audioManager.hasCurrentAudio()) {
              finishMMD();
              return;
            }
            audio.play().catch((err) => {
              console.error('MMD audio play error:', err);
              finishMMD();
            });
            mmdLipSync.onAudioStart(audio, volumes || [], sliceLength || 20);
            // Prosody: feed the same volume array to the expression layer so the
            // face pulses with speech emphasis (C).
            mmdExpression.onAudioStart(audio, volumes || [], sliceLength || 20);
            // The per-sentence emotion KEYWORD (backend actions.emotions, e.g.
            // "joy_grin") is the reliable signal — it's present on every sentence,
            // unlike `motions` which is sparse (only on emotion change). Fall back
            // to the motion group for older backends.
            const emoKeyword =
              emotions && emotions.length > 0 ? emotions[0] : null;
            const bodyEmotion = emoKeyword
              ? mmdEmotionFromKeyword(emoKeyword)
              : mmdEmotionFromGroup(
                  motions && motions.length > 0 ? motions[0] : null,
                );
            // Body motion. Two clip kinds carry a per-sentence id:
            //  - "gesture": audio-driven co-speech (GestureLSM). It's generated
            //    from the finished audio and lands ~2-3s in, so we start the
            //    correct-emotion pool loop NOW (instant, always-moving body) and
            //    POLL for the gesture, crossfading into it as a natural loop when
            //    ready. The pool is the fallback if the gesture never arrives.
            //  - HY-Motion bespoke: a one-shot clip sized to this sentence; play it
            //    when ready, else fall back to the emotion pool (unchanged).
            // With no clip id, just loop the emotion/gesture pool.
            if (motion?.id && motion.key === 'gesture') {
              // Body moves WITH speech from t=0. co-speech is generated from the
              // finished audio (lands ~2-3s in for the 1st sentence), so: (1) start
              // an INSTANT warm talk loop (generic GestureLSM clip from the pool) so
              // the body is never idle at speech onset; (2) crossfade to THIS
              // sentence's REAL gesture, SEEKED to the audio's current position so it
              // lands in step (frame-0 aligned if pre-fetched during the prior
              // sentence, else mid-sentence sync-in). guard drops it if the sentence
              // already ended (don't smear a stale gesture onto the next one).
              // Warm loop ONLY if the real gesture isn't already pre-fetched: when it
              // is (2nd+ sentences) we crossfade straight from the prior gesture, so a
              // slower warm-pool fetch can't clobber the already-aligned gesture.
              if (!mmdMotionPlayer.hasReadyGesture(motion.id)) {
                mmdMotionPlayer.playEmotion(bodyEmotion);
              }
              mmdMotionPlayer.playGesture(
                `${mmdMotionPlayer.getBaseUrl()}/motion/clip/${motion.id}`,
                {
                  id: motion.id,
                  startAtSec: () =>
                    Number.isFinite(audio.currentTime) ? audio.currentTime : 0,
                  guard: () => !audio.ended,
                },
              );
            } else if (motion?.id) {
              const targetDur =
                Number.isFinite(audio.duration) && audio.duration > 0
                  ? audio.duration
                  : motion.duration || 0;
              const poolKey = motion.key || bodyEmotion;
              mmdMotionPlayer.playClip(
                `${mmdMotionPlayer.getBaseUrl()}/motion/clip/${motion.id}`,
                {
                  durationSec: targetDur,
                  onMiss: () => mmdMotionPlayer.playEmotion(poolKey),
                },
              );
            } else {
              mmdMotionPlayer.playEmotion(motion?.key || bodyEmotion);
            }
            // Facial expression: drive the FACE off the fine-grained keyword (B) —
            // joy_grin / anger_pout / sadness_tearful map to distinct presets — so
            // it reacts to exactly what she says, richer than the body's 7 pools.
            mmdExpression.playEmotion(emoKeyword || bodyEmotion);
          });
          audio.addEventListener('ended', finishMMD);
          audio.addEventListener('error', finishMMD);
          audio.load();
          return;
        }

        if (!live2dManager) {
          console.error('Live2D manager not found');
          resolve();
          return;
        }

        const model = live2dManager.getModel(0);
        if (!model) {
          console.error('Live2D model not found at index 0');
          resolve();
          return;
        }
        console.log('Found model for audio playback');

        if (!model._wavFileHandler) {
          console.warn('Model does not have _wavFileHandler for lip sync');
        } else {
          console.log('Model has _wavFileHandler available');
        }

        // Set expression if available
        const lappAdapter = (window as any).getLAppAdapter?.();
        if (lappAdapter && expressions?.[0] !== undefined) {
          setExpression(
            expressions[0],
            lappAdapter,
            `Set expression to: ${expressions[0]}`,
          );
        }

        // Start the body motion the backend asked for, but ONLY for genuine
        // emotion changes (Joy_big, Sadness_worry, ...). Filler / neutral groups
        // (Idle, Idle_nod, Look, ...) are intentionally NOT started here:
        // re-triggering a full Cubism motion every sentence cross-fades the body
        // and neck to a new base pose, which reads as an abrupt "reset" between
        // sentences. The smooth additive Talking Motion layer (sway, nods,
        // tilts, gaze) provides the per-sentence liveliness instead; the SDK
        // idle loop still runs underneath. This keeps transitions continuous.
        if (LAppDefine && LAppDefine.PriorityNormal) {
          const group = motions && motions.length > 0 ? motions[0] : null;
          if (group && !FILLER_MOTION_GROUPS.has(group)) {
            startGroupMotion(model, group, LAppDefine.PriorityNormal);
          }
        } else {
          console.warn("LAppDefine.PriorityNormal not found - cannot start motion");
        }

        // Setup audio element
        const audio = new Audio(audioDataUrl);
        
        // Register with global audio manager IMMEDIATELY after creating audio
        audioManager.setCurrentAudio(audio, model);
        let isFinished = false;

        const cleanup = () => {
          audioManager.clearCurrentAudio(audio);
          if (!isFinished) {
            isFinished = true;
            resolve();
          }
        };

        // Enhance lip sync sensitivity
        const lipSyncScale = 2.0;

        audio.addEventListener('canplaythrough', () => {
          // Check for interruption before playback
          if (stateRef.current.aiState === 'interrupted' || !audioManager.hasCurrentAudio()) {
            console.warn('Audio playback cancelled due to interruption or audio was stopped');
            cleanup();
            return;
          }

          console.log('Starting audio playback with lip sync');
          audio.play().catch((err) => {
            console.error("Audio play error:", err);
            cleanup();
          });

          // Drive the Talking Motion System for this sentence: enable the
          // subtle speaking sway and schedule rhythm beats / keyword reactions.
          // audio.duration is known by canplaythrough; emotion is derived from
          // the backend's motion group for this sentence.
          talkingMotionController.onSentenceStart(
            displayText?.text ?? '',
            Number.isFinite(audio.duration) ? audio.duration : 0,
            motions && motions.length > 0 ? motions[0] : null,
            volumes && volumes.length > 0 ? volumes : null,
            sliceLength,
            typeof expressions?.[0] === 'string' ? (expressions[0] as string) : null,
          );

          // Setup lip sync
          if (model._wavFileHandler) {
            if (!model._wavFileHandler._initialized) {
              console.log('Applying enhanced lip sync');
              model._wavFileHandler._initialized = true;

              const originalUpdate = model._wavFileHandler.update.bind(model._wavFileHandler);
              model._wavFileHandler.update = function (deltaTimeSeconds: number) {
                const result = originalUpdate(deltaTimeSeconds);
                // @ts-ignore
                this._lastRms = Math.min(2.0, this._lastRms * lipSyncScale);
                return result;
              };
            }

            if (audioManager.hasCurrentAudio()) {
              model._wavFileHandler.start(audioDataUrl);
            } else {
              console.warn('WavFileHandler start skipped - audio was stopped');
            }
          }
        });

        audio.addEventListener('ended', () => {
          console.log("Audio playback completed");
          cleanup();
        });

        audio.addEventListener('error', (error) => {
          console.error("Audio playback error:", error);
          cleanup();
        });

        audio.load();
      } else {
        resolve();
      }
    } catch (error) {
      console.error('Audio playback setup error:', error);
      toaster.create({
        title: `${t('error.audioPlayback')}: ${error}`,
        type: "error",
        duration: 2000,
      });
      resolve();
    }
  });

  // Handle backend synthesis completion
  useEffect(() => {
    let isMounted = true;

    const handleComplete = async () => {
      await audioTaskQueue.waitForCompletion();
      // Streamed turn: audio flows through the StreamingAudioPlayer, so this queue
      // is empty and would report completion immediately (~2.5x too early, before
      // the streamed audio finishes). The streaming path reports playback-complete
      // itself when its player drains — skip here.
      if (streamingPlaybackState.active) return;
      if (isMounted && backendSynthComplete) {
        stopCurrentAudioAndLipSync();
        sendMessage({ type: "frontend-playback-complete" });
        setBackendSynthComplete(false);
      }
    };

    handleComplete();

    return () => {
      isMounted = false;
    };
  }, [backendSynthComplete, sendMessage, setBackendSynthComplete, stopCurrentAudioAndLipSync]);

  /**
   * Add a new audio task to the queue
   */
  const addAudioTask = async (options: AudioTaskOptions) => {
    const { aiState: currentState } = stateRef.current;

    if (currentState === 'interrupted') {
      console.log('Skipping audio task due to interrupted state');
      return;
    }

    console.log(`Adding audio task ${options.displayText?.text} to queue`);
    // Estimate this clip's playback duration so the queue can buffer ahead by
    // SECONDS (not clip count): volumes holds one RMS sample per `sliceLength`
    // ms chunk, so duration ≈ volumes.length * sliceLength / 1000.
    const estDurationSec = options.volumes?.length && options.sliceLength
      ? (options.volumes.length * options.sliceLength) / 1000
      : 0;
    audioTaskQueue.addTask(() => handleAudioPlayback(options), estDurationSec);
  };

  return {
    addAudioTask,
    appendResponse,
    stopCurrentAudioAndLipSync,
  };
};
