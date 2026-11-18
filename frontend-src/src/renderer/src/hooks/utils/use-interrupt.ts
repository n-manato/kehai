import { useAiState } from '@/context/ai-state-context';
import { useWebSocket } from '@/context/websocket-context';
import { useChatHistory } from '@/context/chat-history-context';
import { audioTaskQueue } from '@/utils/task-queue';
import { useSubtitle } from '@/context/subtitle-context';
import { talkingMotionController } from '@/live2d/talking-motion';
import { mmdMotionPlayer } from '@/utils/mmd-motion-player';
import { vrmMotionPlayer } from '@/utils/vrm-motion-player';
import { smplBodyPlayer } from '@/utils/smpl-body-player';
import { mmdExpression } from '@/utils/mmd-expression';
import { vrmExpression } from '@/utils/vrm-expression';
import { vrmLipSync } from '@/utils/vrm-lipsync';
import { streamingControl } from '@/utils/streaming-audio-player';
import { useAudioTask } from './use-audio-task';

export const useInterrupt = () => {
  const { getAiState, setAiState } = useAiState();
  const { sendMessage } = useWebSocket();
  const { fullResponse, clearResponse } = useChatHistory();
  // const { currentModel } = useLive2DModel();
  const { subtitleText, setSubtitleText } = useSubtitle();
  const { stopCurrentAudioAndLipSync } = useAudioTask();

  const interrupt = (sendSignal = true) => {
    if (getAiState() !== 'thinking-speaking') return;
    console.log('Interrupting conversation chain');

    streamingControl.stopNow?.(); // the streamed reply (see streamingControl)
    stopCurrentAudioAndLipSync();
    talkingMotionController.onSpeechEnd(); // fade talking sway back to idle
    mmdMotionPlayer.stop(); // fade the MMD HY-Motion clip back to idle
    vrmMotionPlayer.cancelGesture(); // stop any pending VRM gesture poll
    vrmMotionPlayer.interrupt(); // HOLD the pose (clearing it left 1.8s with nothing on the body)
    vrmMotionPlayer.stopTalking(); // stop the volume-driven talking body
    smplBodyPlayer.cancelGesture(); // stop any pending SMPL body gesture poll
    mmdExpression.relax(); // ease the face back to its resting expression
    vrmExpression.relax();
    vrmLipSync.onAudioEnd(); // close the mouth (stop lip-sync)

    audioTaskQueue.clearQueue();

    setAiState('interrupted');
    window.dispatchEvent(new CustomEvent('kehai-turn', { detail: 'end' }));

    if (sendSignal) {
      sendMessage({
        type: 'interrupt-signal',
        text: fullResponse,
      });
    }

    clearResponse();

    if (subtitleText === 'Thinking...') {
      setSubtitleText('');
    }
    console.log('Interrupted!');
  };

  return { interrupt };
};
