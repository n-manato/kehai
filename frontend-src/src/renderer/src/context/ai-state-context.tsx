import {
  createContext,
  useState,
  ReactNode,
  useContext,
  useCallback,
  useMemo,
  useRef,
  useEffect,
} from 'react';

/**
 * Enum for all possible AI states
 * @description Defines all possible states that the AI can be in
 */
// eslint-disable-next-line no-shadow
export const enum AiStateEnum {
  /**
   * - Can be triggered to speak proactively
   * - Ready to receive user input
   */
  IDLE = 'idle',

  /**
   * - Can be interrupted by user
   */
  THINKING_SPEAKING = 'thinking-speaking',

  /**
   * - Triggered by sending text / detecting speech / clicking interrupt button / creating new chat history / switching character
   */
  INTERRUPTED = 'interrupted',

  /**
   * - Shows during initial load / character switching
   */
  LOADING = 'loading',

  /**
   * - Speech is detected
   */
  LISTENING = 'listening',

  /**
   * - Set when user is typing
   * - Auto returns to IDLE after 2s
   */
  WAITING = 'waiting',
}

export type AiState = `${AiStateEnum}`;

/**
 * Type definition for the AI state context
 */
interface AiStateContextType {
  aiState: AiState;
  /** The latest state, including a set made earlier in the same tick. */
  getAiState: () => AiState;
  setAiState: {
    (state: AiState): void;
    (updater: (currentState: AiState) => AiState): void;
  };
  backendSynthComplete: boolean;
  setBackendSynthComplete: (complete: boolean) => void;
  isIdle: boolean;
  isThinkingSpeaking: boolean;
  isInterrupted: boolean;
  isLoading: boolean;
  isListening: boolean;
  isWaiting: boolean;
  resetState: () => void;
}

/**
 * Initial context value
 */
const initialState: AiState = AiStateEnum.LOADING;

/**
 * Create the AI state context
 */
export const AiStateContext = createContext<AiStateContextType | null>(null);

/**
 * AI State Provider Component
 */
export function AiStateProvider({ children }: { children: ReactNode }) {
  const [aiState, setAiStateInternal] = useState<AiState>(initialState);
  const [backendSynthComplete, setBackendSynthComplete] = useState(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  // The state as last SET, not as last rendered. Updaters used to run against
  // the `aiState` of the render that created this setter, so a callback holding
  // an old setter (the stream player's onEnded is built once per turn) wrote its
  // decision over a newer state — e.g. 'idle' over the 'listening' of a user who
  // had just started talking. Two sets in one tick (interrupt -> 'interrupted'
  // -> 'listening') also need the second to see the first.
  const stateRef = useRef<AiState>(initialState);
  const commit = (s: AiState) => {
    stateRef.current = s;
    setAiStateInternal(s);
  };
  const getAiState = useCallback(() => stateRef.current, []);

  const setAiState = useCallback((newState: AiState | ((currentState: AiState) => AiState)) => {
    const current = stateRef.current;
    const nextState = typeof newState === 'function'
      ? (newState as (currentState: AiState) => AiState)(current)
      : newState;

    if (nextState === AiStateEnum.WAITING) {
      if (current !== AiStateEnum.THINKING_SPEAKING) {
        commit(nextState);

        if (timerRef.current) {
          clearTimeout(timerRef.current);
        }

        timerRef.current = setTimeout(() => {
          commit(AiStateEnum.IDLE);
          timerRef.current = null;
        }, 2000);
      }
    } else {
      commit(nextState);
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    }
  }, [aiState]);

  // Memoized state checks
  const stateChecks = useMemo(
    () => ({
      isIdle: aiState === AiStateEnum.IDLE,
      isThinkingSpeaking: aiState === AiStateEnum.THINKING_SPEAKING,
      isInterrupted: aiState === AiStateEnum.INTERRUPTED,
      isLoading: aiState === AiStateEnum.LOADING,
      isListening: aiState === AiStateEnum.LISTENING,
      isWaiting: aiState === AiStateEnum.WAITING,
    }),
    [aiState],
  );

  // Reset state handler
  const resetState = useCallback(() => {
    setAiState(AiStateEnum.IDLE);
  }, [setAiState]);

  useEffect(() => () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
  }, []);

  // Memoized context value
  const contextValue = useMemo(
    () => ({
      aiState,
      getAiState,
      setAiState,
      backendSynthComplete,
      setBackendSynthComplete,
      ...stateChecks,
      resetState,
    }),
    [aiState, getAiState, setAiState, backendSynthComplete, stateChecks, resetState],
  );

  return (
    <AiStateContext.Provider value={contextValue}>
      {children}
    </AiStateContext.Provider>
  );
}

/**
 * Custom hook to use the AI state context
 * @throws {Error} If used outside of AiStateProvider
 */
export function useAiState() {
  const context = useContext(AiStateContext);

  if (!context) {
    throw new Error('useAiState must be used within a AiStateProvider');
  }

  return context;
}
