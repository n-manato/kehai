import {
  createContext, useState, useMemo, useContext, memo, useCallback,
} from 'react';

/**
 * Subtitle context state interface
 * @interface SubtitleState
 */
interface SubtitleState {
  /** Current subtitle text */
  subtitleText: string

  /** Set subtitle text */
  setSubtitleText: (text: string) => void

  /** Whether to show subtitle */
  showSubtitle: boolean

  /** Toggle subtitle visibility */
  setShowSubtitle: (show: boolean) => void
}

/**
 * Default values and constants
 */
const DEFAULT_SUBTITLE = {
  text: "Hi, I'm some random AI VTuber. Who the hell are ya? "
        + 'Ahh, you must be amazed by my awesomeness, right? right?',
};

/**
 * Create the subtitle context
 */
export const SubtitleContext = createContext<SubtitleState | null>(null);

/**
 * Subtitle Provider Component
 * Manages the subtitle display text state
 *
 * @param {Object} props - Provider props
 * @param {React.ReactNode} props.children - Child components
 */
export const SubtitleProvider = memo(({ children }: { children: React.ReactNode }) => {
  // State management
  const [subtitleText, setSubtitleText] = useState<string>(DEFAULT_SUBTITLE.text);
  // Persisted: on a permanently-installed unit the subtitle setting has to
  // survive a reload, or it silently comes back on every time the page is
  // refreshed and has to be switched off by hand again.
  const [showSubtitle, setShowSubtitleState] = useState<boolean>(() => {
    try {
      // Off unless explicitly switched on: on a permanent installation the
      // subtitle covers her legs and announces that this is an application.
      return localStorage.getItem("showSubtitle") === "1";
    } catch {
      return false;
    }
  });
  const setShowSubtitle = useCallback((v: boolean) => {
    setShowSubtitleState(v);
    try {
      localStorage.setItem("showSubtitle", v ? "1" : "0");
    } catch {
      /* private mode / quota */
    }
  }, []);

  // Memoized context value
  const contextValue = useMemo(
    () => ({
      subtitleText,
      setSubtitleText,
      showSubtitle,
      setShowSubtitle,
    }),
    [subtitleText, showSubtitle],
  );

  return (
    <SubtitleContext.Provider value={contextValue}>
      {children}
    </SubtitleContext.Provider>
  );
});

/**
 * Custom hook to use the subtitle context
 * @throws {Error} If used outside of SubtitleProvider
 */
export function useSubtitle() {
  const context = useContext(SubtitleContext);

  if (!context) {
    throw new Error('useSubtitle must be used within a SubtitleProvider');
  }

  return context;
}
