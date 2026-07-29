import { SPACE } from "./space-config";

/**
 * Fade the overlay chrome out when nobody is touching the screen, and bring it
 * back on contact — the way a video player does.
 *
 * On a permanently-installed unit the controls are the only thing on screen that
 * is obviously a *user interface*. A connection badge and a row of buttons
 * floating over her are a constant reminder that this is an application, which
 * is the one thing the whole spatial layer is trying to make you forget. They
 * still need to be reachable, so: hide on idle, reveal on contact.
 *
 * Driven by a class on <body> plus one stylesheet rule rather than by touching
 * each element's style. The badge is a React component and the buttons are
 * imperative DOM created by the viewer; a class is the only thing both can opt
 * into without one of them having to know about the other. Anything added later
 * joins in by carrying the class.
 */

const CLASS = "auto-hide-ui";
const IDLE_CLASS = "ui-idle";
const STYLE_ID = "auto-hide-ui-style";

let timer = 0;
let installed = false;
let listeners: (() => void)[] = [];

function ensureStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
.${CLASS} { transition: opacity .45s ease; }
body.${IDLE_CLASS} .${CLASS} { opacity: 0; pointer-events: none; }
`;
  document.head.appendChild(style);
}

function goIdle() {
  document.body.classList.add(IDLE_CLASS);
}

function wake() {
  document.body.classList.remove(IDLE_CLASS);
  window.clearTimeout(timer);
  if (!SPACE.uiAutoHide) return;
  timer = window.setTimeout(goIdle, Math.max(SPACE.uiIdleMs, 500));
}

/** Start hiding the chrome on idle. Safe to call more than once. */
export function installAutoHide(): () => void {
  if (installed) return () => {};
  installed = true;
  ensureStyle();

  // pointermove covers mouse and stylus; touchstart is what the unit actually
  // gets. keydown so a keyboard user is not left with invisible controls.
  const events = ["pointerdown", "pointermove", "touchstart", "keydown", "wheel"] as const;
  events.forEach((name) => {
    const fn = () => wake();
    window.addEventListener(name, fn, { passive: true });
    listeners.push(() => window.removeEventListener(name, fn));
  });

  wake(); // visible to begin with, then fade

  return () => {
    installed = false;
    window.clearTimeout(timer);
    listeners.forEach((off) => off());
    listeners = [];
    document.body.classList.remove(IDLE_CLASS);
    document.getElementById(STYLE_ID)?.remove();
  };
}

/** Mark an element as chrome that should fade with the rest. */
export function autoHide<T extends HTMLElement>(el: T): T {
  el.classList.add(CLASS);
  return el;
}

/** Re-read the settings (called after `__space({ uiIdleMs: … })`). */
export function refreshAutoHide() {
  if (!SPACE.uiAutoHide) {
    window.clearTimeout(timer);
    document.body.classList.remove(IDLE_CLASS);
    return;
  }
  wake();
}

export const AUTO_HIDE_CLASS = CLASS;
