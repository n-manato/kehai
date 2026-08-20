import { SPACE } from "./space-config";

/**
 * Keeps the WebGL backing store matched to the display the window is actually
 * on.
 *
 * The viewer used to read `window.devicePixelRatio` once, into a `const`, and
 * never look again — so dragging the Chrome window from the desktop monitor onto
 * the PROTO Luma left the renderer rendering at the OLD monitor's ratio and
 * letting the compositor stretch the result. On a 4K panel running at 200%
 * Windows scaling that means drawing 1920x1080 and blowing it up to 3840x2160:
 * the "画質がすごく荒くなる" symptom, and nothing about the scene itself.
 *
 * Two things are needed to fix it, and BOTH matter:
 *
 *  1. Re-apply the ratio when it changes. A `ResizeObserver` catches most moves
 *     (the CSS size usually changes too) but not the case that matters most
 *     here — 1080p@100% to 4K@200% is the same 1920x1080 in CSS pixels, so no
 *     resize fires at all. That needs a devicePixelRatio watcher.
 *  2. Stop capping the RATIO. The old 1.5 cap was chosen to protect frame rate on
 *     hi-DPI laptops, but a ratio cap throws resolution away precisely where the
 *     CSS viewport is small and the device pixel ratio is large — measured on the
 *     unit: 720x1280 css at dpr 3, so 1.5 drew 1080x1920 for a 2160x3840 panel.
 *     The cost that matters is total pixels, so that is what is budgeted.
 */

/**
 * Effective backing-store scale for the current display and mode.
 *
 * Budgeted by TOTAL PIXELS, not by a fixed ratio cap. A ratio cap silently
 * throws away resolution exactly when it is most needed: on the PROTO the CSS
 * viewport is small and the device pixel ratio is large, so a cap of 2 renders a
 * fraction of the pixels the panel is about to display and the compositor
 * stretches the difference. What actually costs frame time is the number of
 * pixels pushed through GTAO, two blooms, DOF and SMAA — so budget that directly
 * and let the ratio be whatever reaches native within it.
 */
export function resolvePixelRatio(cssW?: number, cssH?: number): number {
  const dpr = window.devicePixelRatio || 1;
  const w = cssW ?? window.innerWidth ?? 1;
  const h = cssH ?? window.innerHeight ?? 1;
  const area = Math.max(w * h, 1);
  // sqrt because the ratio scales BOTH axes.
  const budgetRatio = Math.sqrt(SPACE.maxDrawPixels / area);

  // `superSample` deliberately allows going PAST the device ratio.
  //
  // This used to be a hard `min(dpr, …)` on the reasoning that "past native,
  // extra samples buy nothing but cost everything". That holds for texture
  // detail — it does not hold for a thin, high-contrast edge. Her outline is an
  // extruded hull about a pixel wide; at exactly 1.0 it lands differently on the
  // pixel grid every frame she moves, and the eye reads that as the line
  // crawling and flickering. MSAA cannot fix it (it samples geometry coverage,
  // not the sub-pixel motion of a line between frames) and SMAA cannot either
  // (it is a single-frame spatial filter on the finished image). Rendering above
  // native and letting the downscale average is the one thing that does.
  //
  // Still bounded by pixelRatioCap and the total-pixel budget, so it can never
  // outrun the frame time those exist to protect.
  const ss = Math.max(1, SPACE.superSample || 1);
  return Math.max(0.5, Math.min(dpr * ss, SPACE.pixelRatioCap, budgetRatio));
}

/**
 * Call `onChange` whenever `devicePixelRatio` changes.
 *
 * There is no `devicepixelratiochange` event; the standard trick is a media
 * query pinned to the CURRENT ratio, which stops matching the moment it moves.
 * It has to be re-armed at the new ratio each time, hence `once`.
 */
export function watchDevicePixelRatio(onChange: () => void): () => void {
  let disposed = false;
  let cleanup: (() => void) | null = null;

  const arm = () => {
    if (disposed) return;
    const dpr = window.devicePixelRatio || 1;
    const mql = window.matchMedia(`(resolution: ${dpr}dppx)`);
    const handler = () => {
      cleanup = null;
      arm();
      onChange();
    };
    mql.addEventListener("change", handler, { once: true });
    cleanup = () => mql.removeEventListener("change", handler);
  };

  arm();
  return () => {
    disposed = true;
    cleanup?.();
  };
}

/** Human-readable readout for diagnosing sharpness on the real hardware. */
export function describePixelRatio(container: HTMLElement, canvas: HTMLCanvasElement | null) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = container.clientWidth;
  const cssH = container.clientHeight;
  // Same inputs the render path uses, or the readout would report a ratio that
  // is not the one in effect.
  const applied = resolvePixelRatio(cssW, cssH);
  const bufW = canvas?.width ?? 0;
  const bufH = canvas?.height ?? 0;
  return {
    devicePixelRatio: dpr,
    maxDrawMPix: +(SPACE.maxDrawPixels / 1e6).toFixed(2),
    cap: SPACE.pixelRatioCap,
    appliedRatio: applied,
    cssSize: `${cssW}x${cssH}`,
    drawingBuffer: `${bufW}x${bufH}`,
    // What the panel can actually show for this element. If the drawing buffer
    // is smaller than this, the compositor is upscaling and it will look soft.
    nativePixels: `${Math.round(cssW * dpr)}x${Math.round(cssH * dpr)}`,
    viewport: `${window.innerWidth}x${window.innerHeight} css`,
    screen: `${window.screen.width}x${window.screen.height} css, ${Math.round(window.screen.width * dpr)}x${Math.round(window.screen.height * dpr)} native`,
    sharp: bufW >= Math.round(cssW * dpr) - 1,
    // If this is well under 1, the browser is handing the panel far fewer
    // pixels than it has — a display-mode or window-size problem, not a
    // renderer one, and no setting in here can recover it.
    bufferVsScreenWidth: +((bufW || 0) / Math.max(window.screen.width * dpr, 1)).toFixed(3),
  };
}
