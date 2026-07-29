import { SPACE } from "./space-config";
import { headTracker } from "./head-tracker";
import { viewerTransform } from "./viewer-transform";
import { impliedPanelHeightM } from "./off-axis";
import { describePixelRatio } from "./pixel-ratio";

/**
 * On-screen status for setting the spatial layer up on the real unit.
 *
 * Everything here is already reachable from the console, but the console is the
 * wrong tool at a 2.2 m kiosk: bringing up head tracking means standing in front
 * of the camera, stepping left and right, and watching whether the numbers
 * follow — which you cannot do while typing. So the same state is put on the
 * glass, updated live.
 *
 * Deliberately throttled and text-only: it must not become a thing that itself
 * costs frame time while you are judging frame time.
 */

let hud: HTMLDivElement | null = null;
let timer = 0;
let rafId = 0;
let frames = 0;
let fpsWindow = 0;
let renderFps = 0;

/**
 * Which JS bundle is actually running. Every fix here ships as a new hashed
 * file, and a missed hard-reload looks exactly like a fix that did not work —
 * this makes the difference visible without opening the network tab.
 */
function bundleId(): string {
  const el = document.querySelector('script[src*="assets/main-"]') as HTMLScriptElement | null;
  return el?.src.match(/main-([^.]+)\.js/)?.[1] ?? "unknown";
}

const countFrame = () => {
  rafId = requestAnimationFrame(countFrame);
  frames += 1;
  const now = performance.now();
  if (now - fpsWindow > 1000) {
    renderFps = (frames * 1000) / (now - fpsWindow);
    frames = 0;
    fpsWindow = now;
  }
};

export function hideSpaceHud() {
  if (timer) clearInterval(timer);
  timer = 0;
  cancelAnimationFrame(rafId);
  rafId = 0;
  hud?.remove();
  hud = null;
}

export function showSpaceHud(container: HTMLElement, canvas: HTMLCanvasElement | null): string {
  hideSpaceHud();

  const el = document.createElement("div");
  el.style.cssText = [
    "position:absolute",
    "left:8px",
    "top:64px", // clear of the connection badge in the corner
    "z-index:9998",
    "pointer-events:none",
    "background:rgba(6,4,16,0.82)",
    "border:1px solid rgba(53,215,255,0.45)",
    "border-radius:6px",
    "padding:8px 10px",
    "font:11px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace",
    "color:#dff1ff",
    "white-space:pre",
  ].join(";");
  container.appendChild(el);
  hud = el;

  fpsWindow = performance.now();
  countFrame();

  const paint = () => {
    if (!hud) return;
    const t = headTracker.status();
    const p = viewerTransform.get();
    const px = describePixelRatio(container, canvas);
    const implied = impliedPanelHeightM();
    const agree = implied / Math.max(SPACE.panelHeightM, 1e-6);
    // Off-axis only takes the camera when a head is actually tracked, so say
    // which of the two states we are in rather than just whether it is enabled.
    const offAxisLive = SPACE.offAxis && viewerTransform.isTracking();

    const cm = (v: number) => `${(v * 100).toFixed(0).padStart(4)}cm`;

    hud.textContent = [
      `RENDER   ${renderFps.toFixed(0).padStart(3)} fps   buffer ${px.drawingBuffer}`,
      `         native ${px.nativePixels}  ${px.sharp ? "SHARP" : "UPSCALED <-"}`,
      `         budget ${px.maxDrawMPix}MP  ratio ${px.appliedRatio.toFixed(2)} / dpr ${px.devicePixelRatio}`,
      `DISPLAY  fill ${(px.bufferVsScreenWidth * 100).toFixed(0)}% of ${px.screen.split(",")[1]?.trim() ?? "?"}  ${document.fullscreenElement ? "[fullscreen]" : "[windowed]"}`,
      px.bufferVsScreenWidth < 0.9
        ? "         <- not filling the display: press F11"
        : "         OK",
      `BUILD    ${bundleId()}`,
      "",
      `TRACK    ${t.running ? (t.tracking ? "LOCKED" : "searching") : "off"}   ${t.fps} fps  ${t.delegate ?? "-"}`,
      t.error ? `  ERR ${t.error}` : `  cam ${t.videoSize ?? "-"}`,
      `POSE     x${cm(p.x)} y${cm(p.y)} z${cm(p.z)}`,
      "",
      `OFF-AXIS ${offAxisLive ? "ACTIVE" : "standby (needs a tracked head)"}`,
      `PANEL    ${SPACE.panelWidthM.toFixed(2)} x ${SPACE.panelHeightM.toFixed(2)} m`,
      `         fit ${agree.toFixed(2)}  ${Math.abs(agree - 1) < 0.1 ? "OK" : "<- measure it (__protoCalibrate)"}`,
    ].join("\n");
  };

  paint();
  timer = window.setInterval(paint, 250);
  return "HUD shown — __spaceHud(false) to hide";
}
