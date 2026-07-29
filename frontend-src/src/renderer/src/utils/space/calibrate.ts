import { SPACE, saveSpaceConfig } from "./space-config";

/**
 * Panel calibration overlay.
 *
 * Off-axis projection is a claim about real-world geometry: it only works if the
 * virtual window is the same size and in the same place as the physical one.
 * Proto publish the CABINET size and an "86-inch display" figure, but not the
 * visible image rectangle — and on a unit fed over HDMI the image may not even
 * fill the panel. No amount of software can discover this; someone has to hold a
 * tape measure against the glass once.
 *
 * So: draw a frame around exactly the pixels the scene occupies, with the
 * numbers to type back in. Two minutes, once per installation.
 */

let overlay: HTMLDivElement | null = null;

export function hidePanelCalibration() {
  overlay?.remove();
  overlay = null;
}

export function showPanelCalibration(
  container: HTMLElement,
  canvas?: HTMLCanvasElement | null,
): string {
  hidePanelCalibration();

  const el = document.createElement("div");
  el.style.cssText = [
    "position:absolute",
    "inset:0",
    "z-index:9999",
    "pointer-events:none",
    "box-sizing:border-box",
    "border:2px solid #35d7ff",
    "font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace",
    "color:#eaf6ff",
  ].join(";");

  // Ruler ticks every 10% so a tape measure can be checked against more than
  // just the outer edge — a mis-set HDMI scaling shows up as uneven spacing.
  const ticks = document.createElement("div");
  ticks.style.cssText = "position:absolute;inset:0";
  for (let i = 1; i < 10; i += 1) {
    const v = document.createElement("div");
    v.style.cssText = `position:absolute;top:0;bottom:0;left:${i * 10}%;width:1px;background:rgba(53,215,255,${i === 5 ? 0.85 : 0.28})`;
    ticks.appendChild(v);
    const h = document.createElement("div");
    h.style.cssText = `position:absolute;left:0;right:0;top:${i * 10}%;height:1px;background:rgba(53,215,255,${i === 5 ? 0.85 : 0.28})`;
    ticks.appendChild(h);
  }
  el.appendChild(ticks);

  const w = container.clientWidth;
  const h = container.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const buf = canvas ? `${canvas.width}x${canvas.height}` : "n/a";
  const nativeW = Math.round(w * dpr);
  const nativeH = Math.round(h * dpr);
  const sharp = canvas ? canvas.width >= nativeW - 1 : false;
  const box = document.createElement("div");
  box.style.cssText = [
    "position:absolute",
    "left:50%",
    "top:50%",
    "transform:translate(-50%,-50%)",
    "background:rgba(6,4,16,0.88)",
    "border:1px solid rgba(53,215,255,0.5)",
    "padding:14px 18px",
    "max-width:88%",
    "white-space:pre-wrap",
  ].join(";");
  box.textContent = [
    "PANEL CALIBRATION",
    "",
    "-- 実測して入れる --------------------------------",
    "1) この枠の外周を実測（メートル）",
    "2) 枠の中心が床から何mかを実測",
    "3) 数値を置き換えて実行:",
    `   __space({ panelWidthM: ${SPACE.panelWidthM}, panelHeightM: ${SPACE.panelHeightM}, panelCenterY: ${SPACE.panelCenterY} })`,
    "4) __protoCalibrate(false) で閉じる",
    "",
    "-- 現在の状態 ------------------------------------",
    `ブラウザ viewport : ${vw} x ${vh} css`,
    `3D 描画領域       : ${w} x ${h} css  (比 ${(w / h).toFixed(3)})`,
    `devicePixelRatio  : ${dpr}`,
    `描画バッファ      : ${buf}`,
    `パネル実画素相当  : ${nativeW} x ${nativeH}`,
    sharp
      ? "✓ バッファが実画素に足りています（拡大されていません）"
      : "⚠ バッファ不足＝拡大表示。__spacePixels(8.3) で最大解像度",
    "",
    vw < 700
      ? `⚠ viewport が ${vw}css px しかありません。ウィンドウが最大化されていないか、\n  表示スケーリングが極端です。全画面(F11)にしてから測り直してください。`
      : "✓ viewport 幅は妥当です",
  ].join("\n");
  el.appendChild(box);

  container.appendChild(el);
  overlay = el;
  return `calibration overlay shown (${w}x${h} css px)`;
}

/**
 * Fix the tracker's distance scale from one measurement.
 *
 * Reported Z is `eyeSpan * focal / eyePixels`, and `focal` depends on the
 * camera's horizontal FOV — the one quantity a browser cannot query. Stand a
 * measured distance from the camera, call this with that distance, and the FOV
 * is solved so the report matches reality. Everything downstream (parallax
 * amount, distance reactions in Phase 4) rides on this being right.
 */
export function calibrateHeadDistance(trueMetres: number, reportedMetres: number): string {
  if (!(trueMetres > 0.1) || !(reportedMetres > 0.05)) {
    return "need a real measured distance (m) and an active track";
  }
  const ratio = reportedMetres / trueMetres;
  const halfRad = (SPACE.headCamFovDeg * Math.PI) / 360;
  const newHalf = Math.atan(Math.tan(halfRad) * ratio);
  const next = (newHalf * 360) / Math.PI;
  if (!Number.isFinite(next) || next < 20 || next > 140) {
    return `refusing an implausible FOV (${next.toFixed(1)}deg) — re-measure`;
  }
  const before = SPACE.headCamFovDeg;
  SPACE.headCamFovDeg = next;
  saveSpaceConfig();
  return `headCamFovDeg ${before.toFixed(1)} -> ${next.toFixed(1)} (reported ${reportedMetres.toFixed(2)}m vs actual ${trueMetres.toFixed(2)}m)`;
}
