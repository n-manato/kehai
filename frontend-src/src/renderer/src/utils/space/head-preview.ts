import { headTracker } from "./head-tracker";
import { viewerTransform } from "./viewer-transform";

/**
 * Camera preview with the detection drawn on top.
 *
 * "TRACK: searching" tells you detection is failing but not why, and the
 * candidates — camera aimed at the wrong side of the cabinet, face backlit by a
 * 4K panel, standing too far for a 640x480 frame, wrong device picked — all look
 * identical from the console. Showing the actual frame with the actual landmarks
 * collapses all of that into one glance.
 *
 * It draws the landmarks the TRACKER produced, not a second inference of its
 * own: a separate detection pass could disagree with the one driving the scene,
 * which would make the preview actively misleading.
 */

let box: HTMLDivElement | null = null;
let canvas: HTMLCanvasElement | null = null;
let timer = 0;

const WIDTH = 240;

/** Break a message into canvas-drawable lines — no text wrapping on a canvas. */
function wrap(text: string, chars: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length && out.length < 4; i += chars) {
    out.push(text.slice(i, i + chars));
  }
  return out;
}

export function hideHeadPreview() {
  if (timer) clearInterval(timer);
  timer = 0;
  box?.remove();
  box = null;
  canvas = null;
}

export function showHeadPreview(container: HTMLElement): string {
  hideHeadPreview();

  box = document.createElement("div");
  box.style.cssText = [
    "position:absolute",
    "right:8px",
    "bottom:150px", // clear of the chat input bar that overlays the canvas
    "z-index:9998",
    "pointer-events:none",
    "background:rgba(6,4,16,0.85)",
    "border:1px solid rgba(53,215,255,0.45)",
    "border-radius:6px",
    "padding:6px",
    "font:11px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace",
    "color:#dff1ff",
  ].join(";");

  canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;width:240px;height:180px;background:#000;border-radius:3px";
  box.appendChild(canvas);

  const label = document.createElement("div");
  label.style.cssText = "margin-top:4px;white-space:pre";
  box.appendChild(label);

  container.appendChild(box);

  const draw = () => {
    const v = headTracker.getVideo();
    const c = canvas;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;

    if (!v || v.readyState < 2) {
      c.width = WIDTH;
      c.height = Math.round(WIDTH * 0.75);
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, c.width, c.height);
      const s = headTracker.status();
      // Say WHICH of the three no-picture states this is, in the picture. "no
      // camera stream" alone reads as "the camera is broken" when the usual
      // cause is simply that nothing has been started yet.
      const lines = s.error
        ? ["ERROR", ...wrap(s.error, 30)]
        : s.running
          ? ["starting...", "loading model / waiting", "for the first frame"]
          : ["TRACKER IS OFF", "", "run:", "__headTracker(true)"];
      ctx.fillStyle = s.error ? "#ff6b6b" : s.running ? "#ffe066" : "#7fc9ff";
      ctx.font = "bold 13px monospace";
      lines.forEach((line, i) => ctx.fillText(line, 10, 26 + i * 17));
      label.textContent = s.activeCamera ? `dev: ${s.activeCamera}` : "no device yet";
      return;
    }

    const vw = v.videoWidth || 640;
    const vh = v.videoHeight || 480;
    if (c.width !== WIDTH) {
      c.width = WIDTH;
      c.height = Math.round((WIDTH * vh) / vw);
      c.style.height = `${c.height}px`;
    }
    ctx.drawImage(v, 0, 0, c.width, c.height);

    const face = headTracker.getLastFace();
    if (face && face.length) {
      ctx.fillStyle = "rgba(53,215,255,0.75)";
      for (let i = 0; i < face.length; i += 4) {
        ctx.fillRect(face[i].x * c.width - 1, face[i].y * c.height - 1, 2, 2);
      }
      // The three landmarks the distance/position maths actually uses, so a bad
      // estimate can be traced to a bad landmark rather than guessed at.
      const marks: [number, string][] = [[33, "#ffe066"], [263, "#ffe066"], [168, "#ff4fd8"]];
      marks.forEach(([idx, colour]) => {
        const p = face[idx];
        if (!p) return;
        ctx.strokeStyle = colour;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(p.x * c.width, p.y * c.height, 4, 0, Math.PI * 2);
        ctx.stroke();
      });
    } else {
      ctx.strokeStyle = "#ff6b6b";
      ctx.lineWidth = 2;
      ctx.strokeRect(1, 1, c.width - 2, c.height - 2);
    }

    const p = viewerTransform.get();
    const s = headTracker.status();
    label.textContent = [
      `${face ? "FACE OK" : "no face"}   ${vw}x${vh}  ${s.fps}fps`,
      `x ${(p.x * 100).toFixed(0)}cm  y ${(p.y * 100).toFixed(0)}cm  z ${(p.z * 100).toFixed(0)}cm`,
    ].join("\n");
  };

  draw();
  // 10 Hz: enough to see the tracking respond, cheap enough not to disturb it.
  timer = window.setInterval(draw, 100);
  return "camera preview shown — __headPreview(false) to hide";
}
