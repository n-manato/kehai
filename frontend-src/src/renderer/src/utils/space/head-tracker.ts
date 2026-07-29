import type { FaceLandmarker } from "@mediapipe/tasks-vision";
import { SPACE, saveSpaceConfig } from "./space-config";
import { viewerTransform } from "./viewer-transform";

/**
 * Phase 2 — find the viewer's head and write it into `viewerTransform`.
 *
 * Camera source is abstracted on purpose. The PROTO Luma has a built-in
 * "AI-Enhanced 4K Camera", but whether that camera is exposed to a browser
 * running on an EXTERNAL PC driving the panel over HDMI is not something we can
 * assume — from the PC's point of view it may simply not exist. So this takes an
 * optional `deviceId` and works with any `videoinput`: the built-in camera if it
 * enumerates, otherwise a USB webcam stuck on top of the cabinet. `listCameras()`
 * is there so a picker can be built without touching this file again.
 *
 * Everything is self-hosted (wasm under /libs/mediapipe/, model vendored beside
 * it) because the app is meant to run offline and the illusion must not depend
 * on a CDN being reachable at startup.
 */

const WASM_PATH = "libs/mediapipe";
const MODEL_PATH = "libs/mediapipe/face_landmarker.task";

/**
 * MediaPipe landmark indices for the outer eye corners. The distance between
 * them scales as 1/distance, which is what gives us Z from a single camera.
 * (Outer corners, not pupils: they sit on bone and barely move with expression
 * or gaze, so the estimate doesn't wobble when she makes her look around.)
 */
const LEFT_EYE_OUTER = 33;
const RIGHT_EYE_OUTER = 263;
/** Between-the-eyes landmark — a stable proxy for "where the head is". */
const NOSE_BRIDGE = 168;

export interface CameraOption {
  deviceId: string;
  label: string;
}

class HeadTracker {
  private video: HTMLVideoElement | null = null;

  private stream: MediaStream | null = null;

  private landmarker: FaceLandmarker | null = null;

  private raf = 0;

  private lastVideoTime = -1;

  private running = false;

  private startingPromise: Promise<boolean> | null = null;

  private lastError: string | null = null;

  private fps = 0;

  private frameCount = 0;

  private fpsWindowStart = 0;

  private delegate: "GPU" | "CPU" | null = null;

  /** Which device the browser actually handed us — not necessarily the one asked for. */
  private activeCamera: string | null = null;

  /** performance.now() when start() finished, to notice a stream that never produces frames. */
  private startedAt = 0;

  private lastFace: { x: number; y: number }[] | null = null;

  /** The live camera element, for the debug preview. */
  getVideo(): HTMLVideoElement | null {
    return this.video;
  }

  /** Landmarks from the most recent detection (normalised 0..1), or null. */
  getLastFace() {
    return this.lastFace;
  }

  /** Cameras the browser can see. Labels need permission to be non-empty. */
  async listCameras(): Promise<CameraOption[]> {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices
        .filter((d) => d.kind === "videoinput")
        .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Camera ${i + 1}` }));
    } catch {
      return [];
    }
  }

  isRunning() {
    return this.running;
  }

  /**
   * Resolve once any in-flight start has settled.
   *
   * Starting takes a couple of seconds (permission, ~15MB of wasm and model), so
   * anything that inspects the tracker right after asking it to start would
   * otherwise see "not running" and draw the wrong conclusion — or worse, open a
   * competing camera stream.
   */
  async whenReady(): Promise<void> {
    if (this.startingPromise) await this.startingPromise;
  }

  /** Stop, switch device, start again. Index into listCameras(), or a deviceId. */
  async useCamera(which: number | string): Promise<boolean> {
    const cams = await this.listCameras();
    const id = typeof which === "number" ? cams[which]?.deviceId : which;
    if (!id) return false;
    SPACE.headCamDeviceId = id;
    saveSpaceConfig();
    this.stop();
    return this.start(id);
  }

  status() {
    return {
      running: this.running,
      tracking: viewerTransform.isTracking(),
      fps: Math.round(this.fps),
      error: this.lastError ?? this.stallReason(),
      delegate: this.delegate,
      activeCamera: this.activeCamera,
      videoSize: this.video ? `${this.video.videoWidth}x${this.video.videoHeight}` : null,
      pose: { ...viewerTransform.get() },
      deviceId: SPACE.headCamDeviceId || "(default)",
    };
  }

  /**
   * A stream can be granted and then never deliver a frame — a disconnected or
   * in-use device, or the wrong one of several. Silence looks identical to
   * "still starting", so turn it into a stated problem after a few seconds.
   */
  private stallReason(): string | null {
    if (!this.running || !this.startedAt) return null;
    if (performance.now() - this.startedAt < 5000) return null;
    const v = this.video;
    if (!v || v.readyState < 2) {
      return `camera "${this.activeCamera ?? "?"}" granted but sending no frames — try another with __headCamera(n)`;
    }
    if (this.fps < 1) return "video is playing but detection is not running";
    return null;
  }

  async start(deviceId?: string): Promise<boolean> {
    if (this.running) return true;
    if (this.startingPromise) return this.startingPromise;
    this.startingPromise = this.doStart(deviceId).finally(() => {
      this.startingPromise = null;
    });
    return this.startingPromise;
  }

  private async doStart(deviceId?: string): Promise<boolean> {
    this.lastError = null;
    try {
      const id = deviceId ?? SPACE.headCamDeviceId ?? undefined;
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          /**
           * 1280x720, not 640x480. Work the geometry: at the PROTO's viewing
           * distance of ~2 m through a 60-degree lens, a 16 cm face spans
           *   0.16 / (2 * 2 * tan(30)) = 7% of the frame
           * which is only ~44 px at 640 wide — below what the face detector can
           * reliably find. At 1280 the same face is ~90 px and detects. Still
           * far short of the camera's 4K, because the cost is per pixel and the
           * illusion needs latency more than it needs detail.
           */
          width: { ideal: SPACE.headCamWidth },
          height: { ideal: Math.round((SPACE.headCamWidth * 9) / 16) },
          frameRate: { ideal: 30 },
          ...(id ? { deviceId: { exact: id } } : {}),
        },
        audio: false,
      });

      const video = document.createElement("video");
      video.autoplay = true;
      video.muted = true;
      video.playsInline = true;
      video.srcObject = this.stream;
      // Kept out of the layout entirely; the tracker never needs to show it.
      video.style.cssText = "position:absolute;width:1px;height:1px;opacity:0;pointer-events:none";
      document.body.appendChild(video);
      this.video = video;
      await video.play();

      // Loaded on demand, not at module scope: with tracking off this keeps
      // ~300KB of MediaPipe out of the main bundle and out of the parse budget,
      // so "camera off" really means nothing camera-related runs at all.
      const { FaceLandmarker, FilesetResolver } = await import("@mediapipe/tasks-vision");
      const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
      const makeLandmarker = (delegate: "GPU" | "CPU") =>
        FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: MODEL_PATH, delegate },
          runningMode: "VIDEO",
          numFaces: 1, // single primary viewer by design — see the plan
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
        });
      try {
        this.landmarker = await makeLandmarker("GPU");
        this.delegate = "GPU";
      } catch (gpuErr) {
        // The GPU delegate needs a WebGL2 context MediaPipe can claim, and it is
        // not always available — notably when the page is already running a
        // heavy WebGL renderer, which is exactly our situation. CPU is slower
        // but the tracker only needs ~30Hz on a 640x480 frame.
        console.warn("[head-tracker] GPU delegate unavailable, falling back to CPU:", gpuErr);
        this.landmarker = await makeLandmarker("CPU");
        this.delegate = "CPU";
      }

      const track = this.stream.getVideoTracks()[0];
      this.activeCamera = track
        ? `${track.label || "unnamed"} (${track.getSettings().deviceId?.slice(0, 8) ?? "?"})`
        : null;

      this.running = true;
      this.startedAt = performance.now();
      this.fpsWindowStart = performance.now();
      this.loop();
      return true;
    } catch (err) {
      this.lastError = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
      // eslint-disable-next-line no-console
      console.warn("[head-tracker] start failed:", this.lastError);
      this.stop();
      return false;
    }
  }

  stop() {
    this.running = false;
    cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.landmarker?.close();
    this.landmarker = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    if (this.video) {
      this.video.srcObject = null;
      this.video.remove();
      this.video = null;
    }
    this.lastVideoTime = -1;
    this.delegate = null;
    this.lastFace = null;
    this.activeCamera = null;
    this.startedAt = 0;
    this.fps = 0;
    viewerTransform.reset();
  }

  private loop = () => {
    if (!this.running) return;
    this.raf = requestAnimationFrame(this.loop);

    const video = this.video;
    const landmarker = this.landmarker;
    if (!video || !landmarker || video.readyState < 2) return;
    // MediaPipe VIDEO mode requires strictly increasing timestamps, and the
    // display refresh usually outruns the camera — feeding the same frame twice
    // throws rather than returning a cached result.
    if (video.currentTime === this.lastVideoTime) return;
    this.lastVideoTime = video.currentTime;

    const nowMs = performance.now();
    let result;
    try {
      result = landmarker.detectForVideo(video, nowMs);
    } catch (err) {
      this.lastError = err instanceof Error ? err.message : String(err);
      return;
    }

    this.frameCount += 1;
    if (nowMs - this.fpsWindowStart > 1000) {
      this.fps = (this.frameCount * 1000) / (nowMs - this.fpsWindowStart);
      this.frameCount = 0;
      this.fpsWindowStart = nowMs;
    }

    const face = result.faceLandmarks?.[0];
    const tSec = nowMs / 1000;
    this.lastFace = face ?? null;
    if (!face) {
      viewerTransform.markLost(tSec);
      return;
    }

    const pose = this.landmarksToPose(face, video.videoWidth || 640, video.videoHeight || 480);
    if (!pose) {
      viewerTransform.markLost(tSec);
      return;
    }
    viewerTransform.setMeasurement(pose.x, pose.y, pose.z, pose.yaw, pose.pitch, pose.roll, tSec);
  };

  /**
   * Normalised landmarks -> metres in panel-centre coordinates.
   *
   * Z comes from apparent eye separation: the outer eye corners are a known
   * real-world width, so their width in pixels pins the distance. That needs the
   * camera's horizontal FOV, which we can't read from the browser — hence
   * `SPACE.headCamFovDeg`, calibrated once against the real camera by standing a
   * measured distance away and comparing `__headTracker().pose.z`.
   */
  private landmarksToPose(
    face: { x: number; y: number; z: number }[],
    videoW: number,
    videoH: number,
  ) {
    const le = face[LEFT_EYE_OUTER];
    const re = face[RIGHT_EYE_OUTER];
    const nb = face[NOSE_BRIDGE];
    if (!le || !re || !nb) return null;

    // Landmark x/y are normalised to the frame, so convert to pixels before
    // measuring a width — otherwise a non-square frame skews the distance.
    const dxPx = (le.x - re.x) * videoW;
    const dyPx = (le.y - re.y) * videoH;
    const eyePx = Math.hypot(dxPx, dyPx);
    if (eyePx < 1) return null;

    const hFov = (SPACE.headCamFovDeg * Math.PI) / 180;
    const focalPx = videoW / (2 * Math.tan(hFov / 2));

    // Pinhole: apparent size = real size * focal / distance.
    let z = (SPACE.eyeSpanM * focalPx) / eyePx;
    z = Math.min(Math.max(z, 0.25), 6);

    // Head centre in pixels, then back-projected at that distance.
    const cxPx = nb.x * videoW;
    const cyPx = nb.y * videoH;
    let x = ((cxPx - videoW / 2) / focalPx) * z;
    const y = -((cyPx - videoH / 2) / focalPx) * z;

    // Webcams present a non-mirrored image, but the viewer moving to THEIR right
    // must move the virtual eye to the right of the panel too.
    if (SPACE.headCamMirror) x = -x;

    // Roll is the only orientation we can read reliably from 2D landmarks;
    // yaw/pitch from a flat landmark set are noisy enough to be worse than
    // nothing, and off-axis projection needs only the position anyway.
    const roll = (Math.atan2(dyPx, dxPx) * 180) / Math.PI;

    return {
      // Camera is mounted above/below the panel centre, not at it.
      x: x + SPACE.headCamOffsetX,
      y: y + SPACE.headCamOffsetY,
      z: z + SPACE.headCamOffsetZ,
      yaw: 0,
      pitch: 0,
      roll,
    };
  }
}

export const headTracker = new HeadTracker();

/**
 * Stop, restart, wait for detection to settle, then diagnose — in that order.
 *
 * Handing over three commands to paste made the ordering the operator's problem:
 * run the diagnosis immediately after the start and it inspects a tracker that
 * has not finished starting, and opens a second camera stream that fights the
 * first. One command, correct order, no race.
 */
export async function setupHeadTracking(): Promise<Record<string, unknown>> {
  console.log("[headSetup] stopping any existing tracker...");
  headTracker.stop();
  await new Promise((r) => setTimeout(r, 300));
  console.log("[headSetup] starting (camera permission + model load)...");
  const ok = await headTracker.start();
  console.log(`[headSetup] start ${ok ? "succeeded" : "FAILED"}`);
  if (ok) {
    console.log("[headSetup] letting detection settle for 3s...");
    await new Promise((r) => setTimeout(r, 3000));
  }
  return diagnoseHeadTracking();
}

/**
 * Walk the whole start-up path and report where it actually breaks.
 *
 * Camera bring-up has about eight ways to fail — insecure context, no device,
 * permission, a device that grants but never sends frames, the wrong one of
 * several, wasm that will not fetch, a model that will not parse, a detector
 * that runs but finds no face — and from the outside they all look the same:
 * a black box. Colour-coding the states only moved the guessing from me to the
 * person standing at the machine. This checks each step in order and says which
 * one failed, so nobody has to interpret anything.
 *
 * Safe to run while the tracker is live: it reuses the running stream rather
 * than opening a second one.
 */
export async function diagnoseHeadTracking(): Promise<Record<string, unknown>> {
  const steps: { step: string; ok: boolean; detail: string }[] = [];
  const add = (step: string, ok: boolean, detail: unknown = "") => {
    const row = { step, ok, detail: String(detail) };
    steps.push(row);
    // Logged as they happen, so a step that HANGS is visible too — a returned
    // summary only ever shows up if the whole thing completes.
    console.log(`[headDiag] ${ok ? "OK  " : "FAIL"} ${step}${row.detail ? ` — ${row.detail}` : ""}`);
    return ok;
  };

  add("secure context (required for getUserMedia)", window.isSecureContext, window.location.origin);
  if (!navigator.mediaDevices?.getUserMedia) {
    add("navigator.mediaDevices.getUserMedia exists", false, "missing");
    return { steps, verdict: "this browser/origin cannot open a camera at all" };
  }
  add("navigator.mediaDevices.getUserMedia exists", true);

  let devices: MediaDeviceInfo[] = [];
  try {
    devices = await navigator.mediaDevices.enumerateDevices();
  } catch (e) {
    add("enumerateDevices", false, e);
  }
  const cams = devices.filter((d) => d.kind === "videoinput");
  add("video inputs found", cams.length > 0, cams.map((d, i) => `[${i}] ${d.label || "(unnamed)"}`).join(" | "));

  // If it is already running, judge the live stream instead of grabbing another.
  if (headTracker.isRunning()) {
    const st = headTracker.status();
    const v = headTracker.getVideo();
    add("tracker running", true, `delegate ${st.delegate}, ${st.fps} fps`);
    add("active device", !!st.activeCamera, st.activeCamera ?? "unknown");
    add("video has frames", !!v && v.readyState >= 2, v ? `readyState ${v.readyState}, ${v.videoWidth}x${v.videoHeight}` : "no element");
    add("detector producing results", st.fps > 0, `${st.fps} fps`);
    const face = headTracker.getLastFace();
    add("face detected", !!face, face ? `${face.length} landmarks` : "no face in the current frame");
    return {
      steps,
      verdict: face
        ? "working"
        : (v && v.readyState >= 2
          ? "camera and detector are fine, but no FACE is in frame — check where the camera points, lighting, and distance (__headPreview)"
          : "the stream is not delivering frames — try __headCamera(n) with another device"),
    };
  }

  // Wait out an in-flight start rather than racing it.
  await headTracker.whenReady();
  if (headTracker.isRunning()) {
    add("tracker running", true, "(finished starting while diagnosing — re-run for detector detail)");
    return { steps, verdict: "tracker just finished starting — run __headDiag() again" };
  }

  add("tracker running", false, "not started — probing the camera directly");
  let stream: MediaStream | null = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } } });
    const t = stream.getVideoTracks()[0];
    add("getUserMedia", true, t ? `${t.label} (${t.readyState})` : "no track");
  } catch (e) {
    add("getUserMedia", false, e);
    stream?.getTracks().forEach((t) => t.stop());
    return { steps, verdict: "the browser refused or could not open a camera — see the failing step" };
  }

  const probe = document.createElement("video");
  probe.muted = true;
  probe.playsInline = true;
  probe.srcObject = stream;
  document.body.appendChild(probe);
  try {
    await probe.play();
    add("video.play()", true);
  } catch (e) {
    add("video.play()", false, e);
  }
  await new Promise((r) => setTimeout(r, 1500));
  add("frames arriving", probe.readyState >= 2, `readyState ${probe.readyState}, ${probe.videoWidth}x${probe.videoHeight}`);
  probe.srcObject = null;
  probe.remove();
  stream.getTracks().forEach((t) => t.stop());

  return {
    steps,
    verdict: "camera itself is fine — now run __headTracker(true) and re-run __headDiag() to check the detector",
  };
}
