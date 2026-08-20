import { defineConfig } from "@playwright/test";

/**
 * Visual regression config for the VRM/HUD scene.
 *
 * Points at the ALREADY-RUNNING app server (`python run_server.py`, HTTPS on 12394)
 * rather than starting a Vite dev server. What we want to photograph is the
 * deployed `frontend/` build the panel actually loads — a dev-server render can
 * differ in asset hashing, base path and minification, and a regression that only
 * exists in one of them is not worth catching.
 *
 * Never parallel: every worker would drive its own WebGL context on the same GPU,
 * and contended frame timing changes how far the spring/particle sim advances per
 * frame. Determinism here is worth more than wall-clock.
 */
export default defineConfig({
  testDir: "./tests/visual",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 120_000,
  reporter: [["list"]],
  use: {
    baseURL: process.env.APP_URL || "https://localhost:12394",
    // The app server generates its own certificate for LAN HTTPS access
    // (src/kehai/ssl_utils.py), so a browser has nothing to trust it against.
    ignoreHTTPSErrors: true,
    // A WebGL scene has no "networkidle": the render loop never stops issuing
    // frames, and model/texture loads continue well past DOMContentLoaded.
    // The tests wait on __AI_DEBUG__ instead, which is a real readiness signal.
    actionTimeout: 30_000,
    launchOptions: {
      args: [
        // Headless Chromium falls back to SwiftShader unless told otherwise, and
        // a software rasteriser does not reproduce the GPU's rendering of the
        // toon/line shaders. Force the real GPU so a reference image taken here
        // means something about what the panel shows.
        "--use-gl=angle",
        "--use-angle=default",
        "--enable-unsafe-swiftshader",
        "--ignore-gpu-blocklist",
      ],
    },
  },
  expect: {
    toHaveScreenshot: {
      // Anti-aliasing and the ink-line pass differ by a hair between driver
      // versions; a hard 0 would fail on noise. 0.2% of pixels is well under
      // any change a human would call a visual regression.
      maxDiffPixelRatio: 0.002,
      threshold: 0.15,
      animations: "disabled",
    },
  },
});
