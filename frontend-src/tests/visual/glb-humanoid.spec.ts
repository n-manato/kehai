import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";

/**
 * A plain .glb has to arrive as a DRIVEABLE humanoid, not just as geometry.
 *
 * Goes through the real path — switch the character the way the UI does, then
 * inspect what the viewer actually built — rather than importing the mapper in
 * the page. Module specifiers are bundled away in the shipped build so an
 * in-page `import()` cannot resolve them; more to the point, a unit test of the
 * mapper would still pass if the viewer stopped calling it.
 *
 * Asserts the bones the MOTION path reads. The failure this guards against is a
 * rig that loads and renders perfectly and then moves wrongly because one name
 * silently missed — which no screenshot would catch.
 */

/** Bones the motion, gaze and pose code address by name. */
const NEEDED = [
  "hips", "spine", "chest", "upperChest", "neck", "head",
  "leftEye", "rightEye",
  "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand",
  "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand",
  "leftUpperLeg", "leftLowerLeg", "leftFoot", "leftToes",
  "rightUpperLeg", "rightLowerLeg", "rightFoot", "rightToes",
  "leftThumbMetacarpal", "leftIndexProximal", "leftLittleDistal",
  "rightThumbMetacarpal", "rightIndexProximal", "rightLittleDistal",
];

/** The app opens its socket before any test code runs, so grab it at construction. */
const CAPTURE_SOCKET = `window.__socks = [];
const OrigWS = window.WebSocket;
window.WebSocket = function (...a) { const s = new OrigWS(...a); window.__socks.push(s); return s; };
window.WebSocket.prototype = OrigWS.prototype;
Object.assign(window.WebSocket, OrigWS);`;

async function switchCharacter(page: Page, file: string) {
  await page.evaluate((f) => {
    const socks = (window as unknown as { __socks: WebSocket[] }).__socks;
    const sock = socks.find((s) => s.readyState === 1);
    if (!sock) throw new Error("no open websocket to switch on");
    sock.send(JSON.stringify({ type: "switch-config", file: f }));
  }, file);
  // The swap tears the old model down and loads the new one.
  await page.waitForTimeout(16_000);
}

test("a plain .glb loads as a full VRM humanoid, and Miku still works", async ({ page }) => {
  test.setTimeout(240_000);

  await page.addInitScript(CAPTURE_SOCKET);
  await page.goto("/");
  await page.waitForFunction(
    () => !!(window as unknown as { __vrmScene?: unknown }).__vrmScene,
    { timeout: 120_000 },
  );
  await page.waitForTimeout(4000);

  // --- the .glb ------------------------------------------------------------
  await switchCharacter(page, "chancelor.yaml");

  const glb = await page.evaluate((needed) => {
    const w = window as unknown as {
      __vrm?: {
        scene: unknown;
        springBoneManager?: unknown;
        humanoid?: {
          getRawBoneNode?: (n: string) => { name: string } | null;
          getNormalizedBoneNode?: (n: string) => unknown;
        };
      };
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      __three: any;
    };
    const v = w.__vrm;
    if (!v) return null;
    const raw: Record<string, string | null> = {};
    let normalized = 0;
    for (const n of needed) {
      const r = v.humanoid?.getRawBoneNode?.(n);
      raw[n] = r ? r.name : null;
      if (v.humanoid?.getNormalizedBoneNode?.(n)) normalized += 1;
    }
    const box = new w.__three.Box3().setFromObject(v.scene);
    return {
      raw,
      normalized,
      heightM: box.max.y - box.min.y,
      footY: box.min.y,
      springBones: !!v.springBoneManager,
    };
  }, NEEDED);

  expect(glb, "no VRM built for the .glb").not.toBeNull();
  if (!glb) return;

  const unmapped = Object.entries(glb.raw).filter(([, v]) => v === null).map(([k]) => k);
  expect(unmapped, `unmapped humanoid bones: ${unmapped.join(", ")}`).toHaveLength(0);
  // The NORMALIZED rig is what motion is applied to. Without it every pose would
  // land on the raw A-pose and the character would stand crooked.
  expect(glb.normalized, "normalized rig incomplete").toBe(NEEDED.length);
  // Real-world scale, standing ON the floor rather than sunk or hovering.
  expect(glb.heightM).toBeGreaterThan(1.4);
  expect(glb.heightM).toBeLessThan(2.2);
  expect(Math.abs(glb.footY)).toBeLessThan(0.05);
  // A plain .glb cannot carry spring bones. Asserted so a future "why is the
  // hair stiff on this model" has its answer already written down.
  expect(glb.springBones).toBe(false);

  // --- and Miku is unaffected ---------------------------------------------
  await switchCharacter(page, "hatsune_miku.yaml");

  const miku = await page.evaluate(() => {
    const w = window as unknown as {
      __vrm?: {
        scene: unknown;
        springBoneManager?: unknown;
        humanoid?: { getNormalizedBoneNode?: (n: string) => unknown };
      };
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      __three: any;
    };
    const v = w.__vrm;
    if (!v) return null;
    const box = new w.__three.Box3().setFromObject(v.scene);
    return {
      hasHips: !!v.humanoid?.getNormalizedBoneNode?.("hips"),
      springBones: !!v.springBoneManager,
      heightM: box.max.y - box.min.y,
    };
  });

  expect(miku, "Miku failed to come back").not.toBeNull();
  expect(miku?.hasHips).toBe(true);
  expect(miku?.springBones, "Miku lost her spring bones").toBe(true);
  expect(miku?.heightM).toBeGreaterThan(1.4);
});

test("the face audition panel reads whatever model is loaded", async ({ page }) => {
  test.setTimeout(240_000);

  await page.addInitScript(CAPTURE_SOCKET);
  await page.goto("/");
  await page.waitForFunction(
    () => !!(window as unknown as { __vrmScene?: unknown }).__vrmScene,
    { timeout: 120_000 },
  );
  await page.waitForTimeout(4000);

  await switchCharacter(page, "chancelor.yaml");

  const open = async () => page.evaluate(async () => {
    const btn = [...document.querySelectorAll("button")].find((b) => b.textContent === "😀");
    if (!btn) throw new Error("no face-audition button");
    btn.click();
    await new Promise((r) => setTimeout(r, 700));
  });

  await open();

  const glb = await page.evaluate(async () => {
    const panel = document.querySelector('.vrm-tune-panel[style*="flex"]');
    if (!panel) return null;
    const names = [...panel.querySelectorAll("span[title]")].map((s) => (s as HTMLElement).title);
    const rows = [...panel.querySelectorAll("input[type=range]")] as HTMLInputElement[];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const v = (window as any).__vrm;
    const read = (n: string) => {
      const out: number[] = [];
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      v.scene.traverse((o: any) => {
        if (o.isMesh && o.morphTargetDictionary && o.morphTargetInfluences) {
          const i = o.morphTargetDictionary[n];
          if (i != null) out.push(o.morphTargetInfluences[i]);
        }
      });
      return out;
    };
    const target = names.find((n) => n === "jawOpen") || names[0];
    const before = read(target);
    const idx = names.indexOf(target);
    rows[idx].value = "1";
    rows[idx].dispatchEvent(new Event("input", { bubbles: true }));
    await new Promise((r) => setTimeout(r, 900));
    const after = read(target);
    const heads = [...panel.querySelectorAll("div")].map((d) => d.textContent)
      .filter((t) => ["viseme", "mouth", "eye", "brow", "face", "body-shape", "other"].includes(t || ""));
    const clipLabel = (panel.children[1] as HTMLElement)?.innerText?.split("\n")[0] || "";
    (([...panel.querySelectorAll("button")].find((b) => b.textContent === "全解除")) as HTMLButtonElement)?.click();
    return { count: names.length, target, before, after, heads, clipLabel };
  });

  expect(glb, "panel did not open for the .glb").not.toBeNull();
  if (!glb) return;
  expect(glb.count).toBeGreaterThan(50);
  // The morph must actually reach the meshes — the panel writing to nothing is
  // the failure that looks identical to a working panel in a screenshot.
  expect(glb.before.every((v) => v < 0.01)).toBe(true);
  expect(glb.after.every((v) => v > 0.99)).toBe(true);
  // Group headings appear once each, in order — they interleaved before sorting.
  expect(glb.heads.length).toBe(new Set(glb.heads).size);
  expect(glb.clipLabel).toContain("内蔵アニメーション (1)");

  // --- same panel, a model with completely different morph naming ----------
  await switchCharacter(page, "hatsune_miku.yaml");
  await open();

  const miku = await page.evaluate(() => {
    const panel = document.querySelector('.vrm-tune-panel[style*="flex"]');
    if (!panel) return null;
    const names = [...panel.querySelectorAll("span[title]")].map((s) => (s as HTMLElement).title);
    const clipLabel = (panel.children[1] as HTMLElement)?.innerText?.split("\n")[0] || "";
    return { count: names.length, hasJp: names.includes("まばたき"), clipLabel };
  });

  expect(miku, "panel did not open for Miku").not.toBeNull();
  expect(miku?.count).toBeGreaterThan(20);
  expect(miku?.hasJp, "Miku's own morph names should be listed").toBe(true);
  expect(miku?.clipLabel).toContain("なし");
});
