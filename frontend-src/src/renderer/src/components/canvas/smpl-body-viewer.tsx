/* eslint-disable no-underscore-dangle */
import { memo, useEffect, useRef } from "react";
import * as THREE from "three";

/**
 * Raw SMPL-X BODY viewer — renders the actual GestureLSM output as the SMPL-X
 * body MESH (the skin surface the paper/demos use), ZERO retarget. Loads a
 * pre-baked co-speech clip (vertices per frame + faces, served as compact binary
 * from /Live2D Characters/vrm/smpl_mesh_*) and loops it. This is the ground-truth
 * motion on the real human body — the reference to retarget other models against.
 * (Per-frame vertices are ~40MB/clip over the gateway, too heavy to stream live;
 * a baked loop shows the real body smoothly. Live-driven needs pose-only LBS.)
 * Used when the model URL is the "smplx-body" sentinel (see App.tsx).
 */
interface Props { url: string }

const BASE = "/Live2D%20Characters/vrm";

export const SMPLBodyViewer = memo((_props: Props): JSX.Element => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;
    let raf = 0;
    let disposed = false;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      32,
      Math.max(container.clientWidth, 1) / Math.max(container.clientHeight, 1),
      0.01,
      100,
    );

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.style.cssText = "width:100%;height:100%;display:block";
    container.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 1.1));
    const dl = new THREE.DirectionalLight(0xffffff, 1.4);
    dl.position.set(0.6, 1.4, 1.6); scene.add(dl);
    const dl2 = new THREE.DirectionalLight(0x88aaff, 0.5);
    dl2.position.set(-0.8, 0.4, -1); scene.add(dl2);

    let geo: THREE.BufferGeometry | null = null;
    let mesh: THREE.Mesh | null = null;
    let verts: Float32Array | null = null;
    let nverts = 0; let nframes = 0; let fps = 15;
    const clock = new THREE.Clock();
    let elapsed = 0;

    Promise.all([
      fetch(`${BASE}/smpl_mesh_meta.json`).then((r) => r.json()),
      fetch(`${BASE}/smpl_mesh_verts.f32`).then((r) => r.arrayBuffer()),
      fetch(`${BASE}/smpl_mesh_faces.i32`).then((r) => r.arrayBuffer()),
    ]).then(([meta, vb, fb]) => {
      if (disposed) return;
      verts = new Float32Array(vb);
      const faces = new Int32Array(fb);
      nverts = meta.nverts; nframes = meta.nframes; fps = meta.fps || 15;
      geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(verts.slice(0, nverts * 3), 3));
      geo.setIndex(new THREE.BufferAttribute(new Uint32Array(faces), 1));
      geo.computeVertexNormals();
      const mat = new THREE.MeshStandardMaterial({ color: 0xc8b8a8, roughness: 0.85, metalness: 0.0 });
      mesh = new THREE.Mesh(geo, mat); scene.add(mesh);
      // frame the body
      geo.computeBoundingBox();
      const ctr = geo.boundingBox!.getCenter(new THREE.Vector3());
      camera.position.set(0, ctr.y, 2.4);
      camera.lookAt(ctr.x, ctr.y, ctr.z);
      (window as unknown as { __smpl?: boolean }).__smpl = true;
    }).catch((e) => console.error("[SMPL] mesh load failed:", e));

    const posAttr = () => geo?.getAttribute("position") as THREE.BufferAttribute | undefined;

    const animate = () => {
      raf = requestAnimationFrame(animate);
      if (verts && geo && nframes > 0) {
        elapsed += clock.getDelta();
        const fi = Math.floor(elapsed * fps) % nframes;
        const off = fi * nverts * 3;
        const attr = posAttr();
        if (attr) {
          (attr.array as Float32Array).set(verts.subarray(off, off + nverts * 3));
          attr.needsUpdate = true;
          geo.computeVertexNormals();
        }
      }
      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      camera.aspect = Math.max(container.clientWidth, 1) / Math.max(container.clientHeight, 1);
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener("resize", onResize);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      delete (window as unknown as { __smpl?: boolean }).__smpl;
      geo?.dispose();
      (mesh?.material as THREE.Material | undefined)?.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === container) container.removeChild(renderer.domElement);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "100%", overflow: "hidden", position: "relative" }}
    />
  );
});

SMPLBodyViewer.displayName = "SMPLBodyViewer";
