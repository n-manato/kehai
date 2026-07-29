/* eslint-disable no-underscore-dangle */
// Holds the current co-speech gesture as raw SMPL-X JOINT positions (the exact,
// zero-retarget output of GestureLSM — the ground-truth motion). The SMPLBodyViewer
// reads currentFrame() each rAF and draws an articulated body figure. Singleton
// (mirrors mmd/vrm players) so the audio layer can drive it without prop drilling.

type Clip = { joints: number[][][]; fps: number };

class SMPLBodyPlayer {
  private clip: Clip | null = null;

  private elapsed = 0;

  private speed = 0.8; // calmer playback (co-speech peaks can be fast)

  private gestureSeq = 0;

  /** Poll /motion/clip-smpl/{id} until the raw SMPL-X joints are ready, then play
   *  them forward once at 0.8x, holding the last frame. */
  playGesture(url: string, opts: { tries?: number; intervalMs?: number } = {}) {
    const tries = opts.tries ?? 8;
    const intervalMs = opts.intervalMs ?? 1000;
    const gseq = ++this.gestureSeq;
    let attempt = 0;
    const poll = () => {
      if (gseq !== this.gestureSeq) return;
      const ctrl = new AbortController();
      const to = setTimeout(() => ctrl.abort(), 4000);
      fetch(url, { signal: ctrl.signal })
        .then((r) => {
          if (r.status === 200) return r.json();
          if (r.status === 202) return null;
          throw new Error(String(r.status));
        })
        .then((data) => {
          if (gseq !== this.gestureSeq) return;
          if (data && data.joints) {
            this.clip = { joints: data.joints, fps: data.fps || 30 };
            this.elapsed = 0;
          } else if (++attempt < tries) {
            setTimeout(poll, intervalMs);
          }
        })
        .catch(() => { /* give up quietly */ })
        .finally(() => clearTimeout(to));
    };
    poll();
  }

  update(dt: number) { if (this.clip) this.elapsed += dt * this.speed; }

  /** Current frame's 55 joint positions ([x,y,z] each). Falls back to a neutral
   *  STANDING pose when idle (before the first gesture) so the body is always
   *  visible — otherwise the viewer had nothing to draw and looked empty. */
  currentFrame(): number[][] {
    if (!this.clip) return DEFAULT_POSE;
    const fi = Math.min(this.clip.joints.length - 1, Math.round(this.elapsed * this.clip.fps));
    return this.clip.joints[fi];
  }

  cancelGesture() { this.gestureSeq += 1; }

  stop() { this.clip = null; }

  isActive() { return !!this.clip; }
}

export const smplBodyPlayer = new SMPLBodyPlayer();

// Neutral standing pose (55 SMPL-X joints, from a captured clip's frame 0) shown
// at idle so the body is visible before/without a gesture.
const DEFAULT_POSE: number[][] = [[-0.002,0.936,0.014],[0.042,0.835,0.007],[-0.066,0.835,0.039],[-0.006,1.039,-0.041],[0.087,0.456,-0.063],[-0.205,0.493,0.045],[0.028,1.178,-0.053],[0.002,0.05,-0.062],[-0.279,0.079,0.025],[0.034,1.234,-0.018],[0.098,-0.015,0.032],[-0.323,0.017,0.151],[0.047,1.407,-0.016],[0.096,1.313,-0.016],[-0.001,1.328,-0.006],[0.087,1.541,0.04],[0.236,1.317,-0.038],[-0.113,1.377,0.013],[0.282,1.064,-0.105],[-0.253,1.143,-0.022],[0.332,1.0,0.144],[-0.33,1.126,0.224],[0.071,1.542,0.032],[0.097,1.567,0.116],[0.031,1.567,0.106],[0.335,1.005,0.255],[0.328,0.995,0.286],[0.305,0.991,0.284],[0.34,0.978,0.259],[0.316,0.971,0.28],[0.293,0.969,0.271],[0.333,0.94,0.226],[0.315,0.94,0.239],[0.296,0.943,0.231],[0.338,0.955,0.243],[0.317,0.953,0.265],[0.294,0.957,0.254],[0.315,1.022,0.193],[0.302,1.021,0.223],[0.304,1.011,0.249],[-0.343,1.149,0.33],[-0.329,1.133,0.357],[-0.31,1.121,0.35],[-0.357,1.126,0.336],[-0.329,1.112,0.342],[-0.311,1.106,0.326],[-0.359,1.082,0.31],[-0.338,1.082,0.317],[-0.323,1.084,0.303],[-0.361,1.101,0.323],[-0.332,1.093,0.332],[-0.316,1.094,0.312],[-0.31,1.148,0.271],[-0.302,1.146,0.303],[-0.305,1.148,0.331]];

// SMPL-X 55-joint order (index -> name) as returned by the gesture service.
export const SMPL_NAMES = [
  "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle",
  "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
  "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "Jaw", "L_Eye", "R_Eye",
  "L_Index1", "L_Index2", "L_Index3", "L_Middle1", "L_Middle2", "L_Middle3",
  "L_Pinky1", "L_Pinky2", "L_Pinky3", "L_Ring1", "L_Ring2", "L_Ring3", "L_Thumb1", "L_Thumb2", "L_Thumb3",
  "R_Index1", "R_Index2", "R_Index3", "R_Middle1", "R_Middle2", "R_Middle3",
  "R_Pinky1", "R_Pinky2", "R_Pinky3", "R_Ring1", "R_Ring2", "R_Ring3", "R_Thumb1", "R_Thumb2", "R_Thumb3",
];

// SMPL-X kinematic tree as [parentIndex, childIndex] bone segments to draw.
const N: Record<string, number> = {};
SMPL_NAMES.forEach((n, i) => { N[n] = i; });
export const SMPL_BONES: [number, number][] = ([
  ["Pelvis", "Spine1"], ["Spine1", "Spine2"], ["Spine2", "Spine3"], ["Spine3", "Neck"], ["Neck", "Head"],
  ["Spine3", "L_Collar"], ["L_Collar", "L_Shoulder"], ["L_Shoulder", "L_Elbow"], ["L_Elbow", "L_Wrist"],
  ["Spine3", "R_Collar"], ["R_Collar", "R_Shoulder"], ["R_Shoulder", "R_Elbow"], ["R_Elbow", "R_Wrist"],
  ["Pelvis", "L_Hip"], ["L_Hip", "L_Knee"], ["L_Knee", "L_Ankle"], ["L_Ankle", "L_Foot"],
  ["Pelvis", "R_Hip"], ["R_Hip", "R_Knee"], ["R_Knee", "R_Ankle"], ["R_Ankle", "R_Foot"],
  // fingers (thin) — show hand gestures
  ["L_Wrist", "L_Index1"], ["L_Index1", "L_Index2"], ["L_Index2", "L_Index3"],
  ["L_Wrist", "L_Middle1"], ["L_Middle1", "L_Middle2"], ["L_Middle2", "L_Middle3"],
  ["L_Wrist", "L_Ring1"], ["L_Ring1", "L_Ring2"], ["L_Ring2", "L_Ring3"],
  ["L_Wrist", "L_Pinky1"], ["L_Pinky1", "L_Pinky2"], ["L_Pinky2", "L_Pinky3"],
  ["L_Wrist", "L_Thumb1"], ["L_Thumb1", "L_Thumb2"], ["L_Thumb2", "L_Thumb3"],
  ["R_Wrist", "R_Index1"], ["R_Index1", "R_Index2"], ["R_Index2", "R_Index3"],
  ["R_Wrist", "R_Middle1"], ["R_Middle1", "R_Middle2"], ["R_Middle2", "R_Middle3"],
  ["R_Wrist", "R_Ring1"], ["R_Ring1", "R_Ring2"], ["R_Ring2", "R_Ring3"],
  ["R_Wrist", "R_Pinky1"], ["R_Pinky1", "R_Pinky2"], ["R_Pinky2", "R_Pinky3"],
  ["R_Wrist", "R_Thumb1"], ["R_Thumb1", "R_Thumb2"], ["R_Thumb2", "R_Thumb3"],
] as [string, string][]).map(([a, b]) => [N[a], N[b]] as [number, number]);
