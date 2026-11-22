import { describe, expect, it } from 'vitest';
import { Gaussian, Mat3, Vec3, m3TmulVec, m3mul, m3mulVec, mulberry32, so3Exp, v3add, v3normalize, v3scale, v3sub, v3norm } from '../src/core/math';
import { defaultExtrinsics, defaultIntrinsics, project, bodyToCamera } from '../src/core/camera';
import { estimateRelativePose, Correspondence, relativeRotationAngle, translationDirectionSecondFrame } from '../src/estimation/relpose';
import { generateScene, renderScene } from '../src/vision/scene';
import { describe as describeFrame, trackFrame } from '../src/vision/tracker';

describe('relative pose', () => {
  it('recovers rotation and translation direction from exact bearings', () => {
    const ext = defaultExtrinsics();
    const K = defaultIntrinsics(320, 240);
    const R_bc = ext.R_bc;
    // Camera 1 identity body pose; camera 2 known relative motion.
    const R_c1 = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const R_c2 = so3Exp([0.02, 0.19, -0.05]);
    const t_c2 = [0.42, 0.07, -0.03] as Vec3;
    const R_bc_T = new Float64Array([R_bc[0], R_bc[3], R_bc[6], R_bc[1], R_bc[4], R_bc[7], R_bc[2], R_bc[5], R_bc[8]]);
    const rng = new Gaussian(mulberry32(31));
    const corr: Correspondence[] = [];
    for (let i = 0; i < 80; i++) {
      // Random world point in front of camera 1 (camera frame directly).
      const z = 2.5 + rng.next() * 2;
      const x = (rng.next() - 0.5) * 2 * z * 0.5;
      const y = (rng.next() - 0.5) * 2 * z * 0.4;
      const p1 = [x, y, z] as Vec3;
      // Build correct camera-2 coordinates: X2 = R21 X1 + t21 with R21 = R_c2 (camera-frame)
      const X2 = v3add(m3mulVec(so3Exp([0.02, 0.19, -0.05]), p1), t_c2);
      const a = project(p1, K);
      const b = project(X2, K);
      corr.push({ u1: a.u, v1: a.v, u2: b.u, v2: b.v });
    }
    const rel = estimateRelativePose(corr, { fx: K.fx, fy: K.fy, cx: K.cx, cy: K.cy, thresholdPx: 2, ransacIterations: 200 });
    expect(rel.ok).toBe(true);
    const Rtrue = so3Exp([0.02, 0.19, -0.05]);
    const err = m3mul(new Float64Array([Rtrue[0],Rtrue[3],Rtrue[6],Rtrue[1],Rtrue[4],Rtrue[7],Rtrue[2],Rtrue[5],Rtrue[8]]), rel.R);
    expect(v3norm(v3sub(rel.t, v3normalize(t_c2)))).toBeLessThan(0.05);
    expect(relativeRotationAngle(rel.R)).toBeCloseTo(relativeRotationAngle(Rtrue), 1);
  });
});

describe('tracker + renderer', () => {
  it('tracks flow between two rendered frames', () => {
    const scene = generateScene(3);
    const K = defaultIntrinsics(320, 240);
    const ext = defaultExtrinsics();
    const R = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const g = new Gaussian(mulberry32(9));
    const a = renderScene(scene, { R_wb: R, p_wb: [0.2, 0.1, 0], R_bc: ext.R_bc, t_bc: ext.t_bc }, K, { exposure: 1, pixelNoise: 0, gaussian: g });
    const b = renderScene(scene, { R_wb: so3Exp([0, 0.03, 0]), p_wb: [0.26, 0.1, 0], R_bc: ext.R_bc, t_bc: ext.t_bc }, K, { exposure: 1, pixelNoise: 0, gaussian: g });
    const fa = describeFrame(a.gray, K.width, K.height);
    const fb = describeFrame(b.gray, K.width, K.height);
    expect(fa.keypoints.length).toBeGreaterThan(30);
    const tracks = fa.keypoints.slice(0, 60).map((kp, i) => ({ id: i, u: kp.u, v: kp.v, prevU: kp.u, prevV: kp.v, bornAt: 0, age: 0, confidence: 1, stale: false }));
    const res = trackFrame(fa, fb, tracks, null, null, { minTracks: 20, maxTracks: 120, fbTol: 1.5, nccMin: 0.6 }, 1);
    expect(res.tracks.length).toBeGreaterThan(20);
    // Tracked points should move consistently with the yaw/pitch translation.
    const moved = res.tracks.filter((t) => t.age > 0);
    expect(moved.length).toBeGreaterThan(10);
  });
});
