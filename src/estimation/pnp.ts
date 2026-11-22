import { Matrix, SingularValueDecomposition } from 'ml-matrix';
import { Mat3, Vec3, m3mul, m3mulVec, m3transpose, so3Exp, v3add, v3norm, v3scale, v3sub } from '../core/math';
import { Intrinsics } from '../core/camera';

// PnP for relocalisation: world landmark -> pixel. The estimator calls this when
// vision returns after a dropout and the inertial prior is too weak to trust.
// DLT inside RANSAC, then 6-DOF Gauss-Newton on the inliers. Pose is represented
// as { R_wc (camera-to-world), C (camera centre in world) } and converted to
// body pose with the camera extrinsic at the end.

export interface PnpCorrespondence {
  p_w: Vec3;
  u: number;
  v: number;
  id?: number;
}

export interface PnpOptions {
  K: Intrinsics;
  R_bc: Mat3;
  t_bc: Vec3;
  ransacIterations?: number;
  thresholdPx?: number;
  minInliers?: number;
  seed?: number;
}

export interface PnpResult {
  ok: boolean;
  R_wb: Mat3;
  p_wb: Vec3;
  inliers: number[];
  inlierCount: number;
  medianReprojPx: number;
}

interface CamPose {
  R_wc: Mat3;
  C: Vec3;
}

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function det3(m: Matrix): number {
  return (
    m.get(0, 0) * (m.get(1, 1) * m.get(2, 2) - m.get(1, 2) * m.get(2, 1)) -
    m.get(0, 1) * (m.get(1, 0) * m.get(2, 2) - m.get(1, 2) * m.get(2, 0)) +
    m.get(0, 2) * (m.get(1, 0) * m.get(2, 1) - m.get(1, 1) * m.get(2, 0))
  );
}

function projectPose(pose: CamPose, K: Intrinsics, X: Vec3): { u: number; v: number; x: number; y: number; z: number } | null {
  const d = v3sub(X, pose.C);
  const x = pose.R_wc[0] * d[0] + pose.R_wc[3] * d[1] + pose.R_wc[6] * d[2];
  const y = pose.R_wc[1] * d[0] + pose.R_wc[4] * d[1] + pose.R_wc[7] * d[2];
  const z = pose.R_wc[2] * d[0] + pose.R_wc[5] * d[1] + pose.R_wc[8] * d[2];
  if (z <= 0.08) return null;
  return { u: K.fx * (x / z) + K.cx, v: K.fy * (y / z) + K.cy, x, y, z };
}

function dlt(points: PnpCorrespondence[], K: Intrinsics): CamPose | null {
  const n = points.length;
  if (n < 6) return null;
  const A = new Matrix(2 * n, 12);
  for (let i = 0; i < n; i++) {
    const p = points[i];
    const xn = (p.u - K.cx) / K.fx;
    const yn = (p.v - K.cy) / K.fy;
    const X = p.p_w[0], Y = p.p_w[1], Z = p.p_w[2];
    const r1 = [X, Y, Z, 1, 0, 0, 0, 0, -xn * X, -xn * Y, -xn * Z, -xn];
    const r2 = [0, 0, 0, 0, X, Y, Z, 1, -yn * X, -yn * Y, -yn * Z, -yn];
    for (let j = 0; j < 12; j++) {
      A.set(2 * i, j, r1[j]);
      A.set(2 * i + 1, j, r2[j]);
    }
  }
  const svd = new SingularValueDecomposition(A, { autoTranspose: true });
  const V = svd.rightSingularVectors;
  const col = V.columns - 1;
  const M = new Matrix(3, 3);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) M.set(r, c, V.get(r * 4 + c, col));
  const p4 = [V.get(3, col), V.get(7, col), V.get(11, col)];
  const svdM = new SingularValueDecomposition(M, { autoTranspose: true });
  const s = svdM.diagonal;
  if (s[1] < 1e-12) return null;
  let U = svdM.leftSingularVectors;
  let Vm = svdM.rightSingularVectors;
  if (det3(U) < 0) U = U.mulColumn(2, -1);
  if (det3(Vm) < 0) Vm = Vm.mulColumn(2, -1);
  // M = s R_cw, so U V^T is R_cw and its transpose is camera-to-world.
  const R_wc = U.mmul(Vm.transpose()).transpose();
  // s = trace(R_cw^T M) / 3.
  let tr = 0;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) tr += R_wc.get(j, i) * M.get(i, j);
  const scale = tr / 3;
  if (!Number.isFinite(scale) || Math.abs(scale) < 1e-12) return null;
  const tCw: Vec3 = [p4[0] / scale, p4[1] / scale, p4[2] / scale];
  const R_wcOut = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) R_wcOut[r * 3 + c] = R_wc.get(r, c);
  const C = v3scale(m3mulVec(R_wcOut, tCw), -1);
  return { R_wc: R_wcOut, C };
}

function refine(points: PnpCorrespondence[], K: Intrinsics, init: CamPose): CamPose {
  let R_wc: Mat3 = new Float64Array(init.R_wc);
  let C: Vec3 = [...init.C];
  for (let iter = 0; iter < 10; iter++) {
    const H = new Float64Array(36);
    const b = new Float64Array(6);
    for (const p of points) {
      const pr = projectPose({ R_wc, C }, K, p.p_w);
      if (!pr) continue;
      const { x, y, z } = pr;
      const ru = p.u - pr.u;
      const rv = p.v - pr.v;
      const iz = 1 / z;
      const Jp0 = [K.fx * iz, 0, -K.fx * x * iz * iz];
      const Jp1 = [0, K.fy * iz, -K.fy * y * iz * iz];
      // d x_cam / dtheta = [x_cam]x for R_wc <- R_wc Exp(dth).
      const sx = [0, -z, y, z, 0, -x, -y, x, 0];
      const RT = m3transpose(R_wc);
      const J = [new Float64Array(6), new Float64Array(6)];
      for (let row = 0; row < 2; row++) {
        const jp = row === 0 ? Jp0 : Jp1;
        for (let c = 0; c < 3; c++) {
          J[row][c] = jp[0] * sx[c] + jp[1] * sx[3 + c] + jp[2] * sx[6 + c];
          J[row][3 + c] = -(jp[0] * RT[c] + jp[1] * RT[3 + c] + jp[2] * RT[6 + c]);
        }
      }
      for (let i = 0; i < 6; i++) {
        b[i] += J[0][i] * ru + J[1][i] * rv;
        for (let j = 0; j < 6; j++) H[i * 6 + j] += J[0][i] * J[0][j] + J[1][i] * J[1][j];
      }
    }
    for (let i = 0; i < 6; i++) H[i * 6 + i] += 1e-7;
    const dx = solve6(H, b);
    if (!dx) break;
    R_wc = m3mul(R_wc, so3Exp([dx[0], dx[1], dx[2]]));
    C = v3add(C, [dx[3], dx[4], dx[5]]);
    if (Math.abs(dx[0]) + Math.abs(dx[1]) + Math.abs(dx[2]) + v3norm([dx[3], dx[4], dx[5]]) < 1e-8) break;
  }
  return { R_wc, C };
}

function solve6(H: Float64Array, b: Float64Array): Float64Array | null {
  const n = 6;
  const aug = new Float64Array(n * (n + 1));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) aug[i * (n + 1) + j] = H[i * n + j];
    aug[i * (n + 1) + n] = b[i];
  }
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(aug[r * (n + 1) + col]) > Math.abs(aug[piv * (n + 1) + col])) piv = r;
    if (Math.abs(aug[piv * (n + 1) + col]) < 1e-12) return null;
    if (piv !== col) {
      for (let j = 0; j <= n; j++) {
        const tmp = aug[col * (n + 1) + j];
        aug[col * (n + 1) + j] = aug[piv * (n + 1) + j];
        aug[piv * (n + 1) + j] = tmp;
      }
    }
    const p = aug[col * (n + 1) + col];
    for (let j = col; j <= n; j++) aug[col * (n + 1) + j] /= p;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = aug[r * (n + 1) + col];
      for (let j = col; j <= n; j++) aug[r * (n + 1) + j] -= f * aug[col * (n + 1) + j];
    }
  }
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = aug[i * (n + 1) + n];
  return out;
}

export function estimatePnP(corr: PnpCorrespondence[], opts: PnpOptions): PnpResult {
  const K = opts.K;
  const empty: PnpResult = {
    ok: false,
    R_wb: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
    p_wb: [0, 0, 0],
    inliers: [],
    inlierCount: 0,
    medianReprojPx: Infinity,
  };
  if (corr.length < 6) return empty;
  const rand = rng(opts.seed ?? 777);
  const thr = opts.thresholdPx ?? 2.5;
  const iters = opts.ransacIterations ?? 160;
  let bestMask: number[] = [];
  let bestPose: CamPose | null = null;
  let bestErr = Infinity;
  const sample: PnpCorrespondence[] = [];
  for (let it = 0; it < iters; it++) {
    sample.length = 0;
    const used = new Set<number>();
    while (sample.length < 6) {
      const k = Math.floor(rand() * corr.length);
      if (used.has(k)) continue;
      used.add(k);
      sample.push(corr[k]);
    }
    const d = dlt(sample, K);
    if (!d) continue;
    const refined = refine(sample, K, d);
    const mask: number[] = [];
    let errSum = 0;
    for (let i = 0; i < corr.length; i++) {
      const pr = projectPose(refined, K, corr[i].p_w);
      if (!pr) continue;
      const e = Math.hypot(pr.u - corr[i].u, pr.v - corr[i].v);
      if (e < thr) {
        mask.push(i);
        errSum += e;
      }
    }
    if (mask.length > bestMask.length || (mask.length === bestMask.length && errSum / Math.max(1, mask.length) < bestErr)) {
      bestMask = mask;
      bestPose = refined;
      bestErr = errSum / Math.max(1, mask.length);
    }
  }
  if (!bestPose || bestMask.length < (opts.minInliers ?? 8)) return empty;
  let pose = refine(bestMask.map((i) => corr[i]), K, bestPose);
  let mask = bestMask;
  for (let round = 0; round < 2; round++) {
    const next: number[] = [];
    for (let i = 0; i < corr.length; i++) {
      const pr = projectPose(pose, K, corr[i].p_w);
      if (!pr) continue;
      const e = Math.hypot(pr.u - corr[i].u, pr.v - corr[i].v);
      if (e < thr) next.push(i);
    }
    if (next.length < 6) break;
    mask = next;
    pose = refine(mask.map((i) => corr[i]), K, pose);
  }
  const errs = mask
    .map((i) => {
      const pr = projectPose(pose, K, corr[i].p_w);
      return pr ? Math.hypot(pr.u - corr[i].u, pr.v - corr[i].v) : Infinity;
    })
    .sort((a, b) => a - b);
  const medianReprojPx = errs.length ? errs[errs.length >> 1] : Infinity;
  // R_wc = R_wb R_bc and C = p_wb + R_wb t_bc, so invert the extrinsic.
  const R_wb = m3mul(pose.R_wc, m3transpose(opts.R_bc));
  const p_wb = v3sub(pose.C, m3mulVec(R_wb, opts.t_bc));
  return {
    ok: true,
    R_wb,
    p_wb,
    inliers: mask,
    inlierCount: mask.length,
    medianReprojPx,
  };
}
