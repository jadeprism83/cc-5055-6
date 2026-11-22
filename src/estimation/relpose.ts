import { Matrix, SingularValueDecomposition } from 'ml-matrix';
import { Mat3, Vec3, m3mul, m3mulVec, m3transpose, so3Exp, v3cross, v3dot, v3norm, v3normalize, v3scale, v3sub } from '../core/math';

// Two-view relative pose from calibrated bearings using the 8-point algorithm
// inside RANSAC. Returns the rotation and unit translation direction between
// camera frames: x2 = R x1 + t, with |t| not observable from bearings alone.

export interface Correspondence {
  u1: number;
  v1: number;
  u2: number;
  v2: number;
  id?: number;
}


// ---------------------------------------------------------------------------
// Rotation-first relative pose. With a narrow baseline at 20 Hz, the 8-point
// essential matrix is ill-conditioned: pure-rotation data has no translation to
// recover, and a near-forward baseline lets rotation error leak into t. Instead
// we estimate R directly from paired bearings (Wahba), then solve t linearly
// from the epipolar constraint with R fixed. This is far better conditioned for
// the handheld/near-static segments in this app.

function bearings(corr: Correspondence[], K: { fx: number; fy: number; cx: number; cy: number }): { b1: Vec3; b2: Vec3 }[] {
  return corr.map((c) => ({
    b1: v3normalize([(c.u1 - K.cx) / K.fx, (c.v1 - K.cy) / K.fy, 1]),
    b2: v3normalize([(c.u2 - K.cx) / K.fx, (c.v2 - K.cy) / K.fy, 1]),
  }));
}

/** Rotation minimising sum |R b1 - b2|, via Kabsch (SVD). */
function wahbaRotation(pairs: Array<{ b1: Vec3; b2: Vec3 }>, weights?: number[]): Mat3 | null {
  if (pairs.length < 3) return null;
  const H = new Float64Array(9);
  for (let i = 0; i < pairs.length; i++) {
    const w = weights ? weights[i] : 1;
    if (w <= 0) continue;
    const a = pairs[i].b1;
    const b = pairs[i].b2;
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) H[r * 3 + c] += w * a[r] * b[c];
  }
  const M = new Matrix(3, 3);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) M.set(r, c, H[r * 3 + c]);
  const svd = new SingularValueDecomposition(M, { autoTranspose: true });
  let U = svd.leftSingularVectors;
  let V = svd.rightSingularVectors;
  if (det3(U) < 0) U = U.mulColumn(2, -1);
  if (det3(V) < 0) V = V.mulColumn(2, -1);
  const s3 = Math.sign(det3(V) * det3(U)) || 1;
  const D = Matrix.diag([1, 1, s3]);
  const Rm = V.mmul(D).mmul(U.transpose());
  const R = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) R[r * 3 + c] = Rm.get(r, c);
  return R;
}

/** Linear translation direction from x2^T [t] R x1 = 0 with R fixed. */
function translationDirection(pairs: Array<{ b1: Vec3; b2: Vec3 }>, R: Mat3): Vec3 | null {
  if (pairs.length < 2) return null;
  const H = new Float64Array(9);
  for (const p of pairs) {
    const Rb1 = m3mulVec(R, p.b1);
    const a = v3cross(p.b2, Rb1); // coefficients of t
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) H[r * 3 + c] += a[r] * a[c];
  }
  // Smallest eigenvector of the 3x3 symmetric matrix. Use Jacobi.
  const ev = smallestEigenvector3(H);
  if (!ev) return null;
  return v3normalize(ev);
}

function smallestEigenvector3(A: Float64Array): Vec3 | null {
  // Cyclic Jacobi on a symmetric 3x3.
  const a = new Float64Array(A);
  const V = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  for (let sweep = 0; sweep < 30; sweep++) {
    let off = Math.abs(a[1]) + Math.abs(a[2]) + Math.abs(a[5]);
    if (off < 1e-14) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]] as const) {
      const apq = a[p * 3 + q];
      if (Math.abs(apq) < 1e-18) continue;
      const app = a[p * 3 + p], aqq = a[q * 3 + q];
      const theta = 0.5 * Math.atan2(2 * apq, aqq - app);
      const c = Math.cos(theta), s = Math.sin(theta);
      // Rotate rows/cols p,q of a and accumulate V.
      for (let k = 0; k < 3; k++) {
        const akp = a[k * 3 + p], akq = a[k * 3 + q];
        a[k * 3 + p] = c * akp - s * akq;
        a[k * 3 + q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p * 3 + k], aqk = a[q * 3 + k];
        a[p * 3 + k] = c * apk - s * aqk;
        a[q * 3 + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = V[k * 3 + p], vkq = V[k * 3 + q];
        V[k * 3 + p] = c * vkp - s * vkq;
        V[k * 3 + q] = s * vkp + c * vkq;
      }
    }
  }
  let minI = 0;
  for (let i = 1; i < 3; i++) if (a[i * 3 + i] < a[minI * 3 + minI]) minI = i;
  return [V[minI], V[3 + minI], V[6 + minI]];
}

export interface RotationFirstPose {
  ok: boolean;
  R: Mat3;
  t: Vec3;
  inlierCount: number;
  inliers: boolean[];
  medianRotationResidualRad: number;
  parallax: number;
  medianErrorPx: number;
}

/**
 * Robust rotation-first relative pose. RANSAC over rotation hypotheses from
 * minimal triples, then a linear translation solve on the inliers. Returns the
 * camera-frame relation x2 = R x1 + t (t unit).
 */
export function estimateRotationFirst(
  corr: Correspondence[],
  K: { fx: number; fy: number; cx: number; cy: number },
  opts: { ransacIterations?: number; inlierRad?: number; minInliers?: number; seed?: number } = {},
): RotationFirstPose {
  const n = corr.length;
  const empty: RotationFirstPose = {
    ok: false,
    R: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
    t: [0, 0, 1],
    inlierCount: 0,
    inliers: new Array(n).fill(false),
    medianRotationResidualRad: Infinity,
    parallax: 0,
    medianErrorPx: Infinity,
  };
  if (n < 4) return empty;
  const pairs = bearings(corr, K);
  const rand = rng(opts.seed ?? 2468);
  const inlierRad = opts.inlierRad ?? 0.0045; // ~0.26 deg
  const iters = opts.ransacIterations ?? 200;
  let bestMask: boolean[] = [];
  let bestR: Mat3 | null = null;
  let bestCount = -1;
  let bestSum = Infinity;
  for (let it = 0; it < iters; it++) {
    const idx = [Math.floor(rand() * n)];
    while (idx.length < 3) {
      const k = Math.floor(rand() * n);
      if (!idx.includes(k)) idx.push(k);
    }
    const R = wahbaRotation(idx.map((i) => pairs[i]));
    if (!R) continue;
    const mask = new Array<boolean>(n).fill(false);
    let count = 0;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const pred = m3mulVec(R, pairs[i].b1);
      const c = Math.min(1, Math.max(-1, v3dot(pred, pairs[i].b2)));
      const e = Math.acos(c);
      if (e < inlierRad) {
        mask[i] = true;
        count++;
        sum += e;
      }
    }
    if (count > bestCount || (count === bestCount && sum < bestSum)) {
      bestCount = count;
      bestSum = sum;
      bestMask = mask;
      bestR = R;
    }
  }
  if (!bestR || bestCount < (opts.minInliers ?? Math.min(10, n))) return empty;
  // Refit rotation and translation on inliers, iterating twice.
  let R = bestR;
  let mask = bestMask;
  let t: Vec3 = [0, 0, 1];
  for (let round = 0; round < 3; round++) {
    const inl = pairs.filter((_, i) => mask[i]);
    const refit = wahbaRotation(inl);
    if (!refit) break;
    R = refit;
    const td = translationDirection(inl, R);
    if (td) t = td;
    const next = new Array<boolean>(n).fill(false);
    let count = 0;
    for (let i = 0; i < n; i++) {
      const pred = m3mulVec(R, pairs[i].b1);
      const c = Math.min(1, Math.max(-1, v3dot(pred, pairs[i].b2)));
      if (Math.acos(c) < inlierRad) {
        next[i] = true;
        count++;
      }
    }
    if (count < (opts.minInliers ?? Math.min(10, n))) break;
    mask = next;
  }
  // Triangulation sign test fixes the direction of t.
  let posDepth = 0;
  let negDepth = 0;
  const inliersIdx = pairs.map((_, i) => i).filter((i) => mask[i]);
  for (const i of inliersIdx) {
    const d = triangulateDepth(R, t, { x: pairs[i].b1[0] / pairs[i].b1[2], y: pairs[i].b1[1] / pairs[i].b1[2] }, { x: pairs[i].b2[0] / pairs[i].b2[2], y: pairs[i].b2[1] / pairs[i].b2[2] });
    if (d.d1 > 0 && d.d2 > 0) posDepth++;
    else negDepth++;
  }
  if (negDepth > posDepth) t = v3scale(t, -1);
  // Parallax and angular residual.
  let parSum = 0;
  let rotSum = 0;
  let parN = 0;
  for (const i of inliersIdx) {
    const pred = m3mulVec(R, pairs[i].b1);
    rotSum += Math.acos(Math.min(1, Math.max(-1, v3dot(pred, pairs[i].b2))));
    const d = triangulateDepth(R, t, { x: pairs[i].b1[0] / pairs[i].b1[2], y: pairs[i].b1[1] / pairs[i].b1[2] }, { x: pairs[i].b2[0] / pairs[i].b2[2], y: pairs[i].b2[1] / pairs[i].b2[2] });
    if (d.d1 > 0 && d.d2 > 0) {
      // Parallax at the landmark: angle between ray 1 and ray 2 expressed in
      // the first camera, i.e. between b1 and R^T b2.
      const b2in1 = m3mulVec(m3transpose(R), pairs[i].b2);
      const c = Math.min(1, Math.max(-1, v3dot(pairs[i].b1, b2in1)));
      parSum += Math.acos(c);
      parN++;
    }
  }
  const medianRotationResidualRad = inliersIdx.length ? rotSum / inliersIdx.length : Infinity;
  return {
    ok: true,
    R,
    t,
    inlierCount: inliersIdx.length,
    inliers: mask,
    medianRotationResidualRad,
    parallax: parN ? parSum / parN : 0,
    medianErrorPx: (medianRotationResidualRad * (K.fx + K.fy)) / 2,
  };
}

export interface RelPoseOptions {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  ransacIterations?: number;
  thresholdPx?: number;
  minInliers?: number;
  seed?: number;
}

export interface RelPose {
  ok: boolean;
  R: Mat3;
  /** Unit translation in the first camera frame. */
  t: Vec3;
  inliers: number[];
  inlierCount: number;
  /** Median Sampson distance in pixels. */
  medianErrorPx: number;
  /** Median triangulated parallax, radians. */
  parallax: number;
  E: Mat3;
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

function normalizePoints(pts: Array<{ x: number; y: number }>): {
  norm: Array<{ x: number; y: number }>;
  T: Mat3;
} {
  let mx = 0, my = 0;
  for (const p of pts) {
    mx += p.x;
    my += p.y;
  }
  mx /= pts.length;
  my /= pts.length;
  let meanDist = 0;
  for (const p of pts) meanDist += Math.hypot(p.x - mx, p.y - my);
  meanDist /= pts.length;
  const s = meanDist > 1e-12 ? Math.SQRT2 / meanDist : 1;
  const T = new Float64Array([s, 0, -s * mx, 0, s, -s * my, 0, 0, 1]);
  const norm = pts.map((p) => ({ x: s * (p.x - mx), y: s * (p.y - my) }));
  return { norm, T };
}

function eightPoint(x1: Array<{ x: number; y: number }>, x2: Array<{ x: number; y: number }>): Mat3 | null {
  const n1 = normalizePoints(x1);
  const n2 = normalizePoints(x2);
  const n = x1.length;
  const A = new Matrix(n, 9);
  for (let i = 0; i < n; i++) {
    const a = n1.norm[i];
    const b = n2.norm[i];
    A.set(i, 0, b.x * a.x);
    A.set(i, 1, b.x * a.y);
    A.set(i, 2, b.x);
    A.set(i, 3, b.y * a.x);
    A.set(i, 4, b.y * a.y);
    A.set(i, 5, b.y);
    A.set(i, 6, a.x);
    A.set(i, 7, a.y);
    A.set(i, 8, 1);
  }
  const svd = new SingularValueDecomposition(A, { autoTranspose: true });
  const V = svd.rightSingularVectors;
  const last = V.columns - 1;
  const e = new Float64Array(9);
  for (let i = 0; i < 9; i++) e[i] = V.get(i, last);
  const E0 = new Matrix(3, 3);
  for (let i = 0; i < 9; i++) E0.set(Math.floor(i / 3), i % 3, e[i]);
  const svdE = new SingularValueDecomposition(E0, { autoTranspose: true });
  const U = svdE.leftSingularVectors;
  const Vt = svdE.rightSingularVectors.transpose();
  const s = svdE.diagonal;
  const s1 = (s[0] + s[1]) / 2;
  const Eclean = U.mmul(Matrix.diag([s1, s1, 0])).mmul(Vt);
  // Denormalise: E = T2^T E_norm T1.
  const T1 = new Matrix(3, 3);
  const T2 = new Matrix(3, 3);
  for (let i = 0; i < 9; i++) {
    T1.set(Math.floor(i / 3), i % 3, n1.T[i]);
    T2.set(Math.floor(i / 3), i % 3, n2.T[i]);
  }
  const E = T2.transpose().mmul(Eclean).mmul(T1);
  const out = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out[r * 3 + c] = E.get(r, c);
  return out;
}

function sampson(E: Mat3, p1: { x: number; y: number }, p2: { x: number; y: number }): number {
  const x1 = [p1.x, p1.y, 1];
  const x2 = [p2.x, p2.y, 1];
  const Ex1 = m3mulVec(E, x1 as Vec3);
  const Etx2 = m3mulVec(new Float64Array([E[0], E[3], E[6], E[1], E[4], E[7], E[2], E[5], E[8]]), x2 as Vec3);
  const num = x2[0] * Ex1[0] + x2[1] * Ex1[1] + x2[2] * Ex1[2];
  const den = Ex1[0] * Ex1[0] + Ex1[1] * Ex1[1] + Etx2[0] * Etx2[0] + Etx2[1] * Etx2[1];
  return den < 1e-18 ? 1e9 : (num * num) / den;
}

function decompose(E: Mat3): Array<{ R: Mat3; t: Vec3 }> {
  const M = new Matrix(3, 3);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) M.set(r, c, E[r * 3 + c]);
  const svd = new SingularValueDecomposition(M, { autoTranspose: true });
  let U = svd.leftSingularVectors.clone();
  let V = svd.rightSingularVectors.clone();
  if (det3(U) < 0) U = U.mulColumn(2, -1);
  if (det3(V) < 0) V = V.mulColumn(2, -1);
  const Vt = V.transpose();
  const W = new Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]]);
  const Wt = W.transpose();
  const cands: Array<{ R: Mat3; t: Vec3 }> = [];
  // Standard decomposition of E = U diag(1,1,0) V^T.
  const Rs = [U.mmul(W).mmul(Vt), U.mmul(Wt).mmul(Vt)];
  const tcol = [U.get(0, 2), U.get(1, 2), U.get(2, 2)] as Vec3;
  for (const Rm of Rs) {
    for (const sgn of [1, -1]) {
      const R = new Float64Array(9);
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) R[r * 3 + c] = Rm.get(r, c);
      cands.push({ R, t: v3scale(tcol, sgn) });
    }
  }
  return cands;
}

function det3(m: Matrix): number {
  return (
    m.get(0, 0) * (m.get(1, 1) * m.get(2, 2) - m.get(1, 2) * m.get(2, 1)) -
    m.get(0, 1) * (m.get(1, 0) * m.get(2, 2) - m.get(1, 2) * m.get(2, 0)) +
    m.get(0, 2) * (m.get(1, 0) * m.get(2, 1) - m.get(1, 1) * m.get(2, 0))
  );
}

function triangulateDepth(R: Mat3, t: Vec3, p1: { x: number; y: number }, p2: { x: number; y: number }): { d1: number; d2: number } {
  // Ray from cam1 origin along x1; ray from cam2 centre C2=-R^T t.
  const x1 = v3normalize([p1.x, p1.y, 1]);
  const x2 = v3normalize([p2.x, p2.y, 1]);
  const C2 = m3mulVec(new Float64Array([R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]]), v3scale(t, -1));
  const d2w = m3mulVec(R, x2);
  const r = v3scale(C2, -1);
  const a = 1;
  const b = x1[0] * d2w[0] + x1[1] * d2w[1] + x1[2] * d2w[2];
  const c = 1;
  const d = x1[0] * r[0] + x1[1] * r[1] + x1[2] * r[2];
  const e = d2w[0] * r[0] + d2w[1] * r[1] + d2w[2] * r[2];
  const den = a * c - b * b;
  if (Math.abs(den) < 1e-12) return { d1: -1, d2: -1 };
  const s1 = (b * e - c * d) / den;
  const s2 = (a * e - b * d) / den;
  return { d1: s1, d2: s2 };
}

export function estimateRelativePose(corr: Correspondence[], opts: RelPoseOptions): RelPose {
  const n = corr.length;
  const fx = opts.fx, fy = opts.fy, cx = opts.cx, cy = opts.cy;
  const p1 = corr.map((c) => ({ x: (c.u1 - cx) / fx, y: (c.v1 - cy) / fy }));
  const p2 = corr.map((c) => ({ x: (c.u2 - cx) / fx, y: (c.v2 - cy) / fy }));
  const empty: RelPose = {
    ok: false,
    R: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
    t: [0, 0, 1],
    inliers: [],
    inlierCount: 0,
    medianErrorPx: Infinity,
    parallax: 0,
    E: new Float64Array(9),
  };
  if (n < 8) return empty;
  const rand = rng(opts.seed ?? 12345);
  // Sampson distances are in normalised units; thresholds are normalised too.
  const thr = (opts.thresholdPx ?? 1.5) / ((fx + fy) / 2);
  const iters = opts.ransacIterations ?? 220;
  let bestMask: number[] = [];
  let bestScore = -1;
  let bestE: Mat3 | null = null;
  for (let it = 0; it < iters; it++) {
    const idx: number[] = [];
    while (idx.length < 8) {
      const k = Math.floor(rand() * n);
      if (!idx.includes(k)) idx.push(k);
    }
    const E = eightPoint(idx.map((i) => p1[i]), idx.map((i) => p2[i]));
    if (!E) continue;
    const mask: number[] = [];
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const s = sampson(E, p1[i], p2[i]);
      if (Math.sqrt(s) < thr) {
        mask.push(i);
        sum += s;
      }
    }
    if (mask.length > bestScore) {
      bestScore = mask.length;
      bestMask = mask;
      bestE = E;
    }
  }
  if (bestMask.length < 8) return empty;
  // Re-fit on all inliers, then one more inlier pass.
  let E = eightPoint(bestMask.map((i) => p1[i]), bestMask.map((i) => p2[i]));
  if (!E) return empty;
  let mask = bestMask;
  for (let round = 0; round < 2; round++) {
    const next: number[] = [];
    for (let i = 0; i < n; i++) {
      const s = sampson(E, p1[i], p2[i]);
      if (Math.sqrt(s) < thr) next.push(i);
    }
    if (next.length < 8) break;
    mask = next;
    const refit = eightPoint(mask.map((i) => p1[i]), mask.map((i) => p2[i]));
    if (refit) E = refit;
  }
  const errors = mask.map((i) => Math.sqrt(sampson(E, p1[i], p2[i])) * ((fx + fy) / 2));
  errors.sort((a, b) => a - b);
  const medianErrorPx = errors.length ? errors[errors.length >> 1] : Infinity;

  const cands = decompose(E);
  let best: { R: Mat3; t: Vec3; good: number; parallax: number } | null = null;
  for (const c of cands) {
    let good = 0;
    let parSum = 0;
    let parCount = 0;
    for (const i of mask) {
      const d = triangulateDepth(c.R, c.t, p1[i], p2[i]);
      if (d.d1 > 0.01 && d.d2 > 0.01) {
        good++;
        const x1 = v3normalize([p1[i].x, p1[i].y, 1]);
        const x2w = m3mulVec(c.R, v3normalize([p2[i].x, p2[i].y, 1]));
        const cosang = Math.min(1, Math.max(-1, x1[0] * x2w[0] + x1[1] * x2w[1] + x1[2] * x2w[2]));
        parSum += Math.acos(cosang);
        parCount++;
      }
    }
    if (!best || good > best.good) best = { R: c.R, t: v3normalize(c.t), good, parallax: parCount ? parSum / parCount : 0 };
  }
  if (!best || best.good < (opts.minInliers ?? 10)) return { ...empty, E };
  return {
    ok: true,
    R: best.R,
    t: best.t,
    inliers: mask,
    inlierCount: mask.length,
    medianErrorPx,
    parallax: best.parallax,
    E,
  };
}

export function relativeRotationAngle(R: Mat3): number {
  const c = (R[0] + R[4] + R[8] - 1) / 2;
  return Math.acos(Math.min(1, Math.max(-1, c)));
}

// Relative translation direction expressed in the second camera frame.
// x2 = R21 x1 + t21, so the second camera sits at C2 = -R^T t in frame 1.
export function translationDirectionSecondFrame(R: Mat3, t: Vec3): Vec3 {
  // Direction from camera1 to camera2 in frame 2 is R * (-t) / |t|.
  return v3normalize(m3mulVec(R, v3scale(t, -1)));
}

export function rotationFromTwoVectors(a: Vec3, b: Vec3): Mat3 {
  // Small utility for tests: rotation taking a to b.
  const u = v3normalize(a);
  const w = v3normalize(b);
  const v = v3cross(u, w);
  const c = Math.min(1, Math.max(-1, u[0] * w[0] + u[1] * w[1] + u[2] * w[2]));
  const s = v3norm(v);
  if (s < 1e-12) {
    if (c > 0) return new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    return new Float64Array([-1, 0, 0, 0, -1, 0, 0, 0, 1]);
  }
  const axis = v3scale(v, 1 / s);
  const ang = Math.atan2(s, c);
  const k = axis;
  const K = new Float64Array([0, -k[2], k[1], k[2], 0, -k[0], -k[1], k[0], 0]);
  const K2 = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let cc = 0; cc < 3; cc++) K2[r * 3 + cc] = K[r * 3] * K[cc] + K[r * 3 + 1] * K[3 + cc] + K[r * 3 + 2] * K[6 + cc];
  const I = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const out = new Float64Array(9);
  for (let i = 0; i < 9; i++) out[i] = I[i] + Math.sin(ang) * K[i] + (1 - Math.cos(ang)) * K2[i];
  return out;
}

/**
 * Two-view bundle refinement of the rotation-first solution. The Wahba rotation
 * from paired bearings is biased by inter-frame translation: on a handheld rig
 * the baseline is not negligible, so parallax leaks into R. Here we minimise the
 * symmetric reprojection error over R, the unit translation direction and one
 * inverse depth per correspondence, which separates the two effects.
 *
 * Parametrisation: R <- Exp(dr) R0, t <- normalize(t0 + dt). Depth is solved
 * linearly inside each iteration by minimising the two reprojection residuals,
 * which keeps the normal equations small (6 unknowns).
 */
export function debugTwoViewCosts(
  corr: Correspondence[],
  K: { fx: number; fy: number; cx: number; cy: number },
  R: Mat3, t: Vec3,
): { cost: number; depths: number[] } {
  const x1 = corr.map((c) => [(c.u1 - K.cx) / K.fx, (c.v1 - K.cy) / K.fy]);
  const x2 = corr.map((c) => [(c.u2 - K.cx) / K.fx, (c.v2 - K.cy) / K.fy]);
  const n = corr.length;
  const d0 = new Float64Array(n);
  for (let i = 0; i < n; i++) d0[i] = initialDepth(R, t, x1[i], x2[i]);
  const f = 0.5 * (K.fx + K.fy);
  return { cost: twoViewCost(R, t, x1, x2, d0, f), depths: Array.from(d0) };
}

function twoViewCost(
  R: Mat3, t: Vec3, x1: number[][], x2: number[][], depth: Float64Array,
  f: number,
): number {
  let sum = 0;
  let count = 0;
  for (let i = 0; i < x1.length; i++) {
    const d = depth[i];
    if (!(d > 0.1) || !isFinite(d)) continue;
    const P1: Vec3 = [d * x1[i][0], d * x1[i][1], d];
    const P2 = m3add3(m3mulVec(R, P1), t);
    if (P2[2] <= 1e-4) continue;
    const eu = P2[0] / P2[2] - x2[i][0];
    const ev = P2[1] / P2[2] - x2[i][1];
    // Truncated quadratic (Huber-like) at 4 px, so a handful of bad depths
    // cannot dominate the acceptance test.
    const e2 = f * f * (eu * eu + ev * ev);
    sum += Math.min(e2, 16);
    count++;
  }
  return count ? sum / count : Infinity;
}

export function refineTwoViewBundle(
  corr: Correspondence[],
  K: { fx: number; fy: number; cx: number; cy: number },
  init: { R: Mat3; t: Vec3 },
  iterations = 12,
): { R: Mat3; t: Vec3; rmsPx: number; medianPx: number } {
  const n = corr.length;
  if (n < 4) return { R: init.R, t: init.t, rmsPx: Infinity, medianPx: Infinity };
  const x1 = corr.map((c) => [(c.u1 - K.cx) / K.fx, (c.v1 - K.cy) / K.fy]);
  const x2 = corr.map((c) => [(c.u2 - K.cx) / K.fx, (c.v2 - K.cy) / K.fy]);
  const f = 0.5 * (K.fx + K.fy);
  let R: Mat3 = new Float64Array(init.R);
  let t = v3normalize(init.t);
  // Initial depths from the current model, one per point (in frame-1 depth).
  let depth = new Float64Array(n);
  for (let i = 0; i < n; i++) depth[i] = initialDepth(R, t, x1[i], x2[i]);

  let rmsPx = Infinity;
  let medianPx = Infinity;
  for (let it = 0; it < iterations; it++) {
    // Gauss-Newton on [dr(3), dt(2)] with one depth per point solved inside.
    // Only the frame-2 reprojection carries information about depth/R/t; the
    // frame-1 residual is zero by construction (P1 lies on the ray x1).
    const H = new Float64Array(25);
    const g = new Float64Array(5);
    const { e1, e2 } = tangentBasis(t);
    for (let i = 0; i < n; i++) {
      const P1: Vec3 = [depth[i] * x1[i][0], depth[i] * x1[i][1], depth[i]];
      const P2 = m3mulVec(R, P1);
      const P2t: Vec3 = [P2[0] + t[0], P2[1] + t[1], P2[2] + t[2]];
      if (P2t[2] <= 1e-4) continue;
      const p2u = P2t[0] / P2t[2], p2v = P2t[1] / P2t[2];
      const r = [p2u - x2[i][0], p2v - x2[i][1]];
      const Jp2 = [1 / P2t[2], 0, -p2u / P2t[2], 0, 1 / P2t[2], -p2v / P2t[2]];
      // d(Exp(dr) P2)/ddr = -skew(P2) dr.
      const dRr = mul26(Jp2, negMat(skewMat(P2)));
      const dtAxes: Vec3[] = [e1, e2];
      const dRt: number[][] = [
        [Jp2[0] * e1[0] + Jp2[1] * e1[1] + Jp2[2] * e1[2], Jp2[0] * e2[0] + Jp2[1] * e2[1] + Jp2[2] * e2[2]],
        [Jp2[3] * e1[0] + Jp2[4] * e1[1] + Jp2[5] * e1[2], Jp2[3] * e2[0] + Jp2[4] * e2[1] + Jp2[5] * e2[2]],
      ];
      void dtAxes;
      // H[0..2,0..2] += Jdr^T Jdr
      for (let a = 0; a < 2; a++) {
        for (let u = 0; u < 3; u++) {
          for (let v = 0; v < 3; v++) H[u * 5 + v] += dRr[a][u] * dRr[a][v];
          g[u] += dRr[a][u] * r[a];
        }
      }
      // H[0..2,3..4] += Jdr^T Jdt and transpose
      for (let a = 0; a < 2; a++) {
        for (let u = 0; u < 3; u++) {
          for (let v = 0; v < 2; v++) {
            const val = dRr[a][u] * dRt[a][v];
            H[u * 5 + (3 + v)] += val;
            H[(3 + v) * 5 + u] += val;
          }
        }
        for (let v = 0; v < 2; v++) g[3 + v] += dRt[a][v] * r[a];
      }
      // H[3..4,3..4] += Jdt^T Jdt
      for (let a = 0; a < 2; a++) {
        for (let u = 0; u < 2; u++) {
          for (let v = 0; v < 2; v++) H[(3 + u) * 5 + (3 + v)] += dRt[a][u] * dRt[a][v];
        }
      }
    }
    const delta = solve5(H, g);
    if (!delta) break;
    for (let i = 0; i < 5; i++) delta[i] = -delta[i];
    // Damped step: cap rotation and translation increments so a bad depth
    // linearisation cannot throw the model across the scene in one iteration.
    const rotMag = Math.hypot(delta[0], delta[1], delta[2]);
    if (rotMag > 0.05) for (let i = 0; i < 3; i++) delta[i] *= 0.05 / rotMag;
    const trMag = Math.hypot(delta[3], delta[4]);
    if (trMag > 0.3) { delta[3] *= 0.3 / trMag; delta[4] *= 0.3 / trMag; }
    const costBefore = twoViewCost(R, t, x1, x2, depth, f);
    const Rcand: Mat3 = m3mul(so3Exp([delta[0], delta[1], delta[2]]), R);
    const tb = tangentBasis(t);
    const dtv: Vec3 = [delta[3] * tb.e1[0] + delta[4] * tb.e2[0], delta[3] * tb.e1[1] + delta[4] * tb.e2[1], delta[3] * tb.e1[2] + delta[4] * tb.e2[2]];
    const tcand = v3normalize([t[0] + dtv[0], t[1] + dtv[1], t[2] + dtv[2]]);
    const depthCand = new Float64Array(n);
    for (let i = 0; i < n; i++) depthCand[i] = refineDepth(Rcand, tcand, x1[i], x2[i], depth[i]);
    const costAfter = twoViewCost(Rcand, tcand, x1, x2, depthCand, f);
    if (!(costAfter < costBefore)) break;
    R = Rcand;
    t = tcand;
    depth = depthCand;
    const errs: number[] = [];
    for (let i = 0; i < n; i++) {
      const P1: Vec3 = [depth[i] * x1[i][0], depth[i] * x1[i][1], depth[i]];
      const P2 = m3add3(m3mulVec(R, P1), t);
      const pr1 = { u: P1[0] / P1[2], v: P1[1] / P1[2] };
      const pr2 = { u: P2[0] / P2[2], v: P2[1] / P2[2] };
      errs.push(f * Math.hypot(pr1.u - x1[i][0], pr1.v - x1[i][1]) + f * Math.hypot(pr2.u - x2[i][0], pr2.v - x2[i][1]));
    }
    rmsPx = Math.sqrt(errs.reduce((a, b) => a + b * b, 0) / n) / 2;
    errs.sort((a, b) => a - b);
    medianPx = errs[n >> 1] / 2;
  }
  return { R, t, rmsPx, medianPx };
}

function skewMat(v: Vec3): Float64Array {
  return new Float64Array([0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0]);
}
function negMat(a: Float64Array): Float64Array {
  const o = new Float64Array(9);
  for (let i = 0; i < 9; i++) o[i] = -a[i];
  return o;
}
function m3add3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}
function mul26(J: number[], M: Float64Array): number[][] {
  const out: number[][] = [[0, 0, 0], [0, 0, 0]];
  for (let r = 0; r < 2; r++) for (let c = 0; c < 3; c++) out[r][c] = J[r * 3] * M[0 * 3 + c] + J[r * 3 + 1] * M[1 * 3 + c] + J[r * 3 + 2] * M[2 * 3 + c];
  return out;
}
function tangentBasis(t: Vec3): { e1: Vec3; e2: Vec3 } {
  const a: Vec3 = Math.abs(t[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const e1 = v3normalize(v3cross(t, a));
  const e2 = v3normalize(v3cross(t, e1));
  return { e1, e2 };
}
function solve5(H: Float64Array, g: Float64Array): Float64Array | null {
  const A = new Float64Array(H);
  const b = new Float64Array(g);
  const n = 5;
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(A[r * n + col]) > Math.abs(A[piv * n + col])) piv = r;
    if (Math.abs(A[piv * n + col]) < 1e-14) return null;
    if (piv !== col) {
      for (let c = 0; c < n; c++) { const t = A[col * n + c]; A[col * n + c] = A[piv * n + c]; A[piv * n + c] = t; }
      const t = b[col]; b[col] = b[piv]; b[piv] = t;
    }
    const d = A[col * n + col];
    for (let r = col + 1; r < n; r++) {
      const f = A[r * n + col] / d;
      for (let c = col; c < n; c++) A[r * n + c] -= f * A[col * n + c];
      b[r] -= f * b[col];
    }
  }
  const x = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let c = r + 1; c < n; c++) s -= A[r * n + c] * x[c];
    x[r] = s / A[r * n + r];
  }
  return x;
}
function initialDepth(R: Mat3, t: Vec3, x1: number[], x2: number[]): number {
  // Depth along ray 1 from d2 b2 - d1 R b1 = t. Dotting with R b1 and b2 gives
  // the (non-symmetric) system:
  //   -d1 + c d2 = p,  -c d1 + d2 = q,  c = Rb1.b2, p = t.Rb1, q = t.b2
  const b1 = v3normalize([x1[0], x1[1], 1]);
  const b2 = v3normalize([x2[0], x2[1], 1]);
  const Rb1 = m3mulVec(R, b1);
  const c = v3dot(Rb1, b2);
  const p = v3dot(t, Rb1);
  const q = v3dot(t, b2);
  const den = 1 - c * c;
  if (den < 1e-9) return 5;
  // d1 Rb1 + t = d2 b2 with Rb1, b2 unit. Dot with Rb1 and b2:
  //   d1 - c d2 = -p ;  c d1 - d2 = -q,  c = Rb1.b2, p = t.Rb1, q = t.b2
  const d1Len = (c * q - p) / den;
  const rayNorm = Math.hypot(x1[0], x1[1], 1);
  let d1 = d1Len / rayNorm; // convert to z-coordinate, as P1 = d * [x1,1]
  // A near-parallel ray pair gives a meaningless (often negative) depth. Fall
  // back to a mid-scene guess rather than poisoning the cost with a huge value.
  if (!isFinite(d1) || d1 <= 0.1) return 6;
  if (d1 > 400) return 400;
  return d1;
}
function refineDepth(R: Mat3, t: Vec3, x1: number[], x2: number[], d0: number): number {
  let d = d0;
  for (let it = 0; it < 4; it++) {
    const P1: Vec3 = [d * x1[0], d * x1[1], d];
    const P2 = m3add3(m3mulVec(R, P1), t);
    if (P2[2] <= 1e-4) { d *= 0.5; continue; }
    const ru = P2[0] / P2[2] - x2[0];
    const rv = P2[1] / P2[2] - x2[1];
    // dP2/dt = R * [x1, 1] (since P1 = d * [x1,1]).
    const dP2 = m3mulVec(R, [x1[0], x1[1], 1]);
    const ju = [dP2[0] / P2[2] - (P2[0] / (P2[2] * P2[2])) * dP2[2]];
    const jv = [dP2[1] / P2[2] - (P2[1] / (P2[2] * P2[2])) * dP2[2]];
    const den = ju[0] * ju[0] + jv[0] * jv[0];
    if (den < 1e-12) break;
    const step = -(ju[0] * ru + jv[0] * rv) / den;
    const nd = d + step;
    if (!isFinite(nd) || nd <= 0.05 || nd > 200) break;
    d = nd;
  }
  return d;
}
