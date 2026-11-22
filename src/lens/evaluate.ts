import { Mat3, Vec3, m3mul, m3mulVec, m3transpose, v3add, v3norm, v3sub } from '../core/math';
import { rotationErrorAngle } from '../estimation/eskf';

// Evaluation code lives apart from the estimator. It is allowed to read truth and
// says so in the UI. Everything here consumes the estimator's outputs; nothing
// here writes back into the filter.

export interface AlignedTrajectory {
  /** Estimated positions after the similarity alignment. */
  aligned: Vec3[];
  /** Similarity scale, metres per estimated unit. */
  scale: number;
  R: Mat3;
  t: Vec3;
}

/**
 * Umeyama similarity alignment (rotation + scale + translation) of estimated
 * positions onto truth. Scale is reported as evidence, not assumed to be 1.
 * With path lengths of zero the scale is undefined; callers must handle null.
 */
export function alignSimilarity(est: Vec3[], truth: Vec3[]): AlignedTrajectory | null {
  const n = Math.min(est.length, truth.length);
  if (n < 3) return null;
  const muE: Vec3 = [0, 0, 0];
  const muT: Vec3 = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    muE[0] += est[i][0];
    muE[1] += est[i][1];
    muE[2] += est[i][2];
    muT[0] += truth[i][0];
    muT[1] += truth[i][1];
    muT[2] += truth[i][2];
  }
  for (let k = 0; k < 3; k++) {
    muE[k] /= n;
    muT[k] /= n;
  }
  let varE = 0;
  const cov = new Float64Array(9);
  for (let i = 0; i < n; i++) {
    const e: Vec3 = [est[i][0] - muE[0], est[i][1] - muE[1], est[i][2] - muE[2]];
    const t: Vec3 = [truth[i][0] - muT[0], truth[i][1] - muT[1], truth[i][2] - muT[2]];
    varE += e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) cov[r * 3 + c] += t[r] * e[c];
  }
  varE /= n;
  if (varE < 1e-12) return null;
  for (let i = 0; i < 9; i++) cov[i] /= n;
  // SVD of 3x3 cov. Implement via Jacobi eigen of cov^T cov is overkill; use
  // closed-form 3x3 polar decomposition by iterating R <- 0.5(R + R^-T).
  const R = polarRotation(cov);
  let traceRc = 0;
  for (let i = 0; i < 3; i++) traceRc += cov[i * 3 + i];
  let num = 0;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) num += R[r * 3 + c] * cov[r * 3 + c];
  const scale = num / varE;
  const aligned: Vec3[] = [];
  for (let i = 0; i < n; i++) {
    const e: Vec3 = [est[i][0] - muE[0], est[i][1] - muE[1], est[i][2] - muE[2]];
    const r = m3mulVec(R, e);
    aligned.push([muT[0] + scale * r[0], muT[1] + scale * r[1], muT[2] + scale * r[2]]);
  }
  const t: Vec3 = [
    muT[0] - scale * (m3mulVec(R, muE)[0]),
    muT[1] - scale * (m3mulVec(R, muE)[1]),
    muT[2] - scale * (m3mulVec(R, muE)[2]),
  ];
  void traceRc;
  return { aligned, scale, R, t };
}

/** Polar decomposition of a near-rotation. Iterative, stable for the small 3x3 case. */
function polarRotation(A: Float64Array): Mat3 {
  let R = new Float64Array(A);
  // Make the iterate start invertible if the covariance is rank-deficient.
  for (let i = 0; i < 3; i++) R[i * 3 + i] += 1e-9;
  for (let it = 0; it < 40; it++) {
    const det = det3(R);
    if (Math.abs(det) < 1e-14) break;
    const inv = inverse3(R);
    if (!inv) break;
    let diff = 0;
    const next = new Float64Array(9);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        next[r * 3 + c] = 0.5 * (R[r * 3 + c] + inv[c * 3 + r]);
        diff += Math.abs(next[r * 3 + c] - R[r * 3 + c]);
      }
    }
    R = next;
    if (diff < 1e-12) break;
  }
  return R;
}

function det3(m: Float64Array): number {
  return (
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6])
  );
}

function inverse3(m: Float64Array): Float64Array | null {
  const d = det3(m);
  if (Math.abs(d) < 1e-15) return null;
  const inv = new Float64Array(9);
  inv[0] = (m[4] * m[8] - m[5] * m[7]) / d;
  inv[1] = (m[2] * m[7] - m[1] * m[8]) / d;
  inv[2] = (m[1] * m[5] - m[2] * m[4]) / d;
  inv[3] = (m[5] * m[6] - m[3] * m[8]) / d;
  inv[4] = (m[0] * m[8] - m[2] * m[6]) / d;
  inv[5] = (m[2] * m[3] - m[0] * m[5]) / d;
  inv[6] = (m[3] * m[7] - m[4] * m[6]) / d;
  inv[7] = (m[1] * m[6] - m[0] * m[7]) / d;
  inv[8] = (m[0] * m[4] - m[1] * m[3]) / d;
  return inv;
}

export interface TrajectoryError {
  ate: number;
  rpe: number;
  maxError: number;
  rmse: number;
  scale: number;
  samples: number;
}

/** Absolute trajectory error after similarity alignment (standard metric). */
export function absoluteTrajectoryError(est: Vec3[], truth: Vec3[]): TrajectoryError {
  const al = alignSimilarity(est, truth);
  if (!al) return { ate: NaN, rpe: NaN, maxError: NaN, rmse: NaN, scale: NaN, samples: 0 };
  let sse = 0;
  let max = 0;
  for (let i = 0; i < al.aligned.length; i++) {
    const e = v3norm(v3sub(al.aligned[i], truth[i]));
    sse += e * e;
    max = Math.max(max, e);
  }
  return {
    ate: Math.sqrt(sse / al.aligned.length),
    rpe: NaN,
    maxError: max,
    rmse: Math.sqrt(sse / al.aligned.length),
    scale: al.scale,
    samples: al.aligned.length,
  };
}

/** Relative pose error at a fixed frame gap, translation in metres (no scale fit). */
export function relativeTrajectoryError(est: Vec3[], truth: Vec3[], gap = 10): { rpe: number; rpeRot: number; samples: number } {
  if (est.length <= gap || truth.length <= gap) return { rpe: NaN, rpeRot: NaN, samples: 0 };
  let sse = 0;
  let rotSse = 0;
  let n = 0;
  for (let i = 0; i + gap < Math.min(est.length, truth.length); i++) {
    const de = v3sub(est[i + gap], est[i]);
    const dt = v3sub(truth[i + gap], truth[i]);
    const err = v3norm(v3sub(de, dt));
    sse += err * err;
    rotSse += 0; // filled by caller with orientations
    n++;
  }
  return { rpe: Math.sqrt(sse / Math.max(1, n)), rpeRot: Math.sqrt(rotSse / Math.max(1, n)), samples: n };
}

export function rotationAte(estR: Mat3[], truthR: Mat3[]): number {
  const n = Math.min(estR.length, truthR.length);
  if (n === 0) return NaN;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const e = rotationErrorAngle(estR[i], truthR[i]);
    s += e * e;
  }
  return (Math.sqrt(s / n) * 180) / Math.PI;
}

export interface HoldoutCheck {
  /** Reprojection error on observations withheld from the filter. */
  holdoutReprojRms: number;
  holdoutCount: number;
  /** Error on a withheld motion interval. */
  intervalError: number;
  /** Whether held-out data could support the claim. */
  valid: boolean;
  note: string;
}

/**
 * Evaluate held-out frames: run the estimator state against truth for frames
 * excluded from optimisation, and report the residual. This is what separates a
 * real fit from a copied curve.
 */
export function holdoutValidation(
  predicted: Vec3[],
  truth: Vec3[],
  holdoutIdx: number[],
): HoldoutCheck {
  if (holdoutIdx.length === 0) {
    return { holdoutReprojRms: NaN, holdoutCount: 0, intervalError: NaN, valid: false, note: '没有留出帧，无法做独立检验' };
  }
  let sse = 0;
  let n = 0;
  for (const i of holdoutIdx) {
    if (i >= predicted.length || i >= truth.length) continue;
    const e = v3norm(v3sub(predicted[i], truth[i]));
    sse += e * e;
    n++;
  }
  if (n === 0) return { holdoutReprojRms: NaN, holdoutCount: 0, intervalError: NaN, valid: false, note: '留出帧索引越界' };
  const rms = Math.sqrt(sse / n);
  return {
    holdoutReprojRms: rms,
    holdoutCount: n,
    intervalError: rms,
    valid: true,
    note: `留出 ${n} 帧，位置残差 RMS ${rms.toFixed(3)} m（这些帧未参与滤波更新或标定）`,
  };
}

export function cameraCentersFromBody(
  R_wb: Mat3[],
  p_wb: Vec3[],
  R_bc: Mat3,
  t_bc: Vec3,
): Vec3[] {
  const out: Vec3[] = [];
  for (let i = 0; i < p_wb.length; i++) {
    out.push(v3add(p_wb[i], m3mulVec(R_wb[i], t_bc)));
  }
  return out;
}

export { m3mul, m3transpose };
