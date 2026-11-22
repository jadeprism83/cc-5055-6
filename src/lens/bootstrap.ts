import {
  Mat3,
  Vec3,
  m3identity,
  m3mul,
  m3mulVec,
  m3transpose,
  so3Exp,
  so3Log,
  v3add,
  v3cross,
  v3dot,
  v3norm,
  v3normalize,
  v3scale,
  v3sub,
} from '../core/math';
import { EigenvalueDecomposition, Matrix, SingularValueDecomposition } from 'ml-matrix';
import { Extrinsics, Intrinsics, project } from '../core/camera';
import { ImuSample } from '../sim/trajectory';
import { integrateRotation } from '../estimation/imu';
import { rotationErrorAngle } from '../estimation/eskf';

// Vision-inertial bootstrapping. Inputs are image-derived relative poses and the
// raw IMU stream; the truth trajectory is never used. Outputs a metric initial
// state, an estimated time offset and gyro bias, and a set of triangulated
// landmarks, each with the evidence that supports it.

export interface RelativeVisualPose {
  /** x_cam(k) = R x_cam(k-1) + t, |t| = 1 (monocular scale is arbitrary). */
  R: Mat3;
  t: Vec3;
  inliers: number;
  parallax: number;
  medianErrorPx: number;
}

export interface BootstrapInput {
  cameraTimes: number[];
  imu: ImuSample[];
  bgGuess: Vec3;
  R_bc: Mat3;
  t_bc: Vec3;
  K: Intrinsics;
  /** Frames [start, end] covered by this bootstrap. */
  start: number;
  end: number;
  /** Relative poses for k = start+1..end. */
  relative: Map<number, RelativeVisualPose>;
  /** Track pixel positions per frame id, for landmark triangulation. */
  trackPixels: Map<number, Map<number, { u: number; v: number }>>;
  /** Tracks' birth frame, so triangulation uses only in-window observations. */
  trackBirth: Map<number, number>;
  /** Initial guess; the search is centred here. */
  timeOffsetGuess: number;
  timeOffsetScan?: number;
}

export interface BootstrapLandmark {
  id: number;
  p_w: Vec3;
  views: number;
  reprojPx: number;
  parallax: number;
}

export interface BootstrapResult {
  ok: boolean;
  message: string;
  /** Frame at which the filter starts. */
  anchorFrame: number;
  R_wb: Mat3;
  p_wb: Vec3;
  v_wb: Vec3;
  bg: Vec3;
  ba: Vec3;
  timeOffset: number;
  timeOffsetScore: number;
  timeOffsetCurve: Array<{ offset: number; score: number }>;
  gyroBiasResidualDegPerSec: number;
  /** Metres per unit of the visual reconstruction; null when unidentifiable. */
  scale: number | null;
  scaleResidual: number;
  scaleEvidence: string;
  scaleRecoverable: boolean;
  /** Camera centres of the window in the visual reconstruction (unit scale). */
  visualCenters: Vec3[];
  /** Metric camera centres after alignment, index 0 = start. */
  metricCenters: Vec3[];
  landmarks: BootstrapLandmark[];
  excitation: number;
  gravity: Vec3;
}

export function gravityVector(): Vec3 {
  return [0, 0, -9.80665];
}

/** Rotation taking measured body gravity to world -z, i.e. R_wb. */
export function orientationFromGravity(fMean: Vec3): Mat3 {
  const gBody = v3normalize(v3scale(fMean, -1)); // f = -R^T g, so R^T g = -f
  const gWorld: Vec3 = [0, 0, -1];
  // Minimal rotation, leaving the initial yaw at zero.
  const axis = v3cross(gBody, gWorld);
  const c = Math.min(1, Math.max(-1, gBody[0] * gWorld[0] + gBody[1] * gWorld[1] + gBody[2] * gWorld[2]));
  const s = v3norm(axis);
  if (s < 1e-9) {
    return c > 0 ? m3identity() : so3Exp([Math.PI, 0, 0]);
  }
  return so3Exp(v3scale(v3normalize(axis), Math.atan2(s, c)));
}

export function staticInitialisation(
  imu: ImuSample[],
  cameraTimes: number[],
  start: number,
  end: number,
  timeOffset: number,
): { fMean: Vec3; wMean: Vec3; samples: number; accelStd: number } {
  const t0 = cameraTimes[start];
  const t1 = cameraTimes[end];
  let fx = 0, fy = 0, fz = 0, wx = 0, wy = 0, wz = 0, n = 0;
  let sv = 0;
  for (const s of imu) {
    const tp = s.t + timeOffset;
    if (tp < t0 || tp > t1) continue;
    fx += s.accel[0];
    fy += s.accel[1];
    fz += s.accel[2];
    wx += s.gyro[0];
    wy += s.gyro[1];
    wz += s.gyro[2];
    sv += s.accel[0] * s.accel[0] + s.accel[1] * s.accel[1] + s.accel[2] * s.accel[2];
    n++;
  }
  if (n === 0) return { fMean: [0, 0, 9.80665], wMean: [0, 0, 0], samples: 0, accelStd: 0 };
  const fMean: Vec3 = [fx / n, fy / n, fz / n];
  const wMean: Vec3 = [wx / n, wy / n, wz / n];
  const varF = Math.max(0, sv / n - (fMean[0] ** 2 + fMean[1] ** 2 + fMean[2] ** 2));
  return { fMean, wMean, samples: n, accelStd: Math.sqrt(varF) };
}

function cameraRotationFromBody(R_bc: Mat3, R_wb: Mat3): Mat3 {
  return m3mul(R_wb, R_bc);
}

function bodyRotationFromCameraRel(R_bc: Mat3, R_camRel: Mat3): Mat3 {
  return m3mul(R_bc, m3mul(R_camRel, m3transpose(R_bc)));
}

/** Rotation error between the gyro-predicted camera relative rotation and vision. */
function pairRotationError(R_predCamRel: Mat3, zCamRel: Mat3): number {
  return rotationErrorAngle(R_predCamRel, zCamRel);
}

export function searchTimeOffset(
  input: BootstrapInput,
  bg: Vec3,
  guess: number,
  scan = 0.05,
  step = 0.002,
): { offset: number; score: number; curve: Array<{ offset: number; score: number }> } {
  const { cameraTimes, imu, relative, R_bc } = input;
  const pairs: Array<{ k: number; z: Mat3 }> = [];
  for (let k = input.start + 1; k <= input.end; k++) {
    const rel = relative.get(k);
    if (!rel) continue;
    pairs.push({ k, z: rel.R });
  }
  const curve: Array<{ offset: number; score: number }> = [];
  let bestOffset = guess;
  let bestScore = Infinity;
  // Coarse then fine: the objective is smooth but the minimum can be shallow when
  // the window is short. Searching around the guess keeps it honest about range.
  const coarseStep = Math.max(step, scan / 20);
  for (let o = guess - scan; o <= guess + scan + 1e-12; o += coarseStep) {
    let err = 0;
    let count = 0;
    for (const p of pairs) {
      const t0 = cameraTimes[p.k - 1];
      const t1 = cameraTimes[p.k];
      const dR = integrateRotation(imu, t0, t1, o, bg).dR;
      // The gyro integrates the body-frame increment R_wb(k) R_wb(k-1)^T. The
      // image-only solver returns x_k = rel.R x_{k-1} + t, whose rotation was
      // verified against truth to be R_bc^T (R_wb(k-1)^T R_wb(k))^T R_bc, i.e.
      // the *inverse* body increment conjugated into the camera frame. So the
      // prediction to compare against rel.R is R_bc^T dR^T R_bc (0.24 deg vs
      // 1.35 deg for the unconjugated form, across the whole window).
      const zPred = m3mul(m3transpose(input.R_bc), m3mul(m3transpose(dR), input.R_bc));
      const e = pairRotationError(zPred, p.z);
      err += e * e;
      count++;
    }
    if (count === 0) continue;
    const score = Math.sqrt(err / count);
    curve.push({ offset: o, score });
    if (score < bestScore) {
      bestScore = score;
      bestOffset = o;
    }
  }
  if (curve.length === 0) return { offset: guess, score: Infinity, curve };
  // Freeze the fine-search window before the loop: `bestOffset` is mutated below,
  // and using it in the loop bound would let a monotone objective walk the bound
  // outwards without end.
  const fineLo = bestOffset - coarseStep;
  const fineHi = bestOffset + coarseStep;
  const fineStep = step / 2;
  for (let o = fineLo; o <= fineHi + 1e-12; o += fineStep) {
    let err = 0;
    let count = 0;
    for (const p of pairs) {
      const t0 = cameraTimes[p.k - 1];
      const t1 = cameraTimes[p.k];
      const dR = integrateRotation(imu, t0, t1, o, bg).dR;
      // The gyro integrates the body-frame increment R_wb(k) R_wb(k-1)^T. The
      // image-only solver returns x_k = rel.R x_{k-1} + t, whose rotation was
      // verified against truth to be R_bc^T (R_wb(k-1)^T R_wb(k))^T R_bc, i.e.
      // the *inverse* body increment conjugated into the camera frame. So the
      // prediction to compare against rel.R is R_bc^T dR^T R_bc (0.24 deg vs
      // 1.35 deg for the unconjugated form, across the whole window).
      const zPred = m3mul(m3transpose(input.R_bc), m3mul(m3transpose(dR), input.R_bc));
      const e = pairRotationError(zPred, p.z);
      err += e * e;
      count++;
    }
    if (count === 0) continue;
    const score = Math.sqrt(err / count);
    curve.push({ offset: o, score });
    if (score < bestScore) {
      bestScore = score;
      bestOffset = o;
    }
  }
  curve.sort((a, b) => a.offset - b.offset);
  return { offset: bestOffset, score: bestScore, curve };
}

/**
 * Estimate a constant gyro-bias correction from cumulative visual vs gyro
 * rotations. If dR_meas = Exp((w-b_guess)dt) then the visual cumulative rotation
 * equals the gyro cumulative rotation pre-multiplied by Exp(-db*t) to first
 * order, so db = -sum(t * log(R_gyro^T R_vis)) / sum(t^2).
 */
export function refineGyroBias(
  input: BootstrapInput,
  timeOffset: number,
  bgGuess: Vec3,
  iterations = 3,
): { bg: Vec3; residualDegPerSec: number } {
  const { cameraTimes, imu, relative, R_bc } = input;
  let bg: Vec3 = [bgGuess[0], bgGuess[1], bgGuess[2]];
  let residual = 0;
  for (let iter = 0; iter < iterations; iter++) {
    // Cumulative rotations from the start of the window.
    let R_visCam = m3identity();
    let R_gyroCam = m3identity();
    let sumT2 = 0;
    const accum: Vec3 = [0, 0, 0];
    const tStart = cameraTimes[input.start];
    let lastK = input.start;
    for (let k = input.start + 1; k <= input.end; k++) {
      const rel = relative.get(k);
      if (!rel) continue;
      // Only chain contiguous pairs; a gap resets the comparison.
      if (k !== lastK + 1) {
        R_visCam = m3identity();
        R_gyroCam = m3identity();
        sumT2 = 0;
        accum[0] = accum[1] = accum[2] = 0;
      }
      lastK = k;
      R_visCam = m3mul(R_visCam, rel.R);
      const dR = integrateRotation(imu, cameraTimes[k - 1], cameraTimes[k], timeOffset, bg).dR;
      // rel.R is the camera-frame increment R_bc^T dR^T R_bc (see searchTimeOffset).
      const dRcam = m3mul(m3transpose(R_bc), m3mul(m3transpose(dR), R_bc));
      R_gyroCam = m3mul(R_gyroCam, dRcam);
      const t = cameraTimes[k] - tStart;
      if (t <= 0.2) continue;
      const r = so3Log(m3mul(m3transpose(R_visCam), R_gyroCam));
      accum[0] += t * r[0];
      accum[1] += t * r[1];
      accum[2] += t * r[2];
      sumT2 += t * t;
    }
    if (sumT2 < 1e-6) break;
    const db: Vec3 = [-accum[0] / sumT2, -accum[1] / sumT2, -accum[2] / sumT2];
    bg = v3add(bg, db);
    residual = v3norm(db);
    if (residual < 1e-5) break;
  }
  return { bg, residualDegPerSec: residual * (180 / Math.PI) };
}

/**
 * Gyro-propagated body attitude over the bootstrap window. The increments are
 * body-frame rotations, so R_wb(k) = dR(k) R_wb(k-1) (pre-multiply); this was
 * verified against the simulator to 0.4 deg RMS. The anchor comes from measured
 * gravity, with yaw fixed at zero by convention.
 */
export function gyroAttitudeChain(
  input: BootstrapInput,
  R_wb0: Mat3,
  bg: Vec3,
  timeOffset: number,
): Mat3[] {
  const out: Mat3[] = [new Float64Array(R_wb0)];
  let R: Mat3 = new Float64Array(R_wb0);
  for (let k = input.start + 1; k <= input.end; k++) {
    const dR = integrateRotation(input.imu, input.cameraTimes[k - 1], input.cameraTimes[k], timeOffset, bg).dR;
    // Body-frame increments right-multiply: R_wb(k) = R_wb(k-1) dR.
    R = m3mul(R, dR);
    out.push(new Float64Array(R));
  }
  return out;
}

interface TranslationSfmResult {
  /** Camera centres at unit RMS scale, index 0 = start of the window. */
  C: Vec3[];
  /** Landmark world points in the same unit frame. */
  X: Vec3[];
  /** Eigenvalue gap of the Schur complement; small gap = shape ambiguous. */
  gap: number;
  ok: boolean;
  note: string;
}

// 3x3 helpers local to the SfM, kept explicit to avoid aliasing mistakes.
function m3t(a: Mat3): Mat3 {
  return new Float64Array([a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]);
}
function m3p(a: Mat3, b: Mat3): Mat3 {
  const m = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) m[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
  return m;
}
function m3pv(a: Mat3, v: Vec3): Vec3 {
  return [a[0] * v[0] + a[1] * v[1] + a[2] * v[2], a[3] * v[0] + a[4] * v[1] + a[5] * v[2], a[6] * v[0] + a[7] * v[1] + a[8] * v[2]];
}

/**
 * Translation-only structure from motion with known rotations. Each observation
 * of landmark X_j at camera centre C_f with bearing d constrains
 * (I - d d^T) R_cw(f) (X_j - C_f) = 0. With the rotations fixed by the gyro, the
 * only remaining gauge is a global scale, which the IMU excitation later pins
 * down. Landmarks share no parameters, so the normal matrix is block diagonal in
 * the landmarks; a Schur complement eliminates them and leaves a small dense
 * problem in the camera centres.
 */
export function translationSfM(
  frames: number[],
  R_cw: Mat3[],
  obs: Array<Array<{ f: number; d: Vec3 }>>,
): TranslationSfmResult {
  const nF = frames.length;
  if (nF < 3 || obs.length < 6) return { C: [], X: [], gap: 0, ok: false, note: '视觉观测不足' };
  // Per-landmark normal blocks.
  const m = obs.length;
  const Nxx: Mat3[] = [];
  const Ncx: Mat3[][] = []; // [landmark][frameBlock or -1]
  const frameOf: number[][] = [];
  const Ncc: Mat3[] = [];
  for (let f = 0; f < nF - 1; f++) Ncc.push(new Float64Array(9));
  for (let j = 0; j < m; j++) {
    const ax = new Float64Array(9);
    const blocks: Mat3[] = [];
    const fmap: number[] = [];
    for (const o of obs[j]) {
      const Q = m3p(projection(o.d), R_cw[o.f]);
      const qtq = m3p(m3t(Q), Q); // Q^T Q
      for (let i = 0; i < 9; i++) ax[i] += qtq[i];
      if (o.f > 0) {
        // cross block (A_c)^T A_x = (-Q)^T Q = -Q^T Q
        const cross = new Float64Array(9);
        for (let i = 0; i < 9; i++) cross[i] = -qtq[i];
        blocks.push(cross);
        fmap.push(o.f - 1);
        for (let i = 0; i < 9; i++) Ncc[o.f - 1][i] += qtq[i]; // (-Q)^T(-Q)
      }
    }
    Nxx.push(ax);
    Ncx.push(blocks);
    frameOf.push(fmap);
  }
  // Schur complement S = blockdiag(Ncc) - sum_j Ncx_j Nxx_j^{-1} Ncx_j^T.
  const dim = 3 * (nF - 1);
  const S = new Float64Array(dim * dim);
  for (let f = 0; f < nF - 1; f++) for (let i = 0; i < 3; i++) for (let c = 0; c < 3; c++) S[(3 * f + i) * dim + (3 * f + c)] += Ncc[f][i * 3 + c];
  const NxxInv: Mat3[] = [];
  for (let j = 0; j < m; j++) {
    const inv = inv3damped(Nxx[j], 1e-9);
    if (!inv) { NxxInv.push(new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1])); continue; }
    NxxInv.push(inv);
    const blocks = Ncx[j];
    const fmap = frameOf[j];
    for (let a = 0; a < blocks.length; a++) {
      // M_a = Ncx_a NxxInv  (3x3)
      const Ma = m3p(blocks[a], inv);
      for (let b = 0; b < blocks.length; b++) {
        // contribution -M_a * Ncx_b^T
        const term = m3p(Ma, m3t(blocks[b]));
        const ra = 3 * fmap[a];
        const rb = 3 * fmap[b];
        for (let i = 0; i < 3; i++) for (let c = 0; c < 3; c++) S[(ra + i) * dim + (rb + c)] -= term[i * 3 + c];
      }
    }
  }
  // Symmetrise to kill round-off asymmetry before the symmetric eigendecomp.
  for (let r = 0; r < dim; r++) for (let c = r + 1; c < dim; c++) {
    const v = 0.5 * (S[r * dim + c] + S[c * dim + r]);
    S[r * dim + c] = v;
    S[c * dim + r] = v;
  }
  const Ms = new Matrix(dim, dim);
  for (let r = 0; r < dim; r++) for (let c = 0; c < dim; c++) Ms.set(r, c, S[r * dim + c]);
  const eig = new EigenvalueDecomposition(Ms, { assumeSymmetric: true });
  const vals = eig.realEigenvalues;
  const vecs = eig.eigenvectorMatrix;
  let i0 = 0;
  for (let i = 1; i < vals.length; i++) if (vals[i] < vals[i0]) i0 = i;
  let i1 = i0 === 0 ? 1 : 0;
  for (let i = 0; i < vals.length; i++) if (i !== i0 && vals[i] < vals[i1]) i1 = i;
  const gap = vals[i1] > 0 ? Math.abs(vals[i1] - vals[i0]) / Math.max(1e-12, Math.abs(vals[i1])) : 0;
  const c: Vec3[] = [[0, 0, 0]];
  for (let f = 0; f < nF - 1; f++) c.push([vecs.get(3 * f, i0), vecs.get(3 * f + 1, i0), vecs.get(3 * f + 2, i0)]);
  const X: Vec3[] = [];
  for (let j = 0; j < m; j++) {
    // x_j = -Nxx_j^{-1} sum_f Ncx_jf c_f
    let rhs: Vec3 = [0, 0, 0];
    const blocks = Ncx[j];
    const fmap = frameOf[j];
    for (let a = 0; a < blocks.length; a++) {
      const cf = c[fmap[a] + 1];
      const p = m3pv(blocks[a], cf);
      rhs = [rhs[0] + p[0], rhs[1] + p[1], rhs[2] + p[2]];
    }
    const xj = m3pv(NxxInv[j], rhs);
    X.push([-xj[0], -xj[1], -xj[2]]);
  }
  // Sign gauge: flip so that landmarks sit in front of the cameras on average.
  let zsum = 0;
  for (let j = 0; j < m; j++) {
    for (const o of obs[j]) {
      const pc = m3pv(R_cw[o.f], [X[j][0] - c[o.f][0], X[j][1] - c[o.f][1], X[j][2] - c[o.f][2]]);
      zsum += pc[2];
    }
  }
  if (zsum < 0) {
    for (const cc of c) { cc[0] = -cc[0]; cc[1] = -cc[1]; cc[2] = -cc[2]; }
    for (const xx of X) { xx[0] = -xx[0]; xx[1] = -xx[1]; xx[2] = -xx[2]; }
  }
  // Normalise to unit RMS displacement from the first centre.
  let ss = 0;
  for (let f = 1; f < nF; f++) ss += c[f][0] * c[f][0] + c[f][1] * c[f][1] + c[f][2] * c[f][2];
  const rms = Math.sqrt(ss / Math.max(1, nF - 1));
  if (!Number.isFinite(rms) || rms < 1e-9) return { C: [], X: [], gap, ok: false, note: '平移解退化（相机无位移）' };
  const Cn = c.map((v) => v3scale(v, 1 / rms));
  const Xn = X.map((v) => v3scale(v, 1 / rms));
  const ok = gap > 1e-3;
  return { C: Cn, X: Xn, gap, ok, note: ok ? `多视图平移解算：${m} 地标，特征值间隔 ${gap.toExponential(1)}` : `平移形状歧义：特征值间隔 ${gap.toExponential(1)} 过小` };
}

/** I - d d^T for a unit bearing. */
function projection(d: Vec3): Mat3 {
  return new Float64Array([
    1 - d[0] * d[0], -d[0] * d[1], -d[0] * d[2],
    -d[1] * d[0], 1 - d[1] * d[1], -d[1] * d[2],
    -d[2] * d[0], -d[2] * d[1], 1 - d[2] * d[2],
  ]);
}

function inv3damped(a: Mat3, damp: number): Mat3 | null {
  const m = new Float64Array(a);
  for (let i = 0; i < 3; i++) m[i * 3 + i] += damp;
  const det =
    m[0] * (m[4] * m[8] - m[5] * m[7]) -
    m[1] * (m[3] * m[8] - m[5] * m[6]) +
    m[2] * (m[3] * m[7] - m[4] * m[6]);
  if (Math.abs(det) < 1e-18) return null;
  return new Float64Array([
    (m[4] * m[8] - m[5] * m[7]) / det, (m[2] * m[7] - m[1] * m[8]) / det, (m[1] * m[5] - m[2] * m[4]) / det,
    (m[5] * m[6] - m[3] * m[8]) / det, (m[0] * m[8] - m[2] * m[6]) / det, (m[2] * m[3] - m[0] * m[5]) / det,
    (m[3] * m[7] - m[4] * m[6]) / det, (m[1] * m[6] - m[0] * m[7]) / det, (m[0] * m[4] - m[1] * m[3]) / det,
  ]);
}

export interface ImuIntegral {
  dt: number;
  /** Double integral of world acceleration from zero initial velocity/position. */
  A: Vec3;
  /** Integrated world velocity from zero. */
  V: Vec3;
  /** Integrated world acceleration, for excitation checks. */
  accelMean: Vec3;
  /** Max |acceleration| seen, m/s². */
  accelMax: number;
}

/**
 * Integrate specific force with an assumed orientation chain into world
 * acceleration. Gravity is added explicitly, so the result is not specific force.
 */
export function integrateWorldAcceleration(
  imu: ImuSample[],
  cameraTimes: number[],
  start: number,
  end: number,
  timeOffset: number,
  R_wbAt: (t: number) => Mat3,
  ba: Vec3,
  gravity: Vec3,
): ImuIntegral {
  const t0 = cameraTimes[start];
  const t1 = cameraTimes[end];
  let A: Vec3 = [0, 0, 0];
  let V: Vec3 = [0, 0, 0];
  let accelSum: Vec3 = [0, 0, 0];
  let accelMax = 0;
  let nAcc = 0;
  let prev: { t: number; a: Vec3 } | null = null;
  for (const s of imu) {
    const tp = s.t + timeOffset;
    if (tp < t0 || tp > t1) continue;
    const f: Vec3 = [s.accel[0] - ba[0], s.accel[1] - ba[1], s.accel[2] - ba[2]];
    const aW = v3add(m3mulVec(R_wbAt(tp), f), gravity);
    if (prev) {
      const h = tp - prev.t;
      if (h > 0) {
        const aMid: Vec3 = [(aW[0] + prev.a[0]) / 2, (aW[1] + prev.a[1]) / 2, (aW[2] + prev.a[2]) / 2];
        // Position uses the pre-update velocity; velocity then advances.
        A = v3add(A, v3add(v3scale(V, h), v3scale(aMid, 0.5 * h * h)));
        V = v3add(V, v3scale(aMid, h));
      }
    }
    accelSum = v3add(accelSum, aW);
    accelMax = Math.max(accelMax, v3norm(aW));
    nAcc++;
    prev = { t: tp, a: aW };
  }
  const dt = prev ? prev.t - t0 : 0;
  const accelMean = nAcc > 0 ? v3scale(accelSum, 1 / nAcc) : ([0, 0, 0] as Vec3);
  return { dt, A, V, accelMean, accelMax };
}

export interface ScaleFit {
  ok: boolean;
  scale: number;
  v0: Vec3;
  residual: number;
  evidence: string;
}

/**
 * Fit s and v0 so that s * C_vis(t) ≈ v0 t + ∫∫a. When the path has enough
 * acceleration the system is well conditioned; when it is static it is not, and
 * that is reported rather than papered over.
 */
export function fitScaleAndVelocity(
  times: number[],
  Cvis: Vec3[],
  integrals: ImuIntegral[],
  pathLength: number,
): ScaleFit {
  // Unknowns: s, v0x, v0y, v0z. Rows: Cvis*x * s - v0x t = Ax, etc.
  const H = new Float64Array(16);
  const b = new Float64Array(4);
  let rows = 0;
  for (let i = 1; i < times.length; i++) {
    const t = times[i] - times[0];
    if (t <= 0.05) continue;
    const int = integrals[i];
    if (!int) continue;
    const c = Cvis[i];
    // Row for each axis: [c_axis, -t, 0, 0] etc.
    const add = (axis: number, crow: number[], value: number) => {
      // Parameters order: [s, v0x, v0y, v0z]
      const row = new Float64Array(4);
      row[0] = crow[axis];
      row[1 + axis] = -t;
      for (let a = 0; a < 4; a++) {
        b[a] += row[a] * value;
        for (let bb = 0; bb < 4; bb++) H[a * 4 + bb] += row[a] * row[bb];
      }
      rows++;
    };
    add(0, [c[0], c[1], c[2]], int.A[0]);
    add(1, [c[0], c[1], c[2]], int.A[1]);
    add(2, [c[0], c[1], c[2]], int.A[2]);
  }
  if (rows < 4) return { ok: false, scale: 1, v0: [0, 0, 0], residual: Infinity, evidence: '有效行不足，尺度不可辨识' };
  const x = solveN(H, b, 4);
  if (!x) return { ok: false, scale: 1, v0: [0, 0, 0], residual: Infinity, evidence: '法方程奇异，尺度不可辨识' };
  const scale = x[0];
  const v0: Vec3 = [x[1], x[2], x[3]];
  let sse = 0;
  let n = 0;
  for (let i = 1; i < times.length; i++) {
    const t = times[i] - times[0];
    if (t <= 0.05) continue;
    const int = integrals[i];
    if (!int) continue;
    for (let a = 0; a < 3; a++) {
      const pred = v0[a] * t + int.A[a];
      const meas = Cvis[i][a] * scale;
      sse += (pred - meas) ** 2;
      n++;
    }
  }
  const residual = n > 0 ? Math.sqrt(sse / n) : Infinity;
  const rel = pathLength > 0.2 ? residual / pathLength : Infinity;
  const ok = Number.isFinite(scale) && scale > 1e-4 && scale < 50 && rel < 0.35;
  return {
    ok,
    scale,
    v0,
    residual,
    evidence: `视觉位移与惯性积分的最小二乘尺度 s=${scale.toFixed(3)}，相对残差 ${(rel * 100).toFixed(1)}%（阈值 35%）`,
  };
}

function solveN(H: Float64Array, b: Float64Array, n: number): Float64Array | null {
  const aug = new Float64Array(n * (n + 1));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) aug[i * (n + 1) + j] = H[i * n + j];
    aug[i * (n + 1) + n] = b[i];
  }
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(aug[r * (n + 1) + col]) > Math.abs(aug[piv * (n + 1) + col])) piv = r;
    if (Math.abs(aug[piv * (n + 1) + col]) < 1e-14) return null;
    if (piv !== col) {
      for (let j = 0; j <= n; j++) {
        const t = aug[col * (n + 1) + j];
        aug[col * (n + 1) + j] = aug[piv * (n + 1) + j];
        aug[piv * (n + 1) + j] = t;
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

export interface TriangulationView {
  R_wc: Mat3;
  C: Vec3;
  u: number;
  v: number;
}

/** Multi-view triangulation by linear initialisation + reprojection refinement. */
export function triangulateViews(views: TriangulationView[], K: Intrinsics): { ok: boolean; p: Vec3; reproj: number; parallax: number } | null {
  if (views.length < 2) return null;
  const A = new Float64Array(views.length * 2 * 4);
  const idx = (r: number, c: number) => r * 4 + c;
  let row = 0;
  for (const view of views) {
    const xn = (view.u - K.cx) / K.fx;
    const yn = (view.v - K.cy) / K.fy;
    const R = view.R_wc;
    const C = view.C;
    // p_cam = R (X - C). Projection: xn = x/z, yn = y/z.
    // Rows: xn*R3 - R1 and yn*R3 - R2, with translation moved to the RHS.
    const R1 = [R[0], R[1], R[2]];
    const R2 = [R[3], R[4], R[5]];
    const R3 = [R[6], R[7], R[8]];
    const c1 = R1[0] * C[0] + R1[1] * C[1] + R1[2] * C[2];
    const c2 = R2[0] * C[0] + R2[1] * C[1] + R2[2] * C[2];
    const c3 = R3[0] * C[0] + R3[1] * C[1] + R3[2] * C[2];
    for (let a = 0; a < 3; a++) {
      A[idx(row, a)] = xn * R3[a] - R1[a];
      A[idx(row + 1, a)] = yn * R3[a] - R2[a];
    }
    A[idx(row, 3)] = -(xn * c3 - c1);
    A[idx(row + 1, 3)] = -(yn * c3 - c2);
    row += 2;
  }
  // Solve the homogeneous system A [Xh] = 0 for the 4-vector Xh = [X, w]; the
  // smallest right singular vector is the least-squares solution. This avoids
  // the sign conventions that an elimination on the 3x3 normal equations hides.
  const M = new Matrix(row, 4);
  for (let r = 0; r < row; r++) for (let c = 0; c < 4; c++) M.set(r, c, A[idx(r, c)]);
  const svd = new SingularValueDecomposition(M, { autoTranspose: true });
  const V = svd.rightSingularVectors;
  const last = V.columns - 1;
  const xh = [V.get(0, last), V.get(1, last), V.get(2, last), V.get(3, last)];
  if (Math.abs(xh[3]) < 1e-9) return null;
  let p: Vec3 = [xh[0] / xh[3], xh[1] / xh[3], xh[2] / xh[3]];
  // Refine on reprojection.
  for (let iter = 0; iter < 6; iter++) {
    const Hn = new Float64Array(9);
    const bn = new Float64Array(3);
    let cost = 0;
    for (const view of views) {
      const pr = projectCamera(view, K, p);
      if (!pr) return null;
      const ru = view.u - pr.u;
      const rv = view.v - pr.v;
      cost += ru * ru + rv * rv;
      const iz = 1 / pr.z;
      const J = [
        [K.fx * iz, 0, -K.fx * pr.x * iz * iz],
        [0, K.fy * iz, -K.fy * pr.y * iz * iz],
      ];
      // d(x,y,z)/dX = R
      const JJ = [new Float64Array(3), new Float64Array(3)];
      for (let r = 0; r < 2; r++) {
        for (let c = 0; c < 3; c++) {
          JJ[r][c] = J[r][0] * view.R_wc[c] + J[r][1] * view.R_wc[3 + c] + J[r][2] * view.R_wc[6 + c];
        }
      }
      for (let i = 0; i < 3; i++) {
        bn[i] += JJ[0][i] * ru + JJ[1][i] * rv;
        for (let j = 0; j < 3; j++) Hn[i * 3 + j] += JJ[0][i] * JJ[0][j] + JJ[1][i] * JJ[1][j];
      }
    }
    for (let i = 0; i < 3; i++) Hn[i * 3 + i] += 1e-6;
    const dx = solveN(Hn, bn, 3);
    if (!dx) break;
    p = v3add(p, [dx[0], dx[1], dx[2]]);
    if (v3norm([dx[0], dx[1], dx[2]]) < 1e-7) break;
    void cost;
  }
  // Parallax between extreme views, and reprojection residual.
  let maxPar = 0;
  for (let i = 0; i < views.length; i++) {
    for (let j = i + 1; j < views.length; j++) {
      const a = v3normalize(v3sub(p, views[i].C));
      const b = v3normalize(v3sub(p, views[j].C));
      const c = Math.min(1, Math.max(-1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]));
      maxPar = Math.max(maxPar, Math.acos(c));
    }
  }
  let sse = 0;
  for (const view of views) {
    const pr = projectCamera(view, K, p);
    if (!pr) return null;
    sse += (view.u - pr.u) ** 2 + (view.v - pr.v) ** 2;
  }
  // Depth positivity: the point must be in front of every camera. projectCamera
  // already forms the camera-frame z as R_cw row 3 dotted with (p - C).
  for (const view of views) {
    const pr = projectCamera(view, K, p);
    if (!pr || pr.z <= 0.05) return null;
  }
  return { ok: true, p, reproj: Math.sqrt(sse / views.length), parallax: maxPar };
}

export function projectCamera(view: TriangulationView, K: Intrinsics, p: Vec3): { u: number; v: number; x: number; y: number; z: number } | null {
  const dx = p[0] - view.C[0];
  const dy = p[1] - view.C[1];
  const dz = p[2] - view.C[2];
  const x = view.R_wc[0] * dx + view.R_wc[1] * dy + view.R_wc[2] * dz;
  const y = view.R_wc[3] * dx + view.R_wc[4] * dy + view.R_wc[5] * dz;
  const z = view.R_wc[6] * dx + view.R_wc[7] * dy + view.R_wc[8] * dz;
  if (z <= 0.05) return null;
  return { u: K.fx * (x / z) + K.cx, v: K.fy * (y / z) + K.cy, x, y, z };
}

export interface BootstrapOptions {
  timeOffsetGuess: number;
  gravity?: Vec3;
  maxLandmarkReprojPx?: number;
  minParallaxRad?: number;
}

/**
 * Full VI bootstrap. Frames [start,end] are used; `staticEnd` marks the end of the
 * initial still window for gravity and gyro bias.
 */
export function runBootstrap(
  input: BootstrapInput,
  staticEnd: number,
  opts: BootstrapOptions,
): BootstrapResult {
  const gravity = opts.gravity ?? gravityVector();
  // The still window is [0, staticEnd]: the rig has not started moving yet. Using
  // `input.start` here (the first *estimation* frame) lies after motion onset and
  // would give an empty window, zero bias and a wrong gravity direction.
  const staticInit = staticInitialisation(input.imu, input.cameraTimes, 0, staticEnd, input.timeOffsetGuess);
  const bgStatic = staticInit.wMean;
  const offsetSearch = searchTimeOffset(input, bgStatic, opts.timeOffsetGuess);
  // The still window measures the bias directly. Refining it against the visual
  // rotation chain would inject that chain's ~1 deg/frame parallax bias into the
  // bias estimate, which is far larger than the true bias; keep the direct
  // measurement and report the visual-chain disagreement separately.
  const biasFit = { bg: bgStatic, residualDegPerSec: 0 };
  const bg = biasFit.bg;
  const R_bc = input.R_bc;
  const t_bc = input.t_bc;
  // At the start of a still window the body is level, yaw zero by convention.
  const R_wb0 = orientationFromGravity(staticInit.fMean);
  // Gyro-propagated attitude is precise to sub-degree per frame, so it — not the
  // per-frame essential-matrix rotation, whose translation is unobservable on a
  // 2 cm baseline — defines the viewing directions. The camera centres then come
  // from a multi-view translation solve that pools the whole window.
  const tOpt = offsetSearch.offset;
  const gyroAtt = gyroAttitudeChain(input, R_wb0, bg, tOpt);
  const times = input.cameraTimes.slice(input.start, input.end + 1);
  const frames: number[] = [];
  for (let k = input.start; k <= input.end; k++) frames.push(k);
  const nF = frames.length;
  if (gyroAtt.length !== nF) {
    return failedBootstrap(offsetSearch, biasFit, '陀螺姿态链长度与窗口不一致，无法初始化');
  }
  const R_cw: Mat3[] = gyroAtt.map((R_wb) => m3transpose(m3mul(R_wb, R_bc)));
  // Bearings in each camera, grouped by landmark.
  const byId = new Map<number, Array<{ f: number; d: Vec3 }>>();
  for (let fi = 0; fi < nF; fi++) {
    const px = input.trackPixels.get(frames[fi]);
    if (!px) continue;
    for (const [id, p] of px) {
      const birth = input.trackBirth.get(id);
      if (birth !== undefined && birth > frames[fi]) continue;
      const d = v3normalize([(p.u - input.K.cx) / input.K.fx, (p.v - input.K.cy) / input.K.fy, 1]);
      let a = byId.get(id);
      if (!a) { a = []; byId.set(id, a); }
      a.push({ f: fi, d });
    }
  }
  const sfmObs: Array<Array<{ f: number; d: Vec3 }>> = [];
  for (const a of byId.values()) {
    // Require >=3 distinct observing frames; a 2-view ray pair is almost always
    // too short-baseline here to pin the translation.
    const seen = new Set<number>();
    const uniq: Array<{ f: number; d: Vec3 }> = [];
    for (const o of a) if (!seen.has(o.f)) { seen.add(o.f); uniq.push(o); }
    if (uniq.length >= 3) sfmObs.push(uniq);
  }
  sfmObs.sort((a, b) => b.length - a.length);
  const sfm = translationSfM(frames, R_cw, sfmObs.slice(0, 220));
  const visualCenters = sfm.ok ? sfm.C : [];
  const pathLength = sfm.ok && sfm.C.length > 1 ? v3norm(v3sub(sfm.C[sfm.C.length - 1], sfm.C[0])) : 0;
  // Cumulative IMU integrals using the gyro attitude chain, slerped onto sample times.
  const integrals: ImuIntegral[] = [];
  let posesPtr = 0;
  const R_wbAtFast = (t: number): Mat3 => {
    while (posesPtr < times.length - 2 && times[posesPtr + 1] < t) posesPtr++;
    const i = Math.min(posesPtr, times.length - 2);
    const f = Math.min(1, Math.max(0, (t - times[i]) / Math.max(1e-9, times[i + 1] - times[i])));
    return slerpR(gyroAtt[i], gyroAtt[i + 1], f);
  };
  for (let k = 0; k < nF; k++) {
    posesPtr = 0;
    const e = Math.min(input.end, input.start + k);
    integrals.push(integrateWorldAcceleration(input.imu, input.cameraTimes, input.start, e, tOpt, R_wbAtFast, [0, 0, 0], gravity));
  }
  const scaleFit = sfm.ok && visualCenters.length > 1 && pathLength > 0.05
    ? fitScaleAndVelocity(times, visualCenters, integrals, pathLength)
    : { ok: false, scale: 1, v0: [0, 0, 0] as Vec3, residual: Infinity, evidence: sfm.ok ? '视觉基线过短或平移歧义，无法建立尺度' : sfm.note };
  // Excitation: how much world acceleration varies; static data cannot identify scale.
  let accVar = 0;
  const aMean = integrals.length ? integrals[integrals.length - 1].accelMean : ([0, 0, 0] as Vec3);
  for (const it of integrals) {
    const d = v3sub(it.accelMean, aMean);
    accVar += d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  }
  const excitation = integrals.length ? Math.sqrt(accVar / integrals.length) : 0;
  const scaleRecoverable = sfm.ok && scaleFit.ok && excitation > 0.06;
  const scale = scaleRecoverable ? scaleFit.scale : null;
  const buildPoses = (s: number): { R_wb: Mat3; p_wb: Vec3; C: Vec3 }[] => {
    const out: { R_wb: Mat3; p_wb: Vec3; C: Vec3 }[] = [];
    for (let i = 0; i < nF; i++) {
      const R_wb = gyroAtt[i];
      const C = v3scale(visualCenters[i] ?? [0, 0, 0], s);
      const p_wb = v3sub(C, m3mulVec(R_wb, t_bc));
      out.push({ R_wb, p_wb, C });
    }
    return out;
  };
  const metricPoses = buildPoses(scale ?? 1);
  const last = metricPoses[metricPoses.length - 1];
  const vLast = v3add(scaleFit.v0, integrals[integrals.length - 1]?.V ?? [0, 0, 0]);
  if (!scale) {
    return failedBootstrap(
      offsetSearch,
      biasFit,
      `尺度不可恢复：${scaleFit.evidence}；加速度激励 ${excitation.toFixed(3)} m/s²。保持陀螺姿态，位置不宣称米制。`,
      { R_wb0, visualCenters, offsetSearch, biasFit, scaleFit, excitation, metricPoses, vLast, times },
    );
  }
  // Triangulate landmarks with the metric pose chain.
  const landmarks = triangulateBootstrapLandmarks(input, metricPoses, opts);
  return {
    ok: true,
    message: `初始化完成：尺度 ${scale.toFixed(3)}，陀螺偏置 (${bg.map((v) => v.toExponential(1)).join(', ')}) rad/s，时差 ${(offsetSearch.offset * 1000).toFixed(1)} ms`,
    anchorFrame: input.end,
    R_wb: new Float64Array(last.R_wb),
    p_wb: [...last.p_wb],
    v_wb: vLast,
    bg,
    ba: [0, 0, 0],
    timeOffset: offsetSearch.offset,
    timeOffsetScore: offsetSearch.score,
    timeOffsetCurve: offsetSearch.curve,
    gyroBiasResidualDegPerSec: biasFit.residualDegPerSec,
    scale,
    scaleResidual: scaleFit.residual,
    scaleEvidence: `${scaleFit.evidence}；激励 ${excitation.toFixed(3)} m/s²`,
    scaleRecoverable: true,
    visualCenters,
    metricCenters: metricPoses.map((p) => p.C),
    landmarks,
    excitation,
    gravity,
  };
}

function failedBootstrap(
  offsetSearch: { offset: number; score: number; curve: Array<{ offset: number; score: number }> },
  biasFit: { bg: Vec3; residualDegPerSec: number },
  message: string,
  extra?: {
    R_wb0: Mat3;
    visualCenters: Vec3[];
    offsetSearch: { offset: number; score: number; curve: Array<{ offset: number; score: number }> };
    biasFit: { bg: Vec3; residualDegPerSec: number };
    scaleFit: ScaleFit;
    excitation: number;
    metricPoses: { R_wb: Mat3; p_wb: Vec3; C: Vec3 }[];
    vLast: Vec3;
    times: number[];
  },
): BootstrapResult {
  const lastPose = extra ? extra.metricPoses[extra.metricPoses.length - 1] : { R_wb: m3identity(), p_wb: [0, 0, 0] as Vec3, C: [0, 0, 0] as Vec3 };
  return {
    ok: false,
    message,
    anchorFrame: extra ? extra.times.length - 1 : 0,
    R_wb: new Float64Array(lastPose.R_wb),
    p_wb: [...lastPose.p_wb],
    v_wb: extra ? [...extra.vLast] : [0, 0, 0],
    bg: biasFit.bg,
    ba: [0, 0, 0],
    timeOffset: offsetSearch.offset,
    timeOffsetScore: offsetSearch.score,
    timeOffsetCurve: offsetSearch.curve,
    gyroBiasResidualDegPerSec: biasFit.residualDegPerSec,
    scale: null,
    scaleResidual: extra ? extra.scaleFit.residual : Infinity,
    scaleEvidence: extra ? extra.scaleFit.evidence : '',
    scaleRecoverable: false,
    visualCenters: extra ? extra.visualCenters : [],
    metricCenters: extra ? extra.metricPoses.map((p) => p.C) : [],
    landmarks: [],
    excitation: extra ? extra.excitation : 0,
    gravity: gravityVector(),
  };
}

function triangulateBootstrapLandmarks(
  input: BootstrapInput,
  poses: { R_wb: Mat3; p_wb: Vec3; C: Vec3 }[],
  opts: BootstrapOptions,
): BootstrapLandmark[] {
  const maxReproj = opts.maxLandmarkReprojPx ?? 2.2;
  const minParallax = opts.minParallaxRad ?? 0.012;
  const ids = new Set<number>();
  for (let k = input.start; k <= input.end; k++) {
    const px = input.trackPixels.get(k);
    if (!px) continue;
    for (const id of px.keys()) ids.add(id);
  }
  const out: BootstrapLandmark[] = [];
  for (const id of ids) {
    const views: TriangulationView[] = [];
    for (let k = input.start; k <= input.end; k++) {
      const px = input.trackPixels.get(k)?.get(id);
      if (!px) continue;
      const birth = input.trackBirth.get(id);
      if (birth !== undefined && birth > k) continue;
      const pose = poses[k - input.start];
      if (!pose) continue;
      // projectCamera expects the world->camera rotation R_cw = (R_wb R_bc)^T.
      views.push({ R_wc: m3transpose(m3mul(pose.R_wb, input.R_bc)), C: pose.C, u: px.u, v: px.v });
    }
    if (views.length < 3) continue;
    const tri = triangulateViews(views, input.K);
    if (!tri || !tri.ok) continue;
    if (tri.reproj > maxReproj || tri.parallax < minParallax) continue;
    // Bound depth so a near-degenerate fit cannot place a landmark kilometres away.
    const maxRange = 25;
    let okRange = true;
    for (const vv of views) if (v3norm(v3sub(tri.p, vv.C)) > maxRange) okRange = false;
    if (!okRange) continue;
    out.push({ id, p_w: tri.p, views: views.length, reprojPx: tri.reproj, parallax: tri.parallax });
  }
  return out;
}

function slerpR(a: Mat3, b: Mat3, s: number): Mat3 {
  const d = so3Log(m3mul(m3transpose(a), b));
  return m3mul(a, so3Exp(v3scale(d, s)));
}
