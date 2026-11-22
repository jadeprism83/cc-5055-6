import {
  Mat,
  Mat3,
  Vec3,
  clamp,
  mat,
  matAdd,
  matEye,
  matGet,
  matInverse,
  matMul,
  matScale,
  matSet,
  matSub,
  matSymmetrize,
  matTranspose,
  matZero,
  m3identity,
  m3mul,
  m3mulVec,
  m3transpose,
  skew,
  so3Exp,
  so3Log,
  v3add,
  v3norm,
  v3scale,
  v3sub,
} from '../core/math';
import { Extrinsics, Intrinsics, project } from '../core/camera';

export interface EskfNoise {
  gyro: number;
  accel: number;
  gyroWalk: number;
  accelWalk: number;
  pixel: number;
  gravity: Vec3;
}

// Error-state ordering: [dtheta(0..2), dv(3..5), dp(6..8), dbg(9..11), dba(12..14)]
export const IDX_THETA = 0;
export const IDX_V = 3;
export const IDX_P = 6;
export const IDX_BG = 9;
export const IDX_BA = 12;

export interface EskfState {
  R: Mat3;
  p: Vec3;
  v: Vec3;
  bg: Vec3;
  ba: Vec3;
}

export class Eskf {
  R: Mat3;
  p: Vec3;
  v: Vec3;
  bg: Vec3;
  ba: Vec3;
  P: Mat;
  noise: EskfNoise;
  /** Innovation history for the residual panel, in pixels. */
  lastVisualResiduals: number[] = [];

  constructor(state: EskfState, P: Mat, noise: EskfNoise) {
    this.R = new Float64Array(state.R);
    this.p = [...state.p];
    this.v = [...state.v];
    this.bg = [...state.bg];
    this.ba = [...state.ba];
    this.P = { r: P.r, c: P.c, d: new Float64Array(P.d) };
    this.noise = noise;
  }

  static initial(
    R0: Mat3,
    p0: Vec3,
    v0: Vec3,
    bg0: Vec3,
    ba0: Vec3,
    noise: EskfNoise,
    sigmas = { theta: 0.03, v: 0.2, p: 0.05, bg: 0.002, ba: 0.08 },
  ): Eskf {
    const P = mat(15, 15);
    for (let i = 0; i < 3; i++) {
      matSet(P, IDX_THETA + i, IDX_THETA + i, sigmas.theta ** 2);
      matSet(P, IDX_V + i, IDX_V + i, sigmas.v ** 2);
      matSet(P, IDX_P + i, IDX_P + i, sigmas.p ** 2);
      matSet(P, IDX_BG + i, IDX_BG + i, sigmas.bg ** 2);
      matSet(P, IDX_BA + i, IDX_BA + i, sigmas.ba ** 2);
    }
    return new Eskf({ R: R0, p: p0, v: v0, bg: bg0, ba: ba0 }, P, noise);
  }

  clone(): Eskf {
    const e = new Eskf({ R: this.R, p: this.p, v: this.v, bg: this.bg, ba: this.ba }, this.P, this.noise);
    e.lastVisualResiduals = [...this.lastVisualResiduals];
    return e;
  }

  sigmaTheta(): number {
    let m = 0;
    for (let i = 0; i < 3; i++) m = Math.max(m, matGet(this.P, IDX_THETA + i, IDX_THETA + i));
    return Math.sqrt(Math.max(0, m));
  }
  sigmaPosition(): number {
    let m = 0;
    for (let i = 0; i < 3; i++) m = Math.max(m, matGet(this.P, IDX_P + i, IDX_P + i));
    return Math.sqrt(Math.max(0, m));
  }
  sigmaGyroBias(): number {
    let m = 0;
    for (let i = 0; i < 3; i++) m = Math.max(m, matGet(this.P, IDX_BG + i, IDX_BG + i));
    return Math.sqrt(Math.max(0, m));
  }
  sigmaAccelBias(): number {
    let m = 0;
    for (let i = 0; i < 3; i++) m = Math.max(m, matGet(this.P, IDX_BA + i, IDX_BA + i));
    return Math.sqrt(Math.max(0, m));
  }

  predict(gyro: Vec3, accel: Vec3, dt: number): void {
    if (dt <= 0) return;
    // Gyro and accelerometer are both body-frame: strapdown right-multiplies the
    // increment R_new = R_old Exp(w dt), and specific force maps to the world
    // with R_new.
    const w: Vec3 = [gyro[0] - this.bg[0], gyro[1] - this.bg[1], gyro[2] - this.bg[2]];
    const f: Vec3 = [accel[0] - this.ba[0], accel[1] - this.ba[1], accel[2] - this.ba[2]];
    const dR = so3Exp(v3scale(w, dt));
    const Rprev = this.R;
    const Rnew = m3mul(Rprev, dR);
    const aWorld = v3add(m3mulVec(Rnew, f), this.noise.gravity);
    this.p = v3add(this.p, v3add(v3scale(this.v, dt), v3scale(aWorld, 0.5 * dt * dt)));
    this.v = v3add(this.v, v3scale(aWorld, dt));
    this.R = Rnew;

    const F = matEye(15);
    // Orientation error propagates by dR^T and the gyro bias error.
    const dRT = m3transpose(dR);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) matSet(F, IDX_THETA + r, IDX_THETA + c, dRT[r * 3 + c]);
      matSet(F, IDX_THETA + r, IDX_BG + r, -dt);
    }
    // Velocity error: -R [f]x dtheta - R dba.
    const RskewF = m3mul(Rnew, skew(f));
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        matSet(F, IDX_V + r, IDX_THETA + c, -RskewF[r * 3 + c] * dt);
        matSet(F, IDX_V + r, IDX_BA + c, -Rnew[r * 3 + c] * dt);
      }
    }
    for (let i = 0; i < 3; i++) matSet(F, IDX_P + i, IDX_V + i, dt);

    const FP = matMul(F, this.P);
    const FPFt = matMul(FP, matTranspose(F));
    const Q = mat(15, 15);
    const sG = this.noise.gyro * this.noise.gyro * dt;
    const sA = this.noise.accel * this.noise.accel * dt;
    const qG = this.noise.gyroWalk * this.noise.gyroWalk * dt;
    const qA = this.noise.accelWalk * this.noise.accelWalk * dt;
    for (let i = 0; i < 3; i++) {
      matSet(Q, IDX_THETA + i, IDX_THETA + i, sG);
      matSet(Q, IDX_V + i, IDX_V + i, sA);
      matSet(Q, IDX_BG + i, IDX_BG + i, qG);
      matSet(Q, IDX_BA + i, IDX_BA + i, qA);
    }
    this.P = matAdd(FPFt, Q);
    matSymmetrize(this.P);
  }

  /** Pixel reprojection update of one world landmark. Returns true if accepted. */
  updateLandmark(
    R_bc: Mat3,
    t_bc: Vec3,
    K: Intrinsics,
    Xw: Vec3,
    u: number,
    v: number,
    sigmaPx = this.noise.pixel,
    gate = 13.8,
  ): { accepted: boolean; residual: number; rejectedReason?: 'depth' | 'gate' } {
    const d = v3sub(Xw, this.p);
    const pb = m3mulVec(m3transpose(this.R), d);
    const pc = m3mulVec(m3transpose(R_bc), v3sub(pb, t_bc));
    if (pc[2] <= 0.15) return { accepted: false, residual: Infinity, rejectedReason: 'depth' };
    const z = project(pc, K);
    if (!z.valid) return { accepted: false, residual: Infinity, rejectedReason: 'depth' };
    const r0 = u - z.u;
    const r1 = v - z.v;
    const res = Math.hypot(r0, r1);
    const H = mat(2, 15);
    // d pc / d theta = R_bc^T [pb]x ; d pc / d p = -R_bc^T R^T
    const Jproj = new Float64Array(6);
    const iz = 1 / pc[2];
    Jproj[0] = K.fx * iz;
    Jproj[1] = 0;
    Jproj[2] = -K.fx * pc[0] * iz * iz;
    Jproj[3] = 0;
    Jproj[4] = K.fy * iz;
    Jproj[5] = -K.fy * pc[1] * iz * iz;
    const Rbct = m3transpose(R_bc);
    const A = m3mul(Rbct, skew(pb)); // dpc/dtheta
    const B = m3mul(Rbct, m3transpose(this.R)); // -dpc/dp
    for (let r = 0; r < 2; r++) {
      const j0 = Jproj[r * 3];
      const j1 = Jproj[r * 3 + 1];
      const j2 = Jproj[r * 3 + 2];
      for (let c = 0; c < 3; c++) {
        matSet(H, r, IDX_THETA + c, j0 * A[c] + j1 * A[3 + c] + j2 * A[6 + c]);
        matSet(H, r, IDX_P + c, -(j0 * B[c] + j1 * B[3 + c] + j2 * B[6 + c]));
      }
    }
    const S = matMul(matMul(H, this.P), matTranspose(H));
    matSet(S, 0, 0, matGet(S, 0, 0) + sigmaPx * sigmaPx);
    matSet(S, 1, 1, matGet(S, 1, 1) + sigmaPx * sigmaPx);
    const Sinv = matInverse(S);
    const d2 = r0 * (matGet(Sinv, 0, 0) * r0 + matGet(Sinv, 0, 1) * r1) + r1 * (matGet(Sinv, 1, 0) * r0 + matGet(Sinv, 1, 1) * r1);
    if (d2 > gate) return { accepted: false, residual: res, rejectedReason: 'gate' };
    const Kk = matMul(matMul(this.P, matTranspose(H)), Sinv);
    const dx = new Float64Array(15);
    for (let i = 0; i < 15; i++) dx[i] = matGet(Kk, i, 0) * r0 + matGet(Kk, i, 1) * r1;
    this.inject(dx);
    // Joseph form.
    const IKH = matSub(matEye(15), matMul(Kk, H));
    const Rm = mat(2, 2);
    matSet(Rm, 0, 0, sigmaPx * sigmaPx);
    matSet(Rm, 1, 1, sigmaPx * sigmaPx);
    const Pnew = matAdd(matMul(matMul(IKH, this.P), matTranspose(IKH)), matMul(matMul(Kk, Rm), matTranspose(Kk)));
    this.P = Pnew;
    matSymmetrize(this.P);
    this.lastVisualResiduals.push(res);
    if (this.lastVisualResiduals.length > 200) this.lastVisualResiduals.shift();
    return { accepted: true, residual: res };
  }

  /**
   * Relative rotation update from a keyframe. `z` is the camera-frame rotation
   * from keyframe to current: X_cur = z X_kf + t. Only the rotation is used.
   */
  updateRelativeRotation(
    R_bc: Mat3,
    zCam: Mat3,
    R_wb_key: Mat3,
    sigmaRad: number,
    gateChi2 = 11.8,
  ): { accepted: boolean; residual: number } {
    const pred = relativeRotationCamera(R_bc, R_wb_key, this.R);
    const r = so3Log(m3mul(m3transpose(zCam), pred));
    const resNorm = v3norm(r);
    // Numerical Jacobian wrt a right perturbation of the current body rotation.
    const H = mat(3, 15);
    const eps = 1e-6;
    for (let axis = 0; axis < 3; axis++) {
      const d = [0, 0, 0] as Vec3;
      d[axis] = eps;
      const Rpert = m3mul(this.R, so3Exp(d));
      const predPert = relativeRotationCamera(R_bc, R_wb_key, Rpert);
      const rp = so3Log(m3mul(m3transpose(zCam), predPert));
      for (let row = 0; row < 3; row++) matSet(H, row, IDX_THETA + axis, (rp[row] - r[row]) / eps);
    }
    const S = matMul(matMul(H, this.P), matTranspose(H));
    for (let i = 0; i < 3; i++) matSet(S, i, i, matGet(S, i, i) + sigmaRad * sigmaRad);
    const Sinv = matInverse(S);
    let d2 = 0;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) d2 += r[i] * matGet(Sinv, i, j) * r[j];
    if (d2 > gateChi2) return { accepted: false, residual: resNorm };
    const Kk = matMul(matMul(this.P, matTranspose(H)), Sinv);
    const dx = new Float64Array(15);
    for (let i = 0; i < 15; i++) for (let j = 0; j < 3; j++) dx[i] += matGet(Kk, i, j) * r[j];
    this.inject(dx);
    const IKH = matSub(matEye(15), matMul(Kk, H));
    const Rm = mat(3, 3);
    for (let i = 0; i < 3; i++) matSet(Rm, i, i, sigmaRad * sigmaRad);
    this.P = matAdd(matMul(matMul(IKH, this.P), matTranspose(IKH)), matMul(matMul(Kk, Rm), matTranspose(Kk)));
    matSymmetrize(this.P);
    return { accepted: true, residual: resNorm };
  }

  /** Absolute pose measurement from relocalisation, used as a soft update. */
  updatePose(
    pMeas: Vec3,
    RMeas: Mat3,
    sigmaPos: number,
    sigmaRot: number,
  ): { accepted: boolean; posResidual: number; rotResidual: number } {
    const rp = v3sub(pMeas, this.p);
    const Rerr = m3mul(m3transpose(this.R), RMeas);
    const rth = so3Log(Rerr);
    const H = mat(6, 15);
    for (let i = 0; i < 3; i++) {
      matSet(H, i, IDX_THETA + i, 1);
      matSet(H, 3 + i, IDX_P + i, 1);
    }
    const r = [rth[0], rth[1], rth[2], rp[0], rp[1], rp[2]];
    const S = matMul(matMul(H, this.P), matTranspose(H));
    for (let i = 0; i < 3; i++) {
      matSet(S, i, i, matGet(S, i, i) + sigmaRot * sigmaRot);
      matSet(S, 3 + i, 3 + i, matGet(S, 3 + i, 3 + i) + sigmaPos * sigmaPos);
    }
    let d2 = 0;
    const Sinv = matInverse(S);
    for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) d2 += r[i] * matGet(Sinv, i, j) * r[j];
    if (d2 > 24.3) return { accepted: false, posResidual: v3norm(rp), rotResidual: v3norm(rth) };
    const Kk = matMul(matMul(this.P, matTranspose(H)), Sinv);
    const dx = new Float64Array(15);
    for (let i = 0; i < 15; i++) for (let j = 0; j < 6; j++) dx[i] += matGet(Kk, i, j) * r[j];
    this.inject(dx);
    const IKH = matSub(matEye(15), matMul(Kk, H));
    const Rm = mat(6, 6);
    for (let i = 0; i < 3; i++) {
      matSet(Rm, i, i, sigmaRot * sigmaRot);
      matSet(Rm, 3 + i, 3 + i, sigmaPos * sigmaPos);
    }
    this.P = matAdd(matMul(matMul(IKH, this.P), matTranspose(IKH)), matMul(matMul(Kk, Rm), matTranspose(Kk)));
    matSymmetrize(this.P);
    return { accepted: true, posResidual: v3norm(rp), rotResidual: v3norm(rth) };
  }

  /** Reset a specific pose/velocity from relocalisation, keeping healthy covariance structure. */
  hardSetPose(p: Vec3, R: Mat3, v: Vec3, sigmaPos: number, sigmaRot: number): void {
    this.p = [...p];
    this.R = new Float64Array(R);
    this.v = [...v];
    for (let i = 0; i < 3; i++) {
      matSet(this.P, IDX_THETA + i, IDX_THETA + i, sigmaRot * sigmaRot);
      matSet(this.P, IDX_P + i, IDX_P + i, sigmaPos * sigmaPos);
      matSet(this.P, IDX_V + i, IDX_V + i, 0.15 ** 2);
    }
  }

  setGyroBias(bg: Vec3, sigma: number): void {
    this.bg = [...bg];
    for (let i = 0; i < 3; i++) matSet(this.P, IDX_BG + i, IDX_BG + i, sigma * sigma);
  }

  /**
   * Re-open the gyro-bias covariance after a jump is detected. Cross covariances
   * with biases are zeroed so the subsequent visual updates can pull them.
   */
  inflateGyroBias(sigma: number): void {
    for (let i = 0; i < 15; i++) {
      if (i >= IDX_BG && i < IDX_BG + 3) continue;
      for (let j = IDX_BG; j < IDX_BG + 3; j++) {
        matSet(this.P, i, j, 0);
        matSet(this.P, j, i, 0);
      }
    }
    for (let i = 0; i < 3; i++) matSet(this.P, IDX_BG + i, IDX_BG + i, sigma * sigma);
    matSymmetrize(this.P);
  }

  inject(dx: Float64Array): void {
    const dth: Vec3 = [dx[0], dx[1], dx[2]];
    this.R = m3mul(this.R, so3Exp(dth));
    this.v = v3add(this.v, [dx[3], dx[4], dx[5]]);
    this.p = v3add(this.p, [dx[6], dx[7], dx[8]]);
    this.bg = v3add(this.bg, [dx[9], dx[10], dx[11]]);
    this.ba = v3add(this.ba, [dx[12], dx[13], dx[14]]);
  }

  state(): EskfState {
    return { R: new Float64Array(this.R), p: [...this.p], v: [...this.v], bg: [...this.bg], ba: [...this.ba] };
  }
}

export function relativeRotationCamera(R_bc: Mat3, R_wb_key: Mat3, R_wb_cur: Mat3): Mat3 {
  // Camera-frame rotation of the key frame relative to the current frame:
  //   zCam = R_bc^T R_wb_key^T R_wb_cur R_bc = (R_bc^T R_wb_cur^T R_wb_key R_bc)^{-1}
  // This is the direction the two-view solver reports as x_cur = zCam x_key + t,
  // verified against the visual relative poses (0.2-0.3 deg agreement on
  // well-conditioned frames).
  return m3mul(m3transpose(R_bc), m3mul(m3mul(m3transpose(R_wb_cur), R_wb_key), R_bc));
}

export function rotationErrorAngle(Rest: Mat3, Rtruth: Mat3): number {
  const E = m3mul(m3transpose(Rtruth), Rest);
  const c = clamp((E[0] + E[4] + E[8] - 1) / 2, -1, 1);
  return Math.acos(c);
}

export function makeNoise(partial: Partial<EskfNoise>): EskfNoise {
  return {
    gyro: partial.gyro ?? 0.0015,
    accel: partial.accel ?? 0.012,
    gyroWalk: partial.gyroWalk ?? 2e-5,
    accelWalk: partial.accelWalk ?? 1.5e-4,
    pixel: partial.pixel ?? 0.6,
    gravity: partial.gravity ?? [0, 0, -9.80665],
  };
}

export function normaliseAngle(a: number): number {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}
