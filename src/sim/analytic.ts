import { Mat3, Vec3, eulerZYX, m3TmulVec, m3mul, so3Exp, v3scale } from '../core/math';
import { gravityWorld } from './trajectory';

// Analytic motion generator: closed-form trajectories whose body rates and
// specific force are exact formulas (no finite differences). Used to verify
// the documented convention examples and to feed the maintenance checks with
// data whose attitude trajectory and reprojection are known exactly.

export interface AnalyticMotion {
  R(t: number): Mat3;
  p(t: number): Vec3;
  v(t: number): Vec3;
  /** Body-frame angular rate, rad/s. */
  omegaB(t: number): Vec3;
  /** Body-frame specific force, m/s². */
  accelB(t: number): Vec3;
  duration: number;
}

export interface SinusoidParams {
  yawAmp: number;
  yawRate: number;
  yawPhase: number;
  pitchAmp: number;
  pitchRate: number;
  pitchPhase: number;
  rollAmp: number;
  rollRate: number;
  rollPhase: number;
  pathRadius: number;
  pathRate: number;
  zAmp: number;
  zRate: number;
  duration: number;
}

export function defaultSinusoidParams(): SinusoidParams {
  return {
    yawAmp: 0.5,
    yawRate: 0.9,
    yawPhase: 0.3,
    pitchAmp: 0.35,
    pitchRate: 1.3,
    pitchPhase: 1.1,
    rollAmp: 0.25,
    rollRate: 1.7,
    rollPhase: 2.0,
    pathRadius: 0.8,
    pathRate: 0.7,
    zAmp: 0.3,
    zRate: 1.1,
    duration: 4,
  };
}

/**
 * Smooth handheld-like motion: ZYX Euler angles are sinusoids, the translation
 * is a circle with a vertical bob. Body rates use the exact Euler-rate
 * Jacobian; specific force is Rᵀ(a − g) with the analytic acceleration.
 */
export function sinusoidMotion(params: SinusoidParams = defaultSinusoidParams()): AnalyticMotion {
  const P = params;
  const yaw = (t: number) => P.yawAmp * Math.sin(P.yawRate * t + P.yawPhase);
  const dyaw = (t: number) => P.yawAmp * P.yawRate * Math.cos(P.yawRate * t + P.yawPhase);
  const pitch = (t: number) => P.pitchAmp * Math.sin(P.pitchRate * t + P.pitchPhase);
  const dpitch = (t: number) => P.pitchAmp * P.pitchRate * Math.cos(P.pitchRate * t + P.pitchPhase);
  const roll = (t: number) => P.rollAmp * Math.sin(P.rollRate * t + P.rollPhase);
  const droll = (t: number) => P.rollAmp * P.rollRate * Math.cos(P.rollRate * t + P.rollPhase);
  const R = (t: number): Mat3 => eulerZYX(yaw(t), pitch(t), roll(t));
  const omegaB = (t: number): Vec3 => {
    const th = pitch(t);
    const ph = roll(t);
    // Exact ZYX Euler-rate mapping: ω_b = [φ̇ − ψ̇ sinθ, θ̇ cosφ + ψ̇ sinφ cosθ, −θ̇ sinφ + ψ̇ cosφ cosθ].
    return [
      droll(t) - dyaw(t) * Math.sin(th),
      dpitch(t) * Math.cos(ph) + dyaw(t) * Math.sin(ph) * Math.cos(th),
      -dpitch(t) * Math.sin(ph) + dyaw(t) * Math.cos(ph) * Math.cos(th),
    ];
  };
  const p = (t: number): Vec3 => [
    P.pathRadius * (Math.cos(P.pathRate * t) - 1),
    P.pathRadius * Math.sin(P.pathRate * t),
    P.zAmp * Math.sin(P.zRate * t),
  ];
  const v = (t: number): Vec3 => [
    -P.pathRadius * P.pathRate * Math.sin(P.pathRate * t),
    P.pathRadius * P.pathRate * Math.cos(P.pathRate * t),
    P.zAmp * P.zRate * Math.cos(P.zRate * t),
  ];
  const accelW = (t: number): Vec3 => [
    -P.pathRadius * P.pathRate * P.pathRate * Math.cos(P.pathRate * t),
    -P.pathRadius * P.pathRate * P.pathRate * Math.sin(P.pathRate * t),
    -P.zAmp * P.zRate * P.zRate * Math.sin(P.zRate * t),
  ];
  const accelB = (t: number): Vec3 => {
    const a = accelW(t);
    const g = gravityWorld();
    return m3TmulVec(R(t), [a[0] - g[0], a[1] - g[1], a[2] - g[2]]);
  };
  return { R, p, v, omegaB, accelB, duration: P.duration };
}

export interface RotationPhase {
  /** Body-fixed axis the rig turns about during this phase. */
  axis: Vec3;
  /** Signed rate, rad/s. */
  rate: number;
  duration: number;
}

export interface RotationRunMotion extends AnalyticMotion {
  /** Attitude at the end of the ordered sequence. */
  finalR: Mat3;
}

/**
 * Ordered rotations about known body axes. Within each phase the body rate is
 * constant, so R(t) = R_phaseStart · Exp(ω (t − t_start)) exactly. The order
 * matters: a mirrored axis convention integrates the same samples to a
 * different final attitude, which is what excludes the mirror solution.
 */
export function orderedRotationRun(phases: RotationPhase[]): RotationRunMotion {
  const bounds: Array<{ t0: number; t1: number; R0: Mat3; phase: RotationPhase }> = [];
  let t = 0;
  let R = so3Exp([0, 0, 0]);
  for (const phase of phases) {
    bounds.push({ t0: t, t1: t + phase.duration, R0: R, phase });
    R = m3mul(R, so3Exp(v3scale(phase.axis, phase.rate * phase.duration)));
    t += phase.duration;
  }
  const finalR = R;
  const Rat = (time: number): Mat3 => {
    for (const b of bounds) {
      if (time <= b.t1) return m3mul(b.R0, so3Exp(v3scale(b.phase.axis, b.phase.rate * Math.max(0, time - b.t0))));
    }
    return finalR;
  };
  const omegaB = (time: number): Vec3 => {
    for (const b of bounds) {
      if (time >= b.t0 && time < b.t1) return v3scale(b.phase.axis, b.phase.rate);
    }
    return [0, 0, 0];
  };
  const accelB = (time: number): Vec3 => {
    const g = gravityWorld();
    const Ri = Rat(time);
    return m3TmulVec(Ri, [-g[0], -g[1], -g[2]]);
  };
  const zero: Vec3 = [0, 0, 0];
  return {
    R: Rat,
    p: () => [...zero] as Vec3,
    v: () => [...zero] as Vec3,
    omegaB,
    accelB,
    duration: t,
    finalR,
  };
}

/** Constant-rate spin about one axis; after 2π the attitude closes the loop. */
export function closedLoopSpin(axis: Vec3, rate: number): RotationRunMotion {
  return orderedRotationRun([{ axis, rate, duration: (2 * Math.PI) / rate }]);
}
