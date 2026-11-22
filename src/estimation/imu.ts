import { Mat3, Vec3, m3identity, m3mul, so3Exp, v3add, v3scale, v3sub } from '../core/math';
import { ImuSample } from '../sim/trajectory';

export interface ImuSlice {
  /** Interpolated samples, including both boundaries, in nominal stamp order. */
  samples: Array<{ t: number; gyro: Vec3; accel: Vec3 }>;
  /** Physical interval actually covered. */
  t0: number;
  t1: number;
}

function lerpSample(a: ImuSample, b: ImuSample, t: number): { t: number; gyro: Vec3; accel: Vec3 } {
  if (b.t - a.t < 1e-12) return { t, gyro: a.gyro, accel: a.accel };
  const s = (t - a.t) / (b.t - a.t);
  return {
    t,
    gyro: v3add(a.gyro, v3scale(v3sub(b.gyro, a.gyro), s)),
    accel: v3add(a.accel, v3scale(v3sub(b.accel, a.accel), s)),
  };
}

/**
 * Slice the IMU stream between physical times t0 and t1. Raw stamps are shifted
 * by `timeOffset` before comparison; the returned samples keep raw stamp times
 * so angular integration is unaffected by the search.
 */
export function imuSlice(imu: ImuSample[], t0: number, t1: number, timeOffset: number): ImuSlice {
  const samples: Array<{ t: number; gyro: Vec3; accel: Vec3 }> = [];
  const a0 = t0 - timeOffset;
  const a1 = t1 - timeOffset;
  if (a1 <= a0) return { samples: [], t0, t1 };
  const inside: ImuSample[] = [];
  for (const s of imu) {
    if (s.t > a0 && s.t < a1) inside.push({ t: s.t, gyro: s.gyro, accel: s.accel, trueGyroBias: s.trueGyroBias, trueAccelBias: s.trueAccelBias });
  }
  // Boundary interpolation.
  const before0 = findBefore(imu, a0);
  const after1 = findAfter(imu, a1);
  if (before0) samples.push(lerpSample(before0[0], before0[1], a0));
  else samples.push({ t: a0, gyro: imu[0].gyro, accel: imu[0].accel });
  for (const s of inside) samples.push(s);
  if (after1) samples.push(lerpSample(after1[0], after1[1], a1));
  else {
    const last = imu[imu.length - 1];
    samples.push({ t: a1, gyro: last.gyro, accel: last.accel });
  }
  return { samples, t0, t1 };
}

function findBefore(imu: ImuSample[], t: number): [ImuSample, ImuSample] | null {
  for (let i = 1; i < imu.length; i++) {
    if (imu[i].t >= t) return [imu[i - 1], imu[i]];
  }
  return null;
}
function findAfter(imu: ImuSample[], t: number): [ImuSample, ImuSample] | null {
  for (let i = 1; i < imu.length; i++) {
    if (imu[i].t >= t) return [imu[i - 1], imu[i]];
  }
  return null;
}

export interface DeltaRotation {
  dR: Mat3;
  /** Integrated effective time, seconds. */
  dt: number;
  samples: number;
}

/** Body-frame rotation change from physical t0 to t1, gyro bias removed. */
export function integrateRotation(imu: ImuSample[], t0: number, t1: number, timeOffset: number, bias: Vec3): DeltaRotation {
  const slice = imuSlice(imu, t0, t1, timeOffset);
  let dR = m3identity();
  let dt = 0;
  for (let i = 1; i < slice.samples.length; i++) {
    const a = slice.samples[i - 1];
    const b = slice.samples[i];
    const h = b.t - a.t;
    if (h <= 0) continue;
    const w = v3scale(v3add(a.gyro, b.gyro), 0.5);
    const wc: Vec3 = [w[0] - bias[0], w[1] - bias[1], w[2] - bias[2]];
    dR = m3mul(dR, so3Exp(v3scale(wc, h)));
    dt += h;
  }
  return { dR, dt, samples: Math.max(0, slice.samples.length - 1) };
}
