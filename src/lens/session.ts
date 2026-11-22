import { Mat3, Vec3, m3mul, m3transpose, so3Exp, v3add, v3scale, v3sub } from '../core/math';
import { Extrinsics, Intrinsics } from '../core/camera';
import { ImuSample } from '../sim/trajectory';
import { FrameFeatures, TrackPoint } from '../vision/tracker';
import { estimateRotationFirst } from '../estimation/relpose';
import { integrateRotation } from '../estimation/imu';
import { RelativeVisualPose } from './bootstrap';

// A sensor-only session: everything the kernel is allowed to see. Built from the
// synthetic capture (image features, raw IMU, timestamps) with no truth access.

export interface TrackObs {
  id: number;
  u: number;
  v: number;
  prevU: number;
  prevV: number;
  age: number;
  confidence: number;
}

export interface SensorSession {
  cameraTimes: number[];
  imu: ImuSample[];
  K: Intrinsics;
  /** Approximate / user-supplied extrinsic handed to the kernel. */
  initialExtrinsics: Extrinsics;
  /** Per-frame KLT tracks. */
  tracks: TrackObs[][];
  /** Per-frame detector output (keypoints + descriptors). */
  vision: FrameFeatures[];
  /** Image-only relative pose for k-1 -> k, when solvable. */
  relativePoses: Map<number, RelativeVisualPose>;
  /** Median forward-backward flow residual per frame, pixels. */
  flowResidual: number[];
  /** Interpolated IMU between physical times (camera clock). */
  imuBetween(t0: number, t1: number, offset: number): ImuSample[];
  /** Body-frame rotation between two camera frames, bias removed. */
  gyroRotation(fromK: number, toK: number, offset: number, bg: Vec3): Mat3 | null;
}

export interface SessionBuildOptions {
  givenExtrinsics: Extrinsics;
  pixelNoise: number;
  seed: number;
}

export function buildSession(
  cameraTimes: number[],
  imu: ImuSample[],
  features: FrameFeatures[],
  trackSets: TrackPoint[][],
  flowResidual: number[],
  K: Intrinsics,
  opts: SessionBuildOptions,
): SensorSession {
  const tracks: TrackObs[][] = trackSets.map((set) =>
    set.map((t) => ({ id: t.id, u: t.u, v: t.v, prevU: t.prevU, prevV: t.prevV, age: t.age, confidence: t.confidence })),
  );
  // Image-only relative poses. This only uses image measurements; scale is
  // arbitrary by construction and is not pretended to be metric.
  const relativePoses = new Map<number, RelativeVisualPose>();
  for (let k = 1; k < trackSets.length; k++) {
    const corr: Array<{ u1: number; v1: number; u2: number; v2: number }> = [];
    for (const tr of trackSets[k]) {
      if (tr.age > 0 && tr.confidence > 0.5) corr.push({ u1: tr.prevU, v1: tr.prevV, u2: tr.u, v2: tr.v });
    }
    if (corr.length < 10) continue;
    const rel = estimateRotationFirst(corr, K, {
      ransacIterations: 140,
      minInliers: 10,
      seed: opts.seed + k * 37,
      inlierRad: 0.005,
    });
    if (!rel.ok) continue;
    if (rel.medianErrorPx > 2.2) continue;
    relativePoses.set(k, {
      R: rel.R,
      t: rel.t,
      inliers: rel.inlierCount,
      parallax: rel.parallax,
      medianErrorPx: rel.medianErrorPx,
    });
  }

  const imuBetween = (t0: number, t1: number, offset: number): ImuSample[] => {
    const a0 = t0 - offset;
    const a1 = t1 - offset;
    const out: ImuSample[] = [];
    if (a1 <= a0) return out;
    let prev: ImuSample | null = null;
    for (const s of imu) {
      if (s.t <= a0) {
        prev = s;
        continue;
      }
      if (s.t >= a1) {
        // Close the interval at a1. The a0 boundary is only added here if the
        // interval contained no interior sample (short gaps between IMU ticks).
        if (out.length === 0) out.push(interpImu(prev ?? s, s, a0));
        out.push(interpImu(prev ?? s, s, a1));
        return out;
      }
      if (out.length === 0) out.push(interpImu(prev ?? s, s, a0));
      out.push(s);
    }
    if (out.length === 0 && prev) out.push(prev);
    return out;
  };

  const gyroRotation = (fromK: number, toK: number, offset: number, bg: Vec3): Mat3 | null => {
    if (fromK < 0 || toK >= cameraTimes.length || toK <= fromK) return null;
    const r = integrateRotation(imu, cameraTimes[fromK], cameraTimes[toK], offset, bg);
    if (r.samples < 2) return null;
    return r.dR;
  };

  return {
    cameraTimes,
    imu,
    K,
    initialExtrinsics: {
      R_bc: new Float64Array(opts.givenExtrinsics.R_bc),
      t_bc: [...opts.givenExtrinsics.t_bc],
    },
    tracks,
    vision: features,
    relativePoses,
    flowResidual,
    imuBetween,
    gyroRotation,
  };
}

function interpImu(a: ImuSample, b: ImuSample, t: number): ImuSample {
  if (Math.abs(b.t - a.t) < 1e-12) return { t, gyro: a.gyro, accel: a.accel };
  const s = (t - a.t) / (b.t - a.t);
  return {
    t,
    gyro: v3add(a.gyro, v3scale(v3sub(b.gyro, a.gyro), s)),
    accel: v3add(a.accel, v3scale(v3sub(b.accel, a.accel), s)),
  };
}

// Small helpers so the session can be transformed without touching truth.
export function rotateBy(R: Mat3, v: Vec3): Vec3 {
  return [R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2]];
}
export const cameraRelFromBody = (R_bc: Mat3, dR_body: Mat3): Mat3 =>
  m3mul(m3transpose(R_bc), m3mul(dR_body, R_bc));
export const bodyRelFromCamera = (R_bc: Mat3, dR_cam: Mat3): Mat3 =>
  m3mul(R_bc, m3mul(dR_cam, m3transpose(R_bc)));
export const rotExp = so3Exp;
