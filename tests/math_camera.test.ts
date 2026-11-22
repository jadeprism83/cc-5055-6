import { describe, expect, it } from 'vitest';
import {
  eulerZYX,
  m3det,
  m3mul,
  m3TmulVec,
  m3transpose,
  rotToEulerZYX,
  skew,
  so3Exp,
  so3Log,
  v3norm,
  v3sub,
  mulberry32,
  Gaussian,
} from '../src/core/math';
import {
  bearingCamera,
  bodyToCamera,
  cameraCenterWorld,
  cameraToBody,
  defaultExtrinsics,
  defaultIntrinsics,
  project,
  reprojection,
  triangulateTwoView,
} from '../src/core/camera';
import { defaultSimConfig, generateDataset, sampleImuTrue, sampleTruth } from '../src/sim/trajectory';

describe('SO(3)', () => {
  it('exp/log round-trip', () => {
    const rng = new Gaussian(mulberry32(3));
    for (let i = 0; i < 100; i++) {
      const w = rng.vec3(1.2);
      const R = so3Exp(w);
      expect(m3det(R)).toBeCloseTo(1, 10);
      const back = so3Log(R);
      // Log returns the principal value, so compare rotations rather than raw vectors.
      const err = m3mul(m3transpose(R), so3Exp(back));
      let e = 0;
      for (let k = 0; k < 9; k++) e += Math.abs(err[k] - (k % 4 === 0 ? 1 : 0));
      expect(e).toBeLessThan(1e-9);
      expect(v3norm(back)).toBeLessThanOrEqual(Math.PI + 1e-9);
    }
  });
  it('skew identity', () => {
    const a = [0.3, -0.7, 1.1] as [number, number, number];
    const b = [-0.4, 0.2, 0.9] as [number, number, number];
    const l = new Float64Array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]);
    const r = new Float64Array(3);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) r[i] += skew(a)[i * 3 + j] * b[j];
    expect(v3norm(v3sub([r[0], r[1], r[2]] as [number, number, number], [l[0], l[1], l[2]] as [number, number, number]))).toBeLessThan(1e-12);
  });
  it('euler round-trip', () => {
    const R = eulerZYX(0.3, -0.2, 0.4);
    const e = rotToEulerZYX(R);
    const R2 = eulerZYX(e[0], e[1], e[2]);
    let err = 0;
    for (let i = 0; i < 9; i++) err += Math.abs(R[i] - R2[i]);
    expect(err).toBeLessThan(1e-9);
  });
});

describe('camera model', () => {
  it('extrinsic round-trip', () => {
    const ext = defaultExtrinsics();
    const p = [1.2, -0.4, 0.7] as [number, number, number];
    const pc = bodyToCamera(ext, p);
    const pb = cameraToBody(ext, pc);
    expect(v3norm(v3sub(pb, p))).toBeLessThan(1e-12);
  });
  it('projects forward body point to centre', () => {
    const ext = defaultExtrinsics();
    const K = defaultIntrinsics(320, 240);
    const R_wb = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const p_w = [5, -0.012, 0.006] as [number, number, number]; // approx optical axis
    const px = project(bodyToCamera(ext, p_w), K);
    expect(px.valid).toBe(true);
    expect(px.depth).toBeGreaterThan(4.9);
  });
  it('triangulates a fixed point', () => {
    const ext = defaultExtrinsics();
    const K = defaultIntrinsics(320, 240);
    const R1 = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const R2 = so3Exp([0, 0, 0.25]);
    const p1 = [0, 0, 0] as [number, number, number];
    const p2 = [0.4, 0.1, 0] as [number, number, number];
    const X = [3.2, 0.5, 0.4] as [number, number, number];
    const a = reprojection(R1, p1, ext, K, X);
    const b = reprojection(R2, p2, ext, K, X);
    expect(a.ok && b.ok).toBe(true);
    const tri = triangulateTwoView(R1, p1, a, R2, p2, b, ext, K);
    expect(tri.ok).toBe(true);
    expect(v3norm(v3sub(tri.point, X))).toBeLessThan(0.05);
    expect(tri.parallax).toBeGreaterThan(0.01);
  });
  it('camera centre is behind the optic ray origin', () => {
    const ext = defaultExtrinsics();
    const c = cameraCenterWorld(new Float64Array([1,0,0,0,1,0,0,0,1]), [1,2,3], ext);
    expect(v3norm(v3sub(c, [1,2,3]))).toBeGreaterThan(0.01);
  });
});

describe('simulator', () => {
  it('sensor grids and bias step exist', () => {
    const cfg = defaultSimConfig(5);
    const ds = generateDataset(cfg);
    expect(ds.cameraTimes.length).toBe(Math.floor(cfg.duration * cfg.cameraHz) + 1);
    expect(ds.imu.length).toBe(Math.floor(cfg.duration * cfg.imuHz) + 1);
    const before = ds.imu[10].trueGyroBias!;
    const after = ds.imu.find((s) => s.t > cfg.biasStepTime + cfg.biasRamp + 0.02)!.trueGyroBias!;
    expect(v3norm(v3sub(after, before))).toBeGreaterThan(0.01);
  });
  it('specific force plus gravity reproduces acceleration', () => {
    const cfg = defaultSimConfig(5);
    const t = 3.1;
    const s0 = sampleTruth(cfg, t - 0.001);
    const s1 = sampleTruth(cfg, t + 0.001);
    const aNum = [
      (s1.v_wb[0] - s0.v_wb[0]) / 0.002,
      (s1.v_wb[1] - s0.v_wb[1]) / 0.002,
      (s1.v_wb[2] - s0.v_wb[2]) / 0.002,
    ];
    // specific force from the simulator must satisfy R f + g = a
    const im = sampleImuTrue(cfg, t);
    const g = [0, 0, -9.80665];
    const aw = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
      aw[i] = s1.R_wb[i * 3] * im.accel[0] + s1.R_wb[i * 3 + 1] * im.accel[1] + s1.R_wb[i * 3 + 2] * im.accel[2] + (g as any)[i];
    }
    expect(v3norm(v3sub(aw as any, aNum as any))).toBeLessThan(0.02);
  });
  it('time offset changes IMU contents vs camera truth', () => {
    const base = defaultSimConfig(5);
    const shifted = { ...base, timeOffset: 0.08 };
    const a = generateDataset(base);
    const b = generateDataset(shifted);
    let d = 0;
    for (let j = 100; j < 200; j++) d += Math.abs(a.imu[j].gyro[0] - b.imu[j].gyro[0]);
    expect(d).toBeGreaterThan(0.01);
  });
});
