import { Gaussian, Mat3, Vec3, m3mulVec, mulberry32, v3add, v3cross, v3dot, v3norm, v3scale } from '../core/math';

export interface DropoutWindow {
  start: number;
  end: number;
  kind: 'lost' | 'blur';
}

export interface SimConfig {
  seed: number;
  duration: number;
  cameraHz: number;
  imuHz: number;
  /** Physical IMU time = nominal IMU stamp + timeOffset (seconds). */
  timeOffset: number;
  /** Camera-to-body (IMU) rotation, truth. */
  R_bc: Mat3;
  /** Camera origin in body frame, truth. */
  t_bc: Vec3;
  gyroBiasBefore: Vec3;
  gyroBiasAfter: Vec3;
  accelBias: Vec3;
  biasStepTime: number;
  biasRamp: number;
  gyroNoise: number;
  accelNoise: number;
  gyroWalk: number;
  accelWalk: number;
  pixelNoise: number;
  dropouts: DropoutWindow[];
  blurExposure: number;
  motionScale: number;
  /** Motion ramp start/end: the rig is still before `motionStart`. */
  motionStart: number;
  motionRamp: number;
}

export interface CameraSampleTrue {
  t: number;
  R_wb: Mat3;
  p_wb: Vec3;
  v_wb: Vec3;
}

export interface ImuSample {
  t: number;
  gyro: Vec3;
  accel: Vec3;
  trueGyroBias?: Vec3;
  trueAccelBias?: Vec3;
}

export interface TrajectoryTruth {
  cameras: CameraSampleTrue[];
  imu: ImuSample[];
  duration: number;
  gravity: Vec3;
}

export function defaultSimConfig(seed = 7): SimConfig {
  return {
    seed,
    duration: 14,
    cameraHz: 20,
    imuHz: 200,
    timeOffset: 0.018,
    R_bc: new Float64Array([0, 0, 1, -1, 0, 0, 0, -1, 0]),
    t_bc: [0.015, -0.012, 0.006],
    gyroBiasBefore: [0.0004, -0.00025, 0.0003],
    gyroBiasAfter: [0.019, -0.014, 0.012],
    accelBias: [0.028, -0.034, 0.019],
    biasStepTime: 5.0,
    biasRamp: 0.2,
    gyroNoise: 0.0011,
    accelNoise: 0.008,
    gyroWalk: 0.000012,
    accelWalk: 0.0001,
    pixelNoise: 0.3,
    dropouts: [
      { start: 4.6, end: 6.0, kind: 'blur' },
      { start: 7.0, end: 8.4, kind: 'lost' },
    ],
    blurExposure: 0.08,
    motionScale: 1,
    motionStart: 0.7,
    motionRamp: 0.8,
  };
}

export function gravityWorld(): Vec3 {
  return [0, 0, -9.80665];
}

interface Motion {
  R(t: number): Mat3;
  p(t: number): Vec3;
  v(t: number): Vec3;
  omegaB(t: number): Vec3;
  accelB(t: number): Vec3;
}

function smoothstep(x: number): number {
  const t = Math.min(1, Math.max(0, x));
  return t * t * (3 - 2 * t);
}

// Handheld motion with reversals and rotation. The rig is still for the first
// `motionStart` seconds so accelerometer/gyro initialisation is honest. Body
// rates and specific force are finite differences of the truth trajectory,
// which keeps the ground truth independent of hand-written derivatives.
function buildMotion(cfg: SimConfig): Motion {
  const scale = cfg.motionScale;
  const a = 0.55 * scale;
  const b = 0.22 * scale;
  const c = 0.16 * scale;
  const w1 = 1.05;
  const w2 = 1.7;
  const w3 = 0.8;
  const ramp = (t: number) => smoothstep((t - cfg.motionStart) / cfg.motionRamp);

  const R = (t: number): Mat3 => {
    const r = ramp(t);
    const yaw = r * (0.36 * Math.sin(w1 * t) + 0.09 * Math.sin(w2 * t + 0.7));
    const pitch = r * (0.08 * Math.sin(w3 * t + 0.2) + 0.04 * Math.sin(1.9 * t));
    const roll = r * (0.07 * Math.sin(1.35 * t + 1.1) + 0.03 * Math.sin(2.6 * t));
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const cr = Math.cos(roll), sr = Math.sin(roll);
    return new Float64Array([
      cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
      sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
      -sp, cp * sr, cp * cr,
    ]);
  };
  const p = (t: number): Vec3 => {
    const r = ramp(t);
    return [
      r * (a * Math.sin(w1 * t) + b * Math.sin(w2 * t + 0.4)),
      r * (c * (1 - Math.cos(w3 * t)) + 0.05 * Math.sin(2.2 * t)),
      r * (a * (1 - Math.cos(w3 * t)) + 0.12 * Math.sin(w1 * t + 0.8)),
    ];
  };
  const v = (t: number): Vec3 => {
    const h = 1e-4;
    const p0 = p(t - h);
    const p1 = p(t + h);
    return [(p1[0] - p0[0]) / (2 * h), (p1[1] - p0[1]) / (2 * h), (p1[2] - p0[2]) / (2 * h)];
  };
  const omegaB = (t: number): Vec3 => {
    const h = 1e-5;
    const R0 = R(t - h);
    const R1 = R(t + h);
    const Rt = R(t);
    const dR = new Float64Array(9);
    for (let i = 0; i < 9; i++) dR[i] = (R1[i] - R0[i]) / (2 * h);
    // Body-frame angular rate: Rdot = R [w]_x, so [w]_x = R^T Rdot. This matches
    // accelB, which is also body-frame; a gyro and accelerometer on the same
    // rigid body report in the same (body) axes.
    const W = mul3(transpose3(Rt), dR);
    return [(W[7] - W[5]) / 2, (W[2] - W[6]) / 2, (W[3] - W[1]) / 2];
  };
  const accelB = (t: number): Vec3 => {
    const h = 1e-4;
    const v0 = v(t - h);
    const v1 = v(t + h);
    const aw: Vec3 = [(v1[0] - v0[0]) / (2 * h), (v1[1] - v0[1]) / (2 * h), (v1[2] - v0[2]) / (2 * h)];
    const g = gravityWorld();
    return m3mulVec(transpose3(R(t)), [aw[0] - g[0], aw[1] - g[1], aw[2] - g[2]]);
  };
  return { R, p, v, omegaB, accelB };
}

export function transpose3(a: Mat3): Mat3 {
  return new Float64Array([a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]);
}
function mul3(a: Mat3, b: Mat3): Mat3 {
  const m = new Float64Array(9);
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      m[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
  return m;
}

export interface GeneratedDataset {
  config: SimConfig;
  cameraTimes: number[];
  imuTimes: number[];
  imu: ImuSample[];
  cameraTruth: CameraSampleTrue[];
  truth: TrajectoryTruth;
  R_bc: Mat3;
  t_bc: Vec3;
  /** Time at which motion actually starts, for UI capability labels. */
  staticUntil: number;
}

export function generateDataset(cfg: SimConfig): GeneratedDataset {
  const motion = buildMotion(cfg);
  const rng = new Gaussian(mulberry32(cfg.seed));
  const nCam = Math.floor(cfg.duration * cfg.cameraHz) + 1;
  const nImu = Math.floor(cfg.duration * cfg.imuHz) + 1;
  const cameraTimes: number[] = [];
  const cameraTruth: CameraSampleTrue[] = [];
  for (let k = 0; k < nCam; k++) {
    const t = k / cfg.cameraHz;
    cameraTimes.push(t);
    cameraTruth.push({ t, R_wb: motion.R(t), p_wb: motion.p(t), v_wb: motion.v(t) });
  }
  const imuTimes: number[] = [];
  const imu: ImuSample[] = [];
  let biasWG: Vec3 = [0, 0, 0];
  let biasWA: Vec3 = [0, 0, 0];
  const dtImu = 1 / cfg.imuHz;
  for (let j = 0; j < nImu; j++) {
    const tNominal = j / cfg.imuHz;
    const tPhys = tNominal + cfg.timeOffset;
    imuTimes.push(tNominal);
    biasWG = v3add(biasWG, rng.vec3(cfg.gyroWalk * Math.sqrt(dtImu)));
    biasWA = v3add(biasWA, rng.vec3(cfg.accelWalk * Math.sqrt(dtImu)));
    const stepFrac = Math.min(1, Math.max(0, (tPhys - cfg.biasStepTime) / cfg.biasRamp));
    const biasG: Vec3 = [
      biasWG[0] + cfg.gyroBiasBefore[0] + (cfg.gyroBiasAfter[0] - cfg.gyroBiasBefore[0]) * stepFrac,
      biasWG[1] + cfg.gyroBiasBefore[1] + (cfg.gyroBiasAfter[1] - cfg.gyroBiasBefore[1]) * stepFrac,
      biasWG[2] + cfg.gyroBiasBefore[2] + (cfg.gyroBiasAfter[2] - cfg.gyroBiasBefore[2]) * stepFrac,
    ];
    const biasA: Vec3 = [
      biasWA[0] + cfg.accelBias[0],
      biasWA[1] + cfg.accelBias[1],
      biasWA[2] + cfg.accelBias[2],
    ];
    const wTrue = motion.omegaB(tPhys);
    const fTrue = motion.accelB(tPhys);
    const noiseG = rng.vec3(cfg.gyroNoise);
    const noiseA = rng.vec3(cfg.accelNoise);
    imu.push({
      t: tNominal,
      gyro: [wTrue[0] + biasG[0] + noiseG[0], wTrue[1] + biasG[1] + noiseG[1], wTrue[2] + biasG[2] + noiseG[2]],
      accel: [fTrue[0] + biasA[0] + noiseA[0], fTrue[1] + biasA[1] + noiseA[1], fTrue[2] + biasA[2] + noiseA[2]],
      trueGyroBias: biasG,
      trueAccelBias: biasA,
    });
  }
  const truth: TrajectoryTruth = { cameras: cameraTruth, imu, duration: cfg.duration, gravity: gravityWorld() };
  return {
    config: cfg,
    cameraTimes,
    imuTimes,
    imu,
    cameraTruth,
    truth,
    R_bc: cfg.R_bc,
    t_bc: cfg.t_bc,
    staticUntil: cfg.motionStart,
  };
}

export function sampleTruth(cfg: SimConfig, t: number): CameraSampleTrue {
  const motion = buildMotion(cfg);
  return { t, R_wb: motion.R(t), p_wb: motion.p(t), v_wb: motion.v(t) };
}

export function sampleImuTrue(cfg: SimConfig, tPhys: number): { omega: Vec3; accel: Vec3; gravity: Vec3 } {
  const motion = buildMotion(cfg);
  return { omega: motion.omegaB(tPhys), accel: motion.accelB(tPhys), gravity: gravityWorld() };
}

export function cameraCenterOf(R_wb: Mat3, p_wb: Vec3, t_bc: Vec3): Vec3 {
  return v3add(p_wb, m3mulVec(R_wb, t_bc));
}

export function isInDropout(cfg: SimConfig, t: number): DropoutWindow | null {
  for (const d of cfg.dropouts) if (t >= d.start && t <= d.end) return d;
  return null;
}
