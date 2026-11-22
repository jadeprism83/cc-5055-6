import {
  Mat3,
  Vec3,
  m3det,
  m3mul,
  m3mulVec,
  m3TmulVec,
  m3transpose,
  so3Exp,
  v3norm,
  v3scale,
} from './math';

// Sensor conventions: how a firmware's raw gyro/accel readings map into the
// body frame used by the estimator (body: x forward, y left, z up). A
// convention is an axis map (signed permutation R_bs, v_body = R_bs · v_sensor)
// plus the physical units of the gyro and accelerometer channels. Everything
// here is pure description; the calibration record in lens/record.ts decides
// which convention is committed for which firmware.

export type GyroUnit = 'rad/s' | 'deg/s';
export type AccelUnit = 'm/s^2' | 'g';

export const GRAVITY_MAG = 9.80665;
export const RAD2DEG = 180 / Math.PI;

export interface SensorConvention {
  /** Firmware identifier, e.g. 'fw-A'. */
  firmware: string;
  /** v_body = R_bs · v_sensor. Signed permutation, |det| = 1. */
  R_bs: Mat3;
  gyroUnit: GyroUnit;
  accelUnit: AccelUnit;
}

/** Factory firmware: sensor axes coincide with the body, SI units. */
export function firmwareA(): SensorConvention {
  return {
    firmware: 'fw-A',
    R_bs: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
    gyroUnit: 'rad/s',
    accelUnit: 'm/s^2',
  };
}

/**
 * Replacement firmware: the chip is mounted rotated (sensor x = -body y,
 * sensor y = +body z, sensor z = -body x) and reports deg/s and g.
 */
export function firmwareB(): SensorConvention {
  return {
    firmware: 'fw-B',
    // Columns are the sensor axes expressed in the body frame.
    R_bs: new Float64Array([0, 0, -1, -1, 0, 0, 0, 1, 0]),
    gyroUnit: 'deg/s',
    accelUnit: 'g',
  };
}

function gyroScaleToBody(u: GyroUnit): number {
  return u === 'rad/s' ? 1 : 1 / RAD2DEG;
}
function accelScaleToBody(u: AccelUnit): number {
  return u === 'm/s^2' ? 1 : GRAVITY_MAG;
}

/** Raw gyro reading (sensor axes, sensor units) → body frame, rad/s. */
export function gyroToBody(conv: SensorConvention, w: Vec3): Vec3 {
  return m3mulVec(conv.R_bs, v3scale(w, gyroScaleToBody(conv.gyroUnit)));
}
/** Raw accel reading (sensor axes, sensor units) → body frame, m/s². */
export function accelToBody(conv: SensorConvention, a: Vec3): Vec3 {
  return m3mulVec(conv.R_bs, v3scale(a, accelScaleToBody(conv.accelUnit)));
}
/** Body-frame rate (rad/s) → raw gyro reading the firmware would report. */
export function gyroFromBody(conv: SensorConvention, wBody: Vec3): Vec3 {
  return m3TmulVec(conv.R_bs, v3scale(wBody, 1 / gyroScaleToBody(conv.gyroUnit)));
}
/** Body-frame specific force (m/s²) → raw accel reading the firmware would report. */
export function accelFromBody(conv: SensorConvention, aBody: Vec3): Vec3 {
  return m3TmulVec(conv.R_bs, v3scale(aBody, 1 / accelScaleToBody(conv.accelUnit)));
}

/**
 * Invertible converter between two firmware conventions. Takes a sample
 * expressed in `from` conventions (axes + units) and expresses the same
 * physical quantity in `to` conventions. `invertConverter` undoes it exactly.
 */
export interface AxisConverter {
  from: string;
  to: string;
  /** v_to(sensor axes) = R · v_from(sensor axes), axes only. */
  R: Mat3;
  /** Unit rescale applied after R for gyro / accel channels. */
  gyroScale: number;
  accelScale: number;
}

export function converterBetween(from: SensorConvention, to: SensorConvention): AxisConverter {
  return {
    from: from.firmware,
    to: to.firmware,
    R: m3mul(m3transpose(to.R_bs), from.R_bs),
    gyroScale: gyroScaleToBody(from.gyroUnit) / gyroScaleToBody(to.gyroUnit),
    accelScale: accelScaleToBody(from.accelUnit) / accelScaleToBody(to.accelUnit),
  };
}

export function invertConverter(c: AxisConverter): AxisConverter {
  return {
    from: c.to,
    to: c.from,
    R: m3transpose(c.R),
    gyroScale: 1 / c.gyroScale,
    accelScale: 1 / c.accelScale,
  };
}

export function convertGyroSample(c: AxisConverter, w: Vec3): Vec3 {
  return m3mulVec(c.R, v3scale(w, c.gyroScale));
}
export function convertAccelSample(c: AxisConverter, a: Vec3): Vec3 {
  return m3mulVec(c.R, v3scale(a, c.accelScale));
}

/** Structural validation: a convention must be a signed permutation with known units. */
export function checkConventionShape(conv: SensorConvention, requireProper: boolean): string | null {
  if (!conv || typeof conv.firmware !== 'string' || conv.firmware.length === 0) return '缺少固件标识';
  if (!(conv.R_bs instanceof Float64Array) || conv.R_bs.length !== 9) return 'R_bs 必须是 3×3 矩阵';
  const sq = (x: number) => x * x;
  for (let r = 0; r < 3; r++) {
    const rowSum = sq(conv.R_bs[r * 3]) + sq(conv.R_bs[r * 3 + 1]) + sq(conv.R_bs[r * 3 + 2]);
    const colSum = sq(conv.R_bs[r]) + sq(conv.R_bs[r + 3]) + sq(conv.R_bs[r + 6]);
    if (Math.abs(rowSum - 1) > 1e-9 || Math.abs(colSum - 1) > 1e-9) return 'R_bs 不是带符号置换矩阵';
  }
  const det = m3det(conv.R_bs);
  if (Math.abs(Math.abs(det) - 1) > 1e-9) return 'R_bs 行列式必须为 ±1';
  if (requireProper && det < 0) return '已固化约定的 R_bs 必须是正常旋转（det=+1），镜像解不允许固化';
  if (conv.gyroUnit !== 'rad/s' && conv.gyroUnit !== 'deg/s') return `未知角速度单位 ${String(conv.gyroUnit)}`;
  if (conv.accelUnit !== 'm/s^2' && conv.accelUnit !== 'g') return `未知加速度单位 ${String(conv.accelUnit)}`;
  return null;
}

export interface StaticConventionCheck {
  ok: boolean;
  /** Measured specific-force magnitude after conversion, m/s². */
  gravityMag: number;
  gravityMagErr: number;
  /** cos(angle) between converted specific force and body +z (level rig). */
  verticalCos: number;
  problems: string[];
}

/**
 * Static-window convention check. The calibration rig sits still and level, so
 * the converted specific force must have magnitude g and point along body +z.
 * A swapped axis map or a wrong accel unit fails here; nothing is assumed.
 */
export function checkStaticConvention(conv: SensorConvention, accelMeanSensor: Vec3): StaticConventionCheck {
  const f = accelToBody(conv, accelMeanSensor);
  const mag = v3norm(f);
  const gravityMagErr = Math.abs(mag - GRAVITY_MAG);
  const verticalCos = mag > 1e-9 ? f[2] / mag : 0;
  const problems: string[] = [];
  if (gravityMagErr > 0.5) {
    problems.push(`|f|=${mag.toFixed(3)} m/s² 与重力 ${GRAVITY_MAG} 不符（加速度单位可能配置错误）`);
  }
  if (verticalCos < 0.98) {
    const deg = Math.acos(Math.min(1, Math.max(-1, verticalCos))) * RAD2DEG;
    problems.push(`比力与机体 +z 夹角 ${deg.toFixed(1)}°（轴系映射可能配置错误）`);
  }
  return { ok: problems.length === 0, gravityMag: mag, gravityMagErr, verticalCos, problems };
}

/** Trapezoid integration of body-frame rates (rad/s) into an attitude increment. */
export function integrateGyroBody(samples: Array<{ t: number; w: Vec3 }>): Mat3 {
  let dR = so3Exp([0, 0, 0]);
  for (let i = 1; i < samples.length; i++) {
    const a = samples[i - 1];
    const b = samples[i];
    const h = b.t - a.t;
    if (h <= 0) continue;
    const w: Vec3 = [(a.w[0] + b.w[0]) / 2, (a.w[1] + b.w[1]) / 2, (a.w[2] + b.w[2]) / 2];
    dR = m3mul(dR, so3Exp(v3scale(w, h)));
  }
  return dR;
}

export interface RotationConventionCheck {
  ok: boolean;
  /** Final-attitude error against the known rotation, degrees. */
  errDeg: number;
  problems: string[];
}

/**
 * Rotation-run convention check: convert raw gyro with `conv`, integrate, and
 * compare the final attitude with the known one. A swapped axis map or a wrong
 * gyro unit (e.g. deg/s read as rad/s) moves the final attitude far beyond the
 * tolerance, so the check fails instead of silently absorbing the mismatch.
 */
export function checkRotationConvention(
  conv: SensorConvention,
  gyro: Array<{ t: number; w: Vec3 }>,
  expectedFinalR: Mat3,
  tolDeg = 3,
): RotationConventionCheck {
  const body = gyro.map((s) => ({ t: s.t, w: gyroToBody(conv, s.w) }));
  const R = integrateGyroBody(body);
  const E = m3mul(m3transpose(expectedFinalR), R);
  const c = Math.min(1, Math.max(-1, (E[0] + E[4] + E[8] - 1) / 2));
  const errDeg = Math.acos(c) * RAD2DEG;
  const problems: string[] = [];
  if (!(errDeg < tolDeg)) {
    problems.push(`转动素材末态姿态误差 ${errDeg.toFixed(2)}° 超过 ${tolDeg}°（换轴或角速度单位配置错误）`);
  }
  return { ok: problems.length === 0, errDeg, problems };
}
