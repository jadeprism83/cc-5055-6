import { Mat3, Vec3, m3mul, m3transpose, v3norm } from '../core/math';
import { Extrinsics } from '../core/camera';
import {
  SensorConvention,
  accelToBody,
  checkConventionShape,
  checkRotationConvention,
  checkStaticConvention,
  firmwareA,
  firmwareB,
  GRAVITY_MAG,
} from '../core/conventions';

// The calibration record is the single artefact the gyro preintegrator and the
// rolling-shutter compensator both reference. It carries the committed sensor
// convention (axes + units) for the mounted firmware, the sampling time
// offset, and the camera-IMU extrinsics. A record is only `committed` after
// the rotation-run loader has excluded the mirror solutions that a static
// dataset cannot resolve; until then it stays a `preview` with its candidates.

export interface CalibrationRecord {
  id: string;
  /** Bumped on every firmware/convention change; snapshots stamped with an
   * older generation are stale and must be rejected by consumers. */
  generation: number;
  /** Firmware the committed convention applies to. */
  firmware: string;
  convention: SensorConvention;
  /** Candidate conventions still alive (preview). Mirrors included. */
  previews: SensorConvention[];
  status: 'preview' | 'committed';
  /** Gravity direction measured in the body frame from the static dataset. */
  gravityBody: Vec3 | null;
  timeOffset: number;
  extrinsics: Extrinsics;
  evidence: string[];
}

export function defaultCalibrationRecord(extrinsics: Extrinsics, timeOffset: number): CalibrationRecord {
  const fw = firmwareA();
  return {
    id: 'inlens-cal',
    generation: 1,
    firmware: fw.firmware,
    convention: fw,
    previews: [fw],
    status: 'committed',
    gravityBody: null,
    timeOffset,
    extrinsics: { R_bc: new Float64Array(extrinsics.R_bc), t_bc: [...extrinsics.t_bc] },
    evidence: ['出厂默认：fw-A 轴系与机体一致（rad/s，m/s²）'],
  };
}

/** Committed record for the replacement firmware, as shipped in its calibration file. */
export function firmwareBRecord(generation: number, extrinsics: Extrinsics, timeOffset: number): CalibrationRecord {
  const fw = firmwareB();
  return {
    id: 'inlens-cal',
    generation,
    firmware: fw.firmware,
    convention: fw,
    previews: [fw],
    status: 'committed',
    gravityBody: null,
    timeOffset,
    extrinsics: { R_bc: new Float64Array(extrinsics.R_bc), t_bc: [...extrinsics.t_bc] },
    evidence: ['fw-B：静止+转动素材已固化（见标定流程）'],
  };
}

export interface StaticCalibrationData {
  /** Static-window raw samples (sensor axes, sensor units). */
  imu: Array<{ t: number; gyro: Vec3; accel: Vec3 }>;
}

/**
 * Static dataset loader. Gravity fixes only two of the three attitude degrees
 * of freedom: from still imagery plus the accelerometer the sign of rotation
 * about the vertical axis is unobservable, so every convention consistent with
 * the static data is kept as a preview candidate (the true one and its mirror
 * about the vertical). Nothing is committed here.
 */
export function loadStaticCalibration(
  data: StaticCalibrationData,
  candidates: SensorConvention[],
  base: { id?: string; generation?: number; timeOffset: number; extrinsics: Extrinsics },
): CalibrationRecord {
  if (data.imu.length < 5) {
    throw new Error(`静止标定素材不足：仅 ${data.imu.length} 个样本（需要 ≥5），拒绝以默认值继续`);
  }
  const mean: Vec3 = [0, 0, 0];
  for (const s of data.imu) {
    mean[0] += s.accel[0];
    mean[1] += s.accel[1];
    mean[2] += s.accel[2];
  }
  mean[0] /= data.imu.length;
  mean[1] /= data.imu.length;
  mean[2] /= data.imu.length;
  const alive = candidates.filter((c) => checkStaticConvention(c, mean).ok);
  const evidence: string[] = [
    `静止素材：|f| 检查通过 ${alive.length}/${candidates.length} 个候选（重力 ${GRAVITY_MAG} m/s²）`,
    '静止素材只能估计重力方向，无法从画面认出绕竖轴的正负：保留全部一致候选供预览',
  ];
  if (alive.length === 0) evidence.push('警告：没有任何候选与静止素材一致（轴系或单位配置错误）');
  const first = alive[0] ?? candidates[0];
  return {
    id: base.id ?? 'inlens-cal',
    generation: base.generation ?? 1,
    firmware: first.firmware,
    convention: first,
    previews: alive,
    status: 'preview',
    gravityBody: alive.length > 0 ? accelToBody(alive[0], mean) : null,
    timeOffset: base.timeOffset,
    extrinsics: { R_bc: new Float64Array(base.extrinsics.R_bc), t_bc: [...base.extrinsics.t_bc] },
    evidence,
  };
}

export interface RotationRun {
  label: string;
  /** Raw gyro during the run (sensor axes, sensor units). */
  gyro: Array<{ t: number; w: Vec3 }>;
  /** Known final attitude R_wb at the end of the ordered rotations. */
  finalR_wb: Mat3;
}

/**
 * Rotation-run refinement. The rig is rotated about two known axes in a known
 * order; because rotations do not commute, the ordered sequence gives the
 * mirror candidate a different final attitude, and it is excluded. Only when
 * exactly one candidate survives may the convention be committed.
 */
export function refineWithRotationRuns(
  record: CalibrationRecord,
  runs: RotationRun[],
  tolDeg = 3,
): CalibrationRecord {
  if (record.previews.length === 0) {
    return { ...record, evidence: [...record.evidence, '转动细化：无候选可评估'] };
  }
  const survivors = record.previews.filter((cand) =>
    runs.every((run) => checkRotationConvention(cand, run.gyro, run.finalR_wb, tolDeg).ok),
  );
  const evidence = [...record.evidence];
  if (survivors.length === 0) {
    evidence.push('转动素材排除全部候选：换轴或角速度单位配置错误，拒绝固化');
    return { ...record, previews: [], status: 'preview', evidence };
  }
  if (survivors.length > 1) {
    evidence.push(`转动素材后仍剩 ${survivors.length} 个候选：歧义未排除，保持预览`);
    return { ...record, previews: survivors, status: 'preview', evidence };
  }
  const only = survivors[0];
  const shapeErr = checkConventionShape(only, true);
  if (shapeErr) {
    evidence.push(`唯一幸存候选不是正常旋转，拒绝固化：${shapeErr}`);
    return { ...record, previews: survivors, status: 'preview', evidence };
  }
  evidence.push(`旋转顺序排除镜像解：固化 ${only.firmware}（${only.gyroUnit}，${only.accelUnit}）`);
  return {
    ...record,
    generation: record.generation + 1,
    firmware: only.firmware,
    convention: only,
    previews: survivors,
    status: 'committed',
    evidence,
  };
}

/**
 * Gate between preview and committed use. The preintegrator and the shutter
 * compensator must only ever see a record that passed through here.
 */
export function solidifyConvention(record: CalibrationRecord): CalibrationRecord | null {
  return record.status === 'committed' ? record : null;
}

/** Reflection through the vertical plane: preserves gravity, flips yaw sign. */
export function mirrorAboutVertical(conv: SensorConvention): SensorConvention {
  const M = new Float64Array([1, 0, 0, 0, -1, 0, 0, 0, 1]);
  return { ...conv, R_bs: m3mul(M, conv.R_bs) };
}

// ---------------------------------------------------------------------------
// Export / import. The file format is explicit and fully validated: a missing
// or inconsistent field is an error, never an occasion to invent a default.

const RECORD_SCHEMA = 'inertial-lens/calibration@1';

export function exportCalibration(rec: CalibrationRecord): string {
  const convToJson = (c: SensorConvention) => ({
    firmware: c.firmware,
    R_bs: [...c.R_bs],
    gyroUnit: c.gyroUnit,
    accelUnit: c.accelUnit,
  });
  return JSON.stringify(
    {
      schema: RECORD_SCHEMA,
      id: rec.id,
      generation: rec.generation,
      firmware: rec.firmware,
      status: rec.status,
      convention: convToJson(rec.convention),
      previews: rec.previews.map(convToJson),
      gravityBody: rec.gravityBody,
      timeOffset: rec.timeOffset,
      extrinsics: { R_bc: [...rec.extrinsics.R_bc], t_bc: [...rec.extrinsics.t_bc] },
      evidence: rec.evidence,
    },
    null,
    2,
  );
}

export function importCalibration(json: string): CalibrationRecord {
  let raw: unknown;
  try {
    raw = JSON.parse(json);
  } catch {
    throw new Error('标定记录无效：不是合法 JSON');
  }
  const o = raw as Record<string, unknown>;
  const fail = (why: string): never => {
    throw new Error(`标定记录无效：${why}`);
  };
  if (o.schema !== RECORD_SCHEMA) fail(`schema 应为 ${RECORD_SCHEMA}，实际 ${String(o.schema)}`);
  const id: string = typeof o.id === 'string' && o.id.length > 0 ? o.id : fail('缺少 id');
  const generation: number =
    Number.isInteger(o.generation) && (o.generation as number) >= 1
      ? (o.generation as number)
      : fail('generation 必须是正整数');
  const firmware: string = typeof o.firmware === 'string' && o.firmware.length > 0 ? o.firmware : fail('缺少 firmware');
  const status: 'preview' | 'committed' =
    o.status === 'preview' || o.status === 'committed' ? o.status : fail('status 必须是 preview 或 committed');
  const timeOffset: number =
    typeof o.timeOffset === 'number' && Number.isFinite(o.timeOffset) ? o.timeOffset : fail('timeOffset 缺失或不是有限数');
  const parseConv = (x: unknown, requireProper: boolean): SensorConvention => {
    const c = x as Record<string, unknown>;
    if (!c || typeof c !== 'object') fail('约定项不是对象');
    const fw: string = typeof c.firmware === 'string' ? c.firmware : fail('约定缺少 firmware');
    if (!Array.isArray(c.R_bs) || c.R_bs.length !== 9 || !c.R_bs.every((v) => Number.isFinite(v))) {
      fail('约定 R_bs 必须是 9 个有限数');
    }
    const conv: SensorConvention = {
      firmware: fw,
      R_bs: new Float64Array(c.R_bs as number[]),
      gyroUnit: c.gyroUnit as SensorConvention['gyroUnit'],
      accelUnit: c.accelUnit as SensorConvention['accelUnit'],
    };
    const err = checkConventionShape(conv, requireProper);
    if (err) fail(err);
    return conv;
  };
  const convention = parseConv(o.convention, status === 'committed');
  if (!Array.isArray(o.previews)) fail('previews 缺失');
  const previews = (o.previews as unknown[]).map((p) => parseConv(p, false));
  if (status === 'committed' && previews.length !== 1) fail('已固化记录必须恰好保留 1 个候选');
  const ext = o.extrinsics as Record<string, unknown>;
  if (!ext || typeof ext !== 'object') fail('缺少 extrinsics');
  if (!Array.isArray(ext.R_bc) || ext.R_bc.length !== 9 || !ext.R_bc.every((v) => Number.isFinite(v))) {
    fail('extrinsics.R_bc 必须是 9 个有限数');
  }
  if (!Array.isArray(ext.t_bc) || ext.t_bc.length !== 3 || !ext.t_bc.every((v) => Number.isFinite(v))) {
    fail('extrinsics.t_bc 必须是 3 个有限数');
  }
  const R_bc = new Float64Array(ext.R_bc as number[]);
  // The extrinsic rotation must be a proper rotation; checked, not assumed.
  const det =
    R_bc[0] * (R_bc[4] * R_bc[8] - R_bc[5] * R_bc[7]) -
    R_bc[1] * (R_bc[3] * R_bc[8] - R_bc[5] * R_bc[6]) +
    R_bc[2] * (R_bc[3] * R_bc[7] - R_bc[4] * R_bc[6]);
  if (Math.abs(det - 1) > 1e-6) fail(`extrinsics.R_bc 行列式 ${det.toFixed(6)} ≠ 1`);
  for (let c = 0; c < 3; c++) {
    const n = Math.hypot(R_bc[c], R_bc[c + 3], R_bc[c + 6]);
    if (Math.abs(n - 1) > 1e-6) fail('extrinsics.R_bc 列不是单位向量');
  }
  const gravityBody = o.gravityBody;
  if (gravityBody !== null && (!Array.isArray(gravityBody) || gravityBody.length !== 3 || !gravityBody.every((v) => Number.isFinite(v)))) {
    fail('gravityBody 必须是 null 或 3 个有限数');
  }
  if (!Array.isArray(o.evidence) || !o.evidence.every((e) => typeof e === 'string')) fail('evidence 必须是字符串数组');
  return {
    id,
    generation,
    firmware,
    status,
    convention,
    previews,
    gravityBody: gravityBody === null ? null : ([...(gravityBody as number[])] as Vec3),
    timeOffset,
    extrinsics: { R_bc, t_bc: [...(ext.t_bc as number[])] as Vec3 },
    evidence: [...(o.evidence as string[])],
  };
}

/** Rotation error in degrees between two rotation matrices. */
export function rotationErrorDegLocal(a: Mat3, b: Mat3): number {
  const E = m3mul(m3transpose(a), b);
  const c = Math.min(1, Math.max(-1, (E[0] + E[4] + E[8] - 1) / 2));
  return (Math.acos(c) * 180) / Math.PI;
}

export function vec3NormLocal(v: Vec3): number {
  return v3norm(v);
}
