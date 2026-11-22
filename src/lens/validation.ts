import { Mat3, Vec3, m3identity, m3mul, m3transpose, mulberry32, v3norm } from '../core/math';
import { defaultExtrinsics, defaultIntrinsics, projectWorld } from '../core/camera';
import {
  GRAVITY_MAG,
  SensorConvention,
  accelFromBody,
  checkRotationConvention,
  checkStaticConvention,
  converterBetween,
  convertGyroSample,
  firmwareA,
  firmwareB,
  gyroFromBody,
  gyroToBody,
  integrateGyroBody,
  invertConverter,
} from '../core/conventions';
import {
  CalibrationRecord,
  defaultCalibrationRecord,
  exportCalibration,
  firmwareBRecord,
  importCalibration,
  loadStaticCalibration,
  mirrorAboutVertical,
  refineWithRotationRuns,
  solidifyConvention,
} from './record';
import { GyroPreintegrator } from './preintegration';
import { ShutterCompensator } from './shutter';
import { PoseSnapshot, groundContactProjection, markerProjection } from './overlays';
import { closedLoopSpin, orderedRotationRun, sinusoidMotion } from '../sim/analytic';
import { defaultSimConfig } from '../sim/trajectory';
import { BootstrapInput, fitScaleAndVelocity, integrateWorldAcceleration, runBootstrap } from './bootstrap';
import { buildCapture, sessionFromCapture } from './replay';
import { InertialLensKernel, KernelFrame, defaultKernelOptions } from './kernel';
import { makeNoise } from '../estimation/eskf';
import { absoluteTrajectoryError, rotationAte } from './evaluate';

// Inertial-data validation and convention-maintenance checks. Every check
// computes from generated data (analytic motion or the synthetic capture);
// nothing is hand-edited, and no check substitutes a default when data is
// missing — failures are reported, not masked. Runs in a clean directory:
// `npm run validate` builds this file standalone and executes it with node.

export interface ValidationCheck {
  id: string;
  title: string;
  ok: boolean;
  detail: string;
  metrics: Record<string, number>;
}

export interface ValidationReport {
  ok: boolean;
  checks: ValidationCheck[];
  durationMs: number;
}

const DEG = 180 / Math.PI;

function angleDeg(R: Mat3): number {
  const c = Math.min(1, Math.max(-1, (R[0] + R[4] + R[8] - 1) / 2));
  return Math.acos(c) * DEG;
}

function errDeg(a: Mat3, b: Mat3): number {
  const E = new Float64Array(9);
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += a[k * 3 + r] * b[k * 3 + c];
      E[r * 3 + c] = s;
    }
  return angleDeg(E);
}

/** fw-B axes with the x/y sensor columns exchanged: a signed permutation, but the wrong one. */
function swappedAxisConvention(): SensorConvention {
  const fwB = firmwareB();
  return { ...fwB, R_bs: new Float64Array([0, 0, -1, 0, 1, 0, -1, 0, 0]) };
}

function checkStaticConventions(): ValidationCheck {
  const fwB = firmwareB();
  const fBody: Vec3 = [0, 0, GRAVITY_MAG];
  const rawB = accelFromBody(fwB, fBody);
  const good = checkStaticConvention(fwB, rawB);
  const wrongConv = checkStaticConvention(firmwareA(), rawB);
  const swapped = checkStaticConvention(swappedAxisConvention(), rawB);
  const wrongUnit = checkStaticConvention({ ...fwB, accelUnit: 'm/s^2' }, rawB);
  const ok = good.ok && !wrongConv.ok && !swapped.ok && !wrongUnit.ok;
  return {
    id: 'static-convention',
    title: '静止素材：轴系/单位检查通过，换轴与改单位被拒绝',
    ok,
    detail: ok
      ? `fw-B 静止比力 |f|=${good.gravityMag.toFixed(3)} m/s²、竖直余弦 ${good.verticalCos.toFixed(3)}；` +
        `错误约定 |f|=${wrongConv.gravityMag.toFixed(3)}、换轴竖直余弦 ${swapped.verticalCos.toFixed(3)}、` +
        `错单位 |f|=${wrongUnit.gravityMag.toFixed(3)}，均被判不合格`
      : '静止检查未按预期区分正确/错误约定',
    metrics: {
      gravityMag: good.gravityMag,
      verticalCos: good.verticalCos,
      wrongConventionMag: wrongConv.gravityMag,
      swappedVerticalCos: swapped.verticalCos,
      wrongUnitMag: wrongUnit.gravityMag,
    },
  };
}

function sampleGyro(motion: { omegaB(t: number): Vec3; duration: number }, hz: number, jitter: number, seed: number): Array<{ t: number; w: Vec3 }> {
  const fwB = firmwareB();
  const rng = mulberry32(seed);
  const out: Array<{ t: number; w: Vec3 }> = [];
  const n = Math.floor(motion.duration * hz);
  let tPrev = -1;
  for (let k = 0; k <= n; k++) {
    let t = k / hz;
    if (jitter > 0 && k > 0 && k < n) {
      t += (rng() - 0.5) * 2 * jitter * (1 / hz);
      if (t <= tPrev + 1e-6) t = tPrev + 1e-6;
    }
    tPrev = t;
    out.push({ t, w: gyroFromBody(fwB, motion.omegaB(t)) });
  }
  return out;
}

function checkAttitudeTrajectory(): ValidationCheck {
  const fwB = firmwareB();
  const motion = sinusoidMotion();
  // Gyro integration yields the increment from the initial attitude, so the
  // expectation is R(0)ᵀ·R(T), not R(T).
  const expected = m3mul(m3transpose(motion.R(0)), motion.R(motion.duration));
  const integrate = (samples: Array<{ t: number; w: Vec3 }>, conv: SensorConvention) =>
    integrateGyroBody(samples.map((s) => ({ t: s.t, w: gyroToBody(conv, s.w) })));

  const uniform = sampleGyro(motion, 200, 0, 11);
  const errUniform = errDeg(integrate(uniform, fwB), expected);
  const jittered = sampleGyro(motion, 200, 0.3, 12);
  const errJitter = errDeg(integrate(jittered, fwB), expected);
  const errSwapped = errDeg(integrate(uniform, swappedAxisConvention()), expected);
  const errUnit = errDeg(integrate(uniform, { ...fwB, gyroUnit: 'rad/s' }), expected);
  const rotCheck = checkRotationConvention(fwB, uniform, expected, 3);
  const rotCheckBad = checkRotationConvention(swappedAxisConvention(), uniform, expected, 3);
  const ok =
    errUniform < 1 &&
    errJitter < 1.5 &&
    errSwapped > 10 &&
    errUnit > 30 &&
    rotCheck.ok &&
    !rotCheckBad.ok;
  return {
    id: 'attitude-trajectory',
    title: '姿态轨迹：解析运动积分吻合（均匀+非匀速采样），换轴/改单位被查出',
    ok,
    detail:
      `均匀采样末态误差 ${errUniform.toFixed(3)}°，非匀速采样 ${errJitter.toFixed(3)}°；` +
      `换轴积分误差 ${errSwapped.toFixed(1)}°、角速度单位错误 ${errUnit.toFixed(1)}°，均超阈值被判不合格`,
    metrics: {
      errUniformDeg: errUniform,
      errJitteredDeg: errJitter,
      errSwappedDeg: errSwapped,
      errWrongUnitDeg: errUnit,
    },
  };
}

function checkReprojectionDeskew(): ValidationCheck {
  const ext = defaultExtrinsics();
  const K = defaultIntrinsics();
  const motion = sinusoidMotion();
  // Use the instant of peak body rate so the shutter skew is clearly visible.
  let t = 0;
  let wMax = 0;
  for (let k = 0; k <= 400; k++) {
    const tk = (k / 400) * motion.duration;
    const w = v3norm(motion.omegaB(tk));
    if (w > wMax) {
      wMax = w;
      t = tk;
    }
  }
  const R_wb = motion.R(t);
  const p_wb = motion.p(t);
  const fwd: Vec3 = [
    R_wb[0] * ext.R_bc[2] + R_wb[1] * ext.R_bc[5] + R_wb[2] * ext.R_bc[8],
    R_wb[3] * ext.R_bc[2] + R_wb[4] * ext.R_bc[5] + R_wb[5] * ext.R_bc[8],
    R_wb[6] * ext.R_bc[2] + R_wb[7] * ext.R_bc[5] + R_wb[8] * ext.R_bc[8],
  ];
  const C: Vec3 = [
    p_wb[0] + R_wb[0] * ext.t_bc[0] + R_wb[1] * ext.t_bc[1] + R_wb[2] * ext.t_bc[2],
    p_wb[1] + R_wb[3] * ext.t_bc[0] + R_wb[4] * ext.t_bc[1] + R_wb[5] * ext.t_bc[2],
    p_wb[2] + R_wb[6] * ext.t_bc[0] + R_wb[7] * ext.t_bc[1] + R_wb[8] * ext.t_bc[2],
  ];
  const point: Vec3 = [C[0] + 3 * fwd[0], C[1] + 3 * fwd[1], C[2] + 3 * fwd[2]];
  const px0 = projectWorld(R_wb, p_wb, ext, K, point);

  // Rolling shutter: row v is read dt later, at the rotated attitude.
  const readout = 40e-6;
  const v = 200;
  const omega = motion.omegaB(t);
  const dt = v * readout;
  const dR = integrateGyroBody([{ t: 0, w: [0, 0, 0] }, { t: dt, w: omega }]);
  const R_row = new Float64Array(9);
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += R_wb[r * 3 + k] * dR[k * 3 + c];
      R_row[r * 3 + c] = s;
    }
  const pxRow = projectWorld(R_row, p_wb, ext, K, point);

  const record = firmwareBRecord(2, ext, 0.018);
  const shutter = new ShutterCompensator(record, readout);
  const rawB = gyroFromBody(firmwareB(), omega);
  const fixed = shutter.deskewWithRaw(pxRow.u, pxRow.v, K, rawB, 'fw-B');
  const unknownFw = shutter.deskewWithRaw(pxRow.u, pxRow.v, K, rawB, 'fw-A');
  const deskewErr = fixed ? Math.hypot(fixed.u - px0.u, fixed.v - px0.v) : Infinity;
  const skewPx = Math.hypot(pxRow.u - px0.u, pxRow.v - px0.v);
  const ok = fixed !== null && deskewErr < 0.5 && unknownFw === null && skewPx > 0.5;
  return {
    id: 'reprojection-deskew',
    title: '重投影与快门去斜：同一标定记录恢复参考行投影，未知固件被拒绝',
    ok,
    detail:
      `峰值角速率 ${wMax.toFixed(2)} rad/s（t=${t.toFixed(2)} s）下行 ${v} 的卷帘快门偏移 ${skewPx.toFixed(2)} px，` +
      `去斜后与参考行投影差 ${deskewErr.toFixed(3)} px；未固化固件 fw-A 的去斜请求返回 null（不用默认轴系换算）`,
    metrics: { skewPx, deskewErrPx: deskewErr },
  };
}

function checkClosedLoop(): ValidationCheck {
  const fwA = firmwareA();
  const fwB = firmwareB();
  const spin = closedLoopSpin([0, 0, 1], 1.0);
  const hz = 200;
  const n = Math.floor(spin.duration * hz);
  const rawA: Array<{ t: number; w: Vec3 }> = [];
  for (let k = 0; k <= n; k++) {
    const t = k / hz;
    rawA.push({ t, w: gyroFromBody(fwA, spin.omegaB(t)) });
  }
  const conv = converterBetween(fwA, fwB);
  const inv = invertConverter(conv);
  // Converter invertibility: A → B → A reproduces the sample exactly.
  let roundTrip = 0;
  for (const s of rawA) {
    const back = convertGyroSample(inv, convertGyroSample(conv, s.w));
    roundTrip = Math.max(roundTrip, v3norm([back[0] - s.w[0], back[1] - s.w[1], back[2] - s.w[2]]));
  }
  // Correct chain: swap once, integrate with the committed fw-B convention.
  const asB = rawA.map((s) => ({ t: s.t, w: convertGyroSample(conv, s.w) }));
  const Rgood = integrateGyroBody(asB.map((s) => ({ t: s.t, w: gyroToBody(fwB, s.w) })));
  const errGood = angleDeg(Rgood);
  // Double-swap bug: the converter is applied a second time to already-swapped
  // data. The 2π loop still closes in attitude, so the check images a fixed
  // world point at mid-loop, where the wrong axis shows up in the projection.
  const doubleSwapped = asB.map((s) => ({ t: s.t, w: convertGyroSample(conv, s.w) }));
  const ext = defaultExtrinsics();
  const K = defaultIntrinsics();
  // Mid-loop the rig has turned π about z and looks toward world -x, so the
  // marker sits on that side to stay in frame.
  const marker: Vec3 = [-2.5, 0.4, 1.1];
  const midT = spin.duration / 2;
  const imageErrAt = (samples: Array<{ t: number; w: Vec3 }>): number => {
    const half = samples.filter((s) => s.t <= midT);
    const Rmid = integrateGyroBody(half.map((s) => ({ t: s.t, w: gyroToBody(fwB, s.w) })));
    const px = projectWorld(Rmid, [0, 0, 0], ext, K, marker);
    const pxTrue = projectWorld(spin.R(midT), [0, 0, 0], ext, K, marker);
    if (!px.valid || !pxTrue.valid) return Infinity;
    return Math.hypot(px.u - pxTrue.u, px.v - pxTrue.v);
  };
  const imgGood = imageErrAt(asB);
  const imgDouble = imageErrAt(doubleSwapped);
  const ok = roundTrip < 1e-12 && errGood < 0.5 && imgGood < 1 && imgDouble > 50;
  return {
    id: 'closed-loop-double-swap',
    title: '可逆转换器 + 闭环转动：串联换轴/积分/成像，双重换轴被查出',
    ok,
    detail:
      `A→B→A 往返误差 ${roundTrip.toExponential(1)}，闭环姿态误差 ${errGood.toFixed(3)}°，` +
      `中段成像差 ${imgGood.toFixed(2)} px；双重换轴后中段成像差 ${imgDouble.toFixed(1)} px，闭环检查报警`,
    metrics: { roundTrip, closureErrDeg: errGood, imgErrGoodPx: imgGood, imgErrDoubleSwapPx: imgDouble },
  };
}

function staticDataset(fwB: SensorConvention, n: number): { imu: Array<{ t: number; gyro: Vec3; accel: Vec3 }> } {
  const imu: Array<{ t: number; gyro: Vec3; accel: Vec3 }> = [];
  for (let k = 0; k < n; k++) {
    imu.push({
      t: k / 200,
      gyro: gyroFromBody(fwB, [0, 0, 0]),
      accel: accelFromBody(fwB, [0, 0, GRAVITY_MAG]),
    });
  }
  return { imu };
}

function rotationRunData(): { record0: CalibrationRecord; runs: Parameters<typeof refineWithRotationRuns>[1] } {
  const fwB = firmwareB();
  const ext = defaultExtrinsics();
  const record0 = loadStaticCalibration(staticDataset(fwB, 60), [fwB, mirrorAboutVertical(fwB)], {
    timeOffset: 0.018,
    extrinsics: ext,
  });
  const mkRun = (phases: Array<{ axis: Vec3; rate: number; duration: number }>, label: string) => {
    const motion = orderedRotationRun(phases);
    const gyro: Array<{ t: number; w: Vec3 }> = [];
    const n = Math.floor(motion.duration * 100);
    for (let k = 0; k <= n; k++) {
      const t = k / 100;
      gyro.push({ t, w: gyroFromBody(fwB, motion.omegaB(t)) });
    }
    return { label, gyro, finalR_wb: motion.finalR };
  };
  const runs = [
    mkRun(
      [
        { axis: [1, 0, 0], rate: 0.8, duration: 1.0 },
        { axis: [0, 1, 0], rate: 0.6, duration: 1.2 },
      ],
      '先 x 后 y',
    ),
    mkRun(
      [
        { axis: [0, 1, 0], rate: 0.5, duration: 1.1 },
        { axis: [0, 0, 1], rate: -0.7, duration: 0.9 },
      ],
      '先 y 后 z',
    ),
  ];
  return { record0, runs };
}

function checkLoaderFlow(): ValidationCheck {
  const fwB = firmwareB();
  const { record0, runs } = rotationRunData();
  const previewOk = record0.status === 'preview' && record0.previews.length === 2;
  const refused = solidifyConvention(record0) === null;
  const mirrorFails = !checkRotationConvention(mirrorAboutVertical(fwB), runs[0].gyro, runs[0].finalR_wb, 3).ok;
  const committed = refineWithRotationRuns(record0, runs);
  const commitOk =
    committed.status === 'committed' &&
    committed.generation === record0.generation + 1 &&
    committed.convention.firmware === 'fw-B' &&
    committed.previews.length === 1;
  // Wrong gyro unit in the candidate: the rotation runs must reject everything.
  const wrongUnitRecord = loadStaticCalibration(staticDataset({ ...fwB, gyroUnit: 'rad/s' }, 60), [{ ...fwB, gyroUnit: 'rad/s' }], {
    timeOffset: 0.018,
    extrinsics: defaultExtrinsics(),
  });
  const rejected = refineWithRotationRuns(wrongUnitRecord, runs);
  const rejectOk = rejected.status === 'preview' && rejected.previews.length === 0;
  const ok = previewOk && refused && mirrorFails && commitOk && rejectOk;
  return {
    id: 'loader-flow',
    title: '标定加载：静止保留镜像候选且拒绝固化，有序转动排除镜像后固化唯一解',
    ok,
    detail:
      `静止素材后候选 ${record0.previews.length} 个（含镜像），solidify 拒绝=${refused}；` +
      `两组有序转动后 ${committed.previews.length} 个幸存并固化（代 ${committed.generation}）；` +
      `角速度单位错误的候选被转动素材全部排除`,
    metrics: { previewCandidates: record0.previews.length, committedGeneration: committed.generation },
  };
}

function checkExportImport(): ValidationCheck {
  const ext = defaultExtrinsics();
  const rec = firmwareBRecord(3, ext, 0.018);
  const json = exportCalibration(rec);
  const back = importCalibration(json);
  const fwB = firmwareB();
  const samples = [
    { t: 0, gyro: gyroFromBody(fwB, [0.1, 0.2, -0.1]), fw: 'fw-B' },
    { t: 0.005, gyro: gyroFromBody(fwB, [0.12, 0.18, -0.08]), fw: 'fw-B' },
    { t: 0.01, gyro: gyroFromBody(fwB, [0.08, 0.22, -0.12]), fw: 'fw-B' },
  ];
  const bias: Vec3 = [0.001, -0.001, 0.0005];
  const s1 = new GyroPreintegrator(rec).integrate(samples, bias);
  const s2 = new GyroPreintegrator(back).integrate(samples, bias);
  let integDiff = Infinity;
  if (s1 && s2) {
    integDiff = 0;
    for (let i = 0; i < 9; i++) integDiff = Math.max(integDiff, Math.abs(s1.dR[i] - s2.dR[i]));
  }
  const K = defaultIntrinsics();
  const raw = gyroFromBody(fwB, [0.3, -0.2, 0.1]);
  const d1 = new ShutterCompensator(rec, 25e-6).deskewWithRaw(160, 180, K, raw, 'fw-B');
  const d2 = new ShutterCompensator(back, 25e-6).deskewWithRaw(160, 180, K, raw, 'fw-B');
  const deskewDiff = d1 && d2 ? Math.hypot(d1.u - d2.u, d1.v - d2.v) : Infinity;
  // Tampered records must throw, never be repaired with defaults.
  const base = JSON.parse(json) as Record<string, unknown>;
  const mirrorObj = JSON.parse(json) as { convention: { R_bs: number[] } };
  mirrorObj.convention.R_bs = [0, 0, 1, -1, 0, 0, 0, 1, 0];
  const twoPreviews = JSON.parse(json) as { previews: unknown[] };
  twoPreviews.previews = [twoPreviews.previews[0], twoPreviews.previews[0]];
  const tampered: Array<[string, string]> = [
    ['镜像轴系', JSON.stringify(mirrorObj)],
    ['缺少 generation', JSON.stringify({ ...base, generation: undefined })],
    ['已固化但候选 2 个', JSON.stringify(twoPreviews)],
    ['缺 extrinsics', JSON.stringify({ ...base, extrinsics: undefined })],
  ];
  let thrown = 0;
  for (const [, text] of tampered) {
    try {
      importCalibration(text);
    } catch {
      thrown++;
    }
  }
  const ok = integDiff === 0 && deskewDiff === 0 && thrown === tampered.length;
  return {
    id: 'export-import',
    title: '导出再读入：积分与去斜采用同一被证实约定，篡改记录一律抛错',
    ok,
    detail:
      `读入记录的陀螺积分与原件差 ${integDiff}，快门去斜差 ${deskewDiff} px；` +
      `${thrown}/${tampered.length} 个篡改记录被拒绝（${tampered.map((t) => t[0]).join('、')}）`,
    metrics: { integrateDiff: integDiff, deskewDiffPx: deskewDiff, tamperedRejected: thrown },
  };
}

function checkStaleSnapshot(): ValidationCheck {
  const ext = defaultExtrinsics();
  const K = defaultIntrinsics();
  const marker: Vec3 = [4.6, 0.55, 1.25];
  const groundZ = -0.6;
  const recordA = defaultCalibrationRecord(ext, 0.018);
  const preA = new GyroPreintegrator(recordA);
  const fwA = firmwareA();
  const snap = preA.integrate(
    [
      { t: 0, gyro: gyroFromBody(fwA, [0.2, 0.1, 0]), fw: 'fw-A' },
      { t: 0.005, gyro: gyroFromBody(fwA, [0.2, 0.1, 0]), fw: 'fw-A' },
      { t: 0.01, gyro: gyroFromBody(fwA, [0.2, 0.1, 0]), fw: 'fw-A' },
    ],
    [0, 0, 0],
  );
  const pose: PoseSnapshot = {
    recordId: snap?.recordId ?? recordA.id,
    generation: snap?.generation ?? recordA.generation,
    t: 0.01,
    R_wb: m3identity(),
    p_wb: [0, 0, 0.4],
  };
  const mBefore = markerProjection(pose, recordA, ext, K, marker);
  const cBefore = groundContactProjection(pose, recordA, ext, K, groundZ);
  // Firmware swap: the committed record advances a generation; the snapshot
  // computed under the old conversion arrives late and must be rejected by
  // both overlays.
  const recordB = firmwareBRecord(recordA.generation + 1, ext, 0.018);
  const mAfter = markerProjection(pose, recordB, ext, K, marker);
  const cAfter = groundContactProjection(pose, recordB, ext, K, groundZ);
  const fresh: PoseSnapshot = { ...pose, generation: recordB.generation };
  const mFresh = markerProjection(fresh, recordB, ext, K, marker);
  const cFresh = groundContactProjection(fresh, recordB, ext, K, groundZ);
  const ok = mBefore !== null && cBefore !== null && mAfter === null && cAfter === null && mFresh !== null && cFresh !== null;
  return {
    id: 'stale-snapshot',
    title: '固件更换：旧换算的迟到快照被角标与虚拟地面接触点同时拒绝',
    ok,
    detail:
      `代 ${recordA.generation} 快照在旧记录下两者接受；固件更换到代 ${recordB.generation} 后，` +
      `角标返回 ${mAfter === null ? 'null' : '像素'}、触点返回 ${cAfter === null ? 'null' : '像素'}；` +
      `新代快照两者恢复接受`,
    metrics: { oldGeneration: recordA.generation, newGeneration: recordB.generation },
  };
}

function checkScaleHonesty(): ValidationCheck {
  // Static rig: the visual centres coincide and the IMU integral is ~0, so the
  // scale normal equations are singular. The fit must refuse, not invent 1.0.
  const times: number[] = [];
  const Cvis: Vec3[] = [];
  const integrals: ReturnType<typeof integrateWorldAcceleration>[] = [];
  const imu: Array<{ t: number; gyro: Vec3; accel: Vec3 }> = [];
  for (let k = 0; k <= 40; k++) {
    times.push(k / 20);
    Cvis.push([0, 0, 0]);
    imu.push({ t: k / 200, gyro: [0, 0, 0], accel: [0, 0, GRAVITY_MAG] });
  }
  for (let k = 0; k <= 40; k++) {
    integrals.push(
      integrateWorldAcceleration(imu, times, 0, k, 0, () => m3identity(), [0, 0, 0], [0, 0, -GRAVITY_MAG]),
    );
  }
  const fit = fitScaleAndVelocity(times, Cvis, integrals, 0);
  const ok = !fit.ok;
  return {
    id: 'scale-honesty',
    title: '尺度诚实：静止数据无可观测约束时不宣称尺度已知',
    ok,
    detail: `静止窗口视觉位移为零、惯性积分约为零：尺度拟合 ok=${fit.ok}（${fit.evidence}）`,
    metrics: { fitOk: fit.ok ? 1 : 0 },
  };
}

function checkLensReplay(): ValidationCheck {
  const cfg = defaultSimConfig(7);
  cfg.duration = 13;
  const cap = buildCapture(cfg);
  const session = sessionFromCapture(cap);
  const opts = defaultKernelOptions(session);
  const kernel = new InertialLensKernel(session, opts, makeNoise({}));
  const est: Vec3[] = [];
  const truth: Vec3[] = [];
  const estR: Mat3[] = [];
  const truthR: Mat3[] = [];
  let rec: KernelFrame | null = kernel.advance();
  while (rec) {
    est.push([...rec.p_wb]);
    truth.push(cap.truthP[rec.index]);
    estR.push(new Float64Array(rec.R_wb));
    truthR.push(cap.truthR[rec.index]);
    rec = kernel.advance();
  }
  const br = kernel.bootstrapResult;
  const ate = absoluteTrajectoryError(est, truth);
  const rotAte = rotationAte(estR, truthR);
  const holdout = kernel.frames.filter((f) => f.holdout);
  let sse = 0;
  for (const h of holdout) {
    const j = kernel.frames.indexOf(h);
    sse += (est[j][0] - truth[j][0]) ** 2 + (est[j][1] - truth[j][1]) ** 2 + (est[j][2] - truth[j][2]) ** 2;
  }
  const holdoutRms = holdout.length ? Math.sqrt(sse / holdout.length) : Infinity;
  const tracked = kernel.frames.filter((f) => f.used > 0);
  const meanReproj = tracked.length ? tracked.reduce((s, f) => s + f.reprojRms, 0) / tracked.length : Infinity;
  const ok =
    br !== null &&
    br.ok &&
    br.scale !== null &&
    br.scale > 0.4 &&
    br.scale < 1.0 &&
    ate.ate < 0.5 &&
    rotAte < 4 &&
    holdoutRms < 1.0 &&
    meanReproj < 2.0;
  return {
    id: 'lens-replay',
    title: '惯性镜头样例运行到底：轨迹、姿态、留出帧与重投影全部达标',
    ok,
    detail:
      `初始化尺度 ${br?.scale?.toFixed(3) ?? '—'}，ATE ${ate.ate.toFixed(3)} m，姿态误差 ${rotAte.toFixed(2)}°，` +
      `留出帧 RMS ${holdoutRms.toFixed(3)} m（${holdout.length} 帧），平均重投影 ${meanReproj.toFixed(2)} px`,
    metrics: {
      scale: br?.scale ?? NaN,
      ateM: ate.ate,
      rotAteDeg: rotAte,
      holdoutRmsM: holdoutRms,
      meanReprojPx: meanReproj,
      frames: kernel.frames.length,
    },
  };
}

function checkNoDefaultMasking(): ValidationCheck {
  const input: BootstrapInput = {
    cameraTimes: [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    imu: [],
    bgGuess: [0, 0, 0],
    R_bc: defaultExtrinsics().R_bc,
    t_bc: defaultExtrinsics().t_bc,
    K: defaultIntrinsics(),
    start: 1,
    end: 6,
    relative: new Map(),
    trackPixels: new Map(),
    trackBirth: new Map(),
    timeOffsetGuess: 0,
  };
  const res = runBootstrap(input, 2, { timeOffsetGuess: 0 });
  const mentionsCount = res.message.includes('样本不足');
  const ok = !res.ok && mentionsCount && res.scale === null;
  return {
    id: 'no-default-masking',
    title: '无默认遮错：静止窗口为空时初始化显式失败',
    ok,
    detail: `空 IMU 流的初始化：ok=${res.ok}，尺度=${res.scale === null ? 'null' : String(res.scale)}，消息「${res.message}」`,
    metrics: { bootstrapOk: res.ok ? 1 : 0 },
  };
}

export function runInertialValidation(onCheck?: (c: ValidationCheck, index: number, total: number) => void): ValidationReport {
  const t0 = Date.now();
  const steps: Array<() => ValidationCheck> = [
    checkStaticConventions,
    checkAttitudeTrajectory,
    checkReprojectionDeskew,
    checkClosedLoop,
    checkLoaderFlow,
    checkExportImport,
    checkStaleSnapshot,
    checkScaleHonesty,
    checkNoDefaultMasking,
    checkLensReplay,
  ];
  const checks: ValidationCheck[] = [];
  steps.forEach((step, i) => {
    const c = step();
    checks.push(c);
    onCheck?.(c, i, steps.length);
  });
  return { ok: checks.every((c) => c.ok), checks, durationMs: Date.now() - t0 };
}
