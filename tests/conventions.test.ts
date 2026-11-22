import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { Mat3, Vec3, m3identity, m3mul, m3transpose, mulberry32, v3norm } from '../src/core/math';
import { defaultExtrinsics, defaultIntrinsics, projectWorld } from '../src/core/camera';
import {
  GRAVITY_MAG,
  accelFromBody,
  accelToBody,
  checkConventionShape,
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
} from '../src/core/conventions';
import {
  defaultCalibrationRecord,
  exportCalibration,
  firmwareBRecord,
  importCalibration,
  loadStaticCalibration,
  mirrorAboutVertical,
  refineWithRotationRuns,
  solidifyConvention,
} from '../src/lens/record';
import { GyroPreintegrator } from '../src/lens/preintegration';
import { ShutterCompensator } from '../src/lens/shutter';
import { PoseSnapshot, groundContactProjection, markerProjection } from '../src/lens/overlays';
import { closedLoopSpin, orderedRotationRun, sinusoidMotion } from '../src/sim/analytic';
import { buildSession } from '../src/lens/session';
import { computeDocExamples } from '../src/lens/docExamples';
import { fitScaleAndVelocity, integrateWorldAcceleration } from '../src/lens/bootstrap';
import { ImuSample } from '../src/sim/trajectory';

// Sensor-convention maintenance checks: axes, units, converter invertibility,
// the calibration loader flow, export/import, stale-snapshot rejection, and
// the doc-example block. These run without rendering; the heavy lens replay
// lives in the validation report (tests/validation.test.ts).

const DEG = 180 / Math.PI;

function angleDeg(R: Mat3): number {
  const c = Math.min(1, Math.max(-1, (R[0] + R[4] + R[8] - 1) / 2));
  return Math.acos(c) * DEG;
}

function errDeg(a: Mat3, b: Mat3): number {
  return angleDeg(m3mul(m3transpose(a), b));
}

function v3close(a: Vec3, b: Vec3, tol = 1e-12): void {
  for (let i = 0; i < 3; i++) expect(Math.abs(a[i] - b[i])).toBeLessThan(tol);
}

describe('sensor conventions: axes and units', () => {
  it('converts gyro/accel between body and sensor frames with units', () => {
    const fwB = firmwareB();
    const wBody: Vec3 = [0.1, -0.2, 0.3];
    const raw = gyroFromBody(fwB, wBody);
    // fw-B reports deg/s on permuted axes: raw = R_bsᵀ · w · 180/π.
    expect(raw[0]).toBeCloseTo(0.2 * DEG, 9);
    expect(raw[1]).toBeCloseTo(0.3 * DEG, 9);
    expect(raw[2]).toBeCloseTo(-0.1 * DEG, 9);
    v3close(gyroToBody(fwB, raw), wBody);
    const fBody: Vec3 = [0.5, -0.3, GRAVITY_MAG];
    v3close(accelToBody(fwB, accelFromBody(fwB, fBody)), fBody);
    // fw-A is the identity conversion.
    const fwA = firmwareA();
    v3close(gyroFromBody(fwA, wBody), wBody);
    v3close(accelFromBody(fwA, fBody), fBody);
  });

  it('validates convention shape and rejects improper rotations for commit', () => {
    expect(checkConventionShape(firmwareA(), true)).toBeNull();
    expect(checkConventionShape(firmwareB(), true)).toBeNull();
    const mirrored = mirrorAboutVertical(firmwareB());
    expect(checkConventionShape(mirrored, false)).toBeNull();
    expect(checkConventionShape(mirrored, true)).toMatch(/镜像/);
    expect(checkConventionShape({ ...firmwareA(), gyroUnit: 'grad/s' as never }, false)).toMatch(/角速度单位/);
  });

  it('static check passes the true convention and fails swap/unit mistakes', () => {
    const fwB = firmwareB();
    const raw = accelFromBody(fwB, [0, 0, GRAVITY_MAG]);
    expect(checkStaticConvention(fwB, raw).ok).toBe(true);
    expect(checkStaticConvention(firmwareA(), raw).ok).toBe(false);
    expect(checkStaticConvention({ ...fwB, accelUnit: 'm/s^2' }, raw).ok).toBe(false);
    const swapped = { ...fwB, R_bs: new Float64Array([0, 0, -1, 0, 1, 0, -1, 0, 0]) };
    expect(checkStaticConvention(swapped, raw).ok).toBe(false);
  });
});

describe('axis converter between firmwares', () => {
  it('is exactly invertible on samples', () => {
    const conv = converterBetween(firmwareA(), firmwareB());
    const inv = invertConverter(conv);
    const rng = mulberry32(42);
    for (let k = 0; k < 50; k++) {
      const w: Vec3 = [rng() - 0.5, rng() - 0.5, rng() - 0.5];
      v3close(convertGyroSample(inv, convertGyroSample(conv, w)), w);
    }
  });

  it('closed-loop rotation detects a double axis-swap through imaging', () => {
    const fwA = firmwareA();
    const fwB = firmwareB();
    const spin = closedLoopSpin([0, 0, 1], 1.0);
    const rawA: Array<{ t: number; w: Vec3 }> = [];
    const nSpin = Math.floor(spin.duration * 200);
    for (let k = 0; k <= nSpin; k++) {
      const t = k / 200;
      rawA.push({ t, w: gyroFromBody(fwA, spin.omegaB(t)) });
    }
    const conv = converterBetween(fwA, fwB);
    const asB = rawA.map((s) => ({ t: s.t, w: convertGyroSample(conv, s.w) }));
    const good = integrateGyroBody(asB.map((s) => ({ t: s.t, w: gyroToBody(fwB, s.w) })));
    expect(angleDeg(good)).toBeLessThan(0.5);
    const ext = defaultExtrinsics();
    const K = defaultIntrinsics();
    const marker: Vec3 = [-2.5, 0.4, 1.1];
    const midT = spin.duration / 2;
    const imgErr = (samples: Array<{ t: number; w: Vec3 }>): number => {
      const half = samples.filter((s) => s.t <= midT);
      const Rmid = integrateGyroBody(half.map((s) => ({ t: s.t, w: gyroToBody(fwB, s.w) })));
      const px = projectWorld(Rmid, [0, 0, 0], ext, K, marker);
      const pxTrue = projectWorld(spin.R(midT), [0, 0, 0], ext, K, marker);
      if (!px.valid || !pxTrue.valid) return Infinity;
      return Math.hypot(px.u - pxTrue.u, px.v - pxTrue.v);
    };
    expect(imgErr(asB)).toBeLessThan(1);
    const doubleSwapped = asB.map((s) => ({ t: s.t, w: convertGyroSample(conv, s.w) }));
    expect(imgErr(doubleSwapped)).toBeGreaterThan(50);
  });
});

describe('attitude trajectory from the analytic generator', () => {
  it('matches the exact attitude increment for uniform and non-uniform sampling', () => {
    const fwB = firmwareB();
    const motion = sinusoidMotion();
    const expected = m3mul(m3transpose(motion.R(0)), motion.R(motion.duration));
    const integrate = (samples: Array<{ t: number; w: Vec3 }>) =>
      integrateGyroBody(samples.map((s) => ({ t: s.t, w: gyroToBody(fwB, s.w) })));
    const uniform: Array<{ t: number; w: Vec3 }> = [];
    for (let k = 0; k <= 800; k++) {
      const t = k / 200;
      uniform.push({ t, w: gyroFromBody(fwB, motion.omegaB(t)) });
    }
    expect(errDeg(integrate(uniform), expected)).toBeLessThan(0.01);
    // Non-uniform (jittered) sampling: the same check must still pass.
    const rng = mulberry32(7);
    const jittered: Array<{ t: number; w: Vec3 }> = [{ t: 0, w: gyroFromBody(fwB, motion.omegaB(0)) }];
    let tPrev = 0;
    for (let k = 1; k < 800; k++) {
      let t = k / 200 + (rng() - 0.5) * 0.6 * (1 / 200);
      if (t <= tPrev + 1e-6) t = tPrev + 1e-6;
      tPrev = t;
      jittered.push({ t, w: gyroFromBody(fwB, motion.omegaB(t)) });
    }
    jittered.push({ t: motion.duration, w: gyroFromBody(fwB, motion.omegaB(motion.duration)) });
    expect(errDeg(integrate(jittered), expected)).toBeLessThan(0.01);
    // Axis swap or unit change must fail the rotation-convention check.
    expect(checkRotationConvention(fwB, uniform, expected, 3).ok).toBe(true);
    const swapped = { ...fwB, R_bs: new Float64Array([0, 0, -1, 0, 1, 0, -1, 0, 0]) };
    expect(checkRotationConvention(swapped, uniform, expected, 3).ok).toBe(false);
    expect(checkRotationConvention({ ...fwB, gyroUnit: 'rad/s' }, uniform, expected, 3).ok).toBe(false);
  });
});

describe('calibration loader flow', () => {
  const fwB = firmwareB();
  const ext = defaultExtrinsics();
  const staticData = {
    imu: Array.from({ length: 60 }, (_, k) => ({
      t: k / 200,
      gyro: gyroFromBody(fwB, [0, 0, 0]),
      accel: accelFromBody(fwB, [0, 0, GRAVITY_MAG]),
    })),
  };
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

  it('static data keeps mirror candidates as preview and refuses to commit', () => {
    const rec = loadStaticCalibration(staticData, [fwB, mirrorAboutVertical(fwB)], {
      timeOffset: 0.018,
      extrinsics: ext,
    });
    expect(rec.status).toBe('preview');
    expect(rec.previews.length).toBe(2);
    expect(solidifyConvention(rec)).toBeNull();
  });

  it('ordered rotation runs exclude the mirror and commit exactly one convention', () => {
    const rec = loadStaticCalibration(staticData, [fwB, mirrorAboutVertical(fwB)], {
      timeOffset: 0.018,
      extrinsics: ext,
    });
    const committed = refineWithRotationRuns(rec, runs);
    expect(committed.status).toBe('committed');
    expect(committed.generation).toBe(rec.generation + 1);
    expect(committed.convention.firmware).toBe('fw-B');
    expect(committed.previews.length).toBe(1);
    expect(solidifyConvention(committed)).not.toBeNull();
  });

  it('rotation runs reject a wrong gyro unit instead of committing it', () => {
    const wrong = { ...fwB, gyroUnit: 'rad/s' as const };
    const rec = loadStaticCalibration(staticData, [wrong], { timeOffset: 0.018, extrinsics: ext });
    const out = refineWithRotationRuns(rec, runs);
    expect(out.status).toBe('preview');
    expect(out.previews.length).toBe(0);
  });

  it('refuses to load a static dataset with too few samples (no defaults)', () => {
    expect(() =>
      loadStaticCalibration({ imu: staticData.imu.slice(0, 3) }, [fwB], { timeOffset: 0, extrinsics: ext }),
    ).toThrow(/静止标定素材不足/);
  });
});

describe('export / import of the calibration record', () => {
  it('round-trips and drives identical integration and de-skew', () => {
    const ext = defaultExtrinsics();
    const rec = firmwareBRecord(3, ext, 0.018);
    const back = importCalibration(exportCalibration(rec));
    expect(back.generation).toBe(3);
    expect(back.status).toBe('committed');
    const fwB = firmwareB();
    const samples = [0, 1, 2, 3].map((k) => ({
      t: k * 0.005,
      gyro: gyroFromBody(fwB, [0.1 + k * 0.01, 0.2, -0.1] as Vec3),
      fw: 'fw-B',
    }));
    const s1 = new GyroPreintegrator(rec).integrate(samples, [0.001, -0.001, 0.0005]);
    const s2 = new GyroPreintegrator(back).integrate(samples, [0.001, -0.001, 0.0005]);
    expect(s1).not.toBeNull();
    expect(s2).not.toBeNull();
    for (let i = 0; i < 9; i++) expect(s2!.dR[i]).toBe(s1!.dR[i]);
    const K = defaultIntrinsics();
    const raw = gyroFromBody(fwB, [0.3, -0.2, 0.1]);
    const d1 = new ShutterCompensator(rec, 25e-6).deskewWithRaw(160, 180, K, raw, 'fw-B');
    const d2 = new ShutterCompensator(back, 25e-6).deskewWithRaw(160, 180, K, raw, 'fw-B');
    expect(d1).not.toBeNull();
    expect(d2).toEqual(d1);
  });

  it('throws on tampered or incomplete records', () => {
    const rec = firmwareBRecord(2, defaultExtrinsics(), 0.018);
    const json = exportCalibration(rec);
    const base = JSON.parse(json) as Record<string, unknown>;
    const mirrorObj = JSON.parse(json) as { convention: { R_bs: number[] } };
    mirrorObj.convention.R_bs = [0, 0, 1, -1, 0, 0, 0, 1, 0];
    const twoPreviews = JSON.parse(json) as { previews: unknown[] };
    twoPreviews.previews = [twoPreviews.previews[0], twoPreviews.previews[0]];
    const bad = [
      JSON.stringify(mirrorObj),
      JSON.stringify({ ...base, generation: undefined }),
      JSON.stringify(twoPreviews),
      JSON.stringify({ ...base, extrinsics: undefined }),
      JSON.stringify({ ...base, schema: 'other/1' }),
      'not json',
    ];
    for (const text of bad) expect(() => importCalibration(text)).toThrow(/标定记录无效/);
  });
});

describe('stale snapshot after a firmware swap', () => {
  it('is rejected by both the corner marker and the ground contact point', () => {
    const ext = defaultExtrinsics();
    const K = defaultIntrinsics();
    const marker: Vec3 = [4.6, 0.55, 1.25];
    const recordA = defaultCalibrationRecord(ext, 0.018);
    const preA = new GyroPreintegrator(recordA);
    const fwA = firmwareA();
    const snap = preA.integrate(
      [0, 1, 2].map((k) => ({ t: k * 0.005, gyro: gyroFromBody(fwA, [0.2, 0.1, 0] as Vec3), fw: 'fw-A' })),
      [0, 0, 0],
    );
    expect(snap).not.toBeNull();
    const pose: PoseSnapshot = {
      recordId: snap!.recordId,
      generation: snap!.generation,
      t: 0.01,
      R_wb: m3identity(),
      p_wb: [0, 0, 0.4],
    };
    expect(markerProjection(pose, recordA, ext, K, marker)).not.toBeNull();
    expect(groundContactProjection(pose, recordA, ext, K, -0.6)).not.toBeNull();
    const recordB = firmwareBRecord(recordA.generation + 1, ext, 0.018);
    expect(markerProjection(pose, recordB, ext, K, marker)).toBeNull();
    expect(groundContactProjection(pose, recordB, ext, K, -0.6)).toBeNull();
    const fresh: PoseSnapshot = { ...pose, generation: recordB.generation };
    expect(markerProjection(fresh, recordB, ext, K, marker)).not.toBeNull();
    expect(groundContactProjection(fresh, recordB, ext, K, -0.6)).not.toBeNull();
  });
});

describe('session convention resolver', () => {
  it('drops samples without a committed convention instead of default-converting', () => {
    const fwB = firmwareB();
    const imu: ImuSample[] = [
      { t: 0.0, gyro: [0, 0, 0.1], accel: [0, 0, 9.80665], fw: 'fw-A' },
      { t: 0.005, gyro: [1, 2, 3], accel: [0, 0, 1], fw: 'fw-B' },
      { t: 0.01, gyro: [0, 0, 0.1], accel: [0, 0, 9.80665], fw: 'fw-A' },
    ];
    const session = buildSession([0, 0.05, 0.1], imu, [], [[], [], []], [0, 0, 0], defaultIntrinsics(), {
      givenExtrinsics: defaultExtrinsics(),
      pixelNoise: 0,
      seed: 1,
    });
    // Default resolver knows fw-A only: the fw-B sample is dropped.
    const body = session.imuBody();
    expect(body.length).toBe(2);
    expect(session.unknownFirmwareInfo()).toEqual({ fw: 'fw-B', firstT: 0.005 });
    // After committing fw-B, its sample is converted through its own convention.
    session.setConventionResolver((fw) => (fw === 'fw-B' ? fwB : fw === 'fw-A' ? firmwareA() : null));
    const body2 = session.imuBody();
    expect(body2.length).toBe(3);
    expect(session.unknownFirmwareInfo()).toBeNull();
    const converted = body2.find((s) => s.fw === 'fw-B');
    expect(converted).toBeDefined();
    v3close(converted!.gyro, gyroToBody(fwB, [1, 2, 3]));
  });
});

describe('shutter compensator', () => {
  it('recovers the reference-row projection and refuses unknown firmware', () => {
    const ext = defaultExtrinsics();
    const K = defaultIntrinsics();
    const motion = sinusoidMotion();
    const t = 1.25;
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
    const readout = 40e-6;
    const v = 200;
    const omega = motion.omegaB(t);
    const dR = integrateGyroBody([{ t: 0, w: [0, 0, 0] as Vec3 }, { t: v * readout, w: omega }]);
    const pxRow = projectWorld(m3mul(R_wb, dR), p_wb, ext, K, point);
    const record = firmwareBRecord(2, ext, 0.018);
    const shutter = new ShutterCompensator(record, readout);
    const raw = gyroFromBody(firmwareB(), omega);
    const fixed = shutter.deskewWithRaw(pxRow.u, pxRow.v, K, raw, 'fw-B');
    expect(fixed).not.toBeNull();
    expect(Math.hypot(fixed!.u - px0.u, fixed!.v - px0.v)).toBeLessThan(0.5);
    expect(shutter.deskewWithRaw(pxRow.u, pxRow.v, K, raw, 'fw-A')).toBeNull();
  });
});

describe('scale honesty', () => {
  it('does not claim a scale from unconstrained static data', () => {
    const times: number[] = [];
    const Cvis: Vec3[] = [];
    const imu: ImuSample[] = [];
    for (let k = 0; k <= 40; k++) {
      times.push(k / 20);
      Cvis.push([0, 0, 0]);
      imu.push({ t: k / 200, gyro: [0, 0, 0], accel: [0, 0, GRAVITY_MAG] });
    }
    const integrals = times.map((_, k) =>
      integrateWorldAcceleration(imu, times, 0, k, 0, () => m3identity(), [0, 0, 0], [0, 0, -GRAVITY_MAG]),
    );
    const fit = fitScaleAndVelocity(times, Cvis, integrals, 0);
    expect(fit.ok).toBe(false);
  });
});

describe('documentation examples', () => {
  it('match the machine-generated block in docs/sensor-conventions.md', () => {
    const text = readFileSync(resolve(__dirname, '../docs/sensor-conventions.md'), 'utf8');
    const b = text.indexOf('<!-- BEGIN EXAMPLES -->');
    const e = text.indexOf('<!-- END EXAMPLES -->');
    expect(b).toBeGreaterThanOrEqual(0);
    expect(e).toBeGreaterThan(b);
    const block = text.slice(b, e);
    const jsonMatch = block.match(/```json\n([\s\S]*?)\n```/);
    expect(jsonMatch).not.toBeNull();
    const parsed = JSON.parse(jsonMatch![1]) as unknown;
    expect(parsed).toEqual(computeDocExamples());
  });
});
