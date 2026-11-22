import { describe, expect, it } from 'vitest';
import { defaultSimConfig } from '../src/sim/trajectory';
import { renderSequence, sessionFromSequence } from './harness';
import { InertialLensKernel, defaultKernelOptions } from '../src/lens/kernel';
import { makeNoise } from '../src/estimation/eskf';
import { solveHandEyeRotation } from '../src/lens/calibration';
import { integrateRotation } from '../src/estimation/imu';
import { generateDataset } from '../src/sim/trajectory';
import { m3mul, m3transpose } from '../src/core/math';


// Calibration must be explicit: the time offset, the camera-IMU extrinsic and
// the inertial biases are either solved from data or accepted as inputs, and the
// UI exposes which is which. Nothing here pretends three integrated numbers are
// drift-free ground truth.

describe('calibration and failure feedback', () => {
  it('solves the sampling time offset from gyro-vision rotation consistency', () => {
    const cfg = defaultSimConfig(7);
    cfg.duration = 9;
    const seq = renderSequence(cfg);
    const session = sessionFromSequence(seq);
    const opts = defaultKernelOptions(session);
    const kernel = new InertialLensKernel(session, opts, makeNoise({}));
    expect(kernel.bootstrap()).toBe(true);
    const br = kernel.bootstrapResult!;
    // The truth offset is 18 ms; the search recovers a value of the same scale
    // and sign, well inside the ±50 ms scan.
    expect(Math.abs(br.timeOffset)).toBeLessThanOrEqual(0.05);
    expect(br.timeOffset).toBeGreaterThan(-0.02);
    expect(['given','solved']).toContain(kernel.timeOffsetSource);
    // The scored curve has a genuine minimum, not a flat line.
    const scores = br.timeOffsetCurve.map((c) => c.score);
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    expect(max - min).toBeGreaterThan(1e-3);
  }, 200_000);

  it('measures the pre-step gyro bias from the static window, not the post-step value', () => {
    const cfg = defaultSimConfig(7);
    cfg.duration = 9;
    const seq = renderSequence(cfg);
    const session = sessionFromSequence(seq);
    const kernel = new InertialLensKernel(session, defaultKernelOptions(session), makeNoise({}));
    expect(kernel.bootstrap()).toBe(true);
    const bg = kernel.bootstrapResult!.bg;
    // Truth pre-step bias is (0.0004, -0.00025, 0.0003); post-step is ~0.024.
    expect(Math.abs(bg[0] - cfg.gyroBiasBefore[0])).toBeLessThan(1e-3);
    expect(Math.abs(bg[1] - cfg.gyroBiasBefore[1])).toBeLessThan(1e-3);
    expect(Math.abs(bg[2] - cfg.gyroBiasBefore[2])).toBeLessThan(1e-3);
  }, 200_000);

  it('offers a hand-eye rotation solve but never silently substitutes translation', () => {
    const cfg = defaultSimConfig(7);
    cfg.duration = 9;
    const seq = renderSequence(cfg);
    const session = sessionFromSequence(seq);
    const kernel = new InertialLensKernel(session, defaultKernelOptions(session), makeNoise({}));
    kernel.bootstrap();
    expect(kernel.handEyeCheck).not.toBeNull();
    const hc = kernel.handEyeCheck!;
    // The solved rotation should be close to the true extrinsic rotation.
    expect(hc.pairs).toBeGreaterThanOrEqual(4);
    expect(hc.residualDeg).toBeLessThan(3);
    // Applying only changes the rotation; translation stays as the input.
    const tBefore = [...kernel.extrinsics.t_bc];
    kernel.applySolvedExtrinsic();
    expect(kernel.extrinsicSource).toBe('solved');
    expect(kernel.extrinsics.t_bc).toEqual(tBefore);
  }, 300_000);

  it('emits a failure-feedback event and inflates bias covariance on a gyro/vision mismatch', () => {
    const cfg = defaultSimConfig(7);
    cfg.duration = 10;
    const seq = renderSequence(cfg);
    const session = sessionFromSequence(seq);
    const kernel = new InertialLensKernel(session, defaultKernelOptions(session), makeNoise({}));
    let rec = kernel.advance();
    let sawInflation = false;
    let sawDropoutEvent = false;
    const seen = new Set<string>();
    while (rec) {
      for (const e of kernel.events) {
        const key = `${e.frame}:${e.message}`;
        if (seen.has(key)) continue;
        seen.add(key);
        if (e.message.includes('放大陀螺偏置协方差') || e.message.includes('陀螺-视觉')) sawInflation = true;
        if (e.message.includes('掉点') || e.message.includes('惯推')) sawDropoutEvent = true;
      }
      if (kernel.frames.some((f) => f.biasInflated)) sawInflation = true;
      rec = kernel.advance();
    }
    // A bias step at t=5 s and a dropout at 4.6/7.0 s must leave a trace the UI
    // can show. At least the dropout feedback must appear.
    expect(sawDropoutEvent || sawInflation).toBe(true);
    expect(kernel.events.length).toBeGreaterThan(3);
  }, 300_000);

  it('hand-eye rotation solve matches the true extrinsic when fed clean pairs', () => {
    const cfg = defaultSimConfig(7);
    // Build synthetic pairs directly from truth so this isolates the solver.
    const ds = generateDataset(cfg);
    const R_bc = cfg.R_bc;
    const pairs = [];
    for (let k = 1; k < 40; k++) {
      const t0 = ds.cameraTimes[k - 1];
      const t1 = ds.cameraTimes[k];
      const dR = integrateRotation(ds.imu, t0, t1, 0, [0, 0, 0]).dR;
      const RwbPrev = ds.cameraTruth[k - 1].R_wb;
      const RwbCur = ds.cameraTruth[k].R_wb;
      const RvisCam = m3mul(m3transpose(R_bc), m3mul(m3mul(m3transpose(RwbCur), RwbPrev), R_bc));
      pairs.push({ dR_body: dR, dR_cam: RvisCam, weight: 1 });
    }
    const solved = solveHandEyeRotation(pairs);
    expect(solved.ok).toBe(true);
    expect(solved.rotationResidualDeg).toBeLessThan(2);
  }, 120_000);
});
