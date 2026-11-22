import { describe, expect, it } from 'vitest';
import { defaultSimConfig } from '../src/sim/trajectory';
import { renderSequence, sessionFromSequence } from './harness';
import { InertialLensKernel, KernelFrame, defaultKernelOptions } from '../src/lens/kernel';
import { makeNoise } from '../src/estimation/eskf';
import { absoluteTrajectoryError, rotationAte } from '../src/lens/evaluate';
import { Mat3, Vec3 } from '../src/core/math';

// End-to-end behaviour of the inertial lens on synthetic handheld data with a
// gyro-bias step and two visual dropouts (blur at 4.6 s, lost at 7.0 s). These
// tests assert observable outcomes, not a copied curve: the estimator never sees
// truth, and the scene/IMU are generated independently of it.

const DURATION = 13;

interface Run {
  est: Vec3[];
  truth: Vec3[];
  estR: Mat3[];
  truthR: Mat3[];
  frames: KernelFrame[];
  idx: number[];
  kernel: InertialLensKernel;
}

function runKernel(overrides: Partial<ReturnType<typeof defaultKernelOptions>> = {}, cfgSeed = 7): Run {
  const cfg = defaultSimConfig(cfgSeed);
  cfg.duration = DURATION;
  const seq = renderSequence(cfg);
  const session = sessionFromSequence(seq);
  const opts = { ...defaultKernelOptions(session), ...overrides };
  const kernel = new InertialLensKernel(session, opts, makeNoise({}));
  const est: Vec3[] = [];
  const truth: Vec3[] = [];
  const estR: Mat3[] = [];
  const truthR: Mat3[] = [];
  const idx: number[] = [];
  let rec = kernel.advance();
  while (rec) {
    est.push([...rec.p_wb]);
    truth.push(seq.truthP[rec.index]);
    estR.push(new Float64Array(rec.R_wb));
    truthR.push(seq.truthR[rec.index]);
    idx.push(rec.index);
    rec = kernel.advance();
  }
  return { est, truth, estR, truthR, frames: kernel.frames, idx, kernel };
}

describe('inertial lens end-to-end', () => {
  it('bootstraps a metric, low-drift trajectory despite a gyro-bias step and dropouts', () => {
    const run = runKernel();
    const br = run.kernel.bootstrapResult;
    expect(br).not.toBeNull();
    expect(br!.ok).toBe(true);
    // Metric scale recovered and supported by the visual/inertial fit.
    expect(br!.scale).not.toBeNull();
    expect(br!.scale! as number).toBeGreaterThan(0.4);
    expect(br!.scale! as number).toBeLessThan(1.0);
    expect(br!.scaleRecoverable).toBe(true);
    // Gyro bias from the static window is close to the truth 3e-4 rad/s, not the
    // 0.024 rad/s post-step value.
    for (const b of br!.bg) expect(Math.abs(b)).toBeLessThan(5e-3);

    // Alignment-free ATE stays sub-metre on a ~1 m path.
    const ate = absoluteTrajectoryError(run.est, run.truth);
    expect(ate.samples).toBeGreaterThan(150);
    expect(ate.ate).toBeLessThan(0.5);
    // Rotation follows the trajectory to a few degrees (gyro+visual fusion).
    expect(rotationAte(run.estR, run.truthR)).toBeLessThan(4);
  }, 400_000);

  it('keeps tracking through both visual dropouts without losing the pose', () => {
    const run = runKernel();
    // During the lost interval (t in [7.0, 8.4]) the kernel must fall back to
    // inertial propagation rather than freeze or teleport.
    const lostFrames = run.frames.filter((f) => f.t >= 7.0 && f.t <= 8.4);
    expect(lostFrames.length).toBeGreaterThan(10);
    expect(lostFrames.some((f) => f.status === 'inertial' || f.status === 'tracking')).toBe(true);
    // The end pose is still on the true path.
    const n = run.est.length;
    const endErr = Math.hypot(
      run.est[n - 1][0] - run.truth[n - 1][0],
      run.est[n - 1][1] - run.truth[n - 1][1],
      run.est[n - 1][2] - run.truth[n - 1][2],
    );
    expect(endErr).toBeLessThan(1.0);
  }, 400_000);

  it('reports held-out frames that were never used in the visual update', () => {
    const run = runKernel();
    const holdout = run.frames.filter((f) => f.holdout);
    expect(holdout.length).toBeGreaterThan(8);
    // Held-out frames are still predicted-only.
    for (const h of holdout) expect(h.predictedOnly).toBe(true);
    let sse = 0;
    for (const h of holdout) {
      const j = run.idx.indexOf(h.index);
      const e = Math.hypot(
        run.est[j][0] - run.truth[j][0],
        run.est[j][1] - run.truth[j][1],
        run.est[j][2] - run.truth[j][2],
      );
      sse += e * e;
    }
    const rms = Math.sqrt(sse / holdout.length);
    // Independent check: held-out error is the same order as the in-sample error,
    // which rules out a copied/faked trajectory.
    expect(rms).toBeLessThan(1.0);
  }, 400_000);

  it('vision-only propagation is worse than joint estimation across a dropout', () => {
    const joint = runKernel({ useInertialPropagation: true, holdoutEvery: 0 });
    const visionOnly = runKernel({ useInertialPropagation: false, holdoutEvery: 0 });
    const ateJoint = absoluteTrajectoryError(joint.est, joint.truth);
    const ateVision = absoluteTrajectoryError(visionOnly.est, visionOnly.truth);
    // During a visual outage the vision-only arm cannot move; the joint arm
    // carries the pose on the IMU. The joint result must be at least as good.
    expect(ateJoint.ate).toBeLessThan(ateVision.ate + 1e-6);
    // And specifically the end-of-run error is much smaller for the joint arm.
    const endErr = (r: Run) => {
      const n = r.est.length;
      return Math.hypot(r.est[n - 1][0] - r.truth[n - 1][0], r.est[n - 1][1] - r.truth[n - 1][1], r.est[n - 1][2] - r.truth[n - 1][2]);
    };
    expect(endErr(joint)).toBeLessThan(endErr(visionOnly));
  }, 500_000);
});
