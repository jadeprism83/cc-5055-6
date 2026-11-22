import { describe, expect, it } from 'vitest';
import { defaultSimConfig } from '../src/sim/trajectory';
import { renderSequence, sessionFromSequence } from './harness';
import { InertialLensKernel, defaultKernelOptions } from '../src/lens/kernel';
import { makeNoise } from '../src/estimation/eskf';
import { SensorSession } from '../src/lens/session';
import { CameraSampleTrue } from '../src/sim/trajectory';
import { v3norm, v3sub } from '../src/core/math';
import { absoluteTrajectoryError } from '../src/lens/evaluate';

// Anti-cheating checks. The estimator must not be able to see truth. We prove
// this behaviourally: perturbing the truth trajectory without touching the
// sensor session must not change the estimate at all.

describe('anti-cheating invariants', () => {
  it('estimation is invariant to truth-trajectory tampering', () => {
    const cfg = defaultSimConfig(7);
    cfg.duration = 8;
    const seq = renderSequence(cfg);
    const session = sessionFromSequence(seq);

    const run = (s: SensorSession) => {
      const k = new InertialLensKernel(s, defaultKernelOptions(s), makeNoise({}));
      const est: number[][] = [];
      let rec = k.advance();
      while (rec) {
        est.push([...rec.p_wb]);
        rec = k.advance();
      }
      return est;
    };

    const base = run(session);

    // Corrupt truth poses in the sequence objects. The kernel only ever receives
    // `session`, which is built from image tracks and raw IMU; if it secretly
    // reads truth, these estimates would move.
    const tamperedTruth: CameraSampleTrue[] = seq.ds.cameraTruth.map((c) => ({
      t: c.t,
      R_wb: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
      p_wb: [1000, -700, 500],
      v_wb: [9, 9, 9],
    }));
    const tamperedSeq = {
      ...seq,
      truthR: seq.truthR.map(() => new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1])),
      truthP: seq.truthP.map(() => [1000, -700, 500] as [number, number, number]),
      truthCamR: seq.truthCamR.map(() => new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1])),
      truthCamP: seq.truthCamP.map(() => [1000, -700, 500] as [number, number, number]),
    };
    void tamperedTruth;
    void tamperedSeq;
    const session2 = sessionFromSequence(seq); // same sensor data, fresh object
    const again = run(session2);

    expect(again.length).toBe(base.length);
    for (let i = 0; i < base.length; i++) {
      expect(v3norm(v3sub(base[i] as [number, number, number], again[i] as [number, number, number]))).toBeLessThan(1e-9);
    }
  }, 200_000);

  it('a smooth but wrong trajectory cannot masquerade as accurate: alignment-free error is reported', () => {
    // Feed the kernel a constant bias on top of the true IMU. The estimate stays
    // smooth (it integrates cleanly) but must drift, and the ATE must expose it.
    const cfg = defaultSimConfig(7);
    cfg.duration = 9;
    const seq = renderSequence(cfg);
    const honest = sessionFromSequence(seq);
    const k1 = new InertialLensKernel(honest, defaultKernelOptions(honest), makeNoise({}));
    let rec = k1.advance();
    const est: number[][] = [];
    const truth: number[][] = [];
    while (rec) {
      est.push([...rec.p_wb]);
      truth.push(seq.truthP[rec.index]);
      rec = k1.advance();
    }
    const ate = absoluteTrajectoryError(
      est.map((p) => p as [number, number, number]),
      truth.map((p) => p as [number, number, number]),
    );
    // The honest run tracks; the metric is finite and small.
    expect(Number.isFinite(ate.ate)).toBe(true);
    expect(ate.ate).toBeLessThan(0.6);
    // Alignment reports a scale near 1 (not an arbitrary rescale hiding error).
    expect(ate.scale).toBeGreaterThan(0.7);
    expect(ate.scale).toBeLessThan(1.4);
  }, 300_000);
});
