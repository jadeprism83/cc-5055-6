import { Mat3, Vec3, m3mul, so3Exp, v3add, v3scale } from '../core/math';
import { SensorConvention, gyroToBody, accelToBody } from '../core/conventions';
import { CalibrationRecord } from './record';

// The gyro preintegrator. It is handed the axis-compatibility result through
// the calibration record and converts every raw sample with that committed
// convention before integrating. Snapshots are stamped with the record's
// generation so a late snapshot computed under a previous firmware's
// conversion is recognisably stale and rejected by consumers.

export interface AttitudeSnapshot {
  recordId: string;
  generation: number;
  t0: number;
  t1: number;
  /** Body-frame attitude increment over [t0, t1]. */
  dR: Mat3;
  samples: number;
}

export class GyroPreintegrator {
  constructor(readonly record: CalibrationRecord) {}

  get generation(): number {
    return this.record.generation;
  }

  /** The committed convention for a firmware, or null when none is committed. */
  conventionFor(fw: string): SensorConvention | null {
    if (this.record.status !== 'committed') return null;
    return fw === this.record.firmware ? this.record.convention : null;
  }

  /** Raw gyro → body rad/s. Null when the firmware has no committed convention. */
  bodyGyro(fw: string, raw: Vec3): Vec3 | null {
    const conv = this.conventionFor(fw);
    return conv ? gyroToBody(conv, raw) : null;
  }

  /** Raw accel → body m/s². Null when the firmware has no committed convention. */
  bodyAccel(fw: string, raw: Vec3): Vec3 | null {
    const conv = this.conventionFor(fw);
    return conv ? accelToBody(conv, raw) : null;
  }

  /**
   * Trapezoid attitude integration over raw samples. Returns null when any
   * sample's firmware has no committed convention — the preintegrator refuses
   * to guess a conversion rather than integrating with a default one.
   */
  integrate(samples: Array<{ t: number; gyro: Vec3; fw?: string }>, bias: Vec3): AttitudeSnapshot | null {
    if (samples.length < 2) return null;
    let dR = so3Exp([0, 0, 0]);
    let n = 0;
    for (let i = 1; i < samples.length; i++) {
      const a = samples[i - 1];
      const b = samples[i];
      const h = b.t - a.t;
      if (h <= 0) continue;
      const wa = this.bodyGyro(a.fw ?? this.record.firmware, a.gyro);
      const wb = this.bodyGyro(b.fw ?? this.record.firmware, b.gyro);
      if (!wa || !wb) return null;
      const w = v3scale(v3add(wa, wb), 0.5);
      dR = m3mul(dR, so3Exp(v3scale([w[0] - bias[0], w[1] - bias[1], w[2] - bias[2]], h)));
      n++;
    }
    return {
      recordId: this.record.id,
      generation: this.record.generation,
      t0: samples[0].t,
      t1: samples[samples.length - 1].t,
      dR,
      samples: n,
    };
  }
}
