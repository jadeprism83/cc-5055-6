import { Intrinsics, bearingCamera, project } from '../core/camera';
import { Vec3, m3TmulVec, m3mulVec, so3Exp, v3scale } from '../core/math';
import { gyroToBody } from '../core/conventions';
import { CalibrationRecord } from './record';

// Rolling-shutter de-skew. Each row v is read out dt = v · readoutPerRowSec
// after the reference row; during that time the camera keeps rotating, so a
// fixed world point is imaged at the row-time attitude. The compensator
// rotates the measured bearing back to the reference attitude with the gyro
// rate. It references the SAME calibration record as the gyro preintegrator,
// so both always apply the same committed axes/units.

export class ShutterCompensator {
  constructor(readonly record: CalibrationRecord, readonly readoutPerRowSec: number) {
    if (!(readoutPerRowSec > 0)) throw new Error('行读出时间必须为正数');
  }

  get generation(): number {
    return this.record.generation;
  }

  /**
   * De-skew one pixel. `omegaBodyRadS` is the body-frame rate at capture; the
   * record's extrinsics carry it into the camera frame. Row 0 is the reference
   * time, so a pixel measured at row v is rotated by Exp(ω · v · readout).
   */
  deskew(u: number, v: number, K: Intrinsics, omegaBodyRadS: Vec3): { u: number; v: number } {
    const omegaCam = m3TmulVec(this.record.extrinsics.R_bc, omegaBodyRadS);
    const dt = v * this.readoutPerRowSec;
    const b = bearingCamera({ u, v }, K);
    const b0 = m3mulVec(so3Exp(v3scale(omegaCam, dt)), b);
    const p = project(b0, K);
    return { u: p.u, v: p.v };
  }

  /**
   * De-skew with a raw sensor-frame gyro sample, converted through the same
   * record. Returns null when the firmware has no committed convention — the
   * caller must surface that, not fall back to a default conversion.
   */
  deskewWithRaw(u: number, v: number, K: Intrinsics, rawGyro: Vec3, fw?: string): { u: number; v: number } | null {
    if (this.record.status !== 'committed') return null;
    const tag = fw ?? this.record.firmware;
    if (tag !== this.record.firmware) return null;
    return this.deskew(u, v, K, gyroToBody(this.record.convention, rawGyro));
  }
}
