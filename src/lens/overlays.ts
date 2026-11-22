import { Extrinsics, Intrinsics, cameraCenterWorld, project, worldToCamera } from '../core/camera';
import { Mat3, Vec3 } from '../core/math';
import { CalibrationRecord } from './record';

// Overlay projections shared by the lens window and the maintenance checks.
// Both the fixed corner marker and the virtual ground contact point are drawn
// from an attitude/pose snapshot, and both refuse a snapshot whose calibration
// generation no longer matches the live record: after a firmware swap, a
// late-arriving snapshot computed under the old conversion is rejected by
// both, never silently drawn.

export interface PoseSnapshot {
  recordId: string;
  generation: number;
  t: number;
  R_wb: Mat3;
  p_wb: Vec3;
}

export function acceptSnapshot(snap: PoseSnapshot, record: CalibrationRecord): boolean {
  return (
    snap.recordId === record.id &&
    snap.generation === record.generation &&
    record.status === 'committed'
  );
}

export interface OverlayPixel {
  u: number;
  v: number;
  valid: boolean;
}

/** Fixed virtual corner marker anchored in the world. Null = snapshot rejected. */
export function markerProjection(
  snap: PoseSnapshot,
  record: CalibrationRecord,
  ext: Extrinsics,
  K: Intrinsics,
  markerWorld: Vec3,
): OverlayPixel | null {
  if (!acceptSnapshot(snap, record)) return null;
  const px = project(worldToCamera(snap.R_wb, snap.p_wb, ext, markerWorld), K);
  return { u: px.u, v: px.v, valid: px.valid };
}

/** Virtual ground contact: the world point straight below the camera centre. */
export function groundContactWorld(snap: PoseSnapshot, ext: Extrinsics, groundZ: number): Vec3 {
  const C = cameraCenterWorld(snap.R_wb, snap.p_wb, ext);
  return [C[0], C[1], groundZ];
}

/** Ground contact projected into the image. Null = snapshot rejected. */
export function groundContactProjection(
  snap: PoseSnapshot,
  record: CalibrationRecord,
  ext: Extrinsics,
  K: Intrinsics,
  groundZ: number,
): (OverlayPixel & { contact: Vec3 }) | null {
  if (!acceptSnapshot(snap, record)) return null;
  const contact = groundContactWorld(snap, ext, groundZ);
  const px = project(worldToCamera(snap.R_wb, snap.p_wb, ext, contact), K);
  return { u: px.u, v: px.v, valid: px.valid, contact };
}
