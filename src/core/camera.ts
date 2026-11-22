import {
  Mat3,
  Vec3,
  m3TmulVec,
  m3mul,
  m3mulVec,
  v3add,
  v3cross,
  v3dot,
  v3norm,
  v3normalize,
  v3scale,
  v3sub,
} from './math';

export interface Intrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
}

export interface Extrinsics {
  /** Rotation taking a point from camera coordinates to body coordinates. */
  R_bc: Mat3;
  /** Camera origin expressed in body coordinates. */
  t_bc: Vec3;
}

export interface Pixel {
  u: number;
  v: number;
  valid: boolean;
  /** Camera-frame depth, metres. */
  depth: number;
}

/** Default lens: body x forward, y left, z up; camera x right, y down, z forward. */
export function defaultIntrinsics(width = 320, height = 240): Intrinsics {
  return { fx: 250, fy: 250, cx: width / 2, cy: height / 2, width, height };
}

export function defaultExtrinsics(): Extrinsics {
  // R_bc columns are camera axes expressed in body: right=-y, down=-z, forward=+x.
  return {
    R_bc: new Float64Array([0, 0, 1, -1, 0, 0, 0, -1, 0]),
    t_bc: [0.015, -0.012, 0.006],
  };
}

export function cameraToBody(ext: Extrinsics, p_c: Vec3): Vec3 {
  return v3add(m3mulVec(ext.R_bc, p_c), ext.t_bc);
}

export function bodyToCamera(ext: Extrinsics, p_b: Vec3): Vec3 {
  return m3TmulVec(ext.R_bc, v3sub(p_b, ext.t_bc));
}

export function worldToCamera(R_wb: Mat3, p_wb: Vec3, ext: Extrinsics, p_w: Vec3): Vec3 {
  return bodyToCamera(ext, m3TmulVec(R_wb, v3sub(p_w, p_wb)));
}

export function cameraCenterWorld(R_wb: Mat3, p_wb: Vec3, ext: Extrinsics): Vec3 {
  return v3add(p_wb, m3mulVec(R_wb, ext.t_bc));
}

export function project(p_c: Vec3, K: Intrinsics): Pixel {
  if (p_c[2] <= 1e-6) return { u: 0, v: 0, valid: false, depth: p_c[2] };
  const u = K.fx * (p_c[0] / p_c[2]) + K.cx;
  const v = K.fy * (p_c[1] / p_c[2]) + K.cy;
  return {
    u,
    v,
    valid: Number.isFinite(u) && Number.isFinite(v),
    depth: p_c[2],
  };
}

export function projectWorld(
  R_wb: Mat3,
  p_wb: Vec3,
  ext: Extrinsics,
  K: Intrinsics,
  p_w: Vec3,
): Pixel {
  return project(worldToCamera(R_wb, p_wb, ext, p_w), K);
}

export function insideImage(K: Intrinsics, u: number, v: number, margin = 0): boolean {
  return u >= margin && v >= margin && u < K.width - margin && v < K.height - margin;
}

export function bearingCamera(pixel: { u: number; v: number }, K: Intrinsics): Vec3 {
  return v3normalize([(pixel.u - K.cx) / K.fx, (pixel.v - K.cy) / K.fy, 1]);
}

export function bearingWorld(
  R_wb: Mat3,
  ext: Extrinsics,
  K: Intrinsics,
  pixel: { u: number; v: number },
): Vec3 {
  return v3normalize(m3mulVec(R_wb, m3mulVec(ext.R_bc, bearingCamera(pixel, K))));
}

export interface TriangulationResult {
  ok: boolean;
  point: Vec3;
  /** Parallax angle in radians. */
  parallax: number;
  /** Midpoint ray miss distance, metres. */
  miss: number;
}

export function triangulateTwoView(
  R1: Mat3,
  p1: Vec3,
  px1: { u: number; v: number },
  R2: Mat3,
  p2: Vec3,
  px2: { u: number; v: number },
  ext: Extrinsics,
  K: Intrinsics,
): TriangulationResult {
  const c1 = cameraCenterWorld(R1, p1, ext);
  const c2 = cameraCenterWorld(R2, p2, ext);
  const d1 = bearingWorld(R1, ext, K, px1);
  const d2 = bearingWorld(R2, ext, K, px2);
  const parallax = Math.acos(Math.min(1, Math.max(-1, v3dot(d1, d2))));
  // Closest points between the two rays.
  const r = v3sub(c1, c2);
  const a = v3dot(d1, d1);
  const b = v3dot(d1, d2);
  const c = v3dot(d2, d2);
  const d = v3dot(d1, r);
  const e = v3dot(d2, r);
  const denom = a * c - b * b;
  if (Math.abs(denom) < 1e-9) {
    return { ok: false, point: c1, parallax, miss: Infinity };
  }
  const s1 = (b * e - c * d) / denom;
  const s2 = (a * e - b * d) / denom;
  if (s1 <= 0.05 || s2 <= 0.05) return { ok: false, point: c1, parallax, miss: Infinity };
  const q1 = v3add(c1, v3scale(d1, s1));
  const q2 = v3add(c2, v3scale(d2, s2));
  const miss = v3norm(v3sub(q1, q2));
  const point = v3scale(v3add(q1, q2), 0.5);
  const range = v3norm(v3sub(point, c1));
  return { ok: miss < Math.max(0.05, range * 0.02), point, parallax, miss };
}

export function reprojection(
  R_wb: Mat3,
  p_wb: Vec3,
  ext: Extrinsics,
  K: Intrinsics,
  p_w: Vec3,
): { u: number; v: number; depth: number; ok: boolean } {
  const pc = worldToCamera(R_wb, p_wb, ext, p_w);
  const px = project(pc, K);
  return { u: px.u, v: px.v, depth: px.depth, ok: px.valid && insideImage(K, px.u, px.v) && px.depth > 0 };
}
