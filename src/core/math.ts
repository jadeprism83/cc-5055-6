// Small fixed-size linear algebra used by the estimator, simulator and evaluator.
// Conventions:
//   Vec3       plain [x, y, z] in metres / radians / seconds
//   Mat3       Float64Array(9), row-major
//   SO(3)      right-handed, active rotation, Exp(w) maps body rates to rotation
//   Dynamic Mat row-major Float64Array, used for the 15x15 ESKF covariance

export type Vec3 = [number, number, number];
export type Mat3 = Float64Array;

export const EPS = 1e-9;

export function v3(x = 0, y = 0, z = 0): Vec3 {
  return [x, y, z];
}
export function v3add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}
export function v3sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
export function v3scale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}
export function v3dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
export function v3cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
export function v3norm(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}
export function v3normalize(a: Vec3): Vec3 {
  const n = v3norm(a);
  return n < EPS ? [0, 0, 0] : [a[0] / n, a[1] / n, a[2] / n];
}
export function v3dist(a: Vec3, b: Vec3): number {
  return v3norm(v3sub(a, b));
}
export function v3lerp(a: Vec3, b: Vec3, s: number): Vec3 {
  return [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s];
}

export function m3identity(): Mat3 {
  const m = new Float64Array(9);
  m[0] = m[4] = m[8] = 1;
  return m;
}
export function m3fromRows(r0: Vec3, r1: Vec3, r2: Vec3): Mat3 {
  return new Float64Array([r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], r2[0], r2[1], r2[2]]);
}
export function m3fromCols(c0: Vec3, c1: Vec3, c2: Vec3): Mat3 {
  return new Float64Array([c0[0], c1[0], c2[0], c0[1], c1[1], c2[1], c0[2], c1[2], c2[2]]);
}
export function m3clone(a: Mat3): Mat3 {
  return new Float64Array(a);
}
export function m3add(a: Mat3, b: Mat3): Mat3 {
  const m = new Float64Array(9);
  for (let i = 0; i < 9; i++) m[i] = a[i] + b[i];
  return m;
}
export function m3scale(a: Mat3, s: number): Mat3 {
  const m = new Float64Array(9);
  for (let i = 0; i < 9; i++) m[i] = a[i] * s;
  return m;
}
export function m3mul(a: Mat3, b: Mat3): Mat3 {
  const m = new Float64Array(9);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      m[r * 3 + c] =
        a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
    }
  }
  return m;
}
export function m3mulVec(a: Mat3, v: Vec3): Vec3 {
  return [
    a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
    a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
    a[6] * v[0] + a[7] * v[1] + a[8] * v[2],
  ];
}
export function m3TmulVec(a: Mat3, v: Vec3): Vec3 {
  return [
    a[0] * v[0] + a[3] * v[1] + a[6] * v[2],
    a[1] * v[0] + a[4] * v[1] + a[7] * v[2],
    a[2] * v[0] + a[5] * v[1] + a[8] * v[2],
  ];
}
export function m3transpose(a: Mat3): Mat3 {
  return new Float64Array([a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]);
}
export function m3trace(a: Mat3): number {
  return a[0] + a[4] + a[8];
}
export function m3det(a: Mat3): number {
  return (
    a[0] * (a[4] * a[8] - a[5] * a[7]) -
    a[1] * (a[3] * a[8] - a[5] * a[6]) +
    a[2] * (a[3] * a[7] - a[4] * a[6])
  );
}
export function m3orthonormalize(a: Mat3): Mat3 {
  // Gram-Schmidt, robust enough for a 15-state filter that re-normalises each step.
  let x: Vec3 = [a[0], a[3], a[6]];
  let y: Vec3 = [a[1], a[4], a[7]];
  x = v3normalize(x);
  y = v3normalize(v3sub(y, v3scale(x, v3dot(x, y))));
  const z = v3cross(x, y);
  if (m3det(a) < 0) return m3fromCols(x, v3scale(y, -1), v3scale(z, -1));
  return m3fromCols(x, y, z);
}
export function skew(w: Vec3): Mat3 {
  return new Float64Array([0, -w[2], w[1], w[2], 0, -w[0], -w[1], w[0], 0]);
}
export function so3Exp(w: Vec3, thetaOverride?: number): Mat3 {
  const theta = thetaOverride ?? v3norm(w);
  if (theta < 1e-8) {
    const W = skew(w);
    const W2 = m3mul(W, W);
    return m3add(m3add(m3identity(), W), m3scale(W2, 0.5));
  }
  const axis = v3scale(w, 1 / theta);
  const W = skew(axis);
  const W2 = m3mul(W, W);
  const s = Math.sin(theta);
  const c = 1 - Math.cos(theta);
  return m3add(m3add(m3identity(), m3scale(W, s)), m3scale(W2, c));
}
export function so3Log(R: Mat3): Vec3 {
  const tr = m3trace(R);
  const cosTheta = Math.min(1, Math.max(-1, (tr - 1) / 2));
  const theta = Math.acos(cosTheta);
  const vee: Vec3 = [R[7] - R[5], R[2] - R[6], R[3] - R[1]];
  if (theta < 1e-6) return v3scale(vee, 0.5);
  if (theta < Math.PI - 1e-3) return v3scale(vee, theta / (2 * Math.sin(theta)));
  // Near pi use the quaternion axis, where w ~= 0 and the vector part dominates.
  const qw = 0.5 * Math.sqrt(Math.max(0, 1 + tr));
  let qx = (R[7] - R[5]) / (4 * qw);
  let qy = (R[2] - R[6]) / (4 * qw);
  let qz = (R[3] - R[1]) / (4 * qw);
  if (qw < 1e-6) {
    qx = Math.sqrt(Math.max(0, (1 + R[0]) / 2));
    qy = Math.sqrt(Math.max(0, (1 + R[4]) / 2));
    qz = Math.sqrt(Math.max(0, (1 + R[8]) / 2));
    if (R[1] + R[3] < 0) qy = -qy;
    if (R[2] + R[6] < 0) qz = -qz;
  }
  const qv = v3normalize([qx, qy, qz]);
  return v3scale(qv, theta);
}
export function rotX(a: number): Mat3 {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return new Float64Array([1, 0, 0, 0, c, -s, 0, s, c]);
}
export function rotY(a: number): Mat3 {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return new Float64Array([c, 0, s, 0, 1, 0, -s, 0, c]);
}
export function rotZ(a: number): Mat3 {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return new Float64Array([c, -s, 0, s, c, 0, 0, 0, 1]);
}
export function eulerZYX(yaw: number, pitch: number, roll: number): Mat3 {
  return m3mul(m3mul(rotZ(yaw), rotY(pitch)), rotX(roll));
}
export function rotToEulerZYX(R: Mat3): Vec3 {
  const pitch = Math.asin(Math.min(1, Math.max(-1, -R[6])));
  let yaw: number;
  let roll: number;
  if (Math.abs(R[6]) < 0.999999) {
    yaw = Math.atan2(R[3], R[0]);
    roll = Math.atan2(R[7], R[8]);
  } else {
    yaw = Math.atan2(-R[1], R[4]);
    roll = 0;
  }
  return [yaw, pitch, roll];
}
export function rotationBetween(a: Vec3, b: Vec3): Mat3 {
  const u = v3normalize(a);
  const w = v3normalize(b);
  const c = Math.min(1, Math.max(-1, v3dot(u, w)));
  if (c > 1 - 1e-12) return m3identity();
  if (c < -1 + 1e-12) {
    // 180 degrees: choose any orthogonal axis.
    let axis = v3cross(u, [1, 0, 0]);
    if (v3norm(axis) < 1e-6) axis = v3cross(u, [0, 1, 0]);
    axis = v3normalize(axis);
    return so3Exp(v3scale(axis, Math.PI));
  }
  const axis = v3cross(u, w);
  return so3Exp(v3scale(axis, Math.acos(c) / v3norm(axis)));
}
export function slerp(R0: Mat3, R1: Mat3, s: number): Mat3 {
  const d = so3Log(m3mul(m3transpose(R0), R1));
  return m3mul(R0, so3Exp(v3scale(d, s)));
}

// ---------------------------------------------------------------------------
// Dynamic row-major matrices for the 15x15 ESKF.

export interface Mat {
  r: number;
  c: number;
  d: Float64Array;
}
export function mat(r: number, c: number, fill = 0): Mat {
  return { r, c, d: new Float64Array(r * c).fill(fill) };
}
export function matEye(n: number): Mat {
  const m = mat(n, n);
  for (let i = 0; i < n; i++) m.d[i * n + i] = 1;
  return m;
}
export function matGet(m: Mat, r: number, c: number): number {
  return m.d[r * m.c + c];
}
export function matSet(m: Mat, r: number, c: number, v: number): void {
  m.d[r * m.c + c] = v;
}
export function matClone(m: Mat): Mat {
  return { r: m.r, c: m.c, d: new Float64Array(m.d) };
}
export function matZero(m: Mat): void {
  m.d.fill(0);
}
export function matMul(a: Mat, b: Mat): Mat {
  const out = mat(a.r, b.c);
  for (let i = 0; i < a.r; i++) {
    for (let k = 0; k < a.c; k++) {
      const aik = a.d[i * a.c + k];
      if (aik === 0) continue;
      for (let j = 0; j < b.c; j++) out.d[i * b.c + j] += aik * b.d[k * b.c + j];
    }
  }
  return out;
}
export function matAdd(a: Mat, b: Mat): Mat {
  const out = mat(a.r, a.c);
  for (let i = 0; i < a.d.length; i++) out.d[i] = a.d[i] + b.d[i];
  return out;
}
export function matSub(a: Mat, b: Mat): Mat {
  const out = mat(a.r, a.c);
  for (let i = 0; i < a.d.length; i++) out.d[i] = a.d[i] - b.d[i];
  return out;
}
export function matScale(a: Mat, s: number): Mat {
  const out = mat(a.r, a.c);
  for (let i = 0; i < a.d.length; i++) out.d[i] = a.d[i] * s;
  return out;
}
export function matTranspose(a: Mat): Mat {
  const out = mat(a.c, a.r);
  for (let i = 0; i < a.r; i++) for (let j = 0; j < a.c; j++) out.d[j * a.r + i] = a.d[i * a.c + j];
  return out;
}
export function matSymmetrize(m: Mat): void {
  for (let i = 0; i < m.r; i++) {
    for (let j = i + 1; j < m.c; j++) {
      const v = 0.5 * (m.d[i * m.c + j] + m.d[j * m.c + i]);
      m.d[i * m.c + j] = v;
      m.d[j * m.c + i] = v;
    }
  }
}
export function matInverse(a: Mat): Mat {
  const n = a.r;
  const aug = mat(n, 2 * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) aug.d[i * 2 * n + j] = a.d[i * n + j];
    aug.d[i * 2 * n + n + i] = 1;
  }
  for (let col = 0; col < n; col++) {
    let pivot = col;
    let best = Math.abs(aug.d[col * 2 * n + col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(aug.d[r * 2 * n + col]);
      if (v > best) {
        best = v;
        pivot = r;
      }
    }
    if (best < 1e-14) throw new Error('singular matrix');
    if (pivot !== col) {
      for (let j = 0; j < 2 * n; j++) {
        const t = aug.d[col * 2 * n + j];
        aug.d[col * 2 * n + j] = aug.d[pivot * 2 * n + j];
        aug.d[pivot * 2 * n + j] = t;
      }
    }
    const p = aug.d[col * 2 * n + col];
    for (let j = 0; j < 2 * n; j++) aug.d[col * 2 * n + j] /= p;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = aug.d[r * 2 * n + col];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) aug.d[r * 2 * n + j] -= f * aug.d[col * 2 * n + j];
    }
  }
  const out = mat(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out.d[i * n + j] = aug.d[i * 2 * n + n + j];
  return out;
}

// ---------------------------------------------------------------------------
// Deterministic RNG and statistics.

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export class Gaussian {
  private spare: number | null = null;
  constructor(private rand: () => number) {}
  next(): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return v;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = this.rand();
    while (v === 0) v = this.rand();
    const mag = Math.sqrt(-2 * Math.log(u));
    this.spare = mag * Math.sin(2 * Math.PI * v);
    return mag * Math.cos(2 * Math.PI * v);
  }
  vec3(sigma: number | Vec3): Vec3 {
    if (typeof sigma === 'number') return [this.next() * sigma, this.next() * sigma, this.next() * sigma];
    return [this.next() * sigma[0], this.next() * sigma[1], this.next() * sigma[2]];
  }
}
export function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}
export function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const a = [...xs].sort((x, y) => x - y);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : 0.5 * (a[m - 1] + a[m]);
}
export function percentile(xs: number[], p: number): number {
  if (xs.length === 0) return 0;
  const a = [...xs].sort((x, y) => x - y);
  const idx = Math.min(a.length - 1, Math.max(0, Math.round((a.length - 1) * p)));
  return a[idx];
}
export function rms(xs: number[]): number {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x * x;
  return Math.sqrt(s / xs.length);
}
export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}
export function smoothstep(x: number): number {
  const t = clamp(x, 0, 1);
  return t * t * (3 - 2 * t);
}
