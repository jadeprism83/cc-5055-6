import { Gaussian, Vec3, v3add, v3cross, v3dist, v3dot, v3normalize, v3scale, v3sub } from '../core/math';
import { Intrinsics } from '../core/camera';

// Static room texture. Coordinates follow the body convention used across the
// app: x forward, y left, z up. The camera starts near the origin looking +x.
// Nothing here depends on an estimated trajectory.

export interface ScenePoint {
  id: number;
  p_w: Vec3;
  rgb: [number, number, number];
  radius: number;
}

export interface SceneQuad {
  p_w: Vec3;
  uAxis: Vec3;
  vAxis: Vec3;
  texture: Float32Array;
  texW: number;
  texH: number;
  shade: number;
}

export interface Scene {
  points: ScenePoint[];
  quads: SceneQuad[];
  bounds: { min: Vec3; max: Vec3 };
}

function mulberry32Local(a: number): () => number {
  let s = a >>> 0;
  return () => {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Value noise plus hard-edged glyphs. Corners need crisp structure, not a
// smooth showcase gradient.
function makeTexture(rand: () => number, w: number, h: number): Float32Array {
  const base = new Float32Array(w * h);
  const octaves = [6, 12, 24];
  for (let o = 0; o < octaves.length; o++) {
    const g = octaves[o];
    const lat = new Float32Array((g + 1) * (g + 1));
    for (let i = 0; i < lat.length; i++) lat[i] = rand();
    const amp = 0.34 / (o + 1);
    for (let y = 0; y < h; y++) {
      const fy = (y / (h - 1)) * g;
      const y0 = Math.floor(fy);
      const y1 = Math.min(g, y0 + 1);
      const ty = fy - y0;
      for (let x = 0; x < w; x++) {
        const fx = (x / (w - 1)) * g;
        const x0 = Math.floor(fx);
        const x1 = Math.min(g, x0 + 1);
        const tx = fx - x0;
        const a = lat[y0 * (g + 1) + x0];
        const b = lat[y0 * (g + 1) + x1];
        const c = lat[y1 * (g + 1) + x0];
        const d = lat[y1 * (g + 1) + x1];
        const top = a + (b - a) * tx;
        const bot = c + (d - c) * tx;
        base[y * w + x] += amp * (top + (bot - top) * ty);
      }
    }
  }
  for (let i = 0; i < base.length; i++) base[i] = Math.min(1, Math.max(0, 0.25 + 0.7 * base[i]));
  // Hard-edged tiles, bars and rings. These generate the corner response.
  for (let k = 0; k < 70; k++) {
    const cx = rand() * w;
    const cy = rand() * h;
    const rw = 3 + rand() * 18;
    const rh = 3 + rand() * 18;
    const kind = rand();
    const val = rand() < 0.5 ? 0.06 + rand() * 0.15 : 0.8 + rand() * 0.2;
    const angle = (rand() - 0.5) * 0.9;
    const ca = Math.cos(angle);
    const sa = Math.sin(angle);
    for (let y = Math.max(0, Math.floor(cy - rw - rh)); y < Math.min(h, Math.ceil(cy + rw + rh)); y++) {
      for (let x = Math.max(0, Math.floor(cx - rw - rh)); x < Math.min(w, Math.ceil(cx + rw + rh)); x++) {
        const dx = x - cx;
        const dy = y - cy;
        const lx = ca * dx + sa * dy;
        const ly = -sa * dx + ca * dy;
        let inside: boolean;
        if (kind < 0.5) inside = Math.abs(lx) < rw / 2 && Math.abs(ly) < rh / 2;
        else if (kind < 0.8) inside = Math.abs(lx) < 2 && Math.abs(ly) < rh;
        else {
          const rr = Math.hypot(lx, ly);
          inside = rr > Math.min(rw, rh) - 3 && rr < Math.min(rw, rh);
        }
        if (inside) base[y * w + x] = val;
      }
    }
  }
  // Fine grain keeps flow from locking onto a flat region.
  for (let i = 0; i < base.length; i++) base[i] = Math.min(1, Math.max(0, base[i] + (rand() - 0.5) * 0.08));
  return base;
}

export function generateScene(seed = 11): Scene {
  const rand = mulberry32Local(seed * 977 + 13);
  const quads: SceneQuad[] = [];
  const specs: Array<{ c: Vec3; n: Vec3; halfW: number; halfH: number; texW: number }> = [
    // Far wall at x=6, facing back toward the camera.
    { c: [6.0, 0.2, 1.2], n: [-1, 0, 0], halfW: 5.2, halfH: 3.0, texW: 120 },
    // Left wall at y=4.2.
    { c: [3.0, 4.2, 1.2], n: [0, -1, 0], halfW: 4.2, halfH: 3.0, texW: 120 },
    // Right wall at y=-4.2.
    { c: [3.0, -4.2, 1.2], n: [0, 1, 0], halfW: 4.2, halfH: 3.0, texW: 120 },
    // Floor.
    { c: [3.0, 0.2, -1.35], n: [0, 0, 1], halfW: 4.8, halfH: 4.6, texW: 120 },
    // A free-standing board mid-room, at an angle.
    { c: [3.6, 1.05, 1.1], n: [-0.72, 0.69, 0], halfW: 0.7, halfH: 0.9, texW: 64 },
  ];
  for (const sp of specs) {
    const n = v3normalize(sp.n);
    let u = v3normalize(v3cross(n, Math.abs(n[2]) > 0.8 ? [1, 0, 0] : [0, 0, 1]));
    let v = v3normalize(v3cross(n, u));
    if (v[2] < 0) v = v3scale(v, -1);
    u = v3scale(u, sp.halfW);
    v = v3scale(v, sp.halfH);
    const texW = sp.texW;
    const texH = Math.round(texW * (sp.halfH / sp.halfW));
    quads.push({ p_w: sp.c, uAxis: u, vAxis: v, texture: makeTexture(rand, texW, texH), texW, texH, shade: 0.8 + 0.2 * rand() });
  }
  // Physical corner-like objects (small boards) give stable, unambiguous flow.
  for (let k = 0; k < 32; k++) {
    const x = 1.7 + rand() * 3.9;
    const y = -3.3 + rand() * 6.6;
    const z = -1.0 + rand() * 3.6;
    if (Math.hypot(y - 0.0, z - 1.2) < 0.45) continue;
    const n: Vec3 = k % 2 === 0 ? [-0.7, 0.7, 0] : [-1, 0.1, 0.15];
    const nn = v3normalize(n);
    let u = v3normalize(v3cross(nn, [0, 0, 1]));
    let v = v3normalize(v3cross(nn, u));
    const s = 0.10 + rand() * 0.18;
    quads.push({
      p_w: [x, y, z],
      uAxis: v3scale(u, s),
      vAxis: v3scale(v, s * (0.7 + rand() * 0.6)),
      texture: makeTexture(rand, 20, 16),
      texW: 20,
      texH: 16,
      shade: 0.85 + 0.15 * rand(),
    });
  }
  const points: ScenePoint[] = [];
  for (let i = 0; i < 170; i++) {
    const p: Vec3 = [1.5 + rand() * 4.2, -3.4 + rand() * 6.8, -1.05 + rand() * 3.7];
    points.push({
      id: i,
      p_w: p,
      rgb: [40 + Math.floor(rand() * 210), 40 + Math.floor(rand() * 210), 40 + Math.floor(rand() * 210)],
      radius: 1.2 + rand() * 2.2,
    });
  }
  return { points, quads, bounds: { min: [-0.5, -4.5, -0.5], max: [6.5, 4.5, 3.5] } };
}

export interface CameraRenderPose {
  R_wb: Float64Array;
  p_wb: Vec3;
  R_bc: Float64Array;
  t_bc: Vec3;
}

function camCenter(pose: CameraRenderPose): Vec3 {
  const R = pose.R_wb;
  const t = pose.t_bc;
  return [
    pose.p_wb[0] + R[0] * t[0] + R[1] * t[1] + R[2] * t[2],
    pose.p_wb[1] + R[3] * t[0] + R[4] * t[1] + R[5] * t[2],
    pose.p_wb[2] + R[6] * t[0] + R[7] * t[1] + R[8] * t[2],
  ];
}

function worldToCam(pose: CameraRenderPose, p: Vec3): Vec3 {
  const R = pose.R_wb;
  const dx = p[0] - pose.p_wb[0];
  const dy = p[1] - pose.p_wb[1];
  const dz = p[2] - pose.p_wb[2];
  const bx = R[0] * dx + R[3] * dy + R[6] * dz;
  const by = R[1] * dx + R[4] * dy + R[7] * dz;
  const bz = R[2] * dx + R[5] * dy + R[8] * dz;
  const db: Vec3 = [bx - pose.t_bc[0], by - pose.t_bc[1], bz - pose.t_bc[2]];
  const Rb = pose.R_bc;
  return [
    Rb[0] * db[0] + Rb[3] * db[1] + Rb[6] * db[2],
    Rb[1] * db[0] + Rb[4] * db[1] + Rb[7] * db[2],
    Rb[2] * db[0] + Rb[5] * db[1] + Rb[8] * db[2],
  ];
}

export interface RenderResult {
  gray: Float32Array;
  rgb: Uint8ClampedArray;
  visibleCorners: number;
}

export interface RenderOptions {
  exposure: number;
  pixelNoise: number;
  gaussian: Gaussian;
  outGray?: Float32Array;
  outRgb?: Uint8ClampedArray;
  blurRadius?: number;
}

export function renderScene(scene: Scene, pose: CameraRenderPose, K: Intrinsics, opts: RenderOptions): RenderResult {
  const w = K.width;
  const h = K.height;
  const gray = opts.outGray ?? new Float32Array(w * h);
  const rgb = opts.outRgb ?? new Uint8ClampedArray(w * h * 3);
  const zbuf = new Float32Array(w * h);
  gray.fill(0);
  rgb.fill(0);
  zbuf.fill(Infinity);
  const cam = camCenter(pose);

  for (const q of scene.quads) {
    const n = v3normalize(v3cross(q.uAxis, q.vAxis));
    if (v3dot(n, v3sub(cam, q.p_w)) <= 0) continue;
    const corners = [
      v3add(q.p_w, v3add(v3scale(q.uAxis, -1), v3scale(q.vAxis, -1))),
      v3add(q.p_w, v3add(q.uAxis, v3scale(q.vAxis, -1))),
      v3add(q.p_w, v3add(q.uAxis, q.vAxis)),
      v3add(q.p_w, v3add(v3scale(q.uAxis, -1), q.vAxis)),
    ].map((p) => worldToCam(pose, p));
    if (corners.some((c) => c[2] < 0.12)) continue;
    const proj = corners.map((c) => ({
      x: (K.fx * c[0]) / c[2] + K.cx,
      y: (K.fy * c[1]) / c[2] + K.cy,
      iz: 1 / c[2],
    }));
    const minX = Math.max(0, Math.floor(Math.min(...proj.map((p) => p.x))));
    const maxX = Math.min(w - 1, Math.ceil(Math.max(...proj.map((p) => p.x))));
    const minY = Math.max(0, Math.floor(Math.min(...proj.map((p) => p.y))));
    const maxY = Math.min(h - 1, Math.ceil(Math.max(...proj.map((p) => p.y))));
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const bary = quadBary(proj, x + 0.5, y + 0.5);
        if (!bary) continue;
        const { s, t, iz } = bary;
        const tx = Math.min(q.texW - 1, Math.max(0, Math.floor(s * (q.texW - 1))));
        const ty = Math.min(q.texH - 1, Math.max(0, Math.floor(t * (q.texH - 1))));
        const val = q.texture[ty * q.texW + tx] * q.shade * opts.exposure;
        const fog = Math.max(0.35, Math.min(1, 1.25 - 0.07 * (1 / Math.max(iz, 1e-3))));
        const c = Math.max(0, Math.min(1, val * fog));
        const g = Math.round(c * 255);
        const idx = y * w + x;
        const zDepth = 1 / Math.max(iz, 1e-6);
        if (zDepth >= zbuf[idx]) continue;
        zbuf[idx] = zDepth;
        gray[idx] = g;
        rgb[idx * 3] = Math.min(255, Math.round(g * (0.9 + 0.1 * s)));
        rgb[idx * 3 + 1] = Math.min(255, Math.round(g * (0.92 + 0.06 * t)));
        rgb[idx * 3 + 2] = Math.min(255, Math.round(g * 0.86));
      }
    }
  }
  let visible = 0;
  for (const pt of scene.points) {
    const c = worldToCam(pose, pt.p_w);
    if (c[2] < 0.35) continue;
    const u = (K.fx * c[0]) / c[2] + K.cx;
    const v = (K.fy * c[1]) / c[2] + K.cy;
    // Fixed pixel size in the rendered image, scaled by the render resolution
    // factor. Depth-dependent size makes the disk breathe, which biases flow.
    const r = 2.0 * (K.width / 320);
    const x0 = Math.max(0, Math.floor(u - r * 2));
    const x1 = Math.min(w - 1, Math.ceil(u + r * 2));
    const y0 = Math.max(0, Math.floor(v - r * 2));
    const y1 = Math.min(h - 1, Math.ceil(v + r * 2));
    if (x1 < 0 || y1 < 0 || x0 >= w || y0 >= h) continue;
    visible++;
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const d = Math.hypot(x + 0.5 - u, y + 0.5 - v);
        if (d > r * 2) continue;
        const idx = y * w + x;
        if (c[2] >= zbuf[idx]) continue;
        if (d <= r) {
          gray[idx] = Math.min(255, (255 - d * 15) * opts.exposure);
          rgb[idx * 3] = Math.min(255, (pt.rgb[0] + 70) * opts.exposure);
          rgb[idx * 3 + 1] = Math.min(255, (pt.rgb[1] + 70) * opts.exposure);
          rgb[idx * 3 + 2] = Math.min(255, (pt.rgb[2] + 70) * opts.exposure);
        } else if (d <= r * 1.35) {
          zbuf[idx] = c[2];
          gray[idx] = Math.min(gray[idx], 28 * opts.exposure);
        }
      }
    }
  }
  if (opts.blurRadius && opts.blurRadius >= 1) boxBlurGray(gray, w, h, Math.min(12, Math.round(opts.blurRadius)));
  if (opts.pixelNoise > 0) {
    for (let i = 0; i < gray.length; i++) {
      gray[i] = Math.min(255, Math.max(0, gray[i] + opts.gaussian.next() * opts.pixelNoise));
    }
  }
  // Keep display RGB consistent with final tracking-gray (exposure/noise/blur).
  const tintR = 0.92, tintG = 0.96, tintB = 0.88;
  for (let i = 0; i < gray.length; i++) {
    const g = gray[i];
    rgb[i * 3] = Math.min(255, Math.round(g * tintR));
    rgb[i * 3 + 1] = Math.min(255, Math.round(g * tintG));
    rgb[i * 3 + 2] = Math.min(255, Math.round(g * tintB));
  }
  return { gray, rgb, visibleCorners: visible };
}

function quadBary(proj: Array<{ x: number; y: number; iz: number }>, x: number, y: number): { s: number; t: number; iz: number } | null {
  const tris = [
    [0, 1, 2],
    [0, 2, 3],
  ];
  const st = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ];
  for (const [ia, ib, ic] of tris) {
    const a = proj[ia], b = proj[ib], c = proj[ic];
    const d = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    if (Math.abs(d) < 1e-9) continue;
    const l1 = ((b.y - c.y) * (x - c.x) + (c.x - b.x) * (y - c.y)) / d;
    const l2 = ((c.y - a.y) * (x - c.x) + (a.x - c.x) * (y - c.y)) / d;
    const l3 = 1 - l1 - l2;
    if (l1 >= -1e-6 && l2 >= -1e-6 && l3 >= -1e-6) {
      // Perspective-correct interpolation: weight by 1/z. Screen-space affine
      // mapping makes the texture slide as the camera translates, which would
      // corrupt optical flow in a way no estimator could undo.
      const w1 = l1 * a.iz, w2 = l2 * b.iz, w3 = l3 * c.iz;
      const wsum = w1 + w2 + w3;
      const s = (st[ia][0] * w1 + st[ib][0] * w2 + st[ic][0] * w3) / wsum;
      const t = (st[ia][1] * w1 + st[ib][1] * w2 + st[ic][1] * w3) / wsum;
      const iz = wsum;
      return { s, t, iz };
    }
  }
  return null;
}

function boxBlurGray(gray: Float32Array, w: number, h: number, r: number): void {
  const tmp = new Float32Array(gray.length);
  const n = 2 * r + 1;
  for (let y = 0; y < h; y++) {
    let sum = 0;
    for (let x = -r; x <= r; x++) sum += gray[y * w + Math.min(w - 1, Math.max(0, x))];
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = sum / n;
      sum += gray[y * w + Math.min(w - 1, x + r + 1)] - gray[y * w + Math.max(0, x - r)];
    }
  }
  for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let y = -r; y <= r; y++) sum += tmp[Math.min(h - 1, Math.max(0, y)) * w + x];
    for (let y = 0; y < h; y++) {
      gray[y * w + x] = sum / n;
      sum += tmp[Math.min(h - 1, y + r + 1) * w + x] - tmp[Math.max(0, y - r) * w + x];
    }
  }
}

export { camCenter };
