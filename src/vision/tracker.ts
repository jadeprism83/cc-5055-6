import { Intrinsics } from '../core/camera';

// Feature detection and KLT-style tracking on grayscale images.
// Tracking is purely geometric: nothing consults the simulator's pose.

export interface Corner {
  u: number;
  v: number;
  score: number;
}

export interface TrackPoint {
  /** Stable per-track id for the lifetime of the run. */
  id: number;
  u: number;
  v: number;
  /** Image at which the track was born (keyframe index). */
  bornAt: number;
  age: number;
  /** Previous pixel for optical flow. */
  prevU: number;
  prevV: number;
  /** Visual match confidence, low when flow residuals are poor. */
  confidence: number;
  /** Set when flow suggests the camera moved so far the match is rejected. */
  stale: boolean;
}

export interface FrameFeatures {
  gray: Float32Array;
  width: number;
  height: number;
  keypoints: Corner[];
  descriptors: Uint8Array;
  /** Number of descriptor bits set, cheap match bucket. */
  descriptorSum: Uint16Array;
  /** Half-resolution image and dimensions, built lazily for pyramid flow. */
  half?: { gray: Float32Array; width: number; height: number };
}

function halfResolution(gray: Float32Array, w: number, h: number): { gray: Float32Array; width: number; height: number } {
  const hw = Math.floor(w / 2);
  const hh = Math.floor(h / 2);
  const out = new Float32Array(hw * hh);
  for (let y = 0; y < hh; y++) {
    for (let x = 0; x < hw; x++) {
      const i = (2 * y) * w + 2 * x;
      out[y * hw + x] = 0.25 * (gray[i] + gray[i + 1] + gray[i + w] + gray[i + w + 1]);
    }
  }
  return { gray: out, width: hw, height: hh };
}

function ensureHalf(f: FrameFeatures): { gray: Float32Array; width: number; height: number } {
  if (!f.half) f.half = halfResolution(f.gray, f.width, f.height);
  return f.half;
}

/**
 * Coarse-to-fine flow: track on half resolution (captures up to ~+-20 px at full
 * resolution), then refine on the full pyramid level for subpixel accuracy.
 */
export function trackPointPyramid(
  prev: FrameFeatures,
  next: FrameFeatures,
  u: number,
  v: number,
  guessDx: number,
  guessDy: number,
): FlowResult {
  const hp = ensureHalf(prev);
  const hn = ensureHalf(next);
  const coarse = trackPoint(
    hp.gray, hp.width, hp.height,
    hn.gray, hn.width, hn.height,
    u / 2, v / 2,
    guessDx / 2, guessDy / 2,
    5,
  );
  const gx = coarse.ok ? 2 * coarse.u - u : guessDx;
  const gy = coarse.ok ? 2 * coarse.v - v : guessDy;
  return trackPoint(prev.gray, prev.width, prev.height, next.gray, next.width, next.height, u, v, gx, gy, 2);
}

export interface TrackResult {
  tracks: TrackPoint[];
  /** Points matched to a keyframe by descriptor, not flow. */
  relocalised: number;
  /** Tracks dropped for failing the forward-backward test. */
  dropped: number;
  /** Median flow residual in pixels. */
  flowResidual: number;
}

// ---------------------------------------------------------------------------
// Harris-style corner detector with a 5x5 structure tensor, non-max suppression.

const GRAD_BORDER = 3;

export function gradients(gray: Float32Array, w: number, h: number): { gx: Float32Array; gy: Float32Array } {
  const gx = new Float32Array(w * h);
  const gy = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      gx[i] =
        -gray[i - w - 1] + gray[i - w + 1] -
        2 * gray[i - 1] + 2 * gray[i + 1] -
        gray[i + w - 1] + gray[i + w + 1];
      gy[i] =
        -gray[i - w - 1] - 2 * gray[i - w] - gray[i - w + 1] +
        gray[i + w - 1] + 2 * gray[i + w] + gray[i + w + 1];
    }
  }
  return { gx, gy };
}

export function detectCorners(gray: Float32Array, w: number, h: number, maxCorners = 220): Corner[] {
  const { gx, gy } = gradients(gray, w, h);
  const r = 2;
  const score = new Float32Array(w * h);
  for (let y = GRAD_BORDER; y < h - GRAD_BORDER; y++) {
    for (let x = GRAD_BORDER; x < w - GRAD_BORDER; x++) {
      let sxx = 0, sxy = 0, syy = 0;
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          const i = (y + dy) * w + (x + dx);
          sxx += gx[i] * gx[i];
          sxy += gx[i] * gy[i];
          syy += gy[i] * gy[i];
        }
      }
      const tr = sxx + syy;
      const det = sxx * syy - sxy * sxy;
      const minEig = tr / 2 - Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
      score[y * w + x] = minEig;
    }
  }
  // Local maxima with a non-max radius large enough for the patch size.
  const radius = 4;
  const corners: Corner[] = [];
  for (let y = GRAD_BORDER + radius; y < h - GRAD_BORDER - radius; y++) {
    for (let x = GRAD_BORDER + radius; x < w - GRAD_BORDER - radius; x++) {
      const s = score[y * w + x];
      if (s < 40) continue;
      let best = true;
      for (let dy = -radius; dy <= radius && best; dy += 1) {
        for (let dx = -radius; dx <= radius; dx++) {
          if (dx === 0 && dy === 0) continue;
          if (score[(y + dy) * w + (x + dx)] > s) {
            best = false;
            break;
          }
        }
      }
      if (best) corners.push({ u: x, v: y, score: s });
    }
  }
  corners.sort((a, b) => b.score - a.score);
  return corners.slice(0, maxCorners);
}

// ---------------------------------------------------------------------------
// BRIEF-like binary descriptor from a smoothed patch, for relocalisation when
// optical flow cannot continue (blur, large jump, track starvation).

const PATTERN_RADIUS = 5;
const DESCRIPTOR_BITS = 128;
const pattern = buildPattern();

function buildPattern(): Int32Array {
  // Deterministic pattern offsets in [-R, R], pairs chosen to be scan-spread.
  let s = 0x9e3779b9;
  const rnd = () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = new Int32Array(DESCRIPTOR_BITS * 4);
  for (let i = 0; i < DESCRIPTOR_BITS; i++) {
    const a = Math.floor(rnd() * (2 * PATTERN_RADIUS + 1)) - PATTERN_RADIUS;
    const b = Math.floor(rnd() * (2 * PATTERN_RADIUS + 1)) - PATTERN_RADIUS;
    const c = Math.floor(rnd() * (2 * PATTERN_RADIUS + 1)) - PATTERN_RADIUS;
    const d = Math.floor(rnd() * (2 * PATTERN_RADIUS + 1)) - PATTERN_RADIUS;
    out[i * 4] = a;
    out[i * 4 + 1] = b;
    out[i * 4 + 2] = c;
    out[i * 4 + 3] = d;
  }
  return out;
}

export function computeDescriptors(gray: Float32Array, w: number, h: number, corners: Corner[]): { descriptors: Uint8Array; sums: Uint16Array } {
  const descriptors = new Uint8Array(corners.length * DESCRIPTOR_BITS);
  const sums = new Uint16Array(corners.length);
  for (let k = 0; k < corners.length; k++) {
    const kp = corners[k];
    for (let b = 0; b < DESCRIPTOR_BITS / 8; b++) {
      let byteVal = 0;
      for (let j = 0; j < 8; j++) {
        const off = (b * 8 + j) * 4;
        const x1 = Math.min(w - 1, Math.max(0, Math.round(kp.u + pattern[off])));
        const y1 = Math.min(h - 1, Math.max(0, Math.round(kp.v + pattern[off + 1])));
        const x2 = Math.min(w - 1, Math.max(0, Math.round(kp.u + pattern[off + 2])));
        const y2 = Math.min(h - 1, Math.max(0, Math.round(kp.v + pattern[off + 3])));
        if (gray[y1 * w + x1] > gray[y2 * w + x2] + 2) byteVal |= 1 << j;
      }
      descriptors[k * DESCRIPTOR_BITS + b] = byteVal;
    }
    let pop = 0;
    for (let b = 0; b < DESCRIPTOR_BITS / 8; b++) pop += popcount8(descriptors[k * DESCRIPTOR_BITS + b]);
    sums[k] = pop;
  }
  return { descriptors, sums };
}

function popcount8(x: number): number {
  x = x - ((x >> 1) & 0x55);
  x = (x & 0x33) + ((x >> 2) & 0x33);
  return (x + (x >> 4)) & 0x0f;
}

export function hamming(a: Uint8Array, ao: number, b: Uint8Array, bo: number): number {
  let d = 0;
  for (let i = 0; i < DESCRIPTOR_BITS / 8; i++) d += popcount8(a[ao + i] ^ b[bo + i]);
  return d;
}

export function describe(gray: Float32Array, w: number, h: number): FrameFeatures {
  const corners = detectCorners(gray, w, h);
  const { descriptors, sums } = computeDescriptors(gray, w, h, corners);
  return { gray, width: w, height: h, keypoints: corners, descriptors, descriptorSum: sums };
}

// ---------------------------------------------------------------------------
// Optical flow with patch normalisation and a forward-backward consistency test.

const PATCH_R = 4;
const SEARCH_R = 9;

export interface FlowResult {
  u: number;
  v: number;
  ok: boolean;
  ncc: number;
  error: number;
}

function bilinear(img: Float32Array, w: number, h: number, x: number, y: number): number | null {
  if (x < 0 || y < 0 || x >= w - 1 || y >= h - 1) return null;
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const tx = x - x0, ty = y - y0;
  const i = y0 * w + x0;
  return (
    img[i] * (1 - tx) * (1 - ty) +
    img[i + 1] * tx * (1 - ty) +
    img[i + w] * (1 - tx) * ty +
    img[i + w + 1] * tx * ty
  );
}

function patchNcc(
  a: Float32Array, aw: number, ah: number,
  b: Float32Array, bw: number, bh: number,
  ax: number, ay: number, bx: number, by: number,
): number {
  let ma = 0, mb = 0, n = 0;
  const vaBuf = new Float64Array((2 * PATCH_R + 1) ** 2);
  const vbBuf = new Float64Array((2 * PATCH_R + 1) ** 2);
  for (let dy = -PATCH_R; dy <= PATCH_R; dy++) {
    for (let dx = -PATCH_R; dx <= PATCH_R; dx++) {
      const va = bilinear(a, aw, ah, ax + dx, ay + dy);
      const vb = bilinear(b, bw, bh, bx + dx, by + dy);
      if (va === null || vb === null) continue;
      vaBuf[n] = va;
      vbBuf[n] = vb;
      ma += va;
      mb += vb;
      n++;
    }
  }
  if (n < 30) return -1;
  ma /= n;
  mb /= n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    const va = vaBuf[i] - ma;
    const vb = vbBuf[i] - mb;
    num += va * vb;
    da += va * va;
    db += vb * vb;
  }
  const den = Math.sqrt(da * db);
  return den < 1e-6 ? -1 : num / den;
}

export function trackPoint(
  prev: Float32Array, pw: number, ph: number,
  next: Float32Array, nw: number, nh: number,
  u: number, v: number,
  guessDx = 0, guessDy = 0,
  searchR = SEARCH_R,
): FlowResult {
  const gx = u + guessDx;
  const gy = v + guessDy;
  let best = -2;
  let bu = gx, bv = gy;
  for (let dy = -searchR; dy <= searchR; dy += 2) {
    for (let dx = -searchR; dx <= searchR; dx += 2) {
      const x = gx + dx, y = gy + dy;
      if (x < PATCH_R + 1 || y < PATCH_R + 1 || x >= nw - PATCH_R - 1 || y >= nh - PATCH_R - 1) continue;
      const s = patchNcc(prev, pw, ph, next, nw, nh, u, v, x, y);
      if (s > best) {
        best = s;
        bu = x;
        bv = y;
      }
    }
  }
  for (let dy = -2; dy <= 2; dy++) {
    for (let dx = -2; dx <= 2; dx++) {
      const x = Math.round(bu + dx), y = Math.round(bv + dy);
      if (x < PATCH_R + 1 || y < PATCH_R + 1 || x >= nw - PATCH_R - 1 || y >= nh - PATCH_R - 1) continue;
      const s = patchNcc(prev, pw, ph, next, nw, nh, u, v, x, y);
      if (s > best) {
        best = s;
        bu = x;
        bv = y;
      }
    }
  }
  // Subpixel peak by a separable parabola on the NCC surface.
  const sXm = patchNcc(prev, pw, ph, next, nw, nh, u, v, bu - 1, bv);
  const sXp = patchNcc(prev, pw, ph, next, nw, nh, u, v, bu + 1, bv);
  const sYm = patchNcc(prev, pw, ph, next, nw, nh, u, v, bu, bv - 1);
  const sYp = patchNcc(prev, pw, ph, next, nw, nh, u, v, bu, bv + 1);
  const denX = sXm - 2 * best + sXp;
  const denY = sYm - 2 * best + sYp;
  const subX = Math.abs(denX) > 1e-9 ? (0.5 * (sXm - sXp)) / denX : 0;
  const subY = Math.abs(denY) > 1e-9 ? (0.5 * (sYm - sYp)) / denY : 0;
  const fx = bu + Math.max(-0.5, Math.min(0.5, subX));
  const fy = bv + Math.max(-0.5, Math.min(0.5, subY));
  return { u: fx, v: fy, ok: best > 0.55, ncc: best, error: 1 - Math.max(0, best) };
}

export interface TrackOptions {
  minTracks: number;
  maxTracks: number;
  fbTol: number;
  nccMin: number;
  /** Predictor from the gyro, in pixels. Zero when inertial aid is off. */
  predict?: (id: number) => { u: number; v: number } | null;
  /**
   * Global monotonically increasing id source. Without this, ids would be reused
   * after tracks die and cross-frame association would silently mismatch.
   */
  idCounter?: { value: number };
}

export function trackFrame(
  prev: FrameFeatures,
  next: FrameFeatures,
  tracks: TrackPoint[],
  keyframe: FrameFeatures | null,
  keyframeDescriptors: { descriptors: Uint8Array; sums: Uint16Array; corners: Corner[] } | null,
  opts: TrackOptions,
  frameIndex: number,
): TrackResult {
  const nTracks: TrackPoint[] = [];
  let dropped = 0;
  const residuals: number[] = [];
  for (const tr of tracks) {
    if (tr.confidence < 0.15) {
      dropped++;
      continue;
    }
    const guess = opts.predict ? opts.predict(tr.id) : null;
    const guessDx = guess ? guess.u - tr.u : (tr.u - tr.prevU);
    const guessDy = guess ? guess.v - tr.v : (tr.v - tr.prevV);
    // Bounded search: flow is local, inertial prediction narrows it further.
    const searchR = guess ? 3 : SEARCH_R;
    const fwd = trackPointPyramid(prev, next, tr.u, tr.v, guessDx, guessDy);
    if (!fwd.ok || fwd.ncc < opts.nccMin) {
      dropped++;
      continue;
    }
    const back = trackPointPyramid(next, prev, fwd.u, fwd.v, 0, 0);
    const fbErr = Math.hypot(back.u - tr.u, back.v - tr.v);
    if (!back.ok || fbErr > opts.fbTol) {
      dropped++;
      continue;
    }
    residuals.push(fbErr);
    nTracks.push({
      ...tr,
      prevU: tr.u,
      prevV: tr.v,
      u: fwd.u,
      v: fwd.v,
      age: tr.age + 1,
      confidence: Math.min(1, Math.max(0, (fwd.ncc - 0.45) * 2.2)),
      stale: false,
    });
  }

  // Top-up with fresh corners that do not collide existing tracks.
  const occupied = new Uint8Array(next.width * next.height);
  for (const tr of nTracks) {
    for (let dy = -4; dy <= 4; dy++) {
      for (let dx = -4; dx <= 4; dx++) {
        const x = Math.round(tr.u + dx), y = Math.round(tr.v + dy);
        if (x >= 0 && y >= 0 && x < next.width && y < next.height) occupied[y * next.width + x] = 1;
      }
    }
  }
  const counter = opts.idCounter ?? { value: tracks.reduce((m, t) => Math.max(m, t.id + 1), 0) };
  let nextId = counter.value;
  if (nTracks.length < opts.maxTracks) {
    for (const kp of next.keypoints) {
      if (nTracks.length >= opts.maxTracks) break;
      const x = Math.round(kp.u), y = Math.round(kp.v);
      if (occupied[y * next.width + x]) continue;
      for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx >= 0 && yy >= 0 && xx < next.width && yy < next.height) occupied[yy * next.width + xx] = 1;
      }
      nTracks.push({
        id: nextId++,
        u: kp.u,
        v: kp.v,
        prevU: kp.u,
        prevV: kp.v,
        bornAt: frameIndex,
        age: 0,
        confidence: 1,
        stale: false,
      });
    }
  }
  const relocalised = 0;
  if (nTracks.length < opts.minTracks && keyframe && keyframeDescriptors) {
    // Re-acquire by descriptor match to the last accepted keyframe.
    const used = new Set<number>();
    for (let k = 0; k < next.keypoints.length && nTracks.length < opts.maxTracks; k++) {
      const kp = next.keypoints[k];
      let bestD = 999;
      let bestJ = -1;
      for (let j = 0; j < keyframeDescriptors.corners.length; j++) {
        if (used.has(j)) continue;
        const d = hamming(next.descriptors, k * DESCRIPTOR_BITS, keyframeDescriptors.descriptors, j * DESCRIPTOR_BITS);
        if (d < bestD) {
          bestD = d;
          bestJ = j;
        }
      }
      if (bestJ >= 0 && bestD < 34) {
        used.add(bestJ);
        nTracks.push({
          id: nextId++,
          u: kp.u,
          v: kp.v,
          prevU: kp.u,
          prevV: kp.v,
          bornAt: frameIndex,
          age: 0,
          confidence: 1,
          stale: false,
        });
      }
    }
  }
  counter.value = nextId;
  return {
    tracks: nTracks,
    relocalised,
    dropped,
    flowResidual: residuals.length ? residuals.reduce((a, b) => a + b, 0) / residuals.length : 0,
  };
}

export function flowResidualReport(residuals: number[]): { mean: number; p95: number; count: number } {
  if (residuals.length === 0) return { mean: 0, p95: 0, count: 0 };
  const sorted = [...residuals].sort((a, b) => a - b);
  return {
    mean: sorted.reduce((a, b) => a + b, 0) / sorted.length,
    p95: sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))],
    count: sorted.length,
  };
}
