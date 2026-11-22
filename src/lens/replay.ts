import { Gaussian, Mat3, Vec3, m3mul, m3mulVec, mulberry32, v3add } from '../core/math';
import { defaultExtrinsics, defaultIntrinsics, Extrinsics, Intrinsics } from '../core/camera';
import { defaultSimConfig, generateDataset, GeneratedDataset, SimConfig, isInDropout } from '../sim/trajectory';
import { generateScene, renderScene, Scene } from '../vision/scene';
import { describe as describeFrame, FrameFeatures, trackFrame, TrackPoint } from '../vision/tracker';
import { buildSession, SensorSession } from './session';

// Sensor-side capture built from the simulator: images, raw IMU, per-frame
// tracks. The estimator consumes the resulting session only; truth poses are
// kept apart for the evaluation overlay, never fed into the kernel.

export interface Capture {
  config: SimConfig;
  ds: GeneratedDataset;
  scene: Scene;
  K: Intrinsics;
  features: FrameFeatures[];
  trackSets: TrackPoint[][];
  flowResidual: number[];
  /** Display RGB (intrinsics resolution), one per camera frame. */
  rgbFrames: Uint8ClampedArray[];
  giftedExtrinsics: Extrinsics;
  /** Truth body poses, evaluation only. */
  truthR: Mat3[];
  truthP: Vec3[];
  truthCamR: Mat3[];
  truthCamP: Vec3[];
}

export interface CaptureOptions {
  /** Render supersampling factor for anti-aliased tracking texture. */
  supersample?: number;
  onProgress?: (done: number, total: number) => void;
}

export function buildCapture(cfg: SimConfig, opts: CaptureOptions = {}): Capture {
  const ds = generateDataset(cfg);
  const scene = generateScene(13);
  const SS = opts.supersample ?? 2;
  const K = defaultIntrinsics(320, 240);
  const Ks: Intrinsics = {
    fx: K.fx * SS,
    fy: K.fy * SS,
    cx: K.cx * SS,
    cy: K.cy * SS,
    width: K.width * SS,
    height: K.height * SS,
  };
  const g = new Gaussian(mulberry32(99));
  const truth = ds.cameraTruth;
  const rgbHi = new Uint8ClampedArray(Ks.width * Ks.height * 3);
  const grayHi = new Float32Array(Ks.width * Ks.height);
  const grayLo = new Float32Array(K.width * K.height);
  const rgbLo = new Uint8ClampedArray(K.width * K.height * 3);
  const features: FrameFeatures[] = [];
  const trackSets: TrackPoint[][] = [];
  const flowResidual: number[] = [];
  const rgbFrames: Uint8ClampedArray[] = [];
  const truthR: Mat3[] = [];
  const truthP: Vec3[] = [];
  const truthCamR: Mat3[] = [];
  const truthCamP: Vec3[] = [];
  let prev: FrameFeatures | null = null;
  let prevTracks: TrackPoint[] = [];
  const idCounter = { value: 1 };
  for (let k = 0; k < truth.length; k++) {
    const c = truth[k];
    const drop = isInDropout(cfg, c.t);
    const blur = drop?.kind === 'blur';
    const exposure = blur ? cfg.blurExposure : 1;
    renderScene(
      scene,
      { R_wb: c.R_wb, p_wb: c.p_wb, R_bc: cfg.R_bc, t_bc: cfg.t_bc },
      Ks,
      {
        exposure,
        pixelNoise: 0,
        gaussian: g,
        outGray: grayHi,
        outRgb: rgbHi,
        blurRadius: blur ? 4 : 0,
      },
    );
    for (let y = 0; y < K.height; y++) {
      for (let x = 0; x < K.width; x++) {
        let sg = 0;
        let sr = 0;
        let sgg = 0;
        let sb = 0;
        for (let dy = 0; dy < SS; dy++) {
          for (let dx = 0; dx < SS; dx++) {
            const hi = (y * SS + dy) * Ks.width + (x * SS + dx);
            sg += grayHi[hi];
            sr += rgbHi[hi * 3];
            sgg += rgbHi[hi * 3 + 1];
            sb += rgbHi[hi * 3 + 2];
          }
        }
        const n = SS * SS;
        grayLo[y * K.width + x] = sg / n + g.next() * cfg.pixelNoise;
        const lo = (y * K.width + x) * 3;
        rgbLo[lo] = sr / n;
        rgbLo[lo + 1] = sgg / n;
        rgbLo[lo + 2] = sb / n;
      }
    }
    // Each frame owns its pixels: the render buffers are reused, and the tracker
    // would otherwise see the next frame overwrite the previous image mid-flow.
    const feat = describeFrame(new Float32Array(grayLo), K.width, K.height);
    features.push(feat);
    const tracked = trackFrame(
      prev ?? feat,
      feat,
      prevTracks,
      prev,
      null,
      { minTracks: 18, maxTracks: 160, fbTol: 1.4, nccMin: 0.62, idCounter },
      k,
    );
    trackSets.push(tracked.tracks);
    flowResidual.push(tracked.flowResidual);
    prevTracks = tracked.tracks;
    prev = feat;
    rgbFrames.push(new Uint8ClampedArray(rgbLo));
    truthR.push(new Float64Array(c.R_wb));
    truthP.push([...c.p_wb]);
    const Rcam = m3mul(c.R_wb, cfg.R_bc);
    const Ccam = v3add(c.p_wb, m3mulVec(c.R_wb, cfg.t_bc));
    truthCamR.push(Rcam);
    truthCamP.push(Ccam);
    opts.onProgress?.(k + 1, truth.length);
  }
  return {
    config: cfg,
    ds,
    scene,
    K,
    features,
    trackSets,
    flowResidual,
    rgbFrames,
    giftedExtrinsics: defaultExtrinsics(),
    truthR,
    truthP,
    truthCamR,
    truthCamP,
  };
}

export function sessionFromCapture(cap: Capture, givenExtrinsics?: Extrinsics): SensorSession {
  return buildSession(
    cap.ds.cameraTimes,
    cap.ds.imu,
    cap.features,
    cap.trackSets,
    cap.flowResidual,
    cap.K,
    { givenExtrinsics: givenExtrinsics ?? cap.giftedExtrinsics, pixelNoise: cap.config.pixelNoise, seed: 5150 },
  );
}

export function defaultCaptureConfig(seed = 7): SimConfig {
  return defaultSimConfig(seed);
}
