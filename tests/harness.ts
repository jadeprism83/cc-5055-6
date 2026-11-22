import { Mat3, Vec3 } from '../src/core/math';
import { defaultExtrinsics, Intrinsics } from '../src/core/camera';
import { GeneratedDataset, SimConfig } from '../src/sim/trajectory';
import { Scene } from '../src/vision/scene';
import { FrameFeatures, TrackPoint } from '../src/vision/tracker';
import { SensorSession } from '../src/lens/session';
import { buildCapture, sessionFromCapture } from '../src/lens/replay';

export interface RenderedSequence {
  ds: GeneratedDataset;
  scene: Scene;
  features: FrameFeatures[];
  trackSets: TrackPoint[][];
  flowResidual: number[];
  K: Intrinsics;
  /** Camera-clocked truth poses for each rendered frame, evaluation only. */
  truthR: Mat3[];
  truthP: Vec3[];
  truthCamR: Mat3[];
  truthCamP: Vec3[];
}

export function renderSequence(cfg: SimConfig, _seed = 99): RenderedSequence {
  // The test harness and the browser replay share one capture pipeline so their
  // sensor data cannot drift apart.
  const cap = buildCapture(cfg, { supersample: 2 });
  return {
    ds: cap.ds,
    scene: cap.scene,
    features: cap.features,
    trackSets: cap.trackSets,
    flowResidual: cap.flowResidual,
    K: cap.K,
    truthR: cap.truthR,
    truthP: cap.truthP,
    truthCamR: cap.truthCamR,
    truthCamP: cap.truthCamP,
  };
}

export function sessionFromSequence(seq: RenderedSequence, givenExtrinsics = defaultExtrinsics()): SensorSession {
  return sessionFromCapture(
    {
      config: seq.ds.config,
      ds: seq.ds,
      scene: seq.scene,
      K: seq.K,
      features: seq.features,
      trackSets: seq.trackSets,
      flowResidual: seq.flowResidual,
      rgbFrames: [],
      giftedExtrinsics: givenExtrinsics,
      truthR: seq.truthR,
      truthP: seq.truthP,
      truthCamR: seq.truthCamR,
      truthCamP: seq.truthCamP,
    },
    givenExtrinsics,
  );
}
