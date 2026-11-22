import {
  Mat3,
  Vec3,
  m3mul,
  m3transpose,
  so3Log,
  v3norm,
  v3sub,
} from '../core/math';
import { Extrinsics, Intrinsics, cameraCenterWorld } from '../core/camera';
import { Eskf, EskfNoise, rotationErrorAngle } from '../estimation/eskf';
import { estimateRelativePose } from '../estimation/relpose';
import { estimatePnP } from '../estimation/pnp';
import { SensorSession, TrackObs } from './session';
import {
  BootstrapResult,
  runBootstrap,
  RelativeVisualPose,
  triangulateViews,
  TriangulationView,
} from './bootstrap';
import { solveHandEyeRotation, HandEyePair } from './calibration';
import { CalibrationRecord, defaultCalibrationRecord } from './record';
import { GyroPreintegrator } from './preintegration';
import { hamming } from '../vision/tracker';

// The inertial-lens kernel. It consumes only sensor-side data (timestamps, raw
// gyro/accel, image features and tracks) and emits estimator state plus
// diagnostics. Ground truth is not referenced anywhere in this file.

export type LensStatus = 'idle' | 'bootstrapping' | 'tracking' | 'inertial' | 'lost' | 'relocalising' | 'stopped';

export interface KernelEvent {
  frame: number;
  severity: 'info' | 'warn' | 'error';
  message: string;
}

export interface KernelFrame {
  index: number;
  t: number;
  status: LensStatus;
  tracks: number;
  used: number;
  rejectedGate: number;
  reprojRms: number;
  reprojP95: number;
  sigmaThetaDeg: number;
  sigmaPos: number;
  bg: Vec3;
  ba: Vec3;
  R_wb: Mat3;
  p_wb: Vec3;
  v_wb: Vec3;
  predictedOnly: boolean;
  holdout: boolean;
  effect: string;
  relocalised: boolean;
  biasInflated: boolean;
  flowResidualPx: number;
  rotResidualDeg: number;
}

export interface KernelOptions {
  /**
   * When false, the IMU is not used to propagate the state at all: the filter
   * holds its pose between frames and relies only on visual updates. This is the
   * vision-only control arm of the joint-vs-vision comparison.
   */
  useInertialPropagation: boolean;
  /** Frames used to average gravity and the static gyro bias. */
  staticFrameEnd: number;
  /** VI bootstrap window, inclusive. */
  bootstrapStart: number;
  bootstrapEnd: number;
  /** Visual updates are skipped on every Nth frame after bootstrap. */
  holdoutEvery: number;
  timeOffsetGuess: number;
  /** Relative rotation uncertainty used when updating yaw. */
  rotationSigmaRad: number;
  pixelSigma: number;
  /** Enable the mid-run gyro-bias step detector. */
  detectBiasSteps: boolean;
  /** Relocalise when orientation sigma exceeds this. */
  relocSigmaRad: number;
  /** Force relocalisation if an outage lasts this many frames. */
  relocOutageFrames: number;
}

export function defaultKernelOptions(session: SensorSession): KernelOptions {
  const hz = 1 / (session.cameraTimes[1] - session.cameraTimes[0]);
  return {
    useInertialPropagation: true,
    staticFrameEnd: Math.round(0.6 * hz),
    bootstrapStart: Math.round(0.7 * hz),
    bootstrapEnd: Math.round(2.9 * hz),
    holdoutEvery: 11,
    timeOffsetGuess: 0,
    rotationSigmaRad: 0.02,
    pixelSigma: 1.0,
    detectBiasSteps: true,
    relocSigmaRad: 0.05,
    relocOutageFrames: 18,
  };
}

export interface LandmarkRecord {
  id: number;
  p_w: Vec3;
  views: number;
  reprojPx: number;
  parallax: number;
  firstFrame: number;
}

interface KeyframeSnapshot {
  index: number;
  R_wb: Mat3;
  p_wb: Vec3;
  keypoints: Array<{ u: number; v: number }>;
  descriptors: Uint8Array;
  /** Map landmark id per keypoint, or -1. */
  landmarkOf: Int32Array;
}

export class InertialLensKernel {
  readonly session: SensorSession;
  readonly opts: KernelOptions;
  readonly noise: EskfNoise;
  readonly extrinsics: Extrinsics;
  /** Calibration record the preintegrator converts samples with. */
  record: CalibrationRecord;
  preintegrator: GyroPreintegrator;

  status: LensStatus = 'idle';
  frameIndex = 0;
  eskf: Eskf | null = null;
  bootstrapResult: BootstrapResult | null = null;
  inertialAid = true;
  events: KernelEvent[] = [];
  frames: KernelFrame[] = [];
  landmarks = new Map<number, LandmarkRecord>();
  timeOffset: number;
  extrinsicSource: 'given' | 'solved' = 'given';
  timeOffsetSource: 'given' | 'solved' = 'given';
  handEyeCheck: { solved: Mat3; residualDeg: number; pairs: number; message: string } | null = null;

  private trackHist = new Map<number, Array<{ frame: number; u: number; v: number }>>();
  private poseHistory: Array<{ frame: number; R_wb: Mat3; p_wb: Vec3 }> = [];
  private keyframe: KeyframeSnapshot | null = null;
  private lastTrackingFrame = -1;
  private outageFrames = 0;
  private biasMonitor: number[] = [];
  private biasInflated = false;
  private consecutiveRotErr: number[] = [];
  private lastFlowResidual = 0;
  private nestedJump = false;
  private lastGoodFrame = -1;
  private relocalisations = 0;
  private nextLandmarkId = 100000;
  private lastFirmwareWarnFrame = -100;

  constructor(session: SensorSession, opts: KernelOptions, noise: EskfNoise, record?: CalibrationRecord) {
    this.session = session;
    this.opts = opts;
    this.noise = noise;
    this.extrinsics = {
      R_bc: new Float64Array(session.initialExtrinsics.R_bc),
      t_bc: [...session.initialExtrinsics.t_bc],
    };
    this.timeOffset = opts.timeOffsetGuess;
    this.record = record ?? defaultCalibrationRecord(this.extrinsics, this.timeOffset);
    this.preintegrator = new GyroPreintegrator(this.record);
    // The session converts every firmware-tagged sample through the record's
    // committed convention; samples without a committed convention are dropped.
    this.session.setConventionResolver((fw) => this.preintegrator.conventionFor(fw));
  }

  private log(severity: KernelEvent['severity'], message: string): void {
    this.events.push({ frame: this.frameIndex, severity, message });
    if (this.events.length > 300) this.events.shift();
  }

  private relPoseAt(k: number): RelativeVisualPose | null {
    return this.session.relativePoses.get(k) ?? null;
  }

  private buildBootstrapInput(start: number, end: number) {
    const trackPixels = new Map<number, Map<number, { u: number; v: number }>>();
    const trackBirth = new Map<number, number>();
    for (let k = start; k <= end; k++) {
      const map = new Map<number, { u: number; v: number }>();
      for (const tr of this.session.tracks[k]) {
        map.set(tr.id, { u: tr.u, v: tr.v });
        if (!trackBirth.has(tr.id)) trackBirth.set(tr.id, tr.age > 0 ? k - tr.age : k);
      }
      trackPixels.set(k, map);
    }
    return {
      cameraTimes: this.session.cameraTimes,
      imu: this.session.imuBody(),
      bgGuess: [0, 0, 0] as Vec3,
      R_bc: this.extrinsics.R_bc,
      t_bc: this.extrinsics.t_bc,
      K: this.session.K,
      start,
      end,
      relative: this.session.relativePoses,
      trackPixels,
      trackBirth,
      timeOffsetGuess: this.opts.timeOffsetGuess,
    };
  }

  /** Run bootstrap. Returns false and logs the reason when it cannot proceed. */
  bootstrap(): boolean {
    this.status = 'bootstrapping';
    const start = Math.max(1, this.opts.bootstrapStart);
    const end = Math.min(this.session.cameraTimes.length - 1, this.opts.bootstrapEnd);
    if (end <= start + 3) {
      this.log('error', '初始化失败：VI 窗口帧数不足');
      this.status = 'idle';
      return false;
    }
    const input = this.buildBootstrapInput(start, end);
    const res = runBootstrap(input, this.opts.staticFrameEnd, {
      timeOffsetGuess: this.opts.timeOffsetGuess,
      minParallaxRad: 0.012,
      maxLandmarkReprojPx: 2.5,
    });
    this.bootstrapResult = res;
    const keepGivenOffset = res.timeOffsetScore < 0.02 && res.timeOffset !== this.opts.timeOffsetGuess;
    if (keepGivenOffset) {
      this.log('warn', `时差搜索最小值不显著（RMS ${(res.timeOffsetScore * 57.3).toFixed(2)}°），保留输入时差`);
      res.timeOffset = this.opts.timeOffsetGuess;
    }
    this.timeOffset = res.timeOffset;
    this.timeOffsetSource = keepGivenOffset ? 'given' : res.ok ? 'solved' : this.timeOffsetSource;
    if (!res.ok) {
      // Keep orientation even without metric scale, but do not claim position.
      this.log('error', res.message);
      this.status = 'idle';
      this.bootstrapResult = res;
      return false;
    }
    this.eskf = Eskf.initial(res.R_wb, res.p_wb, res.v_wb, res.bg, res.ba, this.noise, {
      theta: 0.02,
      v: 0.12,
      p: Math.max(0.05, res.scaleResidual * 3),
      bg: 0.003,
      ba: 0.05,
    });
    this.frameIndex = end;
    this.poseHistory = [];
    this.trackHist.clear();
    for (const lm of res.landmarks) {
      this.landmarks.set(lm.id, {
        id: lm.id,
        p_w: lm.p_w,
        views: lm.views,
        reprojPx: lm.reprojPx,
        parallax: lm.parallax,
        firstFrame: start,
      });
    }
    this.status = 'tracking';
    this.lastTrackingFrame = end;
    this.lastGoodFrame = end;
    this.captureKeyframe(end);
    this.log('info', res.message);
    this.log('info', res.scaleEvidence);
    this.log('info', `时差来源：${this.timeOffsetSource === 'solved' ? '求解' : '输入'}；外参来源：${this.extrinsicSource === 'given' ? '输入' : '求解'}`);
    if (res.landmarks.length < 10) {
      this.log('warn', `初始化地标仅 ${res.landmarks.length} 个，视觉更新可能不足`);
    }
    // A hand-eye rotation check is offered, but never silently substituted.
    this.runHandEyeCheck();
    return true;
  }

  /** Solve camera-IMU rotation over the bootstrap window and compare to input. */
  runHandEyeCheck(): void {
    const pairs: HandEyePair[] = [];
    for (let k = Math.max(1, this.opts.bootstrapStart); k <= Math.min(this.session.cameraTimes.length - 1, this.opts.bootstrapEnd); k++) {
      const rel = this.relPoseAt(k);
      if (!rel || rel.inliers < 15 || rel.medianErrorPx > 1.2) continue;
      const fromK = k - 1;
      if (fromK < 0) continue;
      // Body relative rotation from raw gyro, bias removed with the solved bias.
      const bg = this.eskf ? this.eskf.bg : ([0, 0, 0] as Vec3);
      const dR = this.session.gyroRotation(fromK, k, this.timeOffset, bg);
      if (!dR) continue;
      pairs.push({ dR_body: dR, dR_cam: rel.R, weight: rel.inliers });
    }
    const solved = solveHandEyeRotation(pairs);
    this.handEyeCheck = {
      solved: solved.R_bc,
      residualDeg: solved.rotationResidualDeg,
      pairs: solved.pairsUsed,
      message: solved.message,
    };
    this.log(solved.ok ? 'info' : 'warn', solved.message);
  }

  applySolvedExtrinsic(): void {
    if (!this.handEyeCheck) return;
    if (!Number.isFinite(this.handEyeCheck.residualDeg) || this.handEyeCheck.residualDeg > 1.5) {
      this.log('error', `拒绝应用外参：残差 ${this.handEyeCheck.residualDeg.toFixed(2)}° 过大`);
      return;
    }
    this.extrinsics.R_bc = new Float64Array(this.handEyeCheck.solved);
    this.extrinsicSource = 'solved';
    this.record.extrinsics = { R_bc: new Float64Array(this.extrinsics.R_bc), t_bc: [...this.extrinsics.t_bc] };
    this.log('info', '已应用手眼旋转外参；平移仍来自输入（旋转外参不能观测平移）');
    // Re-run bootstrap with the corrected rotation; the old map is discarded.
    this.resetState();
    this.bootstrap();
  }

  setTimeOffset(offset: number): void {
    this.timeOffset = offset;
    this.timeOffsetSource = 'given';
    this.record.timeOffset = offset;
    this.resetState();
    this.log('info', `时差已设回输入值 ${(offset * 1000).toFixed(1)} ms，等待重新初始化`);
  }

  setExtrinsic(R_bc: Mat3, t_bc: Vec3): void {
    this.extrinsics.R_bc = new Float64Array(R_bc);
    this.extrinsics.t_bc = [...t_bc];
    this.extrinsicSource = 'given';
    this.record.extrinsics = { R_bc: new Float64Array(R_bc), t_bc: [...t_bc] };
    this.resetState();
    this.log('info', '外参已设回输入值，等待重新初始化');
  }

  /**
   * Commit a new firmware's calibration record. The generation must increase;
   * snapshots stamped with the old generation become stale and are rejected by
   * the overlays. The state resets and re-bootstraps under the new conversion.
   */
  commitFirmware(next: CalibrationRecord): boolean {
    if (next.status !== 'committed') {
      this.log('error', `拒绝启用 ${next.firmware} 标定：记录仍是预览态（镜像歧义未排除）`);
      return false;
    }
    if (next.generation <= this.record.generation) {
      this.log('error', `拒绝启用 ${next.firmware} 标定：代际 ${next.generation} 未超过当前 ${this.record.generation}`);
      return false;
    }
    this.record = next;
    this.preintegrator = new GyroPreintegrator(next);
    this.session.setConventionResolver((fw) => this.preintegrator.conventionFor(fw));
    this.resetState();
    this.log('info', `已固化 ${next.firmware} 标定（代 ${next.generation}），旧换算快照一律拒绝`);
    return true;
  }

  resetState(): void {
    this.eskf = null;
    this.bootstrapResult = null;
    this.status = 'idle';
    this.frames = [];
    this.events = [];
    this.landmarks.clear();
    this.trackHist.clear();
    this.poseHistory = [];
    this.keyframe = null;
    this.frameIndex = 0;
    this.outageFrames = 0;
    this.biasMonitor = [];
    this.biasInflated = false;
    this.consecutiveRotErr = [];
    this.relocalisations = 0;
    this.lastTrackingFrame = -1;
    this.lastGoodFrame = -1;
  }

  toggleInertialAid(): boolean {
    this.inertialAid = !this.inertialAid;
    this.log('info', this.inertialAid ? '惯性辅助开启' : '惯性辅助关闭');
    return this.inertialAid;
  }

  /** Step one camera frame. Returns null once the sequence is exhausted. */
  advance(): KernelFrame | null {
    if (this.frameIndex >= this.session.cameraTimes.length - 1) {
      this.status = 'stopped';
      return null;
    }
    if (this.status === 'idle') {
      if (!this.bootstrap()) return null;
      // Bootstrap consumed frames up to bootstrapEnd; emit that frame.
      return this.emitBootstrapFrame();
    }
    if (!this.eskf) return null;
    this.frameIndex++;
    const t = this.session.cameraTimes[this.frameIndex];
    const tracks = this.session.tracks[this.frameIndex];
    const holdout = this.opts.holdoutEvery > 0 && (this.frameIndex - this.opts.bootstrapEnd) % this.opts.holdoutEvery === 0;
    // 1) IMU propagation between frames.
    this.propagate(t);
    // 2) Visual measurement assessment.
    const relPose = this.relPoseAt(this.frameIndex);
    const flowResidual = this.session.flowResidual[this.frameIndex] ?? 0;
    this.lastFlowResidual = flowResidual;
    const visualHealthy = this.visualHealthy(tracks.length, relPose, flowResidual);
    let used = 0;
    let rejected = 0;
    let reproj: number[] = [];
    let relocalised = false;
    let effect = '';
    const prevStatus = this.status;
    if (holdout) {
      effect = '留出帧：不参与视觉更新，仅惯推（用于独立检验）';
      this.status = this.inertialAid ? 'inertial' : 'lost';
    } else if (visualHealthy) {
      const upd = this.visualUpdate(tracks, relPose);
      used = upd.used;
      rejected = upd.rejected;
      reproj = upd.reproj;
      if (used >= 5) {
        this.status = 'tracking';
        this.outageFrames = 0;
        this.lastTrackingFrame = this.frameIndex;
        this.lastGoodFrame = this.frameIndex;
        effect = `视觉更新 ${used} 点（剔除 ${rejected}）`;
        this.trackNewLandmarks(tracks);
        if (this.frameIndex % 12 === 0) this.captureKeyframe(this.frameIndex);
        this.monitorBiasStep(relPose);
      } else if (this.eskf.sigmaTheta() > this.opts.relocSigmaRad) {
        this.status = 'relocalising';
        relocalised = this.tryRelocalise(prevStatus !== 'tracking');
        if (relocalised) {
          this.status = 'tracking';
          effect = '视觉更新不足，已用关键帧描述子+PnP 重定位';
        } else {
          this.status = this.inertialAid ? 'inertial' : 'lost';
          effect = this.lastRelocMessage;
        }
      } else {
        this.status = this.inertialAid ? 'inertial' : 'lost';
        effect = `视觉量测不足（${used} 点），继续惯推`;
      }
    } else {
      this.outageFrames++;
      this.lastFlowResidual = flowResidual;
      if (this.inertialAid) {
        this.status = 'inertial';
        effect = `视觉掉点（跟踪 ${tracks.length}，流残差 ${flowResidual.toFixed(2)} px）：惯推`;
        if (
          this.outageFrames > this.opts.relocOutageFrames ||
          this.eskf.sigmaTheta() > this.opts.relocSigmaRad
        ) {
          if (this.tryRelocalise(true)) {
            relocalised = true;
            this.status = 'tracking';
            this.outageFrames = 0;
            effect = '掉点后通过描述子+PnP 重定位恢复';
          } else {
            effect = this.lastRelocMessage;
          }
        }
      } else {
        this.status = 'lost';
        effect = '惯性辅助关闭：视觉丢失时冻结状态，不外推';
      }
    }
    if (this.status === 'tracking') this.updateTrackHistory(tracks);
    // Surface status transitions to the UI event log so failure feedback reaches
    // it even when per-frame effects stay in the same state.
    if (this.status !== prevStatus) {
      const sev: KernelEvent['severity'] = this.status === 'tracking' ? 'info' : this.status === 'lost' ? 'error' : 'warn';
      this.log(sev, `状态 ${prevStatus} → ${this.status}：${effect}`);
    } else if (!this.inertialAid && this.status === 'lost') {
      // While lost without inertial aid, keep the reason visible at a slow rate.
      if (this.frameIndex % 10 === 0) this.log('error', effect);
    }
    const rec = this.emitFrame(t, tracks.length, used, rejected, reproj, holdout, effect, relocalised);
    if (this.frameIndex >= this.session.cameraTimes.length - 1) this.status = 'stopped';
    return rec;
  }

  private lastRelocMessage = '';

  private propagate(t: number): void {
    if (!this.eskf) return;
    if (!this.opts.useInertialPropagation) return;
    const prevT = this.session.cameraTimes[this.frameIndex - 1];
    const samples = this.session.imuBetween(prevT, t, this.timeOffset);
    const unknown = this.session.unknownFirmwareInfo();
    if (unknown && t >= unknown.firstT && this.frameIndex - this.lastFirmwareWarnFrame >= 20) {
      this.lastFirmwareWarnFrame = this.frameIndex;
      this.log('warn', `固件 ${unknown.fw} 的样本没有已固化标定：惯推暂停，等待固化新标定（不会用默认轴系换算）`);
    }
    for (let i = 1; i < samples.length; i++) {
      const a = samples[i - 1];
      const b = samples[i];
      const h = b.t - a.t;
      if (h <= 0) continue;
      const g: Vec3 = [(a.gyro[0] + b.gyro[0]) / 2, (a.gyro[1] + b.gyro[1]) / 2, (a.gyro[2] + b.gyro[2]) / 2];
      const acc: Vec3 = [(a.accel[0] + b.accel[0]) / 2, (a.accel[1] + b.accel[1]) / 2, (a.accel[2] + b.accel[2]) / 2];
      this.eskf.predict(g, acc, h);
    }
  }

  private visualHealthy(trackCount: number, rel: RelativeVisualPose | null, flow: number): boolean {
    if (trackCount < 8) return false;
    if (!rel) return false;
    if (rel.medianErrorPx > 2.2 || rel.inliers < 12) return false;
    return true;
  }

  private visualUpdate(tracks: TrackObs[], rel: RelativeVisualPose | null): { used: number; rejected: number; reproj: number[] } {
    if (!this.eskf) return { used: 0, rejected: 0, reproj: [] };
    let used = 0;
    let rejected = 0;
    const reproj: number[] = [];
    for (const tr of tracks) {
      const lm = this.landmarks.get(tr.id);
      if (!lm) continue;
      // Drop landmarks seen too long ago; their world position may be stale.
      if (this.frameIndex - lm.firstFrame > 400) continue;
      const r = this.eskf.updateLandmark(
        this.extrinsics.R_bc,
        this.extrinsics.t_bc,
        this.session.K,
        lm.p_w,
        tr.u,
        tr.v,
        this.opts.pixelSigma,
        9.2,
      );
      if (r.accepted) used++;
      else {
        rejected++;
        if (r.rejectedReason === 'gate' && r.residual > 40) {
          // A gross mismatch means the landmark association or the map is bad;
          // remove it rather than repeatedly poisoning the filter.
          this.landmarks.delete(tr.id);
        }
      }
    }
    // Yaw constraint from the image-only relative rotation. This is independent
    // of the landmark map and resists drift when landmarks go stale.
    if (rel && rel.inliers >= 15 && this.frameIndex >= 1) {
      const posePrev = this.poseHistory.find((p) => p.frame === this.frameIndex - 1);
      if (posePrev) {
        const res = this.eskf.updateRelativeRotation(
          this.extrinsics.R_bc,
          rel.R,
          posePrev.R_wb,
          this.opts.rotationSigmaRad,
          9.2,
        );
        if (res.accepted) used++;
        else rejected++;
      }
    }
    if (used > 0) {
      for (const v of this.eskf.lastVisualResiduals.slice(-used)) reproj.push(v);
      this.eskf.lastVisualResiduals = [];
    }
    return { used, rejected, reproj };
  }

  /** Triangulate new landmarks from tracks with enough baseline. */
  private trackNewLandmarks(tracks: TrackObs[]): void {
    if (!this.eskf) return;
    for (const tr of tracks) {
      if (tr.age < 6 || this.landmarks.has(tr.id)) continue;
      const hist = this.trackHist.get(tr.id);
      if (!hist || hist.length < 5) continue;
      if ((this.frameIndex + tr.id) % 3 !== 0) continue;
      const views: TriangulationView[] = [];
      for (const h of hist) {
        const pose = this.poseHistory.find((p) => p.frame === h.frame);
        if (!pose) continue;
        views.push({ R_wc: m3transpose(m3mul(pose.R_wb, this.extrinsics.R_bc)), C: cameraCenterWorld(pose.R_wb, pose.p_wb, this.extrinsics), u: h.u, v: h.v });
      }
      if (views.length < 5) continue;
      const tri = triangulateViews(views, this.session.K);
      if (!tri || !tri.ok) continue;
      if (tri.reproj > 2.0 || tri.parallax < 0.010) continue;
      let maxRange = 0;
      for (const v of views) maxRange = Math.max(maxRange, v3norm(v3sub(tri.p, v.C)));
      if (maxRange > 30) continue;
      // Reject points behind any observing camera.
      let bad = false;
      for (const v of views) {
        const z = v.R_wc[2] * (tri.p[0] - v.C[0]) + v.R_wc[5] * (tri.p[1] - v.C[1]) + v.R_wc[8] * (tri.p[2] - v.C[2]);
        if (z <= 0.1) bad = true;
      }
      if (bad) continue;
      this.landmarks.set(tr.id, {
        id: tr.id,
        p_w: tri.p,
        views: views.length,
        reprojPx: tri.reproj,
        parallax: tri.parallax,
        firstFrame: this.frameIndex,
      });
    }
  }

  private updateTrackHistory(tracks: TrackObs[]): void {
    for (const tr of tracks) {
      const arr = this.trackHist.get(tr.id) ?? [];
      const last = arr[arr.length - 1];
      if (!last || last.frame !== this.frameIndex) arr.push({ frame: this.frameIndex, u: tr.u, v: tr.v });
      if (arr.length > 30) arr.splice(0, arr.length - 30);
      this.trackHist.set(tr.id, arr);
    }
    if (this.eskf) {
      const exists = this.poseHistory.some((p) => p.frame === this.frameIndex);
      if (!exists) {
        this.poseHistory.push({ frame: this.frameIndex, R_wb: new Float64Array(this.eskf.R), p_wb: [...this.eskf.p] });
        if (this.poseHistory.length > 600) this.poseHistory.shift();
      }
    }
  }

  private monitorBiasStep(rel: RelativeVisualPose | null): void {
    if (!this.opts.detectBiasSteps || !rel || !this.eskf) return;
    const dRgyro = this.session.gyroRotation(this.frameIndex - 1, this.frameIndex, this.timeOffset, this.eskf.bg);
    if (!dRgyro) return;
    // rel.R is the camera-frame forward increment; the body-frame gyro increment
    // equals R_wb(k) R_wb(k-1)^T, and R_bc cancels in that world increment, so
    // the two are directly comparable up to measurement noise.
    const RvisCam = m3mul(m3transpose(this.extrinsics.R_bc), m3mul(m3transpose(dRgyro), this.extrinsics.R_bc));
    const res = so3Log(m3mul(m3transpose(RvisCam), rel.R));
    const deg = v3norm(res) * (180 / Math.PI);
    this.consecutiveRotErr.push(deg);
    if (this.consecutiveRotErr.length > 12) this.consecutiveRotErr.shift();
    if (!this.biasInflated && this.consecutiveRotErr.length >= 6) {
      const recent = this.consecutiveRotErr.slice(-4);
      const avg = recent.reduce((a, b) => a + b, 0) / recent.length;
      const min = Math.min(...recent);
      if (avg > 0.45 && min > 0.3) {
        this.biasInflated = true;
        this.eskf.inflateGyroBias(0.01);
        this.log('warn', `检测到陀螺-视觉旋转不一致（平均 ${avg.toFixed(2)}°/帧）：放大陀螺偏置协方差，交由视觉更新重新收敛`);
      }
    }
  }

  private captureKeyframe(index: number): void {
    if (!this.eskf) return;
    const feat = this.session.vision[index];
    if (!feat) return;
    const landmarkOf = new Int32Array(feat.keypoints.length).fill(-1);
    for (let i = 0; i < feat.keypoints.length; i++) {
      const kp = feat.keypoints[i];
      // Associate to map landmarks by nearest current track pixel.
      for (const tr of this.session.tracks[index]) {
        if (Math.hypot(tr.u - kp.u, tr.v - kp.v) < 1.0 && this.landmarks.has(tr.id)) {
          landmarkOf[i] = tr.id;
          break;
        }
      }
    }
    this.keyframe = {
      index,
      R_wb: new Float64Array(this.eskf.R),
      p_wb: [...this.eskf.p],
      keypoints: feat.keypoints.map((k) => ({ u: k.u, v: k.v })),
      descriptors: new Uint8Array(feat.descriptors),
      landmarkOf,
    };
  }

  private tryRelocalise(allowBroadGate: boolean): boolean {
    if (!this.eskf) return false;
    const feat = this.session.vision[this.frameIndex];
    const kf = this.keyframe;
    this.lastRelocMessage = '重定位失败：无关键帧或当前帧特征';
    if (!feat || !kf) return false;
    const featCount = feat.keypoints.length;
    const kfCount = kf.keypoints.length;
    if (featCount < 8 || kfCount < 8) return false;
    const matches: Array<{ p_w: Vec3; u: number; v: number; id: number; dist: number }> = [];
    for (let i = 0; i < featCount; i++) {
      let best = 1000;
      let bestJ = -1;
      for (let j = 0; j < kfCount; j++) {
        if (kf.landmarkOf[j] < 0) continue;
        const d = hamming(feat.descriptors, i * 128, kf.descriptors, j * 128);
        if (d < best) {
          best = d;
          bestJ = j;
        }
      }
      if (bestJ >= 0 && best < 40) {
        const lm = this.landmarks.get(kf.landmarkOf[bestJ]);
        if (lm) matches.push({ p_w: lm.p_w, u: feat.keypoints[i].u, v: feat.keypoints[i].v, id: lm.id, dist: best });
      }
    }
    if (matches.length < 8) {
      this.lastRelocMessage = `重定位失败：描述子匹配仅 ${matches.length} 个（需 ≥8）`;
      return false;
    }
    // Uniqueness filter: drop ambiguous descriptors.
    const byLandmark = new Map<number, typeof matches>();
    for (const m of matches) {
      const arr = byLandmark.get(m.id) ?? [];
      arr.push(m);
      byLandmark.set(m.id, arr);
    }
    const uniq = [...byLandmark.values()].filter((a) => a.length === 1).map((a) => a[0]);
    const pnp = estimatePnP(
      uniq.map((m) => ({ p_w: m.p_w, u: m.u, v: m.v })),
      {
        K: this.session.K,
        R_bc: this.extrinsics.R_bc,
        t_bc: this.extrinsics.t_bc,
        thresholdPx: 3.0,
        minInliers: 8,
        ransacIterations: 180,
        seed: this.frameIndex * 131 + 7,
      },
    );
    if (!pnp.ok) {
      this.lastRelocMessage = `重定位失败：PnP 内点 ${pnp.inlierCount}/8`;
      return false;
    }
    const dp = v3norm(v3sub(pnp.p_wb, this.eskf.p));
    const dth = rotationErrorAngle(pnp.R_wb, this.eskf.R);
    // Teleport guard: a descriptor repeat can match the wrong wall.
    if (!allowBroadGate && (dp > 0.8 || dth > 0.25)) {
      this.lastRelocMessage = `重定位拒绝：PnP 与惯推相差 ${dp.toFixed(2)} m / ${(dth * 57.3).toFixed(1)}°，超过门限`;
      return false;
    }
    if (allowBroadGate && (dp > 4.0 || dth > 0.9)) {
      this.lastRelocMessage = `重定位拒绝：PnP 与惯推相差 ${dp.toFixed(2)} m / ${(dth * 57.3).toFixed(1)}°，超过重捕获上限`;
      return false;
    }
    this.eskf.hardSetPose(pnp.p_wb, pnp.R_wb, this.eskf.v, 0.03, 0.015);
    // Re-seed the map around the recovered pose by triangulating fresh tracks.
    this.keyframe = null;
    this.captureKeyframe(this.frameIndex);
    this.relocalisations++;
    this.biasInflated = false;
    this.consecutiveRotErr = [];
    this.lastRelocMessage = `重定位成功：PnP 内点 ${pnp.inlierCount}，中位重投影 ${pnp.medianReprojPx.toFixed(2)} px`;
    this.log('info', this.lastRelocMessage);
    return true;
  }

  private emitBootstrapFrame(): KernelFrame {
    if (!this.eskf) throw new Error('no state');
    const t = this.session.cameraTimes[this.frameIndex];
    const rec: KernelFrame = {
      index: this.frameIndex,
      t,
      status: this.status,
      tracks: this.session.tracks[this.frameIndex]?.length ?? 0,
      used: 0,
      rejectedGate: 0,
      reprojRms: 0,
      reprojP95: 0,
      sigmaThetaDeg: this.eskf.sigmaTheta() * (180 / Math.PI),
      sigmaPos: this.eskf.sigmaPosition(),
      bg: [...this.eskf.bg],
      ba: [...this.eskf.ba],
      R_wb: new Float64Array(this.eskf.R),
      p_wb: [...this.eskf.p],
      v_wb: [...this.eskf.v],
      predictedOnly: false,
      holdout: false,
      effect: `初始化：尺度 ${this.bootstrapResult?.scale?.toFixed(3) ?? '不可恢复'}，时差 ${(this.timeOffset * 1000).toFixed(1)} ms`,
      relocalised: false,
      biasInflated: false,
      flowResidualPx: 0,
      rotResidualDeg: 0,
    };
    this.frames.push(rec);
    this.poseHistory.push({ frame: this.frameIndex, R_wb: new Float64Array(this.eskf.R), p_wb: [...this.eskf.p] });
    return rec;
  }

  private emitFrame(
    t: number,
    trackCount: number,
    used: number,
    rejected: number,
    reproj: number[],
    holdout: boolean,
    effect: string,
    relocalised: boolean,
  ): KernelFrame {
    if (!this.eskf) throw new Error('no state');
    const sorted = [...reproj].sort((a, b) => a - b);
    const rms = reproj.length ? Math.sqrt(reproj.reduce((s, v) => s + v * v, 0) / reproj.length) : 0;
    const p95 = sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))] : 0;
    const rec: KernelFrame = {
      index: this.frameIndex,
      t,
      status: this.status,
      tracks: trackCount,
      used,
      rejectedGate: rejected,
      reprojRms: rms,
      reprojP95: p95,
      sigmaThetaDeg: this.eskf.sigmaTheta() * (180 / Math.PI),
      sigmaPos: this.eskf.sigmaPosition(),
      bg: [...this.eskf.bg],
      ba: [...this.eskf.ba],
      R_wb: new Float64Array(this.eskf.R),
      p_wb: [...this.eskf.p],
      v_wb: [...this.eskf.v],
      predictedOnly: holdout || used === 0,
      holdout,
      effect,
      relocalised,
      biasInflated: this.biasInflated,
      flowResidualPx: this.lastFlowResidual,
      rotResidualDeg: this.consecutiveRotErr.length ? this.consecutiveRotErr[this.consecutiveRotErr.length - 1] : 0,
    };
    this.frames.push(rec);
    if (!holdout && this.eskf) {
      // poseHistory is updated by updateTrackHistory for tracking frames; ensure
      // an entry exists for inertial frames too so relative updates can chain.
      const exists = this.poseHistory.some((p) => p.frame === this.frameIndex);
      if (!exists) {
        this.poseHistory.push({ frame: this.frameIndex, R_wb: new Float64Array(this.eskf.R), p_wb: [...this.eskf.p] });
        if (this.poseHistory.length > 600) this.poseHistory.shift();
      }
    }
    return rec;
  }

  /** Current best-estimate camera centre for the UI. */
  cameraCentre(): Vec3 | null {
    if (!this.eskf) return null;
    return cameraCenterWorld(this.eskf.R, this.eskf.p, this.extrinsics);
  }

  /** Pose to use when drawing overlays on the current frame. */
  currentPose(): { R_wb: Mat3; p_wb: Vec3 } | null {
    if (!this.eskf) return null;
    return { R_wb: new Float64Array(this.eskf.R), p_wb: [...this.eskf.p] };
  }

  mapLandmarks(): LandmarkRecord[] {
    return [...this.landmarks.values()];
  }

  counts(): { landmarks: number; relocalisations: number; rejections: number } {
    let rejections = 0;
    for (const f of this.frames) rejections += f.rejectedGate;
    return { landmarks: this.landmarks.size, relocalisations: this.relocalisations, rejections };
  }
}
