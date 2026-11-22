import { Vec3, rotToEulerZYX } from '../core/math';
import { cameraToBody, defaultExtrinsics, defaultIntrinsics, projectWorld } from '../core/camera';
import {
  accelFromBody,
  converterBetween,
  convertGyroSample,
  firmwareA,
  firmwareB,
  gyroFromBody,
  gyroToBody,
  invertConverter,
} from '../core/conventions';
import { imuSlice } from '../estimation/imu';
import { ImuSample } from '../sim/trajectory';
import { sinusoidMotion } from '../sim/analytic';
import { defaultCalibrationRecord } from './record';
import { ShutterCompensator } from './shutter';

// The numbers quoted in docs/sensor-conventions.md are computed here, not
// written by hand. `npm run docs:conventions` regenerates the JSON block in
// the document from computeDocExamples(), and tests/conventions.test.ts fails
// if the block and the code disagree.

const r6 = (x: number): number => {
  const r = Math.round(x * 1e6) / 1e6;
  // JSON has no -0; keep the block and the recomputation identical.
  return r === 0 ? 0 : r;
};
const v6 = (v: Vec3): Vec3 => [r6(v[0]), r6(v[1]), r6(v[2])];
const m6 = (m: Float64Array): number[] => [...m].map(r6);

export interface DocExamples {
  extrinsics: {
    R_bc: number[];
    t_bc: Vec3;
    camPoint: Vec3;
    bodyPoint: Vec3;
  };
  firmwareB: {
    omegaBodyRadS: Vec3;
    rawGyroDegS: Vec3;
    roundTripRadS: Vec3;
    accelBody: Vec3;
    rawAccelG: Vec3;
  };
  converterAB: {
    rawA: Vec3;
    asB: Vec3;
    backToA: Vec3;
    R: number[];
    gyroScale: number;
  };
  timestamps: {
    cameraT0: number;
    cameraT1: number;
    timeOffset: number;
    rawWindowStart: number;
    rawWindowEnd: number;
    sliceSamples: number;
    sliceFirstT: number;
    sliceLastT: number;
  };
  attitudeFigure: {
    t: number;
    eulerDeg: Vec3;
    omegaBody: Vec3;
    accelBody: Vec3;
  };
  reprojectionFigure: {
    pointWorld: Vec3;
    u: number;
    v: number;
    depth: number;
    omegaBody: Vec3;
    readoutPerRowUs: number;
    deskewedU: number;
    deskewedV: number;
  };
}

export function computeDocExamples(): DocExamples {
  const ext = defaultExtrinsics();
  const K = defaultIntrinsics();
  const fwA = firmwareA();
  const fwB = firmwareB();

  // Extrinsics: a camera-frame point mapped into the body frame.
  const camPoint: Vec3 = [0.1, 0.2, 2.0];
  const bodyPoint = cameraToBody(ext, camPoint);

  // Firmware B units/axes: one body rate expressed as the raw fw-B reading.
  const omegaBody: Vec3 = [0.1, -0.2, 0.3];
  const rawGyro = gyroFromBody(fwB, omegaBody);
  const roundTrip = gyroToBody(fwB, rawGyro);
  const accelBody: Vec3 = [0.5, -0.3, 9.80665];
  const rawAccel = accelFromBody(fwB, accelBody);

  // Invertible converter fw-A → fw-B.
  const conv = converterBetween(fwA, fwB);
  const rawA: Vec3 = [0.05, -0.1, 0.2];
  const asB = convertGyroSample(conv, rawA);
  const backToA = convertGyroSample(invertConverter(conv), asB);

  // Timestamps: physical time = raw stamp + timeOffset.
  const timeOffset = 0.018;
  const cameraT0 = 1.0;
  const cameraT1 = 1.05;
  const imu: ImuSample[] = [];
  for (let k = 0; k <= 220; k++) {
    const t = 0.97 + k * 0.0005;
    imu.push({ t, gyro: [0.01, 0.02, -0.01], accel: [0, 0, 9.80665] });
  }
  const slice = imuSlice(imu, cameraT0, cameraT1, timeOffset);

  // Attitude figure from the analytic motion generator.
  const motion = sinusoidMotion();
  const tFig = 1.25;
  const euler = rotToEulerZYX(motion.R(tFig));
  const eulerDeg: Vec3 = [(euler[0] * 180) / Math.PI, (euler[1] * 180) / Math.PI, (euler[2] * 180) / Math.PI];

  // Reprojection figure: a world point 3 m ahead of the camera, plus the
  // rolling-shutter de-skew of its pixel with the same committed record.
  const R_wb = motion.R(tFig);
  const p_wb = motion.p(tFig);
  const center: Vec3 = [
    p_wb[0] + R_wb[0] * ext.t_bc[0] + R_wb[1] * ext.t_bc[1] + R_wb[2] * ext.t_bc[2],
    p_wb[1] + R_wb[3] * ext.t_bc[0] + R_wb[4] * ext.t_bc[1] + R_wb[5] * ext.t_bc[2],
    p_wb[2] + R_wb[6] * ext.t_bc[0] + R_wb[7] * ext.t_bc[1] + R_wb[8] * ext.t_bc[2],
  ];
  const fwd: Vec3 = [
    R_wb[0] * ext.R_bc[2] + R_wb[1] * ext.R_bc[5] + R_wb[2] * ext.R_bc[8],
    R_wb[3] * ext.R_bc[2] + R_wb[4] * ext.R_bc[5] + R_wb[5] * ext.R_bc[8],
    R_wb[6] * ext.R_bc[2] + R_wb[7] * ext.R_bc[5] + R_wb[8] * ext.R_bc[8],
  ];
  const pointWorld: Vec3 = [center[0] + 3 * fwd[0], center[1] + 3 * fwd[1], center[2] + 3 * fwd[2]];
  const px = projectWorld(R_wb, p_wb, ext, K, pointWorld);
  const readoutPerRowSec = 25e-6;
  const record = defaultCalibrationRecord(ext, timeOffset);
  const shutter = new ShutterCompensator(record, readoutPerRowSec);
  const omegaFig = motion.omegaB(tFig);
  const fixed = shutter.deskew(px.u, px.v, K, omegaFig);

  return {
    extrinsics: {
      R_bc: m6(ext.R_bc),
      t_bc: v6(ext.t_bc),
      camPoint: v6(camPoint),
      bodyPoint: v6(bodyPoint),
    },
    firmwareB: {
      omegaBodyRadS: v6(omegaBody),
      rawGyroDegS: v6(rawGyro),
      roundTripRadS: v6(roundTrip),
      accelBody: v6(accelBody),
      rawAccelG: v6(rawAccel),
    },
    converterAB: {
      rawA: v6(rawA),
      asB: v6(asB),
      backToA: v6(backToA),
      R: m6(conv.R),
      gyroScale: r6(conv.gyroScale),
    },
    timestamps: {
      cameraT0: r6(cameraT0),
      cameraT1: r6(cameraT1),
      timeOffset: r6(timeOffset),
      rawWindowStart: r6(cameraT0 - timeOffset),
      rawWindowEnd: r6(cameraT1 - timeOffset),
      sliceSamples: slice.samples.length,
      sliceFirstT: r6(slice.samples[0]?.t ?? NaN),
      sliceLastT: r6(slice.samples[slice.samples.length - 1]?.t ?? NaN),
    },
    attitudeFigure: {
      t: r6(tFig),
      eulerDeg: v6(eulerDeg),
      omegaBody: v6(motion.omegaB(tFig)),
      accelBody: v6(motion.accelB(tFig)),
    },
    reprojectionFigure: {
      pointWorld: v6(pointWorld),
      u: r6(px.u),
      v: r6(px.v),
      depth: r6(px.depth),
      omegaBody: v6(omegaFig),
      readoutPerRowUs: r6(readoutPerRowSec * 1e6),
      deskewedU: r6(fixed.u),
      deskewedV: r6(fixed.v),
    },
  };
}
