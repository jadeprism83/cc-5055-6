import { Matrix, SingularValueDecomposition } from 'ml-matrix';
import { Mat3, Vec3, v3norm, v3sub, so3Log } from '../core/math';

// Camera-IMU rotation calibration (hand-eye, rotation only) from paired relative
// rotations: A_i = dR_body, B_i = dR_cam, with A_i X = X B_i and X = R_bc.
// Taking logs, X * log(B_i) = log(A_i), which is an orthogonal Procrustes problem.
// Translation of the extrinsic is NOT identifiable from rotation alone; it stays
// at the supplied value and the UI states that explicitly.

export interface HandEyePair {
  dR_body: Mat3;
  dR_cam: Mat3;
  /** Baseline between samples, used as a quality weight. */
  weight?: number;
}

export interface HandEyeResult {
  ok: boolean;
  R_bc: Mat3;
  pairsUsed: number;
  rotationResidualDeg: number;
  message: string;
}

export function solveHandEyeRotation(pairs: HandEyePair[]): HandEyeResult {
  const usable: Array<{ a: Vec3; b: Vec3; w: number }> = [];
  for (const p of pairs) {
    const a = so3Log(p.dR_body);
    const b = so3Log(p.dR_cam);
    const angle = v3norm(b);
    if (angle < 2 * (Math.PI / 180) || angle > 2.6) continue;
    const w = (p.weight ?? 1) * angle;
    usable.push({ a, b, w });
  }
  if (usable.length < 3) {
    return {
      ok: false,
      R_bc: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
      pairsUsed: usable.length,
      rotationResidualDeg: NaN,
      message: `手眼标定失败：只有 ${usable.length} 个足够旋转的配对（需要 ≥3）`,
    };
  }
  // Procrustes: find X minimising sum w |X b - a|^2.
  const H = new Float64Array(9);
  for (const u of usable) {
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) H[r * 3 + c] += u.w * u.b[r] * u.a[c];
    }
  }
  const M = new Matrix(3, 3);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) M.set(r, c, H[r * 3 + c]);
  const svd = new SingularValueDecomposition(M, { autoTranspose: true });
  let U = svd.leftSingularVectors;
  let V = svd.rightSingularVectors;
  const d = svd.diagonal;
  const detU = det3(U);
  const detV = det3(V);
  // X = V diag(1,1,det(V U^T)) U^T.
  const s3 = Math.sign(detV * detU) || 1;
  const D = Matrix.diag([1, 1, s3]);
  const X = V.mmul(D).mmul(U.transpose());
  const R = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) R[r * 3 + c] = X.get(r, c);
  // Residual over used pairs.
  let sse = 0;
  let wsum = 0;
  for (const u of usable) {
    const pred = apply3(R, u.b);
    const e = v3norm(v3sub(pred, u.a));
    sse += u.w * e * e;
    wsum += u.w;
  }
  const rms = wsum > 0 ? Math.sqrt(sse / wsum) : NaN;
  const ok = usable.length >= 3 && rms < 0.12;
  return {
    ok,
    R_bc: R,
    pairsUsed: usable.length,
    rotationResidualDeg: rms * (180 / Math.PI),
    message: ok
      ? `手眼旋转已求解：使用 ${usable.length} 对，残差 ${(rms * (180 / Math.PI)).toFixed(2)}°；平移外参仍沿用输入（旋转不可观测其平移）`
      : `手眼残差 ${(rms * (180 / Math.PI)).toFixed(2)}° 过大，拒绝该外参解`,
  };
}

function det3(m: Matrix): number {
  return (
    m.get(0, 0) * (m.get(1, 1) * m.get(2, 2) - m.get(1, 2) * m.get(2, 1)) -
    m.get(0, 1) * (m.get(1, 0) * m.get(2, 2) - m.get(1, 2) * m.get(2, 0)) +
    m.get(0, 2) * (m.get(1, 0) * m.get(2, 1) - m.get(1, 1) * m.get(2, 0))
  );
}

function apply3(R: Mat3, v: Vec3): Vec3 {
  return [R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2]];
}
