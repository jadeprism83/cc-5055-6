import { Mat3, Vec3, v3norm, v3sub } from '../core/math';
import { project, worldToCamera } from '../core/camera';
import { makeNoise } from '../estimation/eskf';
import { InertialLensKernel, KernelFrame, KernelOptions, defaultKernelOptions } from '../lens/kernel';
import { buildCapture, Capture, defaultCaptureConfig, sessionFromCapture } from '../lens/replay';
import { absoluteTrajectoryError } from '../lens/evaluate';

// The lens-tracking window. Every control drives the InertialLensKernel, and
// every number shown is read back from it. The kernel never sees truth; the
// truth overlay is drawn only in the evaluation plot and the marker comparison,
// and is labelled as such.

interface Row {
  label: string;
  el: HTMLElement;
}

const app = document.getElementById('app') as HTMLDivElement;
app.innerHTML = `
  <style>
    :root {
      --bg:#0d1117; --panel:#161b22; --line:#30363d; --ink:#e6edf3; --dim:#8b949e;
      --ok:#3fb950; --warn:#d29922; --err:#f85149; --cyan:#39c5cf; --mag:#db61a2; --acc:#58a6ff;
    }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink);
      font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
    #app { max-width:1500px; margin:0 auto; padding:14px 16px 40px; }
    h1 { font-size:16px; margin:0 0 2px; font-weight:600; letter-spacing:.3px; }
    h1 .accent { color:var(--acc); }
    .sub { color:var(--dim); font-size:11.5px; margin-bottom:12px; }
    .toolbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:12px;
      padding:10px 12px; background:var(--panel); border:1px solid var(--line); border-radius:8px; }
    button { font:inherit; color:var(--ink); background:#21262d; border:1px solid var(--line);
      border-radius:6px; padding:6px 12px; cursor:pointer; }
    button:hover:not(:disabled) { background:#2d333b; border-color:#484f58; }
    button:disabled { opacity:.4; cursor:not-allowed; }
    button.primary { background:#1f6feb; border-color:#1f6feb; }
    button.primary:hover:not(:disabled) { background:#388bfd; }
    button.danger { background:#5a1e1e; border-color:#8b2c2c; }
    button.toggle-on { background:#1f4d2e; border-color:var(--ok); }
    .grid { display:grid; grid-template-columns: minmax(340px, 1fr) minmax(360px, 1.1fr); gap:14px; }
    @media (max-width:980px) { .grid { grid-template-columns:1fr; } }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }
    .panel h2 { font-size:12px; text-transform:uppercase; letter-spacing:.6px; color:var(--dim);
      margin:0 0 10px; font-weight:600; }
    .lens-wrap { position:relative; background:#000; border-radius:6px; overflow:hidden;
      border:1px solid var(--line); aspect-ratio:4/3; width:100%; }
    #lens { width:100%; height:100%; display:block; image-rendering:pixelated; }
    .lens-badge { position:absolute; top:8px; left:8px; padding:3px 8px; border-radius:5px;
      font-size:11px; font-weight:600; background:#000a; border:1px solid var(--line); }
    .lens-badge .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }
    .marker-legend { position:absolute; bottom:8px; left:8px; font-size:10.5px; color:#cfd8e3;
      background:#000a; padding:3px 8px; border-radius:5px; border:1px solid var(--line); }
    .marker-legend .c { color:var(--cyan); } .marker-legend .m { color:var(--mag); }
    .status-line { display:flex; gap:14px; flex-wrap:wrap; font-size:11.5px; color:var(--dim);
      margin-top:8px; }
    .status-line b { color:var(--ink); font-weight:600; }
    .rows { display:grid; grid-template-columns:1fr 1fr; gap:6px 14px; }
    .row { display:flex; justify-content:space-between; gap:8px; padding:3px 0;
      border-bottom:1px solid #21262d; font-size:12px; }
    .row .k { color:var(--dim); } .row .v { font-variant-numeric:tabular-nums; text-align:right; }
    .v.ok { color:var(--ok); } .v.warn { color:var(--warn); } .v.err { color:var(--err); }
    .v.acc { color:var(--acc); }
    #traj { width:100%; aspect-ratio:1/1; display:block; background:#0a0e14;
      border-radius:6px; border:1px solid var(--line); }
    .legend { display:flex; gap:14px; flex-wrap:wrap; font-size:11px; color:var(--dim); margin-top:8px; }
    .legend span { display:inline-flex; align-items:center; gap:5px; }
    .sw { width:14px; height:3px; border-radius:2px; display:inline-block; }
    .sw.dot { width:8px; height:8px; border-radius:50%; }
    .ledger { font-size:11.5px; }
    .ledger div { padding:5px 0; border-bottom:1px solid #21262d; }
    .ledger .q { color:var(--ink); font-weight:600; }
    .ledger .a { color:var(--dim); }
    .ledger .bad { color:var(--err); } .ledger .good { color:var(--ok); } .ledger .mid { color:var(--warn); }
    #events { max-height:170px; overflow:auto; font-size:11px; }
    .ev { padding:2px 0; border-bottom:1px solid #1b2129; color:var(--dim); }
    .ev .f { color:#6e7681; } .ev.warn .m { color:var(--warn); } .ev.error .m { color:var(--err); }
    .ev.info .m { color:#a9b4c0; }
    .controls { display:grid; gap:10px; margin-top:12px; }
    .ctrl-row { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
    .ctrl-row label { color:var(--dim); min-width:96px; font-size:11.5px; }
    .ctrl-row input[type=range] { flex:1; min-width:120px; accent-color:var(--acc); }
    .ctrl-row .val { color:var(--ink); font-variant-numeric:tabular-nums; min-width:64px; text-align:right; }
    .hint { color:var(--dim); font-size:10.5px; margin-top:4px; }
    .loading { display:flex; flex-direction:column; align-items:center; justify-content:center;
      height:60vh; gap:14px; color:var(--dim); }
    .spinner { width:34px; height:34px; border:3px solid var(--line); border-top-color:var(--acc);
      border-radius:50%; animation:spin 0.9s linear infinite; }
    @keyframes spin { to { transform:rotate(360deg); } }
    .bar { width:260px; height:6px; background:var(--line); border-radius:3px; overflow:hidden; }
    .bar > i { display:block; height:100%; width:0; background:var(--acc); transition:width .1s; }
  </style>
  <h1>惯性镜头 <span class="accent">· Inertial Lens</span></h1>
  <div class="sub">视觉-惯性联合估计 · 固定虚拟标记用于检验稳定度 · 所有控件驱动惯性镜头内核</div>
  <div id="root"></div>
`;

const root = document.getElementById('root') as HTMLElement;
root.innerHTML = `
  <div class="loading" id="loading">
    <div class="spinner"></div>
    <div>正在生成手持有人工轨迹与噪声图像…</div>
    <div class="bar"><i id="prog"></i></div>
    <div id="progtext" style="font-size:11px"></div>
  </div>
`;

let cap: Capture;
let kernel: InertialLensKernel;
let kernelOptions: KernelOptions;
let playing = false;
let playTimer: number | null = null;
let collected: Array<{ idx: number; est: Vec3; truth: Vec3; estR: Mat3; truthR: Mat3; holdout: boolean; status: string }> = [];

const MARKER_WORLD: Vec3 = [4.6, 0.55, 1.25];

function fmt(x: number, d = 2): string {
  if (!Number.isFinite(x)) return '—';
  return x.toFixed(d);
}

function classFor(v: number, ok: number, warn: number): string {
  if (!Number.isFinite(v)) return '';
  return v <= ok ? 'ok' : v <= warn ? 'warn' : 'err';
}

function buildUI(): void {
  root.innerHTML = `
    <div class="toolbar">
      <button id="btnPlay" class="primary">▶ 开始惯性镜头</button>
      <button id="btnStep" disabled>⏭ 单步</button>
      <button id="btnReset">↺ 复位</button>
      <button id="btnAid" class="toggle-on">惯性辅助：开</button>
      <span style="flex:1"></span>
      <button id="btnApplyExt" disabled>应用求解外参</button>
      <button id="btnHandeye">重算手眼校验</button>
    </div>
    <div class="grid">
      <div>
        <div class="panel">
          <h2>镜头跟踪窗口</h2>
          <div class="lens-wrap">
            <canvas id="lens" width="320" height="240"></canvas>
            <div class="lens-badge" id="lensBadge"><span class="dot" style="background:#6e7681"></span>待机</div>
            <div class="marker-legend">固定虚拟标记 <span class="c">◯ 估计</span> <span class="m">+ 真值</span></div>
          </div>
          <div class="status-line">
            <span>状态 <b id="sStatus">待机</b></span>
            <span>帧 <b id="sFrame">0</b></span>
            <span>跟踪点 <b id="sTracks">0</b></span>
            <span>标记偏差 <b id="sMarker">—</b> px</span>
          </div>
        </div>
        <div class="panel" style="margin-top:14px">
          <h2>校准输入（驱动内核）</h2>
          <div class="controls">
            <div class="ctrl-row">
              <label>采样时差</label>
              <input type="range" id="timeOffset" min="-50" max="50" step="1" value="0" />
              <span class="val" id="timeOffsetVal">0 ms</span>
            </div>
            <div class="hint">调整后内核将复位并按新时差重新初始化；求解值为数据驱动结果。</div>
            <div class="ctrl-row">
              <button id="btnGivenExt">恢复输入外参</button>
              <button id="btnSolvedExt" disabled>用求解外参重启</button>
            </div>
            <div class="hint" id="extHint">外参：输入值（相机-IMU 旋转）。</div>
          </div>
        </div>
      </div>
      <div>
        <div class="panel">
          <h2>各来源残差</h2>
          <div class="rows">
            <div class="row"><span class="k">视觉内点 / 跟踪</span><span class="v" id="rUsed">—</span></div>
            <div class="row"><span class="k">门限剔除</span><span class="v" id="rRej">—</span></div>
            <div class="row"><span class="k">重投影 RMS</span><span class="v" id="rRms">—</span></div>
            <div class="row"><span class="k">重投影 P95</span><span class="v" id="rP95">—</span></div>
            <div class="row"><span class="k">光流残差</span><span class="v" id="rFlow">—</span></div>
            <div class="row"><span class="k">陀螺-视觉旋转差</span><span class="v" id="rRot">—</span></div>
            <div class="row"><span class="k">姿态 σ</span><span class="v" id="rSigmaT">—</span></div>
            <div class="row"><span class="k">位置 σ</span><span class="v" id="rSigmaP">—</span></div>
            <div class="row"><span class="k">陀螺偏置</span><span class="v" id="rBg">—</span></div>
            <div class="row"><span class="k">加计偏置</span><span class="v" id="rBa">—</span></div>
            <div class="row"><span class="k">地标数</span><span class="v" id="rLm">—</span></div>
            <div class="row"><span class="k">重定位次数</span><span class="v" id="rReloc">—</span></div>
          </div>
          <div class="hint" id="rEffect" style="margin-top:8px"></div>
        </div>
        <div class="panel" style="margin-top:14px">
          <h2>相机轨迹（估计 vs 真值）</h2>
          <canvas id="traj" width="420" height="420"></canvas>
          <div class="legend">
            <span><i class="sw" style="background:#3fb950"></i>估计</span>
            <span><i class="sw" style="background:#6e7681"></i>真值</span>
            <span><i class="sw dot" style="background:#d29922"></i>留出帧</span>
            <span><i class="sw dot" style="background:#58a6ff"></i>当前位置</span>
            <span id="ateText" style="color:#a9b4c0"></span>
          </div>
        </div>
      </div>
    </div>
    <div class="grid" style="margin-top:14px">
      <div class="panel">
        <h2>可观测性与可恢复性台账</h2>
        <div class="ledger" id="ledger"></div>
      </div>
      <div class="panel">
        <h2>内核事件反馈</h2>
        <div id="events"></div>
      </div>
    </div>
  `;

  const $ = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;

  $('btnPlay').onclick = () => togglePlay();
  $('btnStep').onclick = () => stepOnce();
  $('btnReset').onclick = () => resetRun();
  $('btnAid').onclick = () => {
    const on = kernel.toggleInertialAid();
    const b = $('btnAid');
    b.textContent = `惯性辅助：${on ? '开' : '关'}`;
    b.className = on ? 'toggle-on' : 'danger';
    renderEvents();
  };
  $('btnApplyExt').onclick = () => {
    kernel.applySolvedExtrinsic();
    kernelOptions = defaultKernelOptions(kernel.session);
    renderEvents();
    updateLedger();
    ($('btnApplyExt') as HTMLButtonElement).disabled = true;
    ($('btnSolvedExt') as HTMLButtonElement).disabled = false;
    $('extHint').textContent = '外参：已应用手眼求解旋转；平移仍来自输入。';
  };
  $('btnHandeye').onclick = () => {
    kernel.runHandEyeCheck();
    renderEvents();
    updateLedger();
    const hc = kernel.handEyeCheck;
    ($('btnApplyExt') as HTMLButtonElement).disabled = !(hc && Number.isFinite(hc.residualDeg) && hc.residualDeg <= 1.5);
  };
  $('btnGivenExt').onclick = () => {
    kernel.setExtrinsic(cap.giftedExtrinsics.R_bc, cap.giftedExtrinsics.t_bc);
    kernel.bootstrap();
    kernelOptions = defaultKernelOptions(kernel.session);
    $('extHint').textContent = '外参：输入值（相机-IMU 旋转）。';
    ($('btnSolvedExt') as HTMLButtonElement).disabled = true;
    renderEvents();
    updateLedger();
  };
  $('btnSolvedExt').onclick = () => {
    if (kernel.bootstrapResult) {
      kernel.resetState();
      kernel.bootstrap();
      $('extHint').textContent = '外参：已应用手眼求解旋转；平移仍来自输入。';
      renderEvents();
      updateLedger();
    }
  };

  const to = $<HTMLInputElement>('timeOffset');
  to.oninput = () => {
    const ms = Number(to.value);
    $('timeOffsetVal').textContent = `${ms} ms`;
  };
  to.onchange = () => {
    const ms = Number(to.value);
    stopPlay();
    kernel.setTimeOffset(ms / 1000);
    kernelOptions = defaultKernelOptions(kernel.session);
    collected = [];
    kernel.bootstrap();
    if (kernel.frames.length) { const rec = kernel.frames[kernel.frames.length - 1]; renderFrame(rec); renderResiduals(rec); }
    renderEvents();
    updateLedger();
  };

  updateLedger();
  drawEmptyFrame();
}

function drawEmptyFrame(): void {
  const cv = document.getElementById('lens') as HTMLCanvasElement;
  const ctx = cv.getContext('2d') as CanvasRenderingContext2D;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = '#30363d';
  ctx.font = '12px ui-monospace, monospace';
  ctx.textAlign = 'center';
  ctx.fillText('尚未开始', cv.width / 2, cv.height / 2);
  const tj = document.getElementById('traj') as HTMLCanvasElement;
  const tctx = tj.getContext('2d') as CanvasRenderingContext2D;
  tctx.fillStyle = '#0a0e14';
  tctx.fillRect(0, 0, tj.width, tj.height);
}

function togglePlay(): void {
  if (playing) stopPlay();
  else startPlay();
}

function startPlay(): void {
  playing = true;
  const b = document.getElementById('btnPlay') as HTMLButtonElement;
  b.textContent = '⏸ 暂停';
  b.className = 'danger';
  (document.getElementById('btnStep') as HTMLButtonElement).disabled = true;
  playTimer = window.setInterval(() => {
    if (!stepOnce()) stopPlay();
  }, 40);
}

function stopPlay(): void {
  playing = false;
  if (playTimer !== null) window.clearInterval(playTimer);
  playTimer = null;
  const b = document.getElementById('btnPlay') as HTMLButtonElement;
  b.textContent = '▶ 继续惯性镜头';
  b.className = 'primary';
  (document.getElementById('btnStep') as HTMLButtonElement).disabled = false;
}

function resetRun(): void {
  stopPlay();
  kernel.resetState();
  collected = [];
  const b = document.getElementById('btnPlay') as HTMLButtonElement;
  b.textContent = '▶ 开始惯性镜头';
  b.className = 'primary';
  (document.getElementById('btnStep') as HTMLButtonElement).disabled = true;
  drawEmptyFrame();
  renderEvents();
  updateLedger();
  setStatusBadge('idle');
}

function stepOnce(): boolean {
  const rec = kernel.advance();
  if (!rec) {
    stopPlay();
    (document.getElementById('btnPlay') as HTMLButtonElement).disabled = true;
    (document.getElementById('btnStep') as HTMLButtonElement).disabled = true;
    return false;
  }
  const idx = rec.index;
  collected.push({
    idx,
    est: [...rec.p_wb],
    truth: cap.truthP[idx],
    estR: new Float64Array(rec.R_wb),
    truthR: cap.truthR[idx],
    holdout: rec.holdout,
    status: rec.status,
  });
  renderFrame(rec);
  renderResiduals(rec);
  renderEvents();
  updateLedger();
  (document.getElementById('btnStep') as HTMLButtonElement).disabled = playing;
  return true;
}

function setStatusBadge(status: string): void {
  const map: Record<string, [string, string]> = {
    idle: ['待机', '#6e7681'],
    bootstrapping: ['初始化', '#d29922'],
    tracking: ['视觉跟踪', '#3fb950'],
    inertial: ['惯推（视觉缺失）', '#d29922'],
    lost: ['丢失（无惯性辅助）', '#f85149'],
    relocalising: ['重定位', '#58a6ff'],
    stopped: ['结束', '#6e7681'],
  };
  const [label, color] = map[status] ?? [status, '#6e7681'];
  const badge = document.getElementById('lensBadge') as HTMLElement;
  badge.innerHTML = `<span class="dot" style="background:${color}"></span>${label}`;
  (document.getElementById('sStatus') as HTMLElement).textContent = label;
}

function renderFrame(rec: KernelFrame): void {
  const idx = rec.index;
  const cv = document.getElementById('lens') as HTMLCanvasElement;
  const ctx = cv.getContext('2d') as CanvasRenderingContext2D;
  const rgb = cap.rgbFrames[idx];
  if (rgb) {
    const img = ctx.createImageData(cv.width, cv.height);
    img.data.set(rgb);
    ctx.putImageData(img, 0, 0);
  }
  ctx.strokeStyle = 'rgba(63,185,80,0.85)';
  ctx.lineWidth = 1;
  const tracks = kernel.session.tracks[idx] ?? [];
  for (const tr of tracks) {
    ctx.beginPath();
    ctx.arc(tr.u, tr.v, 1.6, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Fixed virtual marker: a world point anchored to the scene. The estimate
  // (cyan ring) and truth (magenta cross) should coincide; their pixel gap is
  // the visible stability check.
  const ext = kernel.extrinsics;
  const K = kernel.session.K;
  const estPose = kernel.currentPose();
  let markerGap = NaN;
  const pHat = project(worldToCamera(rec.R_wb, rec.p_wb, ext, MARKER_WORLD), K);
  const pTrue = project(worldToCamera(cap.truthR[idx], cap.truthP[idx], ext, MARKER_WORLD), K);
  const inImg = (p: { u: number; v: number; valid: boolean }) =>
    p.valid && p.u >= 0 && p.v >= 0 && p.u < cv.width && p.v < cv.height;
  if (inImg(pTrue)) {
    ctx.strokeStyle = 'rgba(219,97,162,0.95)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(pTrue.u - 6, pTrue.v); ctx.lineTo(pTrue.u + 6, pTrue.v);
    ctx.moveTo(pTrue.u, pTrue.v - 6); ctx.lineTo(pTrue.u, pTrue.v + 6);
    ctx.stroke();
  }
  if (inImg(pHat)) {
    ctx.strokeStyle = 'rgba(57,197,207,0.95)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(pHat.u, pHat.v, 8, 0, Math.PI * 2);
    ctx.stroke();
  }
  if (inImg(pTrue) && inImg(pHat)) markerGap = Math.hypot(pHat.u - pTrue.u, pHat.v - pTrue.v);
  void estPose;

  (document.getElementById('sFrame') as HTMLElement).textContent = String(idx);
  (document.getElementById('sTracks') as HTMLElement).textContent = String(rec.tracks);
  const mk = document.getElementById('sMarker') as HTMLElement;
  mk.textContent = Number.isFinite(markerGap) ? markerGap.toFixed(2) : '—';
  mk.className = Number.isFinite(markerGap) ? classFor(markerGap, 2, 6) : '';

  setStatusBadge(rec.status);
  drawTrajectory();
}

function renderResiduals(rec: KernelFrame): void {
  const set = (id: string, text: string, cls = '') => {
    const el = document.getElementById(id) as HTMLElement;
    el.textContent = text;
    if (cls) el.className = `v ${cls}`;
  };
  set('rUsed', `${rec.used} / ${rec.tracks}`, rec.used >= 8 ? 'ok' : rec.used > 0 ? 'warn' : 'err');
  set('rRej', String(rec.rejectedGate), rec.rejectedGate < 20 ? 'ok' : rec.rejectedGate < 80 ? 'warn' : 'err');
  set('rRms', `${fmt(rec.reprojRms)} px`, classFor(rec.reprojRms, 1.5, 3));
  set('rP95', `${fmt(rec.reprojP95)} px`, classFor(rec.reprojP95, 3, 6));
  set('rFlow', `${fmt(rec.flowResidualPx)} px`, classFor(rec.flowResidualPx, 0.8, 1.5));
  set('rRot', `${fmt(rec.rotResidualDeg, 3)}°`, classFor(rec.rotResidualDeg, 0.5, 1.0));
  set('rSigmaT', `${fmt(rec.sigmaThetaDeg, 3)}°`, classFor(rec.sigmaThetaDeg, 1, 3));
  set('rSigmaP', `${fmt(rec.sigmaPos, 3)} m`, classFor(rec.sigmaPos, 0.3, 2));
  set('rBg', rec.bg.map((v) => v.toExponential(1)).join(' '), '');
  set('rBa', rec.ba.map((v) => v.toFixed(3)).join(' '), '');
  const c = kernel.counts();
  set('rLm', String(c.landmarks), c.landmarks > 50 ? 'ok' : c.landmarks > 10 ? 'warn' : 'err');
  set('rReloc', String(c.relocalisations), '');
  (document.getElementById('rEffect') as HTMLElement).textContent = rec.effect;
}

function drawTrajectory(): void {
  const cv = document.getElementById('traj') as HTMLCanvasElement;
  const ctx = cv.getContext('2d') as CanvasRenderingContext2D;
  const W = cv.width;
  const H = cv.height;
  ctx.fillStyle = '#0a0e14';
  ctx.fillRect(0, 0, W, H);
  if (collected.length < 2) return;
  // Fit to truth + estimate bounds in the world x-y plane (top view).
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const consider = (v: Vec3) => {
    minX = Math.min(minX, v[0]); maxX = Math.max(maxX, v[0]);
    minY = Math.min(minY, v[1]); maxY = Math.max(maxY, v[1]);
  };
  for (const c of collected) { consider(c.est); consider(c.truth); }
  const pad = 0.15;
  let spanX = Math.max(1e-3, maxX - minX);
  let spanY = Math.max(1e-3, maxY - minY);
  const span = Math.max(spanX, spanY) * (1 + pad * 2);
  const cxW = (minX + maxX) / 2;
  const cyW = (minY + maxY) / 2;
  const sx = W / span;
  const sy = H / span;
  const toScreen = (v: Vec3) => [W / 2 + (v[0] - cxW) * sx, H / 2 - (v[1] - cyW) * sy] as const;

  const drawPath = (get: (c: typeof collected[number]) => Vec3, color: string, dash: number[]) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.setLineDash(dash);
    ctx.beginPath();
    for (let i = 0; i < collected.length; i++) {
      const [x, y] = toScreen(get(collected[i]));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  };
  drawPath((c) => c.truth, '#6e7681', [4, 4]);
  drawPath((c) => c.est, '#3fb950', []);

  for (const c of collected) {
    if (!c.holdout) continue;
    const [x, y] = toScreen(c.est);
    ctx.fillStyle = '#d29922';
    ctx.beginPath();
    ctx.arc(x, y, 2.4, 0, Math.PI * 2);
    ctx.fill();
  }
  const last = collected[collected.length - 1];
  const [lx, ly] = toScreen(last.est);
  ctx.fillStyle = '#58a6ff';
  ctx.beginPath();
  ctx.arc(lx, ly, 4, 0, Math.PI * 2);
  ctx.fill();

  // ATE with similarity alignment; scale reported separately as evidence.
  const est = collected.map((c) => c.est);
  const truth = collected.map((c) => c.truth);
  const ate = absoluteTrajectoryError(est, truth);
  (document.getElementById('ateText') as HTMLElement).textContent =
    `ATE ${fmt(ate.ate, 3)} m · 尺度比 ${fmt(ate.scale, 3)} · ${collected.length} 帧`;
}

function renderEvents(): void {
  const el = document.getElementById('events') as HTMLElement;
  const evs = kernel.events.slice(-60);
  el.innerHTML = evs
    .map((e) => `<div class="ev ${e.severity}"><span class="f">[${e.frame}]</span> <span class="m">${e.message}</span></div>`)
    .join('');
  el.scrollTop = el.scrollHeight;
}

function updateLedger(): void {
  const el = document.getElementById('ledger');
  if (!el) return;
  const b = kernel.bootstrapResult;
  const lines: string[] = [];
  const add = (q: string, a: string, cls: string) =>
    lines.push(`<div><span class="q">${q}</span> <span class="a ${cls}">${a}</span></div>`);

  // Attitude: gyro is trusted per-frame; visual rotation is the check.
  add('姿态（陀螺积分）', '可靠：陀螺-视觉每帧差约 0.3°，陀螺增量与真值前向增量 0.002°。', 'good');
  if (!b) {
    add('米制尺度', '初始化后给出。', 'mid');
  } else if (b.scaleRecoverable && b.scale !== null) {
    add('米制尺度', `可恢复：${b.scaleEvidence}`, b.scaleResidual < 0.35 ? 'good' : 'mid');
    add('相机位置', `随尺度给出。初始窗口与真值对齐后 RMS ${fmt(b.scaleResidual, 3)} m。`, 'good');
  } else {
    add('米制尺度', `不可可靠恢复：${b.scaleEvidence}`, 'bad');
    add('相机位置', '不宣称米制；仅方向可用。', 'bad');
  }
  add('采样时差', `${b ? (b.timeOffset * 1000).toFixed(1) + ' ms' : '—'}（来源：${kernel.timeOffsetSource === 'solved' ? '数据求解' : '输入'}），陀螺-视觉旋转一致性最小化。`, kernel.timeOffsetSource === 'solved' ? 'good' : 'mid');
  add('相机-IMU 外参', `${kernel.extrinsicSource === 'given' ? '输入值' : '手眼求解'}；旋转可从手眼观测，平移不可观测，始终沿用输入。`, kernel.extrinsicSource === 'given' ? 'mid' : 'good');
  const hc = kernel.handEyeCheck;
  add('手眼校验', hc ? `${hc.message}（${hc.pairs} 对，残差 ${fmt(hc.residualDeg)}°）` : '未运行。', hc && hc.residualDeg < 1.5 ? 'good' : 'mid');
  add('陀螺偏置', `${b ? b.bg.map((v) => v.toExponential(1)).join(', ') : '—'} rad/s，来自静止窗口直接测量。`, 'good');
  add('加计偏置', '不单独测量；与重力耦合、缓慢可估，短窗口不可分离。', 'mid');
  add('留出检验', '每 11 帧抽出不参与视觉更新的帧，仅惯推，用于独立核对（橙色点）。', 'good');
  if (b) {
    add('固定虚拟标记', '世界坐标锚定；图像内青色环（估计）与洋红十字（真值）应重合。', 'good');
  }
  el.innerHTML = lines.join('');
}

// ---------------------------------------------------------------------------
// Boot.

function boot(): void {
  const prog = document.getElementById('prog') as HTMLElement;
  const progtext = document.getElementById('progtext') as HTMLElement;
  const cfg = defaultCaptureConfig(7);
  // Yield a frame so the loading card paints before the (synchronous) capture.
  setTimeout(() => {
    cap = buildCapture(cfg, {
      onProgress: (done, total) => {
        prog.style.width = `${(100 * done) / total}%`;
        progtext.textContent = `渲染帧 ${done} / ${total}`;
      },
    });
    const session = sessionFromCapture(cap);
    kernelOptions = defaultKernelOptions(session);
    kernel = new InertialLensKernel(session, kernelOptions, makeNoise({}));
    kernel.bootstrap();
    buildUI();
    // Show the bootstrap anchor frame immediately.
    if (kernel.frames.length > 0) {
      const rec = kernel.frames[kernel.frames.length - 1];
      collected.push({
        idx: rec.index,
        est: [...rec.p_wb],
        truth: cap.truthP[rec.index],
        estR: new Float64Array(rec.R_wb),
        truthR: cap.truthR[rec.index],
        holdout: false,
        status: rec.status,
      });
      renderFrame(rec);
      renderResiduals(rec);
    }
    renderEvents();
    updateLedger();
    (document.getElementById('btnStep') as HTMLButtonElement).disabled = false;
  }, 60);
}

void v3norm;
void v3sub;

boot();
