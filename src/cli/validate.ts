import { runInertialValidation } from '../lens/validation';

// Inertial-data validation CLI. Assists the inertial lens; it never replaces
// the graphical window. Runs standalone in a clean directory:
//   npm run validate
// Exit code 0 = every check passed, 1 = at least one failed.

const report = runInertialValidation((c, i, n) => {
  const tag = c.ok ? 'PASS' : 'FAIL';
  console.log(`[${i + 1}/${n}] ${tag}  ${c.title}`);
  console.log(`       ${c.detail}`);
});

console.log('');
for (const c of report.checks) {
  const m = Object.entries(c.metrics)
    .map(([k, v]) => `${k}=${Number.isFinite(v) ? v : '—'}`)
    .join(' ');
  console.log(`${c.ok ? '✓' : '✗'} ${c.id}: ${m}`);
}
console.log('');
console.log(`惯性资料校验${report.ok ? '全部通过' : '存在失败项'}（${report.checks.length} 项，${(report.durationMs / 1000).toFixed(1)} s）`);

process.exit(report.ok ? 0 : 1);
