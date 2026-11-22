import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { computeDocExamples } from '../lens/docExamples';

// Regenerates the machine-checked example block in docs/sensor-conventions.md.
// The document text refers to values computed by src/lens/docExamples.ts; this
// CLI rewrites the JSON block between the EXAMPLES markers so the doc and the
// implementation cannot drift apart. tests/conventions.test.ts verifies it.

const BEGIN = '<!-- BEGIN EXAMPLES -->';
const END = '<!-- END EXAMPLES -->';

const docPath = resolve(process.cwd(), 'docs/sensor-conventions.md');
const text = readFileSync(docPath, 'utf8');
const b = text.indexOf(BEGIN);
const e = text.indexOf(END);
if (b < 0 || e < 0 || e < b) {
  console.error('docs/sensor-conventions.md 缺少 EXAMPLES 标记');
  process.exit(1);
}
const block = `${BEGIN}\n\`\`\`json\n${JSON.stringify(computeDocExamples(), null, 2)}\n\`\`\`\n${END}`;
const next = text.slice(0, b) + block + text.slice(e + END.length);
if (next === text) {
  console.log('docs/sensor-conventions.md 示例已是最新');
} else {
  writeFileSync(docPath, next);
  console.log('docs/sensor-conventions.md 示例已更新');
}
