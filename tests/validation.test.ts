import { describe, expect, it } from 'vitest';
import { runInertialValidation } from '../src/lens/validation';

// The inertial-data validation report is itself part of the delivered checks:
// it must be green, and every check must compute from generated data.

describe('inertial-data validation report', () => {
  it('passes all maintenance checks', () => {
    const report = runInertialValidation();
    expect(report.checks.length).toBe(10);
    for (const c of report.checks) {
      expect(c.ok, `${c.id}: ${c.detail}`).toBe(true);
    }
    expect(report.ok).toBe(true);
  }, 300_000);
});
