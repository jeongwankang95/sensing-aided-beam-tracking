"""missing_rescoring_compare.py: every paired estimate ('x [lo, hi]' triple) in the Scenario-31 to 33 logs, as reported
(results/analysis/) against rescored without frames that have a missing reading (results/analysis/missing_rescoring/, written by
analysis/missing_rescoring.py); lists the changes of at least 0.01 in a point estimate and the largest change per log.
Paper: supplementary Sec. S4-A (no system gain or marginal value moves by more than 0.02 dB; the controller against the
unverified or always-on variants changes by up to 0.07 dB; the rescored ranges of the timing conditions).
usage: python analysis/missing_rescoring_compare.py"""
import re, os
A = os.path.join('results', 'analysis'); B = os.path.join(A, 'missing_rescoring')
NAMES = ['arms', 'attribution', 'coverage', 'decimation', 'stale_observations', 'exceedance', 'loso', 'invocation_ablation',
         'controls', 'controls_gate', 'warmup', 'delays', 'delays_sensor_off', 'decimation_slack', 'sensitivity', 'table2',
         'decimation_sensor_off', 'factorial', 'table3']
tri = re.compile(r'([-+]?\d+\.\d+)\s*(?:dB)?\s*(?:95%\s*)?\[\s*([-+]?\s*\d+\.\d+),\s*([-+]?\s*\d+\.\d+)\]')
lab = re.compile(r'([A-Za-z][^|\[\]]{0,60}?)\s*[-+]?\d+\.\d+\s*(?:dB)?\s*(?:95%\s*)?\[')
def lines(p): return [l.rstrip('\n') for l in open(p) if not l.startswith('[missing_rescoring]')]
mx = {}
for s in (31, 32, 33):
    pairs = [(k, f'{A}/{k}_s{s}.log', f'{B}/{k}_s{s}.log') for k in NAMES]
    pairs += [('seeds_s%d' % sd, f'{A}/seeds_s{sd}_s{s}.log', f'{B}/seeds_s{sd}_s{s}.log') for sd in (1, 2)]
    for k, a, b in pairs:
        A_, B_ = lines(a), lines(b)
        hdr = ''
        for x, y in zip(A_, B_):
            if not tri.search(x): hdr = x.strip()[:70]; continue
            ta, tb = tri.findall(x), tri.findall(y)
            if len(ta) != len(tb): print('S%d %s MISMATCH %s' % (s, k, x[:80])); continue
            labels = lab.findall(x)
            for i, (u, v) in enumerate(zip(ta, tb)):
                d = float(v[0]) - float(u[0])
                key = (s, k); mx[key] = max(mx.get(key, 0), abs(d))
                if abs(d) >= 0.01:
                    print('S%d %-9s %+.3f -> %+.3f (%+.3f) [%s,%s]->[%s,%s] | %s :: %s' % (s, k, float(u[0]), float(v[0]), d, u[1], u[2], v[1], v[2],
                          (labels[i] if i < len(labels) else '?').strip()[-45:], hdr[:50]))
print('max |delta| of paired estimates per log:')
print(' '.join('S%d:%s=%.3f' % (s, k, v) for (s, k), v in sorted(mx.items())))
