import os, sys
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")
exec(compile(open(_CORE).read(), _CORE, "exec"))   # same namespace as the original prelude3.py + tail concatenation

# ---- Option 3 (2026-10-02): one-at-a-time perturbation of the re-selected constants (loss AND probes, A-M4 fix 2) ----
def _trk(slack):
    return Ctl(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=slack)
def score(pol):
    cs, ks, sf = [], [], 0
    for P, keys in passes:
        pol.reset(int(P[0].argmax()), float(P[0].max()))
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: cs.append(float(P[t, a]) / m); ks.append(k)
        sf += getattr(pol, 'sense', 0) + getattr(pol, 'trial', 0)
    c = np.array(cs)
    return float(np.mean(-10 * np.log10(np.clip(c, 1e-6, None)))), float(np.mean(ks)), 100 * sf / len(c)
assert 'rs_k' in K
GRID = [('th_on', [0.6, 1.0]), ('th_off', [0.3, 0.65]), ('margin', [0.25, 1.0]), ('t_trial', [1, 4]),
        ('bl', [64, 512]), ('verify', [2, 8]), ('slack', [0.4, 0.8]), ('rs_k', [2, 5]), ('rs_how', ['full'])]
dep = score(Ctl(**K)); fb = score(_trk(TSL))
print('\n=== %s (%d test passes) ===' % (root, len(passes)))
print('selected %.3f dB @ %.2f probes (%.1f%% model)   feedback-only %.3f dB @ %.2f' % (dep + fb[:2]))
rows = []
for k, vals in GRID:
    for v in vals:
        cfg = dict(K); cfg[k] = v; L, pr, mdl = score(Ctl(**cfg)); rows.append((f'{k}={v}', L, L - dep[0], pr, mdl))
for nm, L, d, pr, mdl in rows:
    print('  %-14s %7.3f  %+6.3f  probes %.2f  model %5.1f%%%s' % (nm, L, d, pr, mdl, '' if abs(d) <= 0.1 else '   <-- outside 0.1 dB'))
print('%d of %d perturbations within 0.1 dB of the selected point; %d better; loss range %.3f..%.3f; probes range %.2f..%.2f'
      % (sum(1 for r in rows if abs(r[2]) <= 0.1), len(rows), sum(1 for r in rows if r[2] < 0),
         min(r[1] for r in rows), max(r[1] for r in rows), min(r[3] for r in rows), max(r[3] for r in rows)))
json.dump(dict(selected=dep, fb=fb, rows=rows), open('results/analysis/sweep3_%s.json' % os.path.basename(root), 'w'))
print('SWEEP3_DONE')
