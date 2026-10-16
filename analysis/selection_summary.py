"""selection_summary.py: numbers the supplement quotes about the constant selection (Sec. S17-N), from
results/analysis/sel3_val.json and results/analysis/sel3_val_grid.npz (analysis/selection.py), and the leave-one-site-out
selections of Table S-loso.
Prints: the settings near the selected one (22, 67 and 419 within 0.01, 0.02 and 0.05 dB of mean regret) and how many of them
use the coarse re-sweep three epochs after setup (22 of 22, 64 of 67); which selected constants lie on an edge of the grid
(N_bl only); the rank of the earlier (2026-08-10) constants among the 62,717 settings, with or without the re-sweep, that meet
the probing condition (657th).
Writes results/analysis/loso3_val.json: the selection rule applied to two of the three validation sites
(configs/loso3_val.json is this output; analysis/loso.py evaluates it on the held-out site's test passes).
usage: python analysis/selection_summary.py"""
import json

import numpy as np

z = np.load('results/analysis/sel3_val_grid.npz')
A, cfg, ro = z['A'], z['cfg'], z['ro']                      # A: (setting, [S9, S31, S32], [loss, probes, inv%, sweeps])
SJ = json.load(open('results/analysis/sel3_val.json'))
SITES = ('S9', 'S31', 'S32')
Lref = np.array([SJ['reference'][k]['loss'] for k in SITES]); Pref = np.array([SJ['reference'][k]['probes'] for k in SITES])
L, P, I = A[:, :, 0], A[:, :, 1], A[:, :, 2]
REG = L - Lref; hasrs = ro != 'null'
feas = (P <= Pref + 1e-9).all(1); obj = REG.mean(1)
pi = SJ['selected']['primary']['index']
print('primary setting %d: %s, re-sweep %s, mean regret %+.4f dB' % (pi, cfg[pi], ro[pi], obj[pi]))
for t in (0.01, 0.02, 0.05):
    near = np.where(hasrs & feas & (obj <= obj[pi] + t))[0]
    k3 = sum(json.loads(ro[i]) == ['coarse', 3] for i in near)
    print('within %.2f dB: %d settings, %d of them with the coarse re-sweep three epochs after setup' % (t, len(near), k3))
values = {}
for c in cfg:
    for k, v in json.loads(c).items():
        values.setdefault(k, set()).add(v)
ks = sorted({json.loads(r)[1] for r in ro if r != 'null'})
sel = dict(json.loads(cfg[pi]), resweep_k=json.loads(ro[pi])[1]); values['resweep_k'] = set(ks)
edge = [k for k, v in sel.items() if v in (min(values[k]), max(values[k]))]
print('selected constants on a grid edge:', edge, '| grid:', {k: sorted(v) for k, v in values.items()})
dep = SJ['S4_deployed']['index']
rank = int((obj[feas] < obj[dep] - 1e-12).sum()) + 1
print('earlier constants (2026-08-10): mean regret %+.4f, meets the probing condition: %s; rank %d of the %d settings that meet it '
      '(%d with and %d without the re-sweep); sel3_val.json: rank %d of %d'
      % (obj[dep], bool(feas[dep]), rank, feas.sum(), (feas & hasrs).sum(), (feas & ~hasrs).sum(),
         SJ['S4_deployed']['rank_all_feasible'], SJ['S4_deployed']['n_all_feasible']))

OUT = {}
for name, pool in (('loso_without_S9', [1, 2]), ('loso_without_S31', [0, 2]), ('loso_without_S32', [0, 1])):
    f = (P[:, pool] <= Pref[pool] + 1e-9).all(1); o = REG[:, pool].mean(1)
    i = int(min(np.where(hasrs & f)[0], key=lambda i: (round(o[i], 4), round(P[i, pool].mean(), 3), round(I[i, pool].mean(), 2))))
    OUT[name] = dict(index=i, cfg=json.loads(cfg[i]), resweep=json.loads(ro[i]), pool_objective=float(o[i]),
                     val={s: dict(loss=float(L[i, j]), regret=float(REG[i, j]), probes=float(P[i, j]), ref_probes=float(Pref[j]))
                          for j, s in enumerate(SITES)})
    print('%-17s setting %6d: %s, re-sweep %s, pool mean regret %+.3f' % (name, i, '/'.join(str(v) for v in OUT[name]['cfg'].values()),
                                                                          ro[i], o[i]))
json.dump(OUT, open('results/analysis/loso3_val.json', 'w'), indent=1)
print('-> results/analysis/loso3_val.json')
