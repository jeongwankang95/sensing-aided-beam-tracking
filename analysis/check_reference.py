"""Compare the regenerated outputs with the copies behind the paper.
- results/analysis/<name> against results/reference/<name>, and results/figures/<name> against results/reference/figures/<name>.
- Outputs that are also inputs of other scripts against their configs/ copy: sel3_val.json, sel3_val_grid.npz (analysis/selection.py),
  loso3_val.json (analysis/selection_summary.py), llr_scenario*.json (analysis/lock_loss_selection.py) and resweep_scenario*.json
  (analysis/resweep_selection.py).
Logs and TikZ files are compared line by line; JSON files field by field (floats to a relative tolerance of 1e-9); npz files
array by array (exactly). Files missing from the outputs are listed as not regenerated.
Usage: python analysis/check_reference.py [name ...]   (names relative to results/reference/, e.g. table2_s9.log, figures/fig4_perpass.tikz)"""
import json, math, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT, FIG, REF = REPO / 'results' / 'analysis', REPO / 'results' / 'figures', REPO / 'results' / 'reference'
CONFIG_OUTPUTS = {'sel3_val.json': 'configs/sel3_val.json', 'sel3_val_grid.npz': 'configs/sel3_val_grid.npz',
                  'loso3_val.json': 'configs/loso3_val.json'}
CONFIG_OUTPUTS.update({f'{f}_scenario{s}.json': f'configs/{f}_scenario{s}.json' for f in ('llr', 'resweep') for s in (9, 31, 32, 33)})


def diff(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f'{path}.{k}: missing' for k in a.keys() ^ b.keys()]
        for k in a.keys() & b.keys():
            out += diff(a[k], b[k], f'{path}.{k}')
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f'{path}: length {len(a)} vs {len(b)}']
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in diff(u, v, f'{path}[{i}]')]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        same = (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
        return [] if same else [f'{path}: {a!r} vs {b!r}']
    return [] if a == b else [f'{path}: {a!r} vs {b!r}']


def compare(new, ref):
    if ref.suffix == '.json':
        return diff(json.load(open(new)), json.load(open(ref)))
    if ref.suffix == '.npz':
        x, y = np.load(new), np.load(ref)
        if sorted(x.files) != sorted(y.files):
            return [f'arrays {sorted(x.files)} vs {sorted(y.files)}']
        return [f'array {k} differs' for k in y.files if not np.array_equal(x[k], y[k])]
    a, b = new.read_text().splitlines(), ref.read_text().splitlines()
    d = [f'line {i + 1}: {x!r} vs {y!r}' for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(a) != len(b):
        d.append(f'{len(a)} lines vs {len(b)}')
    return d


pairs = {}
for p in sorted(REF.rglob('*')):
    if p.is_file():
        n = str(p.relative_to(REF))
        pairs[n] = (FIG / n[len('figures/'):] if n.startswith('figures/') else OUT / n, p)
for n, c in CONFIG_OUTPUTS.items():
    pairs[n] = (OUT / n, REPO / c)
names = sys.argv[1:] or list(pairs)
ok = bad = missing = 0
for n in names:
    new, ref = pairs[n]
    if not new.exists():
        missing += 1
        print(f'not regenerated: {n}')
        continue
    d = compare(new, ref)
    if d:
        bad += 1
        print(f'DIFF {n}: ' + '; '.join(d[:3]) + (' ...' if len(d) > 3 else ''))
    else:
        ok += 1
print(f'{ok} identical, {bad} different, {missing} not regenerated (of {len(names)})')
