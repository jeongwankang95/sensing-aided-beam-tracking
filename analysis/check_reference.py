"""Compare the outputs regenerated in results/analysis/ with the copies behind the paper in results/reference/.
Logs are compared line by line; JSON files field by field (floats to a relative tolerance of 1e-9). Files missing from
results/analysis/ are listed as not regenerated. Usage: python analysis/check_reference.py [name ...]"""
import json, math, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT, REF = REPO / 'results' / 'analysis', REPO / 'results' / 'reference'


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


names = sys.argv[1:] or sorted(str(p.relative_to(REF)) for p in REF.rglob('*') if p.is_file())
ok = bad = missing = 0
for n in names:
    new, ref = OUT / n, REF / n
    if not new.exists():
        missing += 1
        print(f'not regenerated: {n}')
        continue
    if n.endswith('.json'):
        d = diff(json.load(open(new)), json.load(open(ref)))
    else:
        a, b = new.read_text().splitlines(), ref.read_text().splitlines()
        d = [f'line {i + 1}: {x!r} vs {y!r}' for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if len(a) != len(b):
            d.append(f'{len(a)} lines vs {len(b)}')
    if d:
        bad += 1
        print(f'DIFF {n}: ' + '; '.join(d[:3]) + (' ...' if len(d) > 3 else ''))
    else:
        ok += 1
print(f'{ok} identical, {bad} different, {missing} not regenerated (of {len(names)})')
