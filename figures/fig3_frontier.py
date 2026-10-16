"""fig3_frontier.py: TikZ source of Fig. 3 (Scenario-9 mean power loss against candidate-beam probes per epoch), written to
results/figures/fig3_frontier.tikz, and a coordinate comparison with the paper's figure (figures/tikz/fig3_frontier.tikz).
Inputs: results/analysis/policy_family_s9.json (analysis/policy_family.py: periodic, fixed-cadence, extrapolation and
innovation-tracker policies and the Bayes/HMM frontier), the test grids of the lock-loss-recovery and re-sweep references
(configs/llr_scenario9.json, configs/resweep_scenario9.json) and Table II (results/analysis/t2_scenario9.json: the controller and
the controller with its sensor disabled). The vertical coordinate is log10(loss in dB).
- Markers: every periodic (18), fixed-cadence (3), extrapolation (2), tracker (7) and Bayes/HMM (5) policy; the Pareto points of
  the re-sweep grid (12) and of the lock-loss-recovery grid (11: the point equal to the plain tracker, which never fires, and
  points within 0.01 probes/epoch of the previous one are not drawn).
- Frontier: lower convex hull of all 205 policies in the linear-dB plane, each edge drawn with max(2, ceil(length/0.25))
  segments; the dotted segment ends where the hull reaches the controller's loss (4.23 probes/epoch, the caption's value).
The paper's figure was redrawn in 2026-09 from an earlier PDF figure, so its periodic and fixed-cadence markers carry
PDF-recovered coordinates; this script recomputes every coordinate. The comparison it prints gives, per series, the largest
difference from the paper's coordinates (probes/epoch and log10 dB).
usage: python figures/fig3_frontier.py"""
import json
import math
import os
import re

A = os.path.join('results', 'analysis')
OUT = os.path.join('results', 'figures', 'fig3_frontier.tikz')
PAPER = os.path.join('figures', 'tikz', 'fig3_frontier.tikz')
F = json.load(open(os.path.join(A, 'policy_family_s9.json')))
T2 = json.load(open(os.path.join(A, 't2_scenario9.json')))['rows']
LLR = json.load(open(os.path.join('configs', 'llr_scenario9.json')))['test_grid']
RS = json.load(open(os.path.join('configs', 'resweep_scenario9.json')))['test_grid']

lg = math.log10
fam = lambda name: sorted((d['probes'], d['loss']) for d in F[name])
periodic, cadence, extrap, tracker = fam('periodic'), fam('fixed cadence'), fam('extrapolation'), fam('tracker')
bayes = sorted({(d['probes'], d['loss']) for d in F['bayes_hmm']})                    # budgets 5.0 and 7.0 pick the same policy
llr = sorted({(round(s[1], 4), round(s[0], 4)) for g, s in LLR})                     # (probes, loss), 72 distinct settings
rs = [(e[1][1], e[1][0]) for e in RS]                                                 # (probes, loss), 98 settings
ctl, off, plain = T2['METHOD'], T2['own tracker + re-sweep, no sensor'], T2['feedback-only tracker']


def pareto(Q):
    out, best = [], 1e9
    for x, y in sorted(Q):
        if y < best - 1e-12:
            out.append((x, y)); best = y
    return out


def hull(P):
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    H = []
    for p in sorted(set(P)):
        while len(H) >= 2 and cross(H[-2], H[-1], p) <= 0: H.pop()
        H.append(p)
    i = min(range(len(H)), key=lambda i: H[i][1]); return H[:i + 1]


allpts = periodic + cadence + extrap + tracker + [(x, y) for x, y in bayes] + llr + rs
H = hull(allpts)
curve = [H[0]]
for (x0, y0), (x1, y1) in zip(H, H[1:]):
    n = max(2, math.ceil((x1 - x0) / 0.25))
    curve += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(1, n + 1)]
yc = ctl['loss']
xc = next(x0 + (yc - y0) * (x1 - x0) / (y1 - y0) for (x0, y0), (x1, y1) in zip(H, H[1:]) if (y0 - yc) * (y1 - yc) <= 0)
llr_drawn, last = [], None
for x, y in pareto(llr):
    if (x, y) == (round(plain['probes'], 4), round(plain['loss'], 4)) or (last is not None and x - last < 0.01):
        continue
    llr_drawn.append((x, y)); last = x
rs_drawn = [(round(x, 3), round(y, 4)) for x, y in pareto(rs)]

xy = lambda x, y: '%.3f/%.5f' % (x, lg(y))


def block(pts, breaks, indent):
    """comma-separated x/y items, a line break after the item counts in breaks (cumulative)"""
    items, lines, start = [xy(*p) for p in pts], [], 0
    for b in breaks + [len(pts)]:
        if b > start: lines.append(','.join(items[start:b])); start = b
    return (',\n' + indent).join(lines)


curve_items = ['(%.3f,%.5f)' % (x, lg(y)) for x, y in curve]
curve_txt = '\n      '.join(' '.join(curve_items[i:j]) for i, j in
                            zip([0] + list(range(3, len(curve_items), 4)), list(range(3, len(curve_items), 4)) + [len(curve_items)]))
TIKZ = open(PAPER).read()
blocks = {
    'curve': ('plot coordinates {' + curve_txt + '};'),
    'periodic': block(periodic, [3, 7, 11, 15], '    '),
    'cadence': block(cadence, [], ''), 'extrap': block(extrap, [], ''), 'tracker': block(tracker, [3], '    '),
    'bayes': ','.join(xy(*p) for p in bayes),
    'llr': ','.join(xy(*p) for p in llr_drawn), 'rs': ','.join(xy(*p) for p in rs_drawn),
}
yctl = lg(ctl['loss'])
tikz = TIKZ
tikz = re.sub(r'plot coordinates \{.*?\};', lambda m: blocks['curve'], tikz, count=1, flags=re.S)
pat = r'(\\foreach \\x/\\y in \{)(.*?)(\}\{)'
order = ['periodic', 'cadence', 'extrap', 'tracker', 'bayes', 'llr', 'rs']
parts, pos = [], 0
for name, m in zip(order, re.finditer(pat, tikz, flags=re.S)):
    parts += [tikz[pos:m.start(2)], blocks[name]]; pos = m.end(2)
tikz = ''.join(parts) + tikz[pos:]
for a, b in (('(3.247,-0.64782) -- (4.232,-0.64782)', '(%.3f,%.5f) -- (%.3f,%.5f)' % (ctl['probes'], yctl, xc, yctl)),
             ('(4.232,-0.68782) -- (4.232,-0.60782)', '(%.3f,%.5f) -- (%.3f,%.5f)' % (xc, yctl - 0.04, xc, yctl + 0.04)),
             ('at (3.247,-0.64782) {}', 'at (%.3f,%.5f) {}' % (ctl['probes'], yctl)),
             ('at (4.119,-0.58536) {}', 'at (%.3f,%.5f) {}' % (off['probes'], lg(off['loss'])))):
    assert tikz.count(a) == 1, a          # anchors: the dotted segment, its tick and the two diamonds of the paper's figure
    tikz = tikz.replace(a, b)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write(tikz)

# comparison with the paper's coordinates, series by series
def coords(t):
    t = t.split('% Compact in-panel legend')[0]
    curve = re.findall(r'\((-?[\d.]+),(-?[\d.]+)\)', re.search(r'plot coordinates \{(.*?)\};', t, re.S).group(1))
    fe = [re.findall(r'(-?[\d.]+)/(-?[\d.]+)', m) for m in re.findall(r'\\foreach \\x/\\y in \{(.*?)\}\{', t, re.S)]
    rest = re.findall(r'\((-?[\d.]+),(-?[\d.]+)\)', t[t.index('densely dotted'):])
    return [('frontier', curve)] + list(zip(order, fe)) + [('diamonds, dotted segment, tick', rest)]
npol = len(periodic) + len(cadence) + len(extrap) + len(tracker) + len(bayes) + len(LLR) + len(RS)
print('%s: %d policies (%d distinct points; the %d lock-loss-recovery settings give %d), hull vertices at %s probes/epoch; '
      'crossing at %.4f probes/epoch (%.3f times the controller\'s %.4f)'
      % (OUT, npol, len(set(allpts)), len(LLR), len(llr), ', '.join('%.3f' % x for x, _ in H), xc, xc / ctl['probes'], ctl['probes']))
print('largest difference from the paper\'s coordinates (probes/epoch, log10 dB):')
for (name, a), (_, b) in zip(coords(TIKZ), coords(tikz)):
    assert len(a) == len(b), (name, len(a), len(b))
    dx = max(abs(float(p[0]) - float(q[0])) for p, q in zip(a, b)); dy = max(abs(float(p[1]) - float(q[1])) for p, q in zip(a, b))
    print('  %-32s %2d points  %.3f  %.5f' % (name, len(a), dx, dy))
