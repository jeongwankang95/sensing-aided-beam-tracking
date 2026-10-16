"""fig5_example.py: TikZ source of Fig. 5 (two test passes over time), written to results/figures/fig5_example.tikz.
Reads results/analysis/trace_scenario9.json and trace_scenario31.json, written by analysis/example_traces.py (locked_v3.json
constants; plain tracker slack 0.4 on Scenario 9 and 0.6 on Scenario 31). Frame-weighted means reproduce Table II (S9 0.225 /
0.260 / 1.045 dB; S31 0.118 / 0.115 / 0.109 dB for controller / sensor disabled / plain tracker).
Pass choice, stated in the caption: in each group, the pass whose marginal value of sensing (sensor-disabled minus controller
loss) is the median: (a) the seven Scenario-9 lock-loss passes -> idx 24; (b) the eleven Scenario-31 passes -> idx 0.
The output equals figures/tikz/fig5_example.tikz, the TikZ code of the paper's figure.
usage: python figures/fig5_example.py"""
import json
import os

TR = os.path.join('results', 'analysis')
OUT = os.path.join('results', 'figures', 'fig5_example.tikz')


def pick(fn, lockloss_only):
    ps = json.load(open(os.path.join(TR, fn)))['passes']
    if lockloss_only:
        ps = [p for p in ps if p['fb_mean'] - p['ctrl_mean'] > 2.0]
    v = sorted(ps, key=lambda p: (p['off_mean'] - p['ctrl_mean'], p['idx']))
    return v[(len(v) - 1) // 2], len(ps)


PA, NA = pick('trace_scenario9.json', True)
PB, NB = pick('trace_scenario31.json', False)
assert (PA['idx'], NA, PB['idx'], NB) == (24, 7, 0, 11), (PA['idx'], NA, PB['idx'], NB)

W, H = 6.9, 2.15          # plot width and height in cm
YMAX = 63
GAP = 0.95                # vertical gap between panels (cm)


def panel(p, y0, label, legend_at, resweep_label):
    n = p['n']
    sx, sy = W / (n + 1), H / YMAX
    X = lambda t: t * sx
    Y = lambda b: y0 + b * sy
    out = []
    for r in p['ctrl']:                       # frames on which the model runs (trial or sensing)
        if r['model']:
            out.append(r'  \fill[gray!18] (%.3f,%.3f) rectangle (%.3f,%.3f);' % (X(r['t'] - .5), Y(0), X(r['t'] + .5), Y(YMAX)))
    for r in p['ctrl']:                       # the scheduled re-sweep
        if r['resweep']:
            out.append(r'  \draw[black!45,line width=0.4pt] (%.3f,%.3f) -- (%.3f,%.3f);' % (X(r['t']), Y(0), X(r['t']), Y(YMAX)))
            if resweep_label:
                out.append(r'  \node[font=\scriptsize,anchor=north west,inner sep=1pt] at (%.3f,%.3f) {re-sweep};' % (X(r['t']) + 0.03, Y(YMAX)))
    out.append(r'  \draw[line width=0.45pt] (0,%.3f) -- (%.3f,%.3f);' % (Y(0), W, Y(0)))
    out.append(r'  \draw[line width=0.45pt] (0,%.3f) -- (0,%.3f);' % (Y(0), Y(YMAX)))
    for b in (0, 32, 63):
        out.append(r'  \draw[line width=0.4pt] (0,%.3f) -- ++(-2.5pt,0) node[left,inner sep=1.5pt,font=\scriptsize] {$%d$};' % (Y(b), b))
    step = 10 if n <= 60 else 40
    for t in range(0, n + 1, step):
        out.append(r'  \draw[line width=0.4pt] (%.3f,%.3f) -- ++(0,-2.5pt) node[below,inner sep=1.5pt,font=\scriptsize] {$%d$};' % (X(t), Y(0), t))
    out.append(r'  \node[font=\scriptsize,anchor=south west,inner sep=1pt] at (%.3f,%.3f) {%s};' % (-0.05, Y(YMAX) + 0.02, label))
    def line(key, style):
        pts = ' '.join('(%.3f,%.3f)' % (X(r['t']), Y(r['a'])) for r in p[key])
        return r'  \draw[%s] plot coordinates {%s};' % (style, pts)
    out.append(line('fb', 'black,dashed,line width=0.55pt'))
    out.append(line('off', 'black!60,densely dotted,line width=0.8pt'))
    out.append(line('ctrl', 'black,line width=0.55pt'))
    for r in p['ctrl']:                       # best beam as dots on top: argmax jumps stay visible as single dots
        out.append(r'  \fill[gray!55] (%.3f,%.3f) circle (0.8pt);' % (X(r['t']), Y(r['best'])))
    mean_k = lambda key: sum(r['k'] for r in p[key]) / len(p[key])
    rows = [(None, 'best beam', None),
            ('black,dashed,line width=0.55pt', 'plain tracker', (p['fb_mean'], mean_k('fb'))),
            ('black!60,densely dotted,line width=0.8pt', 'sensor disabled', (p['off_mean'], mean_k('off'))),
            ('black,line width=0.55pt', 'controller', (p['ctrl_mean'], mean_k('ctrl')))]
    lx, ly = legend_at
    for i, (st, name, val) in enumerate(rows):
        yy = Y(ly) - i * 0.215
        if st is None:
            out.append(r'  \fill[gray!60] (%.3f,%.3f) circle (0.9pt);' % (X(lx) + 0.21, yy))
        else:
            out.append(r'  \draw[%s] (%.3f,%.3f) -- (%.3f,%.3f);' % (st, X(lx), yy, X(lx) + 0.42, yy))
        txt = name if val is None else r'%s: $%.2f$\,dB, $%.2f$' % (name, val[0], val[1])
        out.append(r'  \node[font=\scriptsize,anchor=west,inner sep=1pt] at (%.3f,%.3f) {%s};' % (X(lx) + 0.47, yy, txt))
    return out


o = [r'\begin{tikzpicture}']
o += panel(PA, H + GAP, '(a) Scenario~9, lock-loss pass', legend_at=(17.5, 26.5), resweep_label=True)
o += panel(PB, 0.0, '(b) Scenario~31 pass', legend_at=(20.0, 27.0), resweep_label=False)
o.append(r'  \node[font=\scriptsize,rotate=90] at (-0.62,%.3f) {beam index};' % (H + GAP / 2))
o.append(r'  \node[font=\scriptsize,anchor=north] at (%.3f,-0.30) {decision epoch};' % (W / 2))
o.append(r'\end{tikzpicture}')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write('\n'.join(o) + '\n')
print(f'{OUT}: passes {PA["idx"]} of the {NA} Scenario-9 lock-loss passes and {PB["idx"]} of the {NB} Scenario-31 passes')
