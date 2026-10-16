"""fig4_perpass.py: TikZ source of Fig. 4 (per-pass loss difference on the 28 Scenario-9 test passes), written to
results/figures/fig4_perpass.tikz.
Reads results/analysis/t2_scenario9.json (analysis/table2.py). Filled markers: each pass's mean loss of the plain tracker minus
that of the controller (METHOD), sorted ("per-pass value (fb - METHOD), sorted" in table2_s9.log); hollow markers: the
controller with its sensor disabled ("own tracker + re-sweep, no sensor") minus the controller, in the same pass order; the
dashed line is the frame-weighted mean of the filled values (0.82 dB). Axes, ticks and legend are fixed layout.
Also prints the range of the seven largest gains and the mean of the other 21 (Sec. VI-A: 2.73-3.63 dB; -0.02 dB).
The output equals figures/tikz/fig4_perpass.tikz, the TikZ code of the paper's figure.
usage: python figures/fig4_perpass.py"""
import json
import os

import numpy as np

OUT = os.path.join('results', 'figures', 'fig4_perpass.tikz')
D = json.load(open(os.path.join('results', 'analysis', 't2_scenario9.json')))
mean = lambda row: np.array([s / n for s, n, *_ in D['perpass'][row]])
fb, ctl, off = mean('feedback-only tracker'), mean('METHOD'), mean('own tracker + re-sweep, no sensor')
nfr = np.array([n for _, n, *_ in D['perpass']['METHOD']])
order = np.argsort(fb - ctl, kind='stable')
filled, hollow = (fb - ctl)[order], (off - ctl)[order]
assert np.allclose(filled, D['perpass_value_sorted'])
wmean = float(((fb - ctl) * nfr).sum() / nfr.sum())


def series(vals, cmd):
    items = ['%d/%.2f' % (i + 1, v) for i, v in enumerate(vals)]
    groups = [items[0:7], items[7:13], items[13:19], items[19:25], items[25:]]
    body = (',%\n' + ' ' * 21).join(','.join(g) for g in groups)
    return r'  \foreach \i/\v in {' + body + '}{' + cmd + '}'


TIKZ = r'''\begin{tikzpicture}[x=0.265cm,y=0.84cm]
  \draw[line width=0.45pt] (0,-0.6) -- (29,-0.6);
  \draw[line width=0.45pt] (0,-0.6) -- (0,3.8);
  \draw[gray!55,densely dashed,line width=0.45pt] (0,0) -- (29,0);
  \draw[black,dashed,line width=0.55pt] (0,MEAN3) -- (29,MEAN3);
  \node[font=\scriptsize,anchor=south west,fill=white,inner sep=1pt]
        at (3.0,0.87) {mean $=MEAN2$\,dB};
  \draw[line width=0.4pt] (0,-0.5) -- ++(-3pt,0);
  \node[font=\scriptsize,anchor=east,inner sep=2pt] at (0,-0.5) {$-0.5$};
  \foreach \y in {0,1,2,3}{%
    \draw[line width=0.4pt] (0,\y) -- ++(-3pt,0);
    \node[font=\scriptsize,anchor=east,inner sep=2pt] at (0,\y) {$\y$};}
  \foreach \x in {1,7,14,21,28}{%
    \draw[line width=0.4pt] (\x,-0.6) -- ++(0,-3pt);
    \node[font=\scriptsize,anchor=north] at (\x,-0.67) {$\x$};}
HOLLOW
FILLED
  \fill (1.6,3.55) circle (1.25pt);
  \node[font=\scriptsize,anchor=west] at (2.1,3.55) {vs.\ plain tracker};
  \draw[line width=0.45pt] (1.6,3.15) circle (1.6pt);
  \node[font=\scriptsize,anchor=west] at (2.1,3.15) {vs.\ sensor disabled};
  \node[font=\scriptsize,anchor=north] at (14.5,-0.95)
        {Scenario~9 test pass (ordered by gain over the plain tracker)};
  \node[font=\scriptsize,rotate=90] at (-2.6,1.6) {difference (dB)};
\end{tikzpicture}
'''
tikz = (TIKZ.replace('MEAN3', '%.3f' % wmean).replace('MEAN2', '%.2f' % wmean)
        .replace('HOLLOW', series(hollow, r'\draw[line width=0.45pt] (\i,\v) circle (1.6pt);'))
        .replace('FILLED', series(filled, r'\fill (\i,\v) circle (1.25pt);')))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write(tikz)
print(f'{OUT}: frame-weighted mean gain over the plain tracker {wmean:+.3f} dB; {int((filled > 0).sum())} of {len(filled)} passes positive')
top, rest = order[-7:], order[:-7]
print('the 7 passes with the largest gain: %.3f-%.3f dB; the other %d: mean %+.4f dB (frame-weighted %+.4f)'
      % ((fb - ctl)[top].min(), (fb - ctl)[top].max(), len(rest), (fb - ctl)[rest].mean(),
         ((fb - ctl)[rest] * nfr[rest]).sum() / nfr[rest].sum()))
