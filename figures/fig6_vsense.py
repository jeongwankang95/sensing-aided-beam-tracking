"""fig6_vsense.py: TikZ source of Fig. 6 (marginal value of sensing under stress), written to
results/figures/fig6_vsense.tikz.
V_sense = loss(controller with its sensor disabled) - loss(controller), with paired pass-level 95% intervals; positive means
that sensing reduces loss. Each row is read from an analysis output:
  native interval, stale link power, late probe outcomes, late commitment and the late-sensor rows: the "skeleton (no sensor)
    minus ctl" line of each condition in results/analysis/delays_sensor_off_p5_s9.log (supplementary Table S-lag);
  measurement noise: results/analysis/noise_skeleton_p5_s9.log (Sec. VI-D, 20 noise realizations);
  decimation rows and the other sites at d=4: "sensor off - controller" (consecutive inputs) and "sensor off - controller,
    d-spaced", every-frame ("held") scoring, in results/analysis/hold_scenario<s>.json (analysis/every_frame.py; Table IV).
Values are rounded as the tables they come from print them (three decimals for the Table S-lag estimates, two otherwise; two
for every interval) before they are placed. Scenario 32 (three passes) has no interval.
The output equals figures/tikz/fig6_vsense.tikz, the TikZ code of the paper's figure.
usage: python figures/fig6_vsense.py"""
import json
import os
import re

A = os.path.join('results', 'analysis')
OUT = os.path.join('results', 'figures', 'fig6_vsense.tikz')


def lag_rows(fn):
    """condition header -> (value, lo, hi) of the 'skeleton (no sensor) minus ctl' line under it."""
    rows, cond = {}, None
    for line in open(os.path.join(A, fn)):
        if line.startswith('-- '):
            cond = line[3:].strip()
        m = re.search(r'skeleton \(no sensor\) minus ctl ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]', line)
        if m and cond is not None:
            rows[cond] = tuple(float(x) for x in m.groups())
    return rows


def noise_rows(fn):
    rows = {}
    for line in open(os.path.join(A, fn)):
        m = re.search(r'sigma_m = ([\d.]+) dB .* value ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]', line)
        if m:
            rows[float(m.group(1))] = tuple(float(x) for x in m.groups()[1:])
    return rows


def held(s, d, key):
    """(value, lo, hi) of an every-frame paired difference in hold_scenario<s>.json at decimation d."""
    return tuple(json.load(open(os.path.join(A, f'hold_scenario{s}.json')))[str(d)][key]['held'])


LAG, NOISE = lag_rows('delays_sensor_off_p5_s9.log'), noise_rows('noise_skeleton_p5_s9.log')
lag = lambda c: (round(LAG[c][0], 3), round(LAG[c][1], 2), round(LAG[c][2], 2))
two = lambda v: (round(v[0], 2), round(v[1], 2), round(v[2], 2))
ROWS = [  # (label, estimate, lo, hi, scenario-9 row?)  None interval = no interval (three passes)
    ('native interval', *lag('none (SELF-CHECK Table II)'), True),
    ('hdr', 'feedback stressed'),
    (r'noise $0.5$\,dB', *two(NOISE[0.5]), True),
    (r'noise $1$\,dB', *two(NOISE[1.0]), True),
    (r'noise $2$\,dB', *two(NOISE[2.0]), True),
    ('stale link power', *lag('link delay 1 (SELF-CHECK 1.086/1.904 on S9)'), True),
    ('late probe outcomes', *lag('PROBE LAG (probe results usable next epoch)'), True),
    ('late commitment', *lag('COMMIT LAG 1 (all commitments)'), True),
    (r'$d=2$, consecutive', *two(held(9, 2, 'sensor off - controller')), True),
    (r'$d=4$, consecutive', *two(held(9, 4, 'sensor off - controller')), True),
    (r'$d=2$, $d$-spaced', *two(held(9, 2, 'sensor off - controller, d-spaced')), True),
    (r'$d=4$, $d$-spaced', *two(held(9, 4, 'sensor off - controller, d-spaced')), True),
    ('hdr', 'sensor late or weaker'),
    ('late sensor, 1 epoch', *lag('sensor delay 1 (SELF-CHECK 0.820 on S9)'), True),
    ('late sensor, 2 epochs', *lag('sensor delay 2'), True),
    ('late sensor and probes', *lag('PROBE LAG + sensor delay 1'), True),
    ('late sensor, stale link', *lag('JOINT sensor 1 + link 1'), True),
    (r'S31 arm, $d=4$', *two(held(31, 4, 'sensor off - controller')), False),
    (r'S32 arm, $d=4$', round(held(32, 4, 'sensor off - controller')[0], 2), None, None, False),
    (r'S33 arm, $d=4$', *two(held(33, 4, 'sensor off - controller')), False),
]
VMIN, VMAX, SX, X0 = -1.2, 2.7, 0.80, 0.0   # dB range on the axis, cm per dB, x of VMIN (2.7: d=2 interval ends at 2.64)
DY = 0.25                                    # cm per row


def X(v):
    return X0 + (v - VMIN) * SX


def val_text(e, lo, hi):
    est = f'{e:+.2f}'
    if lo is None:
        return f'${est}$'
    return f'${est}$ $[{lo:.2f},{hi:.2f}]$'


out = []
out.append(r'\begin{tikzpicture}[font=\scriptsize]')
y = 0.0
ys = []
for r in ROWS:
    if r[0] == 'hdr':
        y -= 0.08 if ys else 0.0
        out.append(rf'  \node[anchor=east,font=\scriptsize\itshape] at ({X(VMIN) - 0.12:.3f},{y:.3f}) {{{r[1]}}};')
        y -= DY
        continue
    lab, e, lo, hi, s9 = r
    ys.append(y)
    out.append(rf'  \node[anchor=east] at ({X(VMIN) - 0.12:.3f},{y:.3f}) {{{lab}}};')
    if lo is not None:
        a, b = max(lo, VMIN), min(hi, VMAX)
        if a < VMAX:
            out.append(rf'  \draw[line width=0.5pt] ({X(a):.3f},{y:.3f}) -- ({X(b):.3f},{y:.3f});')
            for v in (lo, hi):
                if VMIN <= v <= VMAX:
                    out.append(rf'  \draw[line width=0.5pt] ({X(v):.3f},{y - 0.05:.3f}) -- ({X(v):.3f},{y + 0.05:.3f});')
    if e <= VMAX:
        style = 'fill=black' if s9 else 'fill=white'
        out.append(rf'  \draw[{style},line width=0.45pt] ({X(e):.3f},{y:.3f}) circle (1.5pt);')
    else:  # off scale: arrow at the right edge
        out.append(rf'  \draw[-{{Latex[length=1.6mm]}},line width=0.6pt] ({X(VMAX) - 0.35:.3f},{y:.3f}) -- ({X(VMAX):.3f},{y:.3f});')
    out.append(rf'  \node[anchor=west] at ({X(VMAX) + 0.12:.3f},{y:.3f}) {{{val_text(e, lo, hi)}}};')
    y -= DY
ytop, ybot = ys[0] + 0.18, ys[-1] - 0.18
# zero line, axis, ticks
out.insert(1, rf'  \draw[gray!60,densely dashed,line width=0.45pt] ({X(0):.3f},{ytop:.3f}) -- ({X(0):.3f},{ybot:.3f});')
out.append(rf'  \draw[line width=0.45pt] ({X(VMIN):.3f},{ybot:.3f}) -- ({X(VMAX):.3f},{ybot:.3f});')
for t in (-1, 0, 1, 2):
    out.append(rf'  \draw[line width=0.4pt] ({X(t):.3f},{ybot:.3f}) -- ++(0,-2.5pt) node[below,inner sep=1.5pt] {{${t}$}};')
out.append(rf'  \node[font=\footnotesize,anchor=north] at ({X((VMIN + VMAX) / 2):.3f},{ybot - 0.33:.3f}) {{marginal value of sensing (dB)}};')
out.append(r'\end{tikzpicture}')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write('\n'.join(out) + '\n')
print(f'{OUT}: {sum(1 for r in ROWS if r[0] != "hdr")} rows')
for r in ROWS:
    if r[0] != 'hdr':
        print('  %-28s %s' % (r[0], val_text(*r[1:4])))
