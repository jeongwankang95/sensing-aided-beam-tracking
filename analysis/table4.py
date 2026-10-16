#!/usr/bin/env python3
"""table4.py: the every-frame decimation numbers of the paper, read from results/analysis/hold_scenario<s>.json
(analysis/every_frame.py) and, for the drift and innovation columns, results/analysis/decimation_s<s>.log
(analysis/decimation.py). 'held' = every native frame scored with the beam held between decisions; 'dec' = decision frames
only. Prints the Table IV rows (main), the rows of supplementary Table S27, the Scenario-9 d-spaced and re-sweep rows quoted in
Sec. VI-C, the probe counts per decision and per frame (Table S28), the re-selected comparator slacks and the Fig. 6
decimation rows.
usage: python analysis/table4.py"""
import json
import os
import re

A = os.path.join('results', 'analysis')
R = {s: json.load(open(os.path.join(A, 'hold_scenario%d.json' % s))) for s in (9, 31, 32, 33)}
DRIFT, INNOV = {}, {}
for s in (9, 31, 32, 33):   # rows " d  drift  innov med | ..." of the decimation table, kept as printed
    rows = [re.match(r'\s*([124])\s+(\S+)\s+(\S+)\s+\|', l) for l in open(os.path.join(A, 'decimation_s%d.log' % s))]
    rows = {m.group(1): m.groups()[1:] for m in rows if m}
    DRIFT[s], INNOV[s] = [rows[d][0] for d in '124'], [rows[d][1] for d in '124']


def g(x, n=2):
    v = round(x, n)
    if v == 0:
        v = 0.0
    return ('$%+.' + str(n) + 'f$') % v if v != 0 else '$+0.00$'


def ci(t, n=2):
    lo, hi = round(t[1], n), round(t[2], n)
    f = '%.' + str(n) + 'f'
    return '$[' + (f % lo).replace('-0.00', '0.00') + ',' + (f % hi).replace('-0.00', '0.00') + ']$'


def L(x):
    return '$%.3f$' % x


print('%% ---- Table IV rows (held) ----')
for s in (9, 31, 32, 33):
    for j, d in enumerate(('1', '2', '4')):
        r = R[s][d]
        lead = ('\\multirow{3}{*}{S%d}' % s) if j == 0 else ''
        print('%s & $%s$ & $%s$ & $%s$ & %s & %s & %s\\\\' % (
            lead, d, DRIFT[s][j], INNOV[s][j], L(r['plain']['held'][0]), L(r['controller']['held'][0]),
            g(r['plain - controller']['held'][0])))
    print('\\midrule' if s != 33 else '\\bottomrule')

print('\n%% ---- supplement t:hold rows: loss held (decision frames); gain over plain and marginal value, held [CI] (dec) ----')
for s in (9, 31, 32, 33):
    for d in ('1', '2', '4'):
        r = R[s][d]
        gp, vs = r['plain - controller'], r['sensor off - controller']
        print('S%d & $%s$ & %s (%s) & %s (%s) & %s (%s) & %s %s (%s) & %s %s (%s)\\\\' % (
            s, d, L(r['plain']['held'][0]), L(r['plain']['dec'][0]), L(r['controller']['held'][0]), L(r['controller']['dec'][0]),
            L(r['sensor off']['held'][0]), L(r['sensor off']['dec'][0]),
            g(gp['held'][0]), ci(gp['held']), g(gp['dec'][0]), g(vs['held'][0]), ci(vs['held']), g(vs['dec'][0])))
r9 = R[9]
print('\n%% S9 d-spaced and re-sweep rows')
for d in ('2', '4'):
    r = r9[d]
    print('S9 d-spaced d=%s: controller %s (%s); gain %s %s (%s); V %s %s (%s); vs re-sweep %s %s (%s)' % (
        d, L(r['controller, d-spaced']['held'][0]), L(r['controller, d-spaced']['dec'][0]),
        g(r['plain - controller, d-spaced']['held'][0]), ci(r['plain - controller, d-spaced']['held']), g(r['plain - controller, d-spaced']['dec'][0]),
        g(r['sensor off - controller, d-spaced']['held'][0]), ci(r['sensor off - controller, d-spaced']['held']), g(r['sensor off - controller, d-spaced']['dec'][0]),
        g(r['re-sweep ref - controller, d-spaced']['held'][0]), ci(r['re-sweep ref - controller, d-spaced']['held']), g(r['re-sweep ref - controller, d-spaced']['dec'][0])))
for d in ('1', '2', '4'):
    r = r9[d]
    print('S9 re-sweep ref d=%s: loss %s (%s), probes/decision %.2f, probes/frame %.2f; gain %s %s (%s)' % (
        d, L(r['re-sweep ref']['held'][0]), L(r['re-sweep ref']['dec'][0]), r['re-sweep ref']['dec'][1], r['re-sweep ref']['held'][1],
        g(r['re-sweep ref - controller']['held'][0]), ci(r['re-sweep ref - controller']['held']), g(r['re-sweep ref - controller']['dec'][0])))
print('\n%% probes per decision / per native frame (controller; plain; sensor off), calls % per decision / per frame')
for s in (9, 31, 32, 33):
    for d in ('1', '2', '4'):
        r = R[s][d]
        print('S%d d=%s: ctl %.2f / %.2f, calls %.1f / %.1f | plain %.2f / %.2f | off %.2f / %.2f | saving %.0f%%' % (
            s, d, r['controller']['dec'][1], r['controller']['held'][1], r['controller']['dec'][2], r['controller']['held'][2],
            r['plain']['dec'][1], r['plain']['held'][1], r['sensor off']['dec'][1], r['sensor off']['held'][1],
            100 * (r['sensor off']['dec'][1] - r['controller']['dec'][1]) / r['sensor off']['dec'][1]))
print('\n%% re-selected comparator slack (held vs native slack held)')
for s in (9, 31, 32, 33):
    for d in ('2', '4'):
        r = R[s][d]
        if 'plain, re-selected slack' in r:
            print('S%d d=%s: re-selected %s vs native %s (dec %s vs %s)' % (s, d, L(r['plain, re-selected slack']['held'][0]),
                  L(r['plain']['held'][0]), L(r['plain, re-selected slack']['dec'][0]), L(r['plain']['dec'][0])))
print('\n%% Fig. 6 decimation rows (held): (label, estimate, lo, hi)')
for d in ('2', '4'):
    v = r9[d]['sensor off - controller']['held']; print("d=%s consecutive: %.2f, %.2f, %.2f" % (d, *v))
for d in ('2', '4'):
    v = r9[d]['sensor off - controller, d-spaced']['held']; print("d=%s d-spaced: %.2f, %.2f, %.2f" % (d, *v))
for s in (31, 32, 33):
    v = R[s]['4']['sensor off - controller']['held']; print("S%d arm d=4: %.2f, %.2f, %.2f" % (s, *v))
