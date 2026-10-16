"""decimation_correlation.py: across the twelve scenario-by-decimation cells of Table IV, the Spearman rank correlation of the
controller's gain over the plain tracker with the vision arm's current-slot accuracy (consecutive inputs), the feedback-only
tracker's innovation median, and the best-beam drift.
Gains: "plain - controller" in results/analysis/hold_scenario<s>.json, every-frame ("held") scoring and decision frames ("dec");
accuracy: the "accuracy, consecutive inputs [cache]" line of results/analysis/table3_s<s>.log; drift and innovation: the
decimation table of results/analysis/decimation_s<s>.log (as printed in Table IV).
Paper: Sec. VI-C, Spearman +0.72 with accuracy, +0.53 with innovation and +0.27 with drift (every-frame scoring).
Ranks with ties averaged (as scipy.stats.spearmanr).
usage: python analysis/decimation_correlation.py"""
import json
import os
import re

import numpy as np

A = os.path.join('results', 'analysis')
S, D = (9, 31, 32, 33), ('1', '2', '4')


def rank(x):
    x = np.asarray(x, float); o = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        r[x == v] = r[x == v].mean()
    return r


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


R = {s: json.load(open(os.path.join(A, f'hold_scenario{s}.json'))) for s in S}
ACC, DRIFT, INNOV = {}, {}, {}
for s in S:
    m = next(re.search(r'accuracy, consecutive inputs \[cache\]: d=1 ([\d.]+)%\s+d=2 ([\d.]+)%\s+d=4 ([\d.]+)%', l)
             for l in open(os.path.join(A, f'table3_s{s}.log')) if 'accuracy, consecutive inputs [cache]' in l)
    ACC[s] = [float(x) for x in m.groups()]
    rows = {m.group(1): m.groups()[1:] for m in (re.match(r'\s*([124])\s+(\S+)\s+(\S+)\s+\|', l)
                                                 for l in open(os.path.join(A, f'decimation_s{s}.log'))) if m}
    DRIFT[s], INNOV[s] = [float(rows[d][0]) for d in D], [float(rows[d][1]) for d in D]
for tag in ('held', 'dec'):
    g, a, i, dr = [], [], [], []
    for s in S:
        for j, d in enumerate(D):
            g.append(R[s][d]['plain - controller'][tag][0]); a.append(ACC[s][j]); i.append(INNOV[s][j]); dr.append(DRIFT[s][j])
    print(tag, 'accuracy %+.3f innovation %+.3f drift %+.3f' % (spearman(g, a), spearman(g, i), spearman(g, dr)))
