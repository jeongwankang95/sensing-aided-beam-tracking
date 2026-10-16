#!/usr/bin/env python3
"""prevalence.py: lock-loss prevalence of the reference tracker (slack 0.4) on all 136 Scenario-9 passes, and whether one
scheduled coarse re-sweep (k=5, slack 0.4 and 1.0) removes it. Lock loss = pass-mean loss >= 2 dB.
Paper: Sec. VI-A, lock loss on 22% of the 136 passes (25% of the 28 test passes) and none after one re-sweep. The tracker and
the coarse sweep are verbatim from the re-sweep analysis (track-mode logic of the lock-loss-recovery tracker); every pass
outside the validation and test lists is a training pass.
usage: python analysis/prevalence.py"""
import csv, os, numpy as np, math
root = 'dataset/scenario9'
rows = list(csv.DictReader(open(f'{root}/scenario9.csv')))
byp = {}
for r in rows: byp.setdefault(r['seq_index'], []).append(r)
C = 64

class Ctl:  # track-mode logic of llr.py Ctl (sensor=None), verbatim
    def __init__(self, slack, drop_db=2.0, refresh=16, wcap=10):
        self.d, self.refresh, self.wcap, self.slack = drop_db, refresh, wcap, slack
    def reset(self, a0, p0):
        self.acq, self.p_acq = int(a0), float(p0); self.v, self.sig, self.n, self.fire, self.esc = 0., 1.5, 0, False, 1
        self.a = int(a0)
    def _a(self, b, p, c):
        self.sig = .7*self.sig + .3*(abs(b-c)/max(self.n, 1) + .1); self.v = .7*self.v + .3*(b-self.acq)/max(self.n, 1)
        good = self.p_acq <= 0 or 10*np.log10(max(p, 1e-12)/self.p_acq) > -2*self.d
        self.esc = 1 if good else min(self.esc*2, 8); self.fire = not good
        self.acq, self.p_acq, self.n = b, p, 0; return good
    def step(self, Pt):
        self.n += 1
        c = int(np.clip(round(self.acq + self.v*self.n), 0, C-1)); unc = max(self.sig, .2)*self.n; k = 0
        if self.fire or unc > self.slack or self.n >= self.refresh:
            W = int(np.clip(np.ceil(unc)*self.esc, 1, self.wcap)); S = list(range(max(0, c-W), min(C, c+W+1)))
            obs = {b: float(Pt[b]) for b in S}; k = len(S); best = max(obs, key=obs.get)
            self._a(best, obs[best], c); self.a = best
        else: self.a = c
        rss = float(Pt[self.a])
        if self.p_acq > 0 and 10*np.log10(max(rss, 1e-12)/self.p_acq) < -self.d: self.fire = True
        return self.a, k

def sweepset(Pt, how):
    if how == 'full': return list(range(C))
    S0 = list(range(0, C, 4)); b = max(S0, key=lambda x: Pt[x])
    return sorted(set(S0) | set(range(max(0, b-2), min(C, b+3))))

va = {r['seq_index'] for r in csv.DictReader(open(f'{root}/val_seqs_RA.csv'))}
te = {r['seq_index'] for r in csv.DictReader(open(f'{root}/test_seqs_RA.csv'))}
tr = set(byp) - va - te   # the training list holds every other pass (and the validation passes, which lab() names first)
def lab(p): return 'test' if p in te else 'val' if p in va else 'train' if p in tr else '?'
def passloss(P, slack, how=None, k=None):
    pol = Ctl(slack); pol.reset(int(P[0].argmax()), float(P[0].max())); L = []
    for t in range(1, len(P)):
        if how and t == k:
            S = sweepset(P[t], how); b = max(S, key=lambda x: P[t, x]); pol.reset(b, float(P[t, b])); a = b
        else: a, _ = pol.step(P[t])
        m = float(P[t].max())
        if m > 0: L.append(-10*np.log10(max(float(P[t, a])/m, 1e-6)))
    return np.mean(L), len(L)
def wilson(k, n, z=1.96):
    p = k/n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d; return c-h, c+h
res = {}
for p in sorted(byp, key=int):
    g = sorted(byp[p], key=lambda r: int(r['index']))
    if len(g) < 10: continue
    P = np.stack([np.nan_to_num(np.loadtxt(os.path.join(root, r['unit1_pwr_60ghz'].lstrip('./'))), nan=0.0) for r in g])
    res[p] = (lab(p), passloss(P, 0.4), passloss(P, 0.4, 'coarse', 5), passloss(P, 1.0, 'coarse', 5), int(P[0].argmax()))
for sp in ('train', 'val', 'test', 'all'):
    R = [v for v in res.values() if sp == 'all' or v[0] == sp]
    n = len(R); k = sum(v[1][0] >= 2 for v in R); lo, hi = wilson(k, n)
    k2 = sum(v[2][0] >= 2 for v in R); k3 = sum(v[3][0] >= 2 for v in R)
    fw = lambda i: sum(v[i][0]*v[i][1] for v in R)/sum(v[i][1] for v in R)
    print('%-5s passes %3d | lock loss (ref >= 2 dB) %2d = %4.1f%% [%4.1f, %4.1f] | ref %.3f dB | re-sweep k=5 coarse: slack0.4 %.3f (%d lock) slack1.0 %.3f (%d lock) | setup argmax 14 among lock-loss: %d'
          % (sp, n, k, 100*k/n, 100*lo, 100*hi, fw(1), fw(2), k2, fw(3), k3, sum(v[1][0] >= 2 and v[4] == 14 for v in R)))
