#!/usr/bin/env python3
"""resweep_selection.py LOCKED ROOT SLACK --cache=CACHE [--inherit=JSON]: the feedback-only reference with one scheduled
re-sweep (class RS, src/rs_classes.py): k epochs after link setup a charged full or coarse re-acquisition, after which the
tracker restarts as at setup. The 98 settings (slack, sweep, k) are selected on the scenario's validation passes by lowest
loss, with and without a cap of 4 probes/epoch, and evaluated once on test; a scenario without validation passes inherits the
selections of JSON. Also the lock-loss passes of each policy (pass-mean loss of at least 2 dB), paired bootstrap intervals
(10^5 resamples) against the controller of LOCKED (the paper's run used the earlier constants, configs/locked_multi.json), and
the test loss of all 98 settings. Writes results/analysis/resweep_<scenario>.json, which reproduces configs/resweep_scenario*.json.
Paper: supplementary Sec. S17-P and Table S23 (lowest validation loss): the 98 settings; selections 0.4/full/3 on Scenario 9,
0.4/coarse/2 on 31, 0.8/full/2 on 32 (33 inherits 32); on Scenario 9 within 4 probes/epoch 1.0/coarse/5 at 0.370 dB and 3.78
probes/epoch, with which no Scenario-9 test pass remains in lock loss.
usage: python analysis/resweep_selection.py configs/locked_multi.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz
       python analysis/resweep_selection.py configs/locked_multi.json dataset/scenario33 0.8 \\
           --cache=results/predictions/both_scenario33.npz --inherit=configs/resweep_scenario32.json"""
import numpy as np, pandas as pd, os, sys, glob, json, itertools
LK = json.load(open(sys.argv[1])); K = LK['selected']
root = sys.argv[2]; TSL = float(sys.argv[3])
CACHE = next(a.split('=',1)[1] for a in sys.argv if a.startswith('--cache='))
INHERIT = next((a.split('=',1)[1] for a in sys.argv if a.startswith('--inherit=')), None)
C = 64
cache = np.load(CACHE, allow_pickle=True)
PRED = {k: dict(zip(cache['anchor'], cache[k])) for k in ('radar','vision')}
if os.path.exists(f'{root}/{os.path.basename(root)}.csv'):
    base = pd.read_csv(f'{root}/{os.path.basename(root)}.csv')
else:
    base = pd.read_csv(glob.glob(f'{root}/*_dev.csv')[0])
def load(split):
    f = f'{root}/{split}_seqs_RA.csv'
    if not os.path.exists(f): return []
    sel = set(pd.read_csv(f)['seq_index'].unique()); out = []
    for p, g in base.groupby('seq_index'):
        if p not in sel: continue
        g = g.sort_values('index')
        P = np.stack([np.nan_to_num(np.loadtxt(os.path.join(root, str(r).lstrip('./'))), nan=0.0)
                      for r in g['unit1_pwr_60ghz']])
        out.append((P, [str(r) for r in g['unit1_pwr_60ghz']], p))
    return out

class Ctl:
    def __init__(self, sensor='vision', drop_db=2.0, refresh=16, wcap=10, **kw):
        self.sensor, self.d, self.refresh, self.wcap = sensor, drop_db, refresh, wcap
        for k in ('th_on','th_off','verify','t_trial','margin','slack'): setattr(self,k,kw[k])
        self.blf = kw['bl']
    def reset(self,a0,p0):
        self.acq,self.p_acq=int(a0),float(p0); self.v,self.sig,self.n,self.fire,self.esc=0.,1.5,0,False,1
        self.a,self.mode,self.sv,self.tr,self.bl=int(a0),'track',0,0,0; self.sense=self.trial=0
    def _a(self,b,p,c):
        self.sig=.7*self.sig+.3*(abs(b-c)/max(self.n,1)+.1); self.v=.7*self.v+.3*(b-self.acq)/max(self.n,1)
        good=self.p_acq<=0 or 10*np.log10(max(p,1e-12)/self.p_acq)>-2*self.d
        self.esc=1 if good else min(self.esc*2,8); self.fire=not good
        self.acq,self.p_acq,self.n=b,p,0; return good
    def step(self,Pt,key):
        self.n+=1
        if self.bl>0: self.bl-=1
        c=int(np.clip(round(self.acq+self.v*self.n),0,C-1)); unc=max(self.sig,.2)*self.n
        pred=PRED[self.sensor].get(key) if self.sensor else None
        if self.mode=='track' and self.sig>self.th_on and pred is not None and self.bl==0: self.mode,self.tr='trial',0
        if self.mode in ('trial','sense') and pred is None: self.mode='track'
        k=0
        if self.mode=='trial':
            self.tr+=1; self.trial+=1; p=int(pred); S=sorted({max(0,p-1),p,min(C-1,p+1)})
            obs={b:float(Pt[b]) for b in S}; k=len(S); best=max(obs,key=obs.get); self.a=c
            if obs[best]>=float(Pt[self.a])*10**(self.margin/10):
                self.mode,self.sv='sense',0; self._a(best,obs[best],c); self.a=best
            elif self.tr>=self.t_trial: self.mode,self.bl='track',self.blf
        elif self.mode=='sense':
            self.sense+=1; self.sv+=1; p=int(pred)
            if self.sv>=self.verify or self.fire:
                S=sorted({max(0,p-1),p,min(C-1,p+1)}); obs={b:float(Pt[b]) for b in S}; k=len(S)
                best=max(obs,key=obs.get); good=self._a(best,obs[best],p); self.sv=0; self.a=best
                if not good: self.mode,self.bl='track',self.blf
            else: self.a=p
            if self.sig<self.th_off: self.mode='track'
        else:
            if self.fire or unc>self.slack or self.n>=self.refresh:
                W=int(np.clip(np.ceil(unc)*self.esc,1,self.wcap)); S=list(range(max(0,c-W),min(C,c+W+1)))
                obs={b:float(Pt[b]) for b in S}; k=len(S); best=max(obs,key=obs.get)
                self._a(best,obs[best],c); self.a=best
            else: self.a=c
        rss=float(Pt[self.a])
        if self.p_acq>0 and 10*np.log10(max(rss,1e-12)/self.p_acq)<-self.d: self.fire=True
        return self.a,k

exec(open('src/rs_classes.py').read())
# ---- cold review 6, D1 (2026-09-30): reference = tracker + one scheduled re-sweep, selected on VALIDATION over
# slack x {full, coarse} x k (98 settings; min loss, and min loss with validation probes <= 4); test once.
# Compared with the deployed controller and with the controller given the same re-sweep. Paired pass bootstrap
# 10^5 (same routine and seed as analysis/lock_loss_selection.py).
def perpass(passes, pol):
    out = []
    for P, keys, pid in passes:
        pol.reset(int(P[0].argmax()), float(P[0].max())); L = []; Kc = []
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: L.append(-10*np.log10(max(float(P[t, a])/m, 1e-6))); Kc.append(k)
        out.append((np.array(L), np.array(Kc), pid, getattr(pol, 'sense', 0) + getattr(pol, 'trial', 0), getattr(pol, 'sweeps', 0)))
    return out
def summ(pp):
    L = np.concatenate([x[0] for x in pp]); Kc = np.concatenate([x[1] for x in pp])
    return float(L.mean()), float(Kc.mean()), 100*sum(x[3] for x in pp)/len(L), sum(x[4] for x in pp)
def boot(ppA, ppB, B=100000, seed=42):
    sA = np.array([x[0].sum() for x in ppA]); sB = np.array([x[0].sum() for x in ppB]); n = np.array([len(x[0]) for x in ppA])
    pt = (sA.sum() - sB.sum())/n.sum(); rng = np.random.default_rng(seed); P = len(n); d = np.empty(B)
    for i in range(B):
        j = rng.integers(0, P, P); d[i] = (sA[j].sum() - sB[j].sum())/n[j].sum()
    return pt, np.percentile(d, 2.5), np.percentile(d, 97.5)
def lockloss(pp): return int(sum(x[0].mean() >= 2.0 for x in pp if len(x[0])))

VAL, TEST = load('val'), load('test')
SCN = os.path.basename(root); OUT = 'results/analysis'; os.makedirs(OUT, exist_ok=True)
fb = lambda sl, **kw: RS(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=sl, **kw)
ctl = lambda **kw: RS(**K, **kw)
GRID = [dict(slack=s, how=h, k=k) for s in (0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0) for h in ('full', 'coarse')
        for k in (2, 3, 5, 7, 10, 15, 20)]
res = {'scenario': SCN, 'n_val': len(VAL), 'n_test': len(TEST), 'val': [], 'test': []}
print(f'\n######## {SCN}: {len(VAL)} val / {len(TEST)} test passes, reference slack {TSL}')
if VAL:
    rv = summ(perpass(VAL, fb(TSL))); print('val  reference tracker: loss %.3f  probes %.2f' % rv[:2])
    for g in GRID:
        s = summ(perpass(VAL, fb(g['slack'], how=g['how'], k=g['k']))); res['val'].append((g, s))
    sel_free = min(res['val'], key=lambda r: r[1][0])[0]
    cap = [r for r in res['val'] if r[1][1] <= 4.0]
    sel_cap = min(cap, key=lambda r: r[1][0])[0] if cap else None
else:
    inh = json.load(open(INHERIT)); sel_free, sel_cap = inh['sel_free'], inh['sel_cap']
    print('no validation split: inheriting', sel_free, sel_cap)
res['sel_free'], res['sel_cap'] = sel_free, sel_cap
print('selected on val (min loss):', sel_free, ' (min loss, val probes <= 4):', sel_cap)
rows = [('reference tracker', fb(TSL)), ('controller (deployed)', ctl()),
        ('re-sweep tracker, val-selected', fb(sel_free['slack'], how=sel_free['how'], k=sel_free['k'])),
        ('controller + same re-sweep', ctl(how=sel_free['how'], k=sel_free['k']))]
if sel_cap and sel_cap != sel_free:
    rows += [('re-sweep tracker, val-selected (<=4 probes)', fb(sel_cap['slack'], how=sel_cap['how'], k=sel_cap['k'])),
             ('controller + same re-sweep (<=4)', ctl(how=sel_cap['how'], k=sel_cap['k']))]
PP = {}
print('%-46s %8s %7s %7s %6s %9s' % ('TEST policy', 'loss dB', 'probes', 'model%', 'sweeps', 'lockloss'))
for nm, pol in rows:
    PP[nm] = perpass(TEST, pol); s = summ(PP[nm]); res['test'].append((nm, s, lockloss(PP[nm])))
    print('%-46s %8.3f %7.2f %7.1f %6d %9d' % (nm, s[0], s[1], s[2], s[3], lockloss(PP[nm])))
    res.setdefault('perpass', {})[nm] = [[float(x[0].sum()), int(len(x[0])), float(x[1].sum()), int(x[3]), str(x[2])] for x in PP[nm]]
pairs = [('reference tracker', 'controller (deployed)'),
         ('re-sweep tracker, val-selected', 'controller (deployed)'), ('re-sweep tracker, val-selected', 'controller + same re-sweep'),
         ('re-sweep tracker, val-selected (<=4 probes)', 'controller (deployed)'),
         ('re-sweep tracker, val-selected (<=4 probes)', 'controller + same re-sweep (<=4)'),
         ('controller (deployed)', 'controller + same re-sweep'), ('controller (deployed)', 'controller + same re-sweep (<=4)')]
for a, b in pairs:
    if a in PP and b in PP:
        pt, lo, hi = boot(PP[a], PP[b]); print('paired  [%s] - [%s]: %+.3f [%+.3f, %+.3f]' % (a, b, pt, lo, hi))
        res.setdefault('boot', []).append((a, b, pt, lo, hi))
print('\nTEST grid (98 re-sweep settings, sorted by probes; per-pass sums saved):')
for g in GRID:
    pp = perpass(TEST, fb(g['slack'], how=g['how'], k=g['k'])); s = summ(pp)
    res.setdefault('test_grid', []).append((g, s, [[float(x[0].sum()), int(len(x[0])), float(x[1].sum())] for x in pp]))
for g, s, _ in sorted(res['test_grid'], key=lambda r: r[1][1]):
    print('  slack=%-3g %-6s k=%-2d  loss %.3f  probes %.2f' % (g['slack'], g['how'], g['k'], s[0], s[1]))
json.dump(res, open(os.path.join(OUT, f'resweep_{SCN}.json'), 'w'), default=float)
print('RESWEEP_DONE', SCN)
