#!/usr/bin/env python3
"""lock_loss_selection.py LOCKED ROOT SLACK --cache=CACHE [--inherit=JSON] [--resamples=B]: the feedback-only reference with
lock-loss recovery (class LLR, src/llr_tail_classes.py) and the velocity-initialized start. When the served power stays more
than X dB below a peak level for N consecutive epochs, the next epoch runs a charged full or coarse re-acquisition sweep and
restarts the tracker. The 72 settings (X, N, sweep, peak level) are selected on the scenario's validation passes by lowest loss,
with and without a cap of 4 probes/epoch, and evaluated once on test; a scenario without validation passes inherits the
selections of JSON. Also the test loss of all 72 settings, and bootstrap intervals (B resamples, default 20,000) of the value
of sensing against the controller of LOCKED (the paper's run used the earlier constants, configs/locked_multi.json).
Writes results/analysis/llr_<scenario>.json, which reproduces configs/llr_scenario*.json (Scenario 9 with --resamples=100000,
as the stored file).
Paper: supplementary Sec. S17-O: the 72 settings; within the cap validation selects X=3 dB, N=2, coarse sweep and the last
sweep's best, the best of the 35 settings within 4 probes/epoch on test; without the cap X=3 dB, N=2, full sweep and the running
maximum; on Scenario 9 0.824 dB at 3.83 probes/epoch within the cap and 0.606 dB at 8.43 without it, the lowest setting 0.474
dB at 10.9; at the low-innovation sites at most three re-acquisitions per scenario and references of 0.108, 0.055 and 0.248 dB
(2.42, 1.68 and 1.82 probes/epoch); the velocity-initialized start 0.895 dB on Scenario 9, 0.274 and 0.771 dB on 31 and 33.
usage: python analysis/lock_loss_selection.py configs/locked_multi.json dataset/scenario9 0.4 \\
           --cache=results/predictions/both_scenario9.npz --resamples=100000
       python analysis/lock_loss_selection.py configs/locked_multi.json dataset/scenario33 0.8 \\
           --cache=results/predictions/both_scenario33.npz --inherit=configs/llr_scenario32.json"""
import numpy as np, pandas as pd, os, sys, glob, json, itertools
LK = json.load(open(sys.argv[1])); K = LK['selected']
root = sys.argv[2]; TSL = float(sys.argv[3])
CACHE = next(a.split('=',1)[1] for a in sys.argv if a.startswith('--cache='))
INHERIT = next((a.split('=',1)[1] for a in sys.argv if a.startswith('--inherit=')), None)
RESAMPLES = int(next((a.split('=',1)[1] for a in sys.argv if a.startswith('--resamples=')), 20000))
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

exec(open('src/llr_tail_classes.py').read())

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
def boot(ppA, ppB, B=RESAMPLES, seed=42):
    sA = np.array([x[0].sum() for x in ppA]); sB = np.array([x[0].sum() for x in ppB]); n = np.array([len(x[0]) for x in ppA])
    pt = (sA.sum() - sB.sum())/n.sum(); rng = np.random.default_rng(seed); P = len(n); d = np.empty(B)
    for i in range(B):
        j = rng.integers(0, P, P); d[i] = (sA[j].sum() - sB[j].sum())/n[j].sum()
    return pt, np.percentile(d, 2.5), np.percentile(d, 97.5)

VAL, TEST = load('val'), load('test')
SCN = os.path.basename(root); OUT = 'results/analysis'; os.makedirs(OUT, exist_ok=True)
fb = lambda **kw: LLR(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=TSL, **kw)
ctl = lambda **kw: LLR(**K, **kw)
GRID = [dict(X=X, N=N, how=h, ref=r) for X in (3, 4, 5, 6, 8, 10) for N in (1, 2, 3)
        for h in ('full', 'coarse') for r in ('sweep', 'runmax')]
res = {'scenario': SCN, 'n_val': len(VAL), 'n_test': len(TEST), 'val': [], 'test': []}
print(f'\n######## {SCN}: {len(VAL)} val / {len(TEST)} test passes, reference slack {TSL}')
if VAL:
    ref_val = summ(perpass(VAL, fb()))
    print('val  reference tracker: loss %.3f  probes %.2f' % ref_val[:2])
    for g in GRID:
        s = summ(perpass(VAL, fb(**g))); res['val'].append((g, s)); 
    sel_free = min(res['val'], key=lambda r: r[1][0])[0]
    cap = [r for r in res['val'] if r[1][1] <= 4.0]
    sel_cap = min(cap, key=lambda r: r[1][0])[0] if cap else None
else:
    inh = json.load(open(INHERIT)); sel_free, sel_cap = inh['sel_free'], inh['sel_cap']
    print('no validation split: inheriting', sel_free, sel_cap)
res['sel_free'], res['sel_cap'] = sel_free, sel_cap
print('selected on val (min loss):', sel_free, ' (min loss, val probes <= 4):', sel_cap)
rows = [('reference tracker', fb()), ('controller (deployed)', ctl()),
        ('velocity-init tracker (2nd sweep)', fb(init2=True)),
        ('LLR tracker, val-selected', fb(**sel_free)), ('controller + same LLR', ctl(**sel_free))]
if sel_cap and sel_cap != sel_free:
    rows += [('LLR tracker, val-selected (<=4 probes)', fb(**sel_cap)), ('controller + same LLR (<=4)', ctl(**sel_cap))]
rows += [('velocity-init + LLR (val-selected)', fb(init2=True, **sel_free))]
PP = {}
print('%-42s %8s %7s %7s %6s' % ('TEST policy', 'loss dB', 'probes', 'model%', 'sweeps'))
for nm, pol in rows:
    PP[nm] = perpass(TEST, pol); s = summ(PP[nm]); res['test'].append((nm, s))
    print('%-42s %8.3f %7.2f %7.1f %6d' % (nm, s[0], s[1], s[2], s[3]))
for a, b in [('reference tracker', 'controller (deployed)'), ('LLR tracker, val-selected', 'controller (deployed)'),
             ('LLR tracker, val-selected', 'controller + same LLR'),
             ('LLR tracker, val-selected (<=4 probes)', 'controller (deployed)'),
             ('LLR tracker, val-selected (<=4 probes)', 'controller + same LLR (<=4)'),
             ('velocity-init tracker (2nd sweep)', 'controller (deployed)')]:
    if a in PP and b in PP:
        pt, lo, hi = boot(PP[a], PP[b]); print('value of sensing  [%s] - [%s]: %+.3f [%+.3f, %+.3f]' % (a, b, pt, lo, hi))
        res.setdefault('voi', []).append((a, b, pt, lo, hi))
print('\nTEST grid (all 72 LLR settings, sorted by probes):')
for g in GRID:
    s = summ(perpass(TEST, fb(**g))); res.setdefault('test_grid', []).append((g, s))
for g, s in sorted(res['test_grid'], key=lambda r: r[1][1]):
    print('  X=%-3g N=%d %-6s %-6s  loss %.3f  probes %.2f  sweeps %d' % (g['X'], g['N'], g['how'], g['ref'], s[0], s[1], s[3]))
w4 = [(g, s) for g, s in res['test_grid'] if s[1] <= 4.0]
if w4:
    g, s = min(w4, key=lambda r: r[1][0])
    print('  %d of the 72 settings within 4 probes/epoch on test; lowest loss among them: X=%g N=%d %s %s  loss %.3f  probes %.2f'
          % (len(w4), g['X'], g['N'], g['how'], g['ref'], s[0], s[1]))
json.dump(res, open(os.path.join(OUT, f'llr_{SCN}.json'), 'w'), default=float, indent=1)
print('LLR_DONE', SCN)
