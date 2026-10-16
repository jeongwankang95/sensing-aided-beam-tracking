#!/usr/bin/env python3
"""reselection_test.py LOCKED ROOT SLACK --cache=CACHE: the one-time test evaluation of the selections made on validation by
analysis/selection.py (configs/sel3_val.json): the controller, its sensitivity selections and the lock-loss-recovery and
re-sweep references selected with the same probe condition, with paired bootstrap intervals (10^5 resamples, seed 42) against
the plain tracker and the re-sweep references; LOCKED (the earlier constants) gives the "deployed 08-10" row. Writes
results/analysis/test3_<scenario>.json.
Paper: supplementary Sec. S17-N and Table S21 (test columns), Secs. S17-O and S17-P and Tables S22-S23.
usage: python analysis/reselection_test.py configs/locked_multi.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz"""
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

# ---- cold review 6, D1 (2026-09-30): one scheduled re-acquisition sweep after link setup ----
class RS(Ctl):
    """Ctl plus one scheduled charged re-acquisition sweep k epochs after link setup: full (all 64 beams) or
    coarse (every 4th beam, then +-2 about the best coarse beam, same epoch). The tracker restarts there as at
    link setup (v=0, sigma=sigma0, escalation 1, alarm cleared), exactly as the LLR sweep of llr.py; a policy
    not in TRACK hands back to TRACK without a blacklist. k=None gives the plain policy."""
    def __init__(self, k=None, how='coarse', **kw):
        super().__init__(**kw); self.k, self.how = k, how
    def reset(self, a0, p0):
        super().reset(a0, p0); self.t = 0; self.sweeps = 0
    def _sweep(self, Pt):
        if self.how == 'full': S = list(range(C))
        else:
            co = list(range(0, C, 4)); cb = max(co, key=lambda b: float(Pt[b]))
            S = sorted(set(co) | set(range(max(0, cb-2), min(C, cb+3))))
        obs = {b: float(Pt[b]) for b in S}; best = max(obs, key=obs.get)
        return best, obs[best], len(S)
    def step(self, Pt, key):
        self.t += 1
        if self.k is not None and self.t == self.k:
            self.sweeps += 1
            if self.bl > 0: self.bl -= 1
            best, pb, kk = self._sweep(Pt)
            self.acq, self.p_acq, self.n = int(best), pb, 0
            self.v, self.sig, self.esc, self.fire = 0.0, 1.5, 1, False
            self.a = int(best)
            if self.mode != 'track': self.mode = 'track'
            return self.a, kk
        return super().step(Pt, key)

class LLR(Ctl):
    """Ctl plus an absolute lock-loss detector. When the served power stays more than X dB below
    a reference power for N consecutive epochs, the next epoch runs a charged re-acquisition sweep
    (full: all 64 beams; coarse: every 4th beam, then +-2 about the best coarse beam, same epoch)
    and restarts the tracker there as at link setup (v=0, sigma=sigma0, escalation 1). Reference:
    best power of the last sweep ('sweep') or running max of acquired powers since it ('runmax').
    init2=True adds a charged second full sweep at the first scored epoch and initializes the drift
    estimate from the two sweeps (velocity-initialized start)."""
    def __init__(self, X=None, N=2, how='full', ref='sweep', init2=False, **kw):
        super().__init__(**kw); self.X, self.N, self.how, self.ref, self.init2 = X, N, how, ref, init2
    def reset(self, a0, p0):
        super().reset(a0, p0); self.pref = float(p0); self.cnt = 0; self.llr = False
        self.sweeps = 0; self.t = 0; self.a0 = int(a0)
    def _sweep(self, Pt):
        if self.how == 'full': S = list(range(C))
        else:
            co = list(range(0, C, 4)); cb = max(co, key=lambda b: float(Pt[b]))
            S = sorted(set(co) | set(range(max(0, cb-2), min(C, cb+3))))
        obs = {b: float(Pt[b]) for b in S}; best = max(obs, key=obs.get)
        return best, obs[best], len(S)
    def step(self, Pt, key):
        self.t += 1
        if (self.init2 and self.t == 1) or self.llr:
            first = self.init2 and self.t == 1
            self.llr = False; self.sweeps += 1
            if self.bl > 0: self.bl -= 1
            best, pb, k = self._sweep(Pt) if not first else (int(Pt.argmax()), float(Pt.max()), C)
            v0 = float(best - self.acq) if first else 0.0
            self.acq, self.p_acq, self.n = int(best), pb, 0
            self.v, self.sig, self.esc, self.fire = v0, 1.5, 1, False
            self.pref, self.cnt, self.a = pb, 0, int(best)
            if self.mode != 'track': self.mode = 'track'
            return self.a, k
        a, k = super().step(Pt, key)
        if self.X is None: return a, k
        if self.ref == 'runmax' and self.p_acq > self.pref: self.pref = self.p_acq
        rss = float(Pt[a])
        self.cnt = self.cnt + 1 if (self.pref > 0 and 10*np.log10(max(rss, 1e-12)/self.pref) < -self.X) else 0
        if self.cnt >= self.N: self.llr = True
        return a, k


# ======================= Option 3 TEST evaluation (2026-10-02) =======================
# Each selection, declared before the validation run and made on validation by analysis/selection.py, is evaluated ONCE here.
# usage: python analysis/reselection_test.py configs/locked_multi.json <root> <ref slack> --cache=results/predictions/both_<scn>.npz
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
def prange(ppA, ppB):
    d = [(a[0].sum() - b[0].sum())/len(a[0]) for a, b in zip(ppA, ppB)]; return min(d), max(d)
def lockloss(pp): return int(sum(x[0].mean() >= 2.0 for x in pp if len(x[0])))

TEST = load('test'); SCN = os.path.basename(root); s = int(SCN.replace('scenario', ''))
fs = s if s in (9, 31, 32) else 32                     # S33 has no validation split: feedback selections inherit S32's
SJ = json.load(open('configs/sel3_val.json'))
FBK = dict(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0)
def mk(d, drop_rs=False):
    ro = None if drop_rs else d['resweep']
    return RS(**d['cfg']) if ro is None else RS(how=ro[0], k=ro[1], **d['cfg'])
SEL = SJ['selected']
assert SEL['S1_frame_weighted']['cfg'] == SEL['primary']['cfg'] and SEL['S1_frame_weighted']['resweep'] == SEL['primary']['resweep']
rsd = SJ['families'][f'resweep_S{fs}']['same_criterion'][0]; lld = SJ['families'][f'llr_S{fs}']['same_criterion'][0]
OLDFREE = {9: dict(slack=0.4, how='full', k=3), 31: dict(slack=0.4, how='coarse', k=2), 32: dict(slack=0.8, how='full', k=2)}[fs]
rows = [('reference tracker', RS(slack=TSL, **FBK)),
        ('re-sweep tracker, same criterion', RS(how=rsd['how'], k=rsd['k'], **dict(FBK, slack=rsd['slack']))),
        ('LLR tracker, same criterion', LLR(**dict(FBK, slack=TSL), **lld)),
        ('re-sweep tracker, any budget (09-30)', RS(how=OLDFREE['how'], k=OLDFREE['k'], **dict(FBK, slack=OLDFREE['slack'])))]
if s == 9: rows.append(('re-sweep tracker, <=4 probes (09-30)', RS(how='coarse', k=5, **dict(FBK, slack=1.0))))
rows += [('controller, deployed 08-10', RS(**K)),
         ('METHOD primary (= S1)', mk(SEL['primary'])),
         ('S2 loss-only', mk(SEL['S2_loss_only'])),
         ('S3 no re-sweep', mk(SEL['S3_no_resweep'])),
         ('post hoc: primary without its re-sweep', mk(SEL['primary'], drop_rs=True))]
print(f'\n######## {SCN}: {len(TEST)} test passes; reference slack {TSL}; feedback selections from S{fs}')
print('  primary:', SEL['primary']['cfg'], SEL['primary']['resweep'])
print('  re-sweep same criterion:', rsd, ' LLR same criterion:', lld, ' re-sweep any budget:', OLDFREE)
PP = {}; res = {'scenario': SCN, 'test': [], 'boot': []}
print('%-42s %8s %7s %7s %6s %9s' % ('TEST policy', 'loss dB', 'probes', 'model%', 'sweeps', 'lockloss'))
for nm, pol in rows:
    PP[nm] = perpass(TEST, pol); sm = summ(PP[nm]); res['test'].append((nm, sm, lockloss(PP[nm])))
    print('%-42s %8.3f %7.2f %7.1f %6d %9d' % (nm, sm[0], sm[1], sm[2], sm[3], lockloss(PP[nm])))
    res.setdefault('perpass', {})[nm] = [[float(x[0].sum()), int(len(x[0])), float(x[1].sum()), int(x[3]), str(x[2])] for x in PP[nm]]
M = 'METHOD primary (= S1)'
pairs = [(r, M) for r, _ in rows if r != M] + [('reference tracker', 'S2 loss-only'), ('reference tracker', 'S3 no re-sweep'),
         ('re-sweep tracker, same criterion', 'S3 no re-sweep'), ('re-sweep tracker, any budget (09-30)', 'S2 loss-only'),
         ('reference tracker', 'controller, deployed 08-10')]
print('value = first minus second (positive: second has lower loss); 10^5 paired pass bootstrap, seed 42'
      + ('; per-pass range shown (3 passes)' if s == 32 else ''))
for a, b in pairs:
    pt, lo, hi = boot(PP[a], PP[b]); extra = ('  range %+.3f..%+.3f' % prange(PP[a], PP[b])) if len(TEST) <= 3 else ''
    print('  [%-38s] - [%-30s] %+.3f [%+.3f, %+.3f]%s' % (a, b, pt, lo, hi, extra)); res['boot'].append((a, b, pt, lo, hi))
json.dump(res, open(f'results/analysis/test3_{SCN}.json', 'w'), default=float)
print('TEST3_DONE', SCN)
