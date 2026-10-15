#!/usr/bin/env python3
"""Cold-review check (2026-09-29), reviewer B-M1: does the Scenario-9 value of sensing survive a
feedback-only reference with an absolute lock-loss detector and a charged re-acquisition sweep
(BFR-like), or a velocity-initialized start? LLR parameters are selected on VALIDATION passes and
evaluated once on TEST. The controller (class Ctl, copied verbatim from needonly.py) is unchanged.
usage: llr.py locked_multi.json <root> <ref slack> --cache=results/predictions/both_<scn>.npz [--inherit=<json>]"""
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

# ---- cold review 6, step 4b (2026-09-30): equal-budget value of sensing. Same 183 candidates as hull3.py. For each
# controller variant: time-shared feedback loss on the lower hull at the variant's probe rate minus the variant's loss
# (equal budget), best single feedback policy within that budget minus the variant's loss, and the equal-loss ratio;
# unconditional bootstrap (hull rebuilt in each of 10^4 pass resamples, seed 42). ----
exec(open('src/llr_tail_classes.py').read())
exec(open('src/rs_classes.py').read())
import importlib.util, io, contextlib
spec = importlib.util.spec_from_file_location('sfb', 'src/strong_fb.py')
_argv = sys.argv; sys.argv = ['sfb', root, 'val']
with contextlib.redirect_stdout(io.StringIO()):
    sfb = importlib.util.module_from_spec(spec); spec.loader.exec_module(sfb)
sys.argv = _argv
VAL, TEST = load('val'), load('test')
def pp(ps, pol):
    L, Kc, N = [], [], []
    for P, keys, pid in ps:
        pol.reset(int(P[0].argmax()), float(P[0].max())); ls = ks = n = 0
        for t in range(1, len(P)):
            a, k = (pol.step(P[t], keys[t]) if not hasattr(pol, 'band') and not hasattr(pol, 'post') else pol.step(P[t]))
            m = float(P[t].max())
            if m > 0: ls += -10*np.log10(max(float(P[t, a])/m, 1e-6)); ks += k; n += 1
        L.append(ls); Kc.append(ks); N.append(n)
    return np.array(L), np.array(Kc, float), np.array(N, float)
fbk = lambda sl, **kw: LLR(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=sl, **kw)
cand = {}
for sl in (0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0): cand['trk%.1f' % sl] = pp(TEST, fbk(sl))
for X in (3, 4, 5, 6, 8, 10):
    for Nn in (1, 2, 3):
        for h in ('full', 'coarse'):
            for r in ('sweep', 'runmax'): cand['llr%d%d%s%s' % (X, Nn, h[0], r[0])] = pp(TEST, fbk(TSL, X=X, N=Nn, how=h, ref=r))
fbr = lambda sl, **kw: RS(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=sl, **kw)
for sl in (0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0):
    for h in ('full', 'coarse'):
        for kk in (2, 3, 5, 7, 10, 15, 20): cand['rs%.1f%s%d' % (sl, h[0], kk)] = pp(TEST, fbr(sl, how=h, k=kk))
CAND = []
for sv in (0.6, 1.0, 1.6):
    for eps in (0.02, 0.05, 0.10, 0.20, 0.35, 0.5, 0.65):
        for tau in (1.0, 3.0):
            for km in (5, 9, 13, 17): CAND.append(('bayes', dict(sigma_v=sv, eps=eps, tau=tau, kmax=km)))
for band in (2, 3, 5):
    for es in (0.5, 1.0, 2.0, 3.5):
        for wm in (2, 4, 6, 8): CAND.append(('viterbi', dict(band=band, ent_scale=es, wmax=wm)))
mk = lambda k, kw: sfb.Bayes(**kw) if k == 'bayes' else sfb.Viterbi(**kw)
vres = []
for k, kw in CAND:
    L, Kc, N = pp(VAL, mk(k, kw)); vres.append((L.sum()/N.sum(), Kc.sum()/N.sum(), k, kw))
for BUD in (2.6, 3.6, 5.0, 7.0, 9.0, 11.0):
    el = [v for v in vres if v[1] <= BUD]
    if not el: continue
    vl, vc, k, kw = min(el, key=lambda v: v[0])
    cand['%s@%.1f' % (k, BUD)] = pp(TEST, mk(k, kw))
    L, Kc, N = cand['%s@%.1f' % (k, BUD)]; print('  frontier %s budget %.1f %s -> test %.3f @ %.2f' % (k, BUD, kw, L.sum()/N.sum(), Kc.sum()/N.sum()))
# ---- Option 3 (2026-10-02): the re-selected method and the pre-registered sensitivity selections (configs/sel3_val.json)
SJ = json.load(open('configs/sel3_val.json'))['selected']
_m = lambda d: RS(**d['cfg']) if d['resweep'] is None else RS(how=d['resweep'][0], k=d['resweep'][1], **d['cfg'])
VARS = {'METHOD primary': pp(TEST, _m(SJ['primary'])),
        'S2 loss-only': pp(TEST, _m(SJ['S2_loss_only'])),
        'S3 no re-sweep': pp(TEST, _m(SJ['S3_no_resweep'])),
        'controller (deployed 08-10)': pp(TEST, Ctl(**K))}
names = list(cand); LS = np.stack([cand[m][0] for m in names], 1); KS = np.stack([cand[m][1] for m in names], 1); NS = cand[names[0]][2]
def hull(w):
    n = w @ NS; loss = (w @ LS)/n; pr = (w @ KS)/n
    pts = sorted(zip(pr, loss)); H = []
    for p in pts:
        while len(H) >= 2 and (H[-1][0]-H[-2][0])*(p[1]-H[-2][1]) - (H[-1][1]-H[-2][1])*(p[0]-H[-2][0]) <= 0: H.pop()
        H.append(p)
    return H, pr, loss
def at(H, xq):
    if xq < H[0][0]: return None
    for (x0, y0), (x1, y1) in zip(H, H[1:]):
        if x0 <= xq <= x1: return y0 + (xq - x0)*(y1 - y0)/(x1 - x0) if x1 > x0 else min(y0, y1)
    return None
def ratio(H, lc, kc):
    for (x0, y0), (x1, y1) in zip(H, H[1:]):
        if (y0 - lc)*(y1 - lc) <= 0 and y0 != y1: return (x0 + (lc - y0)*(x1 - x0)/(y1 - y0))/kc
    return None
def stats(w):
    H, pr, loss = hull(w); out = {}
    for nm, c in VARS.items():
        lc = (w @ c[0])/(w @ c[2]); kc = (w @ c[1])/(w @ c[2]); hb = at(H, kc)
        single = min((l for p_, l in zip(pr, loss) if p_ <= kc), default=None)
        out[nm] = (lc, kc, None if hb is None else hb - lc, None if single is None else single - lc, ratio(H, lc, kc))
    return out, H
w1 = np.ones(len(TEST)); S0, H0 = stats(w1)
print('candidates %d; hull vertices %s' % (len(names), [(round(x, 2), round(y, 3)) for x, y in H0]))
for nm, (lc, kc, vb, vs, r) in S0.items():
    print('POINT %-38s loss %.3f @ %.2f | equal-budget value (hull) %s | best single policy within budget %s | equal-loss ratio %s'
          % (nm, lc, kc, 'n/a' if vb is None else '%+.3f' % vb, 'n/a' if vs is None else '%+.3f' % vs, 'n/a' if r is None else '%.3f' % r))
rng = np.random.default_rng(42); B = {nm: ([], [], []) for nm in VARS}; miss = {nm: [0, 0, 0] for nm in VARS}
for _ in range(10000):
    w = np.bincount(rng.integers(0, len(TEST), len(TEST)), minlength=len(TEST)).astype(float)
    S, _ = stats(w)
    for nm, (lc, kc, vb, vs, r) in S.items():
        for i, v in enumerate((vb, vs, r)):
            if v is None: miss[nm][i] += 1
            else: B[nm][i].append(v)
q = lambda a: '[%+.3f, %+.3f] median %+.3f' % (np.percentile(a, 2.5), np.percentile(a, 97.5), np.median(a)) if a else 'n/a'
for nm in VARS:
    print('BOOT  %-38s equal-budget (hull) %s (undefined %d) | single within budget %s (undefined %d) | ratio %s (undefined %d)'
          % (nm, q(B[nm][0]), miss[nm][0], q(B[nm][1]), miss[nm][1], q(B[nm][2]), miss[nm][2]))
json.dump({nm: dict(point=S0[nm], boot_budget=B[nm][0], boot_single=B[nm][1], boot_ratio=B[nm][2]) for nm in VARS},
          open(('results/analysis/hull5_scenario9.json'), 'w'), default=float)
print('HULL5_DONE')
