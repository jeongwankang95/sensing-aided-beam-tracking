#!/usr/bin/env python3
"""Final locked evaluation: multi-regime val-selected constants, run ONCE per
scenario test set, with the feedback-only tracker at its own val-selected slack."""
import numpy as np, pandas as pd, os, sys, glob, json
LK = json.load(open(sys.argv[1])); K = LK['selected']
root = sys.argv[2]
TSL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
CACHE = next((a.split('=',1)[1] for a in sys.argv if a.startswith('--cache=')), f'{root}/sensor_preds_t0.npz')
C = 64
cache = np.load(CACHE, allow_pickle=True)
PRED = {k: dict(zip(cache['anchor'], cache[k])) for k in ('radar','vision')}
if os.path.exists(f'{root}/{os.path.basename(root)}.csv'):
    base = pd.read_csv(f'{root}/{os.path.basename(root)}.csv')
else:
    base = pd.read_csv(glob.glob(f'{root}/*_dev.csv')[0])
tp = set(pd.read_csv(f'{root}/test_seqs_RA.csv')['seq_index'].unique())
passes = []
for p, g in base.groupby('seq_index'):
    if p not in tp: continue
    g = g.sort_values('index')
    P = np.stack([np.nan_to_num(np.loadtxt(os.path.join(root, str(r).lstrip('./'))), nan=0.0)
                  for r in g['unit1_pwr_60ghz']])
    passes.append((P, [str(r) for r in g['unit1_pwr_60ghz']]))

class Ctl:
    def __init__(self, sensor='vision', drop_db=2.0, refresh=16, wcap=10, **kw):
        self.sensor, self.d, self.refresh, self.wcap = sensor, drop_db, refresh, wcap
        for k in ('th_on','th_off','verify','t_trial','margin','slack'): setattr(self,k,kw[k])
        self.blf = kw['bl']
        self.rs_k, self.rs_how = kw.get('rs_k'), kw.get('rs_how', 'coarse')  # Option 3: post-setup re-sweep
    def reset(self,a0,p0):
        self.acq,self.p_acq=int(a0),float(p0); self.v,self.sig,self.n,self.fire,self.esc=0.,1.5,0,False,1
        self.a,self.mode,self.sv,self.tr,self.bl=int(a0),'track',0,0,0; self.sense=self.trial=0; self.rs_t=0; self.sweeps=0
    def _a(self,b,p,c):
        self.sig=.7*self.sig+.3*(abs(b-c)/max(self.n,1)+.1); self.v=.7*self.v+.3*(b-self.acq)/max(self.n,1)
        good=self.p_acq<=0 or 10*np.log10(max(p,1e-12)/self.p_acq)>-2*self.d
        self.esc=1 if good else min(self.esc*2,8); self.fire=not good
        self.acq,self.p_acq,self.n=b,p,0; return good
    def _rs(self,Pt):
        # Option 3 (2026-10-02): one scheduled re-sweep rs_k epochs after link setup, logic verbatim from
        # rev_20260930/rs_classes.py RS.step; returns (beam, probes) on that epoch and None otherwise
        self.rs_t+=1
        if self.rs_k is None or self.rs_t!=self.rs_k: return None
        self.sweeps+=1
        if self.bl>0: self.bl-=1
        if self.rs_how=='full': S=list(range(C))
        else:
            co=list(range(0,C,4)); cb=max(co,key=lambda b: float(Pt[b]))
            S=sorted(set(co)|set(range(max(0,cb-2),min(C,cb+3))))
        obs={b:float(Pt[b]) for b in S}; best=max(obs,key=obs.get)
        self.acq,self.p_acq,self.n=int(best),obs[best],0
        self.v,self.sig,self.esc,self.fire=0.0,1.5,1,False
        self.a=int(best)
        if self.mode!='track': self.mode='track'
        return self.a,len(S)
    def step(self,Pt,key):
        _r=self._rs(Pt)
        if _r is not None: return _r
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

class SA:
    def __init__(self,K=10**9): self.K=K
    def reset(self,a0,p0): self.a,self.sv,self.sense,self.trial=int(a0),0,0,0
    def step(self,Pt,key):
        pr=PRED['vision'].get(key)
        if pr is None: return self.a,0
        self.sense+=1; self.sv+=1; p=int(pr); k=0
        if self.sv>=self.K:
            S=sorted({max(0,p-1),p,min(C-1,p+1)}); obs={b:float(Pt[b]) for b in S}; k=len(S)
            self.a=max(obs,key=obs.get); self.sv=0
        else: self.a=p
        return self.a,k

def run(pol, pp=False):
    pc,cc,s_,t_=[],[],0,0
    for P,keys in passes:
        pol.reset(int(P[0].argmax()),float(P[0].max())); caps,cost=[],[]
        for t in range(1,len(P)):
            a,k=pol.step(P[t],keys[t]); m=float(P[t].max())
            if m>0: caps.append(float(P[t,a])/m); cost.append(k)
        pc.append(np.array(caps)); cc.append(np.array(cost))
        s_+=getattr(pol,'sense',0); t_+=getattr(pol,'trial',0)
    if pp: return pc
    caps=np.concatenate(pc); cost=np.concatenate(cc)
    loss=-10*np.log10(np.clip(caps,1e-6,None)); n=len(caps)
    return (float(loss.mean()), float(cost.mean()), 100*(s_+t_)/n,
            float(np.percentile(loss,95)), 100*float((caps<0.5).mean()))

# ---- gate-only warm-up: the tracker keeps sigma_0 = 1.5; the gate may open only after `warm`
# acquisitions, and/or reads a separate innovation EMA initialised at g0 (same update rule). ----
import json as _json
SCN = os.path.basename(root); OUT = 'results/analysis'; os.makedirs(OUT, exist_ok=True)
class CtlG(Ctl):
    def __init__(self, warm=0, g0=None, **kw): super().__init__(**kw); self.warm, self.g0 = warm, g0
    def reset(self, a0, p0):
        super().reset(a0, p0); self.nacq = 0; self.t = 0; self.trials = []; self.sig_g = self.sig if self.g0 is None else self.g0
    def _a(self, b, p, c):
        self.sig_g = .7*self.sig_g + .3*(abs(b-c)/max(self.n,1) + .1); self.nacq += 1
        return super()._a(b, p, c)
    def step(self, Pt, key):
        _r = self._rs(Pt)
        if _r is not None: return _r
        self.t += 1; self.n += 1
        if self.bl > 0: self.bl -= 1
        c = int(np.clip(round(self.acq + self.v*self.n), 0, C-1)); unc = max(self.sig, .2)*self.n
        pred = PRED[self.sensor].get(key) if self.sensor else None
        g = self.sig if self.g0 is None else self.sig_g
        if self.mode == 'track' and g > self.th_on and pred is not None and self.bl == 0 and self.nacq >= self.warm: self.mode, self.tr = 'trial', 0
        if self.mode in ('trial', 'sense') and pred is None: self.mode = 'track'
        k = 0
        if self.mode == 'trial':
            self.tr += 1; self.trial += 1; p = int(pred); S = sorted({max(0,p-1), p, min(C-1,p+1)})
            obs = {b: float(Pt[b]) for b in S}; k = len(S); best = max(obs, key=obs.get); self.a = c; g0 = float(g)
            if obs[best] >= float(Pt[self.a]) * 10**(self.margin/10):
                self.mode, self.sv = 'sense', 0; self._a(best, obs[best], c); self.a = best; self.trials.append((self.t, g0, True))
            elif self.tr >= self.t_trial: self.mode, self.bl = 'track', self.blf; self.trials.append((self.t, g0, False))
        elif self.mode == 'sense':
            self.sense += 1; self.sv += 1; p = int(pred)
            if self.sv >= self.verify or self.fire:
                S = sorted({max(0,p-1), p, min(C-1,p+1)}); obs = {b: float(Pt[b]) for b in S}; k = len(S)
                best = max(obs, key=obs.get); good = self._a(best, obs[best], p); self.sv = 0; self.a = best
                if not good: self.mode, self.bl = 'track', self.blf
            else: self.a = p
            if (self.sig if self.g0 is None else self.sig_g) < self.th_off: self.mode = 'track'
        else:
            if self.fire or unc > self.slack or self.n >= self.refresh:
                W = int(np.clip(np.ceil(unc)*self.esc, 1, self.wcap)); S = list(range(max(0,c-W), min(C,c+W+1)))
                obs = {b: float(Pt[b]) for b in S}; k = len(S); best = max(obs, key=obs.get); self._a(best, obs[best], c); self.a = best
            else: self.a = c
        rss = float(Pt[self.a])
        if self.p_acq > 0 and 10*np.log10(max(rss,1e-12)/self.p_acq) < -self.d: self.fire = True
        return self.a, k
def _trk(slack): return Ctl(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=slack)
def runx(pol):
    pc, cc, s_, t_, log = [], [], 0, 0, []
    for P, keys in passes:
        pol.reset(int(P[0].argmax()), float(P[0].max())); caps, cost = [], []
        fc = next((t for t in range(1, len(P)) if PRED['vision'].get(keys[t]) is not None), None)
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: caps.append(float(P[t, a]) / m); cost.append(k)
        pc.append(np.array(caps)); cc.append(np.array(cost)); s_ += getattr(pol, 'sense', 0); t_ += getattr(pol, 'trial', 0)
        log.append((fc, list(getattr(pol, 'trials', []))))
    caps = np.concatenate(pc); cost = np.concatenate(cc); loss = -10*np.log10(np.clip(caps, 1e-6, None))
    tr = [x for _, l in log for x in l]
    return dict(loss=float(loss.mean()), probes=float(cost.mean()), model=100*(s_+t_)/len(caps), p95=float(np.percentile(loss, 95)), pc=pc,
                trials=len(tr), won=sum(1 for x in tr if x[2]), at_first=sum(1 for fc, l in log if l and fc is not None and l[0][0] == fc),
                pw=sum(1 for _, l in log if l), open_med=float(np.median([x[0] for x in tr])) if tr else float('nan'),
                g_med=float(np.median([x[1] for x in tr])) if tr else float('nan'))
sn = lambda cs: (np.array([(-10*np.log10(np.clip(c,1e-6,None))).sum() for c in cs]), np.array([len(c) for c in cs], float))
fb = runx(_trk(TSL)); base = runx(CtlG(**K)); n = len(passes)
print('\n################ %s gate warm-up variants (fb-only %.3f; steady-state innovation medians 0.91/0.50/0.36/0.37)' % (SCN, fb['loss']))
print('%-30s | %7s %6s %7s %6s | %6s %4s %9s %8s %6s | %8s %s' % ('variant', 'loss', 'p/ep', 'model%', 'p95', 'trials', 'won', 'first-cov', 'open-med', 'g-med', 'VoI', 'deployed minus variant [95% CI]'))
V = [('deployed', dict()), ('warm 3 acq', dict(warm=3)), ('warm 6 acq', dict(warm=6)), ('warm 10 acq', dict(warm=10)), ('warm 16 acq', dict(warm=16)),
     ('gate EMA init 0.0', dict(g0=0.0)), ('gate EMA init 0.5', dict(g0=0.5)), ('gate EMA 0.0 + warm 3', dict(g0=0.0, warm=3)), ('gate EMA 0.0 + warm 6', dict(g0=0.0, warm=6))]
RES = {'scenario': SCN, 'fb': fb['loss'], 'rows': {}}
sa, na = sn(base['pc']); idx = np.random.default_rng(42).integers(0, n, (100000, n))
for nm, kw in V:
    r = runx(CtlG(**kw, **K)); sb, nb = sn(r['pc']); d = sa[idx].sum(1)/na[idx].sum(1) - sb[idx].sum(1)/nb[idx].sum(1)
    print('%-30s | %7.3f %6.2f %7.1f %6.2f | %6d %4d %4d/%-4d %8.0f %6.2f | %+8.3f %+.3f [%+.3f, %+.3f]'
          % (nm, r['loss'], r['probes'], r['model'], r['p95'], r['trials'], r['won'], r['at_first'], r['pw'], r['open_med'], r['g_med'], fb['loss']-r['loss'],
             sa.sum()/na.sum()-sb.sum()/nb.sum(), np.percentile(d, 2.5), np.percentile(d, 97.5)))
    RES['rows'][nm] = {k: r[k] for k in r if k != 'pc'}; RES['rows'][nm]['ci'] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
_json.dump(RES, open(os.path.join(OUT, SCN + '_warmup.json'), 'w'), default=float)
print('REV4_DONE %s' % SCN)
