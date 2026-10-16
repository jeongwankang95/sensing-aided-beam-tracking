#!/usr/bin/env python3
"""controls_gate.py LOCKED ROOT SLACK --cache=CACHE: variants of the gate and the blacklist: an initial innovation estimate
below the gate threshold (the gate opens only on measured innovation), blacklisting only proposals whose probed neighborhood
falls below the served beam (proposals merely within the margin get a short blacklist), and both; the selected controller is
reproduced first. Paired bootstrap against the selected controller. Writes results/analysis/<scenario>_variants.json.
Paper: supplementary Sec. S17-C (the blacklist variant changes Scenario-9 loss by +0.003 dB [-0.013, +0.024]).
usage: python analysis/controls_gate.py configs/locked_v3.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz"""
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
        # src/rs_classes.py RS.step; returns (beam, probes) on that epoch and None otherwise
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

# ---- revision follow-up, 2026-09-22: (a) sigma_0 below theta_on (gate opens only on measured
# innovation); (b) blacklist only WRONG proposals (probed best below the served beam), not
# REDUNDANT ones (within the margin); (c) both. Deployed controller reproduced first. ----
import json as _json
SCN = os.path.basename(root)
OUT = 'results/analysis'; os.makedirs(OUT, exist_ok=True)

class CtlX(Ctl):
    """Ctl with sigma_0 (s0), blacklist for wrong proposals (bl_wrong) and for redundant ones (bl_red)."""
    def __init__(self, s0=1.5, bl_red=None, **kw):
        super().__init__(**kw); self.s0 = s0; self.bl_red = self.blf if bl_red is None else bl_red
    def reset(self, a0, p0):
        super().reset(a0, p0); self.sig = self.s0; self.t = 0; self.trials = []; self.n_red = self.n_wrong = 0
    def step(self, Pt, key):
        _r = self._rs(Pt)
        if _r is not None: return _r
        self.t += 1; self.n += 1
        if self.bl > 0: self.bl -= 1
        c = int(np.clip(round(self.acq + self.v*self.n), 0, C-1)); unc = max(self.sig, .2)*self.n
        pred = PRED[self.sensor].get(key) if self.sensor else None
        if self.mode == 'track' and self.sig > self.th_on and pred is not None and self.bl == 0: self.mode, self.tr = 'trial', 0
        if self.mode in ('trial', 'sense') and pred is None: self.mode = 'track'
        k = 0
        if self.mode == 'trial':
            self.tr += 1; self.trial += 1; p = int(pred); S = sorted({max(0,p-1), p, min(C-1,p+1)})
            obs = {b: float(Pt[b]) for b in S}; k = len(S); best = max(obs, key=obs.get); self.a = c
            s0 = float(self.sig)
            if obs[best] >= float(Pt[self.a]) * 10**(self.margin/10):
                self.mode, self.sv = 'sense', 0; self._a(best, obs[best], c); self.a = best
                self.trials.append((self.t, s0, 'won'))
            elif self.tr >= self.t_trial:
                wrong = obs[best] < float(Pt[self.a])          # neighborhood worse than the served beam
                self.mode = 'track'; self.bl = self.blf if wrong else self.bl_red
                self.n_wrong += wrong; self.n_red += (not wrong)
                self.trials.append((self.t, s0, 'wrong' if wrong else 'redundant'))
        elif self.mode == 'sense':
            self.sense += 1; self.sv += 1; p = int(pred)
            if self.sv >= self.verify or self.fire:
                S = sorted({max(0,p-1), p, min(C-1,p+1)}); obs = {b: float(Pt[b]) for b in S}; k = len(S)
                best = max(obs, key=obs.get); good = self._a(best, obs[best], p); self.sv = 0; self.a = best
                if not good: self.mode, self.bl = 'track', self.blf
            else: self.a = p
            if self.sig < self.th_off: self.mode = 'track'
        else:
            if self.fire or unc > self.slack or self.n >= self.refresh:
                W = int(np.clip(np.ceil(unc)*self.esc, 1, self.wcap)); S = list(range(max(0,c-W), min(C,c+W+1)))
                obs = {b: float(Pt[b]) for b in S}; k = len(S); best = max(obs, key=obs.get)
                self._a(best, obs[best], c); self.a = best
            else: self.a = c
        rss = float(Pt[self.a])
        if self.p_acq > 0 and 10*np.log10(max(rss,1e-12)/self.p_acq) < -self.d: self.fire = True
        return self.a, k

def runx(pol):
    pc, cc, s_, t_, log, nw, nr = [], [], 0, 0, [], 0, 0
    for P, keys in passes:
        pol.reset(int(P[0].argmax()), float(P[0].max())); caps, cost = [], []
        fc = next((t for t in range(1, len(P)) if PRED['vision'].get(keys[t]) is not None), None)
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: caps.append(float(P[t, a]) / m); cost.append(k)
        pc.append(np.array(caps)); cc.append(np.array(cost)); s_ += pol.sense; t_ += pol.trial
        log.append(dict(first_cov=fc, trials=list(pol.trials))); nw += pol.n_wrong; nr += pol.n_red
    caps = np.concatenate(pc); cost = np.concatenate(cc); n = len(caps)
    loss = -10*np.log10(np.clip(caps, 1e-6, None))
    ntr = sum(len(l['trials']) for l in log); won = sum(1 for l in log for tr in l['trials'] if tr[2] == 'won')
    at_first = sum(1 for l in log if l['trials'] and l['first_cov'] is not None and l['trials'][0][0] == l['first_cov'])
    opens = [tr[0] for l in log for tr in l['trials']]
    return dict(loss=float(loss.mean()), probes=float(cost.mean()), model=100*(s_+t_)/n, p95=float(np.percentile(loss, 95)),
                trials=ntr, won=won, wrong=nw, red=nr, at_first=at_first, passes_with=sum(1 for l in log if l['trials']),
                open_med=float(np.median(opens)) if opens else float('nan'), pc=pc)

fbL = None
def _trk(slack): return Ctl(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=slack)
_fb = runx.__wrapped__ if hasattr(runx, '__wrapped__') else None
# feedback-only reference (same run() semantics as before)
pc = []
for P, keys in passes:
    pol = _trk(TSL); pol.reset(int(P[0].argmax()), float(P[0].max())); caps = []
    for t in range(1, len(P)):
        a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
        if m > 0: caps.append(float(P[t, a]) / m)
    pc.append(np.array(caps))
fbL = float(np.mean(-10*np.log10(np.clip(np.concatenate(pc), 1e-6, None))))

print('\n################ %s  follow-up variants (fb-only %.3f dB)' % (SCN, fbL))
print('%-34s | %7s %6s %7s %6s | %6s %4s %5s %4s | %9s %8s' % ('variant', 'loss', 'p/ep', 'model%', 'p95', 'trials', 'won', 'wrong', 'red', 'first-cov', 'open-med'))
RES = {'scenario': SCN, 'fb': fbL, 'rows': {}}
variants = [('deployed (s0=1.5, bl 64/64)', dict(s0=1.5, bl_red=64)),
            ('s0=0.5 (gate needs measured innov.)', dict(s0=0.5, bl_red=64)),
            ('s0=0.3', dict(s0=0.3, bl_red=64)),
            ('wrong 64 / redundant 0', dict(s0=1.5, bl_red=0)),
            ('wrong 64 / redundant 8', dict(s0=1.5, bl_red=8)),
            ('wrong 64 / redundant 16', dict(s0=1.5, bl_red=16)),
            ('s0=0.5 + wrong 64 / redundant 8', dict(s0=0.5, bl_red=8)),
            ('s0=0.5 + wrong 64 / redundant 16', dict(s0=0.5, bl_red=16))]
for nm, kw in variants:
    r = runx(CtlX(**kw, **K))
    print('%-34s | %7.3f %6.2f %7.1f %6.2f | %6d %4d %5d %4d | %4d/%-4d %8.0f   VoI %+.3f'
          % (nm, r['loss'], r['probes'], r['model'], r['p95'], r['trials'], r['won'], r['wrong'], r['red'], r['at_first'], r['passes_with'], r['open_med'], fbL - r['loss']))
    RES['rows'][nm] = {k: r[k] for k in r if k != 'pc'}
# paired pass-level bootstrap of the best redundant-variant against the deployed controller
mL = lambda c: float(np.mean(-10*np.log10(np.clip(c, 1e-6, None))))
def sn(cs): return np.array([(-10*np.log10(np.clip(c,1e-6,None))).sum() for c in cs]), np.array([len(c) for c in cs], float)
base = runx(CtlX(s0=1.5, bl_red=64, **K))
for nm in ('wrong 64 / redundant 8', 's0=0.5 (gate needs measured innov.)', 's0=0.5 + wrong 64 / redundant 8'):
    kw = dict(variants)[nm]; v = runx(CtlX(**kw, **K))
    sa, na = sn(base['pc']); sb, nb = sn(v['pc']); n = len(passes)
    idx = np.random.default_rng(42).integers(0, n, (100000, n))
    d = sa[idx].sum(1)/na[idx].sum(1) - sb[idx].sum(1)/nb[idx].sum(1)
    print('   deployed minus [%s]: %+.3f dB  95%% [%+.3f, %+.3f]' % (nm, sa.sum()/na.sum()-sb.sum()/nb.sum(), np.percentile(d, 2.5), np.percentile(d, 97.5)))
_json.dump(RES, open(os.path.join(OUT, SCN + '_variants.json'), 'w'), default=float)
print('REV2_DONE %s' % SCN)
