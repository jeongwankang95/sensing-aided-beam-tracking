#!/usr/bin/env python3
"""delays_sensor_off_p5.py LOCKED ROOT SLACK --cache=CACHE: analysis/delays_sensor_off.py printing five decimals (the values
of Fig. 6). Writes results/analysis/lag_p5/<scenario>_lag.json.
Paper: the delay rows of Fig. 6 (analysis/table4.py and figures/fig6_vsense.py read the values).
usage: python analysis/delays_sensor_off_p5.py configs/locked_v3.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz"""
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

# ---- B-1 (M8): commit lag and joint delay. 'lag' = every commitment takes effect one epoch late;
# 'probe' = only probe-derived commitments are usable from the next epoch (the epoch that probes
# serves its pre-probe beam); sd/ld = sensor / link-observation delay in epochs (paper's studies). ----
import json as _json
SCN = os.path.basename(root); OUT = 'results/analysis/lag_p5'; os.makedirs(OUT, exist_ok=True)
def _trk(slack): return Ctl(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=slack)
class Persist:
    def reset(self, a0, p0): self.a = int(a0); self.sense = self.trial = 0
    def step(self, Pt, key): return self.a, 0
class SAFB:
    def __init__(self, period=None, slack=0.4): self.period, self.slack = period, slack
    def reset(self, a0, p0):
        self.trk = _trk(self.slack); self.trk.reset(a0, p0); self.a = int(a0); self.sv = 0; self.sense = self.trial = 0
    def step(self, Pt, key):
        pr = PRED['vision'].get(key)
        if pr is None:
            self.a, k = self.trk.step(Pt, key); return self.a, k
        self.sense += 1; p = int(pr); k = 0; self.sv += 1
        if self.period is not None and self.sv >= self.period:
            S = sorted({max(0,p-1), p, min(C-1,p+1)}); obs = {b: float(Pt[b]) for b in S}; k = len(S); self.a = max(obs, key=obs.get); self.sv = 0
        else: self.a = p
        self.trk.n += 1; c = int(np.clip(round(self.trk.acq + self.trk.v*self.trk.n), 0, C-1)); self.trk._a(self.a, float(Pt[self.a]), c)
        return self.a, k
def pre_beam(pol, key):
    """beam served this epoch if probe results were not yet usable (state read before step)"""
    if isinstance(pol, Persist): return pol.a
    if isinstance(pol, SAFB):
        pr = PRED['vision'].get(key)
        if pr is not None: return int(pr)
        t = pol.trk; return int(np.clip(round(t.acq + t.v*(t.n+1)), 0, C-1))
    if pol.mode == 'sense' and pol.sensor:
        pr = PRED[pol.sensor].get(key)
        if pr is not None: return int(pr)
    return int(np.clip(round(pol.acq + pol.v*(pol.n+1)), 0, C-1))
def perpass(mk, mode='none', sd=0, ld=0):
    cs, ks, s_, t_ = [], [], 0, 0
    for P, keys in passes:
        pol = mk(); pol.reset(int(P[0].argmax()), float(P[0].max())); prev = int(P[0].argmax()); c, k = [], []
        for t in range(1, len(P)):
            key = keys[t-sd] if (sd and t-sd >= 1) else (None if sd else keys[t])
            if key is None: key = '__none__'
            Pobs = P[max(t-ld, 0)]
            pre = pre_beam(pol, key); a, kk = pol.step(Pobs, key)
            served = prev if mode == 'lag' else (pre if (mode == 'probe' and kk > 0) else a)
            m = float(P[t].max())
            if m > 0: c.append(float(P[t, served]) / m); k.append(kk)
            prev = a
        cs.append(np.array(c)); ks.append(np.array(k)); s_ += getattr(pol, 'sense', 0); t_ += getattr(pol, 'trial', 0)
    caps = np.concatenate(cs); loss = -10*np.log10(np.clip(caps, 1e-6, None))
    return dict(loss=float(loss.mean()), probes=float(np.concatenate(ks).mean()), model=100*(s_+t_)/len(caps), p95=float(np.percentile(loss, 95)), cs=cs)
sn = lambda cs: (np.array([(-10*np.log10(np.clip(c,1e-6,None))).sum() for c in cs]), np.array([len(c) for c in cs], float))
def paired(a, b):
    sa, na = sn(a['cs']); sb, nb = sn(b['cs']); n = len(passes); idx = np.random.default_rng(42).integers(0, n, (100000, n))
    d = sa[idx].sum(1)/na[idx].sum(1) - sb[idx].sum(1)/nb[idx].sum(1)
    return sa.sum()/na.sum()-sb.sum()/nb.sum(), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
POL = [('persistence', lambda: Persist()), ('feedback-only', lambda: _trk(TSL)), ('controller', lambda: Ctl(**K)), ('skeleton', lambda: Ctl(**dict(K, sensor=None))),
       ('always-on', lambda: SAFB(None, TSL)), ('always-on+verify1', lambda: SAFB(1, TSL)), ('always-on+verify2', lambda: SAFB(2, TSL)), ('always-on+verify8', lambda: SAFB(8, TSL))]
CONDS = [('none (SELF-CHECK Table II)', dict()), ('sensor delay 1 (SELF-CHECK 0.820 on S9)', dict(sd=1)), ('sensor delay 2', dict(sd=2)), ('link delay 1 (SELF-CHECK 1.086/1.904 on S9)', dict(ld=1)),
         ('JOINT sensor 1 + link 1', dict(sd=1, ld=1)), ('COMMIT LAG 1 (all commitments)', dict(mode='lag')), ('PROBE LAG (probe results usable next epoch)', dict(mode='probe')),
         ('PROBE LAG + sensor delay 1', dict(mode='probe', sd=1))]
print('\n################ %s  commit-lag / delay conditions (fb slack %.1f)' % (SCN, TSL)); RES = {'scenario': SCN, 'cond': {}}
for cn, kw in CONDS:
    R = {nm: perpass(mk, **kw) for nm, mk in POL}
    v = paired(R['feedback-only'], R['controller']); w = paired(R['always-on+verify1'], R['controller']); x = paired(R['always-on+verify2'], R['controller'])
    print('-- %s' % cn); vs = paired(R['skeleton'], R['controller']); print('   skeleton (no sensor) minus ctl %+.5f [%+.5f, %+.5f]' % vs)
    print('   ' + ' | '.join('%s %.3f@%.2f' % (nm, R[nm]['loss'], R[nm]['probes']) for nm, _ in POL))
    print('   VoI fb-ctl %+.5f [%+.5f, %+.5f]   ao+v1 minus ctl %+.5f [%+.5f, %+.5f]   ao+v2 minus ctl %+.5f [%+.5f, %+.5f]   ctl model %.1f%%' % (v + w + x + (R['controller']['model'],)))
    RES['cond'][cn] = {nm: (R[nm]['loss'], R[nm]['probes'], R[nm]['model']) for nm, _ in POL}; RES['cond'][cn]['voi'] = v; RES['cond'][cn]['aov1_minus_ctl'] = w; RES['cond'][cn]['aov2_minus_ctl'] = x
_json.dump(RES, open(os.path.join(OUT, SCN + '_lag.json'), 'w'), default=float); print('REV6_DONE %s' % SCN)
