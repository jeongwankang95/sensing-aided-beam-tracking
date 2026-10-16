#!/usr/bin/env python3
"""kinematics.py LOCKED ROOT SLACK --cache=CACHE: geometry and beam motion of the validation and test passes of one scenario
(supplementary Table S4): per split the median pass duration, the best beam's net change and net traverse rate (indices/s), mean
step, straightness, the GPS bearing rate seen from the base station (deg/s, slope over 11-frame windows), the median and minimum
range (m) and the plain tracker's median innovation at slack SLACK. Reads the vehicle GPS (unit2_loc_cal when the scenario has
calibrated positions, else unit2_loc) and the base-station position (unit1_loc). The loading code and the controller class come
verbatim from the lock-loss-recovery analysis (analysis/lock_loss_selection.py), which this check extended; the cache and the
constants are read but do not enter the printed values.
usage: python analysis/kinematics.py configs/locked_multi.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz
       (slack 0.6 for Scenario 31, 0.8 for Scenarios 32 and 33)"""
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

# ---- geometry and innovation per scenario and split (validation / test), 2026-09-30 ----
def bearing(bs, G):
    lat0, lon0 = np.radians(bs); lat, lon = np.radians(G[:, 0]), np.radians(G[:, 1])
    y = np.sin(lon-lon0)*np.cos(lat); x = np.cos(lat0)*np.sin(lat) - np.sin(lat0)*np.cos(lat)*np.cos(lon-lon0)
    dist = 6371000*np.arccos(np.clip(np.sin(lat0)*np.sin(lat) + np.cos(lat0)*np.cos(lat)*np.cos(lon-lon0), -1, 1))
    return np.degrees(np.arctan2(y, x)), dist
class Rec(Ctl):
    def reset(self, a0, p0): super().reset(a0, p0); self.inn = []
    def _a(self, b, p, c):
        self.inn.append(abs(b - c) / max(self.n, 1)); return super()._a(b, p, c)
df = base; cols = list(df.columns)
gcal = [c for c in cols if c.startswith('unit2') and 'cal' in c]; graw = [c for c in cols if c.startswith('unit2_loc') and 'cal' not in c]
gcol = gcal[0] if gcal else graw[0]
tcol = [c for c in cols if 'time' in c.lower()][0]
bs = np.loadtxt(os.path.join(root, str(df[[c for c in cols if c.startswith('unit1_loc')][0]].iloc[0]).lstrip('./')))[:2]
ts = df[tcol].astype(str).str.extract(r"(\d+)[-:](\d+)[-:](\d+)[-.](\d+)").astype(float)
df['t'] = ts[0]*3600 + ts[1]*60 + ts[2] + ts[3]/10**ts[3].astype(int).astype(str).str.len()
for split in ('val', 'test'):
    f = f'{root}/{split}_seqs_RA.csv'
    if not os.path.exists(f): continue
    sel = set(pd.read_csv(f)['seq_index'].unique()); R = []; INN = []
    for p, g in df.groupby('seq_index'):
        if p not in sel: continue
        g = g.sort_values('index')
        P = np.stack([np.nan_to_num(np.loadtxt(os.path.join(root, str(r).lstrip('./'))), nan=0.0) for r in g['unit1_pwr_60ghz']])
        G = np.stack([np.loadtxt(os.path.join(root, str(r).lstrip('./')))[:2] for r in g[gcol]])
        b = P.argmax(1); t = g['t'].values; th, rng = bearing(bs, G); thu = np.degrees(np.unwrap(np.radians(th)))
        rate = [abs(np.polyfit(t[i-5:i+6]-t[i], thu[i-5:i+6], 1)[0]) for i in range(5, len(t)-5)]
        span = t[-1] - t[0]
        R.append(dict(dur=span, net=abs(int(b[-1]) - int(b[0])), trav=abs(int(b[-1]) - int(b[0]))/max(span, 1e-9),
                      step=np.abs(np.diff(b)).mean(), straight=abs(int(b[-1]) - int(b[0]))/max(np.abs(np.diff(b)).sum(), 1),
                      rate=np.median(rate) if rate else np.nan, rmed=np.median(rng), rmin=rng.min()))
        pol = Rec(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=TSL)
        pol.reset(int(P[0].argmax()), float(P[0].max()))
        for tt in range(1, len(P)): pol.step(P[tt], None)
        INN += pol.inn
    R = pd.DataFrame(R); m = R.median()
    print('%-11s %-4s gps=%-13s passes=%2d dur=%.4f net=%.1f trav=%.4f step=%.3f straight=%.3f rate=%.4f rmed=%.3f rmin=%.3f innov=%.3f'
          % (os.path.basename(root), split, gcol, len(R), m.dur, m.net, m.trav, m.step, m.straight, m.rate, m.rmed, m.rmin, np.median(INN)))
print('KIN2_DONE')
