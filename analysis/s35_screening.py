#!/usr/bin/env python3
"""s35_screening.py: the pre-registered screening of DeepSense 6G Scenario 35 as a candidate high-innovation site (CPU only,
feedback-only tracker, no training), against Scenario 9: (1) tracker innovation sigma against th_on = 0.6, (2) causal
persistence and hold-previous loss, (3) runner-up +-1 adjacency and an SNR proxy (peak over median power), plus a GPS
angular-rate check. Constants are the 2026-08-10 shared set with the tracker slack varied. Writes results/analysis/s35_screen.json.
Paper: supplementary Sec. S17-M and the limitations (peak 0.7 dB above the median, feedback-only 0.16-0.22 dB, a bound of about
0.2 dB on what sensing could add). Needs dataset/scenario35 (DeepSense 6G Scenario 35: scenario35.csv, unit1/, unit2/) and the
Scenario-9 GPS files (unit1/GPS_data, unit2_loc).
usage: python analysis/s35_screening.py"""
import numpy as np, pandas as pd, os, sys, json, math, collections
C = 64
ROOT = "."
K = dict(th_on=0.6, th_off=0.5, verify=8, t_trial=1, margin=0.5, slack=1.0, bl=64)
PRED = {'vision': {}, 'radar': {}}

# ---- Ctl copied verbatim from needonly.py prelude (server, 2026-09) ----
class Ctl:
    def __init__(self, sensor='vision', drop_db=2.0, refresh=16, wcap=10, **kw):
        self.sensor, self.d, self.refresh, self.wcap = sensor, drop_db, refresh, wcap
        for k in ('th_on','th_off','verify','t_trial','margin','slack'): setattr(self,k,kw[k])
        self.blf = kw['bl']
    def reset(self,a0,p0):
        self.acq,self.p_acq=int(a0),float(p0); self.v,self.sig,self.n,self.fire,self.esc=0.,1.5,0,False,1
        self.a,self.mode,self.sv,self.tr,self.bl=int(a0),'track',0,0,0; self.sense=self.trial=0
        self.innov=[]  # raw innovation |b-c|/n at each acquisition (screening addition)
    def _a(self,b,p,c):
        self.innov.append(abs(b-c)/max(self.n,1))
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

def fb_replay(passes, slack):
    pol = Ctl(sensor=None, **{**K, 'slack': slack})
    caps, cost, sig_ep, innov, nacq, fires = [], [], [], [], 0, 0
    for P in passes:
        pol.reset(int(P[0].argmax()), float(P[0].max()))
        for t in range(1, len(P)):
            sig_ep.append(pol.sig)           # sigma the gate would test at this epoch
            a, k = pol.step(P[t], None); m = float(P[t].max())
            if m > 0: caps.append(float(P[t, a]) / m); cost.append(k)
        innov += pol.innov; nacq += len(pol.innov)
    caps = np.array(caps); loss = -10*np.log10(np.clip(caps, 1e-6, None)); sig_ep = np.array(sig_ep); innov = np.array(innov)
    return dict(loss=loss.mean(), p95=np.percentile(loss, 95), probes=np.mean(cost), out=100*(caps < .5).mean(),
                sig_med=np.median(sig_ep), sig_gt_on=100*(sig_ep > K['th_on']).mean(), sig_lt_off=100*(sig_ep < K['th_off']).mean(),
                innov_med=np.median(innov), innov_mean=innov.mean(), acq_per_100=100*nacq/len(caps))

def load_site(name):
    if name == 'S9':
        root = f"{ROOT}/dataset/scenario9"; df = pd.read_csv(f"{root}/scenario9.csv"); pcol, ocol, gcol = 'unit1_pwr_60ghz', 'index', 'unit2_loc'
        tp = set(pd.read_csv(f"{root}/test_seqs_RA.csv")['seq_index'].unique())
        ts = df['time_stamp[UTC]'].str.extract(r"(\d+)-(\d+)-(\d+)-(\d+)").astype(float); df['t'] = ts[0]*3600+ts[1]*60+ts[2]+ts[3]/1e3
    else:
        root = f"{ROOT}/dataset/scenario35"; df = pd.read_csv(f"{root}/scenario35.csv"); pcol, ocol, gcol = 'unit1_pwr', 'abs_idx', 'unit2_gps'; tp = None
        ts = df['timestamp'].str.extract(r"(\d+):(\d+):(\d+)-(\d+)").astype(float); df['t'] = ts[0]*3600+ts[1]*60+ts[2]+ts[3]/1e6
    bs = np.loadtxt(f"{root}/unit1/{'GPS_data/gps_location.txt' if name=='S9' else 'gps/unit1_static_loc.txt'}")[:2]
    passes = collections.OrderedDict(); seen_pwr = set()
    for p, g in df.groupby('seq_index', sort=True):
        g = g.sort_values(ocol)
        P = np.stack([np.nan_to_num(np.loadtxt(os.path.join(root, str(r).lstrip('./'))), nan=0.0) for r in g[pcol]])
        G = np.stack([np.loadtxt(os.path.join(root, str(r).lstrip('./')))[:2] for r in g[gcol]])
        files = list(g[pcol]); dup = sum(f in seen_pwr for f in files); seen_pwr.update(files)
        passes[p] = dict(P=P, G=G, t=g['t'].values, dup_frac=dup/len(files), test=(tp is None or p in tp))
    return passes, bs

def bearing(bs, G):
    lat0, lon0 = np.radians(bs); lat, lon = np.radians(G[:, 0]), np.radians(G[:, 1])
    y = np.sin(lon-lon0)*np.cos(lat); x = np.cos(lat0)*np.sin(lat)-np.sin(lat0)*np.cos(lat)*np.cos(lon-lon0)
    dist = 6371000*np.arccos(np.clip(np.sin(lat0)*np.sin(lat)+np.cos(lat0)*np.cos(lat)*np.cos(lon-lon0), -1, 1))
    return np.degrees(np.arctan2(y, x)), dist

def stats(passes, bs, sel):
    ps = [v for v in passes.values() if sel(v)]
    out = collections.OrderedDict(); out['passes'] = len(ps); out['frames'] = sum(len(v['P']) for v in ps)
    out['frames/pass med'] = np.median([len(v['P']) for v in ps])
    pk_med, pk_min, adj2, pers, adj1, dB_hold, dB_hold3, tort, mono, ang, dist, jump = [], [], [], [], [], [], [], [], [], [], [], []
    for v in ps:
        P = v['P']; b = P.argmax(1); s = np.sort(P, 1)[:, ::-1]; second = np.argsort(P, 1)[:, -2]
        pk_med += list(10*np.log10(s[:, 0]/np.median(P, 1))); pk_min += list(10*np.log10(s[:, 0]/np.maximum(P.min(1), 1e-12)))
        adj2 += list(np.abs(second-b) <= 1)
        d = np.abs(np.diff(b)); pers += list(d == 0); adj1 += list(d <= 1); jump += list(d)
        dB_hold += list(-10*np.log10(np.maximum(P[np.arange(1, len(P)), b[:-1]], 1e-12)/P[1:].max(1)))
        loc = [max(P[t, max(0, b[t-1]-1):b[t-1]+2]) for t in range(1, len(P))]
        dB_hold3 += list(-10*np.log10(np.maximum(loc, 1e-12)/P[1:].max(1)))
        tort.append(d.sum()/max(abs(int(b[-1])-int(b[0])), 1)); mono.append(abs(np.corrcoef(np.arange(len(b)), b)[0, 1]) if len(b) > 2 else np.nan)
        th, r = bearing(bs, v['G']); t = v['t']
        for i in range(5, len(th)-5):
            sl = np.polyfit(t[i-5:i+6]-t[i], np.unwrap(np.radians(th[i-5:i+6])), 1)[0]; ang.append(abs(np.degrees(sl)))
        dist += list(r)
    out['SNR proxy peak/median dB (med)'] = np.median(pk_med); out['peak/median dB p10'] = np.percentile(pk_med, 10)
    out['peak/min dB (med)'] = np.median(pk_min)
    out['runner-up within +-1 (%)'] = 100*np.mean(adj2)
    out['persistence b*_t==b*_t-1 (%)'] = 100*np.mean(pers); out['|db*|<=1 (%)'] = 100*np.mean(adj1); out['|db*| mean (beams)'] = np.mean(jump); out['|db*| p95'] = np.percentile(jump, 95)
    out['hold-previous loss mean dB'] = np.mean(dB_hold); out['hold-previous loss p95 dB'] = np.percentile(dB_hold, 95)
    out['hold-prev best-of-+-1 loss mean dB'] = np.mean(dB_hold3)
    out['trajectory tortuosity (med)'] = np.median(tort); out['|corr(t,b*)| (med)'] = np.nanmedian(mono)
    out['GPS angular rate deg/s (med)'] = np.median(ang); out['GPS angular rate deg/s p90'] = np.percentile(ang, 90); out['BS-vehicle dist m (med)'] = np.median(dist)
    for sl in (0.4, 0.6, 0.8, 1.0):
        r = fb_replay([v['P'] for v in ps], sl)
        out[f'FB slack {sl}: loss dB'] = r['loss']; out[f'FB slack {sl}: probes/epoch'] = r['probes']; out[f'FB slack {sl}: sigma med'] = r['sig_med']
        out[f'FB slack {sl}: sigma>th_on %'] = r['sig_gt_on']; out[f'FB slack {sl}: sigma<th_off %'] = r['sig_lt_off']; out[f'FB slack {sl}: raw innov med'] = r['innov_med']; out[f'FB slack {sl}: acq/100 ep'] = r['acq_per_100']
    return out

S9, bs9 = load_site('S9'); S35, bs35 = load_site('S35')
dups = [(p, round(v['dup_frac'], 2), len(v['P'])) for p, v in S35.items() if v['dup_frac'] > 0]
print("S35 passes reusing earlier passes' pwr files (pass, frac, n):", dups)
print('  %d of %d passes, %d frames, reuse power files already assigned to an earlier pass; %d passes and %d frames remain'
      % (len(dups), len(S35), round(sum(v['dup_frac'] * len(v['P']) for v in S35.values())),
         sum(v['dup_frac'] == 0 for v in S35.values()), sum(len(v['P']) for v in S35.values() if v['dup_frac'] == 0)))
cols = collections.OrderedDict()
cols['S9 test (28 passes)'] = stats(S9, bs9, lambda v: v['test'])
cols['S9 all'] = stats(S9, bs9, lambda v: True)
cols['S35 all'] = stats(S35, bs35, lambda v: True)
cols['S35 no-dup'] = stats(S35, bs35, lambda v: v['dup_frac'] == 0)
keys = list(next(iter(cols.values())).keys())
print("\n%-40s" % "metric" + "".join("%18s" % c for c in cols))
for k in keys:
    print("%-40s" % k + "".join("%18s" % (("%.3f" % cols[c][k]) if isinstance(cols[c][k], float) else str(cols[c][k])) for c in cols))
json.dump({c: {k: float(v) for k, v in d.items()} for c, d in cols.items()}, open(f"{ROOT}/results/analysis/s35_screen.json", "w"), indent=1)
