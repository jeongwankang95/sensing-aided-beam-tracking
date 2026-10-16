#!/usr/bin/env python3
"""decimation_slack.py LOCKED ROOT SLACK --cache=CACHE: (1) on Scenario 9, the per-pass gains over the plain tracker on the 17
validation passes and the histories of the high-gain test passes; (2) the feedback-only comparator's slack re-selected on the
decimated validation passes at each d = 1, 2, 4 (the reference grid plus 1.2; Scenario 33 uses Scenario 32's) and scored on
test. Writes results/analysis/scenario9_valdist.json and results/analysis/<scenario>_perd_slack.json (read by
analysis/every_frame.py and every_frame_reselect.py).
Paper: supplementary Secs. S17-H (validation losses 0.483 and 0.258 dB, the 17 per-pass gains from -0.19 to +3.44 dB) and
S17-I (the re-selected comparator at d = 2 and 4).
usage: python analysis/decimation_slack.py configs/locked_v3.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz"""
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

# ---- B-3: S9 validation per-pass value distribution and the high-gain test passes' history;
# B-4: feedback-only comparator slack re-selected on validation at each decimation d. ----
import json as _json
SCN = os.path.basename(root); OUT = 'results/analysis'; os.makedirs(OUT, exist_ok=True)
def _trk(slack): return Ctl(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=slack)
def load_split(r, split):
    b = pd.read_csv(f'{r}/{os.path.basename(r)}.csv') if os.path.exists(f'{r}/{os.path.basename(r)}.csv') else pd.read_csv(glob.glob(f'{r}/*_dev.csv')[0])
    sel = set(pd.read_csv(f'{r}/{split}_seqs_RA.csv')['seq_index'].unique()); ps = []
    for p, g in b.groupby('seq_index'):
        if p not in sel: continue
        g = g.sort_values('index')
        P = np.stack([np.nan_to_num(np.loadtxt(os.path.join(r, str(x).lstrip('./'))), nan=0.0) for x in g['unit1_pwr_60ghz']])
        ps.append((P, [str(x) for x in g['unit1_pwr_60ghz']], int(p)))
    return ps
mL = lambda c: float(np.mean(-10*np.log10(np.clip(c, 1e-6, None))))
def caps_of(ps, pol, pred=None):
    global PRED
    if pred is not None: PRED, saved = pred, PRED
    out, cost, esc, trials = [], [], [], []
    for P, keys, pid in ps:
        pol.reset(int(P[0].argmax()), float(P[0].max())); c, k, e0 = [], [], 0; tr0 = 0; won = 0
        for t in range(1, len(P)):
            esc_before = pol.esc; tr_before = pol.trial; mode_before = pol.mode
            a, kk = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: c.append(float(P[t, a]) / m); k.append(kk)
            if pol.esc > esc_before: e0 += 1
            if pol.trial > tr_before: tr0 += 1; won += (pol.mode == 'sense')
        out.append(np.array(c)); cost.append(np.mean(k)); esc.append(e0); trials.append((tr0, won))
    if pred is not None: PRED = saved
    return out, cost, esc, trials
tp = [(P, k, None) for P, k in passes]  # test passes from the prelude (no ids); reload with ids for B-3
if SCN == 'scenario9':
    VAL = load_split(root, 'val'); vc = np.load(f'results/predictions/both_{os.path.basename(root)}_val.npz', allow_pickle=True)
    VPRED = {k: dict(zip(vc['anchor'], vc[k])) for k in ('radar', 'vision')}
    fbv, _, _, _ = caps_of(VAL, _trk(0.4)); ctv, _, _, trv = caps_of(VAL, Ctl(**K), VPRED)
    per = np.array([mL(a) - mL(b) for a, b in zip(fbv, ctv)]); fw = mL(np.concatenate(fbv)) - mL(np.concatenate(ctv))
    print('\n[B-3] S9 VALIDATION (%d passes): fb slack 0.4 loss %.3f (SELF-CHECK locked feedback_reference 0.483), controller %.3f, frame-weighted VoI %+.3f (locked regret -0.062)'
          % (len(VAL), mL(np.concatenate(fbv)), mL(np.concatenate(ctv)), fw))
    print('      per-pass VoI sorted: ' + ' '.join('%+.2f' % v for v in np.sort(per)))
    print('      mean %+.3f median %+.3f positive %d/%d, >=2 dB %d; trials won %d/%d' % (per.mean(), np.median(per), (per > 0).sum(), len(per), (per >= 2).sum(), sum(w for _, w in trv), sum(t for t, _ in trv)))
    TEST = load_split(root, 'test'); fbt, fbc, fbe, _ = caps_of(TEST, _trk(TSL)); ctt, ctc, cte, trt = caps_of(TEST, Ctl(**K))
    rows = []
    for i, (P, keys, pid) in enumerate(TEST):
        fc = next((t for t in range(1, len(P)) if PRED['vision'].get(keys[t]) is not None), None)
        rows.append(dict(pid=pid, n=len(P)-1, fb=mL(fbt[i]), ctl=mL(ctt[i]), voi=mL(fbt[i])-mL(ctt[i]), drift=float(np.abs(np.diff(P.argmax(1))).mean()),
                         jump95=float(np.percentile(np.abs(np.diff(P.argmax(1))), 95)), fb_probes=fbc[i], fb_esc=fbe[i], trials=trt[i][0], won=trt[i][1], first_cov=fc,
                         fb_first10=mL(fbt[i][:10]), ctl_first10=mL(ctt[i][:10])))
    rows.sort(key=lambda r: -r['voi'])
    print('\n[B-3] S9 TEST passes ranked by VoI (fb_esc = failed re-acquisitions of the feedback tracker; first10 = mean loss over the first 10 scored frames)')
    print('%6s %4s %7s %7s %7s %6s %7s %7s %6s %6s %5s %9s %9s' % ('pass', 'n', 'fb', 'ctl', 'VoI', 'drift', 'jump95', 'fb p/e', 'fb_esc', 'trials', 'won', 'fb_1st10', 'ctl_1st10'))
    for r in rows: print('%6d %4d %7.3f %7.3f %+7.3f %6.2f %7.2f %7.2f %6d %6d %5d %9.3f %9.3f' % (r['pid'], r['n'], r['fb'], r['ctl'], r['voi'], r['drift'], r['jump95'], r['fb_probes'], r['fb_esc'], r['trials'], r['won'], r['fb_first10'], r['ctl_first10']))
    top = rows[:7]; rest = rows[7:]
    print('      top-7 mean: fb %.3f ctl %.3f drift %.2f fb_esc %.1f won %d/%d | rest mean: fb %.3f ctl %.3f drift %.2f fb_esc %.1f won %d/%d'
          % (np.mean([r['fb'] for r in top]), np.mean([r['ctl'] for r in top]), np.mean([r['drift'] for r in top]), np.mean([r['fb_esc'] for r in top]), sum(r['won'] for r in top), sum(r['trials'] for r in top),
             np.mean([r['fb'] for r in rest]), np.mean([r['ctl'] for r in rest]), np.mean([r['drift'] for r in rest]), np.mean([r['fb_esc'] for r in rest]), sum(r['won'] for r in rest), sum(r['trials'] for r in rest)))
    _json.dump(dict(val_per=per.tolist(), val_fw=fw, test_rows=rows), open(os.path.join(OUT, 'scenario9_valdist.json'), 'w'), default=float)
# ---- B-4 ----
VROOT = root if os.path.exists(f'{root}/val_seqs_RA.csv') else os.path.join(os.path.dirname(root), 'scenario32')
VAL2 = load_split(VROOT, 'val')
def dec(ps, d): return [(P[::d], k[::d], pid) for P, k, pid in ps if len(P[::d]) >= 6]
def fbrun(ps, slack):
    cs, cost, _, _ = caps_of(ps, _trk(slack)); return mL(np.concatenate(cs)), float(np.mean(cost))
def ctlrun(ps):
    cs, cost, _, _ = caps_of(ps, Ctl(**K)); return mL(np.concatenate(cs)), float(np.mean(cost))
print('\n[B-4] %s comparator slack re-selected on %s validation at each d (deployed d=1 slack %.1f)' % (SCN, os.path.basename(VROOT), TSL))
print('%2s | %-8s %-13s | %-13s | %-13s | %7s %7s' % ('d', 'val-best', 'fb(reselected)', 'fb(deployed)', 'controller', 'VoI-res', 'VoI-dep')); RES = {}
TEST2 = [(P, k, None) for P, k in passes]
for d in (1, 2, 4):
    cand = sorted(((fbrun(dec(VAL2, d), s) + (s,)) for s in (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0)), key=lambda x: (round(x[0], 4), round(x[1], 3)))
    s_best = cand[0][2]; a = fbrun(dec(TEST2, d), s_best); b = fbrun(dec(TEST2, d), TSL); c = ctlrun(dec(TEST2, d))
    print('%2d | %-8.1f %6.3f @ %5.2f | %6.3f @ %5.2f | %6.3f @ %5.2f | %+7.3f %+7.3f' % (d, s_best, a[0], a[1], b[0], b[1], c[0], c[1], a[0]-c[0], b[0]-c[0]))
    RES[d] = dict(slack=s_best, fb_res=a, fb_dep=b, ctl=c)
_json.dump(RES, open(os.path.join(OUT, SCN + '_perd_slack.json'), 'w'), default=float); print('REV7_DONE %s' % SCN)
