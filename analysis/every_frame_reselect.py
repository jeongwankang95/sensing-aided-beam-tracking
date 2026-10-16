#!/usr/bin/env python3
"""every_frame_reselect.py LOCKED ROOT SLACK --cache=CACHE: the comparators of the decimation study re-selected on the
every-frame validation losses (the plain-tracker slack over 0.4-3.0 plus 1.2, with Scenario 33 using Scenario 32's validation
passes; on Scenario 9 the 98 re-sweep settings, with and without 4 validation probes per decision), then scored on every test
frame as in analysis/every_frame.py. The controller and its constants are unchanged and nothing is selected on test. Writes
results/analysis/holdsel_<scenario>.json.
Paper: Sec. VI-C (re-sweeping reference under decimation) and supplementary Secs. S1-G and S17-I (the slack changes only on
Scenario 9 at d = 2, to 0.6; the re-selected re-sweep reference loses 1.862 and 4.935 dB at d = 2 and 4).
usage: python analysis/every_frame_reselect.py configs/locked_v3.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz"""
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
# ---- cold review 6, D1 (2026-09-30): reference = tracker + one scheduled re-sweep, selected on VALIDATION over
# slack x {full, coarse} x k (98 settings; min loss, and min loss with validation probes <= 4); test once.
# Compared with the deployed controller and with the controller given the same re-sweep. Paired pass bootstrap
# 10^5 (same routine and seed as llr_1e5p.py). usage as llr.py.
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

VAL0, TEST0 = load('val'), load('test')
def dec(ps, d): return [(P[::d], k[::d], pid) for P, k, pid in ps if len(P[::d]) >= 6]
SCN = os.path.basename(root); OUT = 'results/analysis'; os.makedirs(OUT, exist_ok=True)
fb = lambda sl, **kw: RS(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=sl, **kw)
ctl = lambda **kw: RS(**K, **kw)

# ======================= decimation comparators re-selected on every-frame validation scores (2026-10-14, r81) ========
# holdsel3.py = resweep_decim3.py lines 1-153 + this tail. The comparators of the decimation study were selected on the
# decision-frame losses of decimated validation passes (rev73.py: plain-tracker slack; resweep_decim3.py: Scenario-9
# re-sweep reference). Now that the decimation study scores every native frame (hold3.py), they are re-selected on the
# every-frame validation losses with the same grids and rules, then scored on test the same way. The controller and
# its constants are unchanged and nothing is selected on test.
#   plain tracker: slack in {0.4,0.6,0.8,1.0,1.2,1.5,2.0,3.0}, min validation loss (ties: fewer probes per decision);
#                  Scenario 33 uses Scenario 32's validation passes (as rev73.py)
#   Scenario-9 re-sweep reference: slack {0.4,...,3.0} x {full, coarse} x k {2,...,20} (98), min validation loss, and min
#                  loss among settings with at most 4 validation probes per decision (as resweep_decim3.py)
def perpass_hold(passes, pol, D):
    out = []
    for P, keys, pid in passes:
        if len(P[::D]) < 6: continue
        pol.reset(int(P[0].argmax()), float(P[0].max())); a = int(P[0].argmax()); L, Kc, Kd = [], [], []
        for t in range(1, len(P)):
            k = 0
            if t % D == 0: a, k = pol.step(P[t], keys[t])
            m = float(P[t].max())
            if m > 0:
                L.append(-10*np.log10(max(float(P[t, a])/m, 1e-6))); Kc.append(k)
                if t % D == 0: Kd.append(k)
        out.append((np.array(L), np.array(Kc), pid, getattr(pol, 'sense', 0) + getattr(pol, 'trial', 0), getattr(pol, 'sweeps', 0), np.array(Kd)))
    return out

def hsum(pp):   # every-frame loss, probes per decision
    return float(np.concatenate([x[0] for x in pp]).mean()), float(np.concatenate([x[5] for x in pp]).mean())

def load_root(r, split):
    b = pd.read_csv(f'{r}/{os.path.basename(r)}.csv') if os.path.exists(f'{r}/{os.path.basename(r)}.csv') else pd.read_csv(glob.glob(f'{r}/*_dev.csv')[0])
    f = f'{r}/{split}_seqs_RA.csv'
    if not os.path.exists(f): return []
    sel = set(pd.read_csv(f)['seq_index'].unique()); out = []
    for p, g in b.groupby('seq_index'):
        if p not in sel: continue
        g = g.sort_values('index')
        P = np.stack([np.nan_to_num(np.loadtxt(os.path.join(r, str(x).lstrip('./'))), nan=0.0) for x in g['unit1_pwr_60ghz']])
        out.append((P, [str(x) for x in g['unit1_pwr_60ghz']], p))
    return out

VROOT = root if os.path.exists(f'{root}/val_seqs_RA.csv') else os.path.join(os.path.dirname(root), 'scenario32')
VALH = load_root(VROOT, 'val')
PERD = json.load(open(os.path.join(OUT, '%s_perd_slack.json' % SCN)))
SEL = json.load(open(os.path.join(OUT, 'resweep_decim_scenario9.json'))) if SCN == 'scenario9' else None
SLACKS = (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0)
GRID = [dict(slack=s, how=h, k=k) for s in (0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0) for h in ('full', 'coarse') for k in (2, 3, 5, 7, 10, 15, 20)]
print('%s: %d validation passes from %s' % (SCN, len(VALH), VROOT))
RESS = {}
for D in (1, 2, 4):
    out = {}
    cand = sorted(((hsum(perpass_hold(VALH, fb(s), D)), s) for s in SLACKS), key=lambda x: (round(x[0][0], 4), round(x[0][1], 3)))
    s_held = cand[0][1]; s_dec = PERD[str(D)]['slack']
    CT = perpass_hold(TEST0, ctl(), D); ct = hsum(CT)
    out['controller'] = ct
    for lab, s in (('plain, native slack', TSL), ('plain, slack re-selected on decision frames', s_dec), ('plain, slack re-selected on every frame', s_held)):
        pp = perpass_hold(TEST0, fb(s), D); out[lab] = dict(slack=s, test=hsum(pp), gain=boot(pp, CT))
    if SEL:
        vres = [(g, hsum(perpass_hold(VALH, fb(g['slack'], how=g['how'], k=g['k']), D))) for g in GRID]
        f_held = min(vres, key=lambda r: r[1][0])[0]; capl = [r for r in vres if r[1][1] <= 4.0]
        c_held = min(capl, key=lambda r: r[1][0])[0] if capl else None
        f_dec, c_dec = SEL[str(D)]['sel_free'], SEL[str(D)]['sel_cap']
        z = None
        if D in (2, 4):
            z = np.load('results/predictions/dec%d_s9.npz' % D, allow_pickle=True)
            keep = dict(PRED['vision']); PRED['vision'] = dict(zip(z['anchor'], z['vision']))
            CP = perpass_hold(TEST0, ctl(), D); PRED['vision'] = keep
        for lab, g in (('re-sweep, selected on decision frames', f_dec), ('re-sweep, selected on every frame', f_held),
                       ('re-sweep <=4, selected on decision frames', c_dec), ('re-sweep <=4, selected on every frame', c_held)):
            if g is None: continue
            pp = perpass_hold(TEST0, fb(g['slack'], how=g['how'], k=g['k']), D)
            out[lab] = dict(setting=g, test=hsum(pp), gain=boot(pp, CT))
            if z is not None: out[lab]['gain_dspaced'] = boot(pp, CP)
    RESS[D] = out
    print('\n######## %s d=%d (every-frame scoring; validation selection on every-frame losses)' % (SCN, D))
    print('controller: %.4f dB, %.3f probes/decision' % ct)
    for lab, r in out.items():
        if lab == 'controller': continue
        extra = ('  vs d-spaced controller %+.3f [%+.3f, %+.3f]' % tuple(r['gain_dspaced'])) if 'gain_dspaced' in r else ''
        print('%-44s %-34s test %.4f dB @ %.2f/dec | controller gain %+.3f [%+.3f, %+.3f]%s' % (
            lab, json.dumps(r.get('setting', r.get('slack'))), r['test'][0], r['test'][1], *r['gain'], extra))
json.dump(RESS, open(os.path.join(OUT, 'holdsel_%s.json' % SCN), 'w'), default=float)
print('HOLDSEL_DONE', SCN)
