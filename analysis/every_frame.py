#!/usr/bin/env python3
"""every_frame.py LOCKED ROOT SLACK --cache=CACHE: frame decimation scored on every native frame. Each policy observes,
decides and probes only on every d-th frame (d = 1, 2, 4) and serves its last committed beam in between; every native frame
with a reading is scored. Policies: the plain tracker at the reference slack and at the slack re-selected for each d on
decimated validation passes (results/analysis/<scenario>_perd_slack.json, analysis/decimation_slack.py), the Scenario-9 re-sweep
reference re-selected the same way (results/analysis/resweep_decim_scenario9.json, analysis/resweep_decimation.py), the
controller with and without its sensor, and on Scenario 9 the controller with d-spaced sensor inputs
(results/predictions/dec2_s9.npz, dec4_s9.npz). Asserts that the decision-frame scores equal the decimated replay and that d = 1
equals the native replay, and checks them against results/analysis/decsk_<scenario>.json (analysis/decimation_sensor_off.py).
Paired bootstrap over passes (10^5 resamples, seed 42). Writes results/analysis/hold_<scenario>.json.
Paper: Table IV and Sec. VI-C, the decimation rows of Fig. 6, supplementary Sec. S1-G and Tables S27-S28 (printed by
analysis/table4.py).
usage: python analysis/every_frame.py configs/locked_v3.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz"""
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

# ======================= held-beam service scoring of decimation (2026-10-14, r81) =======================
# hold3.py = resweep_decim3.py lines 1-153 + this tail (the same construction as decsk3.py). Policies, constants and
# comparator selections are those of the decimation runs of record: the Scenario-9 re-sweep references come from
# resweep_decim_scenario9.json and the re-selected feedback-only slack from <scenario>_perd_slack.json (both selected
# there on decimated validation passes);
# nothing is selected here. The policy observes, decides and updates only on the frames that dec() keeps
# (t = d, 2d, ...). Between decisions it serves its last committed beam (frames 1..d-1 serve the link-setup beam),
# and every native frame t >= 1 with a reading is scored; intermediate frames are used for scoring only. Probes and
# model calls happen on decision frames only and are reported per native frame and per decision.
# Regression (asserted): (a) at d = 1 the held scores equal perpass() on the native test passes; (b) the
# decision-frame scores equal perpass() on dec(TEST0, d), i.e. the decimated replay behind Table IV and Table S28.
# Paired pass-level bootstrap boot() (10^5, seed 42), frame-weighted over the scored frames.
# usage: python analysis/every_frame.py configs/locked_v3.json dataset/scenario<s> <ref slack> --cache=results/predictions/both_scenario<s>.npz
def perpass_hold(passes, pol, D):
    out, dout = [], []
    for P, keys, pid in passes:
        if len(P[::D]) < 6: continue
        pol.reset(int(P[0].argmax()), float(P[0].max())); a = int(P[0].argmax())
        L, Kc, Ld, Kd = [], [], [], []
        for t in range(1, len(P)):
            k = 0
            if t % D == 0: a, k = pol.step(P[t], keys[t])
            m = float(P[t].max())
            if m > 0:
                l = -10*np.log10(max(float(P[t, a])/m, 1e-6)); L.append(l); Kc.append(k)
                if t % D == 0: Ld.append(l); Kd.append(k)
        calls = getattr(pol, 'sense', 0) + getattr(pol, 'trial', 0); sw = getattr(pol, 'sweeps', 0)
        out.append((np.array(L), np.array(Kc), pid, calls, sw)); dout.append((np.array(Ld), np.array(Kd), pid, calls, sw))
    return out, dout

def same(A, B):
    return len(A) == len(B) and all(x[2] == y[2] and len(x[0]) == len(y[0]) and np.allclose(x[0], y[0], atol=1e-12)
                                    and np.array_equal(x[1], y[1]) and x[3] == y[3] for x, y in zip(A, B))

SEL = json.load(open(os.path.join(OUT, 'resweep_decim_scenario9.json'))) if SCN == 'scenario9' else None
REC = json.load(open(os.path.join(OUT, 'decsk_%s.json' % SCN)))
PERD = json.load(open(os.path.join(OUT, '%s_perd_slack.json' % SCN)))
off = lambda: RS(**dict(K, sensor=None))
RES = {}
for D in (1, 2, 4):
    mk = [('plain', lambda: fb(TSL)), ('controller', lambda: ctl()), ('sensor off', off)]
    if SEL:
        s = SEL[str(D)]; sf, sc = s['sel_free'], s['sel_cap']
        mk.append(('re-sweep ref', lambda sf=sf: fb(sf['slack'], how=sf['how'], k=sf['k'])))
        if sc and sc != sf: mk.append(('re-sweep ref <=4', lambda sc=sc: fb(sc['slack'], how=sc['how'], k=sc['k'])))
    sp = PERD[str(D)]['slack']   # feedback-only slack re-selected on decimated validation (scenario*_perd_slack.json)
    if sp != TSL: mk.append(('plain, re-selected slack', lambda sp=sp: fb(sp)))
    H, Dd = {}, {}
    def run(nm, f):
        H[nm], Dd[nm] = perpass_hold(TEST0, f(), D)
        assert same(Dd[nm], perpass(dec(TEST0, D), f())), (SCN, D, nm, 'decision-frame regression failed')
        if D == 1: assert same(H[nm], perpass(TEST0, f())), (SCN, nm, 'd=1 regression failed')
    for nm, f in mk: run(nm, f)
    if SCN == 'scenario9' and D in (2, 4):
        z = np.load('results/predictions/dec%d_s9.npz' % D, allow_pickle=True)
        keep = dict(PRED['vision']); PRED['vision'] = dict(zip(z['anchor'], z['vision']))
        run('controller, d-spaced', ctl)
        PRED['vision'] = keep
    row = {nm: dict(held=summ(H[nm]), dec=summ(Dd[nm]), passes=len(H[nm])) for nm in H}
    pairs = [('plain', 'controller'), ('sensor off', 'controller')]
    if 're-sweep ref' in H: pairs.append(('re-sweep ref', 'controller'))
    if 're-sweep ref <=4' in H: pairs.append(('re-sweep ref <=4', 'controller'))
    if 'plain, re-selected slack' in H: pairs.append(('plain, re-selected slack', 'controller'))
    if 'controller, d-spaced' in H: pairs += [('plain', 'controller, d-spaced'), ('sensor off', 'controller, d-spaced')]
    if 'controller, d-spaced' in H and 're-sweep ref' in H: pairs.append(('re-sweep ref', 'controller, d-spaced'))
    for a_, b_ in pairs: row[a_ + ' - ' + b_] = dict(held=boot(H[a_], H[b_]), dec=boot(Dd[a_], Dd[b_]))
    RES[D] = row
    print('\n######## %s d=%d (%d test passes; held = every native frame, dec = decision frames only)' % (SCN, D, row['plain']['passes']))
    print('%-26s %9s %9s %9s %9s %8s %8s' % ('policy', 'held dB', 'dec dB', 'prb/frm', 'prb/dec', 'call%frm', 'call%dec'))
    for nm in H:
        h, d_ = row[nm]['held'], row[nm]['dec']
        print('%-26s %9.4f %9.4f %9.3f %9.3f %8.1f %8.1f' % (nm, h[0], d_[0], h[1], d_[1], h[2], d_[2]))
    for a_, b_ in pairs:
        h, d_ = row[a_ + ' - ' + b_]['held'], row[a_ + ' - ' + b_]['dec']
        print('paired [%s] - [%s]: held %+.3f [%+.3f, %+.3f] | dec %+.3f [%+.3f, %+.3f]' % (a_, b_, *h, *d_))
    # values of record for the decision-frame scores (decsk run): sensor off / with sensor / V_sense
    r = REC['opt_d%d' % D]
    print('record check (decsk_%s.json opt_d%d): off %.5f vs %.5f | on %.5f vs %.5f | V %+.5f vs %+.5f'
          % (SCN, D, r['off'][0], row['sensor off']['dec'][0], r['on'][0], row['controller']['dec'][0], r['v'][0],
             row['sensor off - controller']['dec'][0]))
    if 'controller, d-spaced' in H:
        r = REC['pess_d%d' % D]
        print('record check (pess_d%d): on %.5f vs %.5f' % (D, r['on'][0], row['controller, d-spaced']['dec'][0]))
    if 'plain, re-selected slack' in H:
        print('record check (perd_slack d=%d, slack %.1f): %.5f vs %.5f' % (D, sp, PERD[str(D)]['fb_res'][0], row['plain, re-selected slack']['dec'][0]))
    if SEL:
        rr = SEL[str(D)]['rows']
        print('record check (resweep_decim): plain %.4f vs %.4f | re-sweep ref %.4f vs %.4f'
              % (rr['reference tracker (native slack)'][0], row['plain']['dec'][0],
                 rr['re-sweep tracker, val-selected'][0], row['re-sweep ref']['dec'][0]))
json.dump(RES, open(os.path.join(OUT, 'hold_%s.json' % SCN), 'w'), default=float)
print('HOLD_DONE', SCN)
