#!/usr/bin/env python3
"""selection.py: the validation re-selection of the controller constants (2026-10-02, pre-registered; Sec. V and supplementary
Sec. S17-N). VALIDATION ONLY: no test pass is loaded by this script. It scores the reference trackers, the re-sweep and
lock-loss-recovery families and the 136,080-setting method grid on the validation passes of Scenarios 9, 31 and 32, applies
the pre-registered criterion, and writes results/analysis/sel3_val.json (configs/sel3_val.json is this output; its "primary"
entry is configs/locked_v3.json) and results/analysis/sel3_val_grid.npz (every setting's validation loss, probes,
invocation share and sweeps). Self-checks: the deployed 2026-08-10 constants on the old caches reproduce
configs/locked_multi.json, and the family grids reproduce configs/resweep_scenario*.json and configs/llr_scenario*.json.
The original script was built by concatenation; classes Ctl (needonly.py), RS (src/rs_classes.py) and LLR
(src/llr_tail_classes.py) are kept verbatim below. Run-time lines go to stderr.
usage: python analysis/selection.py [--workers=N]   (the paper's run used 64 workers; about a minute on 64 cores)"""
import numpy as np, pandas as pd, os, sys, glob, json, itertools, time
import multiprocessing as mp
C = 64
VS = (9, 31, 32)
REF_SL = {9: 0.4, 31: 0.6, 32: 0.8}
NW = int(next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--workers=')), 48))
def load(s, split):
    assert split == 'val'
    root = f'dataset/scenario{s}'
    base = (pd.read_csv(f'{root}/scenario{s}.csv') if os.path.exists(f'{root}/scenario{s}.csv')
            else pd.read_csv(glob.glob(f'{root}/*_dev.csv')[0]))
    sel = set(pd.read_csv(f'{root}/{split}_seqs_RA.csv')['seq_index'].unique()); out = []
    for p, g in base.groupby('seq_index'):
        if p not in sel: continue
        g = g.sort_values('index')
        P = np.stack([np.nan_to_num(np.loadtxt(os.path.join(root, str(r).lstrip('./'))), nan=0.0)
                      for r in g['unit1_pwr_60ghz']])
        out.append((P, [str(r) for r in g['unit1_pwr_60ghz']], p))
    return out
def cache(f):
    d = np.load(f, allow_pickle=True); return {k: dict(zip(d['anchor'], d[k])) for k in ('radar', 'vision')}
VAL = {s: load(s, 'val') for s in VS}
NEW = {s: cache(f'results/predictions/both_scenario{s}_val.npz') for s in VS}      # corrected anchors (2026-10-02)
OLD = {s: cache(f'results/predictions/sensor_preds_val_scenario{s}.npz') for s in VS}  # used by the 2026-08-10 selection
PRED = NEW[9]

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


# ======================= Option 3 tail (2026-10-02) =======================
def score(ps, pol):
    caps, cost, calls, sw = [], [], 0, 0
    for P, keys, pid in ps:
        pol.reset(int(P[0].argmax()), float(P[0].max()))
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: caps.append(float(P[t, a]) / m); cost.append(k)
        calls += getattr(pol, 'sense', 0) + getattr(pol, 'trial', 0); sw += getattr(pol, 'sweeps', 0)
    caps = np.array(caps)
    return (float(np.mean(-10*np.log10(np.clip(caps, 1e-6, None)))), float(np.mean(cost)),
            100*calls/len(caps), sw, len(caps))
def on(s, src):
    global PRED
    PRED = src[s]
FBK = dict(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0)
DEPLOYED = dict(th_on=0.6, th_off=0.5, margin=0.5, t_trial=1, bl=64, verify=8, slack=1.0)
T0 = time.time(); OUT = {}
print('validation passes:', {s: len(VAL[s]) for s in VS})
print('val anchors: new', {s: len(NEW[s]['vision']) for s in VS}, ' old', {s: len(OLD[s]['vision']) for s in VS})

# ---- 1. reference trackers (feedback-only, cache-independent) and self-checks vs 2026-08-10
LK = json.load(open('configs/locked_multi.json')); FB = {}
for s in VS:
    rows = [(score(VAL[s], RS(slack=sl, **FBK)), sl) for sl in (0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0)]
    best = min(rows, key=lambda r: (round(r[0][0], 4), round(r[0][1], 3))); FB[s] = best[0]
    print('reference tracker S%d: val-selected slack %s  loss %.4f  probes %.3f  (%d scored frames)'
          % (s, best[1], best[0][0], best[0][1], best[0][4]))
    assert best[1] == REF_SL[s] and abs(best[0][0] - LK['feedback_reference'][f'dataset/scenario{s}']) < 1e-9
reg = {}
for s in VS:
    on(s, OLD); r = score(VAL[s], RS(**DEPLOYED)); reg[s] = (r[0] - FB[s][0], r[1], r[2])
m_old = (np.mean([v[0] for v in reg.values()]), np.mean([v[1] for v in reg.values()]), np.mean([v[2] for v in reg.values()]))
print('SELF-CHECK deployed constants, OLD caches: regret %s mean %+.6f (locked %+.6f) probes %.4f (locked %.4f) inv %.3f (locked %.3f)'
      % ({s: round(v[0], 4) for s, v in reg.items()}, m_old[0], LK['mean_regret'], m_old[1], LK['val_probes'], m_old[2], LK['val_invocation_pct']))
assert abs(m_old[0] - LK['mean_regret']) < 1e-9 and abs(m_old[1] - LK['val_probes']) < 1e-9

# ---- 2. feedback families on validation (per scenario), self-checked against their saved runs
RSG = [dict(slack=sl, how=h, k=k) for sl in (0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0) for h in ('full', 'coarse')
       for k in (2, 3, 5, 7, 10, 15, 20)]
LLRG = [dict(X=X, N=N, how=h, ref=r) for X in (3, 4, 5, 6, 8, 10) for N in (1, 2, 3)
        for h in ('full', 'coarse') for r in ('sweep', 'runmax')]
FAM = {'resweep': {}, 'llr': {}}
for s in VS:
    rr = [(g, score(VAL[s], RS(how=g['how'], k=g['k'], **dict(FBK, slack=g['slack'])))) for g in RSG]
    old = json.load(open(f'configs/resweep_scenario{s}.json'))['val']
    assert all(a[0] == b[0] for a, b in zip(rr, old))
    print('SELF-CHECK re-sweep val grid S%d vs 2026-09-30: max |dloss|+|dprobes| = %.1e'
          % (s, max(abs(a[1][0] - b[1][0]) + abs(a[1][1] - b[1][1]) for a, b in zip(rr, old))))
    ll = [(g, score(VAL[s], LLR(**dict(FBK, slack=REF_SL[s]), **g))) for g in LLRG]
    old = json.load(open(f'configs/llr_scenario{s}.json'))['val']
    assert all(a[0] == b[0] for a, b in zip(ll, old))
    print('SELF-CHECK LLR val grid S%d vs 2026-09-29: max |dloss|+|dprobes| = %.1e'
          % (s, max(abs(a[1][0] - b[1][0]) + abs(a[1][1] - b[1][1]) for a, b in zip(ll, old))))
    FAM['resweep'][s], FAM['llr'][s] = rr, ll
def pick_fb(rows, cap):
    ok = [r for r in rows if r[1][1] <= cap + 1e-9]
    return min(ok, key=lambda r: (round(r[1][0], 4), round(r[1][1], 3))) if ok else None
OUT['families'] = {}
for fam in ('resweep', 'llr'):
    for s in VS:
        same, free = pick_fb(FAM[fam][s], FB[s][1]), pick_fb(FAM[fam][s], 9e9)
        OUT['families'][f'{fam}_S{s}'] = dict(same_criterion=same and [same[0], same[1][:4]], free=[free[0], free[1][:4]])
        print('%-8s S%-2d same criterion (val probes <= %.3f): %s -> %s | any budget: %s -> %s'
              % (fam, s, FB[s][1], same and same[0], same and '%.4f @ %.3f' % same[1][:2], free[0], '%.4f @ %.3f' % free[1][:2]))
print('[%.0f s] families done' % (time.time() - T0), file=sys.stderr)

# ---- 3. the method grid on the corrected validation caches
GRID = dict(th_on=[0.4, 0.6, 0.8, 1.0], th_off=[0.3, 0.5, 0.65], margin=[0.0, 0.5, 1.0, 2.0], t_trial=[1, 2, 4],
            bl=[32, 64, 256], verify=[2, 4, 8, 16], slack=[0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0])
RSO = [None] + [(h, k) for h in ('full', 'coarse') for k in (2, 3, 5, 7, 10, 15, 20)]
CFGS = [(dict(zip(GRID, v)), ro) for v in itertools.product(*GRID.values()) if v[1] < v[0] for ro in RSO]
def work(i):
    cfg, ro = CFGS[i]; out = []
    for s in VS:
        on(s, NEW)
        out.append(score(VAL[s], RS(**cfg) if ro is None else RS(how=ro[0], k=ro[1], **cfg))[:4])
    return out
print('grid: %d settings (%d with re-sweep), %d workers' % (len(CFGS), sum(ro is not None for _, ro in CFGS), NW))
t1 = time.time(); _ = [work(i) for i in range(20)]
print('  timing: %.2f s per setting single-core' % ((time.time() - t1) / 20), file=sys.stderr)
with mp.get_context('fork').Pool(NW) as pool:
    R = pool.map(work, range(len(CFGS)), chunksize=100)
A = np.array(R, dtype=float)                       # (n, 3 scenarios, [loss, probes, inv%, sweeps])
print('[%.0f s] grid done' % (time.time() - T0), file=sys.stderr)
np.savez_compressed('results/analysis/sel3_val_grid.npz', A=A, cfg=np.array([json.dumps(c) for c, _ in CFGS]),
                    ro=np.array([json.dumps(ro) for _, ro in CFGS]))

# ---- 4. selection (PREREG sec. 4 and 6)
L, P, I = A[:, :, 0], A[:, :, 1], A[:, :, 2]
Lref = np.array([FB[s][0] for s in VS]); Pref = np.array([FB[s][1] for s in VS]); NF = np.array([FB[s][4] for s in VS], float)
REG = L - Lref; feas = (P <= Pref + 1e-9).all(1); hasrs = np.array([ro is not None for _, ro in CFGS])
obj_eq, obj_fw = REG.mean(1), (REG * NF).sum(1) / NF.sum()
def choose(mask, obj):
    idx = np.where(mask)[0]
    if len(idx) == 0: return None
    return int(min(idx, key=lambda i: (round(obj[i], 4), round(P[i].mean(), 3), round(I[i].mean(), 2))))
def desc(i, obj=obj_eq):
    cfg, ro = CFGS[i]
    return dict(index=i, cfg=cfg, resweep=ro, objective=float(obj[i]), feasible=bool(feas[i]),
                per_scenario={f'S{s}': dict(loss=float(L[i, j]), regret=float(REG[i, j]), probes=float(P[i, j]),
                                            ref_probes=float(Pref[j]), inv=float(I[i, j]), sweeps=int(A[i, j, 3]))
                              for j, s in enumerate(VS)}, mean_probes=float(P[i].mean()), mean_inv=float(I[i].mean()))
print('feasible (val probes <= reference on all three): %d of %d with re-sweep, %d of %d without'
      % ((feas & hasrs).sum(), hasrs.sum(), (feas & ~hasrs).sum(), (~hasrs).sum()))
SEL = {'primary': choose(hasrs & feas, obj_eq), 'S1_frame_weighted': choose(hasrs & feas, obj_fw),
       'S2_loss_only': choose(hasrs, obj_eq), 'S3_no_resweep': choose(~hasrs & feas, obj_eq)}
dep = next(i for i, (c, ro) in enumerate(CFGS) if c == DEPLOYED and ro is None)
OUT['reference'] = {f'S{s}': dict(slack=REF_SL[s], loss=FB[s][0], probes=FB[s][1], frames=FB[s][4]) for s in VS}
OUT['selected'] = {k: desc(v, obj_fw if k.startswith('S1') else obj_eq) for k, v in SEL.items()}
rank = lambda i, mask: int((obj_eq[mask] < obj_eq[i] - 1e-12).sum()) + 1
OUT['S4_deployed'] = dict(desc(dep), rank_all_feasible=rank(dep, feas), n_all_feasible=int(feas.sum()),
                          rank_noresweep_feasible=rank(dep, feas & ~hasrs), n_noresweep_feasible=int((feas & ~hasrs).sum()))
for k, d in list(OUT['selected'].items()) + [('S4_deployed', OUT['S4_deployed'])]:
    ps = d['per_scenario']
    print('%-18s obj %+.4f feas %s | %s | resweep %s | %s'
          % (k, d['objective'], d['feasible'], ' '.join('%s=%s' % kv for kv in d['cfg'].items()), d['resweep'],
             '  '.join('%s %.3f(%+.3f)@%.2f/%.2f inv %.1f%%' % (s, v['loss'], v['regret'], v['probes'], v['ref_probes'], v['inv'])
                       for s, v in ps.items())))
pi = SEL['primary']; o = obj_eq[hasrs & feas]
OUT['plateau'] = {f'within_{t}dB': int((o <= obj_eq[pi] + t).sum()) for t in (0.001, 0.005, 0.01, 0.02, 0.05)}
print('plateau (feasible re-sweep settings within t dB of the primary):', OUT['plateau'])
top = sorted(np.where(hasrs & feas)[0], key=lambda i: (round(obj_eq[i], 4), round(P[i].mean(), 3), round(I[i].mean(), 2)))[:15]
print('top 15 under the primary criterion:')
for i in top:
    c, ro = CFGS[i]
    print('  %+.4f  probes %.2f inv %5.1f%% | %s | %s' % (obj_eq[i], P[i].mean(), I[i].mean(), ' '.join('%s=%s' % kv for kv in c.items()), ro))
OUT['top15'] = [desc(int(i)) for i in top]
json.dump(OUT, open('results/analysis/sel3_val.json', 'w'), indent=1, default=float)
print('[%.0f s]' % (time.time() - T0), file=sys.stderr)
print('-> results/analysis/sel3_val.json\nSEL3_DONE')
