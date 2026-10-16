#!/usr/bin/env python3
"""Reviewer check (2026-09-30), self-review 2.1 / cold A-M2, C-M5: re-select the sensing competitors
(confidence admission, sensor-prior fusion, always-on with periodic verification) with the SAME
criterion as the controller: minimize the mean validation regret over Scenarios 9, 31 and 32 against
each scenario's validation-selected feedback-only tracker; then evaluate once on the four test sets.
Controller class Ctl copied verbatim from needonly.py; ConfAdmit/SAFB from rev11/rev6; BayesFuse from rev10.
usage: python analysis/competitors.py configs/locked_v3.json"""
import numpy as np, pandas as pd, os, sys, glob, json, itertools
LK = json.load(open(sys.argv[1])); K = LK['selected']
C = 64
SL = {9: 0.4, 31: 0.6, 32: 0.8, 33: 0.8}          # validation-selected feedback-only slack (S33 inherits S32)
def load(s, split):
    root = f'dataset/scenario{s}'
    f = f'{root}/{split}_seqs_RA.csv'
    b = pd.read_csv(f'{root}/scenario{s}.csv') if os.path.exists(f'{root}/scenario{s}.csv') else pd.read_csv(glob.glob(f'{root}/*_dev.csv')[0])
    sel = set(pd.read_csv(f)['seq_index'].unique()); out = []
    for p, g in b.groupby('seq_index'):
        if p not in sel: continue
        g = g.sort_values('index')
        P = np.stack([np.nan_to_num(np.loadtxt(os.path.join(root, str(r).lstrip('./'))), nan=0.0) for r in g['unit1_pwr_60ghz']])
        out.append((P, [str(r) for r in g['unit1_pwr_60ghz']]))
    return out
def cachepair(s, split):
    """argmax predictions (for Ctl) and posteriors (for competitors) on the same anchors"""
    if split == 'val':
        a = np.load(f'results/predictions/both_scenario{s}_val.npz', allow_pickle=True)   # Option 3: corrected anchors
        p = a
    else:
        a = np.load(f'results/predictions/both_scenario{s}.npz', allow_pickle=True)
        p = np.load(f'results/predictions/post_scenario{s}.npz', allow_pickle=True)
    pred = {'vision': dict(zip(a['anchor'], a['vision'])), 'radar': dict(zip(a['anchor'], a['radar']))}
    post = dict(zip(p['anchor'], p['vision_post']))
    return pred, post
PRED = None   # set per evaluation (Ctl reads the module-level PRED)

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

def _trk(slack): return Ctl(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=slack)
def conf_of(q, kind):
    s = np.sort(q)[::-1]; return float(s[0]) if kind == 'pmax' else float(s[0] - s[1])
class ConfAdmit:
    """rev11 confidence admission; follow=True (published): the fallback tracker is updated with the served
    sensor beam; follow=False: it is left untouched while the sensor serves (only its clock advances)."""
    def __init__(self, th, kind='pmax', period=None, slack=0.4, src=None, follow=True):
        self.th, self.kind, self.period, self.slack, self.src, self.follow = th, kind, period, slack, src, follow
    def reset(self, a0, p0):
        self.trk = _trk(self.slack); self.trk.reset(a0, p0); self.a = int(a0); self.sv = 0; self.sense = self.trial = 0; self.served = 0
    def step(self, Pt, key):
        q = self.src.get(key)
        if q is None:
            self.a, k = self.trk.step(Pt, key); return self.a, k
        self.sense += 1
        if conf_of(q, self.kind) < self.th:
            self.a, k = self.trk.step(Pt, key); return self.a, k
        self.served += 1; p = int(np.argmax(q)); k = 0; self.sv += 1
        if self.period is not None and self.sv >= self.period:
            S = sorted({max(0,p-1), p, min(C-1,p+1)}); obs = {b: float(Pt[b]) for b in S}; k = len(S); self.a = max(obs, key=obs.get); self.sv = 0
        else: self.a = p
        self.trk.n += 1
        if self.follow:
            c = int(np.clip(round(self.trk.acq + self.trk.v*self.trk.n), 0, C-1)); self.trk._a(self.a, float(Pt[self.a]), c)
        return self.a, k
class SAFB:
    """rev6 always-on vision with feedback fallback and optional +-1 verification every `period` epochs"""
    def __init__(self, period=None, slack=0.4, pred=None): self.period, self.slack, self.pred = period, slack, pred
    def reset(self, a0, p0):
        self.trk = _trk(self.slack); self.trk.reset(a0, p0); self.a = int(a0); self.sv = 0; self.sense = self.trial = 0
    def step(self, Pt, key):
        pr = self.pred.get(key)
        if pr is None:
            self.a, k = self.trk.step(Pt, key); return self.a, k
        self.sense += 1; p = int(pr); k = 0; self.sv += 1
        if self.period is not None and self.sv >= self.period:
            S = sorted({max(0,p-1), p, min(C-1,p+1)}); obs = {b: float(Pt[b]) for b in S}; k = len(S); self.a = max(obs, key=obs.get); self.sv = 0
        else: self.a = p
        self.trk.n += 1; c = int(np.clip(round(self.trk.acq + self.trk.v*self.trk.n), 0, C-1)); self.trk._a(self.a, float(Pt[self.a]), c)
        return self.a, k
class BayesFuse:
    """rev10 Bayes filter with the sensor posterior fused into the predicted prior"""
    def __init__(self, sigma_v=1.0, eps=0.15, kmax=13, tau=2.0, fuse='none', a=1.0, src=None):
        self.sv, self.eps, self.kmax, self.tau, self.fuse, self.par, self.src = sigma_v, eps, kmax, tau, fuse, a, src
    def reset(self, a0, p0): self.post = np.zeros(C); self.post[int(a0)] = 1.0; self.vel = 0.0; self.a = int(a0); self.last = int(a0); self.sense = 0; self.trial = 0
    def _diffuse(self):
        m = np.roll(self.post, int(round(self.vel))); w = np.exp(-0.5 * (np.arange(-4, 5) / self.sv) ** 2); w /= w.sum()
        self.post = np.convolve(m, w, mode='same') + 1e-9; self.post /= self.post.sum()
    def step(self, Pt, key):
        self._diffuse(); q = self.src.get(key) if self.fuse != 'none' else None
        if q is not None:
            self.sense += 1; q = np.asarray(q, float) + 1e-6
            if self.fuse == 'pow': self.post = self.post * q ** self.par
            else: self.post = (1 - self.par) * self.post + self.par * q / q.sum()
            self.post /= self.post.sum()
        order = np.argsort(-self.post); c = np.cumsum(self.post[order])
        n = int(np.clip(np.searchsorted(c, 1 - self.eps) + 1, 1, self.kmax)); S = np.sort(order[:n]); obs = Pt[S]
        if obs.max() > 0:
            lik = np.zeros(C) + 1e-6; z = obs / (obs.max() + 1e-12); lik[S] = np.exp(self.tau * (z - 1.0))
            self.post = self.post * lik; self.post /= self.post.sum()
            b = int(S[int(np.argmax(obs))]); self.vel = 0.7 * self.vel + 0.3 * (b - self.last); self.last = b; self.a = b
        else: self.a = int(np.argmax(self.post))
        return self.a, len(S)

def run(ps, pol):
    pc, cc, calls = [], [], 0
    for P, keys in ps:
        pol.reset(int(P[0].argmax()), float(P[0].max())); c_, k_ = [], []
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: c_.append(float(P[t, a]) / m); k_.append(k)
        pc.append(np.array(c_)); cc.append(np.array(k_)); calls += getattr(pol, 'sense', 0) + getattr(pol, 'trial', 0)
    caps = np.concatenate(pc); loss = -10*np.log10(np.clip(caps, 1e-6, None))
    return dict(loss=float(loss.mean()), probes=float(np.concatenate(cc).mean()), model=100*calls/len(caps), pc=pc)
sn = lambda cs: (np.array([(-10*np.log10(np.clip(c,1e-6,None))).sum() for c in cs]), np.array([len(c) for c in cs], float))
def paired(ra, rb, B=100000):
    sa, na = sn(ra['pc']); sb, nb = sn(rb['pc']); n = len(sa); idx = np.random.default_rng(42).integers(0, n, (B, n))
    d = sa[idx].sum(1)/na[idx].sum(1) - sb[idx].sum(1)/nb[idx].sum(1)
    return float(sa.sum()/na.sum() - sb.sum()/nb.sum()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

VS = (9, 31, 32); TS = (9, 31, 32, 33)
VAL = {s: load(s, 'val') for s in VS}; TEST = {s: load(s, 'test') for s in TS}
VC = {s: cachepair(s, 'val') for s in VS}; TC = {s: cachepair(s, 'test') for s in TS}
def ev(s, split, mk):
    global PRED
    pred, post = (VC if split == 'val' else TC)[s]; PRED = pred
    return run((VAL if split == 'val' else TEST)[s], mk(s, pred, post))
print('#### reference and controller (validation / test)')
FBV = {s: ev(s, 'val', lambda s, pr, po: _trk(SL[s])) for s in VS}
CTV = {s: ev(s, 'val', lambda s, pr, po: Ctl(**K)) for s in VS}
for s in VS: print('  val S%-2d fb %.3f  controller %.3f  regret %+.3f' % (s, FBV[s]['loss'], CTV[s]['loss'], CTV[s]['loss'] - FBV[s]['loss']))
print('  method mean val regret %+.4f (sel3 primary: -0.0671)' % np.mean([CTV[s]['loss'] - FBV[s]['loss'] for s in VS]))
FBT = {s: ev(s, 'test', lambda s, pr, po: _trk(SL[s])) for s in TS}
CTT = {s: ev(s, 'test', lambda s, pr, po: Ctl(**K)) for s in TS}
for s in TS: print('  test S%-2d fb %.3f @ %.2f  controller %.3f @ %.2f, model %.1f%%' % (s, FBT[s]['loss'], FBT[s]['probes'], CTT[s]['loss'], CTT[s]['probes'], CTT[s]['model']))
def mreg(res): return float(np.mean([res[s]['loss'] - FBV[s]['loss'] for s in VS]))
RES = {'controller_val_regret': mreg(CTV), 'families': {}}

def select(name, cands, mk, cap=None):
    """cands: list of cfg dicts; mk(cfg) -> factory(s, pred, post). Returns best cfg under mean val regret
    (optionally with validation probes <= cap on every validation scenario)."""
    rows = []
    for cfg in cands:
        R = {s: ev(s, 'val', mk(cfg)) for s in VS}
        rows.append((mreg(R), max(R[s]['probes'] for s in VS), cfg, {s: (R[s]['loss'], R[s]['probes'], R[s]['model']) for s in VS}))
    # Option 3 (PREREG sec. 5): validation probes <= the reference tracker's on every validation scenario;
    # cap (fusion only) is an additional absolute cap kept from comp3 as a secondary row
    ok = [r for r in rows if all(r[3][s][1] <= FBV[s]['probes'] + 1e-9 for s in VS) and (cap is None or r[1] <= cap)]
    if not ok: return (None, None, None, None), rows
    best = min(ok, key=lambda r: (round(r[0], 4), r[1]))
    return best, rows
def report(label, cfg, mk):
    out = {}
    if cfg is None:
        print('  %s -> no setting meets the probe constraint on validation' % label); return None
    print('  %s -> %s' % (label, cfg))
    for s in TS:
        r = ev(s, 'test', mk(cfg)); d = paired(r, CTT[s]); v = paired(FBT[s], r)
        out[s] = dict(loss=r['loss'], probes=r['probes'], model=r['model'], minus_ctl=d, vos=v)
        print('     test S%-2d %.3f @ %.2f, model %5.1f%% | minus controller %+.3f [%+.3f, %+.3f] | value vs fb %+.3f [%+.3f, %+.3f]'
              % (s, r['loss'], r['probes'], r['model'], d[0], d[1], d[2], v[0], v[1], v[2]))
    return out

# ---------- always-on with periodic verification
print('\n#### always-on + verification every K (select K by the controller criterion)')
mk_ao = lambda cfg: (lambda s, pr, po: SAFB(cfg['K'], SL[s], pr['vision']))
best, rows = select('ao', [dict(K=k) for k in (None, 16, 8, 4, 2, 1)], mk_ao)
for r in rows: print('   val K=%-4s mean regret %+.3f  (%s)' % (r[2]['K'], r[0], ', '.join('S%d %.3f@%.2f' % (s, *r[3][s][:2]) for s in VS)))
RES['families']['always_on'] = dict(sel=best[2], val_regret=best[0], test=report('always-on selected', best[2], mk_ao))

# ---------- confidence admission (follow = published fallback update; freeze = fallback untouched)
print('\n#### confidence admission (one absolute threshold across scenarios)')
allv = {kind: np.concatenate([[conf_of(q, kind) for q in VC[s][1].values()] for s in VS]) for kind in ('pmax', 'margin')}
TH = {kind: sorted(set(np.round(np.quantile(allv[kind], np.linspace(0.02, 0.98, 25)), 4))) for kind in allv}
mk_ca = lambda cfg: (lambda s, pr, po: ConfAdmit(cfg['th'], cfg['kind'], cfg['per'], SL[s], po, cfg['follow']))
RES['families']['confidence'] = {}
for kind in ('pmax', 'margin'):
    for per in (None, 1):
        for follow in (True, False):
            cands = [dict(th=float(th), kind=kind, per=per, follow=follow) for th in TH[kind]] + [dict(th=9e9, kind=kind, per=per, follow=follow)]
            best, rows = select('ca', cands, mk_ca)
            lab = 'conf %s verify=%s %s' % (kind, per, 'follow' if follow else 'freeze')
            print('   %s: best val mean regret %s at th=%s (th=inf means never admit)' % (lab, None if best[0] is None else round(best[0], 4), None if best[2] is None else best[2]['th']))
            RES['families']['confidence'][lab] = dict(sel=best[2], val_regret=best[0], test=report(lab, best[2], mk_ca))

# ---------- sensor-prior fusion
print('\n#### sensor-prior fusion (Bayes filter x sensor posterior)')
FG = []
for sv in (1.0, 1.6):
    for eps in (0.05, 0.2, 0.5):
        for tau in (1.0, 3.0):
            for km in (1, 3, 5, 9, 13):
                FG.append(dict(sigma_v=sv, eps=eps, tau=tau, kmax=km, fuse='none', a=1.0))
                for fz, par in (('pow', 0.5), ('pow', 1.0), ('pow', 2.0), ('mix', 0.3), ('mix', 0.5), ('mix', 0.7)):
                    FG.append(dict(sigma_v=sv, eps=eps, tau=tau, kmax=km, fuse=fz, a=par))
mk_bf = lambda cfg: (lambda s, pr, po: BayesFuse(src=po, **cfg))
# SELF-CHECK: S9-only selection at budget 2.64 must reproduce the paper's 0.343 @ 2.28
rows9 = []
for cfg in FG:
    r = ev(9, 'val', mk_bf(cfg)); rows9.append((r['loss'], r['probes'], cfg))
el = sorted([r for r in rows9 if r[1] <= 2.64], key=lambda r: (round(r[0], 4), round(r[1], 3)))
r = ev(9, 'test', mk_bf(el[0][2])); print('   SELF-CHECK S9-only budget 2.64 -> %s : test %.3f @ %.2f (paper 0.343 @ 2.28)' % (el[0][2], r['loss'], r['probes']))
best, rows = select('bf', FG, mk_bf)
RES['families']['fusion'] = {}
print('   same criterion: best val mean regret %s' % (None if best[0] is None else round(best[0], 4)))
RES['families']['fusion']['free'] = dict(sel=best[2], val_regret=best[0], test=report('fusion, 3-site selected (no probe cap)', best[2], mk_bf))
for cap in (2.64, 4.0, 6.0):
    ok = [r for r in rows if r[1] <= cap]
    if not ok: print('   cap %.2f: none feasible' % cap); continue
    b = min(ok, key=lambda r: (round(r[0], 4), r[1]))
    print('   cap %.2f probes on every validation scenario: best val mean regret %+.3f' % (cap, b[0]))
    RES['families']['fusion']['cap%.2f' % cap] = dict(sel=b[2], val_regret=b[0], test=report('fusion, 3-site selected, val probes <= %.2f' % cap, b[2], mk_bf))
json.dump(RES, open('results/analysis/comp4.json', 'w'), default=lambda o: o if not isinstance(o, np.generic) else o.item(), indent=1)
print('COMP4_DONE')
