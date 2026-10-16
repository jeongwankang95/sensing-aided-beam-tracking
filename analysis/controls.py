#!/usr/bin/env python3
"""controls.py LOCKED ROOT SLACK --cache=CACHE: controls of the controller. Each block first reproduces Table II (SELF-CHECK):
the blacklist length with the innovation gate selected or held open, the trial log, decision-interval rows of the feedback-only
tracker and the controller, always-on vision with feedback fallback and verification every K frames, and (Scenario 9) the
per-pass statistics with a sign test and a sign-flip permutation test. Writes results/analysis/<scenario>.json.
Paper: Sec. VI-A (sign test p = 0.57, permutation test p = 0.009) and supplementary Secs. S17-A, S17-B and S17-D (Tables S11,
S12 and S14).
usage: python analysis/controls.py configs/locked_v3.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz"""
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

# ---- revision one-liners, 2026-09-22 ----------------------------------------------
# 1-1 gate x blacklist grid + trial log; 1-3 feedback-only slack 1.0 at d=1,2,4 and
# always-on + verify K=8; 1-5 exact per-pass statistics (S9). Every block first
# reproduces a published number (SELF-CHECK) before printing anything new.
import json as _json
from math import comb, erf, sqrt
SCN = os.path.basename(root)
OUT = 'results/analysis'; os.makedirs(OUT, exist_ok=True)
RES = {'scenario': SCN, 'slack_fb': TSL, 'cache': os.path.basename(CACHE)}

def _trk(slack):
    return Ctl(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=slack)

class SAFB:  # always-on vision with feedback fallback and optional periodic verification (ao_tail.py)
    def __init__(self, period=None, slack=0.4): self.period, self.slack = period, slack
    def reset(self, a0, p0):
        self.trk = _trk(self.slack); self.trk.reset(a0, p0)
        self.a = int(a0); self.sv = 0; self.sense = self.trial = 0
    def step(self, Pt, key):
        pr = PRED['vision'].get(key)
        if pr is None:
            self.a, k = self.trk.step(Pt, key); return self.a, k
        self.sense += 1; p = int(pr); k = 0; self.sv += 1
        if self.period is not None and self.sv >= self.period:
            S = sorted({max(0,p-1), p, min(C-1,p+1)}); obs = {b: float(Pt[b]) for b in S}
            k = len(S); self.a = max(obs, key=obs.get); self.sv = 0
        else: self.a = p
        self.trk.n += 1
        c = int(np.clip(round(self.trk.acq + self.trk.v*self.trk.n), 0, C-1))
        self.trk._a(self.a, float(Pt[self.a]), c)
        return self.a, k

class CtlLog(Ctl):  # deployed controller with an event log (trial opening frame, sigma at opening, outcome, sense exits)
    def reset(self, a0, p0):
        super().reset(a0, p0); self.t = 0; self.trials = []; self.exits = []
    def step(self, Pt, key):
        self.t += 1; tr0, s0, m0 = self.trial, self.sig, self.mode
        a, k = super().step(Pt, key)
        if self.trial > tr0: self.trials.append((self.t, float(s0), self.mode == 'sense'))
        if m0 == 'sense' and self.mode == 'track': self.exits.append('collapse' if self.bl > 0 else 'thoff')
        return a, k

def runp(ps, pol):
    """loss, probes/epoch, model%, trial%, sense%, p95, per-pass caps"""
    pc, cc, s_, t_ = [], [], 0, 0
    for P, keys in ps:
        pol.reset(int(P[0].argmax()), float(P[0].max())); caps, cost = [], []
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: caps.append(float(P[t, a]) / m); cost.append(k)
        pc.append(np.array(caps)); cc.append(np.array(cost))
        s_ += getattr(pol, 'sense', 0); t_ += getattr(pol, 'trial', 0)
    caps = np.concatenate(pc); cost = np.concatenate(cc); n = len(caps)
    loss = -10*np.log10(np.clip(caps, 1e-6, None))
    return dict(loss=float(loss.mean()), probes=float(cost.mean()), model=100*(s_+t_)/n,
                trial=100*t_/n, sense=100*s_/n, p95=float(np.percentile(loss, 95)), pc=pc)

def decim(d):
    ps = [(P[::d], k[::d]) for P, k in passes]
    return [(P, k) for P, k in ps if len(P) >= 6]

print('\n################ %s  (%d test passes, fb slack %.1f, cache %s)' % (SCN, len(passes), TSL, os.path.basename(CACHE)))

# ---------- SELF-CHECK: Table II ----------
fb = runp(passes, _trk(TSL)); ct = runp(passes, Ctl(**K))
print('SELF-CHECK Table II  feedback-only %.3f dB @ %.2f | controller %.3f dB @ %.2f, model %.1f%% (trial %.1f%% + sense %.1f%%)'
      % (fb['loss'], fb['probes'], ct['loss'], ct['probes'], ct['model'], ct['trial'], ct['sense']))
RES['tableII'] = dict(fb=(fb['loss'], fb['probes']), ctl=(ct['loss'], ct['probes'], ct['model'], ct['trial'], ct['sense']))

# ---------- 1-1  gate x blacklist grid ----------
print('\n[1-1] gate x blacklist grid (verification, margin, K unchanged)')
print('%-22s %4s | %7s %6s %7s %7s %7s %6s' % ('gate', 'bl', 'loss', 'p/ep', 'model%', 'trial%', 'sense%', 'p95'))
grid = {}
for gname, th_on, th_off in (('deployed 0.6/0.5', K['th_on'], K['th_off']),
                             ('open, thoff kept', -1.0, K['th_off']),
                             ('open, thoff off', -1.0, -1.0)):
    for bl in (0, 8, 16, 32, 64, 128, 256):
        cfg = dict(K); cfg.update(th_on=th_on, th_off=th_off, bl=bl)
        r = runp(passes, Ctl(**cfg))
        print('%-22s %4d | %7.3f %6.2f %7.1f %7.1f %7.1f %6.2f' % (gname, bl, r['loss'], r['probes'], r['model'], r['trial'], r['sense'], r['p95']))
        grid['%s|%d' % (gname, bl)] = {k: r[k] for k in ('loss', 'probes', 'model', 'trial', 'sense', 'p95')}
RES['grid'] = grid

# ---------- 1-1  trial log of the deployed controller ----------
pol = CtlLog(**K); per = []
for P, keys in passes:
    pol.reset(int(P[0].argmax()), float(P[0].max()))
    first_cov = next((t for t in range(1, len(P)) if PRED['vision'].get(keys[t]) is not None), None)
    for t in range(1, len(P)): pol.step(P[t], keys[t])
    per.append(dict(n=len(P)-1, first_cov=first_cov, trials=list(pol.trials), exits=list(pol.exits)))
ntr = sum(len(p['trials']) for p in per); nwin = sum(sum(1 for tr in p['trials'] if tr[2]) for p in per)
first_at_cov = sum(1 for p in per if p['trials'] and p['first_cov'] is not None and p['trials'][0][0] == p['first_cov'])
p_with = sum(1 for p in per if p['trials'])
open_frames = [tr[0] for p in per for tr in p['trials']]
sig_open = [tr[1] for p in per for tr in p['trials']]
exits = [e for p in per for e in p['exits']]
print('\n[1-1] trial log (deployed): trials %d (won %d) over %d passes; passes with >=1 trial %d; trials/pass max %d'
      % (ntr, nwin, len(per), p_with, max((len(p['trials']) for p in per), default=0)))
print('      first trial opened at the first covered frame in %d of %d passes with a trial; first covered frame median %s'
      % (first_at_cov, p_with, np.median([p['first_cov'] for p in per if p['first_cov'] is not None])))
if open_frames:
    print('      opening frame index: median %.0f, IQR [%.0f, %.0f]; sigma at opening: median %.2f, min %.2f'
          % (np.median(open_frames), np.percentile(open_frames, 25), np.percentile(open_frames, 75), np.median(sig_open), min(sig_open)))
print('      SENSE exits: %d via theta_off, %d via verified collapse (blacklist)' % (exits.count('thoff'), exits.count('collapse')))
print('      mean scored frames per pass %.1f; blacklist B=%d epochs' % (np.mean([p['n'] for p in per]), K['bl']))
RES['triallog'] = dict(trials=ntr, won=nwin, passes=len(per), passes_with_trial=p_with, first_at_cov=first_at_cov,
                       open_frames=open_frames, sig_open=sig_open, exits=exits, per=per)

# ---------- 1-3  feedback-only at slack 1.0 (controller's internal slack) at d = 1, 2, 4 ----------
print('\n[1-3a] decision-interval rows: feedback-only at deployed slack %.1f and at slack 1.0, controller (SELF-CHECK vs Table IV)' % TSL)
print('%2s | %13s | %13s | %20s | %8s %8s' % ('d', 'fb slack dep', 'fb slack 1.0', 'controller (loss p inv%)', 'VoI dep', 'VoI 1.0'))
dec = {}
for d in (1, 2, 4):
    ps = decim(d)
    a = runp(ps, _trk(TSL)); b = runp(ps, _trk(1.0)); c = runp(ps, Ctl(**K))
    print('%2d | %6.3f @%5.2f | %6.3f @%5.2f | %6.3f @%5.2f %5.1f%% | %+8.3f %+8.3f'
          % (d, a['loss'], a['probes'], b['loss'], b['probes'], c['loss'], c['probes'], c['model'], a['loss']-c['loss'], b['loss']-c['loss']))
    dec[d] = dict(fb_dep=(a['loss'], a['probes']), fb_10=(b['loss'], b['probes']), ctl=(c['loss'], c['probes'], c['model']))
RES['decim'] = dec
# slack sweep of the feedback-only tracker at d=1 (for the frontier context)
print('      slack sweep d=1: ' + '  '.join('%.1f:%.3f@%.2f' % ((s,) + (lambda r: (r['loss'], r['probes']))(runp(passes, _trk(s)))) for s in (0.4, 0.6, 0.8, 1.0, 1.2, 1.5)))

# ---------- 1-3b  always-on + verification period K ----------
print('\n[1-3b] always-on vision with feedback fallback, verification every K frames (SELF-CHECK: K=none/1/2 vs Tables II/III)')
ao = {}
for Kp in (None, 1, 2, 4, 8, 16):
    r = runp(passes, SAFB(period=Kp, slack=TSL))
    print('      K=%-4s loss %.3f @ %.2f probes, model %.1f%%' % (Kp, r['loss'], r['probes'], r['model']))
    ao[str(Kp)] = (r['loss'], r['probes'], r['model'])
RES['alwayson'] = ao

# ---------- 1-5  exact per-pass statistics (S9 only) ----------
if SCN == 'scenario9':
    mL = lambda c: float(np.mean(-10*np.log10(np.clip(c, 1e-6, None))))
    v = np.array([mL(a) - mL(b) for a, b in zip(fb['pc'], ct['pc'])]); n = len(v)
    print('\n[1-5] per-pass value: mean %.4f median %.4f positive %d/%d' % (v.mean(), np.median(v), (v > 0).sum(), n))
    kpos = int((v > 0).sum()); p_sign = sum(comb(n, i) for i in range(n+1) if abs(i-n/2) >= abs(kpos-n/2)) / 2**n
    r = np.argsort(np.argsort(np.abs(v))) + 1.0; Wp = r[v > 0].sum(); mu = n*(n+1)/4; sd = sqrt(n*(n+1)*(2*n+1)/24); z = (Wp-mu)/sd
    ncdf = lambda x: 0.5*(1+erf(x/sqrt(2))); p_w = 2*(1-ncdf(abs(z)))
    rng = np.random.default_rng(42); B = 100000
    flips = rng.choice([-1, 1], size=(B, n)); p_perm = float((np.abs((v*flips).mean(1)) >= abs(v.mean())).mean())
    idx = rng.integers(0, n, (B, n)); bs = v[idx].mean(1); pct = np.percentile(bs, [2.5, 97.5])
    def nppf(q):
        lo, hi = -10., 10.
        for _ in range(200):
            m = (lo+hi)/2
            if ncdf(m) < q: lo = m
            else: hi = m
        return (lo+hi)/2
    z0 = nppf(float((bs < v.mean()).mean())); jack = np.array([np.delete(v, i).mean() for i in range(n)]); jm = jack.mean()
    acc = ((jm-jack)**3).sum() / (6*(((jm-jack)**2).sum())**1.5)
    bca = lambda al: float(np.percentile(bs, 100*ncdf(z0 + (z0+nppf(al))/(1-acc*(z0+nppf(al))))))
    # frame-weighted paired bootstrap exactly as the paper (10^5 resamples, seed 42)
    sa = np.array([(-10*np.log10(np.clip(c,1e-6,None))).sum() for c in fb['pc']]); na = np.array([len(c) for c in fb['pc']], float)
    sb = np.array([(-10*np.log10(np.clip(c,1e-6,None))).sum() for c in ct['pc']]); nb = np.array([len(c) for c in ct['pc']], float)
    idx2 = np.random.default_rng(42).integers(0, n, (B, n))
    fw = sa[idx2].sum(1)/na[idx2].sum(1) - sb[idx2].sum(1)/nb[idx2].sum(1)
    print('      sign test p=%.3f | Wilcoxon z=%.2f p=%.3f | sign-flip permutation (mean) p=%.4f' % (p_sign, z, p_w, p_perm))
    print('      pass-mean CI: percentile [%.3f, %.3f], BCa [%.3f, %.3f] (z0=%.3f a=%.3f)' % (pct[0], pct[1], bca(0.025), bca(0.975), z0, acc))
    print('      SELF-CHECK frame-weighted paired bootstrap: %.4f [%.3f, %.3f]  (paper +0.65 [0.19, 1.17])'
          % (sa.sum()/na.sum()-sb.sum()/nb.sum(), np.percentile(fw, 2.5), np.percentile(fw, 97.5)))
    RES['stats'] = dict(v=v.tolist(), p_sign=p_sign, p_wilcoxon=p_w, p_perm=p_perm, pct=pct.tolist(), bca=[bca(0.025), bca(0.975)],
                        fw=[float(sa.sum()/na.sum()-sb.sum()/nb.sum()), float(np.percentile(fw, 2.5)), float(np.percentile(fw, 97.5))])

_json.dump(RES, open(os.path.join(OUT, SCN + '.json'), 'w'), default=float)
print('REV_DONE %s' % SCN)
