import os, sys
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")
exec(compile(open(_CORE).read(), _CORE, "exec"))   # same namespace as the original prelude3.py + tail concatenation

# ======================= Option 3 (2026-10-02): Table II rows with the re-selected method =======================
# Run on prelude3 (patched Ctl) with configs/locked_v3.json. Persist/SAFB verbatim from exceed_tail.py,
# LLR verbatim from rev_20260929/llr_tail_classes.py, RS verbatim from rev_20260930/rs_classes.py.
class Persist:
    """Causal persistence: hold the link-setup beam, never probe."""
    def reset(self, a0, p0): self.a = int(a0); self.sense = self.trial = 0
    def step(self, Pt, key): return self.a, 0
def _trk(slack):
    return Ctl(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=slack)
class SAFB:
    def __init__(self, period=None, slack=0.4):
        self.period, self.slack = period, slack
    def reset(self, a0, p0):
        self.trk = _trk(self.slack); self.trk.reset(a0, p0)
        self.a = int(a0); self.sv = 0; self.sense = self.trial = 0
    def step(self, Pt, key):
        pr = PRED['vision'].get(key)
        if pr is None:
            self.a, k = self.trk.step(Pt, key)
            return self.a, k
        self.sense += 1; p = int(pr); k = 0; self.sv += 1
        if self.period is not None and self.sv >= self.period:
            S = sorted({max(0, p - 1), p, min(C - 1, p + 1)})
            obs = {b: float(Pt[b]) for b in S}
            k = len(S); self.a = max(obs, key=obs.get); self.sv = 0
        else:
            self.a = p
        self.trk.n += 1
        c = int(np.clip(round(self.trk.acq + self.trk.v * self.trk.n), 0, C - 1))
        self.trk._a(self.a, float(Pt[self.a]), c)
        return self.a, k

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

def perpass(pol):
    out = []
    for P, keys in passes:
        pol.reset(int(P[0].argmax()), float(P[0].max())); L, Kc = [], []
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: L.append(-10*np.log10(max(float(P[t, a])/m, 1e-6))); Kc.append(k)
        out.append((np.array(L), np.array(Kc, float), getattr(pol, 'sense', 0) + getattr(pol, 'trial', 0),
                    getattr(pol, 'sweeps', 0)))
    return out
def stats(pp):
    L = np.concatenate([x[0] for x in pp]); Kc = np.concatenate([x[1] for x in pp])
    return dict(loss=float(L.mean()), probes=float(Kc.mean()), model=100*sum(x[2] for x in pp)/len(L),
                p95=float(np.percentile(L, 95)), gt1=100*float((L > 1).mean()), gt2=100*float((L > 2).mean()),
                gt3=100*float((L > 3).mean()), lockloss=int(sum(x[0].mean() >= 2.0 for x in pp)), frames=int(len(L)))
def boot(ppA, ppB, B=100000, seed=42):
    sA = np.array([x[0].sum() for x in ppA]); sB = np.array([x[0].sum() for x in ppB]); n = np.array([len(x[0]) for x in ppA])
    pt = (sA.sum() - sB.sum())/n.sum(); rng = np.random.default_rng(seed); P = len(n); d = np.empty(B)
    for i in range(B):
        j = rng.integers(0, P, P); d[i] = (sA[j].sum() - sB[j].sum())/n[j].sum()
    return float(pt), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
def prange(ppA, ppB):
    d = [(a[0].sum() - b[0].sum())/len(a[0]) for a, b in zip(ppA, ppB)]; return float(min(d)), float(max(d))

SCN = os.path.basename(root); s = int(SCN.replace('scenario', '')); fs = s if s in (9, 31, 32) else 32
SJ = json.load(open('configs/sel3_val.json'))
LLR_OLD = json.load(open(f'configs/llr_scenario{fs}.json'))['sel_cap']
RS_FREE = {9: dict(slack=0.4, how='full', k=3), 31: dict(slack=0.4, how='coarse', k=2), 32: dict(slack=0.8, how='full', k=2)}[fs]
RS_SAME = SJ['families'][f'resweep_S{fs}']['same_criterion'][0]; LLR_SAME = SJ['families'][f'llr_S{fs}']['same_criterion'][0]
FBK = dict(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0)
assert 'rs_k' in K, 'run with configs/locked_v3.json'
noK = dict(K); noK.update(margin=-99.0, verify=10**9, bl=0)
ROWS = [('causal persistence', Persist()), ('feedback-only tracker', _trk(TSL))]
if s == 9: ROWS += [('plain tracker slack %.1f' % sl, _trk(sl)) for sl in (0.6, 0.8, 1.0, 1.5, 2.0)]
ROWS += [('own tracker + re-sweep, no sensor', Ctl(**dict(K, sensor=None))),
         ('LLR, <=4 probes (09-29)', LLR(**dict(FBK, slack=TSL), **LLR_OLD)),
         ('LLR, same criterion', LLR(**dict(FBK, slack=TSL), **LLR_SAME)),
         ('re-sweep, any budget (09-30)', RS(how=RS_FREE['how'], k=RS_FREE['k'], **dict(FBK, slack=RS_FREE['slack']))),
         ('re-sweep, same criterion', RS(how=RS_SAME['how'], k=RS_SAME['k'], **dict(FBK, slack=RS_SAME['slack'])))]
if s == 9: ROWS += [('re-sweep, <=4 probes (09-30)', RS(how='coarse', k=5, **dict(FBK, slack=1.0)))]
ROWS += [('METHOD', Ctl(**K)), ('unverified invocation', Ctl(**noK)),
         ('always-on', SAFB(slack=TSL)), ('always-on + verification', SAFB(period=1, slack=TSL))]
PP, OUT = {}, {'scenario': SCN, 'K': K, 'rows': {}, 'boot': [], 'perpass': {}}
print('\n######## %s: %d test passes, reference slack %s; LLR<=4 %s; LLR same %s; RS free %s; RS same %s'
      % (SCN, len(passes), TSL, LLR_OLD, LLR_SAME, RS_FREE, RS_SAME))
print('%-36s %7s %6s %6s %6s %6s %6s %6s %3s' % ('policy', 'loss', 'probes', 'model%', 'p95', '>1dB%', '>2dB%', '>3dB%', 'LL'))
for nm, pol in ROWS:
    PP[nm] = perpass(pol); st = stats(PP[nm]); OUT['rows'][nm] = st
    OUT['perpass'][nm] = [[float(x[0].sum()), int(len(x[0])), float(x[1].sum()), int(x[2]), int(x[3])] for x in PP[nm]]
    print('%-36s %7.3f %6.2f %6.1f %6.2f %6.1f %6.1f %6.1f %3d' % (nm, st['loss'], st['probes'], st['model'], st['p95'],
                                                              st['gt1'], st['gt2'], st['gt3'], st['lockloss']))
print('value = reference minus METHOD (positive: METHOD better); 10^5 paired pass bootstrap, seed 42' + ('; range over passes' if len(passes) <= 3 else ''))
for nm, _ in ROWS:
    if nm in ('METHOD',) or nm.startswith('plain tracker'): continue
    b = boot(PP[nm], PP['METHOD']); r = prange(PP[nm], PP['METHOD'])
    OUT['boot'].append((nm, 'METHOD') + b + r)
    print('  %-36s %+.3f [%+.3f, %+.3f]   pass range %+.3f..%+.3f' % ((nm,) + b + r))
fbm = np.array([x[0].mean() for x in PP['feedback-only tracker']]); mm = np.array([x[0].mean() for x in PP['METHOD']])
v = np.sort(fbm - mm); OUT['perpass_value_sorted'] = [float(x) for x in v]
nfr = np.array([len(x[0]) for x in PP['METHOD']])
print('per-pass value (fb - METHOD), sorted:', ' '.join('%.2f' % x for x in v))
print('  frame-weighted mean %+.3f, unweighted mean %+.3f, median %+.3f, passes > 1 dB: %d, positive: %d of %d'
      % (float(((fbm - mm) * nfr).sum() / nfr.sum()), float(v.mean()), float(np.median(v)), int((v > 1).sum()), int((v > 0).sum()), len(v)))
json.dump(OUT, open(f'results/analysis/t2_{SCN}.json', 'w'), default=float)
print('T2_DONE', SCN)
