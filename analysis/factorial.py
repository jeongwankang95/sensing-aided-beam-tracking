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



# ======================= sensor x re-sweep factorial (2026-10-06) =======================
# Built as t2.py lines 1-246 + this tail. Same selected constants throughout; only the sensor (vision / none) and the
# scheduled re-sweep (rs_k=3 coarse / none) are switched. t2.boot: 10^5 paired pass resamples, seed 42, 5 decimals.
SCN = os.path.basename(root); s = int(SCN.replace('scenario', ''))
assert 'rs_k' in K
cells = {'neither': Ctl(**dict(K, sensor=None, rs_k=None)), 'sensor only': Ctl(**dict(K, rs_k=None)),
         're-sweep only': Ctl(**dict(K, sensor=None)), 'both (METHOD)': Ctl(**K)}
PP = {nm: perpass(pol) for nm, pol in cells.items()}
PP['plain tracker (ref slack)'] = perpass(_trk(TSL))
print('\n######## %s: %d test passes, factorial at the selected constants (slack %.1f), reference slack %s'
      % (SCN, len(passes), K['slack'], TSL))
for nm, pp in PP.items():
    st = stats(pp)
    print('%-26s loss %8.5f  probes %6.3f  model %5.1f%%  lock-loss passes %d' % (nm, st['loss'], st['probes'], st['model'], st['lockloss']))
def show(lab, a, b):
    r = boot(PP[a], PP[b]); print('  %-58s %+.5f [%+.5f, %+.5f]' % (lab, r[0], r[1], r[2])); return r
print('differences (first minus second; positive = second better):')
show('sensor without re-sweep   (neither - sensor only)', 'neither', 'sensor only')
show('sensor with re-sweep      (re-sweep only - both)', 're-sweep only', 'both (METHOD)')
show('re-sweep without sensor   (neither - re-sweep only)', 'neither', 're-sweep only')
show('re-sweep with sensor      (sensor only - both)', 'sensor only', 'both (METHOD)')
show('sensor only vs re-sweep only (re-sweep only - sensor only)', 're-sweep only', 'sensor only')
show('plain tracker - sensor only', 'plain tracker (ref slack)', 'sensor only')
# interaction: (neither - sensor only) - (re-sweep only - both), paired on the same resamples
sA = {k: np.array([x[0].sum() for x in v]) for k, v in PP.items()}; n_ = np.array([len(x[0]) for x in PP['neither']])
rng = np.random.default_rng(42); P_ = len(n_); d = np.empty(100000)
for i in range(100000):
    j = rng.integers(0, P_, P_); N = n_[j].sum()
    d[i] = ((sA['neither'][j].sum() - sA['sensor only'][j].sum()) - (sA['re-sweep only'][j].sum() - sA['both (METHOD)'][j].sum())) / N
pt = ((sA['neither'].sum() - sA['sensor only'].sum()) - (sA['re-sweep only'].sum() - sA['both (METHOD)'].sum())) / n_.sum()
print('  %-58s %+.5f [%+.5f, %+.5f]' % ('interaction (sensor value without - with re-sweep)', pt, np.percentile(d, 2.5), np.percentile(d, 97.5)))
json.dump({nm: stats(pp) for nm, pp in PP.items()}, open(f'results/analysis/fact3_{SCN}.json', 'w'), default=float)
print('FACT3_DONE', SCN)
