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


# ---- attrib3.py lines 124-172 (CtlVar) and 198-219 (error distribution, hooks), verbatim ----
class CtlVar(Ctl):
    def __init__(self, cand=None, gate=None, seed=1234, **kw):
        super().__init__(**kw)
        self.cand = cand or (lambda pred, Pt, c, s: pred)
        self.gate = gate or (lambda s, pred: s.sig > s.th_on)
        self.seed = seed
    def reset(self, a0, p0):
        super().reset(a0, p0)
        self.rng = np.random.default_rng(self.seed)
        self.hist = []                      # past raw predictions, for the stale control
        self.won = self.tried = 0           # trials won / attempted
    def step(self, Pt, key):
        _r = self._rs(Pt)
        if _r is not None: return _r
        self.n += 1
        if self.bl > 0: self.bl -= 1
        c = int(np.clip(round(self.acq + self.v * self.n), 0, C - 1)); unc = max(self.sig, .2) * self.n
        raw = PRED[self.sensor].get(key) if self.sensor else None
        self.hist.append(raw)
        pred = None if raw is None else self.cand(raw, Pt, c, self)
        if self.mode == 'track' and self.gate(self, pred) and pred is not None and self.bl == 0:
            self.mode, self.tr = 'trial', 0
        if self.mode in ('trial', 'sense') and pred is None: self.mode = 'track'
        k = 0
        if self.mode == 'trial':
            self.tr += 1; self.trial += 1; p = int(pred); S = sorted({max(0, p-1), p, min(C-1, p+1)})
            obs = {b: float(Pt[b]) for b in S}; k = len(S); best = max(obs, key=obs.get); self.a = c
            if self.tr == 1: self.tried += 1
            if obs[best] >= float(Pt[self.a]) * 10 ** (self.margin / 10):
                self.mode, self.sv = 'sense', 0; self._a(best, obs[best], c); self.a = best; self.won += 1
            elif self.tr >= self.t_trial: self.mode, self.bl = 'track', self.blf
        elif self.mode == 'sense':
            self.sense += 1; self.sv += 1; p = int(pred)
            if self.sv >= self.verify or self.fire:
                S = sorted({max(0, p-1), p, min(C-1, p+1)}); obs = {b: float(Pt[b]) for b in S}; k = len(S)
                best = max(obs, key=obs.get); good = self._a(best, obs[best], p); self.sv = 0; self.a = best
                if not good: self.mode, self.bl = 'track', self.blf
            else: self.a = p
            if self.sig < self.th_off: self.mode = 'track'
        else:
            if self.fire or unc > self.slack or self.n >= self.refresh:
                W = int(np.clip(np.ceil(unc) * self.esc, 1, self.wcap)); S = list(range(max(0, c-W), min(C, c+W+1)))
                obs = {b: float(Pt[b]) for b in S}; k = len(S); best = max(obs, key=obs.get)
                self._a(best, obs[best], c); self.a = best
            else: self.a = c
        rss = float(Pt[self.a])
        if self.p_acq > 0 and 10 * np.log10(max(rss, 1e-12) / self.p_acq) < -self.d: self.fire = True
        return self.a, k

# --- the sensor's own signed error distribution, for the random-offset control ---
ERR = []
for P, keys in passes:
    for t in range(1, len(P)):
        pr = PRED['vision'].get(keys[t])
        if pr is not None and float(P[t].max()) > 0:
            ERR.append(int(pr) - int(P[t].argmax()))
ERR = np.array(ERR) if ERR else np.array([0])

# --- candidate hooks ---
def cand_oracle(pred, Pt, c, s):  return int(Pt.argmax())
def cand_extrap(pred, Pt, c, s):  return c
def cand_random(pred, Pt, c, s):  return int(np.clip(c + s.rng.choice(ERR), 0, C - 1))
def cand_stale(pred, Pt, c, s):
    h = [x for x in s.hist[:-20] if x is not None]
    return int(h[-1]) if h else int(pred)

# --- gate hooks ---
def gate_never(s, pred):  return False          # 'no gate at all' uses always-open
def gate_always(s, pred): return True
def gate_age(th):         return lambda s, pred: s.n >= th
def gate_random(rate):    return lambda s, pred: s.rng.random() < rate


# ======================= precision pass (2026-10-06) =======================
# Built as t2.py lines 1-246 (patched prelude, Persist/_trk/SAFB/LLR/RS, perpass, stats, boot) + attrib3.py's CtlVar
# and hooks (verbatim) + this tail. All intervals use t2.boot: 10^5 paired pass resamples, seed 42, a fresh generator
# per comparison, value = row minus METHOD (positive: METHOD better), printed to 5 decimals.
#   (a) every scenario: need-only ablation A (no trial) and B (no check), as defined in needonly3.py;
#   (b) Scenario 9: Table III attribution controls (random-gate rate re-matched by the same 14-step bisection as in
#       attrib3.py) and the always-on rows with verification period none / 2 / 1;
#   (c) Scenario 9: LLR without probing cap (validation selection sel_free of configs/llr_scenario9.json) and the
#       velocity-initialized start, as rows fb(**sel_free) and fb(init2=True) of rev_20260929/llr.py;
#   (d) every scenario: current-slot top-1 accuracy and coverage of the vision cache under decimation d = 1, 2, 4 with
#       consecutive inputs, logic verbatim from decacc_tail.py (acc_cov), for --cache and, if given, --oldcache.
SCN = os.path.basename(root); s = int(SCN.replace('scenario', ''))
assert 'rs_k' in K, 'run with configs/locked_v3.json'
FBK = dict(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0)
OUT = {'scenario': SCN, 'cache': CACHE, 'rows': {}, 'boot': {}, 'acc': {}}
M = perpass(Ctl(**K)); OUT['rows']['METHOD'] = stats(M)
def show(nm, pp, ref=M):
    st = stats(pp); b = boot(pp, ref); OUT['rows'][nm] = st; OUT['boot'][nm] = b
    print('%-34s %8.5f %6.3f %6.2f %6.3f   minus METHOD %+.5f [%+.5f, %+.5f]'
          % (nm, st['loss'], st['probes'], st['model'], st['p95'], b[0], b[1], b[2]))
    return pp
print('\n######## %s: %d test passes, reference slack %s, precision pass (cache %s)' % (SCN, len(passes), TSL, os.path.basename(CACHE)))
print('%-34s %8s %6s %6s %6s' % ('policy', 'loss', 'probes', 'model%', 'p95'))
st = OUT['rows']['METHOD']; print('%-34s %8.5f %6.3f %6.2f %6.3f' % ('METHOD', st['loss'], st['probes'], st['model'], st['p95']))
show('feedback-only tracker', perpass(_trk(TSL)))
show('own tracker + re-sweep, no sensor', perpass(Ctl(**dict(K, sensor=None))))
A = dict(K); A.update(margin=-99.0)
B = dict(K); B.update(margin=-99.0, verify=10**9, bl=0)
show('need only (A: no trial)', perpass(Ctl(**A)))
show('need only (B: no check)', perpass(Ctl(**B)))
if s == 9:
    V0 = stats(perpass(CtlVar(**K)))
    print('  hook check: CtlVar %.6f / %.4f / %.3f%s' % (V0['loss'], V0['probes'], V0['model'],
          '   IDENTICAL' if abs(V0['loss'] - st['loss']) < 1e-12 and abs(V0['model'] - st['model']) < 1e-12 else '   <-- MISMATCH'))
    TARGET = st['model']; lo, hi = 0.0, 1.0
    for _ in range(14):
        mid = (lo + hi) / 2
        if stats(perpass(CtlVar(gate=gate_random(mid), **K)))['model'] < TARGET: lo = mid
        else: hi = mid
    RATE = (lo + hi) / 2; OUT['rate'] = RATE
    print('  random-gate opening probability matched to %.2f%% invocations: p = %.4f' % (TARGET, RATE))
    for nm, kw in [('oracle candidate', dict(cand=cand_oracle)), ('own extrapolation', dict(cand=cand_extrap)),
                   ('random offset', dict(cand=cand_random)), ('stale prediction', dict(cand=cand_stale)),
                   ('random opening', dict(gate=gate_random(RATE))), ('measurement age >=2', dict(gate=gate_age(2))),
                   ('measurement age >=3', dict(gate=gate_age(3))), ('no gate at all', dict(gate=gate_always))]:
        show(nm, perpass(CtlVar(**kw, **K)))
    for nm, per in [('always-on, no verification', None), ('always-on, verify period 2', 2), ('always-on, verify period 1', 1)]:
        show(nm, perpass(SAFB(period=per, slack=TSL)))
    SF = json.load(open('configs/llr_scenario9.json'))['sel_free']
    print('  LLR sel_free (no cap):', SF)
    show('LLR, no probing cap', perpass(LLR(**dict(FBK, slack=TSL), **SF)))
    show('velocity-initialized start', perpass(LLR(init2=True, **dict(FBK, slack=TSL))))

def acc_cov(pred_map, d):
    hit = tot = scored = 0
    for P, keys in passes:
        Pd, kd = P[::d], keys[::d]
        if len(Pd) < 6: continue
        for t in range(1, len(Pd)):
            m = float(Pd[t].max())
            if m <= 0: continue
            scored += 1
            pr = pred_map.get(kd[t])
            if pr is None or int(pr) < 0: continue
            tot += 1
            if int(pr) == int(Pd[t].argmax()): hit += 1
    return (100 * hit / tot if tot else float('nan'), 100 * tot / scored if scored else 0.0)
OLD = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--oldcache=')), None)
caches = [('cache', PRED['vision'])]
if OLD and os.path.exists(OLD):
    oc = np.load(OLD, allow_pickle=True); caches.append(('oldcache', dict(zip(oc['anchor'], oc['vision']))))
elif OLD: print('  oldcache %s not found' % OLD)
for tag, pm in caches:
    r = [acc_cov(pm, d) for d in (1, 2, 4)]; OUT['acc'][tag] = r
    print('  accuracy, consecutive inputs [%s]: d=1 %.2f%%  d=2 %.2f%%  d=4 %.2f%%   coverage %.1f / %.1f / %.1f%%'
          % (tag, r[0][0], r[1][0], r[2][0], r[0][1], r[1][1], r[2][1]))
json.dump(OUT, open(f'results/analysis/prec_{SCN}.json', 'w'), default=float)
print('PREC_DONE', SCN)
