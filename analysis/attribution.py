import os, sys
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")
exec(compile(open(_CORE).read(), _CORE, "exec"))   # same namespace as the original prelude3.py + tail concatenation
# ---- Table III attribution controls, rebuilt on final_eval.py's locked controller ----
# CtlVar is Ctl.step with two hooks and nothing else changed:
#   cand(pred, Pt, c, self) -> the beam the verification trial evaluates
#   gate(self, pred)        -> whether a closed gate opens this frame
# With the default hooks it must reproduce Ctl exactly; that is checked below.
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

def _trk(slack):
    return Ctl(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=slack)

def run(pol):
    cs, ks, sf = [], [], 0
    for P, keys in passes:
        pol.reset(int(P[0].argmax()), float(P[0].max()))
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: cs.append(float(P[t, a]) / m); ks.append(k)
        sf += getattr(pol, 'sense', 0) + getattr(pol, 'trial', 0)
    c = np.array(cs); L = -10 * np.log10(np.clip(c, 1e-6, None))
    return L, np.mean(ks), 100 * sf / len(c), np.percentile(L, 95), \
           getattr(pol, 'won', None), getattr(pol, 'tried', None)

def perpass(pol):
    out = []
    for P, keys in passes:
        pol.reset(int(P[0].argmax()), float(P[0].max())); c = []
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: c.append(float(P[t, a]) / m)
        out.append(np.array(c))
    return out

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

print('\n=== %s (%d test passes) attribution, cache %s ===' % (root, len(passes), os.path.basename(CACHE)))
BASE = run(Ctl(**K)); VAR0 = run(CtlVar(**K))
print('  hook check: Ctl %.3f dB / %.2f p/f / %.1f%%   CtlVar %.3f / %.2f / %.1f%s'
      % (BASE[0].mean(), BASE[1], BASE[2], VAR0[0].mean(), VAR0[1], VAR0[2],
         '   IDENTICAL' if abs(BASE[0].mean()-VAR0[0].mean()) < 1e-9 else '   <-- MISMATCH'))

# --- match the random gate's invocation rate to the deployed one ---
TARGET = BASE[2]
lo, hi = 0.0, 1.0
for _ in range(14):
    mid = (lo + hi) / 2
    r = run(CtlVar(gate=gate_random(mid), **K))
    if r[2] < TARGET: lo = mid
    else: hi = mid
RATE = (lo + hi) / 2
print('  random-gate opening probability matched to %.1f%% invocations: p = %.4f' % (TARGET, RATE))

VARIANTS = [
    ('candidate replaced (perfect sensing information)', None, None),
    ('oracle candidate',        dict(cand=cand_oracle), None),
    ('candidate replaced (no sensing information)', None, None),
    ('own extrapolation',       dict(cand=cand_extrap), None),
    ('random offset',           dict(cand=cand_random), None),
    ('stale prediction',        dict(cand=cand_stale),  None),
    ('gate replaced (everything else unchanged)', None, None),
    ('random opening',          dict(gate=gate_random(RATE)), None),
    ('measurement age >=2',     dict(gate=gate_age(2)), None),
    ('measurement age >=3',     dict(gate=gate_age(3)), None),
    ('no gate at all',          dict(gate=gate_always), None),
]

n = len(passes); full = list(range(n)); rng = np.random.default_rng(42)
Lm = lambda cs, idx: float(np.mean(-10*np.log10(np.clip(np.concatenate([cs[i] for i in idx]), 1e-6, None))))
base_pp = perpass(Ctl(**K))

print('\n%-26s %8s %7s %7s %7s  %s' % ('policy', 'loss', 'p/f', 'model%', 'p95', 'delta vs ours [95% CI]'))
fb = run(_trk(TSL))
print('%-26s %8.3f %7.2f %7.1f %7.2f' % ('feedback-only tracker', fb[0].mean(), fb[1], fb[2], fb[3]))
print('%-26s %8.3f %7.2f %7.1f %7.2f   (deployed)' % ('need x verify (ours)', BASE[0].mean(), BASE[1], BASE[2], BASE[3]))
for nm, kw, _ in VARIANTS:
    if kw is None:
        print('  -- %s --' % nm); continue
    pol = CtlVar(**kw, **K); r = run(pol)
    pp = perpass(CtlVar(**kw, **K))
    d = Lm(pp, full) - Lm(base_pp, full)
    bs = np.array([Lm(pp, i) - Lm(base_pp, i) for i in (rng.integers(0, n, n) for _ in range(4000))])
    extra = ''
    if r[4] is not None and r[5]:
        extra = '   trials %d/%d won' % (r[4], r[5])
    print('%-26s %8.3f %7.2f %7.1f %7.2f  %+.3f [%+.3f, %+.3f]%s'
          % ('  ' + nm, r[0].mean(), r[1], r[2], r[3], d,
             np.percentile(bs, 2.5), np.percentile(bs, 97.5), extra))
print('ATTRIB_DONE')
