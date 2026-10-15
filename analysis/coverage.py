import os, sys
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")
exec(compile(open(_CORE).read(), _CORE, "exec"))   # same namespace as the original prelude3.py + tail concatenation

# ---- controller against always-on under the deployed and the extended cache ----
def _trk(slack):
    return Ctl(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1,
               bl=0, margin=0, slack=slack)

class SAFB:
    def __init__(self, period=None, slack=0.4):
        self.period, self.slack = period, slack
    def reset(self, a0, p0):
        self.trk = _trk(self.slack); self.trk.reset(a0, p0)
        self.a = int(a0); self.sv = 0; self.sense = self.trial = 0
    def step(self, Pt, key):
        pr = PRED['vision'].get(key)
        if pr is None:
            self.a, k = self.trk.step(Pt, key); return self.a, k
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

def perpass(pol):
    cs = []
    for P, keys in passes:
        pol.reset(int(P[0].argmax()), float(P[0].max()))
        c = []
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: c.append(float(P[t, a]) / m)
        cs.append(np.array(c))
    return cs

def L(cs, idx):
    c = np.concatenate([cs[i] for i in idx])
    return float(np.mean(-10 * np.log10(np.clip(c, 1e-6, None))))

ctl = perpass(Ctl(**K)); ao = perpass(SAFB(slack=TSL)); fb = perpass(_trk(TSL))
n = len(passes); full = list(range(n)); rng = np.random.default_rng(42)
print('\ncache %s' % CACHE)
for nm, a, b in [('feedback-only  - controller', fb, ctl),
                 ('always-on      - controller', ao, ctl)]:
    d = L(a, full) - L(b, full)
    bs = np.array([L(a, i) - L(b, i) for i in (rng.integers(0, n, n) for _ in range(4000))])
    print('  %s  %+.3f dB  [%+.3f, %+.3f]' % (nm, d, np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
print('COV_DONE')
