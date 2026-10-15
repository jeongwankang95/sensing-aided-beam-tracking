import os, sys
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")
exec(compile(open(_CORE).read(), _CORE, "exec"))   # same namespace as the original prelude3.py + tail concatenation
# ---- probe/report delay: every power the POLICY observes is d frames stale ----
# Rewards stay the true powers, exactly as the measurement-noise study of Sec. VI-D
# does. The sensor is NOT delayed here; that is the separate sensor-latency study.
class Persist:
    def reset(self, a0, p0): self.a = int(a0); self.sense = self.trial = 0
    def step(self, Pt, key): return self.a, 0

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

def perpass(pol, delay):
    cs, ks = [], []
    for P, keys in passes:
        pol.reset(int(P[0].argmax()), float(P[0].max()))
        c, k = [], []
        for t in range(1, len(P)):
            a, kk = pol.step(P[max(t - delay, 0)], keys[t])   # observations are stale
            m = float(P[t].max())                              # reward is not
            if m > 0: c.append(float(P[t, a]) / m); k.append(kk)
        cs.append(np.array(c)); ks.append(np.array(k))
    return cs, ks

def mean_loss(cs, idx):
    c = np.concatenate([cs[i] for i in idx])
    return float(np.mean(-10 * np.log10(np.clip(c, 1e-6, None))))

n = len(passes); full = list(range(n)); rng = np.random.default_rng(42)
print('\n=== %s (%d test passes) probe/report delay, slack %s ===' % (root, n, TSL))
print('%-22s %8s %8s %8s' % ('policy', 'd=0', 'd=1', 'd=2'))
store = {}
for nm, mk in [('Causal persistence', lambda: Persist()),
               ('Feedback-only tracker', lambda: _trk(TSL)),
               ('Need x verify (ours)', lambda: Ctl(**K)),
               ('Always-on', lambda: SAFB(slack=TSL)),
               ('  + verification', lambda: SAFB(period=1, slack=TSL))]:
    row = []
    for d in (0, 1, 2):
        cs, ks = perpass(mk(), d); store[(nm, d)] = cs
        row.append('%.3f/%.2f' % (mean_loss(cs, full), np.concatenate(ks).mean()))
    print('%-22s %8s %8s %8s' % (nm, row[0], row[1], row[2]))
print('\nvalue of sensing (feedback - controller), paired pass bootstrap:')
for d in (0, 1, 2):
    fb = store[('Feedback-only tracker', d)]; ct = store[('Need x verify (ours)', d)]
    v = mean_loss(fb, full) - mean_loss(ct, full)
    bs = np.array([mean_loss(fb, i) - mean_loss(ct, i)
                   for i in (rng.integers(0, n, n) for _ in range(4000))])
    print('  d=%d  %+.3f dB  [%+.3f, %+.3f]' % (d, v, np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
print('DELAY_DONE')
