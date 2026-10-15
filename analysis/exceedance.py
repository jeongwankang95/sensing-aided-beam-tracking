import os, sys
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")
exec(compile(open(_CORE).read(), _CORE, "exec"))   # same namespace as the original prelude3.py + tail concatenation

# ---- exceedance table: policies of final_eval.py / Table III, extra reporting ----
class Persist:
    """Causal persistence: hold the link-setup beam, never probe."""
    def reset(self, a0, p0): self.a = int(a0); self.sense = self.trial = 0
    def step(self, Pt, key): return self.a, 0

def _trk(slack):
    return Ctl(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1,
               bl=0, margin=0, slack=slack)

class SAFB:
    """Table-III always-on: run the model on every frame that carries a prediction
    and serve its beam; where none exists, fall back on that scenario's own
    validation-selected feedback-only tracker and pay every probe it spends. The
    served beam and its received power are observed without probing (Sec. III-B),
    so the tracker re-anchors on them each frame and is current when it takes over.
    period K probes the sensor's beam +-1 every K frames (None: never)."""
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

THR = [1.0, 2.0, 3.0]

def caps_of(pol):
    cs, ks, sf = [], [], 0
    for P, keys in passes:
        pol.reset(int(P[0].argmax()), float(P[0].max()))
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: cs.append(float(P[t, a]) / m); ks.append(k)
        sf += getattr(pol, 'sense', 0) + getattr(pol, 'trial', 0)
    return np.array(cs), np.array(ks), sf

FB = _trk(TSL)
POLS = [('Causal persistence', Persist()),
        ('Feedback-only tracker', FB),
        ('Need x verify (ours)', Ctl(**K)),
        ('Always-on', SAFB(slack=TSL)),
        ('  + verification', SAFB(period=1, slack=TSL))]
print('\n=== %s (%d test passes) tracker slack %s ===' % (root, len(passes), TSL))
print('%-22s %7s %6s %6s %s %7s' % ('policy', 'mean', 'p/f', 'model%',
      ' '.join('>%.0fdB' % t for t in THR), 'p95'))
for nm, pol in POLS:
    c, k, sf = caps_of(pol); L = -10 * np.log10(np.clip(c, 1e-6, None))
    print('%-22s %7.3f %6.2f %6.1f %s %7.2f' % (nm, L.mean(), k.mean(), 100 * sf / len(c),
          ' '.join('%5.1f' % (100 * (L > t).mean()) for t in THR), np.percentile(L, 95)))
print('EXCEED_DONE')
