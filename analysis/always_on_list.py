import os, sys
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")
exec(compile(open(_CORE).read(), _CORE, "exec"))   # same namespace as the original prelude3.py + tail concatenation
"""always_on_list.py LOCKED ROOT SLACK --cache=CACHE: always-on loss minus controller loss on the test passes, with a paired
bootstrap over passes (10^5 resamples, seed 42). Always-on serves the vision prediction on every frame that has one and tracks
link feedback elsewhere (no re-sweep, slack SLACK); the controller runs the constants of LOCKED.
Paper: supplementary Sec. S10, with the earlier constant set on Scenario 9: -0.023 dB [-0.085, +0.048] with the anchor list
before the correction (results/predictions/sensor_preds_t0_scenario9.npz) and +0.089 dB [+0.006, +0.186] with the corrected
list (results/predictions/both_scenario9.npz).
usage: python analysis/always_on_list.py configs/locked_multi.json dataset/scenario9 0.4 \\
           --cache=results/predictions/sensor_preds_t0_scenario9.npz"""

# ---- always-on minus controller, the comparison the anchor-list section reports ----
def _trk(slack):
    return Ctl(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1,
               bl=0, margin=0, slack=slack)
class SAFB:
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
def perpass(pol):
    out = []
    for P, keys in passes:
        pol.reset(int(P[0].argmax()), float(P[0].max())); c = []
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: c.append(float(P[t, a]) / m)
        out.append(np.array(c))
    return out
def sn(cs):
    L = [-10*np.log10(np.clip(c, 1e-6, None)) for c in cs]
    return np.array([x.sum() for x in L], float), np.array([len(x) for x in L], float)
n = len(passes)
ao, ct = perpass(SAFB(slack=TSL)), perpass(Ctl(**K))
sa, na = sn(ao); sb, nb = sn(ct)
pt = sa.sum()/na.sum() - sb.sum()/nb.sum()
idx = np.random.default_rng(42).integers(0, n, (100000, n))
v = sa[idx].sum(1)/na[idx].sum(1) - sb[idx].sum(1)/nb[idx].sum(1)
print('  cache %-32s always-on %.6f  controller %.6f  diff %+.6f  [%+.6f, %+.6f]'
      % (os.path.basename(CACHE), sa.sum()/na.sum(), sb.sum()/nb.sum(),
         pt, np.percentile(v, 2.5), np.percentile(v, 97.5)))
print('AO_DONE')
