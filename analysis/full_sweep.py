import os, sys
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")
exec(compile(open(_CORE).read(), _CORE, "exec"))   # same namespace as the original prelude3.py + tail concatenation
"""full_sweep.py LOCKED ROOT SLACK --cache=CACHE: feedback-only anchors beyond the plotted range of Fig. 3: a full 64-beam sweep
every N epochs, a full re-sweep on the epoch after the served power drops more than eta dB below the last sweep's best, and the
two combined, with all 64 evaluations charged. For comparison, causal persistence, the innovation tracker at slack SLACK and
the controller with the constants of LOCKED (the paper's run used the earlier constants, configs/locked_multi.json).
Writes results/analysis/<scenario>_fullsweep.json.
Paper: supplementary Sec. S17-E: on Scenario 9 a sweep every 2, 3, 4, 8 and 16 epochs costs
31.6, 20.9, 15.4, 7.3 and 3.2 probes/epoch for 0.503, 1.415, 2.234, 3.852 and 4.933 dB; alarms at eta = 1, 2 and 4 dB give
1.961, 2.667 and 4.632 dB at 17.8, 14.1 and 5.8; every 8 plus a 2-dB alarm 1.477 dB at 20.1; on Scenarios 31-33 every 2
epochs 0.057, 0.047 and 0.101 dB (31.8-31.9 probes/epoch), every 8 epochs 0.348, 0.200 and 0.301 dB, the 1-dB alarm 1.057,
1.282 and 1.228 dB.
usage: python analysis/full_sweep.py configs/locked_multi.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz"""

# ---- feedback-only anchors for Fig. 3: periodic full 64-beam sweep every N epochs, alarm-
# triggered full re-sweep (sweep on the epoch after the served power drops > eta dB below the
# last sweep's best), and their combination. All 64 candidate evaluations are charged. ----
import json as _json
SCN = os.path.basename(root); OUT = os.path.join('results', 'analysis'); os.makedirs(OUT, exist_ok=True)
class FullSweep:
    def __init__(self, N=None, alarm=None): self.N, self.alarm = N, alarm
    def reset(self, a0, p0): self.a, self.p_acq, self.t, self.fire, self.sweeps = int(a0), float(p0), 0, False, 0
    def step(self, Pt, key):
        self.t += 1; k = 0
        if (self.N is not None and self.t % self.N == 0) or self.fire:
            self.a = int(Pt.argmax()); self.p_acq = float(Pt[self.a]); k = C; self.fire = False; self.sweeps += 1
        rss = float(Pt[self.a])
        if self.alarm is not None and self.p_acq > 0 and 10*np.log10(max(rss,1e-12)/self.p_acq) < -self.alarm: self.fire = True
        return self.a, k
def _trk(slack): return Ctl(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=slack)
def runp(pol):
    caps, cost = [], []
    for P, keys in passes:
        pol.reset(int(P[0].argmax()), float(P[0].max()))
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: caps.append(float(P[t, a]) / m); cost.append(k)
    caps = np.array(caps); loss = -10*np.log10(np.clip(caps, 1e-6, None))
    return dict(loss=float(loss.mean()), probes=float(np.mean(cost)), p95=float(np.percentile(loss, 95)), ex3=100*float((loss > 3).mean()))
print('\n################ %s full-sweep feedback-only anchors' % SCN)
rows = [('causal persistence (SELF-CHECK)', FullSweep()), ('innovation tracker, deployed slack', _trk(TSL)), ('controller', Ctl(**K))]
rows += [('periodic full sweep every %d' % N, FullSweep(N=N)) for N in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)]
rows += [('alarm-triggered full sweep, eta=%g dB' % e, FullSweep(alarm=e)) for e in (1, 2, 3, 4, 6)]
rows += [('periodic %d + alarm 2 dB' % N, FullSweep(N=N, alarm=2)) for N in (8, 16, 32)]
RES = {'scenario': SCN, 'rows': {}}
print('%-40s %8s %8s %7s %8s' % ('policy', 'loss dB', 'probes', 'p95', '>3dB %'))
for nm, pol in rows:
    r = runp(pol); print('%-40s %8.3f %8.2f %7.2f %8.1f' % (nm, r['loss'], r['probes'], r['p95'], r['ex3'])); RES['rows'][nm] = r
_json.dump(RES, open(os.path.join(OUT, SCN + '_fullsweep.json'), 'w'), default=float)
print('REV5_DONE %s' % SCN)
