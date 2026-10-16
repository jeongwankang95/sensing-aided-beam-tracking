#!/usr/bin/env python3
"""policy_family.py LOCKED ROOT SLACK --cache=CACHE: the feedback-only policies of Fig. 3 on the 28 Scenario-9 test passes,
written with full precision to results/analysis/policy_family_s9.json for figures/fig3_frontier.py.
- fixed cadence: probe +-K around the current beam every epoch (K = 1, 2, 3); extrapolation: probe +-K around the beam
  extrapolated from the last move (K = 1, 2; one-step history, m = 1). Classes and scoring verbatim from the replay harness of
  the paper's first frontier (exp_replay.py), on the test passes.
- periodic: probe +-W around the current beam every N epochs and hold otherwise (W = 2, 3, 5; N = 2, 3, 4, 6, 8, 12). The class
  behind the first figure was not kept; this definition reproduces its 18 points within 0.002 dB and 0.001 probes/epoch.
- innovation tracker: the plain feedback-only tracker of Table II (src/replay_core.py) at slacks 0.4, 0.6, 1.0, 1.5, 2.0, 3.0
  and 5.0 (0.4-2.0 are also Table II's "plain tracker slack" rows).
- Bayes/HMM: Bayesian and Viterbi trackers (src/strong_fb.py) with parameters selected on the validation passes at each
  probing budget and evaluated once on test (the analysis' frontier9.py; the same rows, rounded, are in frontier_s9.log).
- Sec. IV-B baselines, printed last: causal persistence and naive triggering (a +-2 search after a 1-dB drop; 6.35 and
  6.27 dB, 0.55 probes/epoch, not plotted), and the share of frames with at least 90% of the available power captured
  (69.5% for the tracker against 79.0% for the +-3 cadence).
usage: python analysis/policy_family.py configs/locked_v3.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz"""
import contextlib, importlib.util, io, json, os, sys, glob
import numpy as np, pandas as pd

ARGV = list(sys.argv)
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'replay_core.py')
CORE = {'__name__': 'replay_core'}
exec(compile(open(_CORE).read(), _CORE, 'exec'), CORE)      # test passes, Ctl and run() of Table II
root = CORE['root']
C = 64

# ---------------- exp_replay.py classes and scoring (verbatim), on the test passes it would load with the same filter
passes = [P for P, keys in CORE['passes'] if len(P) >= 10 and P.max() > 0]

class Policy:
    def reset(self, a0, p0):
        self.a, self.ref = int(a0), float(p0)
    def probes(self, t):            # beams to probe this frame
        return []
    def choose(self, observed):     # observed: {beam: power}; returns transmit beam
        if observed:
            self.a = max(observed, key=observed.get)
        return self.a
    def feedback(self, rss):        # RSS of the transmitted beam
        pass


class Probe(Policy):
    def __init__(self, K): self.K = K
    def probes(self, t):
        return list(range(max(0, self.a - self.K), min(C, self.a + self.K + 1)))


class Drift(Policy):
    def __init__(self, K, m=4): self.K, self.m = K, m
    def reset(self, a0, p0):
        super().reset(a0, p0); self.hist = [int(a0)]
    def probes(self, t):
        v = 0
        if len(self.hist) >= 2:
            k = min(self.m, len(self.hist) - 1)
            v = int(round((self.hist[-1] - self.hist[-1 - k]) / k))
        c = int(np.clip(self.a + v, 0, C - 1))
        return list(range(max(0, c - self.K), min(C, c + self.K + 1)))
    def choose(self, observed):
        a = super().choose(observed); self.hist.append(a); return a


class Periodic(Policy):
    """probe +-W around the current beam every N epochs (t mod N = 0), hold otherwise (reconstructed; see the docstring)"""
    def __init__(self, W, N, phase=0): self.W, self.N, self.phase = W, N, phase
    def probes(self, t):
        if (t - self.phase) % self.N != 0: return []
        return list(range(max(0, self.a - self.W), min(C, self.a + self.W + 1)))


def run(policy):
    caps, costs = [], []
    for P in passes:
        policy.reset(int(P[0].argmax()), float(P[0].max()))
        for t in range(1, len(P)):
            S = [b for b in policy.probes(t) if 0 <= b < C]
            a = int(np.clip(policy.choose({b: float(P[t, b]) for b in S}), 0, C - 1))
            rss = float(P[t, a]); policy.feedback(rss)
            m = float(P[t].max())
            if m > 0:
                caps.append(rss / m); costs.append(len(S))
    caps, costs = np.array(caps), np.array(costs)
    loss = -10 * np.log10(np.clip(caps, 1e-6, None))
    return dict(loss=loss.mean(), p5=np.percentile(caps, 5), ge90=100 * (caps >= .9).mean(),
                probes=costs.mean())


OUT = {'passes': len(passes), 'periodic': [], 'fixed cadence': [], 'extrapolation': [], 'tracker': [], 'bayes_hmm': []}
for W in (2, 3, 5):
    for N in (2, 3, 4, 6, 8, 12):
        r = run(Periodic(W, N)); OUT['periodic'].append(dict(W=W, N=N, loss=float(r['loss']), probes=float(r['probes'])))
for K in (1, 2, 3):
    r = run(Probe(K)); OUT['fixed cadence'].append(dict(K=K, loss=float(r['loss']), probes=float(r['probes'])))
for K in (1, 2):
    r = run(Drift(K, m=1)); OUT['extrapolation'].append(dict(K=K, m=1, loss=float(r['loss']), probes=float(r['probes'])))
for sl in (0.4, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0):
    r = CORE['run'](CORE['Ctl'](sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=sl))
    OUT['tracker'].append(dict(slack=sl, loss=r[0], probes=r[1]))
print('%s: %d test passes' % (root, len(passes)))
for fam in ('periodic', 'fixed cadence', 'extrapolation', 'tracker'):
    for d in OUT[fam]:
        print('  %-13s %-22s %.5f dB @ %.4f probes/epoch' % (fam, ' '.join('%s=%s' % kv for kv in d.items() if kv[0] not in ('loss', 'probes')),
                                                               d['loss'], d['probes']))

# ---------------- Bayes/Viterbi frontier (frontier9.py: parameters chosen on validation at each budget, evaluated once on test)
def load(split):
    if os.path.exists(f'{root}/{os.path.basename(root)}.csv'):
        base = pd.read_csv(f'{root}/{os.path.basename(root)}.csv')
    else:
        base = pd.read_csv(glob.glob(f'{root}/*_dev.csv')[0])
    sel = set(pd.read_csv(f'{root}/{split}_seqs_RA.csv')['seq_index'].unique())
    ps = []
    for p, g in base.groupby('seq_index'):
        if p not in sel: continue
        g = g.sort_values('index')
        ps.append(np.stack([np.nan_to_num(np.loadtxt(os.path.join(root, str(r).lstrip('./'))), nan=0.0)
                            for r in g['unit1_pwr_60ghz']]))
    return ps

spec = importlib.util.spec_from_file_location('sfb', 'src/strong_fb.py')
sys.argv = ['sfb', root, 'val']
with contextlib.redirect_stdout(io.StringIO()):
    sfb = importlib.util.module_from_spec(spec); spec.loader.exec_module(sfb)
sys.argv = ARGV
Bayes, Viterbi = sfb.Bayes, sfb.Viterbi

VAL, TEST = load('val'), load('test')
def run_sfb(ps, pol):
    caps, cost = [], []
    for P in ps:
        pol.reset(int(P[0].argmax()), float(P[0].max()))
        for t in range(1, len(P)):
            a, k = pol.step(P[t]); m = float(P[t].max())
            if m > 0: caps.append(float(P[t,a])/m); cost.append(k)
    caps = np.array(caps)
    return float(np.mean(-10*np.log10(np.clip(caps,1e-6,None)))), float(np.mean(cost))

CAND = []
for sv in (0.6, 1.0, 1.6):
    for eps in (0.02, 0.05, 0.10, 0.20, 0.35, 0.5, 0.65):
        for tau in (1.0, 3.0):
            for km in (5, 9, 13, 17):
                CAND.append((f'bayes', dict(sigma_v=sv, eps=eps, tau=tau, kmax=km)))
for band in (2, 3, 5):
    for es in (0.5, 1.0, 2.0, 3.5):
        for wm in (2, 4, 6, 8):
            CAND.append((f'viterbi', dict(band=band, ent_scale=es, wmax=wm)))
mk = lambda k, kw: Bayes(**kw) if k == 'bayes' else Viterbi(**kw)
vres = [(run_sfb(VAL, mk(k, kw)), k, kw) for k, kw in CAND]

print(f'{root}: Bayes/Viterbi frontier (params selected on val, evaluated once on test)')
print(f"{'budget':>7s} | {'val loss':>8s} {'val p/f':>8s} | {'TEST loss':>9s} {'TEST p/f':>8s}  family")
for BUD in (2.6, 3.6, 5.0, 7.0, 9.0, 11.0):
    el = [v for v in vres if v[0][1] <= BUD]
    if not el: continue
    (vl, vc), k, kw = min(el, key=lambda v: v[0][0])
    tl, tc = run_sfb(TEST, mk(k, kw))
    print(f'{BUD:7.1f} | {vl:8.3f} {vc:8.2f} | {tl:9.3f} {tc:8.2f}  {k} {kw}')
    OUT['bayes_hmm'].append(dict(budget=BUD, family=k, params=kw, val_loss=vl, val_probes=vc, loss=tl, probes=tc))
print('STRONG_FRONTIER_DONE')
json.dump(OUT, open('results/analysis/policy_family_s9.json', 'w'), indent=1)
print('-> results/analysis/policy_family_s9.json')

# ---------------- Sec. IV-B baselines not plotted in Fig. 3 (exp_replay.py classes Hold and Trigger, verbatim), and the share of
# scored frames on which a policy captures at least 90% of the available power
class Hold(Policy):
    pass


class Trigger(Policy):
    def __init__(self, drop_db, W): self.d, self.W = drop_db, W
    def reset(self, a0, p0):
        super().reset(a0, p0); self.fire = False
    def probes(self, t):
        if not self.fire:
            return []
        self.fire = False
        return list(range(max(0, self.a - self.W), min(C, self.a + self.W + 1)))
    def choose(self, observed):
        if observed:
            self.a = max(observed, key=observed.get)
            self.ref = max(observed.values())
        return self.a
    def feedback(self, rss):
        if self.ref > 0 and 10 * np.log10(max(rss, 1e-12) / self.ref) < -self.d:
            self.fire = True


print('Sec. IV-B baselines on the same passes:')
for name, pol in (('causal persistence (hold)', Hold()), ('naive triggering, 1 dB, +-2', Trigger(1, 2)),
                  ('fixed cadence K=1', Probe(1)), ('fixed cadence K=3', Probe(3))):
    r = run(pol)
    print('  %-28s %.5f dB @ %.4f probes/epoch; capture >= 90%% on %.2f%% of scored frames' % (name, r['loss'], r['probes'], r['ge90']))
pol, caps = CORE['Ctl'](sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=0.4), []
for P, keys in CORE['passes']:
    pol.reset(int(P[0].argmax()), float(P[0].max()))
    for t in range(1, len(P)):
        a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
        if m > 0: caps.append(float(P[t, a]) / m)
caps = np.array(caps)
print('  %-28s capture >= 90%% on %.2f%% of %d scored frames' % ('tracker slack=0.4', 100 * (caps >= .9).mean(), len(caps)))
