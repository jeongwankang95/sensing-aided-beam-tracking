"""hull_crossing.py: where the lower convex hull of the Scenario-9 feedback-only policies reaches the controller's loss, for three
policy sets: without the lock-loss-recovery and re-sweep settings, with the recovery settings, and with all 205 policies (the
frontier of Fig. 3, as figures/fig3_frontier.py). For each, the crossing in probes/epoch, its ratio to the controller's
budget, and the hull segment it lies on. Then the setup-inclusive ratio: the one full 64-beam sweep of each test pass,
amortized over the scored frames, added to both budgets.
Inputs as figures/fig3_frontier.py: results/analysis/policy_family_s9.json (analysis/policy_family.py), configs/llr_scenario9.json,
configs/resweep_scenario9.json and results/analysis/t2_scenario9.json (controller, passes and scored frames).
Paper: supplementary Sec. S6-A (9.91 probes/epoch, 3.05 times the controller's 3.25, on the segment from the tracker at
1.045 dB and 3.64 to the Bayesian filter at 0.164 dB and 10.38; with the recovery settings 9.55 and 2.94, between a coarse-sweep
recovery tracker at 0.487 dB and 5.98 and the same filter; with the re-sweep settings 4.23 and 1.30, between re-sweeping trackers
at 0.274 dB and 4.06 and 0.176 dB and 4.40) and Sec. S6-B (64 x 28 / 1,138 = 1.57 probes/epoch; 3.25 to 4.82 and 4.23 to 5.81;
the ratio from 1.30 to 1.20); main paper Sec. VI-A (4.23 probes/epoch, 1.30 times, 1.20 times after amortizing the setup sweep).
usage: python analysis/hull_crossing.py"""
import json
import os

A = os.path.join('results', 'analysis')
F = json.load(open(os.path.join(A, 'policy_family_s9.json')))
T2 = json.load(open(os.path.join(A, 't2_scenario9.json')))
LLR = json.load(open(os.path.join('configs', 'llr_scenario9.json')))['test_grid']
RS = json.load(open(os.path.join('configs', 'resweep_scenario9.json')))['test_grid']

fam = lambda name: [(d['probes'], d['loss']) for d in F[name]]
base = fam('periodic') + fam('fixed cadence') + fam('extrapolation') + fam('tracker') + sorted({(d['probes'], d['loss']) for d in F['bayes_hmm']})
llr = [(s[1], s[0]) for g, s in LLR]                      # (probes, loss)
rs = [(e[1][1], e[1][0]) for e in RS]
ctl = T2['rows']['METHOD']


def hull(P):
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    H = []
    for p in sorted(set(P)):
        while len(H) >= 2 and cross(H[-2], H[-1], p) <= 0: H.pop()
        H.append(p)
    i = min(range(len(H)), key=lambda i: H[i][1]); return H[:i + 1]


def crossing(P, y):
    H = hull(P)
    for (x0, y0), (x1, y1) in zip(H, H[1:]):
        if (y0 - y) * (y1 - y) <= 0:
            return x0 + (y - y0) * (x1 - x0) / (y1 - y0), (x0, y0), (x1, y1)


print('controller %.4f dB at %.4f probes/epoch' % (ctl['loss'], ctl['probes']))
res = {}
for name, P in (('without recovery and re-sweep settings', base), ('with the lock-loss-recovery settings', base + llr),
                ('with the recovery and re-sweep settings (all)', base + llr + rs)):
    x, a, b = crossing(P, ctl['loss']); res[name] = x
    print('%-46s %3d policies: crossing %.4f probes/epoch, ratio %.4f, on the segment %.3f dB at %.2f -> %.3f dB at %.2f'
          % (name, len(P), x, x / ctl['probes'], a[1], a[0], b[1], b[0]))
npass, nfr = len(T2['perpass']['METHOD']), ctl['frames']
setup = 64 * npass / nfr; xa = res['with the recovery and re-sweep settings (all)']
print('setup sweep: 64 x %d passes / %d scored frames = %.4f probes/epoch; controller %.4f -> %.4f, crossing %.4f -> %.4f, '
      'ratio %.4f -> %.4f' % (npass, nfr, setup, ctl['probes'], ctl['probes'] + setup, xa, xa + setup,
                              xa / ctl['probes'], (xa + setup) / (ctl['probes'] + setup)))
