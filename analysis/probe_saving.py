"""probe_saving.py TRACE_JSON: paired pass-level bootstrap (10^5 resamples, seed 20261013) of the Scenario-9 probe saving and
of the marginal value of sensing, controller vs the same controller with its sensor disabled, from the per-frame traces
written by analysis/example_traces.py (whose frame-weighted means reproduce Table II).
Paper: Sec. VI-A and VII, probe saving 21% [13%, 29%] (0.87 probes/epoch) and the one-sided 95% bound of -0.05 dB on the loss
sensing adds; probes saved per model call 0.872/0.330 = 2.64 (the 0.330 model calls per epoch are in Table II).
usage: python analysis/probe_saving.py results/analysis/trace_scenario9.json"""
import json, sys, numpy as np
P = json.load(open(sys.argv[1]))['passes']
k_on = np.array([sum(r['k'] for r in p['ctrl']) for p in P], float); k_off = np.array([sum(r['k'] for r in p['off']) for p in P], float)
l_on = np.array([sum(r['loss'] for r in p['ctrl']) for p in P]); l_off = np.array([sum(r['loss'] for r in p['off']) for p in P])
n = np.array([len(p['ctrl']) for p in P], float)
rng = np.random.default_rng(20261013); B = 100000; idx = rng.integers(0, len(P), (B, len(P))); N = n[idx].sum(1)
ps = (k_off[idx].sum(1) - k_on[idx].sum(1)) / N; pf = ps / (k_off[idx].sum(1) / N); vs = (l_off[idx].sum(1) - l_on[idx].sum(1)) / N
e_ps = (k_off.sum() - k_on.sum()) / n.sum(); e_pf = e_ps / (k_off.sum() / n.sum())
print('probe saving %.3f [%.3f, %.3f] probes/epoch; %.1f%% [%.1f, %.1f]' % (e_ps, *np.percentile(ps, [2.5, 97.5]), 100 * e_pf, *(100 * np.percentile(pf, [2.5, 97.5]))))
print('V_sense %.3f [%.3f, %.3f]; one-sided 95%% bound on added loss %.3f dB' % ((l_off.sum() - l_on.sum()) / n.sum(), *np.percentile(vs, [2.5, 97.5]), -np.percentile(vs, 5)))
