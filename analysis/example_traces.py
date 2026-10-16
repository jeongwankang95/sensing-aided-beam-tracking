#!/usr/bin/env python3
"""example_traces.py LOCKED ROOT SLACK --cache=CACHE OUT.json
Per-frame traces of the controller, the plain feedback-only tracker and the controller with its sensor disabled on every test
pass of one scenario: the data behind Fig. 5 (figures/fig5_example.py) and the probe-saving interval
(analysis/probe_saving.py). The data loading and the Ctl class are executed verbatim from src/replay_core.py (everything
before its run() definition), so the replay is the one behind Tables II and III. Writes every pass's per-frame record plus
summary checks (frame-weighted means must match Table II).
usage: python analysis/example_traces.py configs/locked_v3.json dataset/scenario9 0.4
           --cache=results/predictions/both_scenario9.npz results/analysis/trace_scenario9.json
       (Scenario 31 with slack 0.6 for panel (b) of Fig. 5)"""
import sys, json, numpy as np
locked, root, slack = sys.argv[1], sys.argv[2], float(sys.argv[3])
cache = next(a for a in sys.argv if a.startswith('--cache='))
out = sys.argv[-1]
src = open('src/replay_core.py').read()
src = src[:src.index('def run(pol, pp=False):')]
ns = {'__name__': 'attrib3_head'}
sys.argv = ['attrib3.py', locked, root, str(slack), cache]
exec(compile(src, 'attrib3_head', 'exec'), ns)
Ctl, passes, K = ns['Ctl'], ns['passes'], ns['K']

def trace(pol, P, keys):
    pol.reset(int(P[0].argmax()), float(P[0].max()))
    rec = []
    for t in range(1, len(P)):
        m0 = pol.mode if hasattr(pol, 'mode') else 'track'
        tr0, se0 = getattr(pol, 'trial', 0), getattr(pol, 'sense', 0)
        rs_before = getattr(pol, 'rs_t', 0)
        a, k = pol.step(P[t], keys[t])
        mx = float(P[t].max())
        rec.append(dict(t=t, best=int(P[t].argmax()), a=int(a), k=int(k),
                        loss=float(-10*np.log10(max(float(P[t, a])/mx, 1e-6))) if mx > 0 else None,
                        mode=getattr(pol, 'mode', 'track'), mode_before=m0,
                        model=int((getattr(pol, 'trial', 0) - tr0) + (getattr(pol, 'sense', 0) - se0) > 0),
                        resweep=int(getattr(pol, 'rs_k', None) is not None and getattr(pol, 'rs_t', 0) == pol.rs_k and rs_before != pol.rs_t),
                        bl=int(getattr(pol, 'bl', 0))))
    return rec

res = []
for i, (P, keys) in enumerate(passes):
    c = trace(Ctl(**K), P, keys)
    f = trace(Ctl(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=slack), P, keys)
    o = trace(Ctl(**dict(K, sensor=None)), P, keys)      # 're-sweep only' cell of factorial.py: sensor disabled
    lc = [r['loss'] for r in c if r['loss'] is not None]; lf = [r['loss'] for r in f if r['loss'] is not None]
    lo = [r['loss'] for r in o if r['loss'] is not None]
    res.append(dict(idx=i, n=len(lc), ctrl_mean=float(np.mean(lc)), fb_mean=float(np.mean(lf)), off_mean=float(np.mean(lo)), off=o,
                    setup_best=int(P[0].argmax()), setup_peak_to_median_db=float(10*np.log10(P[0].max()/max(np.median(P[0]), 1e-12))),
                    ctrl=c, fb=f))
allc = np.concatenate([[r['loss'] for r in x['ctrl'] if r['loss'] is not None] for x in res])
allf = np.concatenate([[r['loss'] for r in x['fb'] if r['loss'] is not None] for x in res])
allo = np.concatenate([[r['loss'] for r in x['off'] if r['loss'] is not None] for x in res])
summary = dict(root=root, slack=slack, cache=cache, n_passes=len(res), frames=int(len(allc)),
               ctrl_frame_mean=float(allc.mean()), fb_frame_mean=float(allf.mean()), off_frame_mean=float(allo.mean()),
               per_pass_vsense=[round(x['off_mean']-x['ctrl_mean'], 4) for x in res],
               per_pass_gain=[round(x['fb_mean']-x['ctrl_mean'], 4) for x in res])
json.dump(dict(summary=summary, passes=res), open(out, 'w'))
print(json.dumps(summary)[:900])
