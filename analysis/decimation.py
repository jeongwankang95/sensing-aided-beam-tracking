import os, sys
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")
exec(compile(open(_CORE).read(), _CORE, "exec"))   # same namespace as the original prelude3.py + tail concatenation

# ---- frame decimation, rebuilt on final_eval.py's locked controller ----
# Same construction as the original decimate.py: take every d-th frame of a pass, so
# the decision step spans d frame periods while the sensing model keeps reading eight
# consecutive frames. The innovation median reported is the FEEDBACK-ONLY tracker's.
FRAME_S = 0.111

def _trk(slack):
    return Ctl(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1,
               bl=0, margin=0, slack=slack)

def run(ps, pol):
    caps, cost, sn, sig = [], [], 0, []
    for P, keys in ps:
        pol.reset(int(P[0].argmax()), float(P[0].max()))
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            sig.append(pol.sig)
            if m > 0: caps.append(float(P[t, a]) / m); cost.append(k)
        sn += getattr(pol, 'sense', 0) + getattr(pol, 'trial', 0)
    c = np.array(caps)
    return (float(np.mean(-10 * np.log10(np.clip(c, 1e-6, None)))), float(np.mean(cost)),
            100 * sn / len(c), float(np.median(sig)))

print('\n=== %s decimation (cache %s) ===' % (root, os.path.basename(CACHE)))
print('%2s %10s %10s | %8s %7s | %9s %7s %6s | %7s'
      % ('d', 'drift', 'innov med', 'fb loss', 'p/step', 'ctl loss', 'p/step', 'inv%', 'VoI'))
for d in (1, 2, 4):
    ps = [(P[::d], k[::d]) for P, k in passes]
    ps = [(P, k) for P, k in ps if len(P) >= 6]
    drift = np.mean([np.abs(np.diff(P.argmax(1))).mean() for P, _ in ps])
    fb = run(ps, _trk(TSL))
    ct = run(ps, Ctl(**K))
    print('%2d %10.2f %10.2f | %8.3f %7.2f | %9.3f %7.2f %6.1f | %+7.3f'
          % (d, drift, fb[3], fb[0], fb[1], ct[0], ct[1], ct[2], fb[0] - ct[0]))
print('DECIM_DONE')
