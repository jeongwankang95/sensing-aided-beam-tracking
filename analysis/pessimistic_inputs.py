import os, sys
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")
exec(compile(open(_CORE).read(), _CORE, "exec"))   # same namespace as the original prelude3.py + tail concatenation

# ---- the pessimistic bracket: the same decimated replay, but the sensing arm reads a
#      d-spaced window, so its input is out of distribution and its value is a lower bound.
PESS = {int(a.split('=',1)[0][-1]): a.split('=',1)[1] for a in sys.argv
        if a.startswith('--d2=') or a.startswith('--d4=')}
def _trk(slack):
    return Ctl(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1,
               bl=0, margin=0, slack=slack)
def run(ps, pol):
    caps, cost, sn = [], [], 0
    for P, keys in ps:
        pol.reset(int(P[0].argmax()), float(P[0].max()))
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: caps.append(float(P[t, a]) / m); cost.append(k)
        sn += getattr(pol, 'sense', 0) + getattr(pol, 'trial', 0)
    c = np.array(caps)
    return (float(np.mean(-10*np.log10(np.clip(c, 1e-6, None)))), float(np.mean(cost)),
            100*sn/len(c))
print('\n=== %s   pessimistic bracket (d-spaced sensing input) ===' % root)
print('%2s | %8s %8s %6s | %8s %8s %6s | %8s %8s'
      % ('d', 'fb', 'ctl opt', 'inv%', 'ctl pess', 'p/step', 'inv%', 'VoI opt', 'VoI pess'))
for d in (2, 4):
    ps = [(P[::d], k[::d]) for P, k in passes]
    ps = [(P, k) for P, k in ps if len(P) >= 6]
    fb = run(ps, _trk(TSL))
    co = run(ps, Ctl(**K))
    z = np.load(PESS[d], allow_pickle=True)
    PRED['vision'] = dict(zip(z['anchor'], z['vision']))
    PRED['radar'] = dict(zip(z['anchor'], z['radar']))
    cp = run(ps, Ctl(**K))
    print('%2d | %8.3f %8.3f %6.1f | %8.3f %8.2f %6.1f | %+8.3f %+8.3f'
          % (d, fb[0], co[0], co[2], cp[0], cp[1], cp[2], fb[0]-co[0], fb[0]-cp[0]))
print('PESS_DONE')
