import os, sys
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")
exec(compile(open(_CORE).read(), _CORE, "exec"))   # same namespace as the original prelude3.py + tail concatenation
# ---- datasheet and controller result for each sensing arm ----
def _trk(slack):
    return Ctl(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1,
               bl=0, margin=0, slack=slack)

def run(pol):
    cs, ks, sf = [], [], 0
    for P, keys in passes:
        pol.reset(int(P[0].argmax()), float(P[0].max()))
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: cs.append(float(P[t, a]) / m); ks.append(k)
        sf += getattr(pol, 'sense', 0) + getattr(pol, 'trial', 0)
    c = np.array(cs); L = -10 * np.log10(np.clip(c, 1e-6, None))
    return L.mean(), np.mean(ks), 100 * sf / len(c)

def datasheet(arm):
    """Serve the arm's current-slot prediction wherever it has one: the accuracy and
    the power that costs, on covered frames only."""
    hit = tot = 0; loss = []
    for P, keys in passes:
        for t in range(1, len(P)):
            m = float(P[t].max())
            if m <= 0: continue
            pr = PRED[arm].get(keys[t])
            if pr is None or int(pr) < 0: continue
            tot += 1
            if int(pr) == int(P[t].argmax()): hit += 1
            loss.append(-10 * np.log10(max(float(P[t, int(pr)]) / m, 1e-6)))
    n = sum(1 for P, _ in passes for t in range(1, len(P)) if float(P[t].max()) > 0)
    if not tot: return None
    return 100 * hit / tot, float(np.mean(loss)), float(np.median(loss)), 100 * tot / n

print('\n=== %s (%d test passes) slack %s ===' % (root, len(passes), TSL))
print('%-26s %8s %9s %9s %9s' % ('', 'top-1 %', 'mean dB', 'med dB', 'cover %'))
for arm in ('vision', 'radar'):
    d = datasheet(arm)
    if d is None or d[3] == 0:
        print('%-26s   (no predictions in this cache)' % ('arm: ' + arm)); continue
    print('%-26s %8.1f %9.3f %9.3f %9.1f' % ('arm: ' + arm, d[0], d[1], d[2], d[3]))
print()
fb = run(_trk(TSL))
print('%-26s %8s %8s %8s' % ('controller', 'loss', 'p/f', 'model%'))
print('%-26s %8.3f %8.2f %8.1f' % ('feedback-only', fb[0], fb[1], fb[2]))
for arm in ('vision', 'radar'):
    d = datasheet(arm)
    if d is None or d[3] == 0: continue
    r = run(Ctl(sensor=arm, **K))
    print('%-26s %8.3f %8.2f %8.1f   (VoI %+.3f)'
          % ('need x verify, ' + arm, r[0], r[1], r[2], fb[0] - r[0]))
print('ARMS_DONE')
