import os, sys
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")
exec(compile(open(_CORE).read(), _CORE, "exec"))   # same namespace as the original prelude3.py + tail concatenation

# ---- need gate without verification, on every scenario ----
# Need-only: the innovation gate opens and the link is handed over at once, with no
# verification trial, no periodic re-verification and no blacklist. Everything else
# is the locked constant set.
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
    c = np.array(cs)
    L = -10 * np.log10(np.clip(c, 1e-6, None))
    return L, np.mean(ks), 100 * sf / len(c)

def perpass(pol):
    out = []
    for P, keys in passes:
        pol.reset(int(P[0].argmax()), float(P[0].max()))
        c = []
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: c.append(float(P[t, a]) / m)
        out.append(np.array(c))
    return out

# A: the trial is removed (the gate hands over at once) but the periodic
#    re-verification of Sec. V is kept, which is how the main paper describes it.
A = dict(K); A.update(margin=-99.0)
# B: no over-the-air check of the sensor at all.
B = dict(K); B.update(margin=-99.0, verify=10**9, bl=0)
rows = [('Feedback-only tracker', _trk(TSL)),
        ('Need x verify (ours)', Ctl(**K)),
        ('Need only (A: no trial)', Ctl(**A)),
        ('Need only (B: no check)', Ctl(**B))]
print('\n=== %s (%d test passes) slack %s ===' % (root, len(passes), TSL))
print('%-22s %7s %6s %7s %7s' % ('policy', 'loss', 'p/f', 'model%', 'p95'))
store = {}
for nm, pol in rows:
    L, k, mp = run(pol)
    print('%-22s %7.3f %6.2f %7.1f %7.2f' % (nm, L.mean(), k, mp, np.percentile(L, 95)))
for nm, pol in rows:
    store[nm] = perpass(pol)
n = len(passes); full = list(range(n)); rng = np.random.default_rng(42)
def Lm(cs, idx):
    c = np.concatenate([cs[i] for i in idx])
    return float(np.mean(-10 * np.log10(np.clip(c, 1e-6, None))))
for a, b in [('Need only (A: no trial)', 'Need x verify (ours)'),
             ('Need only (B: no check)', 'Need x verify (ours)')]:
    d = Lm(store[a], full) - Lm(store[b], full)
    bs = np.array([Lm(store[a], i) - Lm(store[b], i)
                   for i in (rng.integers(0, n, n) for _ in range(4000))])
    print('  %-22s minus %-22s %+.3f [%+.3f, %+.3f]'
          % (a, b, d, np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
print('NEEDONLY_DONE')
