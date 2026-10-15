import os, sys
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")
exec(compile(open(_CORE).read(), _CORE, "exec"))   # same namespace as the original prelude3.py + tail concatenation

# ---- Sec. VI-D, run officially ----
# Measurement noise. Every power the policy reads carries an independent lognormal
# error of sigma_m dB, drawn once per (pass, frame, beam) and shared by both policies so
# the comparison is paired on one realization; rewards stay the true powers. A single
# realization is not stable enough to quote, so each figure averages R independent
# realizations, and the interval resamples passes over the realization-averaged losses.
#
# Sensor latency. The prediction served at frame t is the one computed for frame
# t-delta. Deterministic. Two readings of the pass start are reported, since the prose
# does not say which: 'none' leaves the first delta frames without a prediction,
# 'clamp' serves the earliest one there.
R = 20
BOOT = 100000
def _trk(slack):
    return Ctl(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1,
               bl=0, margin=0, slack=slack)

def perpass(pol, obs=None, keymap=None):
    out = []
    for pi, (P, keys) in enumerate(passes):
        Q = obs[pi] if obs is not None else [P[t] for t in range(len(P))]
        ks = keymap(keys) if keymap else keys
        pol.reset(int(np.asarray(Q[0]).argmax()), float(np.asarray(Q[0]).max()))
        c = []
        for t in range(1, len(P)):
            a, k = pol.step(Q[t], ks[t]); m = float(P[t].max())
            if m > 0: c.append(float(P[t, a]) / m)
        out.append(np.array(c))
    return out

def sn(cs):
    L = [-10*np.log10(np.clip(c, 1e-6, None)) for c in cs]
    return np.array([x.sum() for x in L], float), np.array([len(x) for x in L], float)
n = len(passes)

def report(lab, sa, na, sb, nb):
    pt = sa.sum()/na.sum() - sb.sum()/nb.sum()
    idx = np.random.default_rng(42).integers(0, n, (BOOT, n))
    v = sa[idx].sum(1)/na[idx].sum(1) - sb[idx].sum(1)/nb[idx].sum(1)
    lo, hi = np.percentile(v, 2.5), np.percentile(v, 97.5)
    print('  %-22s feedback %6.3f  controller %6.3f  value %+.5f [%+.5f, %+.5f]'
          % (lab, sa.sum()/na.sum(), sb.sum()/nb.sum(), pt, lo, hi))
    return pt, lo, hi

f0, c0 = perpass(Ctl(**dict(K, sensor=None))), perpass(Ctl(**K))
sf0, nf0 = sn(f0); sc0, nc0 = sn(c0)
print('\n=== %s   Sec. VI-D, %d passes, R=%d realizations, B=%d ===' % (root, n, R, BOOT))
report('no noise', sf0, nf0, sc0, nc0)

for sg in (0.5, 1.0, 2.0):
    SF = np.zeros(n); SC = np.zeros(n); NF = None
    for rep in range(R):
        ob = []
        for pi, (P, keys) in enumerate(passes):
            rr = np.random.default_rng((7919, rep, pi))
            Q = P * 10.0 ** (rr.normal(0.0, sg, P.shape) / 10.0)
            ob.append([Q[t] for t in range(len(P))])
        a, na_ = sn(perpass(Ctl(**dict(K, sensor=None)), ob)); b, nb_ = sn(perpass(Ctl(**K), ob))
        SF += a / na_; SC += b / nb_; NF = na_
    # per-pass mean loss under noise, then the usual frame weighting
    report('sigma_m = %.1f dB' % sg, (SF/R)*NF, NF, (SC/R)*NF, NF)

print()
for mode in ('none', 'clamp'):
    for dl in (1, 2):
        if mode == 'none':
            km = lambda ks, dl=dl: ['\x00'] * dl + list(ks[:-dl])
        else:
            km = lambda ks, dl=dl: [ks[max(i-dl, 0)] for i in range(len(ks))]
        ct = perpass(Ctl(**K), keymap=km)
        sc, nc = sn(ct)
        report('sensor lag %d (%s)' % (dl, mode), sf0, nf0, sc, nc)
print('VID_DONE (comparator = own tracker + re-sweep, no sensor)')
