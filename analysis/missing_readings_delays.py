"""missing_readings_delays.py LOCKED ROOT SLACK --cache=CACHE: missing (NaN) power readings in the replay of the native
condition of analysis/delays_sensor_off.py (its code up to the condition loop, unchanged). A missing reading is zero power, so
serving that beam scores the 60-dB cap. Rescored two ways: (a) without frames whose served beam is missing (policy-dependent),
(b) without every frame that has a missing reading (the same frames for all policies); system gain, marginal value and the
verified always-on comparison with paired intervals.
Paper: supplementary Sec. S4-A and Table S5 (Scenarios 31-33; no gain or marginal value moves by more than 0.02 dB).
usage: python analysis/missing_readings_delays.py configs/locked_v3.json dataset/scenario31 0.6 --cache=results/predictions/both_scenario31.npz"""
import sys, os, numpy as np
src = open("analysis/delays_sensor_off.py").read()
src = src[:src.index("print(\x27\\n################")]
exec(src)
NM = [np.stack([np.isnan(np.loadtxt(os.path.join(root, k.lstrip("./")))) for k in keys]) for P, keys in passes]
def pp(mk):
    cs, ns, an = [], [], []
    for (P, keys), M in zip(passes, NM):
        pol = mk(); pol.reset(int(P[0].argmax()), float(P[0].max())); c, nn, aa = [], [], []
        for t in range(1, len(P)):
            a, kk = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: c.append(float(P[t, a]) / m); nn.append(bool(M[t, a])); aa.append(bool(M[t].any()))
        cs.append(np.array(c)); ns.append(np.array(nn)); an.append(np.array(aa))
    return cs, ns, an
L = lambda c: -10 * np.log10(np.clip(c, 1e-6, None))
def agg(cs, masks):
    s = np.array([L(c)[~m].sum() for c, m in zip(cs, masks)]); n = np.array([(~m).sum() for m in masks], float); return s, n
def paired(A, B, which):
    sa, na = agg(A[0], A[which]); sb, nb = agg(B[0], B[which]); n = len(passes)
    idx = np.random.default_rng(42).integers(0, n, (100000, n)); d = sa[idx].sum(1)/na[idx].sum(1) - sb[idx].sum(1)/nb[idx].sum(1)
    return sa.sum()/na.sum() - sb.sum()/nb.sum(), np.percentile(d, 2.5), np.percentile(d, 97.5)
R = {nm: pp(mk) for nm, mk in POL}
none = lambda cs: [np.zeros(len(c), bool) for c in cs]
print("%s: scored frames %d, frames with any missing reading %d" % (SCN, sum(len(c) for c in R["persistence"][0]), sum(m.sum() for m in R["persistence"][2])))
for nm, _ in POL:
    cs, ns, an = R[nm]; l0 = np.concatenate([L(c) for c in cs])
    la = np.concatenate([L(c)[~m] for c, m in zip(cs, ns)]); lb = np.concatenate([L(c)[~m] for c, m in zip(cs, an)])
    print("  %-18s served-missing=%3d | loss %.3f | (a) %.3f | (b) %.3f" % (nm, sum(m.sum() for m in ns), l0.mean(), la.mean(), lb.mean()))
for lab, A, B in (("G_sys fb-ctl", "feedback-only", "controller"), ("V_sense skel-ctl", "skeleton", "controller"), ("ao+v1 minus ctl", "always-on+verify1", "controller")):
    R0 = {k: (v[0], none(v[0]), none(v[0])) for k, v in R.items()}
    print("  %-17s current %+.3f [%+.3f, %+.3f] | (a) %+.3f [%+.3f, %+.3f] | (b) %+.3f [%+.3f, %+.3f]" % ((lab,) + paired(R0[A], R0[B], 1) + paired(R[A], R[B], 1) + paired(R[A], R[B], 2)))
