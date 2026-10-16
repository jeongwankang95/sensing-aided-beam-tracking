"""missing_readings_decimation.py LOCKED ROOT SLACK --cache=CACHE: missing (NaN) power readings under decimation
(analysis/decimation.py's code up to its print loop, unchanged): rescoring (b) drops every retained frame that contains a
missing reading (the same frames for both policies); policies still act on every frame.
Paper: supplementary Sec. S4-A (decision-frame scoring of Table IV's predecessor).
usage: python analysis/missing_readings_decimation.py configs/locked_v3.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz"""
import sys, os, numpy as np
src = open("analysis/decimation.py").read()
src = src[:src.index("print('\\n=== %s decimation")]
exec(src)
RAW = {}


def nanmask(key):
    if key not in RAW:
        RAW[key] = np.isnan(np.loadtxt(os.path.join(root, key.lstrip("./"))))
    return RAW[key]


def score(ps, pol):
    L, AM, K_ = [], [], []
    for P, keys in ps:
        pol.reset(int(P[0].argmax()), float(P[0].max()))
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0:
                L.append(-10 * np.log10(max(float(P[t, a]) / m, 1e-6))); AM.append(bool(nanmask(keys[t]).any())); K_.append(k)
    L, AM = np.array(L), np.array(AM)
    return L.mean(), L[~AM].mean(), int(AM.sum())


for d in (1, 2, 4):
    ps = [(P[::d], k[::d]) for P, k in passes]
    ps = [(P, k) for P, k in ps if len(P) >= 6]
    f0, fb, nf = score(ps, _trk(TSL))
    c0, cb, nc = score(ps, Ctl(**K))
    print("%s d=%d excluded %d | feedback-only %.3f -> %.3f | controller %.3f -> %.3f | gain %+.3f -> %+.3f"
          % (os.path.basename(root), d, nf, f0, fb, c0, cb, f0 - c0, fb - cb))
print("NAN4_DONE")
