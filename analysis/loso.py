import os, sys
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")
exec(compile(open(_CORE).read(), _CORE, "exec"))   # same namespace as the original prelude3.py + tail concatenation

# ---- Option 3 (2026-10-02): leave-one-site-out selections (same criterion on two validation sites), tested once ----
def _trk(slack):
    return Ctl(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=slack)
def perpass(pol):
    out = []
    for P, keys in passes:
        pol.reset(int(P[0].argmax()), float(P[0].max())); L, Kc = [], []
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: L.append(-10*np.log10(max(float(P[t, a])/m, 1e-6))); Kc.append(k)
        out.append((np.array(L), np.array(Kc, float), getattr(pol, 'sense', 0) + getattr(pol, 'trial', 0)))
    return out
def st(pp):
    L = np.concatenate([x[0] for x in pp]); Kc = np.concatenate([x[1] for x in pp]); return float(L.mean()), float(Kc.mean()), 100*sum(x[2] for x in pp)/len(L)
def boot(ppA, ppB, B=100000, seed=42):
    sA = np.array([x[0].sum() for x in ppA]); sB = np.array([x[0].sum() for x in ppB]); n = np.array([len(x[0]) for x in ppA])
    pt = (sA.sum() - sB.sum())/n.sum(); rng = np.random.default_rng(seed); P = len(n); d = np.empty(B)
    for i in range(B):
        j = rng.integers(0, P, P); d[i] = (sA[j].sum() - sB[j].sum())/n[j].sum()
    return float(pt), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
LO = json.load(open('configs/loso3_val.json')); SCN = os.path.basename(root)
fb = perpass(_trk(TSL)); me = perpass(Ctl(**K)); res = {}
print('\n=== %s: feedback-only %.3f @ %.2f | selected method %.3f @ %.2f (%.1f%%)' % ((SCN,) + st(fb)[:2] + st(me)))
for nm, d in LO.items():
    cfg = dict(d['cfg']); cfg['rs_how'], cfg['rs_k'] = d['resweep']; pp = perpass(Ctl(**cfg)); s = st(pp)
    v = boot(fb, pp); w = boot(me, pp); res[nm] = dict(test=s, value_vs_fb=v, minus_method=w)
    print('  %-18s %.3f @ %.2f (%.1f%%) | value vs fb %+.3f [%+.3f, %+.3f] | method minus this %+.3f [%+.3f, %+.3f]' % ((nm,) + s + v + w))
json.dump(res, open('results/analysis/loso3_%s.json' % SCN, 'w'))
print('LOSO3_DONE', SCN)
