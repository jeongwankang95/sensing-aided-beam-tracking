"""missing_readings_table2.py LOCKED ROOT SLACK --cache=CACHE: missing (NaN) power readings in the replay of every Table II
row (analysis/table2.py's code up to its print loop, unchanged). Policies act on every frame as before; rescoring (b) drops
every frame that contains a missing reading from the scored set (the same frames for all policies), and (a) drops only frames
whose served beam is missing. Gains use table2.py's boot().
Paper: supplementary Sec. S4-A and Table S5.
usage: python analysis/missing_readings_table2.py configs/locked_v3.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz"""
import sys, os, numpy as np
src = open("analysis/table2.py").read()
src = src[:src.index("PP, OUT = {}")]
exec(src)
NM = [np.stack([np.isnan(np.loadtxt(os.path.join(root, k.lstrip("./")))) for k in keys]) for P, keys in passes]


def pp3(pol):
    out = []
    for (P, keys), M in zip(passes, NM):
        pol.reset(int(P[0].argmax()), float(P[0].max())); L, Kc, sm, am = [], [], [], []
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0:
                L.append(-10 * np.log10(max(float(P[t, a]) / m, 1e-6))); Kc.append(k)
                sm.append(bool(M[t, a])); am.append(bool(M[t].any()))
        out.append((np.array(L), np.array(Kc, float), np.array(sm), np.array(am)))
    return out


def sub(pp, j):
    return [(x[0][~x[j]], x[1][~x[j]]) for x in pp]


PP = {nm: pp3(pol) for nm, pol in ROWS}
nb = sum(int(x[3].sum()) for x in PP["METHOD"])
print("%s: scored %d, frames with a missing reading %d" % (SCN, sum(len(x[0]) for x in PP["METHOD"]), nb))
for nm, _ in ROWS:
    L0 = np.concatenate([x[0] for x in PP[nm]])
    La = np.concatenate([x[0] for x in sub(PP[nm], 2)])
    Lb = np.concatenate([x[0] for x in sub(PP[nm], 3)])
    K0 = np.concatenate([x[1] for x in PP[nm]]); Kb = np.concatenate([x[1] for x in sub(PP[nm], 3)])
    print("  %-34s served-missing %2d | %.3f | (a) %.3f | (b) %.3f | probes %.2f -> (b) %.2f"
          % (nm, sum(int(x[2].sum()) for x in PP[nm]), L0.mean(), La.mean(), Lb.mean(), K0.mean(), Kb.mean()))
print("  value = reference minus METHOD: current | (b)")
for nm, _ in ROWS:
    if nm == "METHOD" or nm.startswith("plain tracker"):
        continue
    c = boot([(x[0],) for x in PP[nm]], [(x[0],) for x in PP["METHOD"]])
    b = boot(sub(PP[nm], 3), sub(PP["METHOD"], 3))
    print("  %-34s %+.3f [%+.3f, %+.3f] | %+.3f [%+.3f, %+.3f]" % ((nm,) + c + b))
print("NAN3_DONE", SCN)
