import os, sys
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")
exec(compile(open(_CORE).read(), _CORE, "exec"))   # same namespace as the original prelude + tail concatenation
"""decimation_accuracy.py: the vision arm's current-slot top-1 accuracy and coverage on the Scenario-9 test passes when decisions
are taken every d native frames (d = 1, 2, 4), with consecutive sensor inputs (the cache of eight consecutive native frames)
and with d-spaced inputs (eight frames d epochs apart; results/predictions/dec2_s9.npz, dec4_s9.npz).
Paper: Sec. VI-C (accuracy falls from 44.7% to 0.9% at d=4 with d-spaced inputs) and supplementary Table S2 (accuracy
44.7 / 18.8 / 0.9%, coverage 85.2 / 70.1 / 38.7% for d-spaced inputs). Only the test passes of the shared loader are used.
usage: python analysis/decimation_accuracy.py configs/locked_v3.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz
           --consec=results/predictions/both_scenario9.npz --d2=results/predictions/dec2_s9.npz --d4=results/predictions/dec4_s9.npz"""


# ---- sensing-arm accuracy and coverage under decimation, both constructions ----
import numpy as _np
CACHES = {}
for a in sys.argv:
    if a.startswith('--consec='): CACHES['consec'] = a.split('=', 1)[1]
    if a.startswith('--d2='):     CACHES['d2'] = a.split('=', 1)[1]
    if a.startswith('--d4='):     CACHES['d4'] = a.split('=', 1)[1]

def load(path):
    c = _np.load(path, allow_pickle=True)
    return dict(zip(c['anchor'], c['vision']))

def acc_cov(pred_map, d):
    hit = tot = scored = 0
    for P, keys in passes:
        Pd, kd = P[::d], keys[::d]
        if len(Pd) < 6: continue
        for t in range(1, len(Pd)):
            m = float(Pd[t].max())
            if m <= 0: continue
            scored += 1
            pr = pred_map.get(kd[t])
            if pr is None or int(pr) < 0: continue
            tot += 1
            if int(pr) == int(Pd[t].argmax()): hit += 1
    return (100 * hit / tot if tot else float('nan'), 100 * tot / scored if scored else 0.0)

con = load(CACHES['consec'])
print('\n=== %s: arm accuracy and coverage under decimation ===' % root)
print('%-34s %8s %8s %8s' % ('', 'd=1', 'd=2', 'd=4'))
row = [acc_cov(con, d) for d in (1, 2, 4)]
print('%-34s %7.1f%% %7.1f%% %7.1f%%' % ('Accuracy, consecutive frames', row[0][0], row[1][0], row[2][0]))
sp = [row[0][0]]
for tag, d in (('d2', 2), ('d4', 4)):
    sp.append(acc_cov(load(CACHES[tag]), d)[0])
print('%-34s %7.1f%% %7.1f%% %7.1f%%' % ('Accuracy, d-spaced frames', sp[0], sp[1], sp[2]))
print('%-34s %7.1f%% %7.1f%% %7.1f%%' % ('Frames carrying a prediction', row[0][1], row[1][1], row[2][1]))
print('DECACC_DONE')
print("\ncoverage per construction:")
for tag,d in (("consec",1),("d2",2),("d4",4)):
    a,c = acc_cov(load(CACHES[tag]), d)
    print("  %-10s d=%d  acc %5.1f%%  coverage %5.1f%%" % (tag,d,a,c))
