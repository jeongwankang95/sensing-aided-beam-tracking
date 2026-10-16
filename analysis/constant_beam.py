import os, sys
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")
exec(compile(open(_CORE).read(), _CORE, "exec"))   # same namespace as the original prelude3.py + tail concatenation
"""constant_beam.py LOCKED ROOT SLACK --cache=CACHE: for the vision and the radar arm of the prediction cache, the number of
distinct predicted beams over the test anchors and the share of the most frequent one, and the arm's top-1 and loss on covered
frames as replayed against those of always predicting that most frequent beam. The constants and the slack are read but not
used.
Paper: supplementary Sec. S12: on Scenarios 32 and 33 the radar arm predicts beam 16 on all 798 and 3,855
anchors, identical to a constant-beam baseline; the vision arm has 12 and 14 distinct indices with modal shares of 30.5% and
26.4%; radar has 29 and 19 distinct indices on Scenarios 9 and 31. With the seed-1 cache (_s1), the Scenario-33 vision arm
predicts one beam on 94% of anchors (caption of Table S15).
usage: python analysis/constant_beam.py configs/locked_v3.json dataset/scenario32 0.8 --cache=results/predictions/both_scenario32.npz"""

# ---- is the radar arm on this scenario a constant-beam emitter? ----
def datasheet(get):
    hit = tot = 0; loss = []
    for P, keys in passes:
        for t in range(1, len(P)):
            m = float(P[t].max())
            if m <= 0: continue
            pr = get(keys[t])
            if pr is None or int(pr) < 0: continue
            tot += 1
            if int(pr) == int(P[t].argmax()): hit += 1
            loss.append(-10*np.log10(max(float(P[t, int(pr)])/m, 1e-6)))
    return (100*hit/tot, float(np.mean(loss))) if tot else None

for arm in ('vision', 'radar'):
    v = np.array(list(PRED[arm].values()))
    u, ct = np.unique(v, return_counts=True)
    top = u[ct.argmax()]; frac = 100*ct.max()/len(v)
    print('\n  %s arm on %s: %d distinct beams over %d anchors; '
          'modal beam %d on %.1f%% of them' % (arm, os.path.basename(root), len(u), len(v), top, frac))
    d = datasheet(lambda k: PRED[arm].get(k))
    dc = datasheet(lambda k: int(top) if k in PRED[arm] else None)
    print('    arm as replayed      top-1 %5.1f%%  loss %6.3f dB' % d)
    print('    constant beam %-2d     top-1 %5.1f%%  loss %6.3f dB   %s'
          % (top, dc[0], dc[1], 'IDENTICAL' if (abs(d[0]-dc[0])<1e-9 and abs(d[1]-dc[1])<1e-9)
             else 'differs'))
print('CONST_DONE')
