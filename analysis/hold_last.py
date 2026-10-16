import os, sys
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")
exec(compile(open(_CORE).read(), _CORE, "exec"))   # same namespace as the original prelude3.py + tail concatenation
"""hold_last.py LOCKED ROOT SLACK --cache=CACHE: the hold-last always-on variant, which serves the vision prediction wherever
there is one and holds the last serving beam elsewhere (class SA of src/replay_core.py with K=inf), and the same policy with a
+-1 probe every K covered frames. The constants and the slack are read but not used. Also the number of scored frames per
test pass and the scored frames before the first prediction of each pass, and for K=inf the loss on the frames with and
without a prediction and the share of the total loss that comes from the frames without one.
Paper: supplementary Sec. S8: predictions cover 85.2% of scored Scenario-9 frames; the first six scored frames of a pass lack
an eight-frame window (about 41 scored frames per Scenario-9 pass; 2-5% uncovered on Scenarios 31-33); the hold-last variant
loses 0.628, 0.480, 1.465 and 2.433 dB on Scenarios 9, 31, 32 and 33; on Scenario 9 it incurs 1.97 dB on the frames without
a prediction, which carry 46% of its total loss and are roughly 15% of the frames.
usage: python analysis/hold_last.py configs/locked_v3.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz"""

# ---- the naive sensor-always family the frontier figure plots ----
# SA serves the prediction wherever it exists, holds elsewhere, and probes the +-1
# neighbourhood every K covered frames. Cost therefore scales with coverage.
def run(pol):
    cs, ks = [], []
    for P, keys in passes:
        pol.reset(int(P[0].argmax()), float(P[0].max()))
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: cs.append(float(P[t, a]) / m); ks.append(k)
    c = np.array(cs)
    return float(np.mean(-10*np.log10(np.clip(c, 1e-6, None)))), float(np.mean(ks))
cov = 100.0*np.mean([k in PRED['vision'] for P, ks in passes for k in ks[1:]])
print('\n  cache %-28s coverage %.1f%%' % (os.path.basename(CACHE), cov))
print('    %-22s %9s %9s' % ('policy', 'loss dB', 'probes/f'))
for K_ in (10**9, 3, 2, 1):
    l, k = run(SA(K_))
    print('    %-22s %9.3f %9.2f' % ('sensor-always K=%s' % ('inf' if K_ > 10**8 else K_), l, k))

# ---- hold-last (K=inf): scored frames with and without a prediction ----
pol, L, cov_, lead, nsc = SA(), [], [], [], []
for P, keys in passes:
    pol.reset(int(P[0].argmax()), float(P[0].max())); c_ = []
    for t in range(1, len(P)):
        a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
        if m > 0: L.append(-10*np.log10(max(float(P[t, a]) / m, 1e-6))); c_.append(keys[t] in PRED['vision'])
    cov_ += c_; nsc.append(len(c_)); lead.append(c_.index(True) if True in c_ else len(c_))
L, cov_ = np.array(L), np.array(cov_)
print('    %d test passes, %.1f scored frames per pass on average; scored frames before the first prediction of a pass: '
      'min %d, max %d' % (len(passes), np.mean(nsc), min(lead), max(lead)))
print('    hold-last by frame: with a prediction %.3f dB on %.2f%% of scored frames, without %.3f dB on %.2f%%; '
      'share of total loss without a prediction %.1f%%'
      % (L[cov_].mean(), 100*cov_.mean(), L[~cov_].mean(), 100*(~cov_).mean(), 100*L[~cov_].sum()/L.sum()))
print('NAIVE_DONE')
