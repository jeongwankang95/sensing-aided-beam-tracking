import os, sys
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")
exec(compile(open(_CORE).read(), _CORE, "exec"))   # test passes and prediction cache of the shared loader
"""confidence_auc.py LOCKED ROOT SLACK --cache=results/predictions/post_<scenario>.npz: how well the vision arm's own confidence
separates its large errors. On every covered scored frame of the test passes, the softmax maximum and the top-two margin of
the arm's posterior, and two events: predicted beam more than one bin from the best, and arm loss above 1 dB. The AUC is the
probability that a frame with the event has lower confidence than a frame without it (ranks, as the Mann-Whitney statistic).
The constants and the slack are read but not used. This is the statistics part of the script behind the paper, whose
confidence-admission rows (earlier constants) the paper replaced by analysis/competitors.py.
Paper: Sec. IV-F (AUC 0.45-0.65) and supplementary Sec. S17-L: for the event "more than one bin", the AUC of the softmax maximum
is 0.53, 0.64, 0.59 and 0.48 on Scenarios 9, 31, 32 and 33 and that of the margin 0.59, 0.58, 0.51 and 0.45; for "arm loss
above 1 dB" 0.60/0.60/0.61/0.54 and 0.65/0.53/0.50/0.51.
usage: python analysis/confidence_auc.py configs/locked_v3.json dataset/scenario9 0.4 --cache=results/predictions/post_scenario9.npz"""
SCN = os.path.basename(root)
POST = dict(zip(cache['anchor'], cache['vision_post']))
def conf_of(q, kind):
    s = np.sort(q)[::-1]; return float(s[0]) if kind == 'pmax' else float(s[0] - s[1])
print('\n################ %s  model-side confidence (post cache %s)' % (SCN, os.path.basename(CACHE)))
confs = {'pmax': [], 'margin': []}; big = []; l1 = []
for P, keys in passes:
    for t in range(1, len(P)):
        q = POST.get(keys[t]); m = float(P[t].max())
        if q is None or m <= 0: continue
        p = int(q.argmax()); confs['pmax'].append(conf_of(q, 'pmax')); confs['margin'].append(conf_of(q, 'margin'))
        big.append(abs(p - int(P[t].argmax())) > 1); l1.append(-10*np.log10(max(float(P[t, p])/m, 1e-6)) > 1.0)
big = np.array(big); l1 = np.array(l1)
def auc(score, event):  # AUC of (low score) predicting the event = P(score_event < score_ok)
    s = np.asarray(score); e = np.asarray(event, bool)
    if e.all() or (~e).all(): return float('nan')
    r = np.argsort(np.argsort(s)) + 1.0; n1 = e.sum(); n0 = (~e).sum()
    return float((r[~e].sum() - n0*(n0+1)/2) / (n0*n1))
print('covered frames %d; large-error (>1 bin) rate %.1f%%; >1 dB rate %.1f%%' % (len(big), 100*big.mean(), 100*l1.mean()))
for kind in ('pmax', 'margin'):
    print('  AUC(%-6s) for large-error event %.3f, for >1 dB event %.3f' % (kind, auc(confs[kind], big), auc(confs[kind], l1)))
print('CONF_DONE %s' % SCN)
