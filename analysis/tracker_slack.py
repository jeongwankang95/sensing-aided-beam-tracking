import os, sys
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")
exec(compile(open(_CORE).read(), _CORE, "exec"))   # controller (class Ctl) of the shared code; the plain tracker reads no prediction
"""tracker_slack.py LOCKED ROOT SLACK --cache=CACHE: the slack of the plain innovation tracker (the controller with its sensor
disabled, no trial and no re-sweep) on the scenario's validation passes: mean loss and probes per epoch for slack 0.1-3.0, and
the lowest-loss value among {0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0}, the selection rule of the feedback-only reference. The
constants, the slack argument and the cache are read but not used.
Paper: supplementary Sec. S5: the selected slack is 0.4 on Scenario 9 (the grid's lowest value), 0.6 on Scenario 31 and 0.8
on Scenario 32 (Scenario 33 has no validation passes and inherits 0.8); on Scenario 9, slack values of 0.1-0.3 lose 0.63
against 0.48 dB on the validation passes.
usage: python analysis/tracker_slack.py configs/locked_v3.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz"""
def load(split):
    f = f'{root}/{split}_seqs_RA.csv'
    if not os.path.exists(f): return []
    sel = set(pd.read_csv(f)['seq_index'].unique()); out = []
    for p, g in base.groupby('seq_index'):
        if p not in sel: continue
        g = g.sort_values('index')
        P = np.stack([np.nan_to_num(np.loadtxt(os.path.join(root, str(r).lstrip('./'))), nan=0.0)
                      for r in g['unit1_pwr_60ghz']])
        out.append((P, [str(r) for r in g['unit1_pwr_60ghz']]))
    return out
def plain(slack): return Ctl(sensor=None, th_on=9e9, th_off=0, verify=4, t_trial=1, bl=0, margin=0, slack=slack)
def score(ps, pol):
    L, Kc = [], []
    for P, keys in ps:
        pol.reset(int(P[0].argmax()), float(P[0].max()))
        for t in range(1, len(P)):
            a, k = pol.step(P[t], keys[t]); m = float(P[t].max())
            if m > 0: L.append(-10*np.log10(max(float(P[t, a])/m, 1e-6))); Kc.append(k)
    return float(np.mean(L)), float(np.mean(Kc))
GRID, RULE = (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0), (0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0)
VAL = load('val')
print('%s: %d validation passes' % (os.path.basename(root), len(VAL)))
if VAL:
    R = {s: score(VAL, plain(s)) for s in GRID}
    for s in GRID:
        print('  slack %.1f  loss %.4f dB  probes %.3f' % (s, R[s][0], R[s][1]))
    sel = min(RULE, key=lambda s: R[s][0])
    print('  selected among %s: %.1f (loss %.4f dB, %.3f probes/epoch)' % ('/'.join('%g' % s for s in RULE), sel, *R[sel]))
