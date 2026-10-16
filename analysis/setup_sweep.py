"""setup_sweep.py: the link-setup sweep of the seven Scenario-9 lock-loss test passes (plain-tracker pass-mean loss of at least
2 dB; seq_index 116, 122, 125, 127, 130, 132, 133): the setup argmax and the argmax three epochs later, and the peak-over-median
power of the setup sweep against the rest of each pass.
Paper: Sec. VI-A and supplementary Sec. S17-P (the setup sweep is nearly flat and its argmax is beam 14 in every lock-loss
pass, about a dozen indices from the beam that dominates three epochs later).
usage: python analysis/setup_sweep.py"""
import csv, os, numpy as np
root = 'dataset/scenario9'
rows = list(csv.DictReader(open(f'{root}/scenario9.csv')))
byp = {}
for r in rows: byp.setdefault(r['seq_index'], []).append(r)
LL = ['116', '127', '130', '122', '125', '132', '133']
setup_pm, rest_pm, rest_pm_passmed, b0, b3 = [], [], [], [], []
for p in LL:
    g = sorted(byp[p], key=lambda r: int(r['index']))
    P = np.stack([np.nan_to_num(np.loadtxt(os.path.join(root, r['unit1_pwr_60ghz'].lstrip('./'))), nan=0.0) for r in g])
    pm = 10*np.log10(P.max(1)/np.median(P, 1))
    setup_pm.append(pm[0]); rest_pm += list(pm[1:]); rest_pm_passmed.append(np.median(pm[1:]))
    b0.append(int(P[0].argmax())); b3.append(int(P[3].argmax()))
print('setup argmax', b0, ' argmax at t=3', b3)
print('setup peak/median dB per pass', np.round(setup_pm, 2), 'median %.2f' % np.median(setup_pm))
print('rest-of-pass peak/median: pooled median %.2f, median of pass medians %.2f' % (np.median(rest_pm), np.median(rest_pm_passmed)))
