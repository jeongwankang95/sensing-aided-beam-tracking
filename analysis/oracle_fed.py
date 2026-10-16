#!/usr/bin/env python3
"""oracle_fed.py ROOT: feedback baselines scored on link metrics, from the recorded 64-beam power vectors of the test windows
(eight observed and six future frames; at most about 800 windows, evenly spaced). "hold (oracle-fb)" keeps the optimal beam of
the window's last observed slot for the whole horizon: persistence initialized from the recorded optimal-beam label in every
window, as the companion study's windowed baselines are; "hold+-K" also probes K neighbours each slot.
Paper: Sec. III-E, oracle-initialized persistence loses 0.17 dB on Scenario 32, 0.25 on 33, 0.30 on 31 and 4.11 on 9 (the
causal rollouts of Table II lose 2.49, 3.07, 1.63 and 6.35 dB).
usage: python analysis/oracle_fed.py dataset/scenario32   (and scenario33, 31, 9)"""
import numpy as np, pandas as pd, os, sys

root = sys.argv[1] if len(sys.argv) > 1 else 'dataset/scenario9'
csv = f'{root}/test_seqs_RA.csv'
te = pd.read_csv(csv)
step = max(1, len(te) // 800)
COLS = ['beam8'] + [f'future_beam{j}' for j in range(1, 7)]
P = np.stack([[np.loadtxt(os.path.join(root, str(te.iloc[i][c]).lstrip('./')))
               for c in COLS] for i in range(0, len(te), step)])   # (N,7,64)
nbad = int(np.isnan(P).sum())
if nbad:
    print(f'  ({nbad} NaN power entries -> replaced with 0)')
    P = np.nan_to_num(P, nan=0.0)
OPT = P.argmax(-1)
Pmax = P.max(-1)
N, H, C = P.shape
print(f'{root}: {N} sequences x {H} slots (step {step})\n')


def report(name, cap, extra=''):
    loss = -10 * np.log10(np.clip(cap, 1e-6, None))
    prof = ' '.join(f'{-10*np.log10(np.clip(cap[:,h].mean(),1e-6,None)):4.2f}' for h in range(H))
    print(f'{name:16s} cap {cap.mean():.3f}  loss {loss.mean():5.2f} dB  '
          f'>=90% {100*(cap>=.9).mean():5.1f}%  >=50% {100*(cap>=.5).mean():5.1f}%'
          f'   per-slot dB [{prof}]{extra}')


b0 = OPT[:, 0]
cap_hold = np.take_along_axis(P, np.repeat(b0[:, None], H, 1)[..., None], -1).squeeze(-1) / (Pmax + 1e-12)
report('hold (oracle-fb)', cap_hold)

for K in (1, 2, 3):
    offs = np.arange(-K, K + 1)
    cand = np.clip(b0[:, None, None] + offs[None, None, :], 0, C - 1)      # (N,1,2K+1)
    cand = np.repeat(cand, H, 1)                                            # (N,H,2K+1)
    probed = np.take_along_axis(P, cand, -1)                                # powers of probed beams
    cap = probed.max(-1) / (Pmax + 1e-12)
    report(f'hold+-{K} ({2*K+1} probes)', cap)

report('oracle', np.ones((N, H)))

d = np.abs(OPT - b0[:, None])
print(f'\nbeam drift from slot-t: per-slot mean [' +
      ' '.join(f'{d[:,h].mean():4.1f}' for h in range(H)) + '] bins')
print('LINK_DONE')
