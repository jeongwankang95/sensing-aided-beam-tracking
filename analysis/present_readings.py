"""present_readings.py: how far below the best beam of its frame a present (non-missing) power reading can lie, in every
recorded frame of Scenarios 9, 31, 32 and 33 and in the frames of their test passes, and the number of present readings equal
to zero. A capture ratio is floored at 1e-6 (a 60-dB loss cap), so if every present reading is within 60 dB of its frame's best
the cap can only be reached by serving a beam whose reading is missing (read as zero power).
Paper: supplementary Sec. S4-A, "Readings that are present lie at most 21 dB below the best beam of their frame".
usage: python analysis/present_readings.py"""
import glob
import os

import numpy as np
import pandas as pd

worst = 0.0
for s in (9, 31, 32, 33):
    root = f'dataset/scenario{s}'
    f = f'{root}/scenario{s}.csv'
    base = pd.read_csv(f) if os.path.exists(f) else pd.read_csv(glob.glob(f'{root}/*_dev.csv')[0])
    test = set(pd.read_csv(f'{root}/test_seqs_RA.csv')['seq_index'].unique())
    gap, gap_test, zeros, n, n_test = 0.0, 0.0, 0, 0, 0
    for p, g in base.groupby('seq_index'):
        for r in g.sort_values('index')['unit1_pwr_60ghz']:
            x = np.loadtxt(os.path.join(root, str(r).lstrip('./')))
            x = x[~np.isnan(x)]
            if x.max() <= 0: continue
            zeros += int((x <= 0).sum()); x = x[x > 0]
            d = 10 * np.log10(x.max() / x.min()); n += 1; gap = max(gap, d)
            if p in test: n_test += 1; gap_test = max(gap_test, d)
    worst = max(worst, gap)
    print('scenario%-2d all %4d passes, %5d frames: largest gap %.2f dB | test %4d frames: %.2f dB | present readings equal to zero: %d'
          % (s, base['seq_index'].nunique(), n, gap, n_test, gap_test, zeros))
print('largest gap over all recorded frames: %.2f dB' % worst)
