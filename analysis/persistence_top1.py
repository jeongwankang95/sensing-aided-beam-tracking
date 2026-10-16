"""persistence_top1.py: top-1 accuracy of oracle-initialized persistence over the seven slots of every Scenario-32 test window
(the last observed slot and six future slots; the optimal beam is the argmax of the measured powers, a missing reading counting
as zero). Paper: Sec. III-E, "a horizon-averaged top-1 accuracy of 43.8%".
usage: python analysis/persistence_top1.py"""
import numpy as np, pandas as pd, os
root = "dataset/scenario32"; te = pd.read_csv(root + "/test_seqs_RA.csv")
COLS = ["beam8"] + ["future_beam%d" % j for j in range(1, 7)]
P = np.nan_to_num(np.stack([[np.loadtxt(os.path.join(root, str(te.iloc[i][c]).lstrip("./"))) for c in COLS] for i in range(len(te))]), nan=0.0)
O = P.argmax(-1); print("seven-slot persistence top-1 %.2f%%" % (100*(O == O[:, [0]]).mean()))
