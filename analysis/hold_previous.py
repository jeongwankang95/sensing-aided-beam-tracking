"""hold_previous.py: loss of holding the previous epoch's best beam, pooled over all passes of each scenario, under the replay
scoring rule (a missing reading counts as zero power; capture clipped at 1e-6, i.e. a 60-dB cap), with transitions whose held
beam has no reading excluded, and with every frame that has a missing reading excluded.
Paper: supplementary Table S1 (hold-previous losses) and Sec. S4-A (on Scenarios 32 and 33 they become 0.14 and 0.16 dB
without the transitions whose held beam is missing).
usage: python analysis/hold_previous.py"""
import numpy as np, pandas as pd, os, glob
for scn in ("scenario9", "scenario31", "scenario32", "scenario33"):
    root = f"dataset/{scn}"
    f = f"{root}/{scn}.csv" if os.path.exists(f"{root}/{scn}.csv") else glob.glob(f"{root}/*_dev.csv")[0]
    base = pd.read_csv(f); ordc = "index" if "index" in base.columns else base.columns[0]
    caps, miss, anym = [], [], []
    for p, g in base.groupby("seq_index"):
        g = g.sort_values(ordc)
        raw = np.stack([np.loadtxt(os.path.join(root, str(r).lstrip("./"))) for r in g["unit1_pwr_60ghz"]])
        P = np.nan_to_num(raw, nan=0.0); b = P.argmax(1)
        for t in range(1, len(P)):
            m = P[t].max()
            if m > 0: caps.append(P[t, b[t-1]] / m); miss.append(np.isnan(raw[t, b[t-1]])); anym.append(np.isnan(raw[t]).any())
    caps, miss, anym = map(np.array, (caps, miss, anym)); L = -10 * np.log10(np.clip(caps, 1e-6, None))
    print("%-10s replay rule %.4f | held-missing excluded %.4f (n=%d) | any-missing frames excluded %.4f (n=%d)" % (scn, L.mean(), L[~miss].mean(), miss.sum(), L[~anym].mean(), anym.sum()))
