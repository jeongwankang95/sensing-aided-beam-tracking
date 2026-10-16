"""dataset_facts.py: dataset facts used in Secs. III-B and III-F, supplementary Table S3 and Sec. S4: the native epoch (median
within-pass timestamp step), the number of passes, the training/validation/test pass counts with each split's longest pass
(frames), and the distance between the base stations of Scenarios 9 and 31 (and of 32 and 33, which share one). The repository
holds the validation and test lists; every other pass is a training pass. Paper: 111 ms in Scenario 9 and about 92 ms elsewhere;
91/17/28 and 35/6/11 passes; "a separate base station about 110 m away"; N_bl = 256 epochs is longer than any validation or test
pass of Scenarios 9 and 31.
usage: python analysis/dataset_facts.py"""
import pandas as pd, numpy as np, os, glob
def base(s):
    root = f"dataset/scenario{s}"; f = f"{root}/scenario{s}.csv"
    if not os.path.exists(f): f = glob.glob(f"{root}/*_dev.csv")[0]
    return root, pd.read_csv(f)
B = {}
for s in (9, 31, 32, 33):
    root, df = base(s)
    tcol = [c for c in df.columns if "time" in c.lower()][0]          # kin2p.py's timestamp parse
    ts = df[tcol].astype(str).str.extract(r"(\d+)[-:](\d+)[-:](\d+)[-.](\d+)").astype(float)
    df["t"] = ts[0]*3600 + ts[1]*60 + ts[2] + ts[3]/10**ts[3].astype(int).astype(str).str.len()
    d = np.concatenate([np.diff(g.sort_values("index")["t"].values) for _, g in df.groupby("seq_index")]); d = d[(d > 0) & (d < 1)]
    lens = df.groupby("seq_index").size(); out = []
    sets = {sp: set(pd.read_csv(f"{root}/{sp}_seqs_RA.csv")["seq_index"].unique()) for sp in ("val", "test")
            if os.path.exists(f"{root}/{sp}_seqs_RA.csv")}
    sets = dict(train=set(df["seq_index"].unique()) - set().union(*sets.values()), **sets)
    for sp, ids in sets.items():
        if ids: out.append(f"{sp}={len(ids)} (max len {int(lens[lens.index.isin(ids)].max())})")
    c = [x for x in df.columns if x.startswith("unit1_loc")][0]
    B[s] = np.loadtxt(os.path.join(root, str(df[c].iloc[0]).lstrip("./")))[:2]
    print(f"S{s}: epoch {1000*np.median(d):.1f} ms; passes {df['seq_index'].nunique()}; " + "; ".join(out))
def hav(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    return 6371000*2*np.arcsin(np.sqrt(np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2))
print("BS distance S9-S31 %.1f m, S32-S33 %.1f m" % (hav(B[9], B[31]), hav(B[32], B[33])))
