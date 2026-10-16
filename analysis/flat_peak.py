"""flat_peak.py: how flat the measured beam profile is at its peak on the Scenario-9 test passes: the median gap between the
best and the runner-up beam (dB) and how often the runner-up is an immediate neighbour of the best, on the scored frames (t >= 1)
and including the link-setup frames.
Paper: Sec. III-G, "the runner-up beam sits within 0.23 dB of the best at the median and is an immediate neighbor 87.0% of the
time" (scored frames).
usage: python analysis/flat_peak.py"""
import csv, os, numpy as np
root = 'dataset/scenario9'
rows = list(csv.DictReader(open(f'{root}/scenario9.csv')))
te = {r['seq_index'] for r in csv.DictReader(open(f'{root}/test_seqs_RA.csv'))}
byp = {}
for r in rows: byp.setdefault(r['seq_index'], []).append(r)
def stats(frames):
    g=[]; nb=[]
    for p in frames:
        o=np.argsort(p)[::-1]; g.append(10*np.log10(p[o[0]]/p[o[1]])); nb.append(abs(int(o[0])-int(o[1]))==1)
    return len(g), float(np.median(g)), 100*float(np.mean(nb))
scored=[]; withsetup=[]
for p in sorted(te, key=int):
    g = sorted(byp[p], key=lambda r: int(r['index']))
    P = np.stack([np.nan_to_num(np.loadtxt(os.path.join(root, r['unit1_pwr_60ghz'].lstrip('./'))), nan=0.0) for r in g])
    for t in range(len(P)):
        if P[t].max() <= 0: continue
        withsetup.append(P[t])
        if t >= 1: scored.append(P[t])
print('scored test frames (t>=1): n=%d median gap %.4f dB, neighbour %.2f%%' % stats(scored))
print('test frames incl. setup: n=%d median gap %.4f dB, neighbour %.2f%%' % stats(withsetup))
