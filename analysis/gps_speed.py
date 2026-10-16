"""gps_speed.py: vehicle speed from the GPS track of each Scenario-9 test pass (raw unit2_loc and calibrated unit2_loc_cal),
for the seven lock-loss passes (plain-tracker pass-mean loss of at least 2 dB; seq_index 116, 122, 125, 127, 130, 132, 133) and
the other 21: per-pass median, mean, net and path speeds, averaged (mean-of-pass) and medianed (median-of-pass) over each group.
Paper: supplementary Sec. S17-H, "a mean GPS speed of 3.76 against 3.77 m/s" (the mean-of-pass of the per-pass mean
speed): the lock-loss passes are not faster.
usage: python analysis/gps_speed.py"""
import csv, os, re, numpy as np
root = 'dataset/scenario9'
rows = list(csv.DictReader(open(f'{root}/scenario9.csv')))
test = {r['seq_index'] for r in csv.DictReader(open(f'{root}/test_seqs_RA.csv'))}
byp = {}
for r in rows:
    if r['seq_index'] in test: byp.setdefault(r['seq_index'], []).append(r)
LL = {'116', '127', '130', '122', '125', '132', '133'}
def hav(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[:, 0], a[:, 1], b[:, 0], b[:, 1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(h))
res = {}
for col in ('unit2_loc', 'unit2_loc_cal'):
    for p, g in byp.items():
        g = sorted(g, key=lambda r: int(r['index']))
        G = np.stack([np.loadtxt(os.path.join(root, r[col].lstrip('./')))[:2] for r in g])
        ts = []
        for r in g:
            m = re.findall(r'(\d+)-(\d+)-(\d+)-(\d+)', r['time_stamp[UTC]'])[0]
            ts.append(int(m[0]) * 3600 + int(m[1]) * 60 + int(m[2]) + int(m[3]) / 1e3)
        ts = np.array(ts); d = hav(G[:-1], G[1:]); dt = np.diff(ts); v = d / np.maximum(dt, 1e-3)
        res[(col, p)] = dict(med=np.median(v), mean=np.mean(v), net=hav(G[:1], G[-1:])[0] / (ts[-1] - ts[0]),
                             path=d.sum() / (ts[-1] - ts[0]), gpsv=None)
    for stat in ('med', 'mean', 'net', 'path'):
        a = np.mean([res[(col, p)][stat] for p in byp if p in LL]); b = np.mean([res[(col, p)][stat] for p in byp if p not in LL])
        am = np.median([res[(col, p)][stat] for p in byp if p in LL]); bm = np.median([res[(col, p)][stat] for p in byp if p not in LL])
        print('%-14s %-5s  mean-of-pass: LL %.2f rest %.2f | median-of-pass: LL %.2f rest %.2f' % (col, stat, a, b, am, bm))
