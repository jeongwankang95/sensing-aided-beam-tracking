"""beam_dynamics.py: how fast the best beam and the beam powers move on the test passes, from the recorded sweeps.
1. Per scenario: the median epoch (time between dataset samples), the best beam's median net displacement per decision when
   decisions are taken every d = 1, 2, 4 epochs (per pass |b_last - b_first| / number of decisions, median over passes), and
   the median net traverse rate (indices per second) times the epoch.
   Paper: Sec. III-B and Fig. 1 (one epoch is 111 ms in Scenario 9, about 92 ms elsewhere); Sec. VI-C (1.4, 2.9 and 6.0
   indices per decision in Scenario 9 at d = 1, 2, 4; 0.16-0.38 at the far sites at d = 1).
2. Scenario 9: inter-epoch power change |P_t - P_t-1| in dB of the beam that was best at t-1 ("best_prev"; a fixed beam), and
   of other beam sets. Paper: limitations in Sec. VII and supplementary Sec. S3 (median 1.42 dB, 95th percentile 5.23 dB).
usage: python analysis/beam_dynamics.py"""
import numpy as np, pandas as pd, os, glob


def load(scn):
    root = f'dataset/{scn}'
    f = f'{root}/{scn}.csv' if os.path.exists(f'{root}/{scn}.csv') else glob.glob(f'{root}/*_dev.csv')[0]
    base = pd.read_csv(f); tp = set(pd.read_csv(f'{root}/test_seqs_RA.csv')['seq_index'].unique())
    out = []
    for p, g in base.groupby('seq_index'):
        if p not in tp: continue
        g = g.sort_values('index')
        raw = np.stack([np.loadtxt(os.path.join(root, str(r).lstrip('./'))) for r in g['unit1_pwr_60ghz']])
        ts = g[[c for c in g.columns if 'time' in c.lower()][0]].astype(str).str.extract(r"(\d+)[-:](\d+)[-:](\d+)[-.](\d+)").astype(float)
        t = (ts[0]*3600 + ts[1]*60 + ts[2] + ts[3]/10**ts[3].astype(int).astype(str).str.len()).values
        out.append((np.nan_to_num(raw, nan=0.0), [str(r) for r in g['unit1_pwr_60ghz']], raw, t))
    return out


# 1. net displacement per decision
for scn in ('scenario9', 'scenario31', 'scenario32', 'scenario33'):
    ps = load(scn)
    ep = np.median(np.concatenate([np.diff(t) for _, _, _, t in ps]))
    row = []
    for d in (1, 2, 4):
        v = []
        for P, _, _, _ in ps:
            b = P.argmax(1)[::d]
            if len(b) >= 2: v.append(abs(int(b[-1]) - int(b[0])) / (len(b) - 1))
        row.append(np.median(v))
    tr = np.median([abs(int(P.argmax(1)[-1]) - int(P.argmax(1)[0])) / (t[-1] - t[0]) for P, _, _, t in ps])
    print('%s median epoch %.4f s | median net/decision d=1,2,4: %.3f %.3f %.3f | trav %.3f idx/s x epoch = %.3f' % (scn, ep, *row, tr, tr*ep))

# 2. inter-epoch power change on the Scenario-9 test passes
root = 'dataset/scenario9'
base = pd.read_csv(f'{root}/scenario9.csv'); tp = set(pd.read_csv(f'{root}/test_seqs_RA.csv')['seq_index'].unique())
V = {k: [] for k in ('best_prev', 'best_cur', 'top5_prev', 'above_med', 'all_lin_rel', 'setup_beam', 'served_window3')}
for p, g in base.groupby('seq_index'):
    if p not in tp: continue
    g = g.sort_values('index')
    P = np.stack([np.nan_to_num(np.loadtxt(os.path.join(root, str(r).lstrip('./'))), nan=0.0) for r in g['unit1_pwr_60ghz']])
    L = 10*np.log10(np.maximum(P, 1e-12)); b = P.argmax(1)
    for t in range(1, len(P)):
        V['best_prev'].append(abs(L[t, b[t-1]] - L[t-1, b[t-1]]))
        V['best_cur'].append(abs(L[t, b[t]] - L[t-1, b[t]]))
        top = np.argsort(P[t-1])[-5:]; V['top5_prev'] += list(np.abs(L[t, top] - L[t-1, top]))
        m = P[t-1] > np.median(P[t-1]); V['above_med'] += list(np.abs(L[t, m] - L[t-1, m]))
        V['setup_beam'].append(abs(L[t, b[0]] - L[t-1, b[0]]))
        w = [x for x in range(b[t-1]-1, b[t-1]+2) if 0 <= x < 64]; V['served_window3'] += list(np.abs(L[t, w] - L[t-1, w]))
    # relative to the per-frame maximum (normalized profile)
    Ln = L - L.max(1, keepdims=True); V['all_lin_rel'] += list(np.abs(np.diff(Ln, axis=0)).ravel())
for k, v in V.items():
    v = np.array(v); v = v[np.isfinite(v)]
    print('%-15s n=%6d median %.3f p95 %.3f' % (k, len(v), np.median(v), np.percentile(v, 95)))
