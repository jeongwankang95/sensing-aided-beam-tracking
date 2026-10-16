"""drift.py: best-beam drift and straightness per scenario, on the test passes and on all passes (supplementary Table S1 and
Sec. S1-A; Table IV's drift column is decimation.py's). Drift per decimation d = 1, 2, 4: mean |step| of the best beam sampled
every d frames, averaged per pass over passes of at least six decisions (pw6) or of at least two (pw), pooled over steps, and
pooled over all d offsets; straightness = net over total movement (per pass and pooled); hold-previous loss (keep the last
epoch's best beam), frame-pooled and pass-mean.
Paper: supplementary Table S1 and Sec. S1-A (net displacement accounts for 0.80 of the total movement in Scenario 9; drift
grows about 3.6 times from d = 1 to d = 4).
usage: python analysis/drift.py"""
import numpy as np, pandas as pd, os, glob
for scn in ('scenario9', 'scenario31', 'scenario32', 'scenario33'):
    root = f'dataset/{scn}'
    f = f'{root}/{scn}.csv' if os.path.exists(f'{root}/{scn}.csv') else glob.glob(f'{root}/*_dev.csv')[0]
    base = pd.read_csv(f)
    tp = set(pd.read_csv(f'{root}/test_seqs_RA.csv')['seq_index'].unique())
    pcol = 'unit1_pwr_60ghz' if 'unit1_pwr_60ghz' in base.columns else [c for c in base.columns if 'pwr' in c][0]
    ordc = 'index' if 'index' in base.columns else base.columns[0]
    passes = {}
    for p, g in base.groupby('seq_index'):
        g = g.sort_values(ordc)
        P = np.stack([np.nan_to_num(np.loadtxt(os.path.join(root, str(r).lstrip('./'))), nan=0.0) for r in g[pcol]])
        passes[p] = P
    for name, sel in (('test', [p for p in passes if p in tp]), ('all', list(passes))):
        row = []
        for d in (1, 2, 4):
            pw6, pw, pn, pdn, on, odn = [], [], 0, 0, 0, 0
            for p in sel:
                b = passes[p].argmax(1)[::d]
                if len(b) >= 2:
                    st = np.abs(np.diff(b)); pw.append(st.mean()); pn += st.sum(); pdn += len(st)
                    if len(b) >= 6: pw6.append(st.mean())
                for o in range(d):
                    bo = passes[p].argmax(1)[o::d]
                    if len(bo) >= 2: so = np.abs(np.diff(bo)); on += so.sum(); odn += len(so)
            row.append('d%d pw6 %.2f pw %.2f pooled %.2f pooled-offsets %.2f' % (d, np.mean(pw6), np.mean(pw), pn / pdn, on / odn))
        # straightness: net/total per pass (pass-mean and pooled); hold-previous loss (frame-pooled and pass-mean)
        net = [abs(int(passes[p].argmax(1)[-1]) - int(passes[p].argmax(1)[0])) for p in sel]
        tot = [np.abs(np.diff(passes[p].argmax(1))).sum() for p in sel]
        st_pass = np.mean([n / t for n, t in zip(net, tot) if t > 0]); st_pool = sum(net) / sum(tot)
        hp = []; hp_pass = []
        for p in sel:
            P = passes[p]; b = P.argmax(1); m = P.max(1)
            l = [-10 * np.log10(max(P[t, b[t - 1]], 1e-12) / m[t]) for t in range(1, len(P)) if m[t] > 0]
            hp += l; hp_pass.append(np.mean(l))
        print('%-10s %-4s n=%3d | %s | straight pass %.2f pooled %.2f | holdprev frame %.2f pass %.2f'
              % (scn, name, len(sel), ' | '.join(row), st_pass, st_pool, np.mean(hp), np.mean(hp_pass)))
