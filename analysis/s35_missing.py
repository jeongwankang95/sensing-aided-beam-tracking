"""s35_missing.py: Scenario-35 screening statistics with missing (NaN) power readings handled two ways: as reported (zero power;
hold-previous floored at 1e-12, replay losses capped at 60 dB) and with frames containing a missing reading excluded from
scoring. Policies still act on every frame. Runs the loading code and the tracker of analysis/s35_screening.py.
Paper: supplementary Sec. S4-A (Scenario 35 lacks a reading on 137 of its 2,423 screened frames, 5.7%; the hold-previous mean
of 0.20 dB falls to 0.10 dB without those frames).
usage: python analysis/s35_missing.py"""
import numpy as np, pandas as pd, os, collections
src = open('analysis/s35_screening.py').read()
exec(src[:src.index("S9, bs9 = load_site('S9')")])
root = f"{ROOT}/dataset/scenario35"; df = pd.read_csv(f"{root}/scenario35.csv")
S35, _ = load_site('S35')
keep = [p for p, v in S35.items() if v['dup_frac'] == 0]
MS = {}
for p, g in df.groupby('seq_index', sort=True):
    g = g.sort_values('abs_idx')
    MS[p] = np.stack([np.isnan(np.loadtxt(os.path.join(root, str(r).lstrip('./')))) for r in g['unit1_pwr']])
nf = sum(len(S35[p]['P']) for p in keep); nm = sum(int(MS[p].any(1).sum()) for p in keep)
print('S35 no-dup: passes %d frames %d, frames with a missing reading %d (%.1f%%)' % (len(keep), nf, nm, 100 * nm / nf))
# hold-previous
L, held, anym = [], [], []
for p in keep:
    P = S35[p]['P']; M = MS[p]; b = P.argmax(1)
    L += list(-10*np.log10(np.maximum(P[np.arange(1, len(P)), b[:-1]], 1e-12)/P[1:].max(1)))
    held += list(M[np.arange(1, len(P)), b[:-1]]); anym += list(M[1:].any(1))
L, held, anym = map(np.array, (L, held, anym))
print('hold-previous mean/p95: reported %.3f / %.3f | held-missing excluded (%d) %.3f / %.3f | any-missing excluded %.3f / %.3f'
      % (L.mean(), np.percentile(L, 95), held.sum(), L[~held].mean(), np.percentile(L[~held], 95), L[~anym].mean(), np.percentile(L[~anym], 95)))
print('   largest hold-previous losses (dB):', np.round(np.sort(L)[-6:], 1))
# causal persistence: hold the link-setup beam, replay rule
c, a2 = [], []
for p in keep:
    P = S35[p]['P']; M = MS[p]; b0 = int(P[0].argmax())
    for t in range(1, len(P)):
        m = P[t].max()
        if m > 0: c.append(P[t, b0] / m); a2.append(M[t].any())
c, a2 = np.array(c), np.array(a2); lp = -10*np.log10(np.clip(c, 1e-6, None))
print('causal persistence: reported %.3f | any-missing excluded %.3f | capped frames %d' % (lp.mean(), lp[~a2].mean(), int((c <= 1e-6).sum())))
# feedback-only tracker at the shared constants (fb_replay logic), replay rule vs excluded
for sl in (0.4, 0.6, 0.8, 1.0):
    pol = Ctl(sensor=None, **{**K, 'slack': sl}); ls, ex = [], []
    for p in keep:
        P = S35[p]['P']; M = MS[p]; pol.reset(int(P[0].argmax()), float(P[0].max()))
        for t in range(1, len(P)):
            a, k = pol.step(P[t], None); m = float(P[t].max())
            if m > 0: ls.append(-10*np.log10(max(float(P[t, a]) / m, 1e-6))); ex.append(M[t].any())
    ls, ex = np.array(ls), np.array(ex)
    print('tracker slack %.1f: reported %.3f | any-missing excluded %.3f | 60-dB frames %d' % (sl, ls.mean(), ls[~ex].mean(), int((ls >= 59.99).sum())))
# hold-previous under the replay rule (capture floored at 1e-6, i.e. 60 dB cap), as in Table S1
Lc = []
for p in keep:
    P = S35[p]['P']; b = P.argmax(1)
    Lc += list(-10*np.log10(np.clip(P[np.arange(1, len(P)), b[:-1]] / P[1:].max(1), 1e-6, None)))
Lc = np.array(Lc)
print('hold-previous under the 60 dB cap: mean %.3f p95 %.3f (n=%d)' % (Lc.mean(), np.percentile(Lc, 95), len(Lc)))
# profile statistics (as in s35_screen.stats) with and without frames that contain a missing reading
pm, pmin, adj, am = [], [], [], []
for p in keep:
    P = S35[p]['P']; M = MS[p]; b = P.argmax(1); s = np.sort(P, 1)[:, ::-1]; second = np.argsort(P, 1)[:, -2]
    pm += list(10*np.log10(s[:, 0]/np.median(P, 1))); pmin += list(10*np.log10(s[:, 0]/np.maximum(P.min(1), 1e-12)))
    adj += list(np.abs(second-b) <= 1); am += list(M.any(1))
pm, pmin, adj, am = map(np.array, (pm, pmin, adj, am))
print('profile: peak/median med %.3f -> %.3f | peak/min med %.3f -> %.3f | runner-up adj %.1f -> %.1f %%'
      % (np.median(pm), np.median(pm[~am]), np.median(pmin), np.median(pmin[~am]), 100*adj.mean(), 100*adj[~am].mean()))
