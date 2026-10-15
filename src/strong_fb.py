#!/usr/bin/env python3
"""A3: stronger feedback-only denominators.

Two estimator families beyond the heuristic frontier, both feedback-only (they see
only the powers of beams they probe):

  BayesTracker  : Bayesian filter over (beam, velocity) on the beam grid. Prediction
                  step convolves the posterior with a velocity-diffusion kernel;
                  update step conditions on the measured powers of the probed beams
                  via a soft likelihood built from the flat-peak profile. Probes the
                  smallest set covering a credible mass 1-eps of the posterior, so
                  cost adapts to posterior width rather than to a hand-made rule.
  ViterbiTracker: HMM over beam index with a banded transition matrix; probes a
                  window around the MAP state, width set by posterior entropy.

Free parameters are chosen on VALIDATION passes, exactly like the controller.
Usage: strong_fb.py <root> <val|test> [--cache=...]
"""
import numpy as np, pandas as pd, os, sys, glob, json

root, SPLIT = sys.argv[1], sys.argv[2]
C = 64
if os.path.exists(f'{root}/{os.path.basename(root)}.csv'):
    base = pd.read_csv(f'{root}/{os.path.basename(root)}.csv')
else:
    base = pd.read_csv(glob.glob(f'{root}/*_dev.csv')[0])
sel = set(pd.read_csv(f'{root}/{SPLIT}_seqs_RA.csv')['seq_index'].unique())
passes = []
for p, g in base.groupby('seq_index'):
    if p not in sel: continue
    g = g.sort_values('index')
    P = np.stack([np.nan_to_num(np.loadtxt(os.path.join(root, str(r).lstrip('./'))), nan=0.0)
                  for r in g['unit1_pwr_60ghz']])
    passes.append(P)
print(f'{root} [{SPLIT}]: {len(passes)} passes')

IDX = np.arange(C)

class Bayes:
    """posterior over beam index with a velocity state; probe a credible set."""
    def __init__(self, sigma_v=1.0, eps=0.15, kmax=13, tau=2.0):
        self.sv, self.eps, self.kmax, self.tau = sigma_v, eps, kmax, tau
    def reset(self, a0, p0):
        self.post = np.zeros(C); self.post[int(a0)] = 1.0
        self.vel = 0.0; self.a = int(a0); self.last = int(a0)
    def _diffuse(self):
        m = np.roll(self.post, int(round(self.vel)))
        w = np.exp(-0.5 * (np.arange(-4, 5) / self.sv) ** 2); w /= w.sum()
        self.post = np.convolve(m, w, mode='same') + 1e-9
        self.post /= self.post.sum()
    def step(self, Pt, key=None):
        self._diffuse()
        order = np.argsort(-self.post)
        c = np.cumsum(self.post[order])
        n = int(np.clip(np.searchsorted(c, 1 - self.eps) + 1, 1, self.kmax))
        S = np.sort(order[:n])
        obs = Pt[S]
        if obs.max() > 0:
            lik = np.zeros(C) + 1e-6
            # soft likelihood: probed powers, normalised, spread by the flat peak
            z = obs / (obs.max() + 1e-12)
            lik[S] = np.exp(self.tau * (z - 1.0))
            self.post = self.post * lik; self.post /= self.post.sum()
            b = int(S[int(np.argmax(obs))])
            self.vel = 0.7 * self.vel + 0.3 * (b - self.last); self.last = b
            self.a = b
        else:
            self.a = int(np.argmax(self.post))
        return self.a, len(S)

class Viterbi:
    """banded HMM; probe window from posterior entropy."""
    def __init__(self, band=3, wmin=1, wmax=8, ent_scale=2.0):
        self.band, self.wmin, self.wmax, self.es = band, wmin, wmax, ent_scale
        k = np.arange(-band, band + 1)
        self.T = np.exp(-0.5 * (k / (band / 2 + 1e-9)) ** 2); self.T /= self.T.sum()
    def reset(self, a0, p0):
        self.post = np.zeros(C); self.post[int(a0)] = 1.0; self.a = int(a0)
    def step(self, Pt, key=None):
        self.post = np.convolve(self.post, self.T, mode='same') + 1e-9
        self.post /= self.post.sum()
        H = -(self.post * np.log(self.post + 1e-12)).sum()
        W = int(np.clip(round(self.es * H), self.wmin, self.wmax))
        c = int(np.argmax(self.post))
        S = np.arange(max(0, c - W), min(C, c + W + 1))
        obs = Pt[S]
        if obs.max() > 0:
            lik = np.zeros(C) + 1e-6
            lik[S] = obs / (obs.max() + 1e-12)
            self.post = self.post * lik; self.post /= self.post.sum()
            self.a = int(S[int(np.argmax(obs))])
        else:
            self.a = c
        return self.a, len(S)

def run(pol):
    caps, cost = [], []
    for P in passes:
        pol.reset(int(P[0].argmax()), float(P[0].max()))
        for t in range(1, len(P)):
            a, k = pol.step(P[t])
            m = float(P[t].max())
            if m > 0: caps.append(float(P[t, a]) / m); cost.append(k)
    caps = np.array(caps)
    return float(np.mean(-10*np.log10(np.clip(caps,1e-6,None)))), float(np.mean(cost))

rows = []
for sv in (0.6, 1.0, 1.6):
    for eps in (0.05, 0.10, 0.20, 0.35, 0.5, 0.65, 0.8):
        for tau in (1.0, 3.0):
            for km in (5, 9, 13):
                l, c = run(Bayes(sigma_v=sv, eps=eps, tau=tau, kmax=km))
                rows.append((l, c, f'bayes(sv={sv},eps={eps},tau={tau},k={km})'))
for band in (2, 3, 5):
    for es in (0.5, 1.0, 2.0, 3.5):
        for wm in (2, 4, 8):
            l, c = run(Viterbi(band=band, ent_scale=es, wmax=wm))
            rows.append((l, c, f'viterbi(band={band},es={es},wmax={wm})'))
rows.sort(key=lambda r: (round(r[0], 4), round(r[1], 3)))
BUD = float(next((a.split('=',1)[1] for a in sys.argv if a.startswith('--budget=')), 2.64))
print(f"\nunconstrained best:")
for l, c, n in rows[:3]:
    print(f'  {l:7.3f} {c:6.2f}  {n}')
elig = [r for r in rows if r[1] <= BUD]
print(f"\nbest at <= {BUD} probes/frame (the controller's cost):")
for l, c, n in elig[:5]:
    print(f'  {l:7.3f} {c:6.2f}  {n}')
if elig:
    json.dump({'best_at_budget': elig[0][2], 'loss': elig[0][0], 'probes': elig[0][1],
               'budget': BUD, 'uncon_best': rows[0][2], 'uncon_loss': rows[0][0],
               'uncon_probes': rows[0][1]},
              open('results/analysis/' + sys.argv[1].rstrip('/').split('/')[-1] + f'_strongfb_{SPLIT}.json','w'), indent=1)
print('STRONG_FB_DONE')
