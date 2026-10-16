import os, sys
_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")
exec(compile(open(_CORE).read(), _CORE, "exec"))   # test passes and prediction cache of the shared loader
"""stale_share.py LOCKED ROOT SLACK --cache=CACHE: how often the stale-prediction control of Table III (analysis/attribution.py,
cand_stale) can use a prediction from at least 20 frames earlier. On every covered frame (a frame with a current prediction) of
the test passes, the control looks for the latest prediction made at least 20 frames earlier in the same pass; where there is
none it uses the current prediction. Prints both counts.
Paper: Sec. VI-B, "A prediction 20 frames earlier, or the current one where none is that old (58% of covered frames)".
usage: python analysis/stale_share.py configs/locked_v3.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz"""
stale = current = 0
for P, keys in passes:
    hist = []
    for t in range(1, len(P)):
        raw = PRED['vision'].get(keys[t])
        hist.append(raw)
        if raw is None: continue
        h = [x for x in hist[:-20] if x is not None]      # as cand_stale in analysis/attribution.py
        if h: stale += 1
        else: current += 1
n = stale + current
print('%s: %d covered frames on %d test passes; a prediction at least 20 frames older exists on %d (%.1f%%); '
      'the current prediction is used on %d (%.1f%%)' % (root, n, len(passes), stale, 100 * stale / n, current, 100 * current / n))
