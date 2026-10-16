"""missing_rescoring.py SCRIPT ARGS...: run an analysis script unchanged except that frames containing a missing (NaN)
per-beam reading are dropped from the scored set ("rescoring (b)"). Policies still act on every frame; only scoring changes.
The scripts that run src/replay_core.py through the three-line header get its code inlined first (the scripts behind the paper
were concatenations), then every scoring line "if m > 0: <list>.append(" is patched. Outputs the script writes to
results/analysis/ (and results/analysis/lag) go to results/analysis/missing_rescoring/ instead.
Paper: supplementary Sec. S4-A: on Scenarios 31-33 no system gain or marginal value of sensing moves by more than 0.02 dB, and
the controller against the unverified or always-on variants by up to 0.07 dB (analysis/missing_rescoring_compare.py).
usage: python analysis/missing_rescoring.py analysis/table2.py configs/locked_v3.json dataset/scenario31 0.6 \\
           --cache=results/predictions/both_scenario31.npz > results/analysis/missing_rescoring/table2_s31.log"""
import sys, re, os
script = sys.argv[1]; sys.argv = [script] + sys.argv[2:]
src = open(script).read()
HEADER = ('import os, sys\n'
          '_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "replay_core.py")\n'
          'exec(compile(open(_CORE).read(), _CORE, "exec"))   # same namespace as the original prelude3.py + tail concatenation\n')
if src.startswith(HEADER):
    src = open(os.path.join('src', 'replay_core.py')).read() + src[len(HEADER):]
pre = r'''
import numpy as _np, os as _os
_MC = {}
def _MISS(key):
    if key not in _MC:
        _MC[key] = bool(_np.isnan(_np.loadtxt(_os.path.join(root, str(key).lstrip('./')))).any())
    return _MC[key]
'''
n0 = 0
for pat, rep in ((r'if m ?> ?0: L\.append\(', 'if m > 0 and not _MISS(keys[t]): L.append('),
                 (r'if m ?> ?0: caps\.append\(', 'if m > 0 and not _MISS(keys[t]): caps.append('),
                 (r'if m ?> ?0: c\.append\(', 'if m > 0 and not _MISS(keys[t]): c.append('),
                 (r'if m ?> ?0: cs\.append\(', 'if m > 0 and not _MISS(keys[t]): cs.append(')):
    src, n = re.subn(pat, rep, src); n0 += n
assert n0 > 0, 'no scoring line patched'
OUT = os.path.join('results', 'analysis', 'missing_rescoring')
src = src.replace('results/analysis/', OUT + '/')
src = src.replace(repr(OUT + '/lag'), repr(OUT)).replace("'results/analysis'", repr(OUT))
os.makedirs(OUT, exist_ok=True)
print('[missing_rescoring] %s: %d scoring lines patched' % (script, n0))
exec(compile(pre + src, script, 'exec'))
