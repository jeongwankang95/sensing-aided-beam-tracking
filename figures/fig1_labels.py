"""fig1_labels.py: the third line under each camera frame of Fig. 1, "<loss> dB · <accuracy>%": the feedback-only tracker's
loss against the recorded-sweep oracle (Table II, results/analysis/t2_scenario<s>.json) and the deployed vision arm's
current-slot top-1 accuracy (the "accuracy, consecutive inputs [cache]: d=1" value in results/analysis/table3_s<s>.log).
Fig. 1 itself is a drawing with dataset camera frames and is not part of this repository.
Paper: S9 1.04 dB · 45%, S31 0.11 dB · 28%, S32 0.06 dB · 7%, S33 0.25 dB · 3%.
usage: python figures/fig1_labels.py"""
import json
import os
import re

A = os.path.join('results', 'analysis')
for s in (9, 31, 32, 33):
    loss = json.load(open(os.path.join(A, f't2_scenario{s}.json')))['rows']['feedback-only tracker']['loss']
    acc = next(float(m.group(1)) for m in (re.search(r'accuracy, consecutive inputs \[cache\]: d=1 ([\d.]+)%', l)
                                             for l in open(os.path.join(A, f'table3_s{s}.log'))) if m)
    print(f'S{s}: {loss:.2f} dB · {acc:.0f}%   (loss {loss:.4f} dB, accuracy {acc:.2f}%)')
