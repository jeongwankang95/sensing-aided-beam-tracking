# Sensing-aided vehicular beam tracking: replay code, constants and cached predictions

This repository accompanies

> J. Kang, M. Bordallo López, N. T. Nguyen, J. Kim and S. Kim, "Sensing-Aided Vehicular Beam Tracking: System Gain Over
> Link Feedback and Probe-Verified Sensing Invocation," arXiv:2610.20196, 2026. <https://arxiv.org/abs/2610.20196>

It contains the action-restricted causal replay of recorded 64-beam sweeps, the feedback-only trackers and re-sweeping
references, the probe-verified controller and the policies it is compared with, the constants selected on validation, the
cached per-frame predictions of the sensing models, and the scripts that turn the replays into the paper's tables and
values. The sensing models and their training come from the companion paper and its repository,
<https://github.com/jeongwankang95/radar-beam-prediction> (arXiv:2610.20185). The DeepSense 6G data are not included (see
`dataset/README.md`).

## Layout

| Folder | Contents |
|---|---|
| `src/` | `replay_core.py`: scenario loading, the controller (Algorithm 1 of the paper, class `Ctl`) and the replay loop; `rs_classes.py` and `llr_tail_classes.py`: the scheduled re-sweep and lock-loss-recovery references; `strong_fb.py`: the model-based feedback-only trackers of the frontier |
| `analysis/` | one script per analysis (table below); `run_analysis.sh` runs them all in dependency order; `check_reference.py` compares the outputs with `results/reference/` |
| `configs/` | `locked_v3.json`, the controller constants used throughout; `sel3_val.json` and `sel3_val_grid.npz`, the validation re-selection of the controller and of the re-sweeping and lock-loss references; `llr_scenario*.json`, `resweep_scenario*.json` and `loso3_val.json`, further validation selections; `locked_multi.json`, the earlier constants shown as the "deployed 08-10" controller in two comparisons |
| `results/predictions/` | cached per-frame beam predictions of the vision and radar arms: `both_scenario<s>.npz` (test passes), `_s1`, `_s2` (other training seeds), `_val` (validation passes); `post_scenario*.npz` (posteriors, for confidence admission and sensor-prior fusion); `dec2_s9.npz`, `dec4_s9.npz` (sensor inputs spaced by the decision interval); `sensor_preds_t0_scenario*.npz` (the caches before the anchor correction, read by one comparison in `table3.py`) |
| `results/reference/` | the outputs behind the paper, one log (and where written, one JSON) per script and scenario |
| `dataset/` | the validation and test pass lists of Scenarios 9, 31, 32 and 33 (the data themselves come from DeepSense 6G) |

## Reproducing the paper's numbers from the cached predictions

This needs Python 3 with numpy and pandas, and DeepSense 6G Scenarios 9, 31, 32 and 33 in `dataset/`, placed as in its
README: every loss is read from the measured beam powers. No GPU and no model checkpoint are needed.

```bash
bash analysis/run_analysis.sh
```

It runs every script below with the arguments of the paper (the constants in `configs/locked_v3.json`, the reference slack of
each scenario: 0.4 in Scenario 9, 0.6 in Scenario 31, 0.8 in Scenarios 32 and 33), writes to `results/analysis/`, and ends with
`analysis/check_reference.py`. It takes about three minutes on a 16-core server and longer on a laptop; each script can also
be run alone, for example

```bash
python analysis/table2.py configs/locked_v3.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz
```

The replay is deterministic, and every interval is a paired bootstrap over test passes with a fixed generator seed, so the
outputs match `results/reference/` exactly. The scripts were checked with Python 3.12.3, NumPy 2.4.4 and pandas 3.0.5. The
reference outputs are those of the runs behind the paper; they differ from them only in the file paths they print.

Each analysis script runs the code of `src/replay_core.py` in its own namespace before its own part (as the scripts behind the
paper did), so a script may redefine any name of the core. Several scripts carry their own copy of the controller with the
same logic; `frontier.py` and `reselection_test.py` take the earlier constants as their first argument because their
"deployed 08-10" row needs them.

### Where each result comes from

| Paper | Script | Output in `results/reference/` |
|---|---|---|
| Table II, per-pass values (Fig. 4) | `table2.py` | `table2_s*.log`, `t2_scenario*.json` |
| Table III; verification ablations; five-decimal values of Table II | `table3.py` | `table3_s*.log`, `prec_scenario*.json` |
| Table IV and Sec. VI-C (every-frame scoring); Fig. 6 decimation rows; supplementary every-frame tables | `every_frame.py`, after `decimation_slack.py`, `resweep_decimation.py` and `decimation_sensor_off.py`; drift and innovation columns from `decimation.py` | `every_frame_s*.log`, `hold_scenario*.json`, `decimation_s*.log` |
| Sec. VI-C re-sweeping reference under decimation; comparators re-selected on validation | `every_frame_reselect.py` | `every_frame_reselect_s*.log`, `holdsel_scenario*.json` |
| Sec. VI-A sensor and re-sweep factorial, marginal value of sensing | `factorial.py` | `factorial_s*.log`, `fact3_scenario*.json` |
| Sec. VI-B and supplementary controls: sign and permutation tests, blacklist grid, trial log, K-curve | `controls.py` | `controls_s*.log` |
| Warm-up, trial counts and the gate controls | `warmup.py`, `controls_gate.py` | `warmup_s*.log`, `controls_gate_s*.log` |
| Invocation without verification; attribution controls at 4,000 resamples | `invocation_ablation.py`, `attribution.py` | `invocation_ablation_s*.log`, `attribution_s*.log` |
| Confidence admission and sensor-prior fusion | `competitors.py` | `competitors.log`, `comp4.json` |
| Sec. VI-D capture-loss exceedance and tails | `exceedance.py` | `exceedance_s*.log` |
| Sec. VI-D measurement noise; Fig. 6 noise rows | `noise.py`, `noise_skeleton.py` (five decimals: `noise_p5.py`, `noise_skeleton_p5.py`) | `noise*_s9.log` |
| Measurement noise with a higher alarm threshold or margin | `noise_threshold.py` | `noise_threshold_s9.log`, `scenario9_noise_retune.json` |
| Sec. VI-D delays, commitment lags and stale sensing; Fig. 6 delay rows | `delays_sensor_off.py` (five decimals: `delays_sensor_off_p5.py`), `delays.py` | `delays_sensor_off*_s*.log`, `lag/`, `lag_p5/`, `delays_s*.log` |
| Stale link-power observations | `stale_observations.py` | `stale_observations_s*.log` |
| Sec. V leave-one-site-out constant selection | `loso.py` | `loso_s*.log`, `loso3_scenario*.json` |
| Parameter sensitivity across scenarios | `sensitivity.py` | `sensitivity_s*.log`, `sweep3_scenario*.json` |
| Vision and radar sensing arms | `arms.py` | `arms_s*.log` |
| Training seeds | `seeds.py` (run with the `_s1` and `_s2` caches) | `seeds_s1_s*.log`, `seeds_s2_s*.log` |
| Sensor coverage and the always-on fallback | `coverage.py` | `coverage_s*.log` |
| Sensor inputs spaced by the decision interval | `pessimistic_inputs.py`, `resweep_pessimistic.py` | `pessimistic_inputs_s9.log`, `resweep_pessimistic_s9.log` |
| Every-frame scoring without missing readings | `every_frame_missing.py` | `every_frame_missing_s*.log` |
| Probe/loss frontier and convex-hull crossing (Fig. 3 caption values) | `frontier.py` | `frontier_s9.log`, `hull5_scenario9.json`, `scenario9_strongfb_val.json` |
| One-time test evaluation of the re-selected constants | `reselection_test.py` | `reselection_test_s*.log`, `test3_scenario*.json` |

Status (2026-10-15): the scripts that draw Figs. 3, 5 and 6, the validation re-selection itself (`configs/sel3_val.json` is
its output), and a few small computations quoted in the text will be added in the next update.

## License

The code is released under the MIT License (`LICENSE`). The cached predictions and results are derived from DeepSense 6G
Scenarios 9, 31, 32 and 33, whose terms of use apply to the data.

## Citation

```bibtex
@misc{kang2026sensingaided,
  author        = {Kang, Jeongwan and Bordallo L{\'o}pez, Miguel and Nguyen, Nhan Thanh and Kim, Junhyeong and Kim, Sunwoo},
  title         = {Sensing-Aided Vehicular Beam Tracking: System Gain Over Link Feedback and Probe-Verified Sensing
                   Invocation},
  year          = {2026},
  eprint        = {2610.20196},
  archivePrefix = {arXiv},
  primaryClass  = {eess.SP},
  url           = {https://arxiv.org/abs/2610.20196}
}
```
