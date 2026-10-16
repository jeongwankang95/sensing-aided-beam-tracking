# Sensing-aided vehicular beam tracking: replay code, constants and cached predictions

This repository accompanies

> J. Kang, M. Bordallo López, N. T. Nguyen, J. Kim and S. Kim, "Sensing-Aided Vehicular Beam Tracking: System Gain Over
> Link Feedback and Probe-Verified Sensing Invocation," arXiv:2610.20196, 2026. <https://arxiv.org/abs/2610.20196>

It contains the action-restricted causal replay of recorded 64-beam sweeps, the feedback-only trackers and re-sweeping
references, the probe-verified controller and the policies it is compared with, the selection of the constants on validation,
the cached per-frame predictions of the sensing models, and the scripts that turn the replays into the paper's tables,
figures and values. The sensing models and their training come from the companion paper and its repository,
<https://github.com/jeongwankang95/radar-beam-prediction> (arXiv:2610.20185). The DeepSense 6G data are not included (see
`dataset/README.md`).

## Layout

| Folder | Contents |
|---|---|
| `src/` | `replay_core.py`: scenario loading, the controller (Algorithm 1 of the paper, class `Ctl`) and the replay loop; `rs_classes.py` and `llr_tail_classes.py`: the scheduled re-sweep and lock-loss-recovery references; `strong_fb.py`: the model-based feedback-only trackers of the frontier |
| `analysis/` | one script per analysis (tables below); `run_analysis.sh` runs them all in dependency order; `check_reference.py` compares the outputs with `results/reference/`; `latency_bench.py` times the sensing models (not part of the run, see below) |
| `figures/` | `fig1_labels.py`, `fig3_frontier.py`, `fig4_perpass.py`, `fig5_example.py`, `fig6_vsense.py`: write the TikZ code of Figs. 3 to 6 (and the data labels of Fig. 1) from the analysis outputs to `results/figures/`; `tikz/`: the TikZ code of Figs. 2 to 6 as printed in the paper; `make_figures.sh` and `make_figures.tex`: compile TikZ code into one PDF per figure |
| `configs/` | `locked_v3.json`, the controller constants used throughout; `sel3_val.json` and `sel3_val_grid.npz`, the validation re-selection of the controller and of the re-sweeping and lock-loss references (written by `analysis/selection.py`); `llr_scenario*.json` and `resweep_scenario*.json`, the validation selections of the lock-loss-recovery and re-sweep references (written by `analysis/lock_loss_selection.py` and `analysis/resweep_selection.py`); `loso3_val.json`, the leave-one-site-out selections (written by `analysis/selection_summary.py`); `locked_multi.json`, the earlier constants (2026-08-10), used where the paper reports them: the "deployed 08-10" rows, the anchor-coverage correction (supplementary Sec. S10) and the controller rows of the reference selections |
| `results/predictions/` | cached per-frame beam predictions of the vision and radar arms: `both_scenario<s>.npz` (test passes), `_s1`, `_s2` (other training seeds), `_val` (validation passes); `post_scenario*.npz` (posteriors, for confidence admission, sensor-prior fusion and the confidence AUCs); `dec2_s9.npz`, `dec4_s9.npz` (sensor inputs spaced by the decision interval); `sensor_preds_t0_scenario*.npz` and `sensor_preds_val_scenario*.npz` (the test and validation caches before the anchor correction, read by the comparisons of supplementary Sec. S10 and by the self-check of `analysis/selection.py`) |
| `results/reference/` | the outputs behind the paper, one log (and where written, one JSON) per script and scenario; `figures/` holds the outputs of the figure scripts |
| `results/latency/` | `bench_infer_20261013_idle.json`, the latency measurement behind Sec. IV-D and supplementary Sec. S9 |
| `dataset/` | the validation and test pass lists of Scenarios 9, 31, 32 and 33 (the data themselves come from DeepSense 6G) |

## Reproducing the paper's numbers and figures from the cached predictions

This needs Python 3 with numpy and pandas (`requirements.txt`), and DeepSense 6G Scenarios 9, 31, 32 and 33 in `dataset/`,
placed as in its README: every loss is read from the measured beam powers, and a few scripts also read the timestamps and
GPS positions. Scenario 35 is optional; its two screening scripts run only if it is present. No GPU and no model checkpoint are
needed.

```bash
bash analysis/run_analysis.sh
```

It runs every script below with the arguments of the paper (the constants in `configs/locked_v3.json`, or the earlier ones
where the paper reports them; the reference slack of each scenario: 0.4 in Scenario 9, 0.6 in Scenario 31, 0.8 in Scenarios 32
and 33), in ten stages because later scripts read earlier outputs. It writes to `results/analysis/` and `results/figures/` and
ends with `analysis/check_reference.py`. On the server used for the paper (224 cores) it takes four to five minutes. The
validation re-selection (`analysis/selection.py`: 136,080 settings, about an hour of single-core time) dominates and uses
`WORKERS` processes (default 64; for example `WORKERS=8 bash analysis/run_analysis.sh` on a laptop). Each script can also be
run alone from the repository root, for example

```bash
python analysis/table2.py configs/locked_v3.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz
```

The replay is deterministic, and every interval is a paired bootstrap over test passes with a fixed generator seed, so the
outputs match `results/reference/` exactly. `analysis/check_reference.py` compares logs and TikZ files line by line, JSON files
field by field (floats to a relative tolerance of 1e-9) and the `.npz` grid exactly; the outputs that other scripts read from
`configs/` (`sel3_val.json`, `sel3_val_grid.npz`, `loso3_val.json`, `llr_scenario*.json`, `resweep_scenario*.json`) are
compared with those files. The scripts were checked with Python 3.12.3, NumPy 2.4.4 and pandas 3.0.5. The reference outputs
are those of the runs behind the paper; they differ from them only in the file paths they print and, for a few scripts, in
summary lines added for this release (for example the counts behind a sentence of the paper). Where a value of the paper came
from a computation that left no output, the reference is the output of the script written for it, checked against the value
the paper prints.

Each analysis script runs the code of `src/replay_core.py` in its own namespace before its own part (as the scripts behind the
paper did), so a script may redefine any name of the core. Several scripts carry their own copy of the controller with the
same logic. Scripts that report the earlier constants take `configs/locked_multi.json` as their first argument.

### Figures

`bash figures/make_figures.sh` compiles Fig. 2 (`figures/tikz/fig2_controller_flow.tikz`, a diagram) and the regenerated
Figs. 3 to 6 (`results/figures/*.tikz`) into one PDF each in `results/figures/`, in the paper's document class;
`bash figures/make_figures.sh paper` compiles the paper's own TikZ code (`figures/tikz/`) into `results/figures/paper/`, which
reproduces the published figures. This needs pdflatex with `-shell-escape` and the IEEEtran class (both in TeX Live).

- Figs. 4, 5 and 6: the regenerated TikZ code equals the paper's (`analysis/check_reference.py` compares it).
- Fig. 3: every coordinate is recomputed from the replays. The paper's figure was redrawn from an earlier PDF figure, so a few
  of its markers carry coordinates recovered from that PDF; `figures/fig3_frontier.py` prints the largest difference per
  series. The lock-loss-recovery and re-sweep markers are identical; the frontier and the periodic, cadence, extrapolation and
  tracker markers differ by at most 0.001 probes/epoch and 0.00013 in log10(dB), the Bayesian-filter markers by at most 0.005
  probes/epoch and 0.0014 in log10(dB), and the two diamonds (the controller with and without its sensor) and the dotted
  crossing segment by at most 0.0004 in log10(dB), because the paper drew them at rounded losses (0.225 and 0.2598 dB).
- Fig. 1 is a schematic of a recorded pass; `figures/fig1_labels.py` prints its data labels (feedback-only loss and vision
  top-1 per scenario).

### Latency

`python analysis/latency_bench.py --companion=PATH` times single-sample inference of the two sensing models, with the model
definitions of the companion repository cloned at `PATH` and random weights (latency does not depend on them). It needs
PyTorch, and the times depend on the machine and its load, so it is not part of `run_analysis.sh`. The run behind the paper is
`results/latency/bench_infer_20261013_idle.json`.

### Where each result comes from

Scripts are in `analysis/` unless the path says otherwise; outputs are in `results/reference/`. Each script's docstring names
the values of the paper it prints. Some quoted values are differences, ratios or roundings of printed values, or come from the
full-precision JSON written by the same script.

**Main paper**

| Paper | Script | Output |
|---|---|---|
| Fig. 1 data labels | `figures/fig1_labels.py` | `figures/fig1_labels.log` |
| Secs. III-B and III-F: epochs, pass counts, base stations | `dataset_facts.py` | `dataset_facts.log` |
| Sec. III-E: oracle-initialized persistence; its top-1 on Scenario 32 | `oracle_fed.py`, `persistence_top1.py` | `oracle_fed_s*.log`, `persistence_top1_s32.log` |
| Sec. III-G: flat peak of the beam profile | `flat_peak.py` | `flat_peak_s9.log` |
| Sec. IV-B: feedback-only baselines; the policies of Fig. 3 | `policy_family.py` | `policy_family_s9.log`, `policy_family_s9.json` |
| Sec. IV-D: inference latency | `latency_bench.py` | `results/latency/bench_infer_20261013_idle.json` |
| Sec. IV-F: AUC of the model's own confidence | `confidence_auc.py` | `confidence_auc_s*.log` |
| Table II; per-pass values (Fig. 4) | `table2.py`, `figures/fig4_perpass.py` | `table2_s*.log`, `t2_scenario*.json`, `figures/fig4_perpass.*` |
| Table III; verification ablations; five-decimal values of Table II | `table3.py` | `table3_s*.log`, `prec_scenario*.json` |
| Sec. VI-A: sensor and re-sweep factorial, marginal value of sensing | `factorial.py` | `factorial_s*.log`, `fact3_scenario*.json` |
| Sec. VI-A and Fig. 3: frontier, equal-budget value and probing ratio | `frontier.py`, `hull_crossing.py`, `figures/fig3_frontier.py` | `frontier_s9.log`, `hull5_scenario9.json`, `scenario9_strongfb_val.json`, `hull_crossing.log`, `figures/fig3_frontier.*` |
| Sec. VI-A: lock-loss prevalence, setup sweep, sign and permutation tests, probe saving | `prevalence.py`, `setup_sweep.py`, `controls.py`, `probe_saving.py` | `prevalence.log`, `setup_sweep_s9.log`, `controls_s9.log`, `probe_saving.log` |
| Fig. 5: example passes | `example_traces.py`, `figures/fig5_example.py` | `trace_scenario*.json`, `example_traces_s*.log`, `figures/fig5_example.*` |
| Sec. VI-B: attribution controls; stale predictions | `attribution.py`, `stale_share.py` | `attribution_s*.log`, `stale_share_s9.log` |
| Table IV and Sec. VI-C (every-frame scoring); Fig. 6 decimation rows | `every_frame.py` (after `decimation_slack.py`, `resweep_decimation.py`, `decimation_sensor_off.py`), `table4.py`; drift and innovation columns from `decimation.py` | `every_frame_s*.log`, `hold_scenario*.json`, `table4.log`, `decimation_s*.log` |
| Sec. VI-C: accuracy under decimation, beam motion per decision, rank correlations | `decimation_accuracy.py`, `beam_dynamics.py`, `decimation_correlation.py` | `decimation_accuracy_s9.log`, `beam_dynamics.log`, `decimation_correlation.log` |
| Sec. VI-C: re-sweeping reference under decimation; comparators re-selected on validation | `every_frame_reselect.py` | `every_frame_reselect_s*.log`, `holdsel_scenario*.json` |
| Sec. VI-D: measurement noise; Fig. 6 noise rows | `noise.py`, `noise_skeleton.py` (five decimals: `noise_p5.py`, `noise_skeleton_p5.py`) | `noise*_s9.log` |
| Sec. VI-D: delays, commitment lags and stale sensing; Fig. 6 delay rows | `delays_sensor_off.py` (five decimals: `delays_sensor_off_p5.py`), `delays.py` | `delays_sensor_off*_s*.log`, `lag/`, `lag_p5/`, `delays_s*.log`, `scenario*_lag.json` |
| Sec. VI-D: stale link-power observations | `stale_observations.py` | `stale_observations_s*.log` |
| Fig. 6 | `figures/fig6_vsense.py` | `figures/fig6_vsense.*` |
| Sec. V: leave-one-site-out constant selection | `loso.py` | `loso_s*.log`, `loso3_scenario*.json` |

**Supplementary material**

| Section | Script | Output |
|---|---|---|
| S1-A, Table S1: beam motion over all passes, hold-previous loss | `drift.py`, `hold_previous.py` | `drift.log`, `hold_previous.log` |
| S1-B to S1-D, Table S2: d-spaced sensor inputs, accuracy and coverage | `pessimistic_inputs.py`, `resweep_pessimistic.py`, `decimation_accuracy.py` | `pessimistic_inputs_s9.log`, `resweep_pessimistic_s9.log`, `resweep_pess_scenario9.json`, `decimation_accuracy_s9.log` |
| S1-G, Tables S27 and S28: every-frame scoring of decimation | `every_frame.py`, `every_frame_reselect.py`, `table4.py` | as for Table IV |
| S3: inter-epoch power change of a fixed beam | `beam_dynamics.py` | `beam_dynamics.log` |
| S4, Tables S3 and S4: splits, geometry, GPS speeds | `dataset_facts.py`, `kinematics.py`, `gps_geometry.py`; scored frames in `missing_readings_table2.py` | `dataset_facts.log`, `kinematics_s*.log`, `gps_geometry.log`, `missing_readings_table2_s*.log` |
| S4-A, Table S5: missing beam readings | `missing_readings_table2.py`, `missing_readings_decimation.py`, `missing_readings_delays.py`, `present_readings.py`, `every_frame_missing.py`, `missing_rescoring.py` and `missing_rescoring_compare.py`, `hold_previous.py`, `s35_missing.py` | `missing_readings_*_s*.log`, `present_readings.log`, `every_frame_missing_s*.log`, `holdx_scenario*.json`, `missing_rescoring/`, `missing_rescoring_compare.log` |
| S5: slack of the plain tracker | `tracker_slack.py` | `tracker_slack_s*.log` |
| S6, S6-A, S6-B: feedback-only comparison set, hull crossings, setup costs | `policy_family.py`, `frontier.py`, `hull_crossing.py` | as for Fig. 3 |
| S7, Table S6: capture-loss exceedance | `exceedance.py` | `exceedance_s*.log` |
| S8: sensor coverage, hold-last and always-on fallbacks | `coverage.py`, `hold_last.py` | `coverage_s*.log`, `hold_last_s*.log` |
| S9: latency budget | `latency_bench.py`; call rates from `hold_last.py` and `table2.py` | `results/latency/` |
| S10: anchor-coverage correction (earlier constants) | `table3.py` (coverage before the correction), `always_on_list.py`, `decimation.py` and `pessimistic_inputs.py` run with `configs/locked_multi.json`; anchor counts in `selection.py` | `table3_s*.log`, `always_on_list_*_s9.log`, `decimation_earlier_*_s9.log`, `pessimistic_inputs_earlier_s9.log`, `selection.log` |
| S11, Table S7: parameter sensitivity | `sensitivity.py` | `sensitivity_s*.log`, `sweep3_scenario*.json` |
| S12, Table S8: vision and radar arms, constant-beam outputs | `arms.py`, `constant_beam.py` | `arms_s*.log`, `constant_beam_s*.log` |
| S13, Table S9: invocation without verification | `invocation_ablation.py`, `table3.py` | `invocation_ablation_s*.log` |
| S14, Table S10: stale link-power observations | `stale_observations.py` | `stale_observations_s*.log` |
| S17-A, S17-B, S17-D, Tables S11, S12, S14: blacklist and gate grid, trial log, always-on with verification | `controls.py` | `controls_s*.log`, `scenario*.json` |
| S17-C, Table S13: delaying the gate, blacklist variants | `warmup.py`, `controls_gate.py` | `warmup_s*.log`, `scenario*_warmup.json`, `controls_gate_s*.log`, `scenario*_variants.json` |
| S17-E: full-sweep feedback-only anchors | `full_sweep.py` | `full_sweep_s*.log`, `scenario*_fullsweep.json` |
| S17-F, Table S15: training seeds | `seeds.py` (run with the `_s1` and `_s2` caches), `constant_beam.py` (the same caches) | `seeds_s{1,2}_s*.log`, `seed_*.json`, `constant_beam_s{1,2}_s*.log` |
| S17-G, Table S16: leave-one-site-out selection | `loso.py`, `selection_summary.py` | `loso_s*.log`, `selection_summary.log` |
| S17-H: validation and test passes on Scenario 9 | `decimation_slack.py`, `gps_speed.py`, `table2.py` | `decimation_slack_s9.log`, `scenario9_valdist.json`, `gps_speed.log` |
| S17-I: comparator slack at each decimation | `decimation_slack.py`, `every_frame_reselect.py`, `resweep_decimation.py` | `decimation_slack_s*.log`, `scenario*_perd_slack.json`, `resweep_decimation_s9.log`, `resweep_decim_scenario9.json` |
| S17-J, Table S17: noise with a higher alarm threshold or margin | `noise_threshold.py` | `noise_threshold_s9.log`, `scenario9_noise_retune.json` |
| S17-K, Table S18: joint delays and commit lags | `delays_sensor_off.py` | as for Sec. VI-D |
| S17-L, Table S19: model-side confidence and sensor-prior fusion | `confidence_auc.py`, `competitors.py` | `confidence_auc_s*.log`, `competitors.log`, `comp4.json` |
| S17-M, Table S20: screening of Scenario 35 | `s35_screening.py`, `s35_missing.py` | `s35_screening.log`, `s35_screen.json`, `s35_missing.log` |
| S17-N, Table S21: re-selection of the controller constants | `selection.py`, `selection_summary.py`, `reselection_test.py`, `frontier.py` | `selection.log`, `selection_summary.log`, `reselection_test_s*.log`, `test3_scenario*.json` |
| S17-O, Table S22: lock-loss recovery for the feedback reference | `lock_loss_selection.py`, `reselection_test.py`, `table2.py`, `table3.py` | `lock_loss_selection_s*.log` |
| S17-P, Tables S23 and S24: scheduled re-sweep reference | `resweep_selection.py`, `reselection_test.py`, `resweep_decimation.py`, `resweep_pessimistic.py`, `prevalence.py`, `setup_sweep.py`, `frontier.py` | `resweep_selection_s*.log` and as above |
| S17-Q, Tables S25 and S26: factorial and the marginal value of sensing | `factorial.py`, `decimation_sensor_off.py` | `factorial_s*.log`, `decimation_sensor_off_s*.log`, `decsk_scenario*.json` |

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
