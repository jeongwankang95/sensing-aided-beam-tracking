#!/bin/bash
# Regenerates every analysis output and figure source behind the paper from the stored predictions, then compares them
# with results/reference/. Needs DeepSense 6G Scenarios 9, 31, 32 and 33 in dataset/ (see dataset/README.md); the two
# Scenario-35 screening scripts run only if dataset/scenario35 is present.
# Stages run in order because later scripts read earlier outputs; within a stage the scripts run in parallel.
# Run from the repository root: bash analysis/run_analysis.sh   (PYTHON overrides the interpreter, WORKERS the
# processes of analysis/selection.py; default 64)
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
mkdir -p results/analysis/lag results/analysis/lag_p5 results/analysis/missing_rescoring results/figures

echo "== stage 1"
# table2: Table II and the per-pass values (Fig. 4)
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/table2.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/table2_s$s.log 2>&1 & done
# table3: Table III (10^5 resamples), verification ablations, five-decimal Table II rows
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/table3.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz --oldcache=results/predictions/sensor_preds_t0_scenario$s.npz > results/analysis/table3_s$s.log 2>&1 & done
# attribution: attribution controls at 4,000 resamples (no-gate rows of Scenarios 31-33)
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/attribution.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/attribution_s$s.log 2>&1 & done
# factorial: sensor and re-sweep factorial, marginal value of sensing
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/factorial.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/factorial_s$s.log 2>&1 & done
# invocation_ablation: invocation without verification (4,000 resamples)
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/invocation_ablation.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/invocation_ablation_s$s.log 2>&1 & done
# exceedance: capture-loss exceedance and tails
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/exceedance.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/exceedance_s$s.log 2>&1 & done
# coverage: sensor coverage and always-on fallback
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/coverage.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/coverage_s$s.log 2>&1 & done
# decimation: drift and innovation medians per decimation (Table IV columns)
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/decimation.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/decimation_s$s.log 2>&1 & done
# stale_observations: stale link-power observations
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/stale_observations.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/stale_observations_s$s.log 2>&1 & done
# arms: vision and radar sensing arms
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/arms.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/arms_s$s.log 2>&1 & done
# sensitivity: parameter sensitivity across scenarios
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/sensitivity.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/sensitivity_s$s.log 2>&1 & done
# loso: leave-one-site-out constant selection, test rows
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/loso.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/loso_s$s.log 2>&1 & done
# noise: measurement-noise study (Fig. 6)
for p in 9:0.4; do s=${p%%:*}; sl=${p##*:}; $PY analysis/noise.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/noise_s$s.log 2>&1 & done
# noise_skeleton: measurement noise against the sensor-disabled controller
for p in 9:0.4; do s=${p%%:*}; sl=${p##*:}; $PY analysis/noise_skeleton.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/noise_skeleton_s$s.log 2>&1 & done
# noise_p5: noise study, five decimals
for p in 9:0.4; do s=${p%%:*}; sl=${p##*:}; $PY analysis/noise_p5.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/noise_p5_s$s.log 2>&1 & done
# noise_skeleton_p5: noise against the sensor-disabled controller, five decimals
for p in 9:0.4; do s=${p%%:*}; sl=${p##*:}; $PY analysis/noise_skeleton_p5.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/noise_skeleton_p5_s$s.log 2>&1 & done
# pessimistic_inputs: decimation with d-spaced (pessimistic) sensor inputs
for p in 9:0.4; do s=${p%%:*}; sl=${p##*:}; $PY analysis/pessimistic_inputs.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz --d2=results/predictions/dec2_s9.npz --d4=results/predictions/dec4_s9.npz > results/analysis/pessimistic_inputs_s$s.log 2>&1 & done
# selection: validation re-selection of the constants (configs/sel3_val.json; run time to selection.time)
$PY analysis/selection.py --workers=${WORKERS:-64} > results/analysis/selection.log 2> results/analysis/selection.time &
# example_traces: per-frame traces behind Fig. 5 and the probe-saving interval
$PY analysis/example_traces.py configs/locked_v3.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz results/analysis/trace_scenario9.json > results/analysis/example_traces_s9.log 2>&1 &
$PY analysis/example_traces.py configs/locked_v3.json dataset/scenario31 0.6 --cache=results/predictions/both_scenario31.npz results/analysis/trace_scenario31.json > results/analysis/example_traces_s31.log 2>&1 &
# decimation_accuracy: arm accuracy and coverage under decimation (Table S2)
for p in 9:0.4; do s=${p%%:*}; sl=${p##*:}; $PY analysis/decimation_accuracy.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz --consec=results/predictions/both_scenario9.npz --d2=results/predictions/dec2_s9.npz --d4=results/predictions/dec4_s9.npz > results/analysis/decimation_accuracy_s$s.log 2>&1 & done
# stale_share: share of covered frames with a prediction 20 frames older (Sec. VI-B)
for p in 9:0.4; do s=${p%%:*}; sl=${p##*:}; $PY analysis/stale_share.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/stale_share_s$s.log 2>&1 & done
# kinematics: geometry and beam motion (Table S4)
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/kinematics.py configs/locked_multi.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/kinematics_s$s.log 2>&1 & done
# oracle_fed: oracle-initialized persistence on the test windows (Sec. III-E)
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/oracle_fed.py dataset/scenario$s > results/analysis/oracle_fed_s$s.log 2>&1 & done
# missing_readings_table2: Table II rows rescored without frames with a missing reading (Table S5)
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/missing_readings_table2.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/missing_readings_table2_s$s.log 2>&1 & done
# missing_readings_decimation: decimation rescored without frames with a missing reading
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/missing_readings_decimation.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/missing_readings_decimation_s$s.log 2>&1 & done
# missing_readings_delays: native condition rescored without frames with a missing reading (Sec. S4-A)
for p in 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/missing_readings_delays.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/missing_readings_delays_s$s.log 2>&1 & done
# prevalence: lock-loss prevalence on all 136 Scenario-9 passes (Sec. VI-A)
$PY analysis/prevalence.py > results/analysis/prevalence.log 2>&1 &
# beam_dynamics: epoch, net displacement per decision, inter-epoch power change
$PY analysis/beam_dynamics.py > results/analysis/beam_dynamics.log 2>&1 &
# gps_speed: GPS speed of the lock-loss passes and the others
$PY analysis/gps_speed.py > results/analysis/gps_speed.log 2>&1 &
# gps_geometry: GPS speed, bearing sweep and range of the test passes (Sec. S4)
$PY analysis/gps_geometry.py > results/analysis/gps_geometry.log 2>&1 &
# persistence_top1: top-1 of oracle-initialized persistence, Scenario 32 (Sec. III-E)
$PY analysis/persistence_top1.py > results/analysis/persistence_top1_s32.log 2>&1 &
# flat_peak: flatness of the beam profile at its peak (Sec. III-G)
$PY analysis/flat_peak.py > results/analysis/flat_peak_s9.log 2>&1 &
# setup_sweep: link-setup sweep of the lock-loss passes
$PY analysis/setup_sweep.py > results/analysis/setup_sweep_s9.log 2>&1 &
# dataset_facts: epochs, split pass counts, base-station distance
$PY analysis/dataset_facts.py > results/analysis/dataset_facts.log 2>&1 &
# drift: best-beam drift and straightness (Table S1)
$PY analysis/drift.py > results/analysis/drift.log 2>&1 &
# hold_previous: hold-previous loss under the replay rule (Table S1, Sec. S4-A)
$PY analysis/hold_previous.py > results/analysis/hold_previous.log 2>&1 &
# constant_beam: distinct predicted beams per arm and the constant-beam comparison (Sec. S12)
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/constant_beam.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/constant_beam_s$s.log 2>&1 & done
# constant_beam: the same for the caches of training seeds 1 and 2 (caption of Table S15)
for seed in s1 s2; do for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/constant_beam.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario${s}_${seed}.npz > results/analysis/constant_beam_${seed}_s$s.log 2>&1 & done; done
# full_sweep: periodic and alarm-triggered full-sweep feedback-only anchors (Sec. S17-E)
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/full_sweep.py configs/locked_multi.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/full_sweep_s$s.log 2>&1 & done
# confidence_auc: AUC of the vision arm's softmax maximum and margin for its large errors (Sec. IV-F, S17-L)
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/confidence_auc.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/post_scenario$s.npz > results/analysis/confidence_auc_s$s.log 2>&1 & done
# lock_loss_selection: lock-loss recovery reference selected on validation (Sec. S17-O); Scenario 9 with 10^5 resamples as configs/llr_scenario9.json; writes llr_scenario*.json (compared with configs/); Scenario 33 inherits the Scenario-32 selection
$PY analysis/lock_loss_selection.py configs/locked_multi.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz --resamples=100000 > results/analysis/lock_loss_selection_s9.log 2>&1 &
for p in 31:0.6 32:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/lock_loss_selection.py configs/locked_multi.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/lock_loss_selection_s$s.log 2>&1 & done
$PY analysis/lock_loss_selection.py configs/locked_multi.json dataset/scenario33 0.8 --cache=results/predictions/both_scenario33.npz --inherit=configs/llr_scenario32.json > results/analysis/lock_loss_selection_s33.log 2>&1 &
# resweep_selection: scheduled re-sweep reference selected on validation (Sec. S17-P); writes resweep_scenario*.json (compared with configs/); Scenario 33 inherits the Scenario-32 selection
$PY analysis/resweep_selection.py configs/locked_multi.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz > results/analysis/resweep_selection_s9.log 2>&1 &
for p in 31:0.6 32:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/resweep_selection.py configs/locked_multi.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/resweep_selection_s$s.log 2>&1 & done
$PY analysis/resweep_selection.py configs/locked_multi.json dataset/scenario33 0.8 --cache=results/predictions/both_scenario33.npz --inherit=configs/resweep_scenario32.json > results/analysis/resweep_selection_s33.log 2>&1 &
# tracker_slack: slack of the plain tracker on the validation passes (Sec. S5)
for p in 9:0.4 31:0.6 32:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/tracker_slack.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/tracker_slack_s$s.log 2>&1 & done
# present_readings: largest gap of a present reading below its frame's best beam (Sec. S4-A)
$PY analysis/present_readings.py > results/analysis/present_readings.log 2>&1 &
# hold_last: hold-last always-on variant, scored frames per pass and coverage at the start of a pass (Sec. S8)
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/hold_last.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/hold_last_s$s.log 2>&1 & done
# always_on_list, decimation, pessimistic_inputs: the earlier constants before and after the anchor correction (Sec. S10)
$PY analysis/always_on_list.py configs/locked_multi.json dataset/scenario9 0.4 --cache=results/predictions/sensor_preds_t0_scenario9.npz > results/analysis/always_on_list_original_s9.log 2>&1 &
$PY analysis/always_on_list.py configs/locked_multi.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz > results/analysis/always_on_list_corrected_s9.log 2>&1 &
$PY analysis/decimation.py configs/locked_multi.json dataset/scenario9 0.4 --cache=results/predictions/sensor_preds_t0_scenario9.npz > results/analysis/decimation_earlier_original_s9.log 2>&1 &
$PY analysis/decimation.py configs/locked_multi.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz > results/analysis/decimation_earlier_corrected_s9.log 2>&1 &
$PY analysis/pessimistic_inputs.py configs/locked_multi.json dataset/scenario9 0.4 --cache=results/predictions/both_scenario9.npz --d2=results/predictions/dec2_s9.npz --d4=results/predictions/dec4_s9.npz > results/analysis/pessimistic_inputs_earlier_s9.log 2>&1 &
# s35_screening, s35_missing: Scenario-35 screening (only if dataset/scenario35 is present)
if [ -f dataset/scenario35/scenario35.csv ]; then $PY analysis/s35_screening.py > results/analysis/s35_screening.log 2>&1 & $PY analysis/s35_missing.py > results/analysis/s35_missing.log 2>&1 & fi
wait

echo "== stage 2"
# controls: sign and permutation tests, blacklist grid, trial log, K-curve
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/controls.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/controls_s$s.log 2>&1 & done
# controls_gate: gate controls
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/controls_gate.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/controls_gate_s$s.log 2>&1 & done
# warmup: warm-up and trial counts, EMA gate
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/warmup.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/warmup_s$s.log 2>&1 & done
# delays: sensor, probe-outcome and commitment delays
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/delays.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/delays_s$s.log 2>&1 & done
# decimation_slack: comparator slack at each decimation (read by every_frame*)
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/decimation_slack.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/decimation_slack_s$s.log 2>&1 & done
# noise_threshold: measurement noise with a higher alarm threshold or margin
for p in 9:0.4; do s=${p%%:*}; sl=${p##*:}; $PY analysis/noise_threshold.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/noise_threshold_s$s.log 2>&1 & done
# seeds: training seeds 1 and 2 (run once per seed cache)
for seed in s1 s2; do for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/seeds.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario${s}_${seed}.npz > results/analysis/seeds_${seed}_s$s.log 2>&1 & done; done
# selection_summary: counts quoted about the selection; leave-one-site-out selections (configs/loso3_val.json)
$PY analysis/selection_summary.py > results/analysis/selection_summary.log 2>&1 &
# probe_saving: probe saving of sensing, 21% [13%, 29%]
$PY analysis/probe_saving.py results/analysis/trace_scenario9.json > results/analysis/probe_saving.log 2>&1 &
wait

echo "== stage 3"
# delays_sensor_off: delays, lags and staleness incl. the sensor-disabled controller (Fig. 6)
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/delays_sensor_off.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/delays_sensor_off_s$s.log 2>&1 & done
# delays_sensor_off_p5: the same, five decimals
for p in 9:0.4; do s=${p%%:*}; sl=${p##*:}; $PY analysis/delays_sensor_off_p5.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/delays_sensor_off_p5_s$s.log 2>&1 & done
# resweep_decimation: re-sweeping reference under decimation (read by every_frame)
for p in 9:0.4; do s=${p%%:*}; sl=${p##*:}; $PY analysis/resweep_decimation.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/resweep_decimation_s$s.log 2>&1 & done
wait

echo "== stage 4"
# resweep_pessimistic: re-sweeping reference with d-spaced inputs
for p in 9:0.4; do s=${p%%:*}; sl=${p##*:}; $PY analysis/resweep_pessimistic.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/resweep_pessimistic_s$s.log 2>&1 & done
# decimation_sensor_off: decimation against the sensor-disabled controller (read by every_frame)
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/decimation_sensor_off.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/decimation_sensor_off_s$s.log 2>&1 & done
wait

echo "== stage 5"
# every_frame: every-frame scoring of decimation (Table IV, Fig. 6 rows)
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/every_frame.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/every_frame_s$s.log 2>&1 & done
# every_frame_reselect: every-frame scoring with comparators re-selected on validation
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/every_frame_reselect.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/every_frame_reselect_s$s.log 2>&1 & done
# every_frame_missing: every-frame scoring without missing readings
for p in 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/every_frame_missing.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/every_frame_missing_s$s.log 2>&1 & done
wait

echo "== stage 6"
# competitors: confidence admission and sensor-prior fusion
$PY analysis/competitors.py configs/locked_v3.json > results/analysis/competitors.log 2>&1 &
# frontier: probe/loss frontier and convex-hull crossing (Fig. 3 caption)
for p in 9:0.4; do s=${p%%:*}; sl=${p##*:}; $PY analysis/frontier.py configs/locked_multi.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/frontier_s$s.log 2>&1 & done
# reselection_test: one-time test evaluation of the re-selected constants
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/reselection_test.py configs/locked_multi.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/reselection_test_s$s.log 2>&1 & done
# table4: Table IV, Table S27 and S28 rows, Fig. 6 decimation rows
$PY analysis/table4.py > results/analysis/table4.log 2>&1 &
# decimation_correlation: Spearman correlations across the Table IV cells (Sec. VI-C)
$PY analysis/decimation_correlation.py > results/analysis/decimation_correlation.log 2>&1 &
wait

echo "== stage 7"
# policy_family: feedback-only policies of Fig. 3 (after frontier: both write scenario9_strongfb_val.json)
for p in 9:0.4; do s=${p%%:*}; sl=${p##*:}; $PY analysis/policy_family.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/policy_family_s$s.log 2>&1 & done
wait

echo "== stage 8"
# hull_crossing: convex-hull crossings of the feedback-only policy sets and the setup-inclusive ratio (Secs. VI-A, S6-A, S6-B)
$PY analysis/hull_crossing.py > results/analysis/hull_crossing.log 2>&1 &
# fig1_labels: third-line labels of Fig. 1
$PY figures/fig1_labels.py > results/figures/fig1_labels.log 2>&1 &
# fig3_frontier: Fig. 3
$PY figures/fig3_frontier.py > results/figures/fig3_frontier.log 2>&1 &
# fig4_perpass: Fig. 4
$PY figures/fig4_perpass.py > results/figures/fig4_perpass.log 2>&1 &
# fig5_example: Fig. 5
$PY figures/fig5_example.py > results/figures/fig5_example.log 2>&1 &
# fig6_vsense: Fig. 6
$PY figures/fig6_vsense.py > results/figures/fig6_vsense.log 2>&1 &
wait

echo "== stage 9"
# missing_rescoring: the Scenario-31 to 33 analyses rescored without frames that have a missing reading
for p in 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; A="configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz"
  for t in arms attribution coverage decimation stale_observations exceedance loso invocation_ablation controls controls_gate warmup delays delays_sensor_off decimation_slack sensitivity table2 decimation_sensor_off factorial; do $PY analysis/missing_rescoring.py analysis/$t.py $A > results/analysis/missing_rescoring/${t}_s$s.log 2>&1 & done
  $PY analysis/missing_rescoring.py analysis/table3.py $A --oldcache=results/predictions/sensor_preds_t0_scenario$s.npz > results/analysis/missing_rescoring/table3_s$s.log 2>&1 &
  for sd in 1 2; do $PY analysis/missing_rescoring.py analysis/seeds.py configs/locked_v3.json dataset/scenario$s $sl --cache=results/predictions/both_scenario${s}_s$sd.npz > results/analysis/missing_rescoring/seeds_s${sd}_s$s.log 2>&1 & done
done
wait

echo "== stage 10"
# missing_rescoring_compare: paired estimates as reported against rescored (Sec. S4-A)
$PY analysis/missing_rescoring_compare.py > results/analysis/missing_rescoring_compare.log 2>&1 &
wait

$PY analysis/check_reference.py
