#!/bin/bash
# Regenerates every analysis output behind the paper from the stored predictions, then compares them with
# results/reference/. Needs DeepSense 6G Scenarios 9, 31, 32 and 33 in dataset/ (see dataset/README.md).
# Stages run in order because later scripts read earlier outputs; within a stage the scenarios run in parallel.
# Run from the repository root: bash analysis/run_analysis.sh   (PYTHON overrides the interpreter)
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
mkdir -p results/analysis/lag results/analysis/lag_p5

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
# loso: leave-one-site-out constant selection
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
# frontier: probe/loss frontier and convex-hull crossing (Fig. 3)
for p in 9:0.4; do s=${p%%:*}; sl=${p##*:}; $PY analysis/frontier.py configs/locked_multi.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/frontier_s$s.log 2>&1 & done
# reselection_test: one-time test evaluation of the re-selected constants
for p in 9:0.4 31:0.6 32:0.8 33:0.8; do s=${p%%:*}; sl=${p##*:}; $PY analysis/reselection_test.py configs/locked_multi.json dataset/scenario$s $sl --cache=results/predictions/both_scenario$s.npz > results/analysis/reselection_test_s$s.log 2>&1 & done
wait

$PY analysis/check_reference.py
