# DeepSense 6G Scenarios 9, 31, 32 and 33: data and pass lists

The DeepSense 6G data are not redistributed here. Download Scenarios 9, 31, 32 and 33 from the DeepSense 6G website
(<https://www.deepsense6g.net>) under its terms of use and extract each into its folder, so that the scenario's CSV file and
its `unit1/` and `unit2/` folders sit next to the pass lists below:

| Folder | Scenario CSV | Read from the data |
|---|---|---|
| `dataset/scenario9/` | `scenario9.csv` | the 64-beam power files `unit1/mmWave_data/*.txt` named in the CSV column `unit1_pwr_60ghz` |
| `dataset/scenario31/` | `scenario31_dev.csv` | the same |
| `dataset/scenario32/` | `scenario32_dev.csv` | the same |
| `dataset/scenario33/` | `scenario33_dev.csv` | the same |

Almost every script reads only the measured beam powers (with the CSV columns `index`, `seq_index` and `unit1_pwr_60ghz`).
A few also read the timestamps and positions:

- the CSV timestamp column (`time_stamp[UTC]` in Scenario 9, `time_stamp` elsewhere): `analysis/dataset_facts.py`,
  `beam_dynamics.py`, `gps_speed.py`, `gps_geometry.py` and `kinematics.py`;
- the base-station position (`unit1_loc`, a file under `unit1/`) and the vehicle positions (`unit2_loc`, and in Scenario 9 the
  calibrated `unit2_loc_cal`, files under `unit2/`): `analysis/dataset_facts.py`, `gps_speed.py`, `gps_geometry.py` and
  `kinematics.py`.

Camera, radar and LiDAR files are not needed, because the sensing models' predictions are cached in `results/predictions/`.
To keep the data elsewhere, link them in, for example `ln -s /data/scenario9/unit1 dataset/scenario9/unit1` (likewise `unit2`
and the CSV).

Some power vectors of Scenarios 31 to 33 have missing entries. The replay treats a missing reading as zero power, so a frame
whose served beam has no reading scores the 60-dB cap of the loss (supplementary Sec. S4-A; `analysis/missing_readings_*.py`
and `analysis/missing_rescoring.py` rescore without those frames).

## Scenario 35 (optional)

`analysis/s35_screening.py` and `analysis/s35_missing.py` screen DeepSense 6G Scenario 35 as a candidate site (supplementary
Sec. S17-M). They need `dataset/scenario35/` with `scenario35.csv` (columns `abs_idx`, `seq_index`, `timestamp`, `unit1_pwr`,
`unit2_gps`), the power files and `unit1/gps/unit1_static_loc.txt`, and the Scenario-9 positions
(`unit1/GPS_data/gps_location.txt` and `unit2_loc`). `analysis/run_analysis.sh` runs them only if
`dataset/scenario35/scenario35.csv` exists. No pass list is needed: the screening uses every Scenario-35 pass.

## Pass lists (part of the repository)

Each file lists windows of eight observed and six future frames with their pass (`seq_index`), as in the companion
repository; the replay uses only the pass index. Passes never straddle two lists, and every pass of the scenario CSV that is
in neither list is a training pass. Training passes are read only where the paper counts them or uses all passes:
`analysis/dataset_facts.py`, `prevalence.py` (all 136 Scenario-9 passes), `drift.py` and `hold_previous.py` (Table S1),
`present_readings.py`, and the Scenario-9 columns of `s35_screening.py`.

| File | Windows | Passes | Use |
|---|---|---|---|
| `scenario9/test_seqs_RA.csv` | 774 | 28 | Scenario 9 test passes |
| `scenario9/val_seqs_RA.csv` | 499 | 17 | Scenario 9 validation passes (constant selection) |
| `scenario31/test_seqs_RA.csv` | 1,259 | 11 | Scenario 31 test passes |
| `scenario31/val_seqs_RA.csv` | 781 | 6 | Scenario 31 validation passes |
| `scenario32/test_seqs_RA.csv` | 780 | 3 | Scenario 32 test passes |
| `scenario32/val_seqs_RA.csv` | 354 | 2 | Scenario 32 validation passes |
| `scenario33/test_seqs_RA.csv` | 3,747 | 18 | Scenario 33: every pass is a test pass (no validation) |
