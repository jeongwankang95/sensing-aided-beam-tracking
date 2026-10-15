# DeepSense 6G Scenarios 9, 31, 32 and 33: data and pass lists

The DeepSense 6G data are not redistributed here. Download Scenarios 9, 31, 32 and 33 from the DeepSense 6G website
(<https://www.deepsense6g.net>) under its terms of use and extract each into its folder, so that the scenario's CSV file and
its `unit1/` folder sit next to the pass lists below:

| Folder | Scenario CSV | Read from the data |
|---|---|---|
| `dataset/scenario9/` | `scenario9.csv` | the 64-beam power files `unit1/mmWave_data/*.txt` named in the CSV column `unit1_pwr_60ghz` |
| `dataset/scenario31/` | `scenario31_dev.csv` | the same |
| `dataset/scenario32/` | `scenario32_dev.csv` | the same |
| `dataset/scenario33/` | `scenario33_dev.csv` | the same |

Only the measured beam powers are read (with the CSV columns `index`, `seq_index` and `unit1_pwr_60ghz`); camera, radar,
LiDAR and GPS files are not needed, because the sensing models' predictions are cached in `results/predictions/`. To keep the
data elsewhere, link them in, for example `ln -s /data/scenario9/unit1 dataset/scenario9/unit1` and likewise the CSV.

Some power vectors of Scenarios 31 to 33 have missing entries. The replay treats a missing reading as zero power, so a frame
whose served beam has no reading scores the 60-dB cap of the loss (supplementary Sec. S4-A).

## Pass lists (part of the repository)

Each file lists windows of eight observed and six future frames with their pass (`seq_index`), as in the companion
repository; the replay uses only the pass index. Passes never straddle two lists.

| File | Windows | Passes | Use |
|---|---|---|---|
| `scenario9/test_seqs_RA.csv` | 774 | 28 | Scenario 9 test passes |
| `scenario9/val_seqs_RA.csv` | 499 | 17 | Scenario 9 validation passes (constant selection) |
| `scenario31/test_seqs_RA.csv` | 1,259 | 11 | Scenario 31 test passes |
| `scenario31/val_seqs_RA.csv` | 781 | 6 | Scenario 31 validation passes |
| `scenario32/test_seqs_RA.csv` | 780 | 3 | Scenario 32 test passes |
| `scenario32/val_seqs_RA.csv` | 354 | 2 | Scenario 32 validation passes |
| `scenario33/test_seqs_RA.csv` | 3,747 | 18 | Scenario 33: every pass is a test pass (no validation) |
