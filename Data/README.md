# Synthetic example data

This folder contains **fully synthetic** data, generated to mimic the structure
of the real trial-level data so the analysis notebooks/scripts can be run and
demonstrated without exposing real participant data (GDPR).

None of the values here were derived from real participants — they are
randomly generated, calibrated to plausible ranges. Do not use for any
scientific interpretation.

## Contents

```
Data/Refined/
  bv_1_results_trials.xlsx        patient group, trial-level data (fake IDs 101-121)
  bv_2_results_thresholds.xlsx    patient group, per-condition thresholds
  bv_3_results_Diff.xlsx          patient group, pre/post training differences
  bv_4_results_ABC_diff.xlsx      patient group, area-under-curve differences
  VR_imp2.xlsx                    VR training improvement, patient group
  VR_imp_con2.xlsx                VR training improvement, control group

Data/Controls/Refined/
  bv_1_results_trials.xlsx        control group, trial-level data (fake IDs 201-216)
  bv_2_results_thresholds.xlsx    control group, per-condition thresholds
  bv_3_results_Diff.xlsx          control group, pre/post training differences
  bv_4_results_ABC_diff.xlsx      control group, area-under-curve differences
  bv_4_results_ABC_diff_CONTROL.xlsx  duplicate, mirrors the patient folder's layout
```

Fake participant IDs (101+ for patients, 201+ for controls) are chosen well
outside the real ID ranges so they can't be confused with real participants.

## How it was generated

- Sheet names, column names, and dtypes match the real files.
- `bv_2` is a genuine subset of `bv_1` (rows where `AMB_eye == 'MEAN'`), and
  derived columns (`Frequency_cpd`, `Sensitivity`, `logBP`, `Noise100`, the
  Snellen/logMAR visual-angle metrics) are computed with the real functions in
  `Analysis/BV_processing.py`, just fed synthetic inputs — same as the real
  pipeline.
- `bv_3` (pre/post diff) and `bv_4` (ABC) values are independently sampled
  from distributions calibrated to the real files' min/max/mean, with
  `Diff = posttraining - pretraining` enforced, and a mild built-in
  "amblyopic eye worse, training improves" effect so plots look directionally
  sensible.
- Trial-level rows in `bv_1` are plausibly-shaped noise, **not** simulated
  adaptive staircases — fine for exercising code paths and demos, but don't
  expect realistic reversal-by-reversal trial dynamics if you inspect that
  level directly.

The analysis notebooks already point `path_data_ref` / `path_data_ref_con` at
`Data/Refined` / `Data/Controls/Refined` (relative to `Analysis/`), so they run
against this synthetic data out of the box — no path changes needed.