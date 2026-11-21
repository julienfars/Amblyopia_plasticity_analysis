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
  bv_1_results_trials.xlsx        patient group, trial-level data (IDs P01-P21)
  bv_2_results_thresholds.xlsx    patient group, per-condition thresholds
  bv_3_results_Diff.xlsx          patient group, pre/post training differences
  bv_4_results_ABC_diff.xlsx      patient group, area-under-curve differences
  VR_imp2.xlsx                    VR training improvement, patient group
  VR_imp_con2.xlsx                VR training improvement, control group

Data/Controls/Refined/
  bv_1_results_trials.xlsx        control group, trial-level data (IDs C01-C16)
  bv_2_results_thresholds.xlsx    control group, per-condition thresholds
  bv_3_results_Diff.xlsx          control group, pre/post training differences
  bv_4_results_ABC_diff.xlsx      control group, area-under-curve differences
  bv_4_results_ABC_diff_CONTROL.xlsx  duplicate, mirrors the patient folder's layout
```

## Participant IDs

Participant IDs are strings: `P01`, `P02`, ... for patients (amblyopia group)
and `C01`, `C02`, ... for controls. This is the same format as the real data
released on Zenodo, so the notebooks run unchanged on either. The synthetic IDs
do **not** correspond to real participants: synthetic `P05` is random data, not
participant 5.

When processing raw data, `Analysis/BV_processing.py` builds these IDs from the
numbers in raw file and folder names (`format_participant_id`, e.g. `sub-005` →
`P05`).

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