# AKT + ALSO Experiment Results

Generated (UTC): 2026-07-13 17:41:36

## Evaluation Criteria

Rank evaluated configurations by higher student-level mean AUC, then lower student-level standard deviation, while monitoring overall dataset AUC and range.

## Completed Runs

| Run | ALSO | Batch | pi_lr | pi_decay | Loss scale | Alpha | Mode | Overall AUC | Student mean | Student std | Student range |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| akt_student_also_v2_baseline_bs64 | 0 | 64 | 0.001 | 0.01 | - | - | - | 0.761477 | 0.628480 | 0.201374 | 1.000000 |
| akt_also_full_20260713_225853_42622 | 1 | 64 | 3e-05 | 0.01 | - | 1.0 | optimistic | 0.692870 | 0.593758 | 0.212514 | 1.000000 |
| akt_also_full_20260713_225853_42622 | 1 | 64 | 0.001 | 0.01 | - | 1.0 | optimistic | 0.689596 | 0.593041 | 0.208252 | 1.000000 |

## Pending Or Incomplete Runs

- `akt_also_stage1_pi_lr_auto_20260713_232100_42622` / `pilr1e4_bs64_42_0_0.2_256_512_8_4_0.0001_64_100_0_0_1_student_id_None_optimistic_1.0_None_None_0.0001_0.01_None_0`: evaluation_missing
- `akt_also_stage1_pi_lr_auto_20260713_232100_42622` / `pilr3e4_bs64_42_0_0.2_256_512_8_4_0.0001_64_100_0_0_1_student_id_None_optimistic_1.0_None_None_0.0003_0.01_None_0`: evaluation_missing

## Notes

- `overall_dataset_auc` is calculated from the saved student-level prediction artifact and is tracked alongside the model-level test AUC in `all_results.json`.
- A student range of 1.0 indicates that at least one student has AUC 0.0 and another has AUC 1.0; interpret it together with sample counts and IQR.
- Historical `akt_abl_*` runs from the old implementation are deliberately excluded.
