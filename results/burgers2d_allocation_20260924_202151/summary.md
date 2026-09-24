# 2D Burgers width/time/allocation comparison

Pilot, seed 0 by default; one trial is not a robustness result.
Same total training-loop budget per case. Different widths have different parameter counts.
C: equal window budgets; E: extra first-window time, less time for later windows (same width as C).
Fluctuation error uses the reference minus its uniform background in the denominator.

| case | mode | hidden_widths | window_budgets_sec | parameter_count_per_model | l2_relative_error | initial_time_l2_relative_error | final_time_l2_relative_error | fluctuation_l2_relative_error | pde_rmse | ic_rmse | max_interface_rmse | training_sec | completed_epochs | inference_time_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C | marching | 128,128,128,128,128 | [120.0, 120.0, 120.0, 120.0, 120.0] | 66818 | 0.0768913 | 0.1105 | 0.055753 | 0.736212 | 0.00759299 | 0.0462803 | 0.00788147 | 600.057 | 20428 | 2.65397 |
| E | marching | 128,128,128,128,128 | [300.0, 75.0, 75.0, 75.0, 75.0] | 66818 | 0.041855 | 0.0443123 | 0.0577343 | 0.400749 | 0.00580999 | 0.0182375 | 0.0124583 | 600.082 | 20116 | 2.55108 |
