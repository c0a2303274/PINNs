# 2D Burgers width/time comparison

Pilot, seed 0 by default; one trial is not a robustness result.
Same total training-loop budget per case. Different widths have different parameter counts.
Fluctuation error uses the reference minus its uniform background in the denominator.

| case | mode | hidden_widths | parameter_count_per_model | l2_relative_error | fluctuation_l2_relative_error | pde_rmse | ic_rmse | max_interface_rmse | training_sec | completed_epochs | inference_time_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | global | 128,128,128,128,128 | 66818 | 0.0154176 | 0.147619 | 0.00717121 | 0.0107201 | 0 | 600.014 | 18736 | 1.19205 |
| B | global | 256,128,64,128,256 | 84034 | 0.0192305 | 0.184126 | 0.00787401 | 0.0112819 | 0 | 600.012 | 19703 | 0.824785 |
| C | marching | 128,128,128,128,128 | 66818 | 0.0807297 | 0.772963 | 0.00772407 | 0.0474419 | 0.00875603 | 600.064 | 20314 | 2.54368 |
| D | marching | 256,128,64,128,256 | 84034 | 0.0880946 | 0.843479 | 0.0054871 | 0.0525734 | 0.00784929 | 600.082 | 19966 | 2.47019 |
