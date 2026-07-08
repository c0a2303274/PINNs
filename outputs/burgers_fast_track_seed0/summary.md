# Burgers fast-track comparison

| config | mode | seed | L2 relative error | PDE loss | IC loss | BC loss | runtime sec | epochs | lbfgs steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hard_icbc_adam | hard-icbc | 0 | 4.331720e-02 | 4.713095e-05 | 0.000000e+00 | 2.534733e-15 | 7200.00 | 758209 | 0 |
| hard_icbc_lbfgs | hard-icbc | 0 | 4.290432e-02 | 9.395084e-06 | 0.000000e+00 | 2.505528e-15 | 3603.47 | 381653 | 119 |
| hard_icbc_lower_lr | hard-icbc | 0 | 4.294771e-02 | 3.169104e-06 | 0.000000e+00 | 2.473336e-15 | 7200.01 | 756519 | 0 |
