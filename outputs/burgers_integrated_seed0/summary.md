# Burgers integrated comparison

| config | mode | seed | L2 relative error | PDE loss | IC loss | BC loss | runtime sec | epochs | lbfgs steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| soft | soft | 0 | 4.314643e-02 | 1.446843e-05 | 1.292169e-06 | 1.498731e-06 | 6815.16 | 1000000 | 0 |
| hard_icbc_lbfgs | hard-icbc | 0 | 4.299398e-02 | 2.578097e-05 | 0.000000e+00 | 2.577591e-15 | 3605.94 | 387664 | 211 |
| bounded_hard_icbc_lbfgs | bounded-hard-icbc | 0 | 1.761268e-01 | 2.652143e+00 | 0.000000e+00 | 2.559882e-15 | 4762.49 | 367385 | 36329 |
