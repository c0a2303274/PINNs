# Burgers constraint comparison

| method | seed | L2 relative error | PDE loss | IC loss | BC loss | runtime sec | epochs | lbfgs steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| soft | 0 | 4.314643e-02 | 1.446843e-05 | 1.292169e-06 | 1.498731e-06 | 6831.85 | 1000000 | 0 |
| hard-icbc | 0 | 4.276393e-02 | 6.784414e-05 | 0.000000e+00 | 2.587685e-15 | 7200.00 | 765951 | 0 |
