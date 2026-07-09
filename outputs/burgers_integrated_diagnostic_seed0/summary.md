# Burgers integrated comparison

| config | mode | seed | L2 relative error | PDE loss | IC loss | BC loss | runtime sec | epochs | lbfgs steps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bounded_amp2_lbfgs | bounded-hard-icbc | 0 | 3.995239e-02 | 6.487312e-02 | 0.000000e+00 | 2.496340e-15 | 3824.65 | 361371 | 12034 |
| hard_icbc_focused_lbfgs | hard-icbc | 0 | 4.291571e-02 | 9.530236e-06 | 0.000000e+00 | 2.558454e-15 | 3602.67 | 370258 | 108 |
