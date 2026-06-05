# Poisson Optimizer Comparison

Fixed setting:

- epochs: 1000000
- max_runtime_sec per run: 600.0
- n_interior: 1024
- n_boundary: 256
- hidden_dim: 100
- hidden_layers: 4
- lr: 0.001
- lambda_bc: 1.0
- seeds: 0, 1, 2

| method | seed | L2 relative error | PDE loss | BC loss | runtime sec | epochs | lbfgs steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adam | 0 | 1.730901e-02 | 3.237131e-03 | 6.756101e-05 | 600.00 | 103490 |  |
| Adam->L-BFGS | 0 | 6.800325e-04 | 3.374550e-05 | 8.032647e-07 | 323.72 | 50567 | 1060 |
| Adam | 1 | 2.904236e-03 | 1.058449e-04 | 5.192193e-06 | 600.00 | 103180 |  |
| Adam->L-BFGS | 1 | 7.711186e-04 | 1.909457e-05 | 7.621805e-07 | 317.01 | 51645 | 1180 |
| Adam | 2 | 4.562751e-03 | 9.249001e-05 | 8.005471e-06 | 600.00 | 103744 |  |
| Adam->L-BFGS | 2 | 4.035247e-04 | 1.595557e-05 | 2.776012e-07 | 319.10 | 51112 | 863 |
