# Poisson Optimizer Comparison

Fixed setting:

- epochs: 1000000
- max_runtime_sec per run: 2400.0
- n_interior: 4096
- n_boundary: 1024
- hidden_dim: 128
- hidden_layers: 5
- lr: 0.001
- lambda_bc: 1.0
- dtype: float64
- eval_grid_size: 201
- seeds: 0, 1, 2

| method | seed | L2 relative error | PDE loss | BC loss | runtime sec | epochs | lbfgs steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adam | 0 | 7.000362e-03 | 9.233681e-04 | 1.550143e-05 | 2400.00 | 180505 |  |
| Adam->L-BFGS | 0 | 5.899649e-04 | 6.162542e-06 | 7.509410e-07 | 1226.00 | 78195 | 934 |
| Adam | 1 | 1.783711e-03 | 1.755474e-04 | 2.249122e-06 | 2400.01 | 158110 |  |
| Adam->L-BFGS | 1 | 4.751285e-04 | 3.706930e-06 | 4.440231e-07 | 1226.84 | 79260 | 966 |
| Adam | 2 | 1.362792e-02 | 2.448356e-03 | 3.591630e-05 | 2400.01 | 158038 |  |
| Adam->L-BFGS | 2 | 4.195673e-04 | 3.833116e-06 | 3.735607e-07 | 1224.73 | 81711 | 1194 |
