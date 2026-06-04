# Poisson Optimizer Comparison

Fixed setting:

- epochs: 1000
- n_interior: 1024
- n_boundary: 256
- hidden_dim: 100
- hidden_layers: 4
- lr: 0.001
- lambda_bc: 1.0
- seeds: 0, 1, 2

| method | seed | L2 relative error | PDE loss | BC loss | runtime sec |
|---|---:|---:|---:|---:|---:|
| Adam | 0 | 1.164399e-01 | 1.389011e-02 | 1.825187e-02 | 22.56 |
| Adam->L-BFGS | 0 | 4.225509e-02 | 3.452691e-03 | 1.929796e-03 | 33.14 |
| Adam | 1 | 9.124102e-02 | 5.243675e-03 | 1.370436e-02 | 21.90 |
| Adam->L-BFGS | 1 | 2.415431e-02 | 2.488705e-03 | 5.770297e-04 | 35.38 |
| Adam | 2 | 1.090553e-01 | 7.005326e-03 | 1.385669e-02 | 19.43 |
| Adam->L-BFGS | 2 | 4.867029e-02 | 1.784475e-03 | 3.722436e-03 | 32.15 |
