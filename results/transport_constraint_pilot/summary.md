# HardNet and HardNet++ PDE bridge comparison

| constraint | method | seed | L2 relative error | PDE RMSE | max violation | runtime sec | epochs | inference ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| linear | soft | 0 | 1.602635e-03 | 1.370559e-04 | 1.018703e-03 | 1800.00 | 376764 | 0.115 |
| linear | hardnet | 0 | 1.098703e-03 | 5.017070e-04 | 2.384186e-07 | 1800.01 | 128016 | 0.385 |
| circle | soft | 0 | 1.817017e-03 | 3.319319e-04 | 3.701508e-03 | 1800.00 | 375088 | 0.112 |
| circle | hardnetpp | 0 | 1.402906e+00 | 4.464694e-03 | 1.192093e-07 | 1800.05 | 24258 | 9.528 |
