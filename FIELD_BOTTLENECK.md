# Field-input bottleneck comparison

This is a **supervised field-to-field prediction pilot**, not the existing coordinate-input PINN.
Do not compare its errors directly with earlier PINN tables: data, supervision and evaluation sets differ.

## Question and fixed conditions

- Compare abrupt compression and gradual halving/doubling with identical input/output dimensions, hidden-layer count and minimum width.
- Each input contains both velocity components over a periodic square grid. A 64x64 grid has 4096 spatial points and **8192 input values**, not 4096.
- A latent width of 128 means 128 continuous features, not 128 classes.
- Input: complete initial field at t=0. Output: complete field at t=0.25. No identity reconstruction task and no time rollout.
- Reference family: the existing unforced periodic 2D Burgers analytic solution, viscosity 0.01, background velocity (0.5,0.25), amplitude uniformly in [0.3,0.8), two phase shifts uniformly in [0,1).
- Teacher data are analytic fields. Loss is MSE of normalized fluctuations (subtract the known background and divide by 0.1). No PDE/IC/BC loss or HardNet layer is used.
- Separate training, validation and test initial conditions with one stored data seed; the same samples are used in both cases. This is interpolation within a restricted smooth three-parameter solution family, not arbitrary-field generalization.
- Adam, lr=0.001, tanh in hidden layers, linear output, seed 0, batch size 32. Final checkpoint is evaluated, without best-test selection.
- Parameter counts are NOT matched. Fixed-time comparisons mix capacity, optimization speed and convergence; do not attribute differences solely to compression shape.

## Local pilot

```powershell
python -m unittest test_field_bottleneck -v
python run_field_bottleneck_comparison.py --grid 16 --bottleneck 128 --runtime-sec-per-case 60 --device cpu --output-root results/field_bottleneck_pilot_20261008
```

- 16x16 spatial points, two components, D=512.
- Abrupt: 512,128,128,128,512.
- Gradual: 512,256,128,256,512.
- 512 training, 128 validation and 128 test fields. The test set is never used for gradient updates or checkpoint selection.
- A single short trial is not evidence of statistical equivalence or a robust ranking.

## GPU readiness and planned one-day comparison

First check the full-size models for approximately one minute each:

```bash
conda activate pinns
python -m unittest test_field_bottleneck -v
python run_field_bottleneck_comparison.py --grid 64 --bottleneck 128 --train-fields 2048 --validation-fields 256 --test-fields 256 --runtime-sec-per-case 60 --device cuda --output-root outputs/field_bottleneck_gpu_check_$(date +%Y%m%d_%H%M%S)
```

After checking finite losses, CUDA memory and output files:

```bash
python -u run_field_bottleneck_comparison.py --grid 64 --bottleneck 128 --train-fields 2048 --validation-fields 256 --test-fields 256 --runtime-sec-per-case 43200 --device cuda --output-root outputs/field_bottleneck_24h_$(date +%Y%m%d_%H%M%S)
```

The training-loop budgets are 12 hours per case, sequentially, about 24 hours total. Setup and final evaluation add overhead. One update can slightly exceed the deadline. `--max-updates` is a second cap; `stop_reason` records which cap was reached. This command is **prepared, not executed** by the local pilot. No tmux requirement is imposed.

At D=8192, gradual hidden widths are 4096,2048,1024,512,256,128,256,512,1024,2048,4096; the abrupt control has eleven hidden layers of width 128. Both have the same depth and bottleneck. The large dense model is intentionally much more expensive. This does not isolate bottleneck shape from parameter count. If matching parameter counts is required, design a separate control rather than relabel this comparison.

## Outputs and reproducibility

The full-size parameter counts are 2,270,592 (abrupt) and 89,480,832 (gradual), about 39 times different. Check GPU memory before the long run. Dense gradual compression is not a memory-saving alternative in this configuration.

- `config.json`, `data_parameters.npz`, `summary.csv`, `summary.json`, `baselines.json`.
- Per case: `metrics.json`, `history.json`, `latest.pt`, `test_examples.npz`.
- Checkpoints save every 300 seconds and at completion. There is no resume flag yet; checkpoint presence is not a promise of automatic resumption.
- Existing output directories are refused, to prevent overwriting earlier runs.
- Report full-field relative L2 AND fluctuation-relative L2, validation/test error, parameter count, updates, training time, inference time for one whole field and GPU peak allocated memory.
- Persistence (output the input field unchanged) and uniform-background baselines are evaluated on the same test cases.
- Share summaries/config/history and compact test examples in Git. Avoid force-adding entire `outputs/` or large GPU checkpoints.

Next: check whether the full-size setup trains and whether longer training changes the validation/test error. Do not present the expectation of a small difference as an observed result.
