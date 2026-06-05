# PINNs Thesis Workspace

This directory is the working repository for the staged PINNs thesis workflow.

## Current scope

1. Poisson boundary-value problem on `[-1, 1] x [-1, 1]`
2. 1D wave equation with standard PINNs
3. Fourier features and GF-PINNs on the same wave setup
4. Burgers equation after the simpler stages are understood

## Current files

- `train_poisson_pinn.py`: baseline Poisson PINN training entry point
- `poisson_problem.py`: exact solution, source term, and PDE residual
- `pinn_model.py`: MLP model definition
- `append_research_log.py`: append structured entries to `research_log.md`
- `research_log.md`: persistent experiment and decision log

## Local run

Default W&B behavior is offline logging. That keeps runs local unless you later sync them.

Adam only:

```powershell
python .\train_poisson_pinn.py --epochs 1000 --n-interior 1024 --n-boundary 256 --output-dir .\outputs\poisson_baseline_1k
```

Adam followed by L-BFGS:

```powershell
python .\train_poisson_pinn.py --epochs 1000 --n-interior 1024 --n-boundary 256 --lbfgs-steps 200 --output-dir .\outputs\poisson_adam_lbfgs
```

Run the current optimizer comparison plan:

```powershell
python .\run_poisson_comparison.py --seeds 0,1,2 --epochs 1000 --n-interior 1024 --n-boundary 256 --lbfgs-steps 200
```

This writes `outputs/poisson_optimizer_comparison/summary.md` and `summary.csv`.

Run the same comparison with an approximate one-hour total budget:

```powershell
python .\run_poisson_comparison.py --seeds 0,1,2 --epochs 1000000 --n-interior 1024 --n-boundary 256 --lbfgs-steps 100000 --total-runtime-sec 3600 --wandb-mode disabled --output-root .\outputs\poisson_optimizer_comparison_1h
```

`--total-runtime-sec 3600` is split across the six runs, so each Adam or Adam->L-BFGS run gets about 10 minutes.
For Adam->L-BFGS timed runs, the default is to spend half of the run budget on Adam and leave the remaining half for L-BFGS.

Disable W&B completely:

```powershell
python .\train_poisson_pinn.py --wandb-mode disabled
```

Run online with an explicit group and tags:

```powershell
python .\train_poisson_pinn.py --wandb-mode online --wandb-group poisson-adam-lbfgs --wandb-tags "poisson,baseline,lbfgs"
```

## W&B usage

Install dependencies:

```powershell
pip install -r requirements.txt
```

Offline-first workflow:

```powershell
$env:WANDB_MODE="offline"
python .\train_poisson_pinn.py
wandb sync .\outputs\poisson_baseline\wandb\latest-run
```

Each run records:

- config for PDE, optimizer, network, seed, and sampling counts
- training metrics for total, PDE, and boundary losses
- final `l2_relative_error` and runtime
- artifacts for `model.pt`, `metrics.json`, and the saved plots

## Log a run

```powershell
python .\append_research_log.py --stage poisson-baseline --pde Poisson --setting "Omega=[-1,1]^2, Dirichlet boundary" --changes "Updated training condition" --done "Ran one experiment" --result "Recorded metrics under outputs" --wandb-run "pinns-thesis/poisson-baseline/abc123" --next-step "Tune the next variable"
```

## Git workflow

Use this repository as the source of truth.

1. Edit and test locally.
2. Commit meaningful checkpoints.
3. Push to GitHub.
4. Pull or clone from Colab for GPU runs.

## Colab quick start

### Open from GitHub

1. Go to `https://colab.research.google.com/`
2. Open the `GitHub` tab
3. Paste `https://github.com/c0a2303274/PINNs`
4. Select `poisson_baseline_colab.ipynb`
5. After the notebook opens, switch runtime to GPU:
   `Runtime` -> `Change runtime type` -> `T4 GPU` or another GPU
6. Run the cells from top to bottom

### Direct notebook link

After GitHub finishes indexing the notebook, this URL should open it directly in Colab:

`https://colab.research.google.com/github/c0a2303274/PINNs/blob/main/poisson_baseline_colab.ipynb`

### Manual Colab cells

If you want to start from an empty Colab notebook, use:

```python
!git clone https://github.com/c0a2303274/PINNs.git
%cd PINNs
!pip install -r requirements.txt
!python train_poisson_pinn.py --epochs 1000 --n-interior 1024 --n-boundary 256 --output-dir outputs/poisson_baseline_1k
```

For an Adam to L-BFGS run on Colab:

```python
!python train_poisson_pinn.py --epochs 1000 --n-interior 1024 --n-boundary 256 --lbfgs-steps 200 --output-dir outputs/poisson_adam_lbfgs
```

### Save outputs from Colab

The current notebook writes results under `outputs/`. If you need to keep outputs after the Colab session ends, either:

1. Download the files manually from the Colab file pane
2. Mount Google Drive and copy `outputs/` there

### Log runs after Colab execution

```python
!python append_research_log.py --stage poisson-baseline --pde Poisson --setting "Omega=[-1,1]^2, epochs=1000, n_interior=1024, n_boundary=256" --changes "Ran baseline on Colab" --done "Completed one GPU run" --result "See outputs/poisson_baseline_1k/metrics.json" --wandb-run "add-run-path-or-url-here" --next-step "Compare runtime and error with local run"
```
