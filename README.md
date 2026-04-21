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

```powershell
python .\train_poisson_pinn.py --epochs 1000 --n-interior 1024 --n-boundary 256 --output-dir .\outputs\poisson_baseline_1k
```

## Log a run

```powershell
python .\append_research_log.py --stage poisson-baseline --pde Poisson --setting "Omega=[-1,1]^2, Dirichlet boundary" --changes "Updated training condition" --done "Ran one experiment" --result "Recorded metrics under outputs" --next-step "Tune the next variable"
```

## Git workflow

Use this repository as the source of truth.

1. Edit and test locally.
2. Commit meaningful checkpoints.
3. Push to GitHub.
4. Pull or clone from Colab for GPU runs.

## Colab quick start

After pushing to GitHub, use a Colab cell like this:

```python
!git clone <YOUR_REPO_URL>
%cd <YOUR_REPO_NAME>
!python train_poisson_pinn.py --epochs 1000 --n-interior 1024 --n-boundary 256 --output-dir outputs/poisson_baseline_1k
```
