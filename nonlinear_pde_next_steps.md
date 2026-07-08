# Nonlinear PDE next steps

## Research storyline

Poisson PINNs experiments are now treated as preliminary baseline work. They established the implementation, logging, evaluation, and optimizer-comparison workflow, but the tested settings plateaued around `1e-4` L2 relative error. The next phase moves to nonlinear PDEs and studies whether hard-constrained methods can reduce constraint error and improve stability.

## First target PDE

Use the 1D viscous Burgers equation as the first nonlinear PDE:

```text
u_t + u u_x = nu u_xx
x in [-1, 1], t in [0, 1]
u(x, 0) = -sin(pi x)
u(-1, t) = u(1, t) = 0
nu = 0.01 / pi
```

The first implementation uses a finite-difference reference solution for L2 relative error. In slides and reports, call this a numerical reference solution, not an analytic solution.

## Baseline command

Start with a short smoke run:

```bash
python train_burgers_pinn.py --epochs 100 --n-interior 256 --n-initial 64 --n-boundary 64 --wandb-mode disabled --output-dir outputs/burgers_smoke
```

Then run a first serious baseline:

```bash
python train_burgers_pinn.py --epochs 20000 --n-interior 4096 --n-initial 512 --n-boundary 512 --hidden-dim 128 --hidden-layers 5 --lr 0.001 --lbfgs-steps 5000 --seed 0 --wandb-mode offline --output-dir outputs/burgers_baseline_seed0
```

## Comparison plan

| step | method | purpose | key metrics |
|---|---|---|---|
| 1 | standard PINNs | nonlinear PDE baseline | L2 relative error, PDE loss, IC loss, BC loss, runtime |
| 2 | hard IC/BC ansatz | first hard-constraint variant | IC/BC violation and L2 error |
| 3 | HardNet / HardNet++ study | identify implementable constraint layer | constraint satisfaction, stability |
| 4 | ECO study | decide if energy constraint is relevant | boundedness and long-time behavior |

## Two-hour comparison command

This runs standard soft-constraint PINNs and a hard IC/BC ansatz for the same seed. The hard version uses
`u(t,x)=(1-t)u0(x)+t(1-x^2)N(t,x)`, so it satisfies the initial and boundary conditions by construction.

```bash
python run_burgers_constraint_comparison.py --seeds 0 --runtime-sec 7200 --epochs 1000000 --n-interior 4096 --n-initial 512 --n-boundary 512 --hidden-dim 128 --hidden-layers 5 --lr 0.001 --wandb-mode offline --output-root outputs/burgers_constraint_comparison_2h
```

After seed 0 is confirmed, run three seeds:

```bash
python run_burgers_constraint_comparison.py --seeds 0,1,2 --runtime-sec 7200 --epochs 1000000 --n-interior 4096 --n-initial 512 --n-boundary 512 --hidden-dim 128 --hidden-layers 5 --lr 0.001 --wandb-mode offline --output-root outputs/burgers_constraint_comparison_2h_seeds012
```

## Immediate research tasks

1. Run the HardNet vector-field comparison for seeds 0,1,2 on GPU.
2. Analyze whether HardNet keeps the constraint violation near machine precision while maintaining L2 accuracy.
3. Run the Burgers fast-track comparison for hard-IC/BC Adam, hard-IC/BC Adam->L-BFGS, and lower learning rate.
4. Implement a HardNet++-style nonlinear equality correction loop after the Burgers fast-track result is collected.
5. Prepare the midterm resume using `midterm_resume_plan.md` and `hardnet_transition_plan.md`.
