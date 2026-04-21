# Research Log

## 2026-04-21

- Stage: planning
- PDE: Poisson boundary-value problem
- Problem setting: Omega = [-1,1] x [-1,1], exact solution u(x,y) = sin(pi x) sin(pi y), Dirichlet boundary condition u = 0, source term f(x,y) = 2 pi^2 sin(pi x) sin(pi y).
- Methods or changes: Started a local thesis workspace and added a baseline PINN implementation for the rectangular Poisson problem.
- What was done: Added baseline Poisson PINN code, plotting, metrics export, and a local persistent research log workflow in this project folder.
- Result: The project now has a concrete entry point for the first staged experiment.
- Open issue: Need to run the baseline training and verify convergence, runtime, and L2 error on this machine.
- Next step: Execute train_poisson_pinn.py, inspect outputs, and record the first actual run result.

## 2026-04-21

- Stage: setup
- PDE: Poisson
- Problem setting: Local workspace initialized in C:\Users\Admin\Desktop\ゼミ_AI\卒研
- Methods or changes: Added baseline PINN and local log workflow
- What was done: Validated syntax and prepared the project directory
- Result: Workspace is ready for the first training run
- Open issue: None
- Next step: Run train_poisson_pinn.py and inspect outputs

## 2026-04-21

- Stage: poisson-baseline
- PDE: Poisson
- Problem setting: Omega=[-1,1]^2, u=sin(pi x)sin(pi y), Dirichlet boundary u=0, epochs=1000, n_interior=1024, n_boundary=256
- Methods or changes: Fixed coordinate-shape bug in pde_residual and ran a lighter baseline instead of the original 5000-epoch setting
- What was done: Completed a 200-epoch smoke test and a 1000-epoch baseline run; generated plots and metrics under outputs/poisson_baseline_1k
- Result: Training converged stably to total loss 2.218e-02 with L2 relative error 1.015e-01 in 116.02 sec on the 1000-epoch run
- Open issue: The original 5000-epoch, larger-sample configuration timed out; need to balance runtime, collocation counts, and accuracy
- Next step: Inspect output plots, then test whether longer training, L-BFGS, or larger collocation sets reduce the L2 error below the current 0.10 baseline

## 2026-04-21

- Stage: setup
- PDE: repository
- Problem setting: Initialized Git repository in the thesis workspace
- Methods or changes: Added .gitignore, README, requirements, and Colab quickstart helper
- What was done: Prepared the workspace for Git-based local and Colab execution
- Result: Repository is ready to be connected to GitHub and used as the single source of truth
- Open issue: No remote is configured yet
- Next step: Create a GitHub repository, add origin, and push main

## 2026-04-21

- Stage: colab-setup
- PDE: Poisson
- Problem setting: Added a Colab notebook entry point for the Poisson baseline
- Methods or changes: Created poisson_baseline_colab.ipynb and updated README with Colab connection steps
- What was done: Prepared a notebook-based Colab workflow linked to the GitHub repository
- Result: The repository now has a direct Colab entry point for GPU runs
- Open issue: None
- Next step: Commit and push the notebook so it can be opened from Colab

## 2026-04-21

- Stage: poisson-optimizer
- PDE: Poisson
- Problem setting: Added optional Adam-to-L-BFGS training for the rectangular Poisson baseline
- Methods or changes: Extended train_poisson_pinn.py with optional L-BFGS steps after Adam and documented the new flags in README
- What was done: Implemented a two-stage optimizer path without changing the default Adam-only baseline
- Result: The workspace can now compare Adam-only training against Adam->L-BFGS refinement under the same Poisson setup
- Open issue: None
- Next step: Run a Colab experiment with --lbfgs-steps 200 and compare L2 error and runtime against the current Adam-only baseline
