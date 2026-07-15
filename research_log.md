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

## 2026-06-04

- Stage: poisson-baseline
- PDE: Poisson
- Problem setting: Omega=[-1,1]^2, optimizer comparison runner for Adam vs Adam->L-BFGS
- Methods or changes: Added run_poisson_comparison.py and final loss fields in metrics.json
- What was done: Verified the comparison runner with a 1-seed smoke test and generated summary.md/summary.csv
- Result: The workspace can now run the planned 3-seed optimizer comparison and produce a compact result table
- W&B run: None
- Open issue: Full 3-seed experiment still needs to be run with the target epochs and sample counts
- Next step: Run python .\run_poisson_comparison.py --seeds 0,1,2 --epochs 1000 --n-interior 1024 --n-boundary 256 --lbfgs-steps 200

## 2026-06-04

- Stage: poisson-baseline
- PDE: Poisson
- Problem setting: Omega=[-1,1]^2, epochs=1000, n_interior=1024, n_boundary=256, hidden_dim=100, hidden_layers=4, seeds=0,1,2
- Methods or changes: Ran Adam vs Adam->L-BFGS optimizer comparison on RTX 5060 Ti 8GB desktop
- What was done: Collected summary.md and summary.csv under results/poisson_optimizer_comparison
- Result: Adam mean L2 relative error was about 1.06e-1; Adam->L-BFGS mean L2 relative error was about 3.84e-2. L-BFGS improved all seeds but increased runtime from about 21.3 sec to 33.2 sec on average.
- W&B run: None
- Open issue: Only 3 seeds were tested; this is sufficient for the next progress report but not a full stability claim.
- Next step: Use this as the Poisson baseline result and start defining the 1D wave equation standard PINNs setup

## 2026-06-26

- Stage: direction-change
- PDE: Nonlinear PDE
- Problem setting: Move away from Poisson-only tuning toward hard-constrained neural methods for simple nonlinear PDEs
- Methods or changes: Advisor recommended HardNet, HardNet++, and ECO as candidate methods; Poisson will be treated as preliminary baseline work
- What was done: Reviewed the new research direction and selected simple nonlinear PDEs, likely 1D Burgers, as the next entry point
- Result: New storyline: standard PINNs established the workflow but Poisson accuracy plateaued around e-4, motivating hard-constraint methods for nonlinear PDEs
- W&B run: None
- Open issue: Need to read the three papers carefully and lock the first nonlinear PDE setting, reference solution, and baseline comparison
- Next step: Summarize HardNet/HardNet++/ECO and define the first Burgers or other simple nonlinear PDE experiment matrix

## 2026-06-26

- Stage: nonlinear-pde-baseline
- PDE: 1D viscous Burgers
- Problem setting: x=[-1,1], t=[0,1], u_t + u u_x = nu u_xx, u(x,0)=-sin(pi x), u(-1,t)=u(1,t)=0, nu=0.01/pi
- Methods or changes: Added Burgers problem definition and standard PINNs training script with finite-difference reference evaluation
- What was done: Created burgers_problem.py, train_burgers_pinn.py, nonlinear_pde_next_steps.md, and ran a 2-epoch CPU smoke test
- Result: Smoke test completed and saved fields, losses, history, metrics, and model files under outputs/burgers_smoke
- W&B run: None
- Open issue: Need serious multi-seed GPU runs and paper-level method summary before claiming improvement
- Next step: Run Burgers baseline on GPU for seeds 0,1,2, then design the first hard-constrained IC/BC variant

## 2026-07-02

- Stage: nonlinear-pde-comparison
- PDE: 1D viscous Burgers
- Problem setting: Compare soft PINNs with a hard IC/BC ansatz under the same Burgers setup
- Methods or changes: Added constraint-mode hard-icbc and a comparison runner for two-hour runs
- What was done: Implemented HardICBCBurgersModel with u(t,x)=(1-t)u0(x)+t(1-x^2)N(t,x), added run_burgers_constraint_comparison.py, and verified a smoke comparison
- Result: Smoke run completed; hard-icbc gave IC loss 0 and BC loss near machine precision, confirming exact condition enforcement
- W&B run: None
- Open issue: This is a simple hard-constraint ansatz, not a full HardNet/HardNet++ implementation; serious two-hour GPU results are still needed
- Next step: Run soft vs hard-icbc Burgers for two hours per method, first seed 0, then seeds 0,1,2 if stable

## 2026-07-03

- Stage: midterm-preparation
- PDE: 1D viscous Burgers
- Problem setting: After advisor presentation, continue Burgers soft vs hard-constraint comparison and prepare midterm resume within about two weeks
- Methods or changes: Added a two-week midterm resume plan and updated nonlinear PDE next steps
- What was done: Recorded the current research story, must-have experiments, GPU command for seeds 0,1,2, and resume outline
- Result: Advisor feedback was positive; next priority is multi-seed validation and resume-ready result organization
- W&B run: None
- Open issue: Need seeds 0,1,2 results before making a stable claim about hard IC/BC effectiveness
- Next step: Run the 12-hour maximum seeds 0,1,2 Burgers soft vs hard-icbc comparison on the research GPU

## 2026-07-03

- Stage: hardnet-transition
- PDE: 1D viscous Burgers and constrained toy problems
- Problem setting: Move quickly from the hard IC/BC ansatz toward HardNet and HardNet++ implementation
- Methods or changes: Added HardNet transition plan and a differentiable affine equality projection utility
- What was done: Implemented project_affine_equality and smoke_hardnet_projection.py; verified projection constraint satisfaction and gradient flow
- Result: Affine equality violation decreased from 3.43e+00 to 1.19e-07 in the smoke test, with nonzero gradient through the projection layer
- W&B run: None
- Open issue: This is only the first HardNet-style affine projection component; it is not yet connected to a PDE model or HardNet++ nonlinear constraints
- Next step: Connect the projection layer to a vector-valued constrained toy problem, then implement a HardNet++-style nonlinear correction loop

## 2026-07-03

- Stage: hardnet-implementation
- PDE: Constrained vector-valued toy problem
- Problem setting: Implement full minimal HardNet architecture as base network plus differentiable affine equality projection layer
- Methods or changes: Added HardNet nn.Module and train_hardnet_affine_demo.py
- What was done: Built a trainable HardNet demo for y0+y1=1 and ran a 50-epoch CPU smoke test
- Result: The demo trained successfully and kept max constraint violation around 1.19e-07, confirming hard constraint satisfaction during training
- W&B run: None
- Open issue: This covers affine equality HardNet only; nonlinear HardNet++ correction and PDE integration are still next steps
- Next step: Create a PDE-like vector-valued constrained toy problem, then implement HardNet++ nonlinear equality correction

## 2026-07-03

- Stage: hardnet-vector-field
- PDE: Constrained vector-field toy problem
- Problem setting: Compare a soft MLP and HardNet on a vector-valued field with constraint u+v=0
- Methods or changes: Added train_hardnet_vector_field.py and run_hardnet_vector_comparison.py
- What was done: Implemented a timed comparison runner and verified a short smoke run for soft and hardnet methods
- Result: Smoke run showed soft constraint violation around 9.88e-01 while HardNet kept max violation around 5.96e-08
- W&B run: None
- Open issue: Need full GPU runs for seeds 0,1,2 and then HardNet++ nonlinear constraint implementation
- Next step: Run run_hardnet_vector_comparison.py on GPU for seeds 0,1,2 with one hour per method and seed

## 2026-07-08

- Stage: burgers-fast-track
- PDE: 1D viscous Burgers
- Problem setting: Prepare promising hard-IC/BC Burgers runs before the next seminar
- Methods or changes: Added Adam runtime split support and run_burgers_fast_track.py
- What was done: Implemented configs for hard_icbc_adam, hard_icbc_lbfgs, hard_icbc_lower_lr and verified a smoke run
- Result: Smoke run completed and confirmed Adam->L-BFGS enters the L-BFGS phase after the Adam runtime budget
- W&B run: None
- Open issue: Need GPU result for seed 0 to choose the best Burgers setting before multi-seed validation
- Next step: Run run_burgers_fast_track.py on GPU for seed 0 with 7200 seconds per config

## 2026-07-08

- Stage: hardnetpp-nonlinear-demo
- PDE: Nonlinear constrained toy problem
- Problem setting: Implement HardNet++-style nonlinear equality enforcement before connecting back to Burgers
- Methods or changes: Added NonlinearEqualityProjection, HardNetPlusPlus, train_hardnetpp_circle_demo.py, and run_hardnetpp_circle_comparison.py
- What was done: Verified a smoke comparison on the unit-circle constraint y0^2+y1^2=1
- Result: HardNet++ smoke run kept max nonlinear constraint violation around 1.19e-07, while soft MLP had violation around 1.00 in the short run
- W&B run: None
- Open issue: This is a nonlinear constraint toy demo, not yet a Burgers or Karman-vortex PDE result
- Next step: Run the HardNet++ circle comparison on GPU for seeds 0,1,2, then connect nonlinear constraints back to a PDE setting

## 2026-07-08

- Stage: burgers-integrated-hard-constraints
- PDE: 1D viscous Burgers
- Problem setting: Integrate hard IC/BC and bounded correction variants into the Burgers training path
- Methods or changes: Added BoundedHardICBCBurgersModel and run_burgers_integrated_comparison.py
- What was done: Implemented soft, hard_icbc_lbfgs, and bounded_hard_icbc_lbfgs comparison path and verified a smoke run
- Result: Smoke run completed; the integrated runner can switch variants and enter L-BFGS where configured
- W&B run: None
- Open issue: Need GPU seed 0 integrated result to see whether bounded correction improves L2, PDE residual, or stability
- Next step: Run the integrated Burgers comparison on GPU for seed 0, then extend the best setting to seeds 0,1,2

## 2026-07-09

- Stage: burgers-diagnostics
- PDE: 1D viscous Burgers
- Problem setting: Skip additional seed checks and diagnose why bounded hard-IC/BC failed
- Methods or changes: Added shock-focused interior sampling and amplitude-2 bounded configurations
- What was done: Implemented configs bounded_amp2_lbfgs, hard_icbc_focused_lbfgs, bounded_amp2_focused_lbfgs and verified a smoke run
- Result: Smoke run completed; next GPU run will test whether bounded failure was due to too-small amplitude or insufficient sampling near the steep transition
- W&B run: None
- Open issue: Need GPU results to determine whether focused sampling or larger bounded correction improves L2/PDE loss
- Next step: Run the Burgers diagnostic comparison for seed 0 on GPU

## 2026-07-16

- Stage: burgers-advisor-feedback
- PDE: 1D and 2D Burgers
- Problem setting: Revise Burgers direction based on advisor feedback: use representative shock solutions instead of only -sin(pi x), and touch 2D Burgers
- Methods or changes: Added analytic traveling shock problem, shock PINN trainer, 2D Burgers residual scaffold, and feedback plan
- What was done: Implemented burgers_shock_problem.py, train_burgers_shock_pinn.py, burgers2d_problem.py, and verified a short hard-IC/BC shock smoke run
- Result: Shock smoke run completed with exact IC/BC losses equal to zero and analytic-reference L2 output generated
- W&B run: None
- Open issue: Need GPU run for the analytic shock setup and a concrete 2D Burgers reference/IC/BC choice
- Next step: Run the 1D analytic shock hard-IC/BC GPU experiment, then define the 2D Burgers manufactured or numerical reference problem
