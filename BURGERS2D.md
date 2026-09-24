# 2D Burgers: variable-width fully connected PINNs and time marching

This is a new baseline following the 2026-09-24 advisor feedback. All models
are fully connected MLPs, not convolutional networks. Varying hidden widths
does not change that. One input is `(t,x,y)`, not a flattened spatial grid.

## Locked problem

On `(x,y) in [0,1]^2`, `t in [0,1]`, solve the **unforced** coupled system

```text
u_t + u*u_x + v*u_y = nu*(u_xx + u_yy)
v_t + u*v_x + v*v_y = nu*(v_xx + v_yy)
nu = 0.01
```

The reference is an explicitly constructed Cole-Hopf potential solution:

```text
k = 2*pi; a = 0.8; c_x = 0.5; c_y = 0.25
X = k*(x-c_x*t); Y = k*(y-c_y*t)
A(t) = a*exp(-2*k*k*nu*t)
phi = 1 + A(t)*cos(X)*cos(Y)
u = c_x + 2*nu*k*A(t)*sin(X)*cos(Y)/phi
v = c_y + 2*nu*k*A(t)*cos(X)*sin(Y)/phi
```

Here `phi >= 1-a > 0` and `phi_t+c_x*phi_x+c_y*phi_y=nu*Delta(phi)`.
The transformation `(u,v)=c-2*nu*grad(log(phi))` gives the Burgers solution.
The formula above is our chosen verification case, not a claim to reproduce a
particular paper's numerical experiment. Tests independently check the PDE,
periodicity, and a polynomial residual with a hand-derived answer.

Initial data is this solution at `t=0`. Opposite edges have matching values
and coordinate-normal derivatives (`u_x,v_x` on x-edges, `u_y,v_y` on y-edges).
Initial, periodic value, and periodic derivative conditions are all Soft losses.
No hard projection, incompressibility loss, or unit-circle constraint is used.

**Limitations:** this is a smooth, irrotational potential flow with drift and
diffusion. It is not Karman vortex shedding, shock formation, or evidence for
arbitrary high-Reynolds-number flows. The uniform background makes ordinary
relative error look smaller, so `fluctuation_l2_relative_error` divides the
same prediction error by `||reference - c||`, and is reported alongside full
field, per-component and per-time errors. No exact solution is used for
interior supervised training, checkpoint selection or later-slab initial data.

Background references:

- [Multidimensional Hopf-Cole transformation](https://www.sciencedirect.com/science/article/pii/S0375960115008208)
- [Time-marching and window-sweeping](https://arxiv.org/abs/2302.14227)

## Comparison matrix

| Case | Hidden widths | Time training | Params/model | Stored models |
|---|---|---|---:|---:|
| A | 128,128,128,128,128 | global | 66,818 | 1 |
| B | 256,128,64,128,256 | global | 84,034 | 1 |
| C | 128,128,128,128,128 | 5 sequential slabs | 66,818 | 5 |
| D | 256,128,64,128,256 | 5 sequential slabs | 84,034 | 5 |

All use tanh, Adam 1e-3, float32, seed 0, 1024 interior points per update,
256 fixed initial points per slab, and 128 paired boundary points **per spatial
direction** (512 boundary coordinates per update). Loss weights are 1 for
PDE, IC, boundary values and boundary derivatives. Interior/boundary points
are resampled; sampling RNG is separated from model initialization RNG.

The default time budget is 600 seconds **per case**, including all windows:
A/B each get 600 seconds and each of C/D's 5 windows gets 120 seconds.
`--epochs` is also a total cap split across windows, not a per-window cap.
Setup, checkpoint saving, evaluation and plotting take additional wall time.
There is at most a one-update overshoot per slab. No convergence early stop is
implemented. A low epoch cap can therefore finish before the time budget.

Each new slab warm-starts the previous network weights, with fresh Adam state.
Time is scaled to [-1,1] within the slab; spatial coordinates also use [-1,1].
This rescaling is part of the computational graph. Later IC targets are fixed,
detached predictions from the previous frozen model, NOT exact reference data.
All slab models are saved and used on their own time intervals. A shared
endpoint belongs to the later slab; interface jumps are measured separately.

Width comparisons are not parameter-matched. Report costs and avoid attributing
improvements solely to compression. Time-marching includes local time scaling
and warm starting; do not claim a pure window-number ablation.

## Verify locally / on GPU server

```bash
conda activate pinns
python -m unittest -v test_burgers2d
```

Dependencies are the existing `torch`, `numpy`, `matplotlib`. W&B is optional
and only imported if `--wandb-mode offline` or `online` is explicitly selected.
CPU tests do not establish GPU compatibility or scientific convergence.

GPU pilot (4 x 600 seconds = approximately 40 minutes plus overhead):

```bash
python run_burgers2d_comparison.py --runtime-sec 600 --device cuda --output-root outputs/burgers2d_pilot --share-dir results/burgers2d_pilot
```

For a quick full-matrix smoke run, use `--runtime-sec 15` with NEW output/share
directories. For a baseline-only run use `--cases A`. All case selection and
sampling/evaluation count overrides are shown by `--help`.

To survive SSH disconnect, start the command inside `tmux new -s burgers2d`,
then detach with Ctrl-B followed by D. Later use `tmux attach -t burgers2d`.

Existing nonempty output directories are rejected to avoid mixing experiments.
Automatic resume is not implemented. Use a new run name for another run.
CUDA requested but unavailable raises an error instead of silently using CPU.

## Follow-up: first-window time allocation

The pilot `results/burgers2d_pilot_20260924_182901/` shows that C already has
a larger error at t=0 (0.112817 vs A's 0.025677). In C's first slab, sampled
IC loss falls from 0.004109 at 89.75 s to 0.002046 at 120.01 s. These are
pre-update training losses, not held-out RMSE. This motivates a time-allocation
test; it does not prove insufficient runtime is the only cause.

Run C and E together on the same GPU, sequentially:

| Case | First slab | Each of remaining 4 slabs | Total training |
|---|---:|---:|---:|
| C (control) | 120 s | 120 s | 600 s |
| E (front-loaded) | 300 s | 75 s | 600 s |

Both keep 128x5 hidden widths, 5 equal physical-time windows, seed 0, all
sampling counts, loss weights, Adam settings, local time normalization and
teacher handoff unchanged. Only training-time allocation changes. Epoch caps
remain split equally (200000/window by default); verify `stop_reason` is
`time_budget`. This tests allocation, not extra total compute, and can harm
later windows by reducing their time. Same seed does not imply identical
later-window samples: sampling uses a continuing RNG and update counts differ.

```bash
conda activate pinns
RUN_ID=$(date +%Y%m%d_%H%M%S)
python run_burgers2d_comparison.py --cases C,E --runtime-sec 600 --device cuda --output-root "outputs/burgers2d_allocation_${RUN_ID}" --share-dir "results/burgers2d_allocation_${RUN_ID}"
```

Expected training time: **20 minutes total**, plus setup/evaluation. Use tmux
as described above if disconnecting. Original A-D defaults remain unchanged.
`--first-window-fraction 0.5` is E's default; C always uses equal allocation.
The trainer itself also accepts this flag, but only in marching mode.

Check initial-time L2/held-out IC RMSE first, then full-time L2, fluctuation
error, terminal error and interface jumps. An initial improvement with worse
terminal error is a tradeoff, not an overall success. If initial accuracy does
not improve, do not blindly extend runtime; investigate losses and scaling.
Use `window_budgets_sec` in config/metrics/summary and `slabs.json` for auditing.
One seed is a diagnostic, not a robustness claim.

```bash
git add "results/burgers2d_allocation_${RUN_ID}"
git commit -m "Add 2D Burgers time-allocation comparison results"
git push origin main
```

## Artifacts and sharing

`outputs/burgers2d_pilot/` contains `summary.csv`, `summary.md`, commands and
per-case directories. Per case:

- `config.json`: complete setup, device/version, Git commit/dirty status, source hashes.
- `metrics.json`: held-out field/residual/condition errors, budget and actual time, memory.
- `time_errors.csv/png`: full-field and fluctuation-normalized error over physical time.
- `interfaces.json`: jumps between adjacent slabs (not reported as PDE residual).
- `fields_u.png`, `fields_v.png`: reference, prediction and absolute error at three times;
  field scales shared within each component, error scales shared across times.
- `fields.npz`: numeric time slices, prediction and reference; periodic grid excludes
  the duplicated right endpoints. Stored array order is `[time,y,x,component]`.
- `history.csv`, `training_losses.png`: sparsely logged pre-update training losses.
- `slabs.json`, `slab_00.pt`, ...: budget/stop reason and final checkpoints per slab.
- `status.json`: complete or failed. No validation-error-based model selection.

Peak GPU memory is PyTorch allocated memory including evaluation and all saved
slab models still on device, not full process VRAM. Inference times cover equal
point counts, include routing across slabs, and exclude data transfer/training.

`--share-dir` exports only JSON/CSV/PNG plus the summary: no `.pt` or `.npz`.
It only copies files; it does not run `git add`, commit, or push.

```bash
git add results/burgers2d_pilot
git commit -m "Add 2D Burgers architecture and time-marching pilot results"
git push origin main
```

After the pilot, inspect time-resolved error, field plots, IC/BC and interface
jumps before extending compute or adding hard constraints. Good results in this
case alone do not validate strong nonlinear regimes or vortical flows.
