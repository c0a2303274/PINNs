# Burgers advisor feedback plan

## Feedback

The previous Burgers setup used the common PINNs benchmark initial condition:

```text
u(x,0) = -sin(pi x)
```

Advisor feedback was that this was not the intended Burgers direction. The next experiments should follow representative Burgers solutions such as those introduced through the Hopf-Cole transform:

- uniform solution
- traveling shock solution
- shock merging

Reference page:

```text
https://aquabreath.jp/2021/06/13/burgers方程式-代表的な解/
```

The advisor also requested touching 2D Burgers.

## Revised direction

1. Keep the previous `-sin(pi x)` experiments as preliminary PINNs/Burgers baseline only.
2. Add a 1D traveling shock solution with analytic reference.
3. Compare soft PINNs and hard IC/BC on this analytic shock setup.
4. Add a 2D Burgers residual implementation as the entry point toward higher-dimensional nonlinear PDEs.
5. After the 1D shock result is stable, move to 2D Burgers with a manufactured or numerically generated reference solution.

## First shock experiment

Use:

```text
u_t + u u_x = nu u_xx
u(x,t) = c - (u_left-u_right)/2 * tanh((u_left-u_right)(x-c t-x0)/(4 nu))
c = (u_left+u_right)/2
```

Default parameters:

```text
u_left = 1
u_right = 0
nu = 0.05
x in [-1,1]
t in [0,1]
```

Run a smoke test:

```bash
python train_burgers_shock_pinn.py --epochs 100 --n-interior 256 --n-initial 64 --n-boundary 64 --hidden-dim 32 --hidden-layers 2 --device cpu --output-dir outputs/burgers_shock_smoke
```

Run a GPU comparison candidate:

```bash
python train_burgers_shock_pinn.py --constraint-mode hard-icbc --epochs 1000000 --max-runtime-sec 7200 --adam-max-runtime-sec 3600 --lbfgs-steps 50000 --n-interior 4096 --n-initial 512 --n-boundary 512 --hidden-dim 128 --hidden-layers 5 --nu 0.05 --u-left 1.0 --u-right 0.0 --output-dir outputs/burgers_shock_hard_icbc_seed0
```

## 2D Burgers entry point

The 2D viscous Burgers system is:

```text
u_t + u u_x + v u_y = nu (u_xx + u_yy)
v_t + u v_x + v v_y = nu (v_xx + v_yy)
```

Implemented first:

```text
burgers2d_problem.py
```

Next required before full training:

- choose exact/manufactured solution or numerical reference
- define IC/BC
- decide whether HardNet constraints apply to boundedness, conservation, or vector-field structure
