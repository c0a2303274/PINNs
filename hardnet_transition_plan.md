# HardNet transition plan

## Position

Move from the current `hard-icbc` ansatz to paper-based hard-constraint methods quickly. The current Burgers comparison remains useful as a bridge:

```text
soft PINNs -> simple hard IC/BC ansatz -> HardNet-style projection -> HardNet++ nonlinear constraint enforcement
```

Do not spend too much time optimizing the simple ansatz unless it is needed for a clear baseline table.

## What is already done

- Standard PINNs baseline for Poisson.
- Standard PINNs baseline for 1D Burgers.
- Simple hard IC/BC ansatz for Burgers:

```text
u(t,x) = (1-t)u0(x) + t(1-x^2)N(t,x)
```

This exactly enforces:

```text
u(x,0) = -sin(pi x)
u(-1,t) = u(1,t) = 0
```

This is not a HardNet implementation. It is a controlled pre-HardNet hard-constraint baseline.

## Paper-based direction

| method | core idea | relevance to current project |
|---|---|---|
| HardNet | append a differentiable projection layer so outputs satisfy affine/convex constraints | first target for equality/constraint projection implementation |
| HardNet++ | iteratively adjust outputs using damped local linearizations to satisfy nonlinear equality/inequality constraints | next target for nonlinear PDE-related constraints |
| ECO | constrain learned dynamics with energy/boundedness structure | later target for chaotic or vortex-like time evolution |

## Implementation route

### Step 1: HardNet-style affine equality projection

Implement a small, testable projection utility:

```text
Given raw output y and affine equality A y = b,
project y to the nearest y_proj satisfying A y_proj = b.
```

For scalar Burgers output, this is not very useful by itself because IC/BC can be enforced more simply by the ansatz. The purpose is to build the projection-layer machinery for later vector-valued outputs.

### Step 2: Vector-valued test problem

Use a controlled vector output problem before Karman vortex:

```text
network output: [u, v]
constraint example: divergence-free or linear conservation-like condition
```

The goal is to verify the projection layer on a constraint that cannot be reduced to the current scalar ansatz.

### Step 3: HardNet++ local nonlinear projection

Implement the nonlinear correction loop:

```text
y_{k+1} = y_k + damped correction from local linearization of c(y)=0
```

Start with a toy nonlinear constraint such as:

```text
c(y) = ||y||^2 - 1 = 0
```

Then connect it to PDE-related nonlinear constraints.

### Step 4: PDE application

Apply the method to nonlinear PDE constraints in this order:

1. Burgers with exact IC/BC and improved PDE residual handling.
2. A simple 2D incompressible-flow-like setup with divergence-free constraint.
3. Karman-vortex-like target if the controlled cases are reproducible.

## Midterm resume framing

The midterm resume should say:

```text
The current implementation has confirmed the benefit of hard constraint enforcement for IC/BC satisfaction.
Based on this, the next step is moving from a hand-designed hard ansatz to HardNet/HardNet++-style projection layers.
```

Avoid saying:

```text
HardNet has already been implemented.
```

until the projection layer is actually coded and tested.

## Current implementation

The first HardNet architecture is implemented:

```text
HardNet = base neural network + differentiable affine equality projection layer
```

Available files:

```text
hard_constraints.py
smoke_hardnet_projection.py
train_hardnet_affine_demo.py
```

The current demo learns a vector-valued function under the constraint:

```text
y0 + y1 = 1
```

The model output is always projected to satisfy the constraint. A short smoke run verified:

```text
max constraint violation: about 1e-7
gradients flow through the projection layer
```

## Immediate next coding task

The HardNet module is now connected to a vector-valued PDE-like toy problem:

```text
network output: [u, v]
constraint: u + v = 0
target: u = sin(pi x) cos(pi y), v = -u
```

Run a timed GPU comparison:

```bash
python run_hardnet_vector_comparison.py --seeds 0,1,2 --runtime-sec 3600 --epochs 1000000 --n-points 4096 --hidden-dim 128 --hidden-layers 5 --lr 0.001 --output-root outputs/hardnet_vector_comparison_1h_seeds012
```

Expected result:

```text
soft MLP may fit the target but can violate u+v=0
HardNet should keep max |u+v| near machine precision
```

Next after this run:

```text
implement HardNet++-style nonlinear equality correction
```

## HardNet++ nonlinear equality demo

The first HardNet++-style layer is implemented for nonlinear equality constraints. It iteratively applies damped local linearization corrections.

Current toy constraint:

```text
y0^2 + y1^2 = 1
```

Run a GPU comparison:

```bash
python run_hardnetpp_circle_comparison.py --seeds 0,1,2 --runtime-sec 1800 --epochs 1000000 --n-points 1024 --hidden-dim 64 --hidden-layers 3 --lr 0.001 --projection-iterations 15 --output-root outputs/hardnetpp_circle_comparison_30m_seeds012
```

Expected result:

```text
soft MLP may fit the circle target but does not guarantee y0^2+y1^2=1
HardNet++ should keep nonlinear constraint violation near machine precision
```
