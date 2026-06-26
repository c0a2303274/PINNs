import math

import numpy as np
import torch


def initial_condition(x: torch.Tensor) -> torch.Tensor:
    return -torch.sin(math.pi * x)


def boundary_value(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(t)


def pde_residual(model, t: torch.Tensor, x: torch.Tensor, nu: float) -> torch.Tensor:
    coords = torch.cat([t, x], dim=1).requires_grad_(True)
    u = model(coords)

    grads = torch.autograd.grad(
        outputs=u,
        inputs=coords,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_t = grads[:, :1]
    u_x = grads[:, 1:]

    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=coords,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1:]

    return u_t + u * u_x - nu * u_xx


def finite_difference_reference(
    t_grid: np.ndarray,
    x_grid: np.ndarray,
    nu: float,
    safety: float = 0.2,
) -> np.ndarray:
    """Explicit finite-difference reference for 1D viscous Burgers.

    This is intended as a local reference for first experiments, not as a
    high-order production solver.
    """
    if len(t_grid) < 2 or len(x_grid) < 3:
        raise ValueError("reference grid must contain at least 2 time points and 3 x points")

    x = x_grid.astype(float)
    t_targets = t_grid.astype(float)
    dx = float(x[1] - x[0])
    t_final = float(t_targets[-1])
    u = -np.sin(math.pi * x)
    u[0] = 0.0
    u[-1] = 0.0

    result = np.empty((len(t_targets), len(x)), dtype=float)
    result[0] = u.copy()
    target_idx = 1
    current_t = 0.0

    while target_idx < len(t_targets):
        max_speed = max(float(np.max(np.abs(u))), 1.0e-6)
        dt_adv = safety * dx / max_speed
        dt_diff = safety * dx * dx / max(nu, 1.0e-12)
        next_target = float(t_targets[target_idx])
        dt = min(dt_adv, dt_diff, next_target - current_t, t_final - current_t)

        if dt <= 0.0:
            result[target_idx] = u.copy()
            target_idx += 1
            continue

        u_old = u.copy()
        du_dx = (u_old[2:] - u_old[:-2]) / (2.0 * dx)
        d2u_dx2 = (u_old[2:] - 2.0 * u_old[1:-1] + u_old[:-2]) / (dx * dx)
        u[1:-1] = u_old[1:-1] - dt * u_old[1:-1] * du_dx + nu * dt * d2u_dx2
        u[0] = 0.0
        u[-1] = 0.0
        current_t += dt

    return result
