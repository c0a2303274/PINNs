"""Periodic, unforced vector Burgers benchmark from the Cole-Hopf transform.

This smooth potential-flow solution is a verification case, not a vortex test.
See BURGERS2D.md for the derivation, boundary conditions and limitations.
"""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from burgers2d_problem import pde_residual_2d
from pinn_model import MLP


@dataclass(frozen=True)
class Problem:
    nu: float = 0.01
    amplitude: float = 0.8
    speed_x: float = 0.5
    speed_y: float = 0.25
    t_final: float = 1.0

    def __post_init__(self):
        if not all(math.isfinite(x) for x in asdict(self).values()):
            raise ValueError("problem parameters must be finite")
        if self.nu <= 0 or self.t_final <= 0 or not 0 < self.amplitude < 1:
            raise ValueError("require nu > 0, t_final > 0, 0 < amplitude < 1")


def exact_solution(coords: torch.Tensor, problem: Problem) -> torch.Tensor:
    t, x, y = coords.split(1, dim=1)
    k = 2 * math.pi
    ax = k * (x - problem.speed_x * t)
    ay = k * (y - problem.speed_y * t)
    a = problem.amplitude * torch.exp(-2 * k * k * problem.nu * t)
    phi = 1 + a * torch.cos(ax) * torch.cos(ay)
    u = problem.speed_x + 2 * problem.nu * k * a * torch.sin(ax) * torch.cos(ay) / phi
    v = problem.speed_y + 2 * problem.nu * k * a * torch.cos(ax) * torch.sin(ay) / phi
    return torch.cat((u, v), dim=1)


def sample_points(n, start, end, device, dtype, generator):
    pts = torch.rand(n, 3, device=device, dtype=dtype, generator=generator)
    pts[:, 0] = start + (end - start) * pts[:, 0]
    return pts


def periodic_pairs(n, start, end, device, dtype, generator):
    """n paired points per direction: 4*n boundary coordinates in total."""
    pairs = []
    for axis in (1, 2):
        left = sample_points(n, start, end, device, dtype, generator)
        left[:, axis] = 0
        right = left.clone()
        right[:, axis] = 1
        pairs.append((left, right, axis))
    return pairs


def component_gradient(values, coords, component):
    return torch.autograd.grad(values[:, component].sum(), coords, create_graph=True)[0]


def periodic_losses(model, pairs):
    value_terms, derivative_terms = [], []
    for left, right, axis in pairs:
        left = left.detach().requires_grad_(True)
        right = right.detach().requires_grad_(True)
        yl, yr = model(left), model(right)
        value_terms.append((yl - yr).square().mean())
        # Match coordinate derivatives, not outward-normal derivatives (opposite signs).
        for component in (0, 1):
            gl = component_gradient(yl, left, component)[:, axis]
            gr = component_gradient(yr, right, component)[:, axis]
            derivative_terms.append((gl - gr).square().mean())
    return torch.stack(value_terms).mean(), torch.stack(derivative_terms).mean()


def residual(model, coords, problem):
    ru, rv = pde_residual_2d(model, coords[:, :1], coords[:, 1:2], coords[:, 2:3], problem.nu)
    return torch.cat((ru, rv), dim=1)


class SlabMLP(nn.Module):
    """Each slab takes physical coordinates; autodiff includes time rescaling."""

    def __init__(self, widths, start, end):
        super().__init__()
        if end <= start:
            raise ValueError("slab end must exceed start")
        self.start, self.end = float(start), float(end)
        self.network = MLP(in_dim=3, out_dim=2, hidden_widths=widths)

    def forward(self, coords):
        time = 2 * (coords[:, :1] - self.start) / (self.end - self.start) - 1
        space = 2 * coords[:, 1:] - 1
        return self.network(torch.cat((time, space), dim=1))


def handoff_target(previous, coords, problem):
    """Only the first slab sees exact initial data; later slabs use the teacher."""
    with torch.no_grad():
        return (exact_solution(coords, problem) if previous is None else previous(coords)).detach()


def slab_index(t, edges):
    """Use the later slab at a shared endpoint; t_final belongs to the last slab."""
    if t < edges[0] - 1e-8 or t > edges[-1] + 1e-8:
        raise ValueError("time is outside the trained domain")
    return min(sum(t >= edge for edge in edges[1:]), len(edges) - 2)


def predict_piecewise(models, edges, coords):
    if (coords[:, 0] < edges[0] - 1e-7).any() or (coords[:, 0] > edges[-1] + 1e-7).any():
        raise ValueError("coordinates outside trained time interval")
    boundaries = coords.new_tensor(edges[1:-1])
    indices = torch.bucketize(coords[:, 0].contiguous(), boundaries, right=True)
    result = coords.new_empty((len(coords), 2))
    for i, model in enumerate(models):
        mask = indices == i
        if mask.any():
            result[mask] = model(coords[mask])
    return result
