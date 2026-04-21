import math

import torch


def exact_solution(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sin(math.pi * x) * torch.sin(math.pi * y)


def source_term(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 2.0 * (math.pi**2) * torch.sin(math.pi * x) * torch.sin(math.pi * y)


def boundary_value(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


def pde_residual(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    coords = torch.cat([x, y], dim=1).requires_grad_(True)
    u = model(coords)

    grads = torch.autograd.grad(
        outputs=u,
        inputs=coords,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    du_dx = grads[:, :1]
    du_dy = grads[:, 1:]

    d2u_dx2 = torch.autograd.grad(
        outputs=du_dx,
        inputs=coords,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True,
        retain_graph=True,
    )[0][:, :1]

    d2u_dy2 = torch.autograd.grad(
        outputs=du_dy,
        inputs=coords,
        grad_outputs=torch.ones_like(du_dy),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1:]

    x_coord = coords[:, :1]
    y_coord = coords[:, 1:]
    return -(d2u_dx2 + d2u_dy2) - source_term(x_coord, y_coord)
