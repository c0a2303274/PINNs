import torch


def shock_solution(
    t: torch.Tensor,
    x: torch.Tensor,
    nu: float,
    u_left: float = 1.0,
    u_right: float = 0.0,
    x0: float = 0.0,
) -> torch.Tensor:
    """Viscous Burgers traveling shock solution.

    Solves u_t + u u_x = nu u_xx and connects u_left to u_right.
    The shock speed is c=(u_left+u_right)/2.
    """
    speed = 0.5 * (u_left + u_right)
    jump = u_left - u_right
    phase = jump * (x - speed * t - x0) / (4.0 * nu)
    return speed - 0.5 * jump * torch.tanh(phase)


def initial_condition(x: torch.Tensor, nu: float, u_left: float, u_right: float, x0: float) -> torch.Tensor:
    return shock_solution(torch.zeros_like(x), x, nu, u_left=u_left, u_right=u_right, x0=x0)


def boundary_left(t: torch.Tensor, nu: float, u_left: float, u_right: float, x0: float, x_min: float) -> torch.Tensor:
    return shock_solution(t, torch.full_like(t, x_min), nu, u_left=u_left, u_right=u_right, x0=x0)


def boundary_right(t: torch.Tensor, nu: float, u_left: float, u_right: float, x0: float, x_max: float) -> torch.Tensor:
    return shock_solution(t, torch.full_like(t, x_max), nu, u_left=u_left, u_right=u_right, x0=x0)


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
