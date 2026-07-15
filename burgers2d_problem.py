import torch


def pde_residual_2d(model, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, nu: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Residuals for the 2D viscous Burgers system.

    u_t + u u_x + v u_y = nu (u_xx + u_yy)
    v_t + u v_x + v v_y = nu (v_xx + v_yy)
    """
    coords = torch.cat([t, x, y], dim=1).requires_grad_(True)
    uv = model(coords)
    u = uv[:, :1]
    v = uv[:, 1:]

    grad_u = torch.autograd.grad(u, coords, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    grad_v = torch.autograd.grad(v, coords, torch.ones_like(v), create_graph=True, retain_graph=True)[0]
    u_t, u_x, u_y = grad_u[:, :1], grad_u[:, 1:2], grad_u[:, 2:]
    v_t, v_x, v_y = grad_v[:, :1], grad_v[:, 1:2], grad_v[:, 2:]

    u_xx = torch.autograd.grad(u_x, coords, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 1:2]
    u_yy = torch.autograd.grad(u_y, coords, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0][:, 2:]
    v_xx = torch.autograd.grad(v_x, coords, torch.ones_like(v_x), create_graph=True, retain_graph=True)[0][:, 1:2]
    v_yy = torch.autograd.grad(v_y, coords, torch.ones_like(v_y), create_graph=True, retain_graph=True)[0][:, 2:]

    residual_u = u_t + u * u_x + v * u_y - nu * (u_xx + u_yy)
    residual_v = v_t + u * v_x + v * v_y - nu * (v_xx + v_yy)
    return residual_u, residual_v
