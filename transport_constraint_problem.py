import torch


def exact_solution(coords: torch.Tensor, constraint: str, wave_speed: float = 1.0) -> torch.Tensor:
    t = coords[:, :1]
    x = coords[:, 1:]
    phase = 2.0 * torch.pi * (x - wave_speed * t)
    if constraint == "linear":
        first = torch.sin(phase)
        return torch.cat([first, -first], dim=1)
    if constraint == "circle":
        return torch.cat([torch.cos(phase), torch.sin(phase)], dim=1)
    raise ValueError(f"unsupported constraint: {constraint}")


def constraint_residual(y: torch.Tensor, constraint: str) -> torch.Tensor:
    if constraint == "linear":
        return y[:, :1] + y[:, 1:2]
    if constraint == "circle":
        return torch.sum(y**2, dim=1, keepdim=True) - 1.0
    raise ValueError(f"unsupported constraint: {constraint}")


def advection_residual(
    model: torch.nn.Module,
    coords: torch.Tensor,
    wave_speed: float = 1.0,
) -> torch.Tensor:
    coords_with_grad = coords.detach().requires_grad_(True)
    prediction = model(coords_with_grad)
    return advection_residual_from_prediction(prediction, coords_with_grad, wave_speed)


def advection_residual_from_prediction(
    prediction: torch.Tensor,
    coords: torch.Tensor,
    wave_speed: float = 1.0,
) -> torch.Tensor:
    component_residuals = []
    for component in range(prediction.shape[1]):
        gradient = torch.autograd.grad(
            prediction[:, component].sum(),
            coords,
            create_graph=True,
            retain_graph=True,
        )[0]
        component_residuals.append(gradient[:, :1] + wave_speed * gradient[:, 1:2])
    return torch.cat(component_residuals, dim=1)


def sample_interior(n_points: int, device: torch.device) -> torch.Tensor:
    return torch.rand(n_points, 2, device=device)


def sample_initial(n_points: int, device: torch.device) -> torch.Tensor:
    t = torch.zeros(n_points, 1, device=device)
    x = torch.rand(n_points, 1, device=device)
    return torch.cat([t, x], dim=1)


def sample_periodic_boundary(n_points: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.rand(n_points, 1, device=device)
    left = torch.cat([t, torch.zeros_like(t)], dim=1)
    right = torch.cat([t, torch.ones_like(t)], dim=1)
    return left, right
