import torch

from hard_constraints import AffineEqualityProjection, HardNet, HardNetPlusPlus, NonlinearEqualityProjection
from pinn_model import MLP
from transport_constraint_problem import advection_residual, constraint_residual, exact_solution


class ExactTransport(torch.nn.Module):
    def __init__(self, constraint: str):
        super().__init__()
        self.constraint = constraint

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return exact_solution(coords, self.constraint)


def circle_constraint(y: torch.Tensor) -> torch.Tensor:
    return constraint_residual(y, "circle")


def main() -> None:
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    coords = torch.rand(32, 2)

    for constraint in ["linear", "circle"]:
        exact = ExactTransport(constraint)
        residual = advection_residual(exact, coords)
        violation = constraint_residual(exact_solution(coords, constraint), constraint)
        assert residual.abs().max().item() < 1.0e-9
        assert violation.abs().max().item() < 1.0e-9

    linear_model = HardNet(
        MLP(in_dim=2, hidden_dim=16, hidden_layers=2, out_dim=2),
        AffineEqualityProjection(torch.tensor([[1.0, 1.0]]), torch.tensor([0.0])),
    ).double()
    linear_prediction = linear_model(coords)
    assert constraint_residual(linear_prediction, "linear").abs().max().item() < 1.0e-7

    circle_projection = NonlinearEqualityProjection(circle_constraint, iterations=15, ridge=1.0e-8)
    raw_circle = torch.randn(32, 2) + torch.tensor([1.0, 0.0])
    projected_circle = circle_projection(raw_circle)
    assert constraint_residual(projected_circle, "circle").abs().max().item() < 1.0e-7

    circle_model = HardNetPlusPlus(
        MLP(in_dim=2, hidden_dim=16, hidden_layers=2, out_dim=2),
        NonlinearEqualityProjection(circle_constraint, iterations=8, ridge=1.0e-8),
    ).double()
    residual = advection_residual(circle_model, coords)
    loss = torch.mean(residual**2)
    loss.backward()
    gradients = [parameter.grad for parameter in circle_model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    print("transport constraint smoke test passed")


if __name__ == "__main__":
    main()
