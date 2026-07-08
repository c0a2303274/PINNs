import torch
import torch.nn as nn

from burgers_problem import initial_condition
from pinn_model import MLP


class HardICBCBurgersModel(nn.Module):
    """Burgers model with initial and boundary conditions satisfied by construction.

    u(t, x) = (1 - t) u0(x) + t (1 - x^2) N(t, x)

    This is not a full HardNet implementation. It is the first controlled
    hard-constraint baseline for testing whether exact IC/BC enforcement helps.
    """

    def __init__(self, hidden_dim: int = 100, hidden_layers: int = 4):
        super().__init__()
        self.network = MLP(in_dim=2, hidden_dim=hidden_dim, hidden_layers=hidden_layers, out_dim=1)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        t = coords[:, :1]
        x = coords[:, 1:]
        correction = t * (1.0 - x**2) * self.network(coords)
        return (1.0 - t) * initial_condition(x) + correction


class BoundedHardICBCBurgersModel(nn.Module):
    """Burgers hard IC/BC model with bounded correction.

    The network correction is passed through tanh before the IC/BC transform.
    This is a first Burgers-integrated boundedness constraint, useful for
    testing whether limiting the learned correction improves stability.
    """

    def __init__(self, hidden_dim: int = 100, hidden_layers: int = 4, amplitude: float = 1.0):
        super().__init__()
        self.network = MLP(in_dim=2, hidden_dim=hidden_dim, hidden_layers=hidden_layers, out_dim=1)
        self.amplitude = amplitude

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        t = coords[:, :1]
        x = coords[:, 1:]
        bounded_correction = self.amplitude * torch.tanh(self.network(coords))
        correction = t * (1.0 - x**2) * bounded_correction
        return (1.0 - t) * initial_condition(x) + correction
