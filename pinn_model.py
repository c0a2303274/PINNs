import torch
import torch.nn as nn
from collections.abc import Sequence


class MLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 100, hidden_layers: int = 4, out_dim: int = 1,
                 hidden_widths: Sequence[int] | None = None):
        super().__init__()
        if in_dim < 1 or out_dim < 1 or hidden_layers < 0:
            raise ValueError("input/output dimensions must be positive and depth nonnegative")
        widths = list(hidden_widths) if hidden_widths is not None else [hidden_dim] * hidden_layers
        if hidden_widths is not None and not widths:
            raise ValueError("hidden_widths must not be empty")
        if any(not isinstance(w, int) or isinstance(w, bool) or w < 1 for w in widths):
            raise ValueError("hidden widths must be positive integers")
        self.hidden_widths = tuple(widths)
        layers = []
        width = in_dim

        for next_width in widths:
            layers.append(nn.Linear(width, next_width))
            layers.append(nn.Tanh())
            width = next_width

        layers.append(nn.Linear(width, out_dim))
        self.network = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.network(coords)
