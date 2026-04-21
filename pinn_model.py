import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 100, hidden_layers: int = 4, out_dim: int = 1):
        super().__init__()
        layers = []
        width = in_dim

        for _ in range(hidden_layers):
            layers.append(nn.Linear(width, hidden_dim))
            layers.append(nn.Tanh())
            width = hidden_dim

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

