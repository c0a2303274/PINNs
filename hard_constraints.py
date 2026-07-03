import torch
import torch.nn as nn


def project_affine_equality(
    y: torch.Tensor,
    a_matrix: torch.Tensor,
    b_vector: torch.Tensor,
    ridge: float = 1.0e-8,
) -> torch.Tensor:
    """Project y to the nearest point satisfying A y = b.

    This is the basic HardNet-style projection for affine equality constraints.
    It supports batched y with shape (batch, dim). A has shape (constraints, dim)
    and b has shape (constraints,) or (batch, constraints).
    """
    if y.ndim != 2:
        raise ValueError("y must have shape (batch, dim)")
    if a_matrix.ndim != 2:
        raise ValueError("a_matrix must have shape (constraints, dim)")
    if a_matrix.shape[1] != y.shape[1]:
        raise ValueError("a_matrix dim must match y dim")

    a = a_matrix.to(device=y.device, dtype=y.dtype)
    if b_vector.ndim == 1:
        b = b_vector.to(device=y.device, dtype=y.dtype).unsqueeze(0).expand(y.shape[0], -1)
    elif b_vector.ndim == 2:
        b = b_vector.to(device=y.device, dtype=y.dtype)
    else:
        raise ValueError("b_vector must have shape (constraints,) or (batch, constraints)")

    if b.shape[0] != y.shape[0] or b.shape[1] != a.shape[0]:
        raise ValueError("b_vector shape is incompatible with y and a_matrix")

    residual = y @ a.T - b
    gram = a @ a.T
    eye = torch.eye(gram.shape[0], device=y.device, dtype=y.dtype)
    correction_weights = torch.linalg.solve(gram + ridge * eye, residual.T).T
    correction = correction_weights @ a
    return y - correction


def affine_violation(y: torch.Tensor, a_matrix: torch.Tensor, b_vector: torch.Tensor) -> torch.Tensor:
    a = a_matrix.to(device=y.device, dtype=y.dtype)
    if b_vector.ndim == 1:
        b = b_vector.to(device=y.device, dtype=y.dtype).unsqueeze(0).expand(y.shape[0], -1)
    else:
        b = b_vector.to(device=y.device, dtype=y.dtype)
    return y @ a.T - b


class AffineEqualityProjection(nn.Module):
    """Differentiable projection layer for affine equality constraints."""

    def __init__(self, a_matrix: torch.Tensor, b_vector: torch.Tensor, ridge: float = 1.0e-8):
        super().__init__()
        self.register_buffer("a_matrix", a_matrix.detach().clone().float())
        self.register_buffer("b_vector", b_vector.detach().clone().float())
        self.ridge = ridge

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return project_affine_equality(y, self.a_matrix, self.b_vector, ridge=self.ridge)

    def violation(self, y: torch.Tensor) -> torch.Tensor:
        return affine_violation(y, self.a_matrix, self.b_vector)


class HardNet(nn.Module):
    """Neural network with a differentiable hard-constraint projection layer.

    This implements the practical HardNet pattern for affine equality
    constraints: unconstrained network output followed by projection.
    """

    def __init__(self, network: nn.Module, projection: nn.Module):
        super().__init__()
        self.network = network
        self.projection = projection

    def raw_forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.network(coords)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.projection(self.raw_forward(coords))
