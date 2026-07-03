import torch

from hard_constraints import affine_violation, project_affine_equality


def main() -> None:
    torch.manual_seed(0)
    y = torch.randn(8, 2, requires_grad=True)

    # Constraint: y0 + y1 = 1.
    a_matrix = torch.tensor([[1.0, 1.0]])
    b_vector = torch.tensor([1.0])

    before = affine_violation(y, a_matrix, b_vector).abs().max()
    projected = project_affine_equality(y, a_matrix, b_vector)
    after = affine_violation(projected, a_matrix, b_vector).abs().max()

    loss = projected.square().mean()
    loss.backward()

    grad_norm = y.grad.norm()
    print(f"max violation before projection: {before.item():.6e}")
    print(f"max violation after projection:  {after.item():.6e}")
    print(f"gradient norm through layer:      {grad_norm.item():.6e}")

    if after.item() > 1.0e-5:
        raise RuntimeError("projection did not satisfy the affine equality constraint")
    if grad_norm.item() == 0.0:
        raise RuntimeError("gradient did not flow through the projection")


if __name__ == "__main__":
    main()
