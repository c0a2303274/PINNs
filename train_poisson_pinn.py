import argparse
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn_model import MLP
from poisson_problem import boundary_value, exact_solution, pde_residual


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_interior(n_points: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.empty(n_points, 1, device=device).uniform_(-1.0, 1.0)
    y = torch.empty(n_points, 1, device=device).uniform_(-1.0, 1.0)
    return x, y


def sample_boundary(n_points: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    n_side = max(n_points // 4, 1)
    left_y = torch.empty(n_side, 1, device=device).uniform_(-1.0, 1.0)
    right_y = torch.empty(n_side, 1, device=device).uniform_(-1.0, 1.0)
    bottom_x = torch.empty(n_side, 1, device=device).uniform_(-1.0, 1.0)
    top_x = torch.empty(n_side, 1, device=device).uniform_(-1.0, 1.0)

    x = torch.cat(
        [
            -torch.ones_like(left_y),
            torch.ones_like(right_y),
            bottom_x,
            top_x,
        ],
        dim=0,
    )
    y = torch.cat(
        [
            left_y,
            right_y,
            -torch.ones_like(bottom_x),
            torch.ones_like(top_x),
        ],
        dim=0,
    )
    return x, y


def compute_losses(
    model: MLP,
    n_interior: int,
    n_boundary: int,
    device: torch.device,
    lambda_bc: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_f, y_f = sample_interior(n_interior, device)
    residual = pde_residual(model, x_f, y_f)
    loss_pde = torch.mean(residual**2)

    x_bc, y_bc = sample_boundary(n_boundary, device)
    coords_bc = torch.cat([x_bc, y_bc], dim=1)
    u_bc = model(coords_bc)
    target_bc = boundary_value(x_bc, y_bc)
    loss_bc = torch.mean((u_bc - target_bc) ** 2)

    total = loss_pde + lambda_bc * loss_bc
    return total, loss_pde, loss_bc


def evaluate(model: MLP, device: torch.device, grid_size: int = 101) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    grid = torch.linspace(-1.0, 1.0, grid_size, device=device)
    xx, yy = torch.meshgrid(grid, grid, indexing="xy")
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    with torch.no_grad():
        pred = model(coords).reshape(grid_size, grid_size)
        truth = exact_solution(xx, yy)

    l2_error = torch.linalg.norm(pred - truth) / torch.linalg.norm(truth)
    return (
        float(l2_error.item()),
        xx.cpu().numpy(),
        yy.cpu().numpy(),
        (pred - truth).cpu().numpy(),
    )


def save_plots(model: MLP, output_dir: Path, device: torch.device, history: dict[str, list[float]], grid_size: int = 101) -> None:
    grid = torch.linspace(-1.0, 1.0, grid_size, device=device)
    xx, yy = torch.meshgrid(grid, grid, indexing="xy")
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    with torch.no_grad():
        pred = model(coords).reshape(grid_size, grid_size).cpu().numpy()
        truth = exact_solution(xx, yy).cpu().numpy()

    error = pred - truth

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, title in zip(
        axes,
        [truth, pred, error],
        ["Exact solution", "PINN prediction", "Prediction error"],
    ):
        image = ax.imshow(data, origin="lower", extent=(-1, 1, -1, 1), cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "poisson_fields.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["total"], label="total")
    ax.plot(history["pde"], label="pde")
    ax.plot(history["bc"], label="bc")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.set_title("Training losses")
    fig.tight_layout()
    fig.savefig(output_dir / "training_losses.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline PINN for the rectangular Poisson problem.")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--n-interior", type=int, default=2048)
    parser.add_argument("--n-boundary", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-bc", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/poisson_baseline"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--print-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MLP(hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"total": [], "pde": [], "bc": []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        total, loss_pde, loss_bc = compute_losses(
            model=model,
            n_interior=args.n_interior,
            n_boundary=args.n_boundary,
            device=device,
            lambda_bc=args.lambda_bc,
        )
        total.backward()
        optimizer.step()

        history["total"].append(float(total.item()))
        history["pde"].append(float(loss_pde.item()))
        history["bc"].append(float(loss_bc.item()))

        if epoch % args.print_every == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"epoch={epoch:5d} total={total.item():.3e} "
                f"pde={loss_pde.item():.3e} bc={loss_bc.item():.3e}"
            )

    elapsed = time.time() - start_time
    l2_error, _, _, _ = evaluate(model, device=device)

    save_plots(model, output_dir, device, history)

    torch.save(model.state_dict(), output_dir / "model.pt")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "problem": "Poisson on [-1,1]^2 with exact solution sin(pi x) sin(pi y)",
                "epochs": args.epochs,
                "n_interior": args.n_interior,
                "n_boundary": args.n_boundary,
                "hidden_dim": args.hidden_dim,
                "hidden_layers": args.hidden_layers,
                "lr": args.lr,
                "lambda_bc": args.lambda_bc,
                "seed": args.seed,
                "device": str(device),
                "runtime_sec": elapsed,
                "l2_relative_error": l2_error,
            },
            fh,
            indent=2,
        )

    print(f"finished in {elapsed:.2f} sec")
    print(f"L2 relative error: {l2_error:.6e}")
    print(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
