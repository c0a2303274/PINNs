import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from hard_constraints import AffineEqualityProjection, HardNet, affine_violation
from pinn_model import MLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_field(coords: torch.Tensor) -> torch.Tensor:
    x = coords[:, :1]
    y = coords[:, 1:]
    u = torch.sin(torch.pi * x) * torch.cos(torch.pi * y)
    v = -u
    return torch.cat([u, v], dim=1)


def sample_points(n_points: int, device: torch.device) -> torch.Tensor:
    return torch.empty(n_points, 2, device=device).uniform_(-1.0, 1.0)


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    base = MLP(in_dim=2, hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers, out_dim=2)
    if args.method == "hardnet":
        projection = AffineEqualityProjection(
            a_matrix=torch.tensor([[1.0, 1.0]]),
            b_vector=torch.tensor([0.0]),
        )
        model = HardNet(base, projection)
    else:
        model = base
    return model.to(device)


def evaluate(model: torch.nn.Module, device: torch.device, grid_size: int) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    grid = torch.linspace(-1.0, 1.0, grid_size, device=device)
    xx, yy = torch.meshgrid(grid, grid, indexing="xy")
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    truth = target_field(coords)
    with torch.no_grad():
        pred = model(coords)

    l2_error = torch.linalg.norm(pred - truth) / torch.linalg.norm(truth)
    a_matrix = torch.tensor([[1.0, 1.0]], device=device)
    b_vector = torch.tensor([0.0], device=device)
    max_violation = affine_violation(pred, a_matrix, b_vector).abs().max()
    return (
        float(l2_error.item()),
        float(max_violation.item()),
        xx.cpu().numpy(),
        yy.cpu().numpy(),
        pred.cpu().numpy(),
    )


def save_history_csv(history: dict[str, list[float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "loss", "max_constraint_violation"])
        for epoch, (loss, violation) in enumerate(zip(history["loss"], history["violation"]), start=1):
            writer.writerow([epoch, loss, violation])


def save_plots(
    model: torch.nn.Module,
    output_dir: Path,
    device: torch.device,
    history: dict[str, list[float]],
    grid_size: int,
) -> tuple[float, float]:
    l2_error, max_violation, xx, yy, pred = evaluate(model, device, grid_size)
    coords = torch.as_tensor(np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1), device=device, dtype=torch.get_default_dtype())
    truth = target_field(coords).detach().cpu().numpy()
    u_pred = pred[:, 0].reshape(grid_size, grid_size)
    u_truth = truth[:, 0].reshape(grid_size, grid_size)
    constraint = (pred[:, 0] + pred[:, 1]).reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, title in zip(
        axes,
        [u_truth, u_pred, constraint],
        ["target u", "predicted u", f"constraint u+v, max={max_violation:.1e}"],
    ):
        image = ax.imshow(data, origin="lower", extent=(-1, 1, -1, 1), cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle(f"Vector field fit, L2={l2_error:.2e}")
    fig.tight_layout()
    fig.savefig(output_dir / "vector_field.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["loss"], label="fit loss")
    ax.plot(history["violation"], label="max |u+v|")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.legend()
    ax.set_title("Training history")
    fig.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    save_history_csv(history, output_dir / "history.csv")
    return l2_error, max_violation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train soft MLP vs HardNet on a constrained vector field.")
    parser.add_argument("--method", choices=["soft", "hardnet"], default="hardnet")
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--max-runtime-sec", type=float, default=None)
    parser.add_argument("--n-points", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-grid-size", type=int, default=101)
    parser.add_argument("--print-every", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hardnet_vector_field"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    a_matrix = torch.tensor([[1.0, 1.0]], device=device)
    b_vector = torch.tensor([0.0], device=device)
    history = {"loss": [], "violation": []}
    start_time = time.time()
    completed_epochs = 0

    for epoch in range(1, args.epochs + 1):
        if args.max_runtime_sec is not None and time.time() - start_time >= args.max_runtime_sec:
            print(f"stopping at epoch={epoch - 1} due to max runtime")
            break
        coords = sample_points(args.n_points, device)
        truth = target_field(coords)
        pred = model(coords)
        loss = torch.mean((pred - truth) ** 2)
        violation = affine_violation(pred, a_matrix, b_vector).abs().max()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        history["loss"].append(float(loss.item()))
        history["violation"].append(float(violation.item()))
        completed_epochs = epoch
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch:6d} loss={loss.item():.3e} max_violation={violation.item():.3e}")

    runtime_sec = time.time() - start_time
    l2_error, max_violation = save_plots(model, output_dir, device, history, args.eval_grid_size)
    metrics = {
        "method": args.method,
        "constraint": "u + v = 0",
        "epochs": args.epochs,
        "completed_epochs": completed_epochs,
        "max_runtime_sec": args.max_runtime_sec,
        "n_points": args.n_points,
        "hidden_dim": args.hidden_dim,
        "hidden_layers": args.hidden_layers,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "runtime_sec": runtime_sec,
        "l2_relative_error": l2_error,
        "max_constraint_violation": max_violation,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"finished in {runtime_sec:.2f} sec")
    print(f"L2 relative error: {l2_error:.6e}")
    print(f"max constraint violation: {max_violation:.6e}")
    print(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
