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


def target_function(x: torch.Tensor) -> torch.Tensor:
    y0 = torch.sin(torch.pi * x)
    y1 = 1.0 - y0
    return torch.cat([y0, y1], dim=1)


def sample_points(n_points: int, device: torch.device) -> torch.Tensor:
    return torch.empty(n_points, 1, device=device).uniform_(-1.0, 1.0)


def evaluate(model: torch.nn.Module, device: torch.device, grid_size: int) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    x = torch.linspace(-1.0, 1.0, grid_size, device=device).reshape(-1, 1)
    truth = target_function(x)
    with torch.no_grad():
        pred = model(x)
    l2_error = torch.linalg.norm(pred - truth) / torch.linalg.norm(truth)
    a_matrix = torch.tensor([[1.0, 1.0]], device=device)
    b_vector = torch.tensor([1.0], device=device)
    max_violation = affine_violation(pred, a_matrix, b_vector).abs().max()
    return (
        float(l2_error.item()),
        float(max_violation.item()),
        x.cpu().numpy(),
        pred.cpu().numpy(),
        truth.cpu().numpy(),
    )


def save_outputs(
    history: dict[str, list[float]],
    model: torch.nn.Module,
    output_dir: Path,
    device: torch.device,
    grid_size: int,
) -> tuple[float, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    l2_error, max_violation, x, pred, truth = evaluate(model, device, grid_size)

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "loss", "max_constraint_violation"])
        for epoch, (loss, violation) in enumerate(zip(history["loss"], history["violation"]), start=1):
            writer.writerow([epoch, loss, violation])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(x, truth[:, 0], label="target y0")
    axes[0].plot(x, pred[:, 0], "--", label="HardNet y0")
    axes[0].plot(x, truth[:, 1], label="target y1")
    axes[0].plot(x, pred[:, 1], "--", label="HardNet y1")
    axes[0].set_title(f"Prediction, L2={l2_error:.2e}")
    axes[0].set_xlabel("x")
    axes[0].legend()

    axes[1].plot(history["loss"], label="fit loss")
    axes[1].plot(history["violation"], label="max |Ay-b|")
    axes[1].set_yscale("log")
    axes[1].set_title("Training history")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "hardnet_affine_demo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return l2_error, max_violation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a HardNet affine-equality demo.")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--n-points", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-grid-size", type=int, default=201)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hardnet_affine_demo"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base = MLP(in_dim=1, hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers, out_dim=2)
    projection = AffineEqualityProjection(
        a_matrix=torch.tensor([[1.0, 1.0]]),
        b_vector=torch.tensor([1.0]),
    )
    model = HardNet(base, projection).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {"loss": [], "violation": []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        x = sample_points(args.n_points, device)
        truth = target_function(x)
        pred = model(x)
        loss = torch.mean((pred - truth) ** 2)
        violation = model.projection.violation(pred).abs().max()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        history["loss"].append(float(loss.item()))
        history["violation"].append(float(violation.item()))
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch:5d} loss={loss.item():.3e} max_violation={violation.item():.3e}")

    runtime_sec = time.time() - start_time
    l2_error, max_violation = save_outputs(history, model, output_dir, device, args.eval_grid_size)
    metrics = {
        "method": "HardNet affine equality projection",
        "constraint": "y0 + y1 = 1",
        "epochs": args.epochs,
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
