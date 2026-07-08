import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from hard_constraints import HardNetPlusPlus, NonlinearEqualityProjection
from pinn_model import MLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unit_circle_constraint(y: torch.Tensor) -> torch.Tensor:
    return torch.sum(y**2, dim=1, keepdim=True) - 1.0


def target_function(x: torch.Tensor) -> torch.Tensor:
    angle = 2.0 * torch.pi * x
    return torch.cat([torch.cos(angle), torch.sin(angle)], dim=1)


def sample_points(n_points: int, device: torch.device) -> torch.Tensor:
    return torch.empty(n_points, 1, device=device).uniform_(0.0, 1.0)


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    base = MLP(in_dim=1, hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers, out_dim=2)
    if args.method == "hardnetpp":
        projection = NonlinearEqualityProjection(
            unit_circle_constraint,
            iterations=args.projection_iterations,
            damping=args.projection_damping,
            ridge=args.projection_ridge,
            max_step_norm=args.projection_max_step_norm,
        )
        model = HardNetPlusPlus(base, projection)
    else:
        model = base
    return model.to(device)


def evaluate(model: torch.nn.Module, device: torch.device, grid_size: int) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    x = torch.linspace(0.0, 1.0, grid_size, device=device).reshape(-1, 1)
    truth = target_function(x)
    pred = model(x)
    l2_error = torch.linalg.norm(pred - truth) / torch.linalg.norm(truth)
    max_violation = unit_circle_constraint(pred).abs().max()
    return (
        float(l2_error.item()),
        float(max_violation.item()),
        x.detach().cpu().numpy(),
        pred.detach().cpu().numpy(),
        truth.detach().cpu().numpy(),
    )


def save_outputs(
    model: torch.nn.Module,
    output_dir: Path,
    device: torch.device,
    history: dict[str, list[float]],
    grid_size: int,
) -> tuple[float, float]:
    l2_error, max_violation, x, pred, truth = evaluate(model, device, grid_size)
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "loss", "max_constraint_violation"])
        for epoch, (loss, violation) in enumerate(zip(history["loss"], history["violation"]), start=1):
            writer.writerow([epoch, loss, violation])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(truth[:, 0], truth[:, 1], label="target circle")
    axes[0].plot(pred[:, 0], pred[:, 1], "--", label="prediction")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title(f"Unit-circle fit, L2={l2_error:.2e}")
    axes[0].set_xlabel("y0")
    axes[0].set_ylabel("y1")
    axes[0].legend()

    axes[1].plot(history["loss"], label="fit loss")
    axes[1].plot(history["violation"], label="max |norm^2-1|")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].set_title("Training history")
    fig.tight_layout()
    fig.savefig(output_dir / "hardnetpp_circle_demo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return l2_error, max_violation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a HardNet++ nonlinear equality demo.")
    parser.add_argument("--method", choices=["soft", "hardnetpp"], default="hardnetpp")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--max-runtime-sec", type=float, default=None)
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--projection-iterations", type=int, default=15)
    parser.add_argument("--projection-damping", type=float, default=1.0)
    parser.add_argument("--projection-ridge", type=float, default=1.0e-6)
    parser.add_argument("--projection-max-step-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-grid-size", type=int, default=201)
    parser.add_argument("--print-every", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hardnetpp_circle_demo"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {"loss": [], "violation": []}
    start_time = time.time()
    completed_epochs = 0

    for epoch in range(1, args.epochs + 1):
        if args.max_runtime_sec is not None and time.time() - start_time >= args.max_runtime_sec:
            print(f"stopping at epoch={epoch - 1} due to max runtime")
            break
        x = sample_points(args.n_points, device)
        truth = target_function(x)
        pred = model(x)
        loss = torch.mean((pred - truth) ** 2)
        violation = unit_circle_constraint(pred).abs().max()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        history["loss"].append(float(loss.item()))
        history["violation"].append(float(violation.item()))
        completed_epochs = epoch
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch:6d} loss={loss.item():.3e} max_violation={violation.item():.3e}")

    runtime_sec = time.time() - start_time
    l2_error, max_violation = save_outputs(model, output_dir, device, history, args.eval_grid_size)
    metrics = {
        "method": args.method,
        "constraint": "y0^2 + y1^2 = 1",
        "epochs": args.epochs,
        "completed_epochs": completed_epochs,
        "max_runtime_sec": args.max_runtime_sec,
        "n_points": args.n_points,
        "hidden_dim": args.hidden_dim,
        "hidden_layers": args.hidden_layers,
        "lr": args.lr,
        "projection_iterations": args.projection_iterations,
        "projection_damping": args.projection_damping,
        "projection_ridge": args.projection_ridge,
        "projection_max_step_norm": args.projection_max_step_norm,
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
