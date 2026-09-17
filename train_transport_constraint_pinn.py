import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from hard_constraints import AffineEqualityProjection, HardNet, HardNetPlusPlus, NonlinearEqualityProjection
from pinn_model import MLP
from transport_constraint_problem import (
    advection_residual,
    advection_residual_from_prediction,
    constraint_residual,
    exact_solution,
    sample_initial,
    sample_interior,
    sample_periodic_boundary,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_dtype(raw_dtype: str) -> torch.dtype:
    if raw_dtype == "float32":
        return torch.float32
    if raw_dtype == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype: {raw_dtype}")


def circle_constraint(y: torch.Tensor) -> torch.Tensor:
    return constraint_residual(y, "circle")


def build_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    base = MLP(in_dim=2, hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers, out_dim=2)
    if args.method == "soft":
        model = base
    elif args.method == "hardnet":
        if args.constraint != "linear":
            raise ValueError("hardnet requires --constraint linear")
        projection = AffineEqualityProjection(
            a_matrix=torch.tensor([[1.0, 1.0]]),
            b_vector=torch.tensor([0.0]),
        )
        model = HardNet(base, projection)
    elif args.method == "hardnetpp":
        if args.constraint != "circle":
            raise ValueError("hardnetpp requires --constraint circle")
        projection = NonlinearEqualityProjection(
            circle_constraint,
            iterations=args.projection_iterations,
            damping=args.projection_damping,
            ridge=args.projection_ridge,
            max_step_norm=args.projection_max_step_norm,
        )
        model = HardNetPlusPlus(base, projection)
    else:
        raise ValueError(f"unsupported method: {args.method}")
    return model.to(device=device, dtype=dtype)


def compute_losses(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    interior = sample_interior(args.n_interior, device).requires_grad_(True)
    prediction = model(interior)
    residual = advection_residual_from_prediction(prediction, interior, args.wave_speed)
    loss_pde = torch.mean(residual**2)
    loss_constraint = torch.mean(constraint_residual(prediction, args.constraint) ** 2)

    initial = sample_initial(args.n_initial, device)
    initial_prediction = model(initial)
    loss_ic = torch.mean((initial_prediction - exact_solution(initial, args.constraint, args.wave_speed)) ** 2)

    left, right = sample_periodic_boundary(args.n_boundary, device)
    loss_bc = torch.mean((model(left) - model(right)) ** 2)

    constraint_weight = args.lambda_constraint if args.method == "soft" else 0.0
    total = (
        loss_pde
        + args.lambda_ic * loss_ic
        + args.lambda_bc * loss_bc
        + constraint_weight * loss_constraint
    )
    return total, loss_pde, loss_ic, loss_bc, loss_constraint


def evaluate(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis = torch.linspace(0.0, 1.0, args.eval_grid_size, device=device)
    tt, xx = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=1)
    with torch.no_grad():
        prediction = model(coords)
        truth = exact_solution(coords, args.constraint, args.wave_speed)
        l2_error = torch.linalg.norm(prediction - truth) / torch.linalg.norm(truth)
        violation = constraint_residual(prediction, args.constraint).abs().max()

    residual_points = sample_interior(args.eval_residual_points, device)
    residual = advection_residual(model, residual_points, args.wave_speed)
    pde_rmse = torch.sqrt(torch.mean(residual**2))
    return (
        float(l2_error.item()),
        float(violation.item()),
        float(pde_rmse.item()),
        tt.cpu().numpy(),
        xx.cpu().numpy(),
        prediction.detach().cpu().numpy(),
        truth.detach().cpu().numpy(),
    )


def measure_inference(
    model: torch.nn.Module,
    device: torch.device,
    grid_size: int,
    repeats: int,
) -> tuple[float, float]:
    axis = torch.linspace(0.0, 1.0, grid_size, device=device)
    tt, xx = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=1)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(coords)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(coords)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if was_training:
        model.train()
    return elapsed * 1000.0 / repeats, coords.shape[0] * repeats / elapsed


def save_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    history: dict[str, list[float]],
    tt: np.ndarray,
    xx: np.ndarray,
    prediction: np.ndarray,
    truth: np.ndarray,
) -> None:
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "total_loss", "pde_loss", "ic_loss", "bc_loss", "constraint_loss"])
        for epoch, values in enumerate(zip(*(history[key] for key in ["total", "pde", "ic", "bc", "constraint"])), start=1):
            writer.writerow([epoch, *values])

    grid_size = args.eval_grid_size
    truth_first = truth[:, 0].reshape(grid_size, grid_size)
    prediction_first = prediction[:, 0].reshape(grid_size, grid_size)
    pointwise_error = np.linalg.norm(prediction - truth, axis=1).reshape(grid_size, grid_size)
    if args.constraint == "linear":
        constraint_map = (prediction[:, 0] + prediction[:, 1]).reshape(grid_size, grid_size)
    else:
        constraint_map = (np.sum(prediction**2, axis=1) - 1.0).reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    datasets = [truth_first, prediction_first, pointwise_error, np.abs(constraint_map)]
    titles = ["Reference y0", "Prediction y0", "Pointwise error", "Absolute constraint violation"]
    for ax, data, title in zip(axes, datasets, titles):
        image = ax.imshow(data, origin="lower", extent=(0, 1, 0, 1), aspect="auto", cmap="viridis")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "transport_fields.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for key, values in history.items():
        ax.plot(values, label=key)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.set_title("Training history")
    fig.tight_layout()
    fig.savefig(output_dir / "training_losses.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def init_wandb(args: argparse.Namespace, output_dir: Path) -> Any:
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install wandb or use --wandb-mode disabled") from exc
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return wandb.init(
        project=args.wandb_project,
        group=f"transport-{args.constraint}",
        mode=args.wandb_mode,
        dir=str(output_dir),
        config=config,
        tags=["transport", "pde-bridge", args.constraint, args.method],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Connect HardNet or HardNet++ to an analytic transport PDE.")
    parser.add_argument("--constraint", choices=["linear", "circle"], required=True)
    parser.add_argument("--method", choices=["soft", "hardnet", "hardnetpp"], required=True)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--max-runtime-sec", type=float, default=None)
    parser.add_argument("--n-interior", type=int, default=1024)
    parser.add_argument("--n-initial", type=int, default=256)
    parser.add_argument("--n-boundary", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--lambda-ic", type=float, default=1.0)
    parser.add_argument("--lambda-bc", type=float, default=1.0)
    parser.add_argument("--lambda-constraint", type=float, default=1.0)
    parser.add_argument("--wave-speed", type=float, default=1.0)
    parser.add_argument("--projection-iterations", type=int, default=15)
    parser.add_argument("--projection-damping", type=float, default=1.0)
    parser.add_argument("--projection-ridge", type=float, default=1.0e-6)
    parser.add_argument("--projection-max-step-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--eval-grid-size", type=int, default=101)
    parser.add_argument("--eval-residual-points", type=int, default=4096)
    parser.add_argument("--inference-repeats", type=int, default=20)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.getenv("WANDB_MODE", "disabled"))
    parser.add_argument("--wandb-project", type=str, default="pinns-thesis")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/transport_constraint_pinn"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    torch.set_default_dtype(dtype)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args, device, dtype)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run = init_wandb(args, output_dir)
    history = {key: [] for key in ["total", "pde", "ic", "bc", "constraint"]}
    start_time = time.time()
    completed_epochs = 0

    for epoch in range(1, args.epochs + 1):
        if args.max_runtime_sec is not None and time.time() - start_time >= args.max_runtime_sec:
            break
        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses(model, args, device)
        losses[0].backward()
        optimizer.step()
        for key, value in zip(history, losses):
            history[key].append(float(value.item()))
        completed_epochs = epoch
        if run is not None:
            run.log({f"train/{key}": float(value.item()) for key, value in zip(history, losses)}, step=epoch)
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:6d} total={losses[0].item():.3e} pde={losses[1].item():.3e} "
                f"ic={losses[2].item():.3e} bc={losses[3].item():.3e} constraint={losses[4].item():.3e}"
            )

    runtime_sec = time.time() - start_time
    final_losses = compute_losses(model, args, device)
    l2_error, max_violation, pde_rmse, tt, xx, prediction, truth = evaluate(model, args, device)
    save_outputs(output_dir, args, history, tt, xx, prediction, truth)
    inference_time_ms, inference_points_per_sec = measure_inference(
        model, device, args.eval_grid_size, args.inference_repeats
    )
    peak_gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
    metrics = {
        "problem": "two-component 1D periodic advection",
        "equation": "y_t + c y_x = 0",
        "constraint": args.constraint,
        "method": args.method,
        "epochs": args.epochs,
        "completed_epochs": completed_epochs,
        "max_runtime_sec": args.max_runtime_sec,
        "n_interior": args.n_interior,
        "n_initial": args.n_initial,
        "n_boundary": args.n_boundary,
        "hidden_dim": args.hidden_dim,
        "hidden_layers": args.hidden_layers,
        "parameter_count": parameter_count,
        "lr": args.lr,
        "lambda_ic": args.lambda_ic,
        "lambda_bc": args.lambda_bc,
        "lambda_constraint": args.lambda_constraint,
        "wave_speed": args.wave_speed,
        "projection_iterations": args.projection_iterations if args.method == "hardnetpp" else None,
        "seed": args.seed,
        "device": str(device),
        "dtype": args.dtype,
        "runtime_sec": runtime_sec,
        "l2_relative_error": l2_error,
        "pde_residual_rmse": pde_rmse,
        "max_constraint_violation": max_violation,
        "final_total_loss": float(final_losses[0].item()),
        "final_pde_loss": float(final_losses[1].item()),
        "final_ic_loss": float(final_losses[2].item()),
        "final_bc_loss": float(final_losses[3].item()),
        "final_constraint_loss": float(final_losses[4].item()),
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "inference_time_ms": inference_time_ms,
        "inference_points_per_sec": inference_points_per_sec,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    torch.save(model.state_dict(), output_dir / "model.pt")

    if run is not None:
        run.summary.update(metrics)
        run.finish()
    print(f"L2 relative error: {l2_error:.6e}")
    print(f"PDE residual RMSE: {pde_rmse:.6e}")
    print(f"max constraint violation: {max_violation:.6e}")
    print(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
