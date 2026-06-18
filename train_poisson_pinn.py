import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn_model import MLP
from poisson_problem import boundary_value, exact_solution, pde_residual


class TimeBudgetExceeded(Exception):
    """Raised when a timed training run reaches its runtime budget."""


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


def append_history(
    history: dict[str, list[float]],
    total: torch.Tensor,
    loss_pde: torch.Tensor,
    loss_bc: torch.Tensor,
) -> None:
    history["total"].append(float(total.item()))
    history["pde"].append(float(loss_pde.item()))
    history["bc"].append(float(loss_bc.item()))


def log_progress(label: str, step: int, total: torch.Tensor, loss_pde: torch.Tensor, loss_bc: torch.Tensor) -> None:
    print(
        f"{label}={step:5d} total={total.item():.3e} "
        f"pde={loss_pde.item():.3e} bc={loss_bc.item():.3e}"
    )


def runtime_exceeded(start_time: float, max_runtime_sec: float | None) -> bool:
    return max_runtime_sec is not None and (time.time() - start_time) >= max_runtime_sec


def parse_dtype(raw_dtype: str) -> torch.dtype:
    if raw_dtype == "float32":
        return torch.float32
    if raw_dtype == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype: {raw_dtype}")


def parse_tags(raw_tags: str) -> list[str]:
    return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]


def init_wandb(args: argparse.Namespace, device: torch.device) -> Any:
    if args.wandb_mode == "disabled":
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging was requested, but wandb is not installed. "
            "Install it with `pip install wandb` or run with `--wandb-mode disabled`."
        ) from exc

    config = {
        "stage": "poisson-baseline",
        "pde": "Poisson",
        "problem_setting": "Omega=[-1,1]^2, Dirichlet boundary, exact solution sin(pi x) sin(pi y)",
        "domain": "[-1,1] x [-1,1]",
        "boundary_condition": "Dirichlet from exact solution",
        "optimizer": "Adam" if args.lbfgs_steps == 0 else "Adam->L-BFGS",
        "network_depth": args.hidden_layers,
        "network_width": args.hidden_dim,
        "activation": "tanh",
        "epochs": args.epochs,
        "max_runtime_sec": args.max_runtime_sec,
        "adam_max_runtime_sec": args.adam_max_runtime_sec,
        "collocation_n": args.n_interior,
        "boundary_n": args.n_boundary,
        "lr": args.lr,
        "bc_weight": args.lambda_bc,
        "lbfgs_steps": args.lbfgs_steps,
        "lbfgs_lr": args.lbfgs_lr,
        "lbfgs_history_size": args.lbfgs_history_size,
        "seed": args.seed,
        "device": str(device),
        "dtype": args.dtype,
        "eval_grid_size": args.eval_grid_size,
        "feature_map": "none",
        "gfpinns_enabled": False,
    }

    return wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        tags=parse_tags(args.wandb_tags),
        notes=args.wandb_notes,
        config=config,
        mode=args.wandb_mode,
        dir=str(args.output_dir),
    )


def log_wandb_metrics(
    run: Any,
    epoch: int,
    total: torch.Tensor,
    loss_pde: torch.Tensor,
    loss_bc: torch.Tensor,
    l2_error: float | None = None,
    epoch_sec: float | None = None,
) -> None:
    if run is None:
        return

    payload = {
        "epoch": epoch,
        "train/total_loss": float(total.item()),
        "train/pde_loss": float(loss_pde.item()),
        "train/bc_loss": float(loss_bc.item()),
    }
    if l2_error is not None:
        payload["eval/l2_relative_error"] = l2_error
    if epoch_sec is not None:
        payload["time/epoch_sec"] = epoch_sec
    if torch.cuda.is_available():
        payload["system/gpu_mem_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
    run.log(payload)


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

    save_history_csv(history, output_dir / "history.csv")
    save_smoothed_loss_plot(history, output_dir / "training_losses_smoothed.png")


def save_history_csv(history: dict[str, list[float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "total_loss", "pde_loss", "bc_loss"])
        for step, (total, pde, bc) in enumerate(zip(history["total"], history["pde"], history["bc"]), start=1):
            writer.writerow([step, total, pde, bc])


def moving_average(values: list[float], window: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return array
    window = max(1, min(window, len(array)))
    kernel = np.ones(window) / window
    return np.convolve(array, kernel, mode="valid")


def save_smoothed_loss_plot(history: dict[str, list[float]], path: Path) -> None:
    length = len(history["total"])
    if length == 0:
        return

    window = max(1, min(length // 200, 500))
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in [("total", "total"), ("pde", "pde"), ("bc", "bc")]:
        smoothed = moving_average(history[key], window)
        x = np.arange(window, window + len(smoothed))
        ax.plot(x, smoothed, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("moving-average loss")
    ax.legend()
    ax.set_title(f"Training losses, moving average window={window}")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline PINN for the rectangular Poisson problem.")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--max-runtime-sec", type=float, default=None, help="stop training after this many seconds")
    parser.add_argument("--adam-max-runtime-sec", type=float, default=None, help="stop the Adam phase after this many seconds")
    parser.add_argument("--n-interior", type=int, default=2048)
    parser.add_argument("--n-boundary", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-bc", type=float, default=1.0)
    parser.add_argument("--lbfgs-steps", type=int, default=0)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-history-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/poisson_baseline"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--eval-grid-size", type=int, default=101)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.getenv("WANDB_MODE", "offline"))
    parser.add_argument("--wandb-project", type=str, default="pinns-thesis")
    parser.add_argument("--wandb-group", type=str, default="poisson-baseline")
    parser.add_argument("--wandb-tags", type=str, default="poisson,baseline")
    parser.add_argument("--wandb-notes", type=str, default="Baseline Poisson PINN run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    torch.set_default_dtype(dtype)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MLP(hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run = init_wandb(args, device)

    history = {"total": [], "pde": [], "bc": []}
    start_time = time.time()
    completed_epochs = 0

    for epoch in range(1, args.epochs + 1):
        if runtime_exceeded(start_time, args.max_runtime_sec):
            print(f"stopping Adam phase at epoch={epoch - 1} due to max runtime")
            break
        if runtime_exceeded(start_time, args.adam_max_runtime_sec):
            print(f"stopping Adam phase at epoch={epoch - 1} due to Adam runtime budget")
            break

        epoch_start = time.time()
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
        append_history(history, total, loss_pde, loss_bc)
        completed_epochs = epoch
        log_wandb_metrics(run, epoch, total, loss_pde, loss_bc, epoch_sec=time.time() - epoch_start)

        if epoch % args.print_every == 0 or epoch == 1 or epoch == args.epochs:
            log_progress("epoch", epoch, total, loss_pde, loss_bc)

    if args.lbfgs_steps > 0:
        x_f_fixed, y_f_fixed = sample_interior(args.n_interior, device)
        x_bc_fixed, y_bc_fixed = sample_boundary(args.n_boundary, device)
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=args.lbfgs_lr,
            max_iter=args.lbfgs_steps,
            history_size=args.lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )
        lbfgs_state = {"step": 0}

        def closure() -> torch.Tensor:
            if runtime_exceeded(start_time, args.max_runtime_sec):
                raise TimeBudgetExceeded

            lbfgs.zero_grad(set_to_none=True)
            residual = pde_residual(model, x_f_fixed, y_f_fixed)
            loss_pde = torch.mean(residual**2)

            coords_bc = torch.cat([x_bc_fixed, y_bc_fixed], dim=1)
            u_bc = model(coords_bc)
            target_bc = boundary_value(x_bc_fixed, y_bc_fixed)
            loss_bc = torch.mean((u_bc - target_bc) ** 2)
            total = loss_pde + args.lambda_bc * loss_bc
            total.backward()

            append_history(history, total, loss_pde, loss_bc)
            lbfgs_state["step"] += 1

            if lbfgs_state["step"] == 1 or lbfgs_state["step"] % args.print_every == 0 or lbfgs_state["step"] == args.lbfgs_steps:
                log_progress("lbfgs", lbfgs_state["step"], total, loss_pde, loss_bc)

            log_wandb_metrics(run, args.epochs + lbfgs_state["step"], total, loss_pde, loss_bc)
            return total

        try:
            lbfgs.step(closure)
        except TimeBudgetExceeded:
            print(f"stopping L-BFGS phase at step={lbfgs_state['step']} due to max runtime")

    elapsed = time.time() - start_time
    final_total, final_loss_pde, final_loss_bc = compute_losses(
        model=model,
        n_interior=args.n_interior,
        n_boundary=args.n_boundary,
        device=device,
        lambda_bc=args.lambda_bc,
    )
    l2_error, _, _, _ = evaluate(model, device=device, grid_size=args.eval_grid_size)

    save_plots(model, output_dir, device, history, grid_size=args.eval_grid_size)

    model_path = output_dir / "model.pt"
    metrics_path = output_dir / "metrics.json"
    torch.save(model.state_dict(), model_path)
    metrics = {
        "problem": "Poisson on [-1,1]^2 with exact solution sin(pi x) sin(pi y)",
        "epochs": args.epochs,
        "completed_epochs": completed_epochs,
        "max_runtime_sec": args.max_runtime_sec,
        "adam_max_runtime_sec": args.adam_max_runtime_sec,
        "n_interior": args.n_interior,
        "n_boundary": args.n_boundary,
        "hidden_dim": args.hidden_dim,
        "hidden_layers": args.hidden_layers,
        "lr": args.lr,
        "lambda_bc": args.lambda_bc,
        "lbfgs_steps": args.lbfgs_steps,
        "completed_lbfgs_steps": lbfgs_state["step"] if args.lbfgs_steps > 0 else 0,
        "lbfgs_lr": args.lbfgs_lr,
        "lbfgs_history_size": args.lbfgs_history_size,
        "seed": args.seed,
        "device": str(device),
        "dtype": args.dtype,
        "eval_grid_size": args.eval_grid_size,
        "runtime_sec": elapsed,
        "l2_relative_error": l2_error,
        "final_total_loss": float(final_total.item()),
        "final_pde_loss": float(final_loss_pde.item()),
        "final_bc_loss": float(final_loss_bc.item()),
        "wandb_mode": args.wandb_mode,
    }
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    if run is not None:
        run.summary["runtime_sec"] = elapsed
        run.summary["l2_relative_error"] = l2_error
        run.summary["output_dir"] = str(output_dir.resolve())
        log_wandb_metrics(
            run,
            args.epochs + args.lbfgs_steps,
            final_total,
            final_loss_pde,
            final_loss_bc,
            l2_error=l2_error,
        )
        run.log_artifact(str(model_path), name=f"{args.wandb_group}-model-seed{args.seed}", type="model")
        run.log_artifact(str(metrics_path), name=f"{args.wandb_group}-metrics-seed{args.seed}", type="metrics")
        run.log_artifact(str(output_dir / "poisson_fields.png"), name=f"{args.wandb_group}-fields-seed{args.seed}", type="plot")
        run.log_artifact(str(output_dir / "training_losses.png"), name=f"{args.wandb_group}-losses-seed{args.seed}", type="plot")
        run.log_artifact(str(output_dir / "training_losses_smoothed.png"), name=f"{args.wandb_group}-smoothed-losses-seed{args.seed}", type="plot")
        run.log_artifact(str(output_dir / "history.csv"), name=f"{args.wandb_group}-history-seed{args.seed}", type="history")
        run.finish()

    print(f"finished in {elapsed:.2f} sec")
    print(f"L2 relative error: {l2_error:.6e}")
    print(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
