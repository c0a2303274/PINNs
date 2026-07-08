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

from burgers_problem import boundary_value, finite_difference_reference, initial_condition, pde_residual
from burgers_models import BoundedHardICBCBurgersModel, HardICBCBurgersModel
from pinn_model import MLP


class TimeBudgetExceeded(Exception):
    """Raised when a timed training run reaches its runtime budget."""


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


def sample_interior(
    n_points: int,
    device: torch.device,
    sampling: str = "uniform",
    focus_fraction: float = 0.5,
    focus_std: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.empty(n_points, 1, device=device).uniform_(0.0, 1.0)
    if sampling == "uniform":
        x = torch.empty(n_points, 1, device=device).uniform_(-1.0, 1.0)
    elif sampling == "shock-focused":
        n_focus = int(n_points * focus_fraction)
        n_uniform = n_points - n_focus
        x_uniform = torch.empty(n_uniform, 1, device=device).uniform_(-1.0, 1.0)
        x_focus = torch.randn(n_focus, 1, device=device) * focus_std
        x_focus = torch.clamp(x_focus, -1.0, 1.0)
        x = torch.cat([x_uniform, x_focus], dim=0)
    else:
        raise ValueError(f"unsupported sampling: {sampling}")
    return t, x


def sample_initial(n_points: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.zeros(n_points, 1, device=device)
    x = torch.empty(n_points, 1, device=device).uniform_(-1.0, 1.0)
    return t, x


def sample_boundary(n_points: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    n_side = max(n_points // 2, 1)
    t_left = torch.empty(n_side, 1, device=device).uniform_(0.0, 1.0)
    t_right = torch.empty(n_side, 1, device=device).uniform_(0.0, 1.0)
    x = torch.cat([-torch.ones_like(t_left), torch.ones_like(t_right)], dim=0)
    t = torch.cat([t_left, t_right], dim=0)
    return t, x


def compute_losses(
    model: MLP,
    n_interior: int,
    n_initial: int,
    n_boundary: int,
    device: torch.device,
    nu: float,
    lambda_ic: float,
    lambda_bc: float,
    sampling: str,
    focus_fraction: float,
    focus_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t_f, x_f = sample_interior(n_interior, device, sampling=sampling, focus_fraction=focus_fraction, focus_std=focus_std)
    residual = pde_residual(model, t_f, x_f, nu)
    loss_pde = torch.mean(residual**2)

    t_ic, x_ic = sample_initial(n_initial, device)
    u_ic = model(torch.cat([t_ic, x_ic], dim=1))
    loss_ic = torch.mean((u_ic - initial_condition(x_ic)) ** 2)

    t_bc, x_bc = sample_boundary(n_boundary, device)
    u_bc = model(torch.cat([t_bc, x_bc], dim=1))
    loss_bc = torch.mean((u_bc - boundary_value(t_bc, x_bc)) ** 2)

    total = loss_pde + lambda_ic * loss_ic + lambda_bc * loss_bc
    return total, loss_pde, loss_ic, loss_bc


def append_history(
    history: dict[str, list[float]],
    total: torch.Tensor,
    loss_pde: torch.Tensor,
    loss_ic: torch.Tensor,
    loss_bc: torch.Tensor,
) -> None:
    history["total"].append(float(total.item()))
    history["pde"].append(float(loss_pde.item()))
    history["ic"].append(float(loss_ic.item()))
    history["bc"].append(float(loss_bc.item()))


def log_progress(
    label: str,
    step: int,
    total: torch.Tensor,
    loss_pde: torch.Tensor,
    loss_ic: torch.Tensor,
    loss_bc: torch.Tensor,
) -> None:
    print(
        f"{label}={step:6d} total={total.item():.3e} "
        f"pde={loss_pde.item():.3e} ic={loss_ic.item():.3e} bc={loss_bc.item():.3e}"
    )


def runtime_exceeded(start_time: float, max_runtime_sec: float | None) -> bool:
    return max_runtime_sec is not None and (time.time() - start_time) >= max_runtime_sec


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
        "stage": "burgers-baseline",
        "constraint_mode": args.constraint_mode,
        "pde": "1D viscous Burgers",
        "problem_setting": "x=[-1,1], t=[0,1], u_t + u u_x = nu u_xx, u(x,0)=-sin(pi x), u(-1,t)=u(1,t)=0",
        "domain": "t=[0,1], x=[-1,1]",
        "initial_condition": "u(x,0)=-sin(pi x)",
        "boundary_condition": "Dirichlet zero at x=-1 and x=1",
        "reference_solution": "explicit finite-difference solver in burgers_problem.py",
        "optimizer": "Adam" if args.lbfgs_steps == 0 else "Adam->L-BFGS",
        "network_depth": args.hidden_layers,
        "network_width": args.hidden_dim,
        "activation": "tanh",
        "epochs": args.epochs,
        "max_runtime_sec": args.max_runtime_sec,
        "adam_max_runtime_sec": args.adam_max_runtime_sec,
        "collocation_n": args.n_interior,
        "initial_n": args.n_initial,
        "boundary_n": args.n_boundary,
        "lr": args.lr,
        "ic_weight": args.lambda_ic,
        "bc_weight": args.lambda_bc,
        "nu": args.nu,
        "sampling": args.sampling,
        "focus_fraction": args.focus_fraction,
        "focus_std": args.focus_std,
        "seed": args.seed,
        "device": str(device),
        "dtype": args.dtype,
        "eval_grid_size": args.eval_grid_size,
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
    step: int,
    total: torch.Tensor,
    loss_pde: torch.Tensor,
    loss_ic: torch.Tensor,
    loss_bc: torch.Tensor,
    l2_error: float | None = None,
) -> None:
    if run is None:
        return

    payload = {
        "step": step,
        "train/total_loss": float(total.item()),
        "train/pde_loss": float(loss_pde.item()),
        "train/ic_loss": float(loss_ic.item()),
        "train/bc_loss": float(loss_bc.item()),
    }
    if l2_error is not None:
        payload["eval/l2_relative_error"] = l2_error
    if torch.cuda.is_available():
        payload["system/gpu_mem_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
    run.log(payload)


def evaluate(model: MLP, device: torch.device, nu: float, grid_size: int = 101) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_grid = torch.linspace(0.0, 1.0, grid_size, device=device)
    x_grid = torch.linspace(-1.0, 1.0, grid_size, device=device)
    tt, xx = torch.meshgrid(t_grid, x_grid, indexing="ij")
    coords = torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=1)

    with torch.no_grad():
        pred = model(coords).reshape(grid_size, grid_size)

    truth = finite_difference_reference(
        t_grid.detach().cpu().numpy(),
        x_grid.detach().cpu().numpy(),
        nu=nu,
    )
    truth_tensor = torch.as_tensor(truth, device=device, dtype=pred.dtype)
    l2_error = torch.linalg.norm(pred - truth_tensor) / torch.linalg.norm(truth_tensor)
    return (
        float(l2_error.item()),
        tt.cpu().numpy(),
        xx.cpu().numpy(),
        pred.cpu().numpy(),
        truth,
    )


def save_history_csv(history: dict[str, list[float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "total_loss", "pde_loss", "ic_loss", "bc_loss"])
        for step, (total, pde, ic, bc) in enumerate(
            zip(history["total"], history["pde"], history["ic"], history["bc"]),
            start=1,
        ):
            writer.writerow([step, total, pde, ic, bc])


def moving_average(values: list[float], window: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return array
    window = max(1, min(window, len(array)))
    kernel = np.ones(window) / window
    return np.convolve(array, kernel, mode="valid")


def save_plots(
    model: MLP,
    output_dir: Path,
    device: torch.device,
    history: dict[str, list[float]],
    nu: float,
    grid_size: int = 101,
) -> None:
    l2_error, tt, xx, pred, truth = evaluate(model, device=device, nu=nu, grid_size=grid_size)
    error = pred - truth

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, title in zip(
        axes,
        [truth, pred, error],
        ["Reference solution", "PINN prediction", f"Prediction error, L2={l2_error:.2e}"],
    ):
        image = ax.imshow(data, origin="lower", extent=(-1, 1, 0, 1), aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "burgers_fields.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ["total", "pde", "ic", "bc"]:
        ax.plot(history[key], label=key)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    ax.set_title("Training losses")
    fig.tight_layout()
    fig.savefig(output_dir / "training_losses.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    length = len(history["total"])
    if length > 0:
        window = max(1, min(length // 200, 500))
        fig, ax = plt.subplots(figsize=(7, 4))
        for key in ["total", "pde", "ic", "bc"]:
            smoothed = moving_average(history[key], window)
            x_axis = np.arange(window, window + len(smoothed))
            ax.plot(x_axis, smoothed, label=key)
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("moving-average loss")
        ax.legend()
        ax.set_title(f"Training losses, moving average window={window}")
        fig.tight_layout()
        fig.savefig(output_dir / "training_losses_smoothed.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    save_history_csv(history, output_dir / "history.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline PINN for the 1D viscous Burgers equation.")
    parser.add_argument("--constraint-mode", choices=["soft", "hard-icbc", "bounded-hard-icbc"], default="soft")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--max-runtime-sec", type=float, default=None, help="stop training after this many seconds")
    parser.add_argument("--adam-max-runtime-sec", type=float, default=None, help="stop the Adam phase after this many seconds")
    parser.add_argument("--n-interior", type=int, default=4096)
    parser.add_argument("--n-initial", type=int, default=512)
    parser.add_argument("--n-boundary", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-ic", type=float, default=1.0)
    parser.add_argument("--lambda-bc", type=float, default=1.0)
    parser.add_argument("--bound-amplitude", type=float, default=1.0)
    parser.add_argument("--sampling", choices=["uniform", "shock-focused"], default="uniform")
    parser.add_argument("--focus-fraction", type=float, default=0.5)
    parser.add_argument("--focus-std", type=float, default=0.2)
    parser.add_argument("--nu", type=float, default=0.01 / math.pi)
    parser.add_argument("--lbfgs-steps", type=int, default=0)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-history-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/burgers_baseline"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--eval-grid-size", type=int, default=101)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.getenv("WANDB_MODE", "offline"))
    parser.add_argument("--wandb-project", type=str, default="pinns-thesis")
    parser.add_argument("--wandb-group", type=str, default="burgers-baseline")
    parser.add_argument("--wandb-tags", type=str, default="burgers,baseline,nonlinear-pde")
    parser.add_argument("--wandb-notes", type=str, default="Standard PINNs baseline for 1D viscous Burgers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    torch.set_default_dtype(dtype)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.constraint_mode == "hard-icbc":
        model = HardICBCBurgersModel(hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers).to(device=device, dtype=dtype)
    elif args.constraint_mode == "bounded-hard-icbc":
        model = BoundedHardICBCBurgersModel(
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            amplitude=args.bound_amplitude,
        ).to(device=device, dtype=dtype)
    else:
        model = MLP(in_dim=2, hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers, out_dim=1).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run = init_wandb(args, device)
    history = {"total": [], "pde": [], "ic": [], "bc": []}
    start_time = time.time()
    completed_epochs = 0

    for epoch in range(1, args.epochs + 1):
        if runtime_exceeded(start_time, args.max_runtime_sec):
            print(f"stopping Adam phase at epoch={epoch - 1} due to max runtime")
            break
        if runtime_exceeded(start_time, args.adam_max_runtime_sec):
            print(f"stopping Adam phase at epoch={epoch - 1} due to Adam runtime budget")
            break

        optimizer.zero_grad(set_to_none=True)
        total, loss_pde, loss_ic, loss_bc = compute_losses(
            model=model,
            n_interior=args.n_interior,
            n_initial=args.n_initial,
            n_boundary=args.n_boundary,
            device=device,
            nu=args.nu,
            lambda_ic=args.lambda_ic,
            lambda_bc=args.lambda_bc,
            sampling=args.sampling,
            focus_fraction=args.focus_fraction,
            focus_std=args.focus_std,
        )
        total.backward()
        optimizer.step()
        append_history(history, total, loss_pde, loss_ic, loss_bc)
        completed_epochs = epoch
        log_wandb_metrics(run, epoch, total, loss_pde, loss_ic, loss_bc)

        if epoch % args.print_every == 0 or epoch == 1 or epoch == args.epochs:
            log_progress("epoch", epoch, total, loss_pde, loss_ic, loss_bc)

    lbfgs_completed_steps = 0
    if args.lbfgs_steps > 0:
        t_f_fixed, x_f_fixed = sample_interior(
            args.n_interior,
            device,
            sampling=args.sampling,
            focus_fraction=args.focus_fraction,
            focus_std=args.focus_std,
        )
        t_ic_fixed, x_ic_fixed = sample_initial(args.n_initial, device)
        t_bc_fixed, x_bc_fixed = sample_boundary(args.n_boundary, device)
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
            residual = pde_residual(model, t_f_fixed, x_f_fixed, args.nu)
            loss_pde = torch.mean(residual**2)
            u_ic = model(torch.cat([t_ic_fixed, x_ic_fixed], dim=1))
            loss_ic = torch.mean((u_ic - initial_condition(x_ic_fixed)) ** 2)
            u_bc = model(torch.cat([t_bc_fixed, x_bc_fixed], dim=1))
            loss_bc = torch.mean((u_bc - boundary_value(t_bc_fixed, x_bc_fixed)) ** 2)
            total = loss_pde + args.lambda_ic * loss_ic + args.lambda_bc * loss_bc
            total.backward()

            append_history(history, total, loss_pde, loss_ic, loss_bc)
            lbfgs_state["step"] += 1
            if lbfgs_state["step"] == 1 or lbfgs_state["step"] % args.print_every == 0:
                log_progress("lbfgs", lbfgs_state["step"], total, loss_pde, loss_ic, loss_bc)
            log_wandb_metrics(run, args.epochs + lbfgs_state["step"], total, loss_pde, loss_ic, loss_bc)
            return total

        try:
            lbfgs.step(closure)
        except TimeBudgetExceeded:
            print(f"stopping L-BFGS phase at step={lbfgs_state['step']} due to max runtime")
        lbfgs_completed_steps = lbfgs_state["step"]

    elapsed = time.time() - start_time
    final_total, final_loss_pde, final_loss_ic, final_loss_bc = compute_losses(
        model=model,
        n_interior=args.n_interior,
        n_initial=args.n_initial,
        n_boundary=args.n_boundary,
        device=device,
        nu=args.nu,
        lambda_ic=args.lambda_ic,
        lambda_bc=args.lambda_bc,
        sampling=args.sampling,
        focus_fraction=args.focus_fraction,
        focus_std=args.focus_std,
    )
    l2_error, _, _, _, _ = evaluate(model, device=device, nu=args.nu, grid_size=args.eval_grid_size)
    save_plots(model, output_dir, device, history, nu=args.nu, grid_size=args.eval_grid_size)

    model_path = output_dir / "model.pt"
    metrics_path = output_dir / "metrics.json"
    torch.save(model.state_dict(), model_path)
    metrics = {
        "problem": "1D viscous Burgers on t=[0,1], x=[-1,1]",
        "constraint_mode": args.constraint_mode,
        "equation": "u_t + u u_x = nu u_xx",
        "initial_condition": "u(x,0)=-sin(pi x)",
        "boundary_condition": "u(-1,t)=u(1,t)=0",
        "reference_solution": "explicit finite-difference solver",
        "epochs": args.epochs,
        "completed_epochs": completed_epochs,
        "max_runtime_sec": args.max_runtime_sec,
        "adam_max_runtime_sec": args.adam_max_runtime_sec,
        "n_interior": args.n_interior,
        "n_initial": args.n_initial,
        "n_boundary": args.n_boundary,
        "hidden_dim": args.hidden_dim,
        "hidden_layers": args.hidden_layers,
        "lr": args.lr,
        "lambda_ic": args.lambda_ic,
        "lambda_bc": args.lambda_bc,
        "bound_amplitude": args.bound_amplitude,
        "sampling": args.sampling,
        "focus_fraction": args.focus_fraction,
        "focus_std": args.focus_std,
        "nu": args.nu,
        "lbfgs_steps": args.lbfgs_steps,
        "completed_lbfgs_steps": lbfgs_completed_steps,
        "seed": args.seed,
        "device": str(device),
        "dtype": args.dtype,
        "eval_grid_size": args.eval_grid_size,
        "runtime_sec": elapsed,
        "l2_relative_error": l2_error,
        "final_total_loss": float(final_total.item()),
        "final_pde_loss": float(final_loss_pde.item()),
        "final_ic_loss": float(final_loss_ic.item()),
        "final_bc_loss": float(final_loss_bc.item()),
        "wandb_mode": args.wandb_mode,
    }
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    if run is not None:
        run.summary["runtime_sec"] = elapsed
        run.summary["l2_relative_error"] = l2_error
        run.summary["output_dir"] = str(output_dir.resolve())
        log_wandb_metrics(run, args.epochs + args.lbfgs_steps, final_total, final_loss_pde, final_loss_ic, final_loss_bc, l2_error=l2_error)
        run.log_artifact(str(model_path), name=f"{args.wandb_group}-model-seed{args.seed}", type="model")
        run.log_artifact(str(metrics_path), name=f"{args.wandb_group}-metrics-seed{args.seed}", type="metrics")
        run.log_artifact(str(output_dir / "burgers_fields.png"), name=f"{args.wandb_group}-fields-seed{args.seed}", type="plot")
        run.log_artifact(str(output_dir / "training_losses.png"), name=f"{args.wandb_group}-losses-seed{args.seed}", type="plot")
        run.log_artifact(str(output_dir / "training_losses_smoothed.png"), name=f"{args.wandb_group}-smoothed-losses-seed{args.seed}", type="plot")
        run.log_artifact(str(output_dir / "history.csv"), name=f"{args.wandb_group}-history-seed{args.seed}", type="history")
        run.finish()

    print(f"finished in {elapsed:.2f} sec")
    print(f"L2 relative error: {l2_error:.6e}")
    print(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
