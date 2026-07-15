import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from burgers_shock_problem import (
    boundary_left,
    boundary_right,
    initial_condition,
    pde_residual,
    shock_solution,
)
from pinn_model import MLP


class TimeBudgetExceeded(Exception):
    pass


class ShockHardICBCModel(nn.Module):
    """Output transform satisfying shock IC and time-dependent boundary values."""

    def __init__(
        self,
        hidden_dim: int,
        hidden_layers: int,
        nu: float,
        u_left: float,
        u_right: float,
        x0: float,
        x_min: float,
        x_max: float,
    ):
        super().__init__()
        self.network = MLP(in_dim=2, hidden_dim=hidden_dim, hidden_layers=hidden_layers, out_dim=1)
        self.nu = nu
        self.u_left = u_left
        self.u_right = u_right
        self.x0 = x0
        self.x_min = x_min
        self.x_max = x_max

    def boundary_blend(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        left_w = (self.x_max - x) / (self.x_max - self.x_min)
        right_w = (x - self.x_min) / (self.x_max - self.x_min)
        g_left = boundary_left(t, self.nu, self.u_left, self.u_right, self.x0, self.x_min)
        g_right = boundary_right(t, self.nu, self.u_left, self.u_right, self.x0, self.x_max)
        return left_w * g_left + right_w * g_right

    def boundary_blend_initial(self, x: torch.Tensor) -> torch.Tensor:
        t0 = torch.zeros_like(x)
        return self.boundary_blend(t0, x)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        t = coords[:, :1]
        x = coords[:, 1:]
        u0 = initial_condition(x, self.nu, self.u_left, self.u_right, self.x0)
        boundary = self.boundary_blend(t, x)
        boundary0 = self.boundary_blend_initial(x)
        interior_shape = (x - self.x_min) * (self.x_max - x)
        return boundary + (1.0 - t) * (u0 - boundary0) + t * interior_shape * self.network(coords)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_interior(n_points: int, device: torch.device, t_max: float, x_min: float, x_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.empty(n_points, 1, device=device).uniform_(0.0, t_max)
    x = torch.empty(n_points, 1, device=device).uniform_(x_min, x_max)
    return t, x


def sample_initial(n_points: int, device: torch.device, x_min: float, x_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.zeros(n_points, 1, device=device)
    x = torch.empty(n_points, 1, device=device).uniform_(x_min, x_max)
    return t, x


def sample_boundary(n_points: int, device: torch.device, t_max: float, x_min: float, x_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    n_side = max(n_points // 2, 1)
    t_left = torch.empty(n_side, 1, device=device).uniform_(0.0, t_max)
    t_right = torch.empty(n_side, 1, device=device).uniform_(0.0, t_max)
    x = torch.cat([torch.full_like(t_left, x_min), torch.full_like(t_right, x_max)], dim=0)
    t = torch.cat([t_left, t_right], dim=0)
    return t, x


def compute_losses(model, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t_f, x_f = sample_interior(args.n_interior, device, args.t_max, args.x_min, args.x_max)
    residual = pde_residual(model, t_f, x_f, args.nu)
    loss_pde = torch.mean(residual**2)

    t_ic, x_ic = sample_initial(args.n_initial, device, args.x_min, args.x_max)
    u_ic = model(torch.cat([t_ic, x_ic], dim=1))
    target_ic = initial_condition(x_ic, args.nu, args.u_left, args.u_right, args.x0)
    loss_ic = torch.mean((u_ic - target_ic) ** 2)

    t_bc, x_bc = sample_boundary(args.n_boundary, device, args.t_max, args.x_min, args.x_max)
    u_bc = model(torch.cat([t_bc, x_bc], dim=1))
    left_mask = x_bc < 0.5 * (args.x_min + args.x_max)
    target_bc = torch.empty_like(u_bc)
    target_bc[left_mask] = boundary_left(t_bc[left_mask], args.nu, args.u_left, args.u_right, args.x0, args.x_min)
    target_bc[~left_mask] = boundary_right(t_bc[~left_mask], args.nu, args.u_left, args.u_right, args.x0, args.x_max)
    loss_bc = torch.mean((u_bc - target_bc) ** 2)
    total = loss_pde + args.lambda_ic * loss_ic + args.lambda_bc * loss_bc
    return total, loss_pde, loss_ic, loss_bc


def evaluate(model, args: argparse.Namespace, device: torch.device) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_grid = torch.linspace(0.0, args.t_max, args.eval_grid_size, device=device)
    x_grid = torch.linspace(args.x_min, args.x_max, args.eval_grid_size, device=device)
    tt, xx = torch.meshgrid(t_grid, x_grid, indexing="ij")
    coords = torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=1)
    with torch.no_grad():
        pred = model(coords).reshape(args.eval_grid_size, args.eval_grid_size)
        truth = shock_solution(tt, xx, args.nu, args.u_left, args.u_right, args.x0)
    l2_error = torch.linalg.norm(pred - truth) / torch.linalg.norm(truth)
    return float(l2_error.item()), tt.cpu().numpy(), xx.cpu().numpy(), pred.cpu().numpy(), truth.cpu().numpy()


def save_outputs(model, args: argparse.Namespace, output_dir: Path, device: torch.device, history: dict[str, list[float]]) -> float:
    l2_error, tt, xx, pred, truth = evaluate(model, args, device)
    error = pred - truth
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, title in zip(axes, [truth, pred, error], ["Exact shock solution", "PINN prediction", f"Error, L2={l2_error:.2e}"]):
        image = ax.imshow(data, origin="lower", extent=(args.x_min, args.x_max, 0, args.t_max), aspect="auto", cmap="viridis")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "burgers_shock_fields.png", dpi=150, bbox_inches="tight")
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

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "total_loss", "pde_loss", "ic_loss", "bc_loss"])
        for i, values in enumerate(zip(history["total"], history["pde"], history["ic"], history["bc"]), start=1):
            writer.writerow([i, *values])
    return l2_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PINNs on the viscous Burgers traveling shock solution.")
    parser.add_argument("--constraint-mode", choices=["soft", "hard-icbc"], default="hard-icbc")
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--max-runtime-sec", type=float, default=None)
    parser.add_argument("--adam-max-runtime-sec", type=float, default=None)
    parser.add_argument("--n-interior", type=int, default=4096)
    parser.add_argument("--n-initial", type=int, default=512)
    parser.add_argument("--n-boundary", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--lambda-ic", type=float, default=1.0)
    parser.add_argument("--lambda-bc", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=0.05)
    parser.add_argument("--u-left", type=float, default=1.0)
    parser.add_argument("--u-right", type=float, default=0.0)
    parser.add_argument("--x0", type=float, default=0.0)
    parser.add_argument("--x-min", type=float, default=-1.0)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument("--t-max", type=float, default=1.0)
    parser.add_argument("--lbfgs-steps", type=int, default=0)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-history-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-grid-size", type=int, default=101)
    parser.add_argument("--print-every", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/burgers_shock"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.constraint_mode == "hard-icbc":
        model = ShockHardICBCModel(args.hidden_dim, args.hidden_layers, args.nu, args.u_left, args.u_right, args.x0, args.x_min, args.x_max).to(device)
    else:
        model = MLP(in_dim=2, hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers, out_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {"total": [], "pde": [], "ic": [], "bc": []}
    start_time = time.time()
    completed_epochs = 0

    for epoch in range(1, args.epochs + 1):
        if args.max_runtime_sec is not None and time.time() - start_time >= args.max_runtime_sec:
            break
        if args.adam_max_runtime_sec is not None and time.time() - start_time >= args.adam_max_runtime_sec:
            break
        optimizer.zero_grad(set_to_none=True)
        total, loss_pde, loss_ic, loss_bc = compute_losses(model, args, device)
        total.backward()
        optimizer.step()
        history["total"].append(float(total.item()))
        history["pde"].append(float(loss_pde.item()))
        history["ic"].append(float(loss_ic.item()))
        history["bc"].append(float(loss_bc.item()))
        completed_epochs = epoch
        if epoch == 1 or epoch % args.print_every == 0:
            print(f"epoch={epoch:6d} total={total.item():.3e} pde={loss_pde.item():.3e} ic={loss_ic.item():.3e} bc={loss_bc.item():.3e}")

    completed_lbfgs_steps = 0
    if args.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=args.lbfgs_lr,
            max_iter=args.lbfgs_steps,
            history_size=args.lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )
        state = {"step": 0}

        def closure() -> torch.Tensor:
            if args.max_runtime_sec is not None and time.time() - start_time >= args.max_runtime_sec:
                raise TimeBudgetExceeded
            lbfgs.zero_grad(set_to_none=True)
            total, loss_pde, loss_ic, loss_bc = compute_losses(model, args, device)
            total.backward()
            history["total"].append(float(total.item()))
            history["pde"].append(float(loss_pde.item()))
            history["ic"].append(float(loss_ic.item()))
            history["bc"].append(float(loss_bc.item()))
            state["step"] += 1
            if state["step"] == 1 or state["step"] % args.print_every == 0:
                print(f"lbfgs={state['step']:6d} total={total.item():.3e} pde={loss_pde.item():.3e} ic={loss_ic.item():.3e} bc={loss_bc.item():.3e}")
            return total

        try:
            lbfgs.step(closure)
        except TimeBudgetExceeded:
            pass
        completed_lbfgs_steps = state["step"]

    runtime_sec = time.time() - start_time
    final_total, final_pde, final_ic, final_bc = compute_losses(model, args, device)
    l2_error = save_outputs(model, args, output_dir, device, history)
    metrics = {
        "problem": "1D viscous Burgers traveling shock",
        "constraint_mode": args.constraint_mode,
        "equation": "u_t + u u_x = nu u_xx",
        "reference_solution": "analytic traveling shock from Hopf-Cole transform",
        "nu": args.nu,
        "u_left": args.u_left,
        "u_right": args.u_right,
        "x0": args.x0,
        "x_min": args.x_min,
        "x_max": args.x_max,
        "t_max": args.t_max,
        "epochs": args.epochs,
        "completed_epochs": completed_epochs,
        "lbfgs_steps": args.lbfgs_steps,
        "completed_lbfgs_steps": completed_lbfgs_steps,
        "runtime_sec": runtime_sec,
        "l2_relative_error": l2_error,
        "final_total_loss": float(final_total.item()),
        "final_pde_loss": float(final_pde.item()),
        "final_ic_loss": float(final_ic.item()),
        "final_bc_loss": float(final_bc.item()),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"finished in {runtime_sec:.2f} sec")
    print(f"L2 relative error: {l2_error:.6e}")
    print(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
