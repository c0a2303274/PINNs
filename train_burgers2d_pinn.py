"""Soft PINNs for analytic 2D Burgers: global time or sequential time slabs."""

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from burgers2d_benchmark import (
    Problem, SlabMLP, exact_solution, handoff_target, periodic_losses,
    periodic_pairs, predict_piecewise, residual, sample_points, slab_index,
)


def widths_arg(value):
    try:
        widths = [int(x.strip()) for x in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated positive integers") from exc
    if not widths or min(widths) < 1:
        raise argparse.ArgumentTypeError("hidden widths must be positive")
    return widths


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["global", "marching"], default="global")
    p.add_argument("--hidden-widths", type=widths_arg, default=widths_arg("128,128,128,128,128"))
    p.add_argument("--windows", type=int, default=5, help="used only in marching mode")
    p.add_argument("--runtime-sec", type=float, default=600, help="total training-loop budget, shared across windows")
    p.add_argument("--epochs", type=int, default=1_000_000, help="total update cap, shared across windows")
    p.add_argument("--n-interior", type=int, default=1024)
    p.add_argument("--n-initial", type=int, default=256)
    p.add_argument("--n-boundary", type=int, default=128, help="paired points per direction; 4*N boundary coordinates")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-ic", type=float, default=1.0)
    p.add_argument("--lambda-bc", type=float, default=1.0)
    p.add_argument("--lambda-bc-grad", type=float, default=1.0)
    p.add_argument("--nu", type=float, default=0.01)
    p.add_argument("--amplitude", type=float, default=0.8)
    p.add_argument("--speed-x", type=float, default=0.5)
    p.add_argument("--speed-y", type=float, default=0.25)
    p.add_argument("--t-final", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--eval-grid-size", type=int, default=48)
    p.add_argument("--eval-times", type=int, default=21)
    p.add_argument("--eval-residual-points", type=int, default=1024)
    p.add_argument("--inference-repeats", type=int, default=20)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    p.add_argument("--wandb-project", default="pinns-burgers2d")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/burgers2d_global"))
    args = p.parse_args(argv)
    positive = ["windows", "epochs", "n_interior", "n_initial", "n_boundary", "cpu_threads",
                "eval_grid_size", "eval_times", "eval_residual_points", "inference_repeats", "log_every"]
    if any(getattr(args, key) < 1 for key in positive):
        p.error("counts must be positive")
    if args.eval_grid_size < 4 or args.eval_times < 2 or args.seed < 0:
        p.error("evaluation needs grid >= 4, times >= 2 and seed >= 0")
    for key in ["runtime_sec", "lr", "lambda_ic", "lambda_bc", "lambda_bc_grad"]:
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(f"{key} must be finite and positive")
    if args.mode == "marching" and (args.windows < 2 or args.epochs < args.windows):
        p.error("marching needs windows >= 2 and total epochs >= windows")
    try:
        Problem(args.nu, args.amplitude, args.speed_x, args.speed_y, args.t_final)
    except ValueError as exc:
        p.error(str(exc))
    return args


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def provenance():
    root = Path(__file__).resolve().parent
    names = [Path(__file__).name, "pinn_model.py", "burgers2d_problem.py", "burgers2d_benchmark.py"]
    result = {"source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}}
    for label, command in [("git_commit", ["git", "rev-parse", "HEAD"]),
                           ("git_status", ["git", "status", "--short"])]:
        try:
            result[label] = subprocess.check_output(command, cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            result[label] = None
    return result


def train_slabs(args, problem, device, dtype, output, tracker=None):
    count = args.windows if args.mode == "marching" else 1
    edges = np.linspace(0, problem.t_final, count + 1).tolist()
    models, summaries, history = [], [], []
    samples = torch.Generator(device=device).manual_seed(args.seed + 10_000)
    total_steps = 0
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
            model = SlabMLP(args.hidden_widths, start, end).to(device=device, dtype=dtype)
            previous = models[-1] if models else None
            if previous is not None:
                model.network.load_state_dict(previous.network.state_dict())
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            initial = sample_points(args.n_initial, start, start, device, dtype, samples)
            target = handoff_target(previous, initial, problem)
            budget = args.runtime_sec / count
            cap = args.epochs // count + int(index < args.epochs % count)
            sync(device)
            begun = time.perf_counter()
            completed = 0
            stop_reason = "epoch_cap"
            for step in range(1, cap + 1):
                if time.perf_counter() - begun >= budget:
                    stop_reason = "time_budget"
                    break
                optimizer.zero_grad(set_to_none=True)
                interior = sample_points(args.n_interior, start, end, device, dtype, samples)
                lpde = residual(model, interior, problem).square().mean()
                lic = (model(initial) - target).square().mean()
                pairs = periodic_pairs(args.n_boundary, start, end, device, dtype, samples)
                lbc, lgrad = periodic_losses(model, pairs)
                loss = lpde + args.lambda_ic * lic + args.lambda_bc * lbc + args.lambda_bc_grad * lgrad
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"nonfinite loss in slab {index}, step {step}")
                loss.backward()
                gradients = [p.grad for p in model.parameters()]
                if any(g is None for g in gradients) or not torch.stack([torch.isfinite(g).all() for g in gradients]).all().item():
                    raise FloatingPointError(f"nonfinite/missing gradient in slab {index}, step {step}")
                optimizer.step()
                sync(device)
                completed = step
                total_steps += 1
                elapsed = time.perf_counter() - begun
                if step == 1 or step % args.log_every == 0 or step == cap or elapsed >= budget:
                    row = {"slab": index, "epoch": step, "total_epoch": total_steps,
                           "training_elapsed_sec": sum(s["training_sec"] for s in summaries) + elapsed,
                           "total_loss": loss.item(), "pde_loss": lpde.item(), "ic_loss": lic.item(),
                           "bc_loss": lbc.item(), "bc_gradient_loss": lgrad.item()}
                    history.append(row)
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    handle.flush()
                    if tracker is not None:
                        tracker.log(row, step=total_steps)
                if elapsed >= budget:
                    stop_reason = "time_budget"
                    break
            sync(device)
            training_sec = time.perf_counter() - begun
            if not completed:
                raise RuntimeError("budget too short to complete one update; increase --runtime-sec")
            model.eval().requires_grad_(False)
            models.append(model)
            info = {"slab": index, "start": start, "end": end, "completed_epochs": completed,
                    "training_sec": training_sec, "budget_sec": budget, "epoch_cap": cap,
                    "stop_reason": stop_reason, "initial_target": "analytic" if index == 0 else "previous_prediction"}
            summaries.append(info)
            torch.save({"network_state": {k: v.detach().cpu() for k, v in model.network.state_dict().items()},
                        "hidden_widths": args.hidden_widths, "start": start, "end": end,
                        "dtype": args.dtype, "problem": asdict(problem)}, output / f"slab_{index:02d}.pt")
            save_json(output / "slabs.json", summaries)
            print(f"slab {index+1}/{count}: {completed} updates, {training_sec:.2f}s, {stop_reason}", flush=True)
    return models, edges, summaries, history


def relative_error(diff, reference):
    return float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(reference).clamp_min(torch.finfo(diff.dtype).tiny))


def evaluate(models, edges, problem, args, device, dtype, output):
    n = args.eval_grid_size
    axis = torch.arange(n, device=device, dtype=dtype) / n
    xx, yy = torch.meshgrid(axis, axis, indexing="xy")
    space = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    times = np.linspace(0, problem.t_final, args.eval_times)
    background = space.new_tensor([problem.speed_x, problem.speed_y])
    rows, predictions, references = [], [], []
    with torch.no_grad():
        for t in times:
            coords = torch.cat((space.new_full((len(space), 1), float(t)), space), dim=1)
            pred = models[slab_index(float(t), edges)](coords)
            ref = exact_solution(coords, problem)
            diff = pred - ref
            rows.append({"t": float(t), "l2_relative_error": relative_error(diff, ref),
                         "l2_u": relative_error(diff[:, 0], ref[:, 0]),
                         "l2_v": relative_error(diff[:, 1], ref[:, 1]),
                         "fluctuation_l2_relative_error": relative_error(diff, ref - background),
                         "max_abs_error": diff.abs().max().item()})
            predictions.append(pred.cpu())
            references.append(ref.cpu())
    prediction, reference = torch.stack(predictions), torch.stack(references)
    error = prediction - reference
    metrics = {"l2_relative_error": relative_error(error, reference),
               "l2_u": relative_error(error[..., 0], reference[..., 0]),
               "l2_v": relative_error(error[..., 1], reference[..., 1]),
               "fluctuation_l2_relative_error": relative_error(error, reference - background.cpu()),
               "final_time_l2_relative_error": rows[-1]["l2_relative_error"],
               "max_abs_error": error.abs().max().item()}
    generator = torch.Generator(device=device).manual_seed(args.seed + 90_000)
    coords = sample_points(args.eval_residual_points, 0, problem.t_final, device, dtype, generator)
    indices = torch.bucketize(coords[:, 0].contiguous(), coords.new_tensor(edges[1:-1]), right=True)
    sum_residual = 0.0
    for i, model in enumerate(models):
        subset = coords[indices == i]
        if len(subset):
            sum_residual += residual(model, subset, problem).detach().square().sum().item()
    metrics["pde_rmse"] = math.sqrt(sum_residual / (2 * len(coords)))
    pairs = periodic_pairs(args.n_boundary, 0, problem.t_final, device, dtype, generator)
    piecewise = lambda pts: predict_piecewise(models, edges, pts)
    bc, bcgrad = periodic_losses(piecewise, pairs)
    metrics["bc_rmse"] = bc.detach().sqrt().item()
    metrics["bc_gradient_rmse"] = bcgrad.detach().sqrt().item()
    jumps = []
    with torch.no_grad():
        initial = sample_points(args.n_initial, 0, 0, device, dtype, generator)
        metrics["ic_rmse"] = (models[0](initial) - exact_solution(initial, problem)).square().mean().sqrt().item()
        for i, t in enumerate(edges[1:-1], start=1):
            pts = torch.cat((space.new_full((len(space), 1), t), space), dim=1)
            difference = models[i](pts) - models[i - 1](pts)
            jumps.append({"t": t, "rmse": difference.square().mean().sqrt().item(),
                          "max_abs_jump": difference.abs().max().item()})
        metrics["max_interface_rmse"] = max((j["rmse"] for j in jumps), default=0.0)
        # Same physical coordinates for all methods; include piecewise routing overhead.
        inference_coords = sample_points(n * n, 0, problem.t_final, device, dtype, generator)
        for _ in range(3):
            piecewise(inference_coords)
        sync(device)
        before = time.perf_counter()
        for _ in range(args.inference_repeats):
            piecewise(inference_coords)
        sync(device)
        metrics["inference_time_ms"] = (time.perf_counter() - before) * 1000 / args.inference_repeats
        metrics["inference_points"] = len(inference_coords)
    write_csv(output / "time_errors.csv", rows)
    save_json(output / "interfaces.json", jumps)
    np.savez_compressed(output / "fields.npz", times=times, axis=axis.cpu().numpy(),
                        prediction=prediction.numpy().reshape(-1, n, n, 2),
                        reference=reference.numpy().reshape(-1, n, n, 2))
    plot_fields(prediction.numpy().reshape(-1, n, n, 2), reference.numpy().reshape(-1, n, n, 2), times, output)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.semilogy(times, [max(r["l2_relative_error"], 1e-16) for r in rows], label="Full field")
    ax.semilogy(times, [max(r["fluctuation_l2_relative_error"], 1e-16) for r in rows], label="Error / fluctuation norm")
    for t in edges[1:-1]:
        ax.axvline(t, color="gray", alpha=.4, linewidth=1)
    ax.set(xlabel="Physical time t", ylabel="Relative L2 error")
    ax.legend()
    fig.savefig(output / "time_errors.png", dpi=160)
    plt.close(fig)
    return metrics


def plot_fields(pred, ref, times, output):
    for component, label in enumerate(("u", "v")):
        selected = sorted(set([0, len(times) // 2, len(times) - 1]))
        vmin = min(ref[..., component].min(), pred[..., component].min())
        vmax = max(ref[..., component].max(), pred[..., component].max())
        errmax = max(float(np.abs(pred[..., component] - ref[..., component]).max()), 1e-12)
        fig, axes = plt.subplots(len(selected), 3, figsize=(10, 3 * len(selected)), squeeze=False, constrained_layout=True)
        for row, index in enumerate(selected):
            fields = [ref[index, ..., component], pred[index, ..., component], np.abs(pred[index, ..., component] - ref[index, ..., component])]
            for col, (data, title) in enumerate(zip(fields, ["Reference", "Prediction", "Absolute error"])):
                ax = axes[row, col]
                image = ax.imshow(data, origin="lower", extent=(0, 1, 0, 1),
                                  vmin=0 if col == 2 else vmin, vmax=errmax if col == 2 else vmax,
                                  cmap="magma" if col == 2 else "viridis")
                ax.set(title=f"{title} {label}, t={times[index]:.2f}", xlabel="x", ylabel="y")
                fig.colorbar(image, ax=ax, shrink=.8)
        fig.savefig(output / f"fields_{label}.png", dpi=160)
        plt.close(fig)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    dtype = getattr(torch, payload["dtype"])
    model = SlabMLP(payload["hidden_widths"], payload["start"], payload["end"]).to(device=device, dtype=dtype)
    model.network.load_state_dict(payload["network_state"])
    return model.eval().requires_grad_(False)


def run(args):
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu only for smoke tests")
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing run: {output}")
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    dtype = getattr(torch, args.dtype)
    problem = Problem(args.nu, args.amplitude, args.speed_x, args.speed_y, args.t_final)
    config = {**vars(args), "output_dir": str(output), "actual_device": str(device), "problem": asdict(problem),
              "torch_version": str(torch.__version__), "python_version": platform.python_version(),
              "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
              "constraint_mode": "soft-periodic-value-and-gradient", **provenance()}
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", config)
    tracker = None
    started = time.perf_counter()
    try:
        if args.wandb_mode != "disabled":
            import wandb
            tracker = wandb.init(project=args.wandb_project, mode=args.wandb_mode, config=config,
                                 name=f"{args.mode}-{'-'.join(map(str,args.hidden_widths))}-seed{args.seed}", dir=str(output))
        models, edges, slabs, history = train_slabs(args, problem, device, dtype, output, tracker)
        metrics = evaluate(models, edges, problem, args, device, dtype, output)
        parameters = sum(p.numel() for p in models[0].parameters())
        metrics.update({"status": "complete", "mode": args.mode, "seed": args.seed, "hidden_widths": args.hidden_widths,
                        "parameter_count_per_model": parameters, "stored_parameter_count": parameters * len(models),
                        "model_count": len(models), "training_sec": sum(s["training_sec"] for s in slabs),
                        "completed_epochs": sum(s["completed_epochs"] for s in slabs),
                        "runtime_budget_sec": args.runtime_sec, "edges": edges,
                        "peak_gpu_mem_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
                        "wall_sec": time.perf_counter() - started})
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        for key in ("total_loss", "pde_loss", "ic_loss", "bc_loss", "bc_gradient_loss"):
            ax.semilogy([r["training_elapsed_sec"] for r in history], [max(r[key], 1e-16) for r in history], label=key)
        ax.set(xlabel="Cumulative training-loop time (s)", ylabel="Sampled training loss (pre-update)")
        ax.legend(fontsize=8)
        fig.savefig(output / "training_losses.png", dpi=160)
        plt.close(fig)
        save_json(output / "metrics.json", metrics)
        save_json(output / "status.json", {"status": "complete"})
        if tracker is not None:
            tracker.summary.update(metrics)
        print(json.dumps(metrics, indent=2), flush=True)
        return metrics
    except BaseException as exc:
        save_json(output / "status.json", {"status": "failed", "error": str(exc)})
        raise
    finally:
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    run(parse_args())
