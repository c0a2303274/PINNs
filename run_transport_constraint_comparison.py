import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


METHODS = {
    "linear": ["soft", "hardnet"],
    "circle": ["soft", "hardnetpp"],
}


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def run_one(
    args: argparse.Namespace,
    constraint: str,
    method: str,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    cmd = [
        sys.executable,
        "train_transport_constraint_pinn.py",
        "--constraint",
        constraint,
        "--method",
        method,
        "--epochs",
        str(args.epochs),
        "--max-runtime-sec",
        str(args.runtime_sec),
        "--n-interior",
        str(args.n_interior),
        "--n-initial",
        str(args.n_initial),
        "--n-boundary",
        str(args.n_boundary),
        "--hidden-dim",
        str(args.hidden_dim),
        "--hidden-layers",
        str(args.hidden_layers),
        "--lr",
        str(args.lr),
        "--lambda-ic",
        str(args.lambda_ic),
        "--lambda-bc",
        str(args.lambda_bc),
        "--lambda-constraint",
        str(args.lambda_constraint),
        "--projection-iterations",
        str(args.projection_iterations),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--eval-grid-size",
        str(args.eval_grid_size),
        "--eval-residual-points",
        str(args.eval_residual_points),
        "--inference-repeats",
        str(args.inference_repeats),
        "--print-every",
        str(args.print_every),
        "--wandb-mode",
        args.wandb_mode,
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)
    with (output_dir / "metrics.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def write_summary(rows: list[dict[str, object]], output_root: Path) -> None:
    keys = [
        "constraint",
        "method",
        "seed",
        "hidden_dim",
        "hidden_layers",
        "parameter_count",
        "l2_relative_error",
        "pde_residual_rmse",
        "max_constraint_violation",
        "final_pde_loss",
        "final_ic_loss",
        "final_bc_loss",
        "runtime_sec",
        "completed_epochs",
        "peak_gpu_mem_mb",
        "inference_time_ms",
    ]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})

    with (output_root / "summary.md").open("w", encoding="utf-8") as fh:
        fh.write("# HardNet and HardNet++ PDE bridge comparison\n\n")
        fh.write(
            "| constraint | method | seed | L2 relative error | PDE RMSE | max violation | "
            "runtime sec | epochs | inference ms |\n"
        )
        fh.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            fh.write(
                f"| {row.get('constraint')} | {row.get('method')} | {row.get('seed')} | "
                f"{float(row.get('l2_relative_error', 0.0)):.6e} | "
                f"{float(row.get('pde_residual_rmse', 0.0)):.6e} | "
                f"{float(row.get('max_constraint_violation', 0.0)):.6e} | "
                f"{float(row.get('runtime_sec', 0.0)):.2f} | {row.get('completed_epochs')} | "
                f"{float(row.get('inference_time_ms', 0.0)):.3f} |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare soft constraints with HardNet and HardNet++ on an analytic transport PDE."
    )
    parser.add_argument("--constraints", type=str, default="linear,circle")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--runtime-sec", type=float, default=600.0, help="runtime budget per method and seed")
    parser.add_argument("--epochs", type=int, default=1_000_000)
    parser.add_argument("--n-interior", type=int, default=1024)
    parser.add_argument("--n-initial", type=int, default=256)
    parser.add_argument("--n-boundary", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--lambda-ic", type=float, default=1.0)
    parser.add_argument("--lambda-bc", type=float, default=1.0)
    parser.add_argument("--lambda-constraint", type=float, default=1.0)
    parser.add_argument("--projection-iterations", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--eval-grid-size", type=int, default=101)
    parser.add_argument("--eval-residual-points", type=int, default=4096)
    parser.add_argument("--inference-repeats", type=int, default=20)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/transport_constraint_comparison"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    constraints = parse_csv(args.constraints)
    unknown = sorted(set(constraints) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown constraints: {unknown}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seed_text in parse_csv(args.seeds):
        seed = int(seed_text)
        for constraint in constraints:
            for method in METHODS[constraint]:
                output_dir = args.output_root / f"{constraint}_{method}_seed{seed}"
                rows.append(run_one(args, constraint, method, seed, output_dir))
                write_summary(rows, args.output_root)
    print(f"saved summary to {args.output_root}")


if __name__ == "__main__":
    main()
