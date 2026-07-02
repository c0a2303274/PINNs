import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_seeds(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def run_one(args: argparse.Namespace, method: str, seed: int, output_dir: Path) -> dict[str, object]:
    cmd = [
        sys.executable,
        "train_burgers_pinn.py",
        "--constraint-mode",
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
        "--nu",
        str(args.nu),
        "--lbfgs-steps",
        str(args.lbfgs_steps),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-group",
        f"burgers-{method}-2h",
        "--wandb-tags",
        f"burgers,{method},2h,nonlinear-pde",
    ]
    subprocess.run(cmd, check=True)

    with (output_dir / "metrics.json").open(encoding="utf-8") as fh:
        metrics = json.load(fh)
    metrics["method"] = method
    return metrics


def write_summary(rows: list[dict[str, object]], output_root: Path) -> None:
    keys = [
        "method",
        "seed",
        "constraint_mode",
        "l2_relative_error",
        "final_total_loss",
        "final_pde_loss",
        "final_ic_loss",
        "final_bc_loss",
        "runtime_sec",
        "completed_epochs",
        "completed_lbfgs_steps",
    ]

    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})

    with (output_root / "summary.md").open("w", encoding="utf-8") as fh:
        fh.write("# Burgers constraint comparison\n\n")
        fh.write("| method | seed | L2 relative error | PDE loss | IC loss | BC loss | runtime sec | epochs | lbfgs steps |\n")
        fh.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            fh.write(
                f"| {row.get('method')} | {row.get('seed')} | "
                f"{float(row.get('l2_relative_error', 0.0)):.6e} | "
                f"{float(row.get('final_pde_loss', 0.0)):.6e} | "
                f"{float(row.get('final_ic_loss', 0.0)):.6e} | "
                f"{float(row.get('final_bc_loss', 0.0)):.6e} | "
                f"{float(row.get('runtime_sec', 0.0)):.2f} | "
                f"{row.get('completed_epochs')} | {row.get('completed_lbfgs_steps')} |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run soft vs hard-IC/BC Burgers PINNs comparisons.")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--runtime-sec", type=float, default=7200.0, help="runtime budget per method and seed")
    parser.add_argument("--epochs", type=int, default=1_000_000)
    parser.add_argument("--n-interior", type=int, default=4096)
    parser.add_argument("--n-initial", type=int, default=512)
    parser.add_argument("--n-boundary", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--lambda-ic", type=float, default=1.0)
    parser.add_argument("--lambda-bc", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=0.01 / 3.141592653589793)
    parser.add_argument("--lbfgs-steps", type=int, default=0)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/burgers_constraint_comparison_2h"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in parse_seeds(args.seeds):
        for method in ["soft", "hard-icbc"]:
            output_dir = args.output_root / f"{method}_seed{seed}"
            rows.append(run_one(args, method, seed, output_dir))
            write_summary(rows, args.output_root)
    write_summary(rows, args.output_root)
    print(f"saved summary to {args.output_root}")


if __name__ == "__main__":
    main()
