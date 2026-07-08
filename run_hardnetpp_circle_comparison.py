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
        "train_hardnetpp_circle_demo.py",
        "--method",
        method,
        "--epochs",
        str(args.epochs),
        "--max-runtime-sec",
        str(args.runtime_sec),
        "--n-points",
        str(args.n_points),
        "--hidden-dim",
        str(args.hidden_dim),
        "--hidden-layers",
        str(args.hidden_layers),
        "--lr",
        str(args.lr),
        "--projection-iterations",
        str(args.projection_iterations),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)
    with (output_dir / "metrics.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def write_summary(rows: list[dict[str, object]], output_root: Path) -> None:
    keys = ["method", "seed", "l2_relative_error", "max_constraint_violation", "runtime_sec", "completed_epochs"]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})

    with (output_root / "summary.md").open("w", encoding="utf-8") as fh:
        fh.write("# HardNet++ circle comparison\n\n")
        fh.write("| method | seed | L2 relative error | max constraint violation | runtime sec | epochs |\n")
        fh.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            fh.write(
                f"| {row.get('method')} | {row.get('seed')} | "
                f"{float(row.get('l2_relative_error', 0.0)):.6e} | "
                f"{float(row.get('max_constraint_violation', 0.0)):.6e} | "
                f"{float(row.get('runtime_sec', 0.0)):.2f} | "
                f"{row.get('completed_epochs')} |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare soft MLP and HardNet++ on a nonlinear unit-circle constraint.")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--runtime-sec", type=float, default=1800.0)
    parser.add_argument("--epochs", type=int, default=1_000_000)
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--projection-iterations", type=int, default=15)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/hardnetpp_circle_comparison"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in parse_seeds(args.seeds):
        for method in ["soft", "hardnetpp"]:
            output_dir = args.output_root / f"{method}_seed{seed}"
            rows.append(run_one(args, method, seed, output_dir))
            write_summary(rows, args.output_root)
    write_summary(rows, args.output_root)
    print(f"saved summary to {args.output_root}")


if __name__ == "__main__":
    main()
