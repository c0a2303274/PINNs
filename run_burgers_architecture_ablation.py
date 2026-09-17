import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_seeds(raw: str) -> list[int]:
    return parse_ints(raw)


def make_config(
    study: str,
    name: str,
    hidden_dim: int,
    hidden_layers: int,
    n_interior: int,
    n_initial: int,
    n_boundary: int,
) -> dict[str, object]:
    return {
        "study": study,
        "config_name": name,
        "hidden_dim": hidden_dim,
        "hidden_layers": hidden_layers,
        "n_interior": n_interior,
        "n_initial": n_initial,
        "n_boundary": n_boundary,
    }


def build_configs(args: argparse.Namespace) -> list[dict[str, object]]:
    configs: list[dict[str, object]] = []
    studies = [args.study] if args.study != "all" else ["width", "depth", "points"]

    if "width" in studies:
        for width in parse_ints(args.widths):
            configs.append(
                make_config(
                    "width",
                    f"width_{width}",
                    width,
                    args.base_layers,
                    args.base_n_interior,
                    args.base_n_initial,
                    args.base_n_boundary,
                )
            )

    if "depth" in studies:
        for depth in parse_ints(args.depths):
            configs.append(
                make_config(
                    "depth",
                    f"depth_{depth}",
                    args.base_width,
                    depth,
                    args.base_n_interior,
                    args.base_n_initial,
                    args.base_n_boundary,
                )
            )

    if "points" in studies:
        for n_interior in parse_ints(args.interior_counts):
            scale = n_interior / args.base_n_interior
            n_initial = max(1, round(args.base_n_initial * scale))
            n_boundary = max(2, round(args.base_n_boundary * scale))
            if n_boundary % 2:
                n_boundary += 1
            configs.append(
                make_config(
                    "points",
                    f"points_{n_interior}",
                    args.base_width,
                    args.base_layers,
                    n_interior,
                    n_initial,
                    n_boundary,
                )
            )

    deduplicated: dict[tuple[int, int, int, int, int], dict[str, object]] = {}
    for config in configs:
        key = (
            int(config["hidden_dim"]),
            int(config["hidden_layers"]),
            int(config["n_interior"]),
            int(config["n_initial"]),
            int(config["n_boundary"]),
        )
        if key in deduplicated:
            previous = deduplicated[key]
            previous["study"] = f"{previous['study']}+{config['study']}"
            previous["config_name"] = "baseline"
        else:
            deduplicated[key] = config
    return list(deduplicated.values())


def run_one(
    args: argparse.Namespace,
    config: dict[str, object],
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    cmd = [
        sys.executable,
        "train_burgers_pinn.py",
        "--constraint-mode",
        args.constraint_mode,
        "--epochs",
        str(args.epochs),
        "--max-runtime-sec",
        str(args.runtime_sec),
        "--n-interior",
        str(config["n_interior"]),
        "--n-initial",
        str(config["n_initial"]),
        "--n-boundary",
        str(config["n_boundary"]),
        "--hidden-dim",
        str(config["hidden_dim"]),
        "--hidden-layers",
        str(config["hidden_layers"]),
        "--lr",
        str(args.lr),
        "--lambda-ic",
        str(args.lambda_ic),
        "--lambda-bc",
        str(args.lambda_bc),
        "--nu",
        str(args.nu),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--eval-grid-size",
        str(args.eval_grid_size),
        "--inference-repeats",
        str(args.inference_repeats),
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-group",
        f"burgers-architecture-{config['study']}",
        "--wandb-tags",
        f"burgers,architecture,{config['study']},{args.constraint_mode}",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)
    with (output_dir / "metrics.json").open(encoding="utf-8") as fh:
        metrics = json.load(fh)
    metrics["study"] = config["study"]
    metrics["config_name"] = config["config_name"]
    return metrics


def write_summary(rows: list[dict[str, object]], output_root: Path) -> None:
    keys = [
        "study",
        "config_name",
        "constraint_mode",
        "seed",
        "hidden_dim",
        "hidden_layers",
        "parameter_count",
        "n_interior",
        "n_initial",
        "n_boundary",
        "l2_relative_error",
        "final_pde_loss",
        "final_ic_loss",
        "final_bc_loss",
        "runtime_sec",
        "completed_epochs",
        "peak_gpu_mem_mb",
        "inference_time_ms",
        "inference_points_per_sec",
    ]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})

    with (output_root / "summary.md").open("w", encoding="utf-8") as fh:
        fh.write("# Burgers architecture ablation\n\n")
        fh.write(
            "| study | config | seed | width | layers | params | interior | "
            "L2 relative error | PDE loss | runtime sec | epochs | peak GPU MB | inference ms |\n"
        )
        fh.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            peak_gpu = row.get("peak_gpu_mem_mb")
            peak_gpu_text = "" if peak_gpu is None else f"{float(peak_gpu):.1f}"
            fh.write(
                f"| {row.get('study')} | {row.get('config_name')} | {row.get('seed')} | "
                f"{row.get('hidden_dim')} | {row.get('hidden_layers')} | {row.get('parameter_count')} | "
                f"{row.get('n_interior')} | {float(row.get('l2_relative_error', 0.0)):.6e} | "
                f"{float(row.get('final_pde_loss', 0.0)):.6e} | "
                f"{float(row.get('runtime_sec', 0.0)):.2f} | {row.get('completed_epochs')} | "
                f"{peak_gpu_text} | {float(row.get('inference_time_ms', 0.0)):.3f} |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled width, depth, and collocation-count ablations for 1D Burgers PINNs."
    )
    parser.add_argument("--study", choices=["width", "depth", "points", "all"], default="all")
    parser.add_argument("--constraint-mode", choices=["soft", "hard-icbc", "bounded-hard-icbc"], default="soft")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--runtime-sec", type=float, default=300.0, help="runtime budget per configuration and seed")
    parser.add_argument("--epochs", type=int, default=1_000_000)
    parser.add_argument("--widths", type=str, default="32,64,128,256")
    parser.add_argument("--depths", type=str, default="3,5,8")
    parser.add_argument("--interior-counts", type=str, default="1024,4096,16384")
    parser.add_argument("--base-width", type=int, default=128)
    parser.add_argument("--base-layers", type=int, default=5)
    parser.add_argument("--base-n-interior", type=int, default=4096)
    parser.add_argument("--base-n-initial", type=int, default=512)
    parser.add_argument("--base-n-boundary", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--lambda-ic", type=float, default=1.0)
    parser.add_argument("--lambda-bc", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=0.01 / math.pi)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--eval-grid-size", type=int, default=101)
    parser.add_argument("--inference-repeats", type=int, default=50)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/burgers_architecture_ablation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    configs = build_configs(args)
    rows: list[dict[str, object]] = []
    for seed in parse_seeds(args.seeds):
        for config in configs:
            output_dir = args.output_root / f"{config['config_name']}_seed{seed}"
            rows.append(run_one(args, config, seed, output_dir))
            write_summary(rows, args.output_root)
    print(f"saved summary to {args.output_root}")


if __name__ == "__main__":
    main()
