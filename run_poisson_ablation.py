#!/usr/bin/env python3
"""Run focused Poisson Adam->L-BFGS ablations for high-accuracy tuning."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


CONFIGS = {
    "base": {
        "description": "4h setting without Adam-only runs",
        "n_interior": 4096,
        "n_boundary": 1024,
        "hidden_dim": 128,
        "hidden_layers": 5,
        "lr": 1e-3,
        "lambda_bc": 1.0,
    },
    "more_points": {
        "description": "increase collocation and boundary points",
        "n_interior": 8192,
        "n_boundary": 2048,
        "hidden_dim": 128,
        "hidden_layers": 5,
        "lr": 1e-3,
        "lambda_bc": 1.0,
    },
    "wider": {
        "description": "increase network width",
        "n_interior": 4096,
        "n_boundary": 1024,
        "hidden_dim": 256,
        "hidden_layers": 5,
        "lr": 1e-3,
        "lambda_bc": 1.0,
    },
    "lower_lr": {
        "description": "use a smaller Adam learning rate before L-BFGS",
        "n_interior": 4096,
        "n_boundary": 1024,
        "hidden_dim": 128,
        "hidden_layers": 5,
        "lr": 5e-4,
        "lambda_bc": 1.0,
    },
}


def parse_seeds(raw: str) -> list[int]:
    return [int(seed.strip()) for seed in raw.split(",") if seed.strip()]


def format_float(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6e}"


def run_one(
    args: argparse.Namespace,
    config_name: str,
    config: dict[str, object],
    seed: int,
    script_path: Path,
) -> dict[str, object]:
    output_dir = args.output_root / config_name / f"seed_{seed}"
    metrics_path = output_dir / "metrics.json"

    if args.skip_existing and metrics_path.exists():
        return load_result(metrics_path, config_name, config, seed)

    command = [
        sys.executable,
        str(script_path),
        "--epochs",
        str(args.epochs),
        "--n-interior",
        str(config["n_interior"]),
        "--n-boundary",
        str(config["n_boundary"]),
        "--hidden-dim",
        str(config["hidden_dim"]),
        "--hidden-layers",
        str(config["hidden_layers"]),
        "--lr",
        str(config["lr"]),
        "--lambda-bc",
        str(config["lambda_bc"]),
        "--lbfgs-steps",
        str(args.lbfgs_steps),
        "--seed",
        str(seed),
        "--dtype",
        args.dtype,
        "--eval-grid-size",
        str(args.eval_grid_size),
        "--output-dir",
        str(output_dir),
        "--print-every",
        str(args.print_every),
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-group",
        f"poisson-ablation-{config_name}",
        "--wandb-tags",
        f"poisson,ablation,{config_name},adam,lbfgs",
        "--wandb-notes",
        f"Poisson high-accuracy ablation: {config_name}, seed={seed}",
    ]
    if args.max_runtime_sec is not None:
        command.extend(["--max-runtime-sec", str(args.max_runtime_sec)])
        command.extend(["--adam-max-runtime-sec", str(args.max_runtime_sec * args.adam_runtime_fraction)])
    if args.device != "auto":
        command.extend(["--device", args.device])

    subprocess.run(command, check=True, cwd=script_path.parent)
    return load_result(metrics_path, config_name, config, seed)


def load_result(metrics_path: Path, config_name: str, config: dict[str, object], seed: int) -> dict[str, object]:
    with metrics_path.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)

    return {
        "config": config_name,
        "description": config["description"],
        "seed": seed,
        "l2_relative_error": metrics.get("l2_relative_error"),
        "final_pde_loss": metrics.get("final_pde_loss"),
        "final_bc_loss": metrics.get("final_bc_loss"),
        "runtime_sec": metrics.get("runtime_sec"),
        "completed_epochs": metrics.get("completed_epochs"),
        "completed_lbfgs_steps": metrics.get("completed_lbfgs_steps"),
        "n_interior": config["n_interior"],
        "n_boundary": config["n_boundary"],
        "hidden_dim": config["hidden_dim"],
        "hidden_layers": config["hidden_layers"],
        "lr": config["lr"],
        "lambda_bc": config["lambda_bc"],
        "output_dir": str(metrics_path.parent),
    }


def write_csv(results: list[dict[str, object]], path: Path) -> None:
    fields = [
        "config",
        "description",
        "seed",
        "l2_relative_error",
        "final_pde_loss",
        "final_bc_loss",
        "runtime_sec",
        "completed_epochs",
        "completed_lbfgs_steps",
        "n_interior",
        "n_boundary",
        "hidden_dim",
        "hidden_layers",
        "lr",
        "lambda_bc",
        "output_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def write_markdown(results: list[dict[str, object]], path: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Poisson High-Accuracy Ablation",
        "",
        "Fixed setting:",
        "",
        f"- optimizer: Adam->L-BFGS",
        f"- epochs: {args.epochs}",
        f"- max_runtime_sec per run: {args.max_runtime_sec if args.max_runtime_sec is not None else 'none'}",
        f"- adam_runtime_fraction: {args.adam_runtime_fraction}",
        f"- dtype: {args.dtype}",
        f"- eval_grid_size: {args.eval_grid_size}",
        f"- seeds: {', '.join(str(seed) for seed in parse_seeds(args.seeds))}",
        "",
        "Per-run results:",
        "",
        "| config | seed | L2 relative error | PDE loss | BC loss | runtime sec | epochs | lbfgs steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| {config} | {seed} | {l2} | {pde} | {bc} | {runtime:.2f} | {epochs} | {lbfgs_steps} |".format(
                config=result["config"],
                seed=result["seed"],
                l2=format_float(result["l2_relative_error"]),
                pde=format_float(result["final_pde_loss"]),
                bc=format_float(result["final_bc_loss"]),
                runtime=float(result["runtime_sec"]),
                epochs=result.get("completed_epochs") or "",
                lbfgs_steps=result.get("completed_lbfgs_steps") or "",
            )
        )

    lines.extend(["", "Mean by config:", "", "| config | mean L2 relative error | mean PDE loss | mean BC loss | mean runtime sec |", "|---|---:|---:|---:|---:|"])
    for config_name in CONFIGS:
        rows = [result for result in results if result["config"] == config_name]
        if not rows:
            continue
        lines.append(
            "| {config} | {l2} | {pde} | {bc} | {runtime:.2f} |".format(
                config=config_name,
                l2=format_float(mean([float(row["l2_relative_error"]) for row in rows])),
                pde=format_float(mean([float(row["final_pde_loss"]) for row in rows])),
                bc=format_float(mean([float(row["final_bc_loss"]) for row in rows])),
                runtime=mean([float(row["runtime_sec"]) for row in rows]),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default="base,more_points,wider,lower_lr", help="comma-separated config names")
    parser.add_argument("--seeds", default="0,1,2", help="comma-separated seed list")
    parser.add_argument("--epochs", type=int, default=1000000)
    parser.add_argument("--max-runtime-sec", type=float, default=None, help="per-run time budget")
    parser.add_argument("--total-runtime-sec", type=float, default=None, help="split total time evenly across all runs")
    parser.add_argument("--adam-runtime-fraction", type=float, default=0.5)
    parser.add_argument("--lbfgs-steps", type=int, default=200000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--eval-grid-size", type=int, default=201)
    parser.add_argument("--print-every", type=int, default=5000)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/poisson_ablation_12h"))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_names = [name.strip() for name in args.configs.split(",") if name.strip()]
    unknown = [name for name in config_names if name not in CONFIGS]
    if unknown:
        raise ValueError(f"unknown configs: {', '.join(unknown)}")

    seeds = parse_seeds(args.seeds)
    if args.total_runtime_sec is not None:
        args.max_runtime_sec = args.total_runtime_sec / (len(config_names) * len(seeds))

    script_path = Path(__file__).with_name("train_poisson_pinn.py")
    args.output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for config_name in config_names:
        for seed in seeds:
            print(f"running {config_name} seed={seed}")
            results.append(run_one(args, config_name, CONFIGS[config_name], seed, script_path))

    csv_path = args.output_root / "summary.csv"
    md_path = args.output_root / "summary.md"
    write_csv(results, csv_path)
    write_markdown(results, md_path, args)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
