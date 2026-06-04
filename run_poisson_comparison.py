#!/usr/bin/env python3
"""Run the Poisson optimizer comparison and write a compact result table."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


METHODS = {
    "adam": {
        "label": "Adam",
        "extra_args": [],
        "wandb_group": "poisson-adam",
        "wandb_tags": "poisson,baseline,adam",
    },
    "adam_lbfgs": {
        "label": "Adam->L-BFGS",
        "extra_args": ["--lbfgs-steps"],
        "wandb_group": "poisson-adam-lbfgs",
        "wandb_tags": "poisson,baseline,adam,lbfgs",
    },
}


def parse_seeds(raw: str) -> list[int]:
    return [int(seed.strip()) for seed in raw.split(",") if seed.strip()]


def format_float(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6e}"


def run_one(args: argparse.Namespace, method_key: str, seed: int, script_path: Path) -> dict[str, object]:
    method = METHODS[method_key]
    output_dir = args.output_root / method_key / f"seed_{seed}"
    metrics_path = output_dir / "metrics.json"

    if args.skip_existing and metrics_path.exists():
        return load_result(metrics_path, method["label"], seed)

    command = [
        sys.executable,
        str(script_path),
        "--epochs",
        str(args.epochs),
        "--n-interior",
        str(args.n_interior),
        "--n-boundary",
        str(args.n_boundary),
        "--hidden-dim",
        str(args.hidden_dim),
        "--hidden-layers",
        str(args.hidden_layers),
        "--lr",
        str(args.lr),
        "--lambda-bc",
        str(args.lambda_bc),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--print-every",
        str(args.print_every),
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-group",
        method["wandb_group"],
        "--wandb-tags",
        method["wandb_tags"],
        "--wandb-notes",
        f"{method['label']} Poisson optimizer comparison, seed={seed}",
    ]
    if method_key == "adam_lbfgs":
        command.extend(["--lbfgs-steps", str(args.lbfgs_steps)])
    if args.device != "auto":
        command.extend(["--device", args.device])

    subprocess.run(command, check=True, cwd=script_path.parent)
    return load_result(metrics_path, method["label"], seed)


def load_result(metrics_path: Path, method: str, seed: int) -> dict[str, object]:
    with metrics_path.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)

    return {
        "method": method,
        "seed": seed,
        "l2_relative_error": metrics.get("l2_relative_error"),
        "final_pde_loss": metrics.get("final_pde_loss"),
        "final_bc_loss": metrics.get("final_bc_loss"),
        "runtime_sec": metrics.get("runtime_sec"),
        "output_dir": str(metrics_path.parent),
    }


def write_csv(results: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "method",
                "seed",
                "l2_relative_error",
                "final_pde_loss",
                "final_bc_loss",
                "runtime_sec",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(results)


def write_markdown(results: list[dict[str, object]], path: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Poisson Optimizer Comparison",
        "",
        "Fixed setting:",
        "",
        f"- epochs: {args.epochs}",
        f"- n_interior: {args.n_interior}",
        f"- n_boundary: {args.n_boundary}",
        f"- hidden_dim: {args.hidden_dim}",
        f"- hidden_layers: {args.hidden_layers}",
        f"- lr: {args.lr}",
        f"- lambda_bc: {args.lambda_bc}",
        f"- seeds: {', '.join(str(seed) for seed in parse_seeds(args.seeds))}",
        "",
        "| method | seed | L2 relative error | PDE loss | BC loss | runtime sec |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| {method} | {seed} | {l2} | {pde} | {bc} | {runtime:.2f} |".format(
                method=result["method"],
                seed=result["seed"],
                l2=format_float(result["l2_relative_error"]),
                pde=format_float(result["final_pde_loss"]),
                bc=format_float(result["final_bc_loss"]),
                runtime=float(result["runtime_sec"]),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2", help="comma-separated seed list")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--n-interior", type=int, default=1024)
    parser.add_argument("--n-boundary", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-bc", type=float, default=1.0)
    parser.add_argument("--lbfgs-steps", type=int, default=200)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device string")
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/poisson_optimizer_comparison"))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).with_name("train_poisson_pinn.py")
    args.output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in parse_seeds(args.seeds):
        for method_key in METHODS:
            print(f"running {METHODS[method_key]['label']} seed={seed}")
            results.append(run_one(args, method_key, seed, script_path))

    csv_path = args.output_root / "summary.csv"
    md_path = args.output_root / "summary.md"
    write_csv(results, csv_path)
    write_markdown(results, md_path, args)

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
