import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


CONFIGS = {
    "soft": {
        "constraint_mode": "soft",
        "lr": 1.0e-3,
        "lbfgs_steps": 0,
        "adam_fraction": None,
    },
    "hard_icbc": {
        "constraint_mode": "hard-icbc",
        "lr": 1.0e-3,
        "lbfgs_steps": 0,
        "adam_fraction": None,
    },
    "hard_icbc_lbfgs": {
        "constraint_mode": "hard-icbc",
        "lr": 1.0e-3,
        "lbfgs_steps": 50000,
        "adam_fraction": 0.5,
    },
    "bounded_hard_icbc": {
        "constraint_mode": "bounded-hard-icbc",
        "lr": 1.0e-3,
        "lbfgs_steps": 0,
        "adam_fraction": None,
    },
    "bounded_hard_icbc_lbfgs": {
        "constraint_mode": "bounded-hard-icbc",
        "lr": 1.0e-3,
        "lbfgs_steps": 50000,
        "adam_fraction": 0.5,
        "bound_amplitude": 1.0,
        "sampling": "uniform",
    },
    "bounded_amp2_lbfgs": {
        "constraint_mode": "bounded-hard-icbc",
        "lr": 1.0e-3,
        "lbfgs_steps": 50000,
        "adam_fraction": 0.5,
        "bound_amplitude": 2.0,
        "sampling": "uniform",
    },
    "hard_icbc_focused_lbfgs": {
        "constraint_mode": "hard-icbc",
        "lr": 1.0e-3,
        "lbfgs_steps": 50000,
        "adam_fraction": 0.5,
        "sampling": "shock-focused",
    },
    "bounded_amp2_focused_lbfgs": {
        "constraint_mode": "bounded-hard-icbc",
        "lr": 1.0e-3,
        "lbfgs_steps": 50000,
        "adam_fraction": 0.5,
        "bound_amplitude": 2.0,
        "sampling": "shock-focused",
    },
}


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def validate_configs(names: list[str]) -> None:
    unknown = sorted(set(names) - set(CONFIGS))
    if unknown:
        raise ValueError(f"unknown configs: {unknown}; choices={sorted(CONFIGS)}")


def run_one(args: argparse.Namespace, config_name: str, seed: int, output_dir: Path) -> dict[str, object]:
    config = CONFIGS[config_name]
    cmd = [
        sys.executable,
        "train_burgers_pinn.py",
        "--constraint-mode",
        str(config["constraint_mode"]),
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
        str(config["lr"]),
        "--lambda-ic",
        str(args.lambda_ic),
        "--lambda-bc",
        str(args.lambda_bc),
        "--bound-amplitude",
        str(config.get("bound_amplitude", args.bound_amplitude)),
        "--sampling",
        str(config.get("sampling", args.sampling)),
        "--focus-fraction",
        str(args.focus_fraction),
        "--focus-std",
        str(args.focus_std),
        "--nu",
        str(args.nu),
        "--lbfgs-steps",
        str(config["lbfgs_steps"]),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-group",
        f"burgers-integrated-{config_name}",
        "--wandb-tags",
        f"burgers,{config_name},integrated",
    ]
    if config["adam_fraction"] is not None:
        cmd.extend(["--adam-max-runtime-sec", str(args.runtime_sec * float(config["adam_fraction"]))])

    subprocess.run(cmd, check=True)
    with (output_dir / "metrics.json").open(encoding="utf-8") as fh:
        metrics = json.load(fh)
    metrics["config_name"] = config_name
    return metrics


def write_summary(rows: list[dict[str, object]], output_root: Path) -> None:
    keys = [
        "config_name",
        "constraint_mode",
        "seed",
        "l2_relative_error",
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
        fh.write("# Burgers integrated comparison\n\n")
        fh.write("| config | mode | seed | L2 relative error | PDE loss | IC loss | BC loss | runtime sec | epochs | lbfgs steps |\n")
        fh.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            fh.write(
                f"| {row.get('config_name')} | {row.get('constraint_mode')} | {row.get('seed')} | "
                f"{float(row.get('l2_relative_error', 0.0)):.6e} | "
                f"{float(row.get('final_pde_loss', 0.0)):.6e} | "
                f"{float(row.get('final_ic_loss', 0.0)):.6e} | "
                f"{float(row.get('final_bc_loss', 0.0)):.6e} | "
                f"{float(row.get('runtime_sec', 0.0)):.2f} | "
                f"{row.get('completed_epochs')} | {row.get('completed_lbfgs_steps')} |\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run integrated Burgers hard-constraint comparisons.")
    parser.add_argument("--configs", type=str, default="soft,hard_icbc_lbfgs,bounded_hard_icbc_lbfgs")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--runtime-sec", type=float, default=7200.0)
    parser.add_argument("--epochs", type=int, default=1_000_000)
    parser.add_argument("--n-interior", type=int, default=4096)
    parser.add_argument("--n-initial", type=int, default=512)
    parser.add_argument("--n-boundary", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=5)
    parser.add_argument("--lambda-ic", type=float, default=1.0)
    parser.add_argument("--lambda-bc", type=float, default=1.0)
    parser.add_argument("--bound-amplitude", type=float, default=1.0)
    parser.add_argument("--sampling", choices=["uniform", "shock-focused"], default="uniform")
    parser.add_argument("--focus-fraction", type=float, default=0.5)
    parser.add_argument("--focus-std", type=float, default=0.2)
    parser.add_argument("--nu", type=float, default=0.01 / 3.141592653589793)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/burgers_integrated_comparison"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_names = parse_csv(args.configs)
    validate_configs(config_names)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed_text in parse_csv(args.seeds):
        seed = int(seed_text)
        for config_name in config_names:
            output_dir = args.output_root / f"{config_name}_seed{seed}"
            rows.append(run_one(args, config_name, seed, output_dir))
            write_summary(rows, args.output_root)
    write_summary(rows, args.output_root)
    print(f"saved summary to {args.output_root}")


if __name__ == "__main__":
    main()
