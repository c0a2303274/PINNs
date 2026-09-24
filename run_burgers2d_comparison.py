"""Run width/time cases A-D or time-allocation cases C,E sequentially on one GPU."""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from train_burgers2d_pinn import save_json, training_budgets, widths_arg, write_csv


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", default="A,B,C,D", help="A-D: original matrix; C,E: equal vs front-loaded time budgets")
    p.add_argument("--first-window-fraction", type=float, default=0.5, help="case E only; C keeps equal budgets")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runtime-sec", type=float, default=600, help="TOTAL training seconds PER CASE, not per window")
    p.add_argument("--epochs", type=int, default=1_000_000, help="TOTAL update cap PER CASE")
    p.add_argument("--windows", type=int, default=5)
    p.add_argument("--uniform-widths", type=widths_arg, default=widths_arg("128,128,128,128,128"))
    p.add_argument("--variable-widths", type=widths_arg, default=widths_arg("256,128,64,128,256"))
    p.add_argument("--n-interior", type=int, default=1024)
    p.add_argument("--n-initial", type=int, default=256)
    p.add_argument("--n-boundary", type=int, default=128)
    p.add_argument("--eval-grid-size", type=int, default=48)
    p.add_argument("--eval-times", type=int, default=21)
    p.add_argument("--eval-residual-points", type=int, default=1024)
    p.add_argument("--inference-repeats", type=int, default=20)
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    p.add_argument("--output-root", type=Path, default=Path("outputs/burgers2d_pilot"))
    p.add_argument("--share-dir", type=Path, help="optional NEW directory for small Git-shareable outputs only")
    args = p.parse_args(argv)
    args.cases = [x.strip().upper() for x in args.cases.split(",")]
    if not args.cases or len(set(args.cases)) != len(args.cases) or any(c not in "ABCDE" or len(c) != 1 for c in args.cases):
        p.error("--cases must contain unique case letters A,B,C,D,E")
    if not math.isfinite(args.runtime_sec) or args.runtime_sec <= 0:
        p.error("runtime must be positive and finite")
    if min(args.epochs, args.n_interior, args.n_initial, args.n_boundary, args.eval_residual_points, args.inference_repeats) < 1:
        p.error("counts must be positive")
    if args.windows < 2 or args.epochs < args.windows or args.eval_grid_size < 4 or args.eval_times < 2 or args.seed < 0:
        p.error("require windows>=2, epochs>=windows, grid>=4, eval-times>=2, seed>=0")
    try:
        training_budgets("marching", args.windows, args.runtime_sec, args.first_window_fraction)
    except ValueError as exc:
        p.error(str(exc))
    return args


def summary(rows, root):
    write_csv(root / "summary.csv", rows)
    columns = ["case", "mode", "hidden_widths", "window_budgets_sec", "parameter_count_per_model", "l2_relative_error",
               "initial_time_l2_relative_error", "final_time_l2_relative_error",
               "fluctuation_l2_relative_error", "pde_rmse", "ic_rmse", "max_interface_rmse",
               "training_sec", "completed_epochs", "inference_time_ms"]
    lines = ["# 2D Burgers width/time/allocation comparison", "", "Pilot, seed 0 by default; one trial is not a robustness result.",
             "Same total training-loop budget per case. Different widths have different parameter counts.",
             "C: equal window budgets; E: extra first-window time, less time for later windows (same width as C).",
             "Fluctuation error uses the reference minus its uniform background in the denominator.", "",
             "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        formatted = [f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k]) for k in columns]
        lines.append("| " + " | ".join(formatted) + " |")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"use a new --output-root, not {root}")
    if args.share_dir:
        share = args.share_dir.resolve()
        if share == root or root in share.parents or share in root.parents:
            raise ValueError("share and output directories must be separate, not nested")
        if share.exists():
            raise FileExistsError(f"use a new --share-dir, not {share}")
    root.mkdir(parents=True, exist_ok=True)
    trainer = Path(__file__).resolve().with_name("train_burgers2d_pinn.py")
    rows, commands = [], []
    print(f"{len(args.cases)} cases, training budget {len(args.cases)*args.runtime_sec/60:.1f} min total + setup/evaluation", flush=True)
    for case in args.cases:
        widths = args.uniform_widths if case in "ACE" else args.variable_widths
        mode = "global" if case in "AB" else "marching"
        name = f"{case}_{mode}_seed{args.seed}"
        dest = root / name
        command = [sys.executable, str(trainer), "--hidden-widths", ",".join(map(str, widths)),
                   "--mode", mode, "--output-dir", str(dest)]
        for key in ["seed", "runtime_sec", "epochs", "windows", "n_interior", "n_initial", "n_boundary",
                    "eval_grid_size", "eval_times", "eval_residual_points", "inference_repeats", "device", "dtype", "wandb_mode"]:
            command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
        if case == "E":
            command += ["--first-window-fraction", str(args.first_window_fraction)]
        commands.append({"case": case, "command": command})
        save_json(root / "commands.json", commands)
        budgets = training_budgets(mode, args.windows, args.runtime_sec,
                                   args.first_window_fraction if case == "E" else None)
        print(f"Starting {name}: {widths}, window budgets {budgets} seconds", flush=True)
        subprocess.run(command, check=True)
        metrics = json.loads((dest / "metrics.json").read_text(encoding="utf-8"))
        row = {"case": case, **metrics, "hidden_widths": ",".join(map(str, widths)),
               "edges": json.dumps(metrics["edges"]),
               "window_budgets_sec": json.dumps(metrics["window_budgets_sec"])}
        rows.append(row)
        summary(rows, root)
        if args.share_dir:
            target = args.share_dir.resolve() / name
            target.mkdir(parents=True)
            for file in dest.iterdir():
                if file.suffix in {".json", ".csv", ".png"}:
                    shutil.copy2(file, target / file.name)
            for filename in ("summary.csv", "summary.md", "commands.json"):
                shutil.copy2(root / filename, args.share_dir / filename)
    return rows


if __name__ == "__main__":
    run(parse_args())
