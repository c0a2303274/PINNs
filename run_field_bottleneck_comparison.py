"""Supervised field-to-field Burgers pilot; deliberately not a coordinate PINN.

Both cases use identical data, depth, bottleneck, optimizer and time budget.
Only hidden widths differ. Parameter counts are NOT matched.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn


def architecture(dimension, bottleneck, kind):
    if dimension <= bottleneck or bottleneck < 1:
        raise ValueError("require dimension > bottleneck >= 1")
    descending = []
    width = dimension
    while width > bottleneck:
        if width % 2 or width // 2 < bottleneck:
            raise ValueError("dimension / bottleneck must be a power of two")
        width //= 2
        descending.append(width)
    hidden = descending + descending[-2::-1]
    if kind == "abrupt":
        hidden = [bottleneck] * len(hidden)
    elif kind != "gradual":
        raise ValueError("unknown architecture")
    return [dimension, *hidden, dimension]


def make_model(widths):
    layers = []
    for i, (left, right) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(left, right))
        if i < len(widths) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def exact_fields(parameters, grid, t, nu=0.01):
    """Periodic Cole-Hopf family: amplitude and two spatial phase shifts."""
    axis = torch.arange(grid, dtype=parameters.dtype, device=parameters.device) / grid
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    a, sx, sy = (parameters[:, i, None, None] for i in range(3))
    k = 2 * math.pi
    decay = a * math.exp(-2 * k * k * nu * t)
    xx, yy = k * (x - sx - 0.5 * t), k * (y - sy - 0.25 * t)
    phi = 1 + decay * torch.cos(xx) * torch.cos(yy)
    u = 0.5 + 2 * nu * k * decay * torch.sin(xx) * torch.cos(yy) / phi
    v = 0.25 + 2 * nu * k * decay * torch.cos(xx) * torch.sin(yy) / phi
    return torch.stack([u, v], dim=1).flatten(1)


def make_data(grid, counts, horizon, seed):
    generator = torch.Generator().manual_seed(seed)
    parameters = torch.rand(sum(counts), 3, generator=generator)
    parameters[:, 0] = 0.3 + 0.5 * parameters[:, 0]
    initial = exact_fields(parameters, grid, 0)
    target = exact_fields(parameters, grid, horizon)
    background = torch.tensor([0.5, 0.25]).repeat_interleave(grid * grid)[None, :]
    splits = []
    offset = 0
    for count in counts:
        part = slice(offset, offset + count)
        splits.append(((initial[part] - background) / 0.1,
                       (target[part] - background) / 0.1))
        offset += count
    return parameters, splits, background


def errors(predicted, target, background):
    # Inputs here are normalized fluctuations, not full velocity values.
    difference = (predicted - target).double()
    target = target.double()
    full = target + background.double() / 0.1
    per_field = difference.norm(dim=1) / target.norm(dim=1).clamp_min(1e-15)
    return {"l2_relative_error": (difference.norm() / full.norm()).item(),
            "fluctuation_l2_relative_error": (difference.norm() / target.norm()).item(),
            "mean_field_fluctuation_error": per_field.mean().item(),
            "max_field_fluctuation_error": per_field.max().item()}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def save_checkpoint(path, model, optimizer, widths, update, config):
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "widths": widths, "updates": update, "config": config}, temporary)
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--train-fields", type=int, default=512)
    parser.add_argument("--validation-fields", type=int, default=128)
    parser.add_argument("--test-fields", type=int, default=128)
    parser.add_argument("--horizon", type=float, default=0.25)
    parser.add_argument("--runtime-sec-per-case", type=float, default=60)
    parser.add_argument("--max-updates", type=int, default=100000000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=20261008)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--checkpoint-sec", type=float, default=300)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ["grid", "bottleneck", "train_fields", "validation_fields", "test_fields",
                 "batch_size", "max_updates", "cpu_threads"]:
        if getattr(args, name) <= 0:
            parser.error(name + " must be positive")
    for name in ["lr", "horizon", "runtime_sec_per_case", "checkpoint_sec"]:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(name + " must be finite and positive")
    try:
        architecture(2 * args.grid ** 2, args.bottleneck, "gradual")
    except ValueError as error:
        parser.error(str(error))
    return args


def run(args):
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    counts = [args.train_fields, args.validation_fields, args.test_fields]
    params, splits, background = make_data(args.grid, counts, args.horizon, args.data_seed)
    np.savez_compressed(root / "data_parameters.npz", parameters=params.numpy(), counts=counts)
    config = {**vars(args), "output_root": str(root), "device": str(device),
              "torch_version": torch.__version__, "platform": platform.platform(),
              "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "data_sha256": hashlib.sha256(params.numpy().tobytes()).hexdigest(),
              "nu": 0.01, "background": [0.5, 0.25], "normalization_scale": 0.1,
              "supervised": True, "pde_loss": False, "checkpoint_selection": "last",
              "budget_scope": "training loop including logging and checkpoint IO; excludes final evaluation"}
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    train, valid, test = [tuple(value.to(device) for value in pair) for pair in splits]
    background = background.to(device)
    baselines = {"persistence": errors(test[0], test[1], background),
                 "uniform_background": errors(torch.zeros_like(test[1]), test[1], background)}
    (root / "baselines.json").write_text(json.dumps(baselines, indent=2), encoding="utf-8")
    rows = []
    for kind in ["abrupt", "gradual"]:
        torch.manual_seed(args.seed)
        widths = architecture(2 * args.grid ** 2, args.bottleneck, kind)
        model = make_model(widths).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        batch_rng = torch.Generator().manual_seed(args.seed + 9000)
        folder = root / kind
        folder.mkdir()
        history = []
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        last_checkpoint = start
        updates = 0
        while updates < args.max_updates and time.perf_counter() - start < args.runtime_sec_per_case:
            indices = torch.randint(len(train[0]), (args.batch_size,), generator=batch_rng).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(train[0][indices]) - train[1][indices]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss: " + kind)
            loss.backward()
            optimizer.step()
            updates += 1
            synchronize(device)
            if updates == 1 or updates % 500 == 0:
                history.append({"updates": updates, "elapsed_sec": time.perf_counter() - start,
                                "normalized_train_batch_mse": loss.item()})
            if time.perf_counter() - last_checkpoint >= args.checkpoint_sec:
                save_checkpoint(folder / "latest.pt", model, optimizer, widths, updates, config)
                last_checkpoint = time.perf_counter()
        training_sec = time.perf_counter() - start
        model.eval()
        def predict(values):
            return torch.cat([model(part) for part in values.split(args.batch_size)])
        with torch.no_grad():
            validation = errors(predict(valid[0]), valid[1], background)
            predicted = predict(test[0])
            result = errors(predicted, test[1], background)
            sample = test[0][:1]
            for _ in range(5):
                model(sample)
            synchronize(device)
            clock = time.perf_counter()
            for _ in range(50):
                model(sample)
            synchronize(device)
            inference_ms = (time.perf_counter() - clock) * 1000 / 50
        row = {"case": kind, "widths": ",".join(map(str, widths)),
               "parameters": sum(p.numel() for p in model.parameters()), **result,
               "validation_fluctuation_error": validation["fluctuation_l2_relative_error"],
               "training_sec": training_sec, "updates": updates, "inference_ms_one_field": inference_ms,
               "stop_reason": "max_updates" if updates >= args.max_updates else "time_budget",
               "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None}
        (folder / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        (folder / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_checkpoint(folder / "latest.pt", model, optimizer, widths, updates, config)
        np.savez_compressed(folder / "test_examples.npz",
                            initial=(test[0][:4] * .1 + background).cpu().numpy(),
                            target=(test[1][:4] * .1 + background).cpu().numpy(),
                            predicted=(predicted[:4] * .1 + background).cpu().numpy(), grid=args.grid)
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model, optimizer, predicted
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


if __name__ == "__main__":
    run(parse_args())
