#!/usr/bin/env python3
"""Append a dated entry to the local thesis research log."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="research_log.md", help="markdown log path")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--pde", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--changes", required=True)
    parser.add_argument("--done", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--open-issue", default="")
    parser.add_argument("--next-step", required=True)
    parser.add_argument("--wandb-run", default="", help="optional W&B run path, URL, or run id")
    args = parser.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = "\n".join(
        [
            f"## {date.today().isoformat()}",
            "",
            f"- Stage: {args.stage}",
            f"- PDE: {args.pde}",
            f"- Problem setting: {args.setting}",
            f"- Methods or changes: {args.changes}",
            f"- What was done: {args.done}",
            f"- Result: {args.result}",
            f"- W&B run: {args.wandb_run or 'None'}",
            f"- Open issue: {args.open_issue or 'None'}",
            f"- Next step: {args.next_step}",
            "",
        ]
    )

    prefix = "" if not log_path.exists() or log_path.read_text(encoding="utf-8").endswith("\n\n") else "\n"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(prefix + entry)

    print(f"Appended entry to {log_path.resolve()}")


if __name__ == "__main__":
    main()
