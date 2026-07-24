"""Generate reproducible train/val/test sequence-list files.

The paper (Sec. 3.1) specifies split *counts* per dataset (e.g. UAVDT: 23 of
50 available sequences, partitioned 12/3/8) but does not publish which exact
sequences go in which split. This script picks a seeded random sample so the
split is reproducible and documented, rather than guessing at the paper's
undisclosed assignment.

Usage:
    python scripts/make_splits.py \
        --pool-dir Datasets/UAVDT/UAV-benchmark-M \
        --out-dir configs/splits --dataset uavdt \
        --select 23 --train 12 --val 3 --test 8 --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def discover_pool(pool_dir: Path, prefix: str | None = None) -> list[str]:
    names = [p.name for p in pool_dir.iterdir() if p.is_dir()]
    if prefix:
        names = [n for n in names if n.upper().startswith(prefix.upper())]
    return sorted(names)


def make_split(
    pool: list[str],
    select: int | None,
    train: int,
    val: int,
    test: int,
    seed: int,
) -> dict[str, list[str]]:
    if select is not None and select > len(pool):
        raise ValueError(f"Requested select={select} but pool only has {len(pool)} sequences.")
    if train + val + test > (select if select is not None else len(pool)):
        raise ValueError("train+val+test exceeds the selected pool size.")

    rng = random.Random(seed)
    chosen = list(pool)
    rng.shuffle(chosen)
    if select is not None:
        chosen = chosen[:select]

    return {
        "train": sorted(chosen[:train]),
        "val": sorted(chosen[train : train + val]),
        "test": sorted(chosen[train + val : train + val + test]),
    }


def write_split_files(splits: dict[str, list[str]], out_dir: Path, dataset: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, names in splits.items():
        path = out_dir / f"{dataset}_{split_name}.txt"
        path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")
        print(f"{path}: {len(names)} sequences")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", required=True, help="Directory with one subfolder per sequence.")
    parser.add_argument("--pool-prefix", default=None, help="Optional name prefix filter, e.g. M for UAVDT.")
    parser.add_argument("--out-dir", default="configs/splits")
    parser.add_argument("--dataset", required=True, help="Dataset name, used in output filenames.")
    parser.add_argument("--select", type=int, default=None, help="Sequences to sample from the pool before splitting.")
    parser.add_argument("--train", type=int, required=True)
    parser.add_argument("--val", type=int, required=True)
    parser.add_argument("--test", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pool = discover_pool(Path(args.pool_dir), prefix=args.pool_prefix)
    print(f"Discovered {len(pool)} sequences under {args.pool_dir}")
    splits = make_split(
        pool,
        select=args.select,
        train=args.train,
        val=args.val,
        test=args.test,
        seed=args.seed,
    )
    write_split_files(splits, Path(args.out_dir), args.dataset)


if __name__ == "__main__":
    main()
