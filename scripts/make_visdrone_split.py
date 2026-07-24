"""Generate a reproducible 45/10/10 VisDrone train/val/test split.

The paper (Sec. 3.1) specifies "45 sequences for training, 10 for validation,
and 10 for testing from the VisDrone2019 dataset" but does not publish which
exact sequences. VisDrone2019-MOT ships as three official folders (train: 56,
val: 7, test-dev: 17 = 80 total). This script pools all 80 sequence names
(tagged with which official folder each physically lives in), takes a seeded
random sample of 65, and assigns 45/10/10 to our own train/val/test split.

Because a sequence's physical folder and its assigned split can now differ,
this writes one sequence-list file per (assigned_split, source_folder) pair,
so prepare_dataset.py can be invoked once per pair with the correct
--source-split.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

OFFICIAL_FOLDERS = {
    "train": "VisDrone2019-MOT-train",
    "val": "VisDrone2019-MOT-val",
    "test-dev": "VisDrone2019-MOT-test-dev",
}


def discover_pool(source_root: Path) -> dict[str, str]:
    """Return {sequence_name: official_source_split}."""

    pool: dict[str, str] = {}
    for source_split, folder_name in OFFICIAL_FOLDERS.items():
        sequences_dir = source_root / folder_name / "sequences"
        if not sequences_dir.is_dir():
            continue
        for path in sequences_dir.iterdir():
            if path.is_dir():
                pool[path.name] = source_split
    return pool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Folder containing the VisDrone2019-MOT-* folders.")
    parser.add_argument("--out-dir", default="configs/splits")
    parser.add_argument("--select", type=int, default=65)
    parser.add_argument("--train", type=int, required=True)
    parser.add_argument("--val", type=int, required=True)
    parser.add_argument("--test", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pool = discover_pool(Path(args.source_root))
    print(f"Discovered {len(pool)} sequences across official folders:")
    counts_by_origin: dict[str, int] = defaultdict(int)
    for origin in pool.values():
        counts_by_origin[origin] += 1
    for origin, count in sorted(counts_by_origin.items()):
        print(f"  {origin}: {count}")

    if args.select > len(pool):
        raise ValueError(f"Requested select={args.select} but pool only has {len(pool)} sequences.")
    if args.train + args.val + args.test > args.select:
        raise ValueError("train+val+test exceeds --select.")

    names = list(pool.keys())
    rng = random.Random(args.seed)
    rng.shuffle(names)
    chosen = names[: args.select]

    assigned_split = {}
    for name in chosen[: args.train]:
        assigned_split[name] = "train"
    for name in chosen[args.train : args.train + args.val]:
        assigned_split[name] = "val"
    for name in chosen[args.train + args.val : args.train + args.val + args.test]:
        assigned_split[name] = "test"

    # Group by (assigned_split, physical origin) so each group can be
    # converted with a single prepare_dataset.py invocation.
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for name, split in assigned_split.items():
        groups[(split, pool[name])].append(name)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nAssignment (split <- origin: count):")
    plan_lines = []
    for (split, origin), seq_names in sorted(groups.items()):
        seq_names = sorted(seq_names)
        list_path = out_dir / f"visdrone_{split}_from-{origin}.txt"
        list_path.write_text("\n".join(seq_names) + "\n", encoding="utf-8")
        print(f"  {split} <- {origin}: {len(seq_names)}  ({list_path})")
        plan_lines.append(f"{split}\t{origin}\t{list_path.as_posix()}")

    plan_path = out_dir / "visdrone_conversion_plan.tsv"
    plan_path.write_text("\n".join(plan_lines) + "\n", encoding="utf-8")
    print(f"\nConversion plan written to {plan_path}")


if __name__ == "__main__":
    main()
