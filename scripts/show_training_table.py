"""Print a detector run's per-epoch metrics as a terminal table.

Ultralytics writes these to results.csv during training but the live console
output is not retained afterwards. This reads that file back, so the table is
the same data the console printed, formatted for a screenshot.

    python scripts/show_training_table.py
    python scripts/show_training_table.py --run uavdt_only_m_lowlr

Exists as a script rather than a `python -c` one-liner because quoting an
f-string through cmd.exe and PowerShell differently is a reliable way to waste
ten minutes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# (csv column, header label, width, decimals)
COLUMNS = [
    ("epoch", "epoch", 6, 0),
    ("train/box_loss", "box_loss", 11, 4),
    ("train/cls_loss", "cls_loss", 11, 4),
    ("train/dfl_loss", "dfl_loss", 11, 4),
    ("metrics/precision(B)", "precision", 11, 4),
    ("metrics/recall(B)", "recall", 10, 4),
    ("metrics/mAP50(B)", "mAP50", 10, 4),
    ("metrics/mAP50-95(B)", "mAP50-95", 11, 4),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="retrain_aug_m_lowlr", help="Run name under outputs/yolo_runs/.")
    parser.add_argument("--project", default="outputs/yolo_runs")
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / args.project / args.run / "results.csv"
    if not csv_path.exists():
        raise SystemExit(f"No results.csv at {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(handle)]

    present = [c for c in COLUMNS if c[0] in rows[0]]
    fitness = "metrics/mAP50-95(B)"   # ultralytics 8.4 selects best.pt on this alone
    best = max(rows, key=lambda r: float(r[fitness]))

    print(f"Detector training - {args.run}")
    print(f"source: {csv_path.relative_to(PROJECT_ROOT)}   ({len(rows)} epochs)")
    print()
    header = "".join(f"{label:>{width}}" for _key, label, width, _dp in present)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = "".join(f"{float(row[key]):>{width}.{dp}f}" for key, _label, width, dp in present)
        print(line + ("   <-- best (best.pt)" if row is best else ""))
    print()
    print(f"best epoch {int(float(best['epoch']))}: "
          f"mAP50 {float(best['metrics/mAP50(B)']):.4f}, "
          f"mAP50-95 {float(best[fitness]):.4f}, "
          f"P {float(best['metrics/precision(B)']):.4f}, "
          f"R {float(best['metrics/recall(B)']):.4f}")


if __name__ == "__main__":
    main()
