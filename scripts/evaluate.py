"""Evaluate JMDST tracking output against ground truth (Phase 10).

Scores the MOT-format results from scripts/run_jmdst.py against the unified
ground truth, computing MOTA, MOTP, IDF1, IDs, FP, FN (CLEAR MOT + IDF1 via
motmetrics) and HOTA, DetA, AssA (implemented in jmdst.eval.hota), plus FPS
if a timing file from run_jmdst.py is present.

Usage:
    python scripts/evaluate.py --results outputs/jmdst_results/uavdt/val \
        --dataset uavdt --split val \
        --report outputs/jmdst_results/uavdt/val_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jmdst.eval import (
    compute_hota,
    evaluate_clearmot,
    filter_predictions_by_ignore,
    load_ground_truth,
    load_ignore_regions,
    load_mot_predictions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Directory of MOT-format <sequence>.txt files.")
    parser.add_argument("--unified-root", default="data/unified")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="CLEAR MOT match IoU threshold.")
    parser.add_argument(
        "--no-ignore-regions",
        action="store_true",
        help="Disable UAVDT ignore-region filtering (score raw, as before). Default: apply if present.",
    )
    parser.add_argument("--timing", default=None, help="Optional timing_<split>.json for FPS.")
    parser.add_argument("--report", default=None, help="Optional markdown report path.")
    parser.add_argument("--json", default=None, help="Optional JSON results path.")
    return parser


def _fps_from_timing(path: Path | None) -> float | None:
    if path and path.is_file():
        return float(json.loads(path.read_text(encoding="utf-8")).get("fps", float("nan")))
    return None


def format_report(dataset, split, clearmot, hota, fps) -> str:
    overall = clearmot["OVERALL"]
    lines = [
        f"# JMDST Evaluation - {dataset}/{split}",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| MOTA | {overall.mota * 100:.1f} |",
        f"| MOTP | {overall.motp * 100:.1f} |",
        f"| IDF1 | {overall.idf1 * 100:.1f} |",
        f"| HOTA | {hota.hota * 100:.1f} |",
        f"| DetA | {hota.deta * 100:.1f} |",
        f"| AssA | {hota.assa * 100:.1f} |",
        f"| IDs | {overall.id_switches} |",
        f"| FP | {overall.false_positives} |",
        f"| FN | {overall.false_negatives} |",
        f"| FPS | {fps:.1f} |" if fps is not None else "| FPS | n/a |",
        "",
        "## Per-sequence (CLEAR MOT + IDF1)",
        "",
        "| Sequence | MOTA | MOTP | IDF1 | IDs | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in sorted(k for k in clearmot if k != "OVERALL"):
        r = clearmot[name]
        lines.append(
            f"| {name} | {r.mota*100:.1f} | {r.motp*100:.1f} | {r.idf1*100:.1f} | "
            f"{r.id_switches} | {r.false_positives} | {r.false_negatives} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()

    ground_truth = load_ground_truth(args.unified_root, args.dataset, args.split)
    predictions = load_mot_predictions(args.results)
    print(f"Loaded {len(ground_truth)} GT sequences, {len(predictions)} prediction files.")

    if not args.no_ignore_regions:
        ignore = load_ignore_regions(args.unified_root, args.dataset, args.split)
        if ignore:
            predictions, removed = filter_predictions_by_ignore(predictions, ground_truth, ignore)
            print(f"Applied ignore-region filtering ({len(ignore)} sequences with regions): {removed} predictions dropped.")

    clearmot = evaluate_clearmot(ground_truth, predictions, iou_threshold=args.iou_threshold)
    hota = compute_hota(ground_truth, predictions)

    timing_path = Path(args.timing) if args.timing else Path(args.results) / "timing.json"
    fps = _fps_from_timing(timing_path)

    report = format_report(args.dataset, args.split, clearmot, hota, fps)
    print("\n" + report)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report, encoding="utf-8")
    if args.json:
        payload = {
            "dataset": args.dataset,
            "split": args.split,
            "overall": {
                "MOTA": clearmot["OVERALL"].mota,
                "MOTP": clearmot["OVERALL"].motp,
                "IDF1": clearmot["OVERALL"].idf1,
                "HOTA": hota.hota,
                "DetA": hota.deta,
                "AssA": hota.assa,
                "IDs": clearmot["OVERALL"].id_switches,
                "FP": clearmot["OVERALL"].false_positives,
                "FN": clearmot["OVERALL"].false_negatives,
                "FPS": fps,
            },
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
