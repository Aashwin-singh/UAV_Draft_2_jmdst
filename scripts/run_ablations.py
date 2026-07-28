"""Run the JMDST ablation study (Phase 11, mirroring paper Sec. 3.4).

Runs a set of pipeline configurations over one dataset split, evaluates each,
and emits a comparison table.

Ablations covered:
  * detection interval tau (paper Sec. 3.4.3 / Table 3) -- the core design
    claim: periodic detection + a tracking branch trades accuracy for speed.
    tau=1 (detect every frame) is included as a reference baseline; the paper's
    own table starts at tau=2.
  * FELNet feature encoding (paper Sec. 3.4.2) -- appearance-based cascade
    matching ON vs OFF (IoU-only). NOTE: the paper compares FELNet against
    DeepSORT's ReIDNet descriptor; we have no ReIDNet weights, so this measures
    FELNet's embeddings against *no* appearance model, which bounds the
    contribution differently. Documented as a deviation.

Not covered: MSFP (paper Sec. 3.4.1) -- blocked, mamba-ssm does not build
(see project_log.md). The paper reports MSFP as worth ~+0.4 MOTA / -7% IDs.

Usage:
    python scripts/run_ablations.py --yolo <weights> --felnet <ckpt> \
        --dataset uavdt --split val --out outputs/ablations
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
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
    parser.add_argument("--yolo", required=True)
    parser.add_argument("--felnet", required=True)
    parser.add_argument("--unified-root", default="data/unified")
    parser.add_argument("--dataset", default="uavdt")
    parser.add_argument("--split", default="val")
    parser.add_argument("--conf", type=float, default=0.55)
    parser.add_argument("--taus", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--out", default="outputs/ablations")
    parser.add_argument("--skip-appearance-ablation", action="store_true")
    parser.add_argument("--report", default=None, help="Markdown report path (default: <out>/report.md).")
    return parser


def run_config(args, name: str, tau: int, no_appearance: bool) -> dict:
    """Run the pipeline for one configuration and evaluate it."""

    results_root = Path(args.out) / name
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "run_jmdst.py"),
        "--yolo", args.yolo, "--felnet", args.felnet,
        "--unified-root", args.unified_root,
        "--datasets", args.dataset, "--split", args.split,
        "--tau", str(tau), "--conf", str(args.conf),
        "--output-root", str(results_root),
    ]
    if no_appearance:
        cmd.append("--no-appearance")

    start = time.perf_counter()
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    wall = time.perf_counter() - start

    results_dir = results_root / args.dataset / args.split
    ground_truth = load_ground_truth(args.unified_root, args.dataset, args.split)
    predictions = load_mot_predictions(results_dir)

    ignore = load_ignore_regions(args.unified_root, args.dataset, args.split)
    if ignore:
        predictions, _ = filter_predictions_by_ignore(predictions, ground_truth, ignore)

    clearmot = evaluate_clearmot(ground_truth, predictions)["OVERALL"]
    hota = compute_hota(ground_truth, predictions)

    timing_path = results_dir / "timing.json"
    fps = json.loads(timing_path.read_text(encoding="utf-8"))["fps"] if timing_path.is_file() else float("nan")

    return {
        "name": name, "tau": tau, "appearance": not no_appearance,
        "MOTA": clearmot.mota * 100, "MOTP": clearmot.motp * 100,
        "IDF1": clearmot.idf1 * 100, "HOTA": hota.hota * 100,
        "DetA": hota.deta * 100, "AssA": hota.assa * 100,
        "IDs": clearmot.id_switches, "FP": clearmot.false_positives,
        "FN": clearmot.false_negatives, "FPS": fps, "wall_seconds": wall,
    }


def format_report(args, rows: list[dict]) -> str:
    header = (
        f"# JMDST Ablation Study - {args.dataset}/{args.split}\n\n"
        f"Detector conf={args.conf}. Ignore regions applied where available.\n"
        f"Mirrors paper Sec. 3.4; see script docstring for deviations.\n\n"
    )

    tau_rows = [r for r in rows if r["appearance"]]
    lines = [
        "## Detection interval tau (paper Sec. 3.4.3 / Table 3)\n",
        "| tau | MOTA | MOTP | IDF1 | HOTA | DetA | AssA | IDs | FP | FN | FPS |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(tau_rows, key=lambda r: r["tau"]):
        lines.append(
            f"| {r['tau']} | {r['MOTA']:.1f} | {r['MOTP']:.1f} | {r['IDF1']:.1f} | {r['HOTA']:.1f} | "
            f"{r['DetA']:.1f} | {r['AssA']:.1f} | {r['IDs']} | {r['FP']} | {r['FN']} | {r['FPS']:.1f} |"
        )

    ablation_rows = [r for r in rows if not r["appearance"]]
    if ablation_rows:
        baseline = next((r for r in tau_rows if r["tau"] == ablation_rows[0]["tau"]), None)
        lines += [
            "\n## FELNet feature encoding (paper Sec. 3.4.2)\n",
            "Appearance cascade ON vs OFF (IoU-only). Deviation: the paper compares",
            "against DeepSORT's ReIDNet; we have no ReIDNet weights, so OFF means",
            "no appearance model at all.\n",
            "| Association | MOTA | IDF1 | HOTA | AssA | IDs |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        if baseline:
            lines.append(
                f"| FELNet embeddings (tau={baseline['tau']}) | {baseline['MOTA']:.1f} | {baseline['IDF1']:.1f} | "
                f"{baseline['HOTA']:.1f} | {baseline['AssA']:.1f} | {baseline['IDs']} |"
            )
        for r in ablation_rows:
            lines.append(
                f"| IoU only (tau={r['tau']}) | {r['MOTA']:.1f} | {r['IDF1']:.1f} | "
                f"{r['HOTA']:.1f} | {r['AssA']:.1f} | {r['IDs']} |"
            )

    lines += [
        "\n## MSFP (paper Sec. 3.4.1)\n",
        "Not run: blocked on mamba-ssm (see project_log.md). The paper reports",
        "MSFP as worth +0.4 MOTA / +0.6 IDF1 / +0.3 HOTA / IDs 361->335.",
    ]
    return header + "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    rows: list[dict] = []

    for tau in args.taus:
        name = f"tau{tau}"
        print(f"[ablation] running {name} ...", flush=True)
        row = run_config(args, name, tau, no_appearance=False)
        rows.append(row)
        print(f"    MOTA={row['MOTA']:.1f} IDF1={row['IDF1']:.1f} HOTA={row['HOTA']:.1f} FPS={row['FPS']:.1f}", flush=True)

    if not args.skip_appearance_ablation:
        tau = 3 if 3 in args.taus else args.taus[0]
        name = f"tau{tau}_noappearance"
        print(f"[ablation] running {name} ...", flush=True)
        row = run_config(args, name, tau, no_appearance=True)
        rows.append(row)
        print(f"    MOTA={row['MOTA']:.1f} IDF1={row['IDF1']:.1f} HOTA={row['HOTA']:.1f}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report = format_report(args, rows)
    report_path = Path(args.report) if args.report else out / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Saved to {report_path}")


if __name__ == "__main__":
    main()
