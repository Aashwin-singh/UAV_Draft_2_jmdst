"""Run an edge/ground sweep plan: one simulation per configuration, scored.

Reads a plan (configs/sweeps/*.json). Each group varies one factor -- edge
hardware, detector placement, link delay / bandwidth / loss, or the reply
deadline -- from a common base; configurations that coincide across groups
(the base itself, usually) run once. Every run uses the modelled ground
station (a detection cache), so ground compute is the same per frame in every
configuration and cannot be slowed by the edge's CPU load.

For each configuration: run_sim.py (or edge_node.py alone for the onboard
design, which has no network), then evaluate.py per dataset. Results collect
into <out>/results.csv, rebuilt from every finished run each time.

**Resumable**: a configuration whose eval.json files all exist is skipped, so
an interrupted sweep continues where it stopped. Delete a run's folder to redo
it.

Usage (one line in cmd.exe):
    python scripts/run_sweep.py --plan configs/sweeps/edge_ground_val.json --dry-run
    python scripts/run_sweep.py --plan configs/sweeps/edge_ground_val.json
    python scripts/run_sweep.py --plan configs/sweeps/edge_ground_val.json --groups hardware loss
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_FIELDS = [
    "run", "groups", "edge", "detector", "delay_ms", "rate_mbit", "loss", "deadline_ms", "dataset",
    "HOTA", "MOTA", "IDF1", "DetA", "AssA", "IDs", "frames", "processed", "skipped_pct",
    "latency_p50_ms", "latency_p95_ms", "detection_frames", "replies_ok_pct", "rtt_p50_ms",
    "edge_cores_used", "cache_misses", "wall_s",
]


def link_specs(cfg: dict) -> tuple[str, str]:
    """netem specs for both directions. Jitter is 20% of the delay; loss applies
    to both directions with independent seeds; the rate limits the uplink,
    which carries the frames (replies are ~1 KB)."""

    d, rate, loss = cfg["delay_ms"], cfg["rate_mbit"], cfg["loss"]
    delay = f"delay {d}ms {d * 0.2:g}ms" if d else ""
    loss_part = f"loss {loss}" if loss else ""
    e2g = " ".join(p for p in (delay, f"rate {rate}mbit" if rate else "", loss_part, "seed 1") if p)
    g2e = " ".join(p for p in (delay, loss_part, "seed 2") if p)
    return e2g, g2e


def expand(plan: dict, groups: list[str] | None) -> list[dict]:
    configs: dict[str, dict] = {}
    for group, spec in plan["groups"].items():
        if groups and group not in groups:
            continue
        for value in spec["values"]:
            cfg = {**plan["base"], **spec.get("set", {}), spec["vary"]: value}
            key = json.dumps({k: cfg[k] for k in sorted(cfg)}, sort_keys=True)
            if key in configs:
                configs[key]["groups"].append(f"{group}={value}")
                continue
            label = str(value).replace(" ", "").replace("%", "pct").replace(".", "p") or "none"
            configs[key] = {**cfg, "name": f"{group}_{label}", "groups": [f"{group}={value}"]}
    return list(configs.values())


def run_one(plan: dict, cfg: dict, out: Path, max_frames: int | None, py: list[str]) -> float:
    run_dir = out / cfg["name"]
    edge_args = [*plan["edge_common"], *plan["edges"][cfg["edge"]], "--datasets", *plan["datasets"],
                 "--split", plan["split"], "--deadline-ms", str(cfg["deadline_ms"])]
    if max_frames:
        edge_args += ["--max-frames", str(max_frames)]
    start = time.perf_counter()
    if cfg["detector"] == "onboard":
        # No network in this design, so no ground or link processes.
        cmd = py + [str(PROJECT_ROOT / "scripts/edge_node.py"), "--felnet", plan["felnet"], "--detector", "onboard",
                    "--yolo", plan["yolo"], "--output-root", str(run_dir), *edge_args]
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "edge.log", "w", encoding="utf-8") as log:
            code = subprocess.call(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
    else:
        e2g, g2e = link_specs(cfg)
        cmd = py + [str(PROJECT_ROOT / "scripts/run_sim.py"), "--yolo", plan["yolo"], "--felnet", plan["felnet"],
                    "--e2g", e2g, "--g2e", g2e, "--ground-cache", plan["ground_cache"],
                    "--host-cpus", plan["host_cpus"], "--output-root", str(run_dir), *edge_args]
        code = subprocess.call(cmd, cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if code != 0:
        raise RuntimeError(f"{cfg['name']}: simulation exited with code {code}; see {run_dir}/logs")
    for dataset in plan["datasets"]:
        split_dir = run_dir / dataset / plan["split"]
        subprocess.check_call(py + [str(PROJECT_ROOT / "scripts/evaluate.py"), "--results", str(split_dir),
                                    "--dataset", dataset, "--split", plan["split"], "--json", str(split_dir / "eval.json")],
                              cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wall = time.perf_counter() - start
    (run_dir / "sweep_config.json").write_text(json.dumps({**cfg, "wall_s": wall}, indent=2), encoding="utf-8")
    return wall


def done(plan: dict, cfg: dict, out: Path) -> bool:
    return all((out / cfg["name"] / d / plan["split"] / "eval.json").exists() for d in plan["datasets"]) \
        and (out / cfg["name"] / "sweep_config.json").exists()


def collect(plan: dict, configs: list[dict], out: Path) -> list[dict]:
    rows = []
    for cfg in configs:
        if not done(plan, cfg, out):
            continue
        run_dir = out / cfg["name"]
        wall = json.loads((run_dir / "sweep_config.json").read_text(encoding="utf-8")).get("wall_s")
        misses = ""
        if (run_dir / "ground.csv").exists():
            with open(run_dir / "ground.csv", encoding="utf-8") as f:
                misses = sum(r.get("source") == "live (cache miss)" for r in csv.DictReader(f))
        for dataset in plan["datasets"]:
            split_dir = run_dir / dataset / plan["split"]
            e = json.loads((split_dir / "eval.json").read_text(encoding="utf-8"))["overall"]
            t = json.loads((split_dir / "timing.json").read_text(encoding="utf-8"))
            sent = t.get("detection_frames", 0)
            rows.append({
                "run": cfg["name"], "groups": " ".join(cfg["groups"]), "edge": cfg["edge"], "detector": cfg["detector"],
                "delay_ms": cfg["delay_ms"], "rate_mbit": cfg["rate_mbit"], "loss": cfg["loss"] or "0",
                "deadline_ms": cfg["deadline_ms"], "dataset": dataset,
                **{k: round(e[k] * 100, 2) for k in ("HOTA", "MOTA", "IDF1", "DetA", "AssA")}, "IDs": e["IDs"],
                "frames": t["frames"], "processed": t["processed"],
                "skipped_pct": round(100 * t["skipped"] / t["frames"], 2),
                "latency_p50_ms": round(t["latency_ms"]["p50"], 1) if t.get("latency_ms") else "",
                "latency_p95_ms": round(t["latency_ms"]["p95"], 1) if t.get("latency_ms") else "",
                "detection_frames": sent,
                "replies_ok_pct": round(100 * t["outcomes"]["ok"] / sent, 1) if sent else "",
                "rtt_p50_ms": round(t["rtt_ms"]["p50"], 1) if t.get("rtt_ms") else "",
                "edge_cores_used": round(t["edge_cpu_cores_used"], 2), "cache_misses": misses,
                "wall_s": round(wall or 0),
            })
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", default="outputs/sim/sweep")
    parser.add_argument("--groups", nargs="+", default=None, help="Only these groups.")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap frames per sequence (pilot runs).")
    parser.add_argument("--dry-run", action="store_true", help="List the configurations and stop.")
    args = parser.parse_args(argv)

    plan = json.loads((PROJECT_ROOT / args.plan).read_text(encoding="utf-8"))
    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    configs = expand(plan, args.groups)
    todo = [c for c in configs if not done(plan, c, out)]
    print(f"{len(configs)} configurations ({len(configs) - len(todo)} already done), datasets {plan['datasets']} "
          f"{plan['split']} -> {out.relative_to(PROJECT_ROOT)}")
    for cfg in configs:
        e2g, g2e = link_specs(cfg)
        where = "edge only (no network)" if cfg["detector"] == "onboard" else f"e2g '{e2g}' | g2e '{g2e}'"
        print(f"  {'done ' if cfg not in todo else 'to do'}  {cfg['name']:22s} edge={cfg['edge']:3s} "
              f"deadline={cfg['deadline_ms']}  {where}   [{', '.join(cfg['groups'])}]")
    if args.dry_run:
        return 0

    py = [sys.executable, "-u"]
    for i, cfg in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {cfg['name']} ...", flush=True)
        wall = run_one(plan, cfg, out, args.max_frames, py)
        rows = [r for r in collect(plan, configs, out) if r["run"] == cfg["name"]]
        print("    " + " | ".join(f"{r['dataset']}: HOTA {r['HOTA']:.1f}, processed {r['processed']}/{r['frames']}, "
                                   f"replies ok {r['replies_ok_pct']}%" for r in rows) + f"  ({wall / 60:.1f} min)", flush=True)
    collect(plan, configs, out)
    print(f"\nResults: {(out / 'results.csv').relative_to(PROJECT_ROOT)}. "
          f"Report: python scripts/sweep_report.py --results {args.out}/results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
