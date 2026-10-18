"""Turn a sweep's results.csv into tables and charts (Markdown + PNG).

One section per sweep group. Each table varies one factor from the base
configuration. Charts are small multiples, one y-axis each -- HOTA, and the
mechanism behind it (detection replies on time, camera frames processed) --
with one line per dataset. Every chart has its table beside it in report.md.

Usage (one line in cmd.exe):
    python scripts/sweep_report.py --results outputs/sim/sweep/results.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Reference palette (validated: two-slot CVD dE 24.7, normal-vision dE 33.6, >= 3:1 on the surface).
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"uavdt": "#2a78d6", "visdrone": "#eb6834"}
DATASET_LABEL = {"uavdt": "UAVDT", "visdrone": "VisDrone"}
EDGE_ORDER = ["gpu", "e8", "e4", "e2", "e1"]
EDGE_LABEL = {"gpu": "GPU", "e8": "8 E-cores", "e4": "4 E-cores", "e2": "2 E-cores", "e1": "1 E-core"}

GROUPS = {
    # group: (axis label, how to read the value, how to print it)
    "hardware": ("Edge compute", None, None),
    "onboard": ("Edge compute", None, None),
    "latency": ("One-way delay (ms), both directions", float, lambda v: f"{v:g} ms"),
    "bandwidth": ("Uplink bandwidth (Mbit/s)", float, lambda v: f"{v:g} Mbit/s"),
    "loss": ("Packet loss, both directions (%)", None, None),
    "deadline": ("Reply deadline (ms)", float, lambda v: f"{v:g} ms"),
}
METRICS = [("HOTA", "HOTA"), ("replies_ok_pct", "Detection replies on time (%)"), ("processed_pct", "Camera frames processed (%)")]


def _loss_value(text: str) -> tuple[float, str]:
    """Plot position and label: random loss at its rate; GE at its mean rate."""

    if not text or text == "0":
        return 0.0, "none"
    m = re.fullmatch(r"gemodel ([0-9.]+)% ([0-9.]+)%", text)
    if m:
        p, r = float(m.group(1)) / 100, float(m.group(2)) / 100
        return 100 * p / (p + r), f"bursty ({text})"
    return float(text.rstrip("%")), text


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["processed_pct"] = 100 * float(r["processed"]) / float(r["frames"])
        r["memberships"] = dict(g.split("=", 1) for g in r["groups"].split())
    return rows


def group_rows(rows: list[dict], group: str) -> list[dict]:
    out = []
    for r in rows:
        if group in r["memberships"]:
            out.append({**r, "value": r["memberships"][group]})
    return out


def x_of(group: str, value: str) -> tuple[float, str]:
    if group in ("hardware", "onboard"):
        return float(EDGE_ORDER.index(value)), EDGE_LABEL[value]
    if group == "loss":
        return _loss_value(value)
    parse, fmt = GROUPS[group][1], GROUPS[group][2]
    return parse(value), fmt(parse(value))


def _f(v, digits=1) -> str:
    return "" if v in ("", None) else f"{float(v):.{digits}f}"


def table(group: str, rows: list[dict]) -> str:
    lines = [f"| {GROUPS[group][0]} | Dataset | HOTA | MOTA | IDF1 | IDs | frames processed | replies on time | RTT p50 | latency p50 | edge cores |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: (x_of(group, r["value"])[0], r["dataset"])):
        lines.append(
            f"| {x_of(group, r['value'])[1]} | {DATASET_LABEL[r['dataset']]} | {_f(r['HOTA'])} | {_f(r['MOTA'])} | "
            f"{_f(r['IDF1'])} | {r['IDs']} | {r['processed']}/{r['frames']} ({r['processed_pct']:.0f}%) | "
            f"{_f(r['replies_ok_pct'], 0) + '%' if r['replies_ok_pct'] else 'n/a (onboard)'} | "
            f"{_f(r['rtt_p50_ms'], 0) + ' ms' if r['rtt_p50_ms'] else '-'} | {_f(r['latency_p50_ms'], 0)} ms | "
            f"{_f(r['edge_cores_used'], 2)} |")
    return "\n".join(lines)


def chart(group: str, rows: list[dict], path: Path, extra: list[dict] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10})
    metrics = [m for m in METRICS if not (group == "onboard" and m[0] == "replies_ok_pct")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.4), facecolor=SURFACE)
    for ax, (key, label) in zip(axes, metrics):
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(AXIS)
        ax.grid(axis="y", color=GRID, linewidth=1, linestyle="-")
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelcolor=INK_2)
        ax.set_title(label, color=INK, fontsize=11, loc="left")
        series = [("split", rows)] + ([("onboard", extra)] if extra else [])
        for design, data in series:
            for dataset in ("uavdt", "visdrone"):
                pts = sorted((x_of(group, r["value"])[0], float(r[key])) for r in data
                             if r["dataset"] == dataset and r[key] not in ("", None))
                if not pts:
                    continue
                xs, ys = zip(*pts)
                style = "-" if design == "split" else (0, (1, 2))
                name = DATASET_LABEL[dataset] + ("" if not extra else f", {design}")
                ax.plot(xs, ys, linestyle=style, color=SERIES[dataset], linewidth=2, solid_capstyle="round",
                        marker="o" if design == "split" else "s", markersize=8,
                        markeredgecolor=SURFACE, markeredgewidth=2, label=name)
                ax.annotate(f"{ys[-1]:.0f}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", color=INK_2, fontsize=9)
        ticks = sorted({x_of(group, r["value"]) for r in rows + (extra or [])})
        ax.set_xticks([t[0] for t in ticks])
        ax.set_xticklabels([t[1] if group in ("hardware", "onboard") else f"{t[0]:g}" for t in ticks], color=INK_2)
        if key != "HOTA":
            ax.set_ylim(0, 105)
        ax.set_xlabel(GROUPS[group][0], color=INK_2)
    axes[0].legend(frameon=False, labelcolor=INK_2, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default=None, help="Output folder (default: beside results.csv, report/).")
    args = parser.parse_args(argv)
    results = PROJECT_ROOT / args.results
    out = Path(args.out) if args.out else results.parent / "report"
    out.mkdir(parents=True, exist_ok=True)
    rows = load(results)
    misses = sum(int(r["cache_misses"] or 0) for r in rows)

    md = ["# Edge/ground simulation sweep", "",
          f"Source: `{args.results}`. Real-time edge (camera 30 fps), streaming-perception scoring, modelled "
          f"ground station (cache misses across all runs: **{misses}**). Each group varies one factor from the base "
          "configuration: 4 E-core CPU edge, 10 ms +-2 ms each way, 50 Mbit/s uplink, no loss, 150 ms deadline.", ""]
    onboard = group_rows(rows, "onboard")
    for group in GROUPS:
        g = group_rows(rows, group)
        if not g or group == "onboard":
            continue
        png = out / f"{group}.png"
        chart(group, g, png, extra=onboard if group == "hardware" and onboard else None)
        md += [f"## {group.capitalize()}", "", f"![{group}]({png.name})", "", table(group, g), ""]
        if group == "hardware" and onboard:
            md += ["Onboard design (YOLO on the edge, no network):", "", table("onboard", onboard), ""]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Report: {out / 'report.md'} ({len(rows)} result rows, cache misses {misses})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
