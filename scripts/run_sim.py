"""Run one edge/ground simulation: ground node + link emulator + edge node.

Starts the ground node and waits until it is listening, starts the link
emulator with the given netem specs, runs the edge node to completion, then
stops the other two (they print their summaries on the way out). For the
snapshot of the deployment, run the three scripts in three terminals instead;
this launcher is for experiments and sweeps.

Usage (one line in cmd.exe; everything after the known options goes to the edge):
    python scripts/run_sim.py --yolo outputs/yolo_runs/retrain_aug_m_lowlr/weights/best.pt --felnet outputs/felnet_runs/full_run/best.pt --e2g "delay 40ms rate 20mbit loss 1%" --g2e "delay 40ms" --output-root outputs/sim/run1 --datasets uavdt --split val --conf 0.55 --no-appearance --nsa --cmc --byte --byte-low-conf 0.4

Writes under --output-root: the edge's MOT results/timing/exchanges, plus
ground.csv, link.csv, logs/{ground,link,edge}.log and run.json.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


class Child:
    """A node process whose output is teed to the console and a log file."""

    def __init__(self, name: str, cmd: list[str], log_path: Path, echo: bool = True) -> None:
        self.name = name
        self.ready = threading.Event()
        self._ready_marker: str | None = None
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_path, "w", encoding="utf-8")
        self._echo = echo
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=flags,
        )
        self._thread = threading.Thread(target=self._pump, daemon=True)

    def start(self, ready_marker: str | None = None) -> "Child":
        self._ready_marker = ready_marker
        self._thread.start()
        return self

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self._log.write(line)
            self._log.flush()
            if self._echo and not line.lstrip().startswith(("WARNING", "warnings.warn")):
                print(line, end="", flush=True)
            if self._ready_marker and self._ready_marker in line:
                self.ready.set()

    def wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while not self.ready.wait(0.2):
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} exited with code {self.proc.returncode} before it was ready")
            if time.monotonic() > deadline:
                raise TimeoutError(f"{self.name} not ready after {timeout:.0f} s")

    def stop(self, timeout: float = 15.0) -> None:
        if self.proc.poll() is None:
            if sys.platform == "win32":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._thread.join(5)
        self._log.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yolo", required=True)
    parser.add_argument("--felnet", required=True)
    parser.add_argument("--e2g", default="", help="Edge->ground netem spec ('' = perfect link).")
    parser.add_argument("--g2e", default="", help="Ground->edge netem spec.")
    parser.add_argument("--no-link", action="store_true", help="Edge talks to the ground directly (no emulator).")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--port-base", type=int, default=47000, help="Link listens on base, ground on base+1.")
    parser.add_argument("--ground-device", default=None)
    parser.add_argument("--host-cpus", default=None,
                        help="Pin ground + link to these CPUs (e.g. 0-15), apart from the edge's --edge-cpus.")
    parser.add_argument("--ground-print-every", type=int, default=0,
                        help="Ground prints a 'serving' line every N requests (0 = quiet).")
    args, edge_args = parser.parse_known_args(argv)

    out = PROJECT_ROOT / args.output_root
    out.mkdir(parents=True, exist_ok=True)
    link_addr, ground_addr = f"127.0.0.1:{args.port_base}", f"127.0.0.1:{args.port_base + 1}"
    (out / "run.json").write_text(json.dumps({
        "e2g": args.e2g, "g2e": args.g2e, "no_link": args.no_link, "edge_args": edge_args,
        "yolo": args.yolo, "felnet": args.felnet, "ground_device": args.ground_device, "host_cpus": args.host_cpus,
    }, indent=2), encoding="utf-8")

    py = [sys.executable, "-u"]
    ground_cmd = py + [str(SCRIPTS / "ground_node.py"), "--yolo", args.yolo, "--listen", ground_addr,
                       "--telemetry", str(out / "ground.csv"), "--print-every", str(args.ground_print_every)]
    if args.ground_device:
        ground_cmd += ["--device", args.ground_device]
    if args.host_cpus:
        ground_cmd += ["--cpus", args.host_cpus]
    children: list[Child] = []
    try:
        ground = Child("ground", ground_cmd, out / "logs" / "ground.log").start("[ground] listening")
        children.append(ground)
        ground.wait_ready(180)
        target = ground_addr
        if not args.no_link:
            link = Child("link", py + [
                str(SCRIPTS / "link_relay.py"), "--edge-listen", link_addr, "--ground", ground_addr,
                "--e2g", args.e2g, "--g2e", args.g2e, "--telemetry", str(out / "link.csv"), "--stats-every", "5",
                *(["--cpus", args.host_cpus] if args.host_cpus else []),
            ], out / "logs" / "link.log").start("[link] listening")
            children.append(link)
            link.wait_ready(60)
            target = link_addr
        edge = Child("edge", py + [
            str(SCRIPTS / "edge_node.py"), "--felnet", args.felnet, "--ground", target,
            "--output-root", str(out), *edge_args,
        ], out / "logs" / "edge.log").start()
        code = edge.proc.wait()
        edge.stop()
        return code
    finally:
        for child in reversed(children):
            child.stop()


if __name__ == "__main__":
    raise SystemExit(main())
