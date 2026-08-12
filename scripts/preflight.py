"""Pre-demo readiness check. Run this before presenting, on the machine you will present from.

Every check corresponds to something that has actually gone wrong or would
plausibly go wrong in a live setting:

  * a CPU-only torch (PyPI's default) drops the demo far below real time
  * checkpoints are gitignored, so a fresh clone has none
  * a laptop on battery throttles the GPU hard
  * the first inference on a cold CUDA context takes ~2.2 s

Exit code is 0 only when nothing is blocking, so it can gate a launcher script.

    python scripts/preflight.py
    python scripts/preflight.py --source data/unified/uavdt/test/M0101
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_YOLO = "outputs/yolo_runs/retrain_aug_m_lowlr/weights/best.pt"
DEFAULT_FELNET = "outputs/felnet_runs/full_run/best.pt"
DEFAULT_SOURCE = "data/unified/uavdt/test/M0101"

OK, WARN, FAIL = "  OK  ", " WARN ", " FAIL "
_results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    _results.append((status, name, detail))
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""), flush=True)


def check_python() -> None:
    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro}"
    record(OK if (v.major, v.minor) == (3, 11) else WARN, "Python 3.11", detail)


def check_torch_cuda() -> None:
    try:
        import torch
    except ImportError as exc:
        record(FAIL, "PyTorch importable", str(exc))
        return

    record(OK, "PyTorch", torch.__version__)
    if not torch.cuda.is_available():
        record(FAIL, "CUDA available",
               "CPU-only build or no driver - the demo will be far below real time. "
               "Reinstall per requirements.txt (note the --extra-index-url).")
        return
    record(OK, "CUDA available", f"{torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}")

    free, total = torch.cuda.mem_get_info()
    gb = 1024 ** 3
    used = (total - free) / gb
    detail = f"{used:.1f} GB in use of {total / gb:.1f} GB"
    # Inference needs ~2 GB; flag if something else has eaten the card.
    record(OK if free / gb > 2.5 else WARN, "GPU memory free", detail)


def check_packages() -> None:
    import importlib
    missing = []
    for mod in ("cv2", "ultralytics", "numpy", "scipy", "PIL", "motmetrics"):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    record(OK if not missing else FAIL, "Required packages",
           "all present" if not missing else f"missing: {', '.join(missing)}")


def check_checkpoints(yolo: str, felnet: str) -> bool:
    ok = True
    for label, path in (("YOLO weights", yolo), ("FELNet checkpoint", felnet)):
        p = Path(path)
        if p.exists():
            record(OK, label, f"{p} ({p.stat().st_size / 1024 ** 2:.0f} MB)")
        else:
            ok = False
            record(FAIL, label, f"missing: {p} - gitignored, copy it in or retrain")
    return ok


def check_source(source: str) -> bool:
    """Verify the source yields a readable frame, not merely that a path exists.

    A unified sequence directory contains no images at all -- it stores
    ABSOLUTE paths into Datasets/. Such a source resolves only on the machine
    that built the dataset, so existence proves nothing about portability.
    Export a self-contained clip (scripts/export_demo_clip.py) to avoid this.
    """

    p = Path(source)
    if not p.exists():
        record(WARN, "Demo footage", f"not found: {p} (pass --source, or use a video file)")
        return False

    portable = p.is_file() or not (p / "seqinfo.json").exists()
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from demo import open_source

        frames, _fps, total = open_source(p, 1, 30.0)
        frame = next(iter(frames), None)
        if frame is None:
            record(FAIL, "Demo footage", f"{p} yielded no frames")
            return False
        h, w = frame.shape[:2]
        detail = f"{p} ({total or '?'} frames, {w}x{h})"
        if not portable:
            detail += " [depends on Datasets/ via absolute paths - not portable]"
        record(OK if portable else WARN, "Demo footage", detail)
        return True
    except Exception as exc:
        record(FAIL, "Demo footage", f"{p}: {type(exc).__name__}: {exc}")
        return False


def check_power() -> None:
    """A laptop on battery throttles the GPU enough to spoil a live demo."""

    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Battery).BatteryStatus"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        # BatteryStatus 2 == AC connected. Empty output == no battery (desktop).
        if not out:
            record(OK, "Power", "no battery detected (desktop)")
        elif "2" in out.split():
            record(OK, "Power", "on AC")
        else:
            record(WARN, "Power", "on battery - plug in, the GPU throttles hard")
    except Exception:
        record(WARN, "Power", "could not determine (check you are plugged in)")


def check_disk() -> None:
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1024 ** 3
    record(OK if free_gb > 2 else WARN, "Disk space", f"{free_gb:.1f} GB free")


def smoke_test(yolo: str, felnet: str, source: str) -> None:
    """Actually run a few frames - the only check that proves the whole path."""

    try:
        import cv2
        from PIL import Image

        from jmdst.pipeline.models import build_jmdst
        from jmdst.tracking import CameraMotionCompensator
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from demo import open_source, warm_up

        frames, _fps, _total = open_source(Path(source), 8, 30.0)
        tracker = build_jmdst(
            yolo_weights=yolo, felnet_checkpoint=felnet, tau=3, conf=0.55,
            use_appearance=False, use_nsa=True, cmc=CameraMotionCompensator(),
            use_byte=True, byte_conf_high=0.55,
        )
        outputs = None
        for i, frame_bgr in enumerate(frames):
            if i == 0:
                warm_up(tracker, frame_bgr, 2)
            h, w = frame_bgr.shape[:2]
            image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            outputs = tracker.process_frame(image, (w, h))
        record(OK, "End-to-end smoke test", f"8 frames processed, {len(outputs or [])} tracks on the last")
    except Exception as exc:
        record(FAIL, "End-to-end smoke test", f"{type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yolo", default=DEFAULT_YOLO)
    parser.add_argument("--felnet", default=DEFAULT_FELNET)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the end-to-end run.")
    args = parser.parse_args()

    print("JMDST demo preflight\n" + "=" * 60)
    check_python()
    check_torch_cuda()
    check_packages()
    have_ckpt = check_checkpoints(args.yolo, args.felnet)
    have_src = check_source(args.source)
    check_power()
    check_disk()
    if not args.skip_smoke and have_ckpt and have_src:
        smoke_test(args.yolo, args.felnet, args.source)

    fails = [r for r in _results if r[0] == FAIL]
    warns = [r for r in _results if r[0] == WARN]
    print("=" * 60)
    if fails:
        print(f"NOT READY - {len(fails)} blocking issue(s):")
        for _s, name, detail in fails:
            print(f"  - {name}: {detail}")
        sys.exit(1)
    print(f"READY{f' ({len(warns)} warning(s) - review above)' if warns else ''}")
    print("\n  python scripts/demo.py --source <video|frames|seq-dir> --show")
    sys.exit(0)


if __name__ == "__main__":
    main()
