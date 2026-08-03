"""Build the self-contained HTML report from the template + generated figures.

Inlines every figure as a base64 data URI so the page has no external
dependencies (required by the Artifact CSP, and makes the file portable).

Usage:
    python scripts/make_presentation_figures.py --output-dir outputs/presentation
    python scripts/build_report.py
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
from pathlib import Path

from PIL import Image


def encode(path: Path, max_width: int, quality: int) -> str:
    """Downscale if needed and return a base64 data URI."""

    img = Image.open(path)
    if img.width > max_width:
        img = img.resize((max_width, int(img.height * max_width / img.width)), Image.LANCZOS)

    suffix = path.suffix.lower()
    tmp = path.with_name(f"_enc{path.stem}{suffix}")
    if suffix in {".jpg", ".jpeg"}:
        img.convert("RGB").save(tmp, quality=quality, optimize=True)
    else:
        img.save(tmp, optimize=True)

    data = base64.b64encode(tmp.read_bytes()).decode("ascii")
    tmp.unlink()
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{data}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figures", default="outputs/presentation")
    p.add_argument("--template", default="scripts/report_template.html")
    p.add_argument("--output", default="outputs/presentation/report.html")
    p.add_argument("--max-width", type=int, default=1000)
    p.add_argument("--quality", type=int, default=85)
    args = p.parse_args()

    figures = Path(args.figures)
    html = Path(args.template).read_text(encoding="utf-8")

    total = 0
    for path in sorted(figures.glob("fig*")):
        token = "{{" + path.stem.split("_")[0].upper() + "}}"
        if token not in html:
            continue
        uri = encode(path, args.max_width, args.quality)
        total += len(uri)
        html = html.replace(token, uri)
        print(f"  embedded {path.name} -> {token} ({len(uri)/1024:.0f} KB)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\nWrote {out} ({out.stat().st_size/1024:.0f} KB, {total/1024:.0f} KB of it images)")

    missing = [t for t in ("{{FIG1}}", "{{FIG9}}") if t in html]
    if missing:
        print(f"WARNING: unreplaced tokens remain: {missing}")


if __name__ == "__main__":
    main()
