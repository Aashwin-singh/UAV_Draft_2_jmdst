# Project Log

Running status for the JMDST reproduction, for teammates picking up work in
parallel. See `PROJECT_CONTEXT.md` for the full technical spec (Section A)
and detailed phase roadmap (Section B) — this file is just a quick "where
are we" snapshot, updated as phases complete.

## Status as of 2026-07-24

**Done and committed** (see `git log` for details):
- Phase 1: VisDrone2019-MOT + UAVDT converted to a unified JSONL format,
  with train/val/test splits matching the paper's counts (VisDrone 45/10/10,
  UAVDT 12/3/8) via a reproducible seed-42 split — see `configs/splits/`.
  Verified: 88 sequences, 42,474 frames, 929,343 annotations, 0 data issues.
- Phase 2 (in progress): unified-to-YOLO exporter (`jmdst/data/yolo_export.py`)
  and training script (`scripts/train_yolo.py`) built and smoke-tested.
  **Full YOLOv11n training (100 epochs) is being run now** — not yet
  committed/available as a checkpoint for other phases to consume.

**Not yet started**: FELNet architecture/training (Phase 3-4), feature
extraction (Phase 5), MSFP/Mamba (Phase 6, blocked on `mamba-ssm`
installing), tracking infrastructure (Phase 7), modified DeepSORT (Phase 8),
full pipeline integration (Phase 9), evaluation (Phase 10), ablations
(Phase 11).

## What a teammate can work on right now, in parallel with YOLO training

YOLO training holds the GPU for hours. Good parallel work is either
CPU-only, or independent of both the GPU and of YOLO's eventual output:

1. **FELNet architecture (Phase 3)** — `PROJECT_CONTEXT.md` Section A.3 has
   the exact channel/kernel/output-shape table. This is just defining
   `nn.Module` classes and unit-testing forward-pass output shapes on CPU
   with a tiny dummy batch — no GPU or trained-YOLO dependency at all.
   `jmdst/data/crops.py` already produces 64x64 SSIs and FELNet anchor
   targets (overlap vectors, confidence labels), so the *data* side of
   Phase 3/4 is ready; only the model + training loop are missing.

2. **Tracking infrastructure (Phase 7)** — Kalman filter (predict/update for
   bounding boxes) and the tracklet state machine (tentative -> confirmed ->
   deleted, `PROJECT_CONTEXT.md` Section A.6). Pure algorithmic Python, no
   ML training or GPU involved, fully testable with synthetic trajectories
   without waiting on YOLO or FELNet.

3. **Evaluation metrics scaffolding (Phase 10)** — `motmetrics` is already
   installed. A script that takes predicted + ground-truth trajectories in a
   standard format and computes MOTA/MOTP/IDF1/HOTA (formulas in
   `PROJECT_CONTEXT.md` A.9 / paper Sec 3.2) can be built and tested against
   synthetic/toy trajectories now, ready to point at real tracker output
   later.

Avoid starting Phase 4 (FELNet *training*, not just architecture) or
anything else that competes for the GPU until YOLO training finishes, to
avoid slowing both down / risking an out-of-memory error on the 8GB laptop
GPU.

## Setting up a fresh clone (if a teammate is on a different machine)

The raw datasets and all derived data (`data/unified/`, `data/yolo/`,
`Datasets/`, `outputs/`) are gitignored — only code is pushed. On a new
machine:

1. `conda env create -f environment.yml` (or see `requirements.txt`), then
   `conda activate jmdst`.
2. Get the raw VisDrone2019-MOT and UAVDT datasets locally under `Datasets/`
   (see `README.md` for expected folder layout).
3. Regenerate the unified format + splits:
   - `python scripts/make_splits.py ...` / `python scripts/make_visdrone_split.py ...`
     (see `configs/splits/` for the exact seed-42 split already decided on —
     reuse those `.txt` files rather than regenerating, so everyone trains
     on the same split).
   - `python scripts/prepare_dataset.py visdrone/uavdt ...` per
     `configs/splits/visdrone_conversion_plan.tsv` and the uavdt split files.
4. `python scripts/verify_converted_dataset.py --unified-root data/unified --skip-ssi`
   to confirm the data converted cleanly.
5. `python scripts/export_yolo_dataset.py --unified-root data/unified --output-root data/yolo`
6. Now `data/yolo/dataset.yaml` exists and `scripts/train_yolo.py` will work.
