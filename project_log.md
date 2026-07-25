# Project Log

Running status for the JMDST reproduction, for teammates picking up work in
parallel. See `PROJECT_CONTEXT.md` for the full technical spec (Section A)
and detailed phase roadmap (Section B) — this file is just a quick "where
are we" snapshot, updated as phases complete.

## Status as of 2026-07-25

**Done and committed** (see `git log` for details):
- Phase 1: VisDrone2019-MOT + UAVDT converted to a unified JSONL format,
  with train/val/test splits matching the paper's counts (VisDrone 45/10/10,
  UAVDT 12/3/8) via a reproducible seed-42 split — see `configs/splits/`.
  Verified: 88 sequences, 42,474 frames, 929,343 annotations, 0 data issues.
- **Phase 2 done**: unified-to-YOLO exporter (`jmdst/data/yolo_export.py`),
  training script (`scripts/train_yolo.py`), and inference script
  (`scripts/infer_yolo.py`, applies the paper's NMS IoU 0.2 / conf 0.55 post-
  processing per Sec. 3.3). YOLOv11n trained 100 epochs on the full train
  set; checkpoint is at `outputs/yolo_runs/full_run/weights/best.pt` (not
  committed — gitignored, ~5MB, ask whoever ran it for a copy, or retrain
  with the command below). Validated: mAP50 0.45 / mAP50-95 0.27 overall;
  per-class mAP50 car 0.76, truck 0.40, bus 0.38, van 0.27 (van/bus are the
  weak spots — expected, they're a small fraction of instances vs. car).
  **Known issue**: validation mAP peaked at epoch ~2 and declined for the
  rest of the 100-epoch run (training loss kept dropping = overfitting).
  `best.pt` (not `last.pt`) correctly holds the early, better-generalizing
  weights, so this doesn't block using it, but a from-scratch retrain should
  use a low `--patience` (e.g. `--patience 10`) to stop automatically instead
  of burning the full epoch budget:
  `python scripts/train_yolo.py --data data/yolo/dataset.yaml --model yolo11n.pt --epochs 100 --patience 10 --name retrain_early_stop`
- **Phase 3 done**: FELNet architecture in `jmdst/models/felnet.py` (paper
  Table 1 exactly — Darknet-derived backbone + 3 conv heads for overlap /
  16-D embedding / confidence), plus `decode_boxes` (Eq. 2) and
  `select_anchor_output` (Sec. 2.2 selection rule). 20 unit tests pass;
  4.35M params, ~0.14 ms/SSI @ batch 32 on the 4070. Architecture only, no
  training yet.

- **Phase 4 done**: FELNet training pipeline (episode sampler in
  `jmdst/data/felnet_episode.py` per paper Algorithm 2, three RMSE losses in
  `jmdst/training/felnet_loss.py`, `scripts/train_felnet.py`). Full 50-epoch
  run completed: `outputs/felnet_runs/full_run/best.pt` (not committed —
  gitignored). Held-out (val-split) evaluation via `scripts/eval_felnet.py`:
  embedding branch generalizes well (same-identity vs different-identity
  cosine similarity separation 0.65 — the critical signal for Phase 8
  matching); overlap/localization branch has a real train/val gap (mean IoU
  ~0.65 at the paper's confidence-based anchor selection, some overfitting).
  Decided to proceed rather than retrain now — revisit localization tuning
  later if Phase 7-9 tracking accuracy suffers.
  **Bug found and fixed** in `select_anchor_output` (the paper's Sec 2.2
  inference-time anchor-selection rule, in `jmdst/models/felnet.py`): it was
  comparing all 4 anchors against one shared reference overlap vector, which
  is dimensionally wrong (a full-crop-relative overlap isn't comparable to a
  32x32-anchor-relative one). Fixed to compute a correct per-anchor reference
  via the same formula training's ground-truth targets use
  (`anchor_reference_overlaps`). This only affects inference-time selection
  (Phase 7+), not the already-completed training run.
- **Phase 5 done (pipeline)**: feature extraction (`jmdst/data/features.py`,
  `scripts/extract_features.py`) runs FELNet over unified sequences and saves
  per-object embeddings grouped by track id — MSFP's (Phase 6) training
  input. Verified on a real sequence (5,404 objects in 12.7s on GPU).
  **The full extraction run itself has not been done yet**:
  `python scripts/extract_features.py --checkpoint outputs/felnet_runs/full_run/best.pt --unified-root data/unified --output-root data/features --splits train val test`
  (~35-40 min estimated, based on the measured per-object rate over all
  929,343 annotations).

**Not yet started**: run the feature extraction (Phase 5 execution),
MSFP/Mamba (Phase 6, blocked on `mamba-ssm` installing), tracking
infrastructure (Phase 7), modified DeepSORT (Phase 8), full pipeline
integration (Phase 9), evaluation (Phase 10), ablations (Phase 11).

## What a teammate can work on right now, in parallel

Good parallel work is either CPU-only, or independent of whatever's
currently using the GPU:

1. **Tracking infrastructure (Phase 7)** — Kalman filter (predict/update for
   bounding boxes) and the tracklet state machine (tentative -> confirmed ->
   deleted, `PROJECT_CONTEXT.md` Section A.6). Pure algorithmic Python, no
   ML training or GPU involved, fully testable with synthetic trajectories
   without waiting on YOLO or FELNet.

2. **Evaluation metrics scaffolding (Phase 10)** — `motmetrics` is already
   installed. A script that takes predicted + ground-truth trajectories in a
   standard format and computes MOTA/MOTP/IDF1/HOTA (formulas in
   `PROJECT_CONTEXT.md` A.9 / paper Sec 3.2) can be built and tested against
   synthetic/toy trajectories now, ready to point at real tracker output
   later.

Phase 6 (MSFP) needs Phase 5's extracted features to exist first (run the
extraction command above). Avoid running anything large on the GPU at the
same time as anyone else, to avoid slowing both down / an out-of-memory
error on the 8GB laptop GPU.

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
