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
- **Phase 5 done**: feature extraction (`jmdst/data/features.py`,
  `scripts/extract_features.py`) runs FELNet over unified sequences and saves
  per-object embeddings grouped by track id — MSFP's (Phase 6) training
  input. Full extraction run completed and verified: 929,343 object
  embeddings (exact match to the Phase 1 annotation count — nothing lost or
  duplicated), 4,985 track sequences (exact match to unique-track-ID count),
  0 non-finite embeddings, embedding L2 norms deviate from 1.0 by 0.000000
  across the entire dataset. Track length distribution is healthy for MSFP:
  median 133 frames/track, only 0.9% single-frame fragments. Saved under
  `data/features/<dataset>/<split>/<sequence>.npz` (gitignored, regenerate
  with the command in Phase 5's original entry above if needed).

- **Phase 6 (MSFP) still blocked on `mamba-ssm`** — root cause now fully
  diagnosed (see below), decided not to sink more time into it right now.

  **mamba-ssm investigation writeup** (so nobody re-does this from scratch):
  CUDA Toolkit 12.8 + VS Build Tools (C++ workload) got installed and
  verified (`nvcc --version` confirms 12.8.61, matching torch's cu128 build).
  Getting `pip install mamba-ssm --no-build-isolation` to even reach the
  compile step needed two extra fixes, both real and necessary but not
  sufficient: (1) `cl.exe` isn't on PATH by default — MSVC needs
  `vcvarsall.bat x64` run in the *same* process as the pip install (its env
  vars don't persist across separate shell invocations, so `conda activate`
  + `vcvarsall.bat` + `pip install` all need to happen in one script block);
  (2) once the VC env is manually activated this way, PyTorch's build
  requires `DISTUTILS_USE_SDK=1` to be set too, or it raises a clear error
  telling you so. With both fixed, the build reached actual compilation —
  and then failed because **the PyPI `mamba-ssm` sdist (checked 2.2.4, and
  by extension likely all versions) is missing its own `csrc/` directory**.
  `setup.py` explicitly lists `csrc/selective_scan/selective_scan.cpp` and
  9 `.cu` kernel files as required sources; none of them exist anywhere in
  the downloaded/extracted package (confirmed: `grep csrc setup.py` lists
  them, `grep csrc mamba_ssm.egg-info/SOURCES.txt` returns nothing). This
  isn't a Windows-specific bug — the source genuinely isn't in the sdist,
  so building from PyPI's source package can't work on any OS. mamba-ssm
  publishes prebuilt Linux wheels via GitHub Releases instead, which is
  presumably why this gap in the sdist has gone unnoticed.

  **Real paths forward, not yet tried**: (a) clone the official
  `state-spaces/mamba` GitHub repo directly (has the complete source
  including `csrc/`) and build from that checkout instead of PyPI — untested
  whether other Windows-specific compile issues surface once real source is
  present; (b) use WSL2, which sidesteps this whole class of Windows-build
  issues and can use the official prebuilt approach. Either is a genuinely
  open-ended time investment (real CUDA kernel compilation is commonly
  10-30+ min even when everything's configured correctly), so it's deferred
  rather than attempted unprompted — whoever picks this up next should treat
  it as its own task, not a quick add-on.

**Not yet started**: MSFP/Mamba (Phase 6, see above), tracking
infrastructure (Phase 7 — starting now), modified DeepSORT (Phase 8), full
pipeline integration (Phase 9), evaluation (Phase 10), ablations (Phase 11).

## What a teammate can work on right now, in parallel

Phase 7 (tracking infrastructure — Kalman filter, tentative/confirmed/deleted
state machine, `PROJECT_CONTEXT.md` Section A.6) is being started in this
session — check `git log` before picking it up yourself to avoid duplicate
work.

**Evaluation metrics scaffolding (Phase 10)** is still open and good parallel
work: `motmetrics` is already installed. A script that takes predicted +
ground-truth trajectories in a standard format and computes
MOTA/MOTP/IDF1/HOTA (formulas in `PROJECT_CONTEXT.md` A.9 / paper Sec 3.2)
can be built and tested against synthetic/toy trajectories now, ready to
point at real tracker output later. Pure CPU work, no GPU/ML dependency.

If anyone wants to tackle the mamba-ssm build (see the writeup above)
instead, that's also independent of everything else — just don't run a long
GPU training job at the same time as someone else on the same machine.

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
