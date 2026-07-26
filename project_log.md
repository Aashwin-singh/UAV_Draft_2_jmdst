# Project Log

Running status for the JMDST reproduction, for teammates picking up work in
parallel. See `PROJECT_CONTEXT.md` for the full technical spec (Section A)
and detailed phase roadmap (Section B) — this file is just a quick "where
are we" snapshot, updated as phases complete.

## Accuracy-improvement work (2026-07-26, post-Phase-10)

Focused on raising results before the Phase 11 ablations. Findings, in order:

1. **MSFP is NOT the bottleneck** (answered "how much does MSFP matter"):
   paper Table 2 shows MSFP adds only +0.4 MOTA / +0.6 IDF1 / -7% IDs. Our
   gap to the paper is ~50 MOTA points, so unblocking mamba-ssm for MSFP is
   not worth it now. FELNet's feature encoding (already implemented) is the
   bigger association win (IDs 541->361 vs standard ReID in the same table).
2. **UAVDT ignore regions — DONE, biggest feasible eval win**: UAVDT val
   MOTA 3.0 -> 17.0 (IDF1 43.6->46.6, HOTA 34.7->36.6). This is the correct
   UAVDT protocol, not a hack. M0401 alone went 9.8 -> 34.9. Committed.
3. **Residual gap is the DETECTOR, not tracker/MSFP**: diagnosed M1101 (our
   worst val seq) — 44% of the detector's raw detections there are true FP
   (hallucinated/unlabeled cars outside ignore regions). Root cause is the
   Phase-2 YOLO overfitting (val mAP peaked ~epoch 2) on out-of-distribution
   scenes. Matches paper Sec 3.4.4 (detector choice = ~8.6% MOTA).
   - Note M0203 val already gets MOTA 61.3 — the full system tracks well when
     the detector is reliable. The average is dragged by M1101/M0401.
4. **Detection-conf sensitivity** (quick tuning check, UAVDT val, with ignore
   regions): conf 0.55->0.65 gives MOTA 17->21 (trades FP for FN) but flat/
   slightly-worse IDF1/HOTA; 0.75 is worse overall. Kept the paper's 0.55 as
   the faithful default (didn't change it) — 0.65 is a documented
   detector-compensating option to validate on TEST if pursued, not a fix.

**Highest-value next step = a better-generalizing detector** (retrain YOLO).
Everything else (tracker tuning, MSFP) is small by comparison. See git log
`bceba3b` for the ignore-region commit and full diagnostics.

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

- **Phase 7 done**: tracking infrastructure in `jmdst/tracking/` —
  `kalman_filter.py` (standard DeepSORT 8-D constant-velocity Kalman filter
  over `[cx, cy, aspect_ratio, height]`; the paper says "modified DeepSORT"
  but doesn't specify Kalman internals, so this is the faithful standard
  baseline) and `track.py` (the `Track` class implementing A.6's
  tentative/confirmed/deleted state machine exactly: N=tau+1=4 consecutive
  detections to confirm, 100-miss deletion for confirmed tracks, immediate
  deletion for unconfirmed ones). 15 new tests (42 total), all passing —
  Kalman numerical correctness, bbox<->state round trips, and every
  state-machine rule from A.6. Scope note: this is just the building blocks;
  cascade/IoU matching, missed-detection expansion, and output filtering are
  Phase 8 (modified DeepSORT), not yet started.

- **Phase 8 done**: modified-DeepSORT association in `jmdst/tracking/`.
  `matching.py` (IoU/cosine cost matrices, Mahalanobis gating, Hungarian
  min-cost matching, DeepSORT age cascade) + `tracker.py` (the `Tracker`:
  detection-frame association per A.6 steps 1-7, plus `expansion_targets` and
  `filter_tracking_outputs` for the tracking-branch rules). 25 new tests (67
  total). End-to-end check on a real UAVDT sequence (GT boxes + real FELNet
  embeddings, per-frame detection): 14/14 GT tracks kept a single stable ID,
  0 ID switches over 200 frames.
  - **max_cosine_distance** defaults to 0.5 (paper doesn't specify it),
    calibrated to FELNet's Phase-4 embedding distribution (same-id cosine
    distance ~0.24, diff-id ~0.89). Tune in Phase 10 if needed.
  - **Known behavior, faithful to paper**: appearance matching is
    Mahalanobis-gated, so it disambiguates nearby targets but does NOT rescue
    large position jumps (a 60px jump for a 20px box is gated out). This is
    the paper's own documented failure mode (Sec. 3.5.4). If Phase 9/10 shows
    too many ID switches from UAV/camera motion, loosening the gate or the
    Kalman `std_weight_position` is the first knob.
  - **Phase 9 wiring insight** (important): a tracklet confirms only after N=4
    *consecutive frames present*, counting BOTH detection-frame updates AND
    tracking-branch updates. So the Phase 9 tracking branch MUST call
    `track.update(kf, localized_box, embedding)` on every non-detection frame
    for the targets it localizes — otherwise tentative tracks get
    `mark_missed()` and are deleted before they can ever confirm (verified
    this failure mode directly). The detection branch calls `tracker.update`;
    the tracking branch calls `track.update` per tracked target, then the
    survivors/filtered outputs update trajectories.

- **Phase 9 done**: full JMDST integration in `jmdst/pipeline/`.
  `jmdst.py` (`JMDSTTracker`: Algorithm 1's dual-branch routine — detection
  branch every tau frames via YOLO+FELNet+Tracker, tracking branch on other
  frames via Kalman-predict -> SSI crop -> FELNet localize/refine -> update,
  with the base/expansion split and c1/i1/i2 output filtering). `models.py`
  (`YoloDetector`, `FELNetLocalizer`, `build_jmdst` factory). Branch logic
  is model-agnostic (injected detector/localizer callables) so it's unit-
  tested with mocks. `scripts/run_jmdst.py` writes MOT-format results (Phase
  10 input). 5 new tests (72 total). Real end-to-end run on 3 UAVDT val
  sequences: ~30-36 FPS (exceeds the paper's 26.6 on UAVDT), 0 duplicate IDs
  per frame, clean stable-ID output, 0 invalid boxes.
  - Two robustness fixes found only via real-data runs (mocks can't surface
    them): FELNet overlap->box decode can be degenerate on off-target crops
    -> `FELNetLocalizer` falls back to the input box; the Kalman box can be
    transiently degenerate for tiny shrinking targets (position stays valid,
    height/aspect overshoots for one frame) -> `_confirmed_outputs` skips
    that frame's output, track survives internally.
  - **Command to generate full MOT results** (for Phase 10 eval):
    `python scripts/run_jmdst.py --yolo outputs/yolo_runs/full_run/weights/best.pt --felnet outputs/felnet_runs/full_run/best.pt --datasets uavdt visdrone --split val --tau 3`
    (drop `--max-frames`; ~30 FPS so a full split is minutes, not hours).

- **Phase 10 done**: evaluation in `jmdst/eval/` (+ `scripts/evaluate.py`).
  `metrics.py` (motmetrics wrapper: MOTA/MOTP/IDF1/IDs/FP/FN, with a
  `np.asfarray` shim for NumPy 2.0 and MOTP converted to the paper's mean-IoU
  convention), `hota.py` (HOTA/DetA/AssA implemented from scratch — no
  available lib has it; unit-tested against known cases), `io.py` (loads
  unified GT + MOT-format predictions). 9 new tests (81 total).
  - **First real numbers, UAVDT val (tau=3, 41.5 FPS inference)**: overall
    MOTA 3.0, MOTP 77.2, IDF1 43.6, HOTA 34.7 (DetA 31.8, AssA 38.0). These
    are LOW and the causes are diagnosed (NOT a pipeline bug — the M0203
    result proves the integration is sound):
    - **M0203**: GT 5.4 vs pipeline 5.5 boxes/frame -> MOTA **59.4**, MOTP
      79.2. When detector output and annotation density align, the full
      system tracks well.
    - **M1101**: GT 8.8 vs pipeline 17.2 -> FP-dominated (9119 FP, MOTA
      -46.2). YOLO detects ~15 real cars/frame but UAVDT labels only ~8.8
      (UAVDT's sparse annotation / ignore-region protocol, which our
      simplified eval does NOT replicate — those extra detections are real
      cars scored as FP). Confirmed YOLO fires only "car" here (no
      class-mismatch FP).
    - **M0401**: GT 16.2 vs pipeline 12.3 -> FN-dominated (5686 FN, MOTA
      9.8). Dense scene; the known Phase-2 YOLO overfitting hurts recall.
  - **Improvement levers** (for whoever pushes accuracy, in rough priority):
    (1) UAVDT ignore-region handling in eval + conversion (biggest UAVDT
    win); (2) a better-generalizing YOLO (retrain with early stopping / more
    augmentation / per-dataset detectors — the Phase 2 overfitting is the
    root detector weakness); (3) tracker FP tuning (lower `max_age`, revisit
    the expansion set). Don't over-tune to val — keep test held out for the
    final Phase 11 comparison.
  - **Command to evaluate** (after generating predictions with run_jmdst):
    `python scripts/evaluate.py --results outputs/jmdst_results/uavdt/val --dataset uavdt --split val --report outputs/jmdst_results/uavdt_val_report.md`

**Not yet started**: MSFP/Mamba (Phase 6, see above), ablations (Phase 11 —
the paper's ablation study: without-MSFP, varying tau, YOLO-only, standard
DeepSORT ReID vs FELNet; mirror paper Sec. 3.4).

## What a teammate can work on right now, in parallel

**Evaluation metrics scaffolding (Phase 10)** is open and good parallel work:
`motmetrics` is already installed. A script that takes predicted +
ground-truth trajectories in a standard format and computes
MOTA/MOTP/IDF1/HOTA (formulas in `PROJECT_CONTEXT.md` A.9 / paper Sec 3.2)
can be built and tested against synthetic/toy trajectories now, ready to
point at real tracker output later. Pure CPU work, no GPU/ML dependency.

If anyone wants to tackle the mamba-ssm build (see the writeup above)
instead, that's also independent of everything else — just don't run a long
GPU training job at the same time as someone else on the same machine.

Phase 11 (ablations) is the natural next step — mirror the paper's Sec. 3.4
ablation study (without-MSFP, varying tau via `--tau`, YOLO-only baseline,
standard-DeepSORT-ReID vs FELNet embeddings). Note MSFP-related ablations are
blocked with Phase 6 on mamba-ssm. Also open: the Phase 10 improvement levers
above (UAVDT ignore regions, better detector) if the goal is to close the gap
to the paper's numbers. Check `git log`
before starting it to
avoid duplicate work if picked up in parallel.

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
