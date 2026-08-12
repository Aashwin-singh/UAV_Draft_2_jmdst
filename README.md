# JMDST Reproduction

This repository is being built step by step to reproduce JMDST:
Joint Multi-target Detection and Single-target Tracking for UAV vehicle tracking.

## Dataset Preparation

The unified annotation format is stored as:

```text
data/unified/
  visdrone/train/<sequence>/
    seqinfo.json
    annotations.jsonl
  uavdt/train/<sequence>/
    seqinfo.json
    annotations.jsonl
```

Each `annotations.jsonl` line contains one frame:

```json
{"frame_id":1,"image_path":"...","objects":[{"track_id":1,"class_id":0,"class_name":"car","bbox_xywh":[10,20,30,40]}]}
```

Convert VisDrone:

```powershell
python scripts/prepare_dataset.py visdrone `
  --source-root C:\path\to\VisDrone `
  --output-root data/unified `
  --split train
```

Convert UAVDT:

```powershell
python scripts/prepare_dataset.py uavdt `
  --source-root C:\path\to\UAVDT `
  --output-root data/unified `
  --split train
```

By default, UAVDT conversion follows the paper's car-only setting. Add
`--all-vehicle-classes` to keep truck and bus labels when present.

Render annotation and SSI crop checks:

```powershell
python scripts/visualize_dataset_samples.py `
  --unified-root data/unified `
  --output-dir outputs/verification_samples `
  --dataset visdrone `
  --split train `
  --num-samples 12
```

Run the full converted-dataset verifier:

```powershell
python scripts/verify_converted_dataset.py `
  --unified-root data/unified `
  --report-path outputs/dataset_verification_report.md `
  --json-path outputs/dataset_verification_report.json
```

## Live demo

`scripts/demo.py` runs the tracker on ordinary video and renders the result —
coloured boxes with persistent track IDs, plus an overlay showing which branch
is active on each frame (the paper's core idea made visible: full detection
only every τ-th frame).

Write an annotated video:

```powershell
python scripts/demo.py --source clip.mp4 --output outputs/demo/clip_tracked.mp4
```

Live window (press `q` to quit):

```powershell
python scripts/demo.py --source data/unified/uavdt/test/M0101 --show
```

`--source` accepts a video file, a directory of frames, or a unified sequence
directory. Defaults are the project's final configuration (see
`RESULTS_SUMMARY.md`); `--appearance`, `--no-nsa`, `--no-cmc` switch the
ablations back on/off.

Requires the trained checkpoints, which are gitignored:
`outputs/yolo_runs/retrain_aug_m_lowlr/weights/best.pt` (~40 MB) and
`outputs/felnet_runs/full_run/best.pt` (~17 MB). The script fails with a clear
message if they are missing. **Datasets are not needed** — only the two
checkpoints and some footage.

Expect no boxes for the first few frames of any clip: a tracklet is only
emitted once confirmed, which needs N = τ+1 = 4 consecutive present frames.
That is the paper's rule (A.6), not a defect.

### Presenting it

`demo_package/` is the portable bundle — see `demo_package/README.md`. Check
readiness first:

```powershell
python scripts/preflight.py --source demo_package/clips/uavdt_M0101.mp4
```

**Do not demo from `data/unified/...`.** Those directories hold annotations
only, and the image paths inside them are *absolute*, so they resolve on the
machine that built the dataset and nowhere else. Export a self-contained clip
instead:

```powershell
python scripts/export_demo_clip.py --source data/unified/uavdt/test/M0101 --output demo_package/clips/uavdt_M0101.mp4 --max-frames 400
```

Clips also run faster than the unified directories (mp4 decode beats per-frame
JPEG): 31–54 FPS end-to-end versus ~19.

## Implemented So Far

- VisDrone2019-MOT converter.
- UAVDT converter.
- Unified sequence metadata and frame annotation schema.
- Paper SSI crop formula resized to 64x64.
- FELNet 2x2 anchor overlap/confidence target generation.
- PyTorch-ready detection, FELNet SSI, and tracking sequence datasets.
- Synthetic dataset generator and visual verification script.

## Verification

Run the local tests:

```powershell
python -m unittest discover -s tests -v
```

If PyTorch is installed, `jmdst.data.datasets.build_dataloader` creates real
`torch.utils.data.DataLoader` instances. Without PyTorch, dataset indexing still
works for converter/crop verification and returns NumPy arrays.
