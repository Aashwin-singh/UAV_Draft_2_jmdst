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
