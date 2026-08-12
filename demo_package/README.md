# JMDST live demo package

Self-contained. Needs no datasets, no training, no network.

## Contents

| Path | What |
|---|---|
| `clips/*.mp4` | Raw UAV footage to run inference on live |
| `prerendered/*_tracked.mp4` | Same clips already tracked — the no-risk fallback |
| `checkpoints/` | The two model files (copy them in — see below) |

## One-time setup on the presentation machine

```powershell
conda env create -f environment.yml
conda activate jmdst
```

Then copy the two checkpoints in — they are gitignored and cannot be
regenerated without hours of GPU training:

| From | To |
|---|---|
| `outputs/yolo_runs/retrain_aug_m_lowlr/weights/best.pt` (39 MB) | `demo_package/checkpoints/yolo.pt` |
| `outputs/felnet_runs/full_run/best.pt` (17 MB) | `demo_package/checkpoints/felnet.pt` |

**CUDA matters.** PyPI's default `torch` is a CPU build and the demo is
unusable on it. `environment.yml` and `requirements.txt` both carry the
`--extra-index-url` that forces the GPU build. Confirm with preflight below.

## Before you present

```powershell
python scripts/preflight.py --source demo_package/clips/uavdt_M0101.mp4
```

Exits 0 only when nothing is blocking. It checks Python, the CUDA build, GPU
memory, both checkpoints, that the footage actually decodes, mains power
(a laptop on battery throttles the GPU hard), and runs 8 real frames
end-to-end.

## Running it

```powershell
python scripts/demo.py --source demo_package/clips/uavdt_M0101.mp4 --show
```

Press `q` to stop. Swap the clip for either VisDrone file to show it
generalising to a second dataset with no setting changed.

Measured end-to-end on an RTX 4070 Laptop:

| Clip | Resolution | Frames | Playback | Runtime |
|---|---|---:|---:|---:|
| `uavdt_M0101.mp4` | 1024x540 | 400 | 31 FPS | ~13 s |
| `visdrone_uav0000306.mp4` | 680x382 | 420 | **54 FPS** | ~8 s |
| `visdrone_uav0000308.mp4` | 1280x720 | 230 | 27 FPS | ~8 s |

Open with **uavdt_M0101** — cleanest tracking. Follow with a VisDrone clip to
show one configuration handling both datasets.

## Two things to say

**Point at the HUD's second line.** It alternates `DETECT (YOLO+FELNet)` and
`TRACK (FELNet only)`. That is the paper's core idea visible in real time: full
detection runs on only 1 frame in 3, and a cheap single-target branch covers
the rest. It is the best explanatory moment in the demo.

**No boxes appear for the first ~4 frames.** A tracklet is emitted only once
confirmed, which needs N = tau+1 = 4 consecutive present frames. That is the
paper's rule (A.6), not a defect — but it looks like one if you are not ready
to say so.

## If the GPU misbehaves

Play `prerendered/*_tracked.mp4` in any media player and keep talking. No
debugging in front of an audience.

## Configuration on display

YOLOv11m (combined-dataset training, lr0 0.0025) + FELNet + IoU-only
association + BYTE two-stage matching + NSA Kalman + ECC camera-motion
compensation, conf 0.55, tau 3. One configuration, both datasets. Full results
in `RESULTS_SUMMARY.md`.

`--appearance`, `--no-nsa`, `--no-cmc` switch the ablations back on if someone
asks to see a negative result live.
