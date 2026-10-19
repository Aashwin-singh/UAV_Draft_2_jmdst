# Architecture & handoff guide

Code map and hard-won invariants for making **architecture-level changes** to
this JMDST reproduction. Read alongside:

- `PROJECT_CONTEXT.md` — **Section A is the paper spec and is ground truth.**
  Implement against it rather than re-deriving from the PDF.
- `project_log.md` — chronological results, every measurement, all deviations.
- This file — where the code lives, how it fits together, and what will bite you.

**Status**: 10 of 11 phases implemented. Only Phase 6 (MSFP) is missing —
blocked because the PyPI `mamba-ssm` sdist ships without its CUDA sources.
184 tests pass. Final held-out test results are in `project_log.md`.

---

## Environment

conda env `jmdst` (Python 3.11, PyTorch 2.11+cu128, **RTX 4070 Laptop — 8 GB
VRAM**, not the 12 GB desktop card). **Activate it for every command** — shell
state does not persist between tool calls:

```bash
conda activate jmdst
```

The 8 GB budget is a real constraint on detector scale, not a footnote:
YOLOv11m at the batch 16 every existing detector used needs 8.19 GB and OOMs.
Measured per-variant VRAM and the correct `--batch` are in
`scripts/train_yolo.py`'s module docstring — read it before launching any
detector run larger than yolo11n.

CUDA Toolkit 12.8 and VS Build Tools are installed. Building any CUDA
extension additionally needs `vcvarsall.bat x64` run **in the same process**
as pip, plus `DISTUTILS_USE_SDK=1` (see the mamba-ssm writeup in
`project_log.md` before attempting that again).

---

## Data flow

```
Datasets/{VisDrone_MOT,UAVDT}/          raw, gitignored
        │  scripts/prepare_dataset.py  (+ make_splits.py / make_visdrone_split.py)
        ▼
data/unified/<dataset>/<split>/<seq>/   seqinfo.json, annotations.jsonl,
        │                               ignore_regions.jsonl (UAVDT only)
        ├─ scripts/export_yolo_dataset.py ─► data/yolo*/ ─► scripts/train_yolo.py
        ├─ jmdst/data/felnet_episode.py    ─────────────► scripts/train_felnet.py
        └─ scripts/run_jmdst.py  (YOLO + FELNet + tracker)
                 ▼
        outputs/jmdst_results/…/<seq>.txt   MOT format
                 ▼
        scripts/evaluate.py ─► MOTA/MOTP/IDF1/HOTA/FPS
```

Everything under `data/`, `Datasets/`, `outputs/` is gitignored and
regenerable. Only code is committed.

---

## Module map

| Module | Responsibility | Paper ref |
|---|---|---|
| `jmdst/data/schema.py` | unified annotation dataclasses, class maps | A.9 |
| `jmdst/data/converters/` | VisDrone + UAVDT → unified; bbox clipping; UAVDT ignore regions | A.9 |
| `jmdst/data/crops.py` | SSI crop geometry, overlap vectors (Eq. 1/2), anchor targets | A.2, A.3 |
| `jmdst/data/datasets.py` | PyTorch datasets (detection / SSI / sequence) | — |
| `jmdst/data/felnet_episode.py` | episode sampler for FELNet training | Algorithm 2 |
| `jmdst/data/yolo_export.py` | unified → YOLO layout (hardlinks images) | A.7 |
| `jmdst/data/features.py` | extract embedding sequences (MSFP input) | A.5 |
| `jmdst/models/felnet.py` | FELNet backbone + 3 heads, decode, anchor selection | A.3, Table 1 |
| `jmdst/training/felnet_loss.py` | overlap / confidence / embedding RMSE losses | Eq. 3-7 |
| `jmdst/tracking/kalman_filter.py` | 8-D constant-velocity KF over `[cx,cy,a,h]` | A.6 |
| `jmdst/tracking/track.py` | tentative→confirmed→deleted lifecycle | A.6 |
| `jmdst/tracking/matching.py` | IoU/cosine costs, gating, Hungarian, cascade | A.6 |
| `jmdst/tracking/tracker.py` | detection-frame association, expansion set, output filter | A.6 |
| `jmdst/pipeline/jmdst.py` | **the dual-branch loop** (`JMDSTTracker`) | Algorithm 1 |
| `jmdst/pipeline/models.py` | YOLO/FELNet adapters + `build_jmdst()` | — |
| `jmdst/eval/` | CLEAR MOT + IDF1 (motmetrics) and HOTA (own impl) | Sec. 3.2 |
| `jmdst/net/link.py` | edge/ground sim: netem-equivalent link model (delay, jitter, rate, buffer, random/GE loss), netem-syntax parser | — |
| `jmdst/net/relay.py` | UDP relay applying one link model per direction, per-packet telemetry, Windows timer handling | — |
| `jmdst/net/protocol.py` | edge<->ground wire format: MTU fragmentation (no retransmission), frame codecs, bit-exact detection replies | — |
| `jmdst/net/detcache.py` | modelled ground station: detections + each frame's unloaded compute time, keyed by encoded-frame hash | — |
| `jmdst/net/nodes.py` | `RemoteDetector` (edge: a drop-in `detector` that returns `None` on loss/deadline) and `GroundServer` | — |
| `jmdst/sim/hardware.py` | edge compute limits via one Windows Job Object: CPU hard cap, core affinity, memory; children break away | — |
| `jmdst/sim/camera.py` | real-time camera (own process, shared-memory ring, seqlock) and `latest_finished` streaming scoring | — |
| `jmdst/pipeline/cli.py` | tracker options shared by `run_jmdst.py` and `edge_node.py`, so both build an identical tracker | — |

`jmdst/pipeline/jmdst.py` takes `detector` and `localizer` as **injected
callables**, so the branch logic is unit-testable with mocks and models are
swappable without touching orchestration. Preserve that seam.

The edge/ground simulation uses exactly that seam: the edge's `detector` is a
`RemoteDetector` that ships the frame to the ground. A detector may return
**`None` ("no result for this frame") as distinct from `[]` ("looked, found
nothing")**; on `None` the frame runs the tracking branch instead. Don't
collapse the two -- `[]` at a detection frame unmatches every track.

---

## Invariants that will bite you

Each of these cost real debugging time. Breaking one usually produces
plausible-looking but wrong numbers rather than a crash.

1. **`overlap_scale` — two coordinate spaces.** FELNet may predict overlaps
   normalized (`overlap_scale=64`) or in SSI pixels (`1.0`, the default and
   what old checkpoints use). Training targets are divided by it; **every
   inference path must multiply the model output back to pixels** before
   anchor selection or box decoding. Already handled in
   `pipeline/models.py`, `data/features.py`, `scripts/eval_felnet.py`. Add a
   new inference path and you must do it there too.
2. **Anchor ordering is row-major** and must agree between
   `crops.anchor_boxes()` and FELNet's `flatten(2)`. Reordering either
   silently mismatches predictions to targets.
3. **`select_anchor_output` needs a per-anchor reference overlap**, not one
   shared vector — a full-crop-relative overlap and a 32×32-anchor-relative
   one are different scales. See `anchor_reference_overlaps`.
4. **The tracking branch must call `track.update()`** for every target it
   localizes. Confirmation needs N=τ+1 *consecutive present frames* counting
   **both** branches; skip the update and tentative tracks are deleted before
   they can ever confirm.
5. **Appearance matching is Mahalanobis-gated** (standard DeepSORT). It
   disambiguates nearby targets but will not rescue large position jumps —
   that is the paper's own documented failure mode, not a bug.
6. **HOTA frame offsets must be shared** between GT and predictions
   (`_global_frames_pair`). Computing them independently desynchronizes every
   sequence after a frame-range mismatch — VisDrone's object-free sequences
   made HOTA read 1.5 instead of 38.5 while MOTA still looked fine.
7. **Object-free sequences produce NaN metrics.** OVERALL MOTP is recomputed
   as a match-weighted mean to survive that.
8. **motmetrics 1.4 needs a `np.asfarray` shim** on NumPy 2 — applied at the
   top of `eval/metrics.py` before importing motmetrics.
9. **Degenerate boxes are guarded in two places**: `FELNetLocalizer` falls
   back to the input box when the overlap decode is degenerate, and
   `_confirmed_outputs` skips a frame whose Kalman box is transiently
   invalid. Both were found only by running on real data. A third guard caps
   the box `FELNetLocalizer` crops at the frame size: under heavy packet loss
   a long-coasting track's box grew without bound and the crop raised
   `MemoryError` (found by the edge/ground sweep at 2% loss).
10. **ultralytics nests relative `--project` paths** under its own
    `runs/detect/…`. Always resolve to an absolute path.
11. **`.gitignore` dataset patterns must stay root-anchored** (`/data/`, not
    `data/`). An unanchored `data/` once shadowed `jmdst/data/` and kept the
    entire source package out of git.
12. **Windows ignores a background process's 1 ms timer request after ~2 s.**
    `timeBeginPeriod(1)` alone works in short tests, then a mostly-idle
    windowless process is throttled back to ~15.6 ms wake-ups. Sparse
    traffic then gets released up to ~14 ms late while busy traffic looks
    fine. Any simulation process that sleeps must use
    `jmdst.net.high_resolution_timer`, which also opts out of that throttling.
13. **In real time, the tau schedule counts processed frames, not camera
    frames.** `JMDSTTracker.skip_frame()` coasts tracks (Kalman predict) for a
    frame the edge was too busy for but does not advance `frame_index`.
    Scheduling on camera frames lets a steadily overloaded edge alias onto, or
    straight past, the detection frames.
14. **Real-time results are scored with the streaming-perception rule**
    (`jmdst.sim.latest_finished`): camera frame j shows the newest result
    *finished* by its capture time, so even a fast edge shows the previous
    frame. Compare real-time numbers with real-time numbers. Compare a codec
    with the same codec too: JPEG q90 alone costs 0.8 HOTA on UAVDT val.
15. **On one laptop the nodes are coupled through the power budget.** A busy
    CPU-only edge makes the laptop down-clock the GPU the ground's YOLO runs
    on: 1980 -> 465 MHz measured, YOLO 15 -> 42 ms per frame, with identical
    requests. A real ground station is a separate machine, so CPU-edge runs
    carry a ground-compute penalty that GPU-edge runs do not. Core pinning
    (`--host-cpus`) and EcoQoS opt-out do not remove it. Read
    `ground_compute_ms` in the telemetry before comparing edge configurations.
    Sweeps avoid it with the modelled ground (`--ground-cache`), which replies
    with live-identical detections after each frame's unloaded compute time.

---

## Running things

```bash
python -m unittest discover -s tests          # 184 tests; use discover, not -m unittest tests.x
python scripts/run_jmdst.py --yolo <w> --felnet <c> --datasets uavdt --split val --tau 3
python scripts/evaluate.py --results outputs/jmdst_results/uavdt/val --dataset uavdt --split val
python scripts/run_ablations.py --datasets uavdt --split val
python scripts/make_presentation_figures.py && python scripts/build_report.py
python scripts/demo.py --source <video|frames|seq-dir> --output out.mp4   # live demo
python scripts/calibrate_link.py                # link emulator: configured vs measured, exits 1 on failure
python scripts/run_sim.py --yolo <w> --felnet <c> --e2g "<netem>" --g2e "<netem>" --output-root <dir> <edge args>   # ground + link + edge
python scripts/ground_node.py --yolo <w>        # or run the three nodes in three terminals:
python scripts/link_relay.py --e2g "<netem>" --g2e "<netem>"
python scripts/edge_node.py --felnet <c> <same tracker args as run_jmdst.py>
python scripts/verify_realtime_replay.py --run <run_sim output> --dataset uavdt   # real-time run must replay byte-identically
python scripts/sim_telemetry.py --run <edge output-root>   # per-detection-frame telemetry + run summary (snapshot 4.11)
python scripts/build_detection_cache.py --yolo <w> --datasets uavdt visdrone --split val --conf 0.55 --byte --byte-low-conf 0.4 --output <npz>   # idle machine only
python scripts/run_sweep.py --plan configs/sweeps/edge_ground_val.json   # resumable; then scripts/sweep_report.py --results <out>/results.csv
python scripts/link_relay.py --e2g "delay 40ms rate 8mbit loss 1%" --g2e "delay 40ms"   # run the emulator
```

`scripts/demo.py` is the presentation path: video/frame in, rendered boxes and
track IDs out. It drives the library through `build_jmdst` + `process_frame`
rather than `run_jmdst.py`, so it cannot affect measured results. Rendering
lives in `scripts/` on purpose — presentation, not pipeline.

Tests import each other by bare module name, so they only work via
`discover`. Long jobs (YOLO/FELNet training, full-split runs) are historically
run by the user, not launched in the background.

---

## Current best configurations

Selected on validation; see `project_log.md` for the full tables and the
held-out test numbers.

### The general configuration (use this)

**One** setting for all inputs — the deployment target, since the model must
handle arbitrary UAV footage rather than a specific benchmark:

| Detector | FELNet | Association | conf | tau | Extras |
|---|---|---|---|---|---|
| `retrain_aug_m_lowlr` (combined yolo11m, low LR) | `full_run` (`overlap_scale=1`) | IoU-only | **0.55** | 3 | `--nsa --cmc --byte --byte-low-conf 0.4` |

**`--nsa` must stay detection-frames-only.** `--nsa-tracking` also applies it
to tracking-branch updates and costs 1.6 HOTA: FELNet's anchor confidence
reports anchor *coverage*, not localisation accuracy, so it is the wrong
signal to scale measurement noise by. See `project_log.md`.

Val, **with `--nsa --cmc`** (test not yet re-run with these — the test column
below predates both):

| | UAVDT val | VisDrone val | UAVDT test | VisDrone test |
|---|---:|---:|---:|---:|
| HOTA | **46.6** | **51.8** | **47.8** | **49.3** |
| IDF1 | 61.1 | 64.8 | 59.7 | 59.7 |
| AssA | 55.8 | 54.4 | 47.1 | 60.6 |
| DetA | 39.1 | 49.8 | 48.7 | 40.4 |
| MOTA | 25.7\* | 56.3 | 55.4 | 44.6 |
| IDs | 60 | 519 | 637 | 317 |
| FPS | 46.9 | 20.2 | 26.9 | 15.6 |



Held-out test confirms val. Against the published paper, **VisDrone now ties
on MOTA (44.6 vs 44.9) and exceeds on HOTA (+2.9), AssA (+3.7), DetA (+2.4)
and MOTP (+2.1)** — and edges QDTrack, the paper's strongest baseline, on HOTA
and AssA. UAVDT still trails on every metric (HOTA -12.3). Splits are not the
paper's, so none of this is like-for-like.

conf 0.55 maximises the *worse* of the two HOTAs (the right criterion for a
general model, rather than the average), is VisDrone's outright optimum, and
sits 0.6 below UAVDT's own peak. **It is also the paper's unmodified value**,
so this configuration carries no threshold deviation — unlike every
per-dataset config that preceded it.

\* **UAVDT MOTA is deflated by a labelling artifact, not by tracking quality.**
The combined detector finds cars generally; UAVDT labels only some of them, so
the extras score as FP (10,628 vs the UAVDT-only detector's 5,271) and MOTA
punishes FP linearly while HOTA does not. On the same run the general config
has *better* association than the UAVDT-specific one (IDs 67 vs 139, AssA 51.9
vs 51.5). **Report HOTA as the headline and state the UAVDT MOTA caveat.**

### Per-dataset optima (reference only — do not ship these)

| Dataset | Detector | FELNet | Association | HOTA |
|---|---|---|---|---:|
| UAVDT | `uavdt_only_m_lowlr` @ conf 0.65 | `rebalanced` (`overlap_scale=64`) | appearance | 44.6 |
| VisDrone | `retrain_aug_m` @ conf 0.45 | `full_run` | IoU-only | 43.5 |

**Generalising now costs essentially nothing.** The general config reaches
HOTA 44.3 on UAVDT (vs the per-dataset 44.6) and 45.4 on VisDrone — *better*
than VisDrone's own per-dataset optimum, because the low-LR detector landed
after that config was chosen. One configuration is no longer a compromise.

Checkpoints live under `outputs/` and are **gitignored** — retrain or copy
them; they do not come with a fresh clone.

**A detector swap invalidates the confidence threshold**, and so does a
dataset swap. Against the paper's 0.55, the current detectors want **0.65**
(UAVDT) and **0.45** (VisDrone) — opposite directions. `uavdt_only_m` reads as
a 7-point MOTA *regression* at 0.55 and a 3.4-point *gain* at 0.75, from
identical weights. Always re-sweep on val, and bracket until the metric turns
over: both datasets' optima were initially missed by sweeping too narrow a
range.

**Judge a detector on the association mode you actually ship.** On VisDrone,
`retrain_aug_m` looks like a *reject* under appearance matching (HOTA -1.3,
AssA -5.1) and a clear *adopt* under IoU-only (HOTA +1.3, AssA +0.1) — same
weights, same sweep. yolo11m's extra detections overwhelm FELNet's appearance
matching but not IoU matching, so the appearance model degrades exactly as the
detector improves.

---

## Open threads

- **Phase 6 / MSFP** — not implemented. Build from the `state-spaces/mamba`
  GitHub source (has `csrc/`, unlike the PyPI sdist) or use WSL2. Worth ~1%
  by the paper's own ablation, so low priority.
- **Detector quality is the dominant accuracy lever**, not the tracker. The
  per-sequence spread (MOTA 80.4 down to negative on the same split) is the
  evidence — and acting on it delivered the project's largest single gain:
  yolo11n -> yolo11m at a corrected LR took UAVDT val MOTA 25.3 -> 36.4
  (HOTA 38.7 -> 44.6). See `project_log.md`.
- **ultralytics' `optimizer='auto'` discards `lr0`** and picks the optimizer by
  iteration count, so `--epochs` alone can switch MuSGD <-> AdamW. Pinning the
  optimizer in turn changes `momentum` and `warmup_bias_lr` away from what
  `auto` sets. `scripts/train_yolo.py` guards the first case and documents all
  three; read its docstring before any comparative training run.
- **Appearance matching underperforms IoU-only** on both datasets, and it
  replicated on held-out test. The loss-rebalance fix confirmed the diagnosed
  mechanism but was a dataset-dependent trade, not a general win.
- **Split variance is large** — val and test disagreed by ±13-18 MOTA in
  opposite directions. Quote per-sequence spread with any single number.
