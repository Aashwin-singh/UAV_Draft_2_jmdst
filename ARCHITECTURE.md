# Architecture & handoff guide

Code map and hard-won invariants for making **architecture-level changes** to
this JMDST reproduction. Read alongside:

- `PROJECT_CONTEXT.md` — **Section A is the paper spec and is ground truth.**
  Implement against it rather than re-deriving from the PDF.
- `project_log.md` — chronological results, every measurement, all deviations.
- This file — where the code lives, how it fits together, and what will bite you.

**Status**: 10 of 11 phases implemented. Only Phase 6 (MSFP) is missing —
blocked because the PyPI `mamba-ssm` sdist ships without its CUDA sources.
93 tests pass. Final held-out test results are in `project_log.md`.

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

`jmdst/pipeline/jmdst.py` takes `detector` and `localizer` as **injected
callables**, so the branch logic is unit-testable with mocks and models are
swappable without touching orchestration. Preserve that seam.

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
   invalid. Both were found only by running on real data.
10. **ultralytics nests relative `--project` paths** under its own
    `runs/detect/…`. Always resolve to an absolute path.
11. **`.gitignore` dataset patterns must stay root-anchored** (`/data/`, not
    `data/`). An unanchored `data/` once shadowed `jmdst/data/` and kept the
    entire source package out of git.

---

## Running things

```bash
python -m unittest discover -s tests          # 93 tests; use discover, not -m unittest tests.x
python scripts/run_jmdst.py --yolo <w> --felnet <c> --datasets uavdt --split val --tau 3
python scripts/evaluate.py --results outputs/jmdst_results/uavdt/val --dataset uavdt --split val
python scripts/run_ablations.py --datasets uavdt --split val
python scripts/make_presentation_figures.py && python scripts/build_report.py
```

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

| Detector | FELNet | Association | conf | tau |
|---|---|---|---|---|
| `retrain_aug_m_lowlr` (combined yolo11m, low LR) | `full_run` (`overlap_scale=1`) | IoU-only | **0.55** | 3 |

| | UAVDT val | VisDrone val | **UAVDT test** | **VisDrone test** |
|---|---:|---:|---:|---:|
| HOTA | 44.3 | 45.4 | **45.0** | **44.5** |
| IDF1 | 58.8 | 56.1 | 57.1 | 52.8 |
| AssA | 53.2 | 47.3 | 43.8 | 53.9 |
| DetA | 37.0 | 44.2 | 46.4 | 37.1 |
| MOTA | 21.8\* | 46.6 | 51.8 | 38.2 |
| IDs | 74 | 849 | 767 | 583 |
| FPS | 45.2 | 19.0 | 30.3 | 15.6 |

Held-out test confirms val: HOTA 45.0 / 44.5. Against the published paper,
VisDrone is within **0.9 DetA / 1.9 HOTA**; UAVDT trails by 15.1 HOTA, and on
test that shortfall is **association-dominated** (AssA -17.5 vs DetA -12.6),
which is what CMC / NSA-Kalman target.

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
