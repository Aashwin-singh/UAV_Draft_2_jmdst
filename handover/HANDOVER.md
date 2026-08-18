# JMDST — handover for evaluation

For teammates presenting without the original author or their laptop.

---

## ⚠️ Read this first: two required snapshots cannot be produced

**Snapshots 4.10 and 4.11 describe a system this project does not contain.**

* **4.10** — "Two terminal windows side-by-side: **Ground Node** and **Edge
  Node**, showing loaded/listening/serving and sent-vs-local frames"
* **4.11** — "Telemetry output/table showing **send time, arrival time, queue
  delay, bytes, loss flag** + run summary"

That is a **distributed edge↔ground streaming system with network telemetry**.
This repository is a single-process tracker. Verified by search: there is no
socket, no client/server, no node split, no message queue, and no telemetry
anywhere in `jmdst/` or `scripts/`. The only match for "telemetry" is a comment
in `cmc.py` stating that camera-motion compensation needs *no* telemetry.

**Nothing in the 10 other snapshots is affected — those are all real and
included here.** But 4.10 and 4.11 cannot be generated from this codebase, and
no amount of running existing scripts will produce them.

**Someone must resolve this before evaluation.** Three possibilities:

1. The snapshot list belongs to a **different module or a different team's
   project**, and was sent by mistake.
2. The capstone brief includes an **edge/ground deployment component** that was
   never in this repository's scope (the roadmap here is 11 phases of detection
   + tracking, none of which mention networking).
3. It is **planned work that was never started**.

Ask whoever issued the snapshot list which it is. If a distributed component is
genuinely required, that is new development — not a screenshot — and the
evaluator needs to know it does not exist rather than discovering it live.

---

## What is in this handover

```
handover/
  HANDOVER.md              <- this file
  snapshots/               <- pre-generated evidence for 4.1-4.9 and 4.12
demo_package/              <- the live demo (clips, fallbacks, launcher, README)
RESULTS_SUMMARY.md         <- every results table, for the presentation
```

Everything in `snapshots/` was generated on the author's machine from the real
system. It can be uploaded as-is. Where a teammate can re-run something on
their own machine, that is noted — a live re-run is stronger evidence, but only
if their machine is set up (see "What teammates need").

---

## Snapshot checklist

| # | Requirement | Status | File / how |
|---|---|---|---|
| 4.1 | Env + PyTorch + CUDA + GPU | **Ready** | `snapshots/4.1_environment.txt` — better re-run live, see below |
| 4.2 | Converter output + verifier passed | **Ready** | `snapshots/4.2_dataset_verify.txt` |
| 4.3 | Split files | **Ready** | `snapshots/4.3_splits/` (11 files) |
| 4.4 | Detector training console + plot | **Ready** | `snapshots/4.4_detector_training.txt` + `4.4_detector_plots/results.png` |
| 4.5 | FELNet loss + component eval | **Ready** | `snapshots/4.5_felnet.txt` |
| 4.6 | Pipeline command + progress + FPS | **Ready** | `snapshots/4.6_pipeline_run.txt` |
| 4.7 | Eval table, per-sequence + overall | **Ready** | `snapshots/4.7_evaluation/{uavdt,visdrone}_test.md` |
| 4.8 | Consecutive DETECTION / TRACKING frames | **Ready** | `snapshots/4.8_branch_frames/` (two PNGs) |
| 4.9 | τ sweep table | **Ready** | `snapshots/4.9_tau_sweep.txt` |
| 4.10 | Ground Node / Edge Node terminals | **NOT POSSIBLE** | does not exist — see above |
| 4.11 | Network telemetry table | **NOT POSSIBLE** | does not exist — see above |
| 4.12 | Test run showing count + OK | **Ready** | `snapshots/4.12_tests.txt` |

### Notes on specific snapshots

**4.1 and 4.12 are worth re-running live** if a teammate has the environment
set up, because they prove *their* machine works. The included files are the
author's machine and serve as a fallback. Commands:

```powershell
python -c "import sys, torch, ultralytics; print(sys.version.split()[0]); print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(ultralytics.__version__)"
python -m unittest discover -s tests
```

**4.4** — the training console was not captured live at the time, so the file
is a per-epoch table rebuilt from `results.csv`, which is the same data the
console printed. The required "actual generated training plot" is
`4.4_detector_plots/results.png`, produced by ultralytics during training.
Note the best epoch is **epoch 1** — that is a real finding (a 20M-parameter
model overfits 8,472 correlated video frames almost immediately), not a broken
run, and it is explained in `RESULTS_SUMMARY.md` §3.1.

**4.8** — the two frames are consecutive (frame 90 and 91) from the same run.
Frame 90 is a detection frame, 91 a tracking frame, and the HUD's second line
names the active branch. **15 of the 15 detection-frame IDs persist into the
tracking frame** — that is the identity continuity the snapshot is asking for.

**4.9** — regenerated with the current shipping configuration, so it matches
the numbers in `RESULTS_SUMMARY.md`. It is real experiment output, not a
diagram.

---

## What teammates need to run the demo

**Minimum: a Windows/Linux machine with an NVIDIA GPU.** On CPU the demo runs
far below real time and is not worth showing.

1. **The repository** (git clone or a zip).
2. **The two model checkpoints** — these are gitignored and cannot be
   regenerated without hours of GPU training. **They must be copied manually:**

   | From the author's machine | Size |
   |---|---|
   | `outputs/yolo_runs/retrain_aug_m_lowlr/weights/best.pt` | 39 MB |
   | `outputs/felnet_runs/full_run/best.pt` | 17 MB |

3. **`demo_package/clips/`** — three self-contained video clips (84 MB). These
   carry their own pixels, so **no datasets are needed for the demo**.
4. **`demo_package/prerendered/`** — the same clips already tracked (89 MB),
   as the no-risk fallback.

Total to send: roughly **230 MB** plus the repository.

### Setup on their machine

```powershell
conda env create -f environment.yml
conda activate jmdst
```

Then copy the two checkpoints into `outputs/yolo_runs/retrain_aug_m_lowlr/weights/`
and `outputs/felnet_runs/full_run/` respectively, keeping those exact paths.

**CUDA matters.** PyPI's default `torch` is a CPU build. `environment.yml`
carries the `--extra-index-url` that forces the GPU build — do not "simplify"
it away.

### Verify before presenting

```powershell
python scripts/preflight.py --source demo_package/clips/uavdt_M0101.mp4
```

Exits 0 only when nothing is blocking. It checks Python, the CUDA build, GPU
memory, both checkpoints, that the footage decodes, mains power, disk, and runs
8 real frames end-to-end. **If this fails, the live demo will fail** — use the
pre-rendered videos instead and say so plainly.

### Running the demo

```powershell
python scripts/demo.py --source demo_package/clips/uavdt_M0101.mp4 --show
```

Press `q` to stop. Or double-click `demo_package/run_demo.bat`, which runs
preflight first and refuses to launch if anything is blocking.

Open with **uavdt_M0101** (cleanest, ~13 s), then a VisDrone clip to show one
configuration handling both datasets without any setting change.

---

## Two things the presenter must know

**Point at the HUD's second line.** It alternates `DETECT (YOLO+FELNet)` and
`TRACK (FELNet only)`. That is the paper's central idea visible live: full
detection runs on only 1 frame in 3, and a cheap single-target branch covers
the rest — which is *why* it is fast enough to run in real time.

**No boxes appear for the first ~4 frames of any clip.** A tracklet is only
output once confirmed, which needs N = τ+1 = 4 consecutive present frames.
That is the paper's rule, not a defect. It looks like a bug if the presenter is
not ready to explain it.

---

## Headline results, for questions

Full tables in `RESULTS_SUMMARY.md`.

| Held-out test | UAVDT | VisDrone |
|---|---:|---:|
| MOTA | 55.4 | 44.6 |
| HOTA | 47.8 | 49.3 |
| IDF1 | 59.7 | 59.7 |
| FPS | 26.9 | 15.6 |

**On VisDrone this matches the published paper on MOTA (44.6 vs 44.9) and
exceeds it on HOTA (+2.9), AssA (+3.7), DetA (+2.4) and MOTP (+2.1).**

**State this caveat whenever the comparison is made:** the paper never
published its sequence splits, so ours are seeded-random and *not* the same
test set. The comparison is indicative, not like-for-like. UAVDT still trails
the paper on every metric.

### Likely questions

* **"Why is UAVDT MOTA low on validation?"** UAVDT annotates only some of the
  visible cars; our detector finds more of them, and the extras score as false
  positives. MOTA penalises FP linearly, HOTA does not — hence HOTA is the
  headline metric here.
* **"Is Phase 6 done?"** No. MSFP is not implemented — its dependency
  (`mamba-ssm`) ships without the CUDA sources its own installer requires. The
  paper's own ablation puts it at ~1% of the metrics.
* **"Why IoU matching instead of the paper's appearance model?"** Measured five
  times across validation and test: IoU-only wins by up to 8.7 HOTA with 72%
  fewer ID switches. The reason is that different vehicles seen from above are
  not visually separable enough — different-identity embedding similarity sits
  at ~0.11 no matter how the loss is weighted.
