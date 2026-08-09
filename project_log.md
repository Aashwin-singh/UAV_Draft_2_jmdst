# Project Log

Running status for the JMDST reproduction, for teammates picking up work in
parallel. See `PROJECT_CONTEXT.md` for the full technical spec (Section A)
and detailed phase roadmap (Section B) — this file is just a quick "where
are we" snapshot, updated as phases complete.

## IN PROGRESS — detector capacity upgrade: yolo11n → yolo11m (2026-08-06)

Post-test improvement work. Rationale is the project's own repeated finding:
**the detector is the dominant accuracy lever, not the tracker** (per-sequence
MOTA spread 80.4 → negative on one split; paper Sec. 3.4.4 puts detector
choice at ~8.6 MOTA). Every detector so far is yolo11n (2.6M params), the
smallest variant; the paper never specifies a size, so scaling up is a
faithful-but-stronger choice rather than a deviation.

**Setup measured before launching anything** (RTX 4070 Laptop, 8 GB):

| | params | train VRAM @ batch 16 | verdict |
|---|---:|---:|---|
| yolo11n (current) | 2.6M | 2.38 GB | fits |
| yolo11m | 20.1M | **8.19 GB** | **OOM — exceeds the card** |
| yolo11m @ batch 8 | 20.1M | 4.25 GB | use this |

So yolo11m runs at `--batch 8`, not 16. This does **not** break the A/B against
the yolo11n detectors: ultralytics accumulates to a nominal batch of 64
(`nbs=64`), so the optimizer step is equivalent and weight decay is rescaled;
only BatchNorm statistics differ. Full table and method in
`scripts/train_yolo.py`'s docstring.

**Cost**: ~20.2 ms/image to train vs ~4.9 for yolo11n (4.1×) — roughly 2.9
min/epoch compute-only on `data/yolo_uavdt`, 8.7 min/epoch on `data/yolo`.
Inference is much cheaper than feared: +3.3 ms/detection-frame on UAVDT,
+2.3 ms on VisDrone, and at tau=3 only a third of that per frame, because the
JMDST loop is dominated by FELNet's per-target SSI work, not YOLO. Expect
roughly −4% FPS on UAVDT and −1% on VisDrone, i.e. speed stays in the paper's
range.

### UAVDT result (2026-08-07) — yolo11m WINS, but only after re-tuning conf UP

`uavdt_only_m` trained (batch 8, else identical to `uavdt_only`). It **peaked
at epoch 1** and early-stopped at 21 — textbook overfitting past that (train
box loss 1.47 -> 0.85 where yolo11n only reaches 1.05, while val cls loss
*rises* 1.15 -> 1.45). Both runs share an identical LR schedule (warmup to
~0.0296 by epoch 3), so this is capacity, not config.

Detector quality at each run's own best epoch, UAVDT val:

| | best ep | mAP50 | mAP50-95 | P | R |
|---|---:|---:|---:|---:|---:|
| `uavdt_only` (yolo11n) | 7 | 0.673 | 0.367 | 0.647 | **0.658** |
| `uavdt_only_m` (yolo11m) | **1** | **0.701** | **0.411** | **0.716** | 0.614 |

**End-to-end conf sweep** (tau=3, FELNet `rebalanced`, appearance matching,
ignore regions — all identical to the baseline row, detector is the only
variable):

| Detector / conf | MOTA | MOTP | IDF1 | HOTA | DetA | AssA | IDs | FP | FN | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `uavdt_only` (n) @ 0.55 | 25.3 | 76.2 | 51.7 | 38.7 | 33.7 | 44.5 | **99** | 7069 | 9964 | 55.9 |
| `uavdt_only_m` @ 0.35 | 7.6 | 76.1 | 52.6 | 39.9 | 33.0 | 48.4 | 210 | 12954 | 8007 | 37.4 |
| `uavdt_only_m` @ 0.45 | 13.2 | 76.2 | 53.6 | 40.6 | 33.9 | 48.7 | 184 | 11470 | 8239 | 33.4 |
| `uavdt_only_m` @ 0.55 | 18.3 | 76.4 | 55.0 | 41.6 | 34.7 | 49.9 | 195 | 10014 | 8509 | 47.7 |
| `uavdt_only_m` @ 0.65 | 22.0 | 76.6 | **56.6** | **42.2** | 34.8 | **51.3** | 147 | 8702 | 9039 | 47.7 |
| **`uavdt_only_m` @ 0.75** | **28.7** | 77.0 | 55.0 | 41.5 | **35.2** | 49.0 | 132 | **6232** | 9978 | 51.9 |
| `uavdt_only_m` @ 0.85 | 18.9 | **78.2** | 29.0 | 25.1 | 20.5 | 30.8 | 70 | 1817 | 16709 | **67.8** |

**Three conclusions:**

1. **The conf threshold had to move UP, not down.** The prediction from the
   detector's P/R (higher precision, lower recall -> lower conf) was **wrong**:
   at a fixed threshold the barely-fine-tuned epoch-1 model retains COCO's
   calibration and passes *more* boxes, so it over-detects. Every metric
   improves monotonically 0.35 -> 0.75. 0.85 collapses (FN 16,709).
2. **At conf 0.75 yolo11m beats the yolo11n baseline on every accuracy metric**
   — MOTA +3.4, IDF1 +3.3, HOTA +2.8, DetA +1.5, AssA +4.5, MOTP +0.8, and
   fewer FP (6232 vs 7069) at equal FN. The only regressions are ID switches
   (132 vs 99) and FPS (51.9 vs 55.9). No trade-off to arbitrate: 0.75
   dominates. (conf 0.65 is the HOTA/IDF1/AssA optimum but drops MOTA below
   baseline, so it is not the better pick.)
3. **The win is concentrated in the FP-dominated sequences.** Per-sequence
   MOTA, nano @0.55 -> m @0.75: M1101 **-11.5 -> -0.6** (+10.9, the project's
   worst sequence nearly fixed), M0203 **48.0 -> 54.6** (+6.6), but M0401
   **40.9 -> 36.9** (-4.0). M0401 is the FN-dominated dense scene, so raising
   conf costs recall exactly where recall was already the weak spot. The
   optimal threshold is genuinely sequence-dependent; 0.75 is the best single
   compromise and per-sequence tuning would be val overfitting.

**Deviation to declare**: conf 0.75 is a large departure from the paper's 0.55.
Same justification as the earlier `retrain_aug` @ 0.45 case — the paper's
threshold was tuned for the paper's own detector, and ours has a different
operating point. **Chosen on val; test remains held out.**

### yolo11s (2026-08-07) — RULED OUT, dominated by both nano and medium

`uavdt_only_s` (9.4M params, batch 16 — so args are *byte-identical* to
`uavdt_only` apart from the model, the cleanest A/B in this project). Peaked
at **epoch 3**, early-stopped at 23.

| conf | MOTA | MOTP | IDF1 | HOTA | DetA | AssA | IDs | FP | FN | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.45 | 6.3 | 76.1 | 47.2 | 37.9 | 32.5 | 44.4 | 223 | 12672 | 8577 | 40.8 |
| 0.55 | 9.6 | 76.4 | 48.6 | 38.2 | 32.3 | 45.3 | 220 | 11206 | 9296 | 44.8 |
| 0.65 | 15.2 | 76.9 | 50.4 | **39.2** | 32.6 | **47.3** | 163 | 9325 | 9962 | 46.4 |
| 0.75 | 22.3 | 77.0 | 50.4 | 39.1 | 32.7 | 46.9 | 139 | 6747 | 10921 | 53.5 |
| 0.85 | **23.6** | **78.1** | 41.5 | 32.7 | 27.5 | 39.1 | **91** | 3503 | 13929 | **61.7** |

**Its best point loses to both neighbours.** Peak MOTA 23.6 (conf 0.85, with
HOTA collapsed to 32.7) or 22.3 at conf 0.75 — either way under yolo11n's 25.3
and far under yolo11m's 28.7. Peak HOTA 39.2 vs yolo11m's 41.5. Not
competitive on any metric; **drop it and keep yolo11m**.

**The peak epoch is monotonic in capacity**: yolo11n epoch 7, yolo11s epoch 3,
yolo11m epoch 1. So going *smaller* did not fix the early-peak problem — 8,472
correlated video frames overfit fast at any size above nano. More usefully:
**peaking early does not mean the checkpoint is bad.** yolo11m peaks earliest
and still wins end-to-end by a clear margin. That makes the low-LR retrain
more interesting, not less — if yolo11m can be made to train productively past
epoch 1, it should go beyond 28.7.

### LOW-LR RETRAIN — the big win (2026-08-07)

`uavdt_only_m_lowlr`: yolo11m, `--optimizer MuSGD --lr0 0.0025 --momentum 0.9
--warmup-bias-lr 0.0`, otherwise identical to `uavdt_only_m`. Peaked at epoch
2 (vs epoch 1) and early-stopped at 22.

Detector quality — **best on every measure, and the recall regression is gone**:

| run | best ep | mAP50 | mAP50-95 | P | R |
|---|---:|---:|---:|---:|---:|
| `uavdt_only` (n) | 7 | 0.673 | 0.367 | 0.647 | 0.658 |
| `uavdt_only_s` | 3 | 0.631 | 0.378 | 0.683 | 0.560 |
| `uavdt_only_m` | 1 | 0.701 | 0.411 | 0.716 | 0.614 |
| **`uavdt_only_m_lowlr`** | 2 | **0.757** | **0.462** | **0.760** | **0.660** |

The LR was the right lever. It did not stop the early peak (epoch 2 vs 1) but
it stopped the *collapse* after it: mAP50-95 holds 0.38-0.46 through 12 epochs
where the default-LR run fell to 0.30-0.36.

**End-to-end** (tau=3, FELNet `rebalanced`, appearance, ignore regions):

| Detector / conf | MOTA | MOTP | IDF1 | HOTA | DetA | AssA | IDs | FP | FN | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `uavdt_only` (n) @ 0.55 | 25.3 | 76.2 | 51.7 | 38.7 | 33.7 | 44.5 | **99** | 7069 | 9964 | 55.9 |
| `uavdt_only_m` @ 0.75 | 28.7 | 77.0 | 55.0 | 41.5 | 35.2 | 49.0 | 132 | 6232 | 9978 | 51.9 |
| `..._m_lowlr` @ 0.55 | 32.4 | 76.7 | 58.6 | 44.2 | 38.5 | 51.0 | 146 | 6994 | 8353 | 41.4 |
| **`..._m_lowlr` @ 0.65** | **36.4** | 77.1 | **59.2** | **44.6** | **38.7** | **51.5** | 139 | 5271 | 9167 | 52.1 |
| `..._m_lowlr` @ 0.75 | 32.3 | 78.4 | 53.3 | 40.9 | 34.2 | 49.2 | 112 | 3796 | 11616 | 52.7 |
| `..._m_lowlr` @ 0.85 | 3.2 | 80.9 | 8.7 | 11.7 | 5.2 | 26.7 | 22 | 662 | 21504 | 95.8 |

**+11.1 MOTA, +7.5 IDF1, +5.9 HOTA, +5.0 DetA, +7.0 AssA over the yolo11n
baseline.** Only ID switches regress (139 vs 99).

Two things worth noting:

1. **The optimum moved back DOWN to conf 0.65**, from the default-LR model's
   0.75. A properly-trained model is better calibrated and needs less
   threshold compensation. And at the paper's **unmodified conf 0.55** it
   still scores MOTA 32.4 / HOTA 44.2 — comfortably above every previous
   config — so the headline result no longer depends on a threshold deviation.
2. **M1101 is finally positive.** Per-sequence MOTA on the project's
   pathological sequence: -11.5 (nano) -> -0.6 (m) -> **+11.8** (m_lowlr).
   M0203 also rises 48.0 -> 68.6. M0401 sits at 37.7, still just under nano's
   40.9 — it stays the FN-dominated hard case.

### VisDrone `retrain_aug_m` (2026-08-07) — wins, but only under IoU-only

yolo11m on the combined set, `optimizer: auto` left alone so it matches
`retrain_aug` exactly. Peaked at epoch **17** (vs UAVDT's 1-2) and ran 37
epochs / 8.0 h — consistent with 3x the data and a harder 4-class task.

Detector: mAP50 0.587 vs 0.498, mAP50-95 **0.373 vs 0.298 (+25% relative)**,
recall 0.561 vs 0.478. Much the largest mAP gain in the project.

**Under the faithful appearance config, almost none of it survived:**

| detector / conf (appearance) | MOTA | IDF1 | HOTA | DetA | AssA | IDs |
|---|---:|---:|---:|---:|---:|---:|
| `retrain_aug` (n) @ 0.55 | 39.9 | **48.7** | **40.1** | 39.3 | **41.5** | **1126** |
| `retrain_aug_m` @ 0.55 | **42.1** | 46.4 | 38.8 | **41.9** | 36.4 | 1446 |
| `retrain_aug_m` @ 0.65 | 41.1 | 47.7 | 39.6 | 40.3 | 39.3 | 1185 |
| `retrain_aug_m` @ 0.75 | 38.3 | 46.0 | 38.0 | 37.4 | 39.1 | 997 |

DetA +2.6 and 3,200 fewer misses, but AssA **-5.1**, IDs +320, HOTA negative
at every threshold. The extra detections buy detection quality and spend it
on association errors.

**Under IoU-only — VisDrone's actually-best association — it wins outright:**

| detector / conf (IoU-only) | MOTA | MOTP | IDF1 | HOTA | DetA | AssA | IDs | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `retrain_aug` (n) @ 0.55 | 40.5 | 77.4 | 51.8 | 42.0 | 39.0 | 45.9 | **656** | 9741 | 31234 |
| `retrain_aug_m` @ 0.35 | 44.7 | 77.3 | **54.1** | **44.1** | **43.4** | 45.3 | 912 | 12565 | 25265 |
| **`retrain_aug_m` @ 0.45** | **45.4** | 77.5 | 52.8 | 43.5 | 43.1 | 44.6 | 871 | 10845 | 26503 |
| `retrain_aug_m` @ 0.55 | 42.8 | 77.7 | 53.0 | 43.3 | 41.4 | **46.0** | 801 | 10966 | 28309 |
| `retrain_aug_m` @ 0.65 | 41.7 | 77.9 | 51.8 | 42.6 | 40.0 | 45.9 | 714 | 9617 | 30480 |

Best MOTA **45.4 at conf 0.45** (+4.9 over baseline); best HOTA **44.1 at 0.35**
(+2.1), which is also best IDF1 (54.1) and DetA (43.4). MOTA turns over below
0.45, so the peak is bracketed. **Recommended primary: conf 0.45** — it wins
MOTA outright and gives up only 0.6 HOTA to 0.35. Only IDs regress vs baseline
(871 vs 656).

Note VisDrone's threshold moves **down** (0.45) while UAVDT's moves **up**
(0.65), from the same paper default of 0.55 and with detectors trained the
same way. Further evidence that conf is not transferable — not across
detectors, and not across datasets either.

**The mechanism, and why it matters beyond this run**: swapping association
turns AssA from **-5.1 to +0.1**. yolo11m's extra detections overwhelm
FELNet's *appearance* matching but not IoU matching. That is a new, concrete
mechanism for the project's standing "IoU-only beats our FELNet embeddings"
finding: the appearance model degrades as detection density rises, so it gets
*worse* precisely as the detector gets better. Judging a detector upgrade on
the weaker association mode would have produced the wrong call — the
appearance table above says "reject yolo11m for VisDrone", the IoU-only table
says "adopt it".

**Cross-dataset contrast worth reporting**: the same upgrade is +11.1 MOTA /
+5.9 HOTA on UAVDT and +2.3 / +1.3 on VisDrone. UAVDT's gain is concentrated
in FP-dominated sequences where extra recall is nearly free; VisDrone is dense
and small-target, already at 656-1,126 ID switches vs UAVDT's ~100, so extra
detections mostly add association load. **The size of a detector gain is a
property of the scenes, not of the detector.**

### `train_yolo.py` trap found while setting those up (2026-08-07)

**`--lr0` was a silent no-op as first shipped.** ultralytics' default
`optimizer='auto'` discards it outright — its own log line reads
`ignoring 'lr0=...'`. A low-LR run would have trained at auto's 0.01 and been
indistinguishable from not passing the flag. The script now refuses `--lr0`
without `--optimizer`. Two related confounds, documented in its docstring:
`auto` selects by iteration count (`>10000` -> MuSGD, else AdamW), so changing
`--epochs` alone can switch optimizers; and pinning the optimizer silently
changes `momentum` (0.937 vs auto's 0.9) and `warmup_bias_lr` (0.1 vs auto's
0.0), the latter governing exactly the early epochs that matter here.

## FINAL HELD-OUT TEST — general configuration (2026-08-09)

**Supersedes the 2026-07-28 test section below**, which predates the yolo11m +
low-LR detector. Config was frozen from val before running; nothing was tuned
on test. Both association modes were run as a pre-registered A/B.

Config: `retrain_aug_m_lowlr` + FELNet `full_run` + conf 0.55 + tau 3.

**UAVDT test** — 8 sequences, ignore regions applied:

| Association | MOTA | MOTP | IDF1 | HOTA | DetA | AssA | IDs | FP | FN | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **IoU-only** | **51.8** | 76.5 | **57.1** | **45.0** | **46.4** | **43.8** | **767** | **22840** | 44140 | **30.3** |
| Faithful (appearance) | 49.2 | 76.4 | 45.6 | 37.0 | 46.2 | 29.9 | 2488 | 26426 | **42503** | 28.7 |

**VisDrone test** — 10 sequences:

| Association | MOTA | MOTP | IDF1 | HOTA | DetA | AssA | IDs | FP | FN | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **IoU-only** | **38.2** | 77.8 | **52.8** | **44.5** | **37.1** | **53.9** | **583** | **11383** | 40113 | **15.6** |
| Faithful (appearance) | 36.2 | 77.8 | 48.3 | 41.5 | 36.9 | 47.1 | 1097 | 12988 | **39696** | 15.2 |

### 1. The appearance negative result replicates, and the gap WIDENS on test

UAVDT HOTA **45.0 vs 37.0 (+8.0)**, AssA +13.9, IDs 767 vs 2488 (**-69%**).
VisDrone HOTA 44.5 vs 41.5 (+3.0), AssA +6.8, IDs 583 vs 1097 (-47%). On val
the UAVDT gap was +4.3; on unseen data it doubled. Three independent
confirmations now exist (2026-07-28 test, val, this test), and the mechanism
is documented (the pinned diff-identity floor). This is a settled finding.

### 2. Large improvement over the previous held-out test

| | old IoU variant (2026-07-28) | new | delta |
|---|---:|---:|---:|
| UAVDT MOTA | 43.1 | **51.8** | **+8.7** |
| UAVDT HOTA | 38.8 | **45.0** | **+6.2** |
| VisDrone MOTA | 29.0 | **38.2** | **+9.2** |
| VisDrone HOTA | 39.6 | **44.5** | **+4.9** |

Detector work drove all of it, on genuinely unseen data.

### 3. Against the paper's published numbers — VisDrone is close

| | ours (test) | paper | gap |
|---|---:|---:|---:|
| VisDrone DetA | 37.1 | 38.0 | **-0.9** |
| VisDrone HOTA | 44.5 | 46.4 | **-1.9** |
| VisDrone AssA | 53.9 | 56.9 | -3.0 |
| VisDrone MOTA | 38.2 | 44.9 | -6.7 |
| VisDrone IDF1 | 52.8 | 62.4 | -9.6 |
| UAVDT DetA | 46.4 | 59.0 | -12.6 |
| UAVDT HOTA | 45.0 | 60.1 | -15.1 |
| UAVDT AssA | 43.8 | 61.3 | -17.5 |

**VisDrone detection accuracy is within 0.9 DetA and 1.9 HOTA of the published
paper** on held-out data, with different splits. That is a credible
reproduction on that dataset.

**Diagnosis update for UAVDT — association is now the larger gap, not
detection.** On val the UAVDT shortfall looked detection-driven (DetA -22).
On test DetA recovered to 46.4 (gap -12.6) while AssA sits at -17.5. IDs are
767 against the paper's 170, and one sequence (M0603) contributes 327 of them
alongside 8,836 FP. **This makes CMC and NSA-Kalman the right next work** —
they target association and motion, which is where the remaining UAVDT gap
actually is.

### 4. Speed — honest regression on VisDrone

UAVDT **30.3 FPS** still beats the paper's 26.6. VisDrone **15.6 FPS is now
below** the paper's 18.6, and below our own earlier 18.3-20.9 — yolo11m costs
more than yolo11n. This is a real trade: we bought +4.9 HOTA for ~3 FPS.
Report it as a trade, not a win.

### 5. Per-sequence spread (IoU-only) — quote this with any single number

| UAVDT test | MOTA | | VisDrone test | MOTA |
|---|---:|---|---|---:|
| M0101 | **78.3** | | uav0000308_00000_v | **69.7** |
| M0703 | 69.4 | | uav0000316_01288_v | 57.0 |
| M0603 | 67.8 | | uav0000306_00230_v | 55.5 |
| M1301 | 49.5 | | uav0000119_02301_v | 49.0 |
| M1006 | 40.3 | | uav0000124_00944_v | 41.4 |
| M1004 | 14.6 | | uav0000315_00000_v | 41.1 |
| M1009 | 13.6 | | uav0000281_00460_v | 21.1 |
| M0210 | 10.2 | | uav0000370_00001_v | 16.6 |
| | | | uav0000263_03289_v | 2.2 |
| | | | uav0000073_00600_v | 0.0\* |

\* 13 GT objects in the entire sequence — meaningless at that scale.

Spread remains large (78.3 to 10.2 on UAVDT). Every headline number here
should be quoted with this table.

Artifacts: `outputs/test_general_{iou,app}/{uavdt,visdrone}.{md,json}`.

## PAPER'S OWN NUMBERS — now available (2026-08-09)

**Supersedes the standing note that the paper's UAVDT/VisDrone tables "are images
in the PDF and their values were never extracted."** They were recovered by
rasterising the pages (external research pass, see
`Desktop/Capstone/Results/JMDST_RESEARCH_FINDINGS.md`). Accuracy comparison to
the paper is now possible where it previously was not.

Paper Table 5 — **UAVDT** (their split, 23-of-50 sequences, 12/3/8):

| Method | MOTA | MOTP | IDs | IDF1 | HOTA | DetA | AssA | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSORT | 31.4 | 75.6 | 200 | 48.3 | 38.4 | 31.3 | 47.7 | 15.8 |
| ByteTrack | 32.0 | 74.9 | 165 | 49.8 | 39.2 | 31.6 | 49.2 | 22.9 |
| QDTrack | **69.7** | **81.5** | 179 | **76.5** | **62.9** | **59.2** | **67.0** | 13.0 |
| **JMDST (paper)** | 67.3 | 81.2 | 170 | 73.5 | 60.1 | 59.0 | 61.3 | **26.6** |

Paper Table 6 — **VisDrone2019**:

| Method | MOTA | MOTP | IDs | IDF1 | HOTA | DetA | AssA | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSORT | 36.3 | 75.8 | 905 | 47.9 | 38.7 | 34.3 | 44.3 | 12.2 |
| ByteTrack | 39.0 | 75.6 | 457 | 53.6 | 41.5 | 34.9 | 50.2 | 16.9 |
| QDTrack | **49.5** | 77.2 | **161** | **66.0** | **49.0** | **41.0** | **58.9** | 11.3 |
| **JMDST (paper)** | 44.9 | 76.5 | 178 | 62.4 | 46.4 | 38.0 | 56.9 | 18.6 |

### Where we actually stand (current general config, val)

| | UAVDT HOTA | VisDrone HOTA |
|---|---:|---:|
| Ours (general config, **val**) | 44.3 | **45.4** |
| Paper's JMDST | 60.1 | 46.4 |
| Gap | **-15.8** | **-1.0** |

**VisDrone is now within 1.0 HOTA of the paper.** That is a much smaller gap
than any earlier comparison suggested — the old "final test" figures (VisDrone
HOTA 36.3) predate the yolo11m + low-LR detector, so any table using them
understates the current system by ~9 HOTA. Re-running test is what makes this
claim quotable; until then it is val-vs-published and must be labelled as such.

**UAVDT remains far behind, and it is a detection gap, not a tracking gap**:
paper DetA **59.0** vs ours **37.0**, with AssA much closer (61.3 vs 53.2).
Their detector finds far more of the ground truth than ours does. Note this is
partly protocol — our UAVDT numbers apply ignore-region filtering and suffer
the sparse-annotation FP artifact, and the splits differ (the paper never
published its sequence lists) — but a 22-point DetA gap is too large to be
explained by split variance alone.

**Corroboration worth citing**: the paper's own Sec. 3.4.4 ablation swaps
YOLOv11 -> YOLOX and loses 8.6% MOTA / 19.3% HOTA. Our "detector quality is the
dominant lever" finding is not just true in this codebase — the source paper
measured the same thing.

## FINAL GENERAL CONFIGURATION (2026-08-09)

**`retrain_aug_m_lowlr` + FELNet `full_run` + IoU-only + conf 0.55, tau 3.**

| | UAVDT val | VisDrone val |
|---|---:|---:|
| HOTA | **44.3** | **45.4** |
| IDF1 | 58.8 | 56.1 |
| AssA | 53.2 | 47.3 |
| DetA | 37.0 | 44.2 |
| MOTA | 21.8\* | 46.6 |
| IDs | 74 | 849 |
| FPS | 45.2 | 19.0 |

\* UAVDT MOTA is deflated by the sparse-annotation artifact (see the
2026-08-07 section) — a stronger detector finds more unlabelled cars, each
scored as FP. HOTA is the honest headline.

**Conf sweep** (all IoU-only, new detector). Selection criterion is the
**worse** of the two HOTAs, not the mean — the right objective for a model
that must work on arbitrary input:

| conf | UAVDT HOTA | VisDrone HOTA | worst-case | UAVDT MOTA | VisDrone MOTA | UAVDT IDs | VisDrone IDs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | 43.4 | 44.4 | 43.4 | 10.8 | 43.1 | 112 | 1152 |
| 0.45 | 43.2 | 44.6 | 43.2 | 15.9 | 45.6 | 96 | 1037 |
| **0.55** | 44.3 | **45.4** | **44.3** | 21.8 | **46.6** | 74 | 849 |
| 0.65 | **44.9** | 43.7 | 43.7 | **28.3** | 45.5 | **62** | 724 |

**conf 0.55 is the paper's own unmodified value**, so the final configuration
carries **no threshold deviation** — unlike every per-dataset config that
preceded it. Worth stating plainly in the writeup: the earlier 0.65/0.45
divergence was an artifact of weaker, differently-calibrated detectors.

Improvement over the 2026-08-07 general config on **both** datasets:
UAVDT HOTA 43.2 -> 44.3, IDF1 57.4 -> 58.8; VisDrone HOTA 43.5 -> 45.4,
IDF1 52.8 -> 56.1. Generalising now costs essentially nothing: 44.3 vs the
UAVDT per-dataset optimum's 44.6, and VisDrone's number *beats* its own
former per-dataset best.

### Step 4 — appearance matching cannot be rescued (2026-08-09)

The decisive test of the lambda1 hypothesis, all at conf 0.45 with the new
detector:

| FELNet / association | dataset | MOTA | IDF1 | HOTA | AssA | IDs |
|---|---|---:|---:|---:|---:|---:|
| `lambda_e30` + appearance | UAVDT | 13.1 | 50.9 | 39.2 | 43.7 | 376 |
| `full_run` + appearance | UAVDT | 12.3 | 50.3 | 38.9 | 42.1 | 282 |
| **`full_run` + IoU-only** | UAVDT | **15.9** | **57.0** | **43.2** | **51.2** | **96** |
| `lambda_e30` + appearance | VisDrone | 40.3 | 47.8 | 39.2 | 36.4 | 2038 |
| `full_run` + appearance | VisDrone | 40.5 | 46.6 | 39.6 | 36.3 | 2041 |
| **`full_run` + IoU-only** | VisDrone | **45.6** | **54.4** | **44.6** | **45.1** | **1037** |

**A 20% better embedding bought nothing.** `lambda_e30` has embedding
separation 0.784 vs `full_run`'s 0.653, and end-to-end that is worth HOTA
+0.3 on UAVDT and **-0.4** on VisDrone, AssA +1.6 and +0.1. On UAVDT it
produced *more* ID switches (376 vs 282), not fewer.

Combined with the pinned diff-identity floor from the lambda sweep below, the
conclusion is well supported: **the ceiling is not embedding quality, it is
that these targets are not visually separable enough for appearance matching
to help at all.** Tighter same-identity clusters do not help when different
identities sit at ~0.11 similarity regardless; a tighter threshold then admits
more false matches, which is exactly the UAVDT ID-switch increase.

**The density prediction held.** IoU-only's HOTA margin over appearance is now
**+4.3 / +5.0**, wider than the +4.7 measured on VisDrone with the weaker
detector, and VisDrone appearance ID switches rose to ~2,040 from ~1,446.
Appearance matching does degrade as the detector improves.

For the writeup this is the strongest form of the negative result: the paper's
own prescribed remedy (rebalance the loss so the embedding branch gets
gradient) was implemented, achieved its intended effect on the component
metric, and did not transfer end-to-end — with a mechanism explaining why.

## FELNet lambda1 sweep (2026-08-09) — NEGATIVE: lambda1 is a *worse* lever than overlap-scale

Tested the one untried option from the Phase-11 diagnosis: raise lambda1 (the
embedding weight in `L = l1*L_E + l2*L_o + l3*L_C`) instead of normalising the
overlap targets. Added `--lambda-embedding/--lambda-overlap/--lambda-confidence`
to `scripts/train_felnet.py` (they were not reachable before). Two 50-epoch
runs at lambda1 = 30 and 100, `--overlap-scale 1` so overlap targets stay in
pixel space.

Held-out val, via `scripts/eval_felnet.py`:

| checkpoint | lever | loc IoU | separation | conf acc |
|---|---|---:|---:|---:|
| `full_run` | none (paper l=1,1,1) | **0.634** | 0.653 | **0.843** |
| `rebalanced` | `--overlap-scale 64` | 0.610 | 0.782 | 0.835 |
| `lambda_e30` | `--lambda-embedding 30` | 0.593 | 0.784 | 0.825 |
| `lambda_e100` | `--lambda-embedding 100` | 0.522 | **0.789** | 0.753 |

**Three conclusions:**

1. **lambda1 buys the same embedding gain at a higher localisation price.**
   `rebalanced` reaches separation 0.782 at IoU 0.610; `lambda_e30` reaches
   0.784 (+0.003) at IoU 0.593 (-0.017). Strictly dominated. The pre-registered
   win condition — separation >= 0.78 *without* IoU dropping below 0.610 — is
   met by neither run.
2. **Embedding separation saturates; localisation does not.** lambda1 30 -> 100
   moved separation +0.005 and IoU **-0.071**, with confidence accuracy also
   falling 0.825 -> 0.753. Past ~0.78 the trade is close to pure loss.
3. **The binding constraint is not the loss weighting.** Same-identity
   similarity is already 0.90-0.92, but different-identity similarity sits at
   0.105-0.126 no matter what lambda1 is. Nothing in the loss balance moves the
   diff-identity floor, which is what association actually needs. That points at
   the 16-D embedding's capacity, or at the task (near-identical small vehicles
   from above), rather than at a hyperparameter.

**Practical upshot**: keep `full_run` for any IoU-only configuration — it has
the best localisation (0.634), and localisation is the only FELNet output that
IoU-only association consumes. Both rebalance levers are now explored and
neither produces a general win.

## GENERAL (single) CONFIGURATION — the deployment target (2026-08-07)

Objective changed: the model must work on arbitrary UAV input, so per-dataset
tuning is out. That means collapsing **four** per-dataset choices into one —
detector, FELNet checkpoint, association mode, and conf.

Two were already settled by earlier work: combined training beats per-dataset
training on detector mAP for *both* datasets (cross-domain diversity is a net
positive, see 2026-07-26 item 7), and `full_run` + IoU-only is the HOTA leader
on *both* (see the loss-rebalance table). The open gap was that
`retrain_aug_m` had never been run on UAVDT.

**`retrain_aug_m` + `full_run` + IoU-only, swept jointly:**

| conf | UAVDT HOTA | VisDrone HOTA | UAVDT MOTA | VisDrone MOTA | UAVDT IDs | VisDrone IDs |
|---|---:|---:|---:|---:|---:|---:|
| **0.45** | **43.2** | **43.5** | 19.9 | **45.4** | **67** | 871 |
| 0.55 | 42.7 | 43.3 | 19.8 | 42.8 | 68 | 801 |
| 0.65 | 42.4 | 42.6 | 21.7 | 41.7 | 47 | 714 |

**Chosen: conf 0.45.** HOTA 43.2 / 43.5 — within 0.3 across two very different
datasets, which is the signature of a config not bent toward either. 0.45 also
maximises the *minimum* of the two, the right criterion for a general model
rather than the mean.

**Cost of generalising**: UAVDT HOTA 44.6 -> 43.2 (**-1.4**); VisDrone
**unchanged**, because its per-dataset optimum already *is* this config.

**UAVDT MOTA 36.4 -> 19.9 is a labelling artifact, and it argues for the
general model rather than against it.** UAVDT FP rises 5,271 -> 10,628: the
UAVDT-only detector learned which cars UAVDT bothers to label, while the
combined detector finds cars generally, and the extras score as FP under
UAVDT's sparse annotation (already diagnosed on M1101 — 44% of raw detections
there are real-but-unlabelled cars). MOTA punishes FP linearly; HOTA does not.
On the same runs the general config has **better** association than the
UAVDT-specific one: IDs 67 vs 139, AssA 51.9 vs 51.5. Detecting more real
vehicles is what a deployed model should do. **Report HOTA as headline; state
the UAVDT MOTA caveat explicitly.**

**Declared deviation**: IoU-only departs from the paper, which specifies
FELNet appearance matching. Justified for a deployment target — it wins on
both datasets and its margin *grows* as the detector improves — but the
faithful appearance numbers stay reported alongside.

**Scope honesty**: two daytime, vehicle-only UAV datasets at similar altitudes
is **not** "every kind of input". Nothing here tests night, weather, varied
altitude, other classes, or other sensors. The only genuine generalisation
evidence available in this project is the held-out test split.

## SUPERSEDED — held-out test results (2026-07-28)

**Kept for history. These predate the yolo11m + low-LR detector; see the
2026-08-09 test section above for the current numbers.**

Configs were fixed from val results **before** running test (detector, FELNet
version, tau=3, conf 0.55, mcd 0.3 — all val-selected; see the sections
below). Nothing was tuned on test. Both the faithful configuration (FELNet
appearance matching, as the paper specifies) and the IoU-only ablation
variant were run, as a clean A/B of association alone.

**UAVDT test** — 8 sequences, 6,471 frames, 140,598 objects
(detector `uavdt_only`, FELNet `rebalanced`, ignore regions applied):

| Configuration | MOTA | MOTP | IDF1 | HOTA | DetA | AssA | IDs | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Faithful (FELNet appearance) | **43.6** | 74.6 | 47.3 | 37.1 | **38.0** | 36.4 | 1,133 | 36.0 |
| IoU-only variant | 43.1 | 74.8 | **49.1** | **38.8** | 37.6 | **40.2** | **682** | **46.9** |

**VisDrone test** — 10 sequences, 3,812 frames, 84,302 objects
(detector `retrain_aug`, FELNet `full_run`):

| Configuration | MOTA | MOTP | IDF1 | HOTA | DetA | AssA | IDs | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Faithful (FELNet appearance) | 27.3 | 78.3 | 41.3 | 36.3 | 29.2 | 45.4 | 643 | 18.3 |
| IoU-only variant | **29.0** | **78.4** | **45.8** | **39.6** | **29.4** | **53.6** | **351** | **20.9** |

### Three things the test split established

1. **The IoU-only result REPLICATES on held-out data, on both datasets.**
   UAVDT HOTA 38.8 vs 37.1 and IDs 682 vs 1,133 (-40%); VisDrone HOTA 39.6 vs
   36.3 and IDs 351 vs 643 (-45%). This was the main risk of the Phase 11
   finding being val overfitting — it was not. The negative result stands on
   genuinely unseen data, which makes it a real finding rather than an artifact.
2. **Val and test moved in OPPOSITE directions, by a lot.** UAVDT val 25.3 ->
   test 43.6 (+18.3); VisDrone val 39.9 -> test 27.3 (-12.6). Cause is split
   composition: the paper never publishes its sequence names, so our splits are
   seeded-random, and UAVDT val happened to contain the pathological M1101
   (MOTA -18 to -38) while VisDrone test drew harder sequences. **Practical
   consequence: single-split numbers from this reproduction carry large
   variance and should be quoted with the per-sequence spread, not alone.**
3. **Speed matches or beats the paper on both datasets.** UAVDT 36.0-46.9 FPS
   vs the paper's 26.6; VisDrone 18.3-20.9 vs 18.6. (Newer GPU — RTX 4070 vs
   the paper's 3070 — and we time inference only, so this is indicative, not a
   like-for-like claim.)

### Per-sequence spread (faithful config) — the average hides a lot

| UAVDT test | MOTA | | VisDrone test | MOTA |
|---|---:|---|---|---:|
| M0101 | **80.4** | | uav0000308_00000_v | **69.6** |
| M0603 | 67.2 | | uav0000306_00230_v | 46.6 |
| M0703 | 61.0 | | uav0000316_01288_v | 38.3 |
| M1301 | 44.6 | | uav0000119_02301_v | 33.3 |
| M1004 | 15.7 | | uav0000124_00944_v | 29.1 |
| M1006 | 10.9 | | uav0000315_00000_v | 26.0 |
| M0210 | -2.5 | | uav0000370_00001_v | 8.4 |
| M1009 | -1.4 | | uav0000263_03289_v | -21.2 |
| | | | uav0000073_00600_v | -69.2* |

\* 22 ground-truth objects in the whole sequence — MOTA is meaningless at that
scale (9 FP + 13 FN). Reported for completeness, not interpretation.

UAVDT M0101 at **MOTA 80.4** and VisDrone uav0000308 at **69.6** are the
strongest evidence in the project that the tracking architecture works when
detection is reliable — the same conclusion the val per-sequence analysis
reached, now confirmed on unseen data.

**Cannot compare accuracy to the paper's UAVDT/VisDrone tables**: those tables
are images in the PDF and their values were never extracted. Only the paper's
FPS figures (abstract) and its MDMT ablation tables are available as numbers,
so accuracy comparisons in this project are limited to trends, not values.

Artifacts: `outputs/test_final/{uavdt,visdrone}_{faithful,iou}.{md,json}`.

## STATUS: all 11 phases implemented (10 complete, Phase 6 blocked)

Phases 1-5 and 7-11 are done, verified, and committed. **Phase 6 (MSFP) is
the only one not implemented** — blocked on `mamba-ssm`, whose PyPI sdist
ships without its CUDA sources (full diagnosis further down). The paper's own
ablation puts MSFP at ~1% of the metrics, so this is a small hole, and it is
documented rather than hidden.

**Headline results** (val splits, tau=3, paper's conf=0.55, ignore regions
applied where available):

| Dataset | Detector | FELNet / assoc | MOTA | IDF1 | HOTA | FPS |
|---|---|---|---:|---:|---:|---:|
| UAVDT | `uavdt_only` | `rebalanced` + appearance | **25.3** | 51.7 | 38.7 | ~50 |
| UAVDT | `uavdt_only` | `full_run` + IoU-only | 20.4 | 54.2 | **41.1** | ~52 |
| VisDrone | `retrain_aug` | `full_run` + IoU-only | **40.5** | **51.8** | **42.0** | ~21 |

Best-MOTA and best-HOTA configs differ (see the loss-rebalance follow-up
below); pick per the metric you are reporting and say which you used.

FPS exceeds the paper's (26.6 UAVDT / 18.6 VisDrone); accuracy is well below
it, dominated by detector quality on hard sequences (see the improvement-work
section). Test splits are deliberately still held out — every number here is
val, and thresholds were tuned on val only.

## Phase 11 — Ablation study (2026-07-27)

Mirrors paper Sec. 3.4. Run with `scripts/run_ablations.py` (tau sweep +
appearance ablation, emits markdown/JSON). Note the paper runs its ablations
on MDMT, which we don't have; ours are on UAVDT val (and VisDrone val for the
appearance ablation).

### A. Detection interval tau (paper Sec. 3.4.3 / Table 3)

UAVDT val, `uavdt_only` detector, conf 0.55, mcd 0.3, ignore regions applied.
tau=1 (detect every frame) is a reference baseline the paper's table omits.

| tau | MOTA | MOTP | IDF1 | HOTA | DetA | AssA | IDs | FP | FN | FPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **34.9** | 78.2 | 48.4 | 38.1 | 37.1 | 39.2 | 362 | 4795 | 9755 | 35.6 |
| 2 | 27.9 | 77.4 | **51.2** | **39.3** | 35.7 | **43.3** | 153 | 6916 | 9469 | 48.2 |
| 3 | 22.6 | 76.8 | 50.3 | 38.0 | 33.9 | 42.9 | 125 | 7886 | 9725 | 51.7 |
| 4 | 19.2 | 76.7 | 48.9 | 37.0 | 32.8 | 42.0 | **96** | 8304 | 10128 | 54.9 |
| 5 | 14.9 | 76.0 | 44.5 | 34.4 | 31.3 | 37.9 | 126 | 8959 | 10429 | 55.9 |
| 6 | 11.0 | 75.9 | 43.4 | 33.7 | 29.5 | 38.6 | 120 | 9189 | 11100 | **57.2** |

**Reproduced 3 of the paper's 4 trends** (checked programmatically over
tau=2..6): MOTA decreases monotonically ✓, FPS increases monotonically ✓,
FN increases ✓. **Diverged on FP**: the paper's FP *decreases* with tau
(14416 -> 11747), ours *increases* (6916 -> 9189).
- Likely cause of the FP divergence: in the paper, larger tau raises the
  N=tau+1 confirmation bar, filtering more false detections. In ours the
  *tracking branch* appears to generate FPs — our FELNet localization is
  weaker (Phase 4: val IoU ~0.65), so longer runs of consecutive tracking
  frames drift more, and drifted tracks persist as FP.
- **Our optimum tau is lower than the paper's**: the paper picks tau=3 as the
  best trade-off; ours peaks at tau=2 for HOTA/IDF1/AssA and tau=1 for MOTA.
  Consistent with a weaker tracking branch — we benefit more from frequent
  detection than the paper's implementation does.
- Absolute MOTA is far below the paper's (27.9 vs 69.0 at tau=2), but that is
  a different dataset (UAVDT vs MDMT) *and* our known detector gap; the
  trends are what this ablation validates.
- Our FPS is much higher (48 vs 16 at tau=2) — YOLOv11n on an RTX 4070 vs the
  paper's RTX 3070, and we time inference only.

### B-follow-up. FELNet loss rebalance (2026-07-28) — DIAGNOSIS CONFIRMED, NOT A NET WIN

Acted on B's root-cause hypothesis: normalized overlap targets to ~[0,1]
(`--overlap-scale 64`, commit `2e5c658`) so L_o stops swamping L_E, *without*
changing the paper's lambda weights. Training losses went from
L_o 27.6 / L_C 0.49 / L_E 0.92 to **0.48 / 0.50 / 0.97** — balanced, embedding
now the largest term. Retrained 50 epochs (`outputs/felnet_runs/rebalanced/`).

**FELNet component level — the diagnosis was right:**

| Metric (held-out val) | old `full_run` | `rebalanced` |
|---|---:|---:|
| same-identity similarity | 0.758 | **0.898** |
| diff-identity similarity | 0.105 | 0.116 |
| **embedding separation** | 0.653 | **0.782 (+20%)** |
| localization IoU | **0.634** | 0.610 |

Embedding quality improved substantially, at a small localization cost —
exactly the predicted trade.

**End-to-end, full 2x2 (tau=3, per-dataset best detector, ignore regions):**

| Dataset | FELNet | Assoc | MOTA | IDF1 | HOTA | AssA | IDs |
|---|---|---|---:|---:|---:|---:|---:|
| UAVDT | old | appearance | 22.6 | 50.3 | 38.0 | 42.9 | 125 |
| UAVDT | old | IoU-only | 20.4 | 54.2 | **41.1** | **50.9** | **54** |
| UAVDT | **rebalanced** | appearance | **25.3** | 51.7 | 38.7 | 44.5 | 99 |
| UAVDT | rebalanced | IoU-only | 24.8 | **54.3** | 40.2 | 48.5 | 63 |
| VisDrone | old | appearance | 39.9 | 48.7 | 40.1 | 41.5 | 1126 |
| VisDrone | old | IoU-only | **40.5** | **51.8** | **42.0** | **45.9** | **656** |
| VisDrone | rebalanced | appearance | 36.2 | 46.0 | 37.9 | 39.2 | 1275 |
| VisDrone | rebalanced | IoU-only | 36.2 | 48.8 | 39.6 | 43.7 | 858 |

**Three honest conclusions:**
1. **The rebalance helps UAVDT and hurts VisDrone.** Best UAVDT MOTA rose
   22.6 -> **25.3** (and appearance-path IDs fell 125 -> 99, -21%). But every
   VisDrone number got worse (best MOTA 40.5 -> 36.2). The small localization
   regression matters more on VisDrone's smaller, denser, more varied targets.
   Net: **not a global win** — it is a dataset-dependent trade.
2. **It did NOT flip the ablation.** IoU-only still beats appearance on
   association metrics on *both* datasets with *both* checkpoints. The gap
   narrowed on UAVDT (HOTA 3.1 -> 1.5) but never closed. Threshold tuning is
   not the lever either: swept `max_cosine_distance` 0.15/0.3/0.5 on the new
   checkpoint, HOTA spans only 38.3-38.8 vs IoU-only's 40.2.
3. **Best configs are unchanged for HOTA**: old checkpoint + IoU-only remains
   the HOTA leader on both datasets (UAVDT 41.1, VisDrone 42.0). The
   rebalanced checkpoint owns exactly one crown: best UAVDT MOTA (25.3).

**IMPORTANT CLAIM CORRECTION** (supersedes the wording in section B below):
the paper's Sec. 3.4.2 compares FELNet against DeepSORT's **ReIDNet** —
appearance-model vs appearance-model. It never tests appearance-vs-no-
appearance. So "IoU-only beats our FELNet embeddings" is a finding **beyond**
the paper's ablation, **not a contradiction of it**. We cannot claim the
paper is wrong here; we can only report that on our data, with our FELNet, a
learned appearance model did not beat spatial matching. Earlier phrasing in
section B ("the opposite of the paper") overstated this.

### B. FELNet feature encoding (paper Sec. 3.4.2) — NEGATIVE RESULT

Appearance cascade ON vs OFF (IoU-only), tau=3, mcd=0.3. **Deviation**: the
paper compares FELNet against DeepSORT's ReIDNet; we have no ReIDNet weights,
so OFF means *no appearance model at all*.

| Dataset | Association | MOTA | IDF1 | HOTA | AssA | IDs |
|---|---|---:|---:|---:|---:|---:|
| UAVDT val | FELNet embeddings | **22.6** | 50.3 | 38.0 | 42.9 | 125 |
| UAVDT val | IoU only | 20.4 | **54.2** | **41.1** | **50.9** | **54** |
| VisDrone val | FELNet embeddings | 39.9 | 48.7 | 40.1 | 41.5 | 1126 |
| VisDrone val | IoU only | **40.5** | **51.8** | **42.0** | **45.9** | **656** |

**Our FELNet embeddings HURT association** — the opposite of the paper
(which reports FELNet cutting IDs 541 -> 335 vs ReIDNet). Turning appearance
off cuts ID switches by 57% on UAVDT (125 -> 54) and 42% on VisDrone
(1126 -> 656), and raises HOTA ~2-3 points on both. Only MOTA on UAVDT
slightly prefers appearance ON.
- Tested and ruled out two explanations: (a) threshold tuning — swept
  max_cosine_distance {0.2, 0.3, 0.5}; 0.3 is best and was adopted as the new
  default, but IoU-only still wins; (b) "UAVDT is car-only so all targets look
  alike" — the same pattern holds on VisDrone's 4 varied classes.
- **Most likely root cause, predicted back in Phase 4**: FELNet's training
  loss is dominated numerically by the overlap term (L_o ~27 vs L_E ~0.9,
  because overlap targets are in SSI pixel units 0-64 while the embedding loss
  is a bounded correlation). With the paper's equal weights (1,1,1) the
  embedding branch gets comparatively little gradient signal on the shared
  backbone. This ablation is the confirmation of that Phase-4 warning.
- **Concrete fix to try** (not run): retrain FELNet with a rebalanced loss —
  e.g. normalize overlap targets to [0,1] or raise lambda1 — then re-run this
  ablation. That is a documented deviation from the paper's stated weights,
  but justified by this evidence.

### C. Detector (paper Sec. 3.4.4)

Covered by the improvement work above (original vs `retrain_aug` vs
`uavdt_only`); the paper swaps YOLOv11 for YOLOX and reports ~8.6% MOTA
swing, consistent with our finding that the detector is our dominant lever.

### D. MSFP (paper Sec. 3.4.1) — NOT RUN

Blocked on mamba-ssm (root cause diagnosed below). The paper reports MSFP as
worth +0.4 MOTA / +0.6 IDF1 / +0.3 HOTA / IDs 361 -> 335, i.e. ~1%, which is
why it was deprioritized relative to the detector.

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

5. **Detector retrain with `--strong-aug` + early stopping — DONE, helped**:
   `outputs/yolo_runs/retrain_aug/` (28 epochs, early-stopped). The
   overfitting peak moved epoch 2 -> **epoch 8**, and detector quality rose:
   mAP50 0.451 -> **0.498**, mAP50-95 0.268 -> **0.298**. Augmentation
   genuinely delayed overfitting.
6. **But the retrained detector needs a LOWER conf threshold** (it's less
   confident after heavy augmentation — output count dropped ~20% at the same
   threshold, trading FP for FN). UAVDT val, ignore regions applied:

   | Detector / conf | MOTA | IDF1 | HOTA | AssA | IDs | FP | FN |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | original @ 0.55 (prev. baseline) | 17.0 | 46.6 | 36.6 | 38.4 | 203 | 10229 | 8585 |
   | retrain_aug @ 0.55 | 19.0 | 43.8 | 34.9 | 36.8 | 171 | 8511 | 9884 |
   | **retrain_aug @ 0.45 (best)** | 18.6 | **50.2** | **38.1** | **42.3** | 219 | 9526 | 8915 |
   | retrain_aug @ 0.40 | 18.2 | 49.1 | 37.5 | 40.8 | 165 | 9870 | 8706 |

   **Best config = retrain_aug @ conf 0.45**: vs the previous baseline that's
   +1.6 MOTA, +3.6 IDF1, +1.5 HOTA, +3.9 AssA. Per-sequence: M0203 58.9,
   M0401 37.4 (up from 34.9), M1101 -34.9 (up from -38.0).
   - Note conf 0.45 deviates from the paper's 0.55, which was tuned for the
     paper's own detector; ours is trained differently so its operating point
     differs. Threshold was chosen on VAL — keep TEST held out and report
     final numbers there with this val-chosen config.
   - Honest read: real but **modest** gains. M1101 is still strongly negative
     (-34.9, 8162 FP) and dominates the average.

7. **Per-dataset (UAVDT-only) detector — DONE. Best MOTA, and more faithful.**
   Trained on `data/yolo_uavdt/` (8472 imgs, early-stopped at 27 epochs, best
   epoch 7).
   - **My domain-mixing hypothesis was WRONG on detector quality**: evaluated
     apples-to-apples on UAVDT val, the *combined* detector is actually
     better (mAP50 **0.697** vs UAVDT-only's **0.673**) — the extra VisDrone
     data helps, cross-domain diversity is a net positive. But the UAVDT-only
     model has higher **recall** (0.658 vs 0.623), and recall/FN was our weak
     spot, so it still won on tracking. Good reminder: **detector mAP does
     not reliably predict tracking metrics** — always evaluate end-to-end.

   Full UAVDT val comparison (all with ignore regions applied):

   | Detector / conf | MOTA | IDF1 | HOTA | AssA | IDs | FP | FN | M1101 MOTA |
   |---|---:|---:|---:|---:|---:|---:|---:|---:|
   | original @ 0.55 (first baseline) | 17.0 | 46.6 | 36.6 | 38.4 | 203 | 10229 | 8585 | -38.0 |
   | retrain_aug @ 0.45 | 18.6 | **50.2** | **38.1** | **42.3** | 219 | 9526 | 8915 | -34.9 |
   | **uavdt_only @ 0.55** | **21.2** | 47.4 | 36.1 | 38.9 | **140** | 8196 | 9723 | **-18.0** |

   **Two defensible best configs, genuine trade-off:**
   - `uavdt_only` @ conf **0.55** — best MOTA (21.2), fewest ID switches
     (140, down from 203), and much the best on our problem sequence M1101
     (-18.0 vs -38.0 baseline). **Also the most faithful option**: it matches
     the paper's per-dataset training protocol AND keeps the paper's stated
     conf=0.55 with no threshold deviation.
   - `retrain_aug` @ conf **0.45** — best IDF1 (50.2), HOTA (38.1), AssA
     (42.3), i.e. better identity/association quality, but requires deviating
     from the paper's conf and uses a domain-mixed detector.

   **Recommended primary = `uavdt_only` @ 0.55** on faithfulness grounds
   (per-dataset protocol + unmodified paper hyperparameters), with the
   retrain_aug/0.45 numbers reported as an association-focused alternative.

8. **VisDrone-only detector + first VisDrone tracking evaluation — DONE.**
   Detector trained (`outputs/yolo_runs/visdrone_only/`, 47 epochs, best at
   epoch 27 — much later than UAVDT's epoch 7, consistent with 2x the data and
   a harder 4-class task).
   - Fair detector comparison **on VisDrone val**: essentially a tie, with the
     combined detector marginally ahead — mAP50 **0.517 (combined)** vs
     **0.507 (visdrone_only)**, identical mAP50-95 (0.312), and combined has
     better recall (0.503 vs 0.473). Per-class shows the expected imbalance
     effect (car 0.83-0.84, bus 0.36-0.41): VisDrone train is 83.9% car /
     9.8% van / 4.6% truck / **1.6% bus** (52:1 car:bus).
   - So **per-dataset training helped UAVDT but NOT VisDrone.** Combined
     training is fine (arguably better) for VisDrone.

   **First-ever VisDrone val tracking results** (conf 0.55, tau=3):

   | Detector | MOTA | MOTP | IDF1 | HOTA | DetA | AssA | IDs | FPS |
   |---|---:|---:|---:|---:|---:|---:|---:|---:|
   | **retrain_aug (combined)** | **38.8** | 77.4 | 46.2 | 38.0 | **39.1** | 37.4 | **1141** | 21.7 |
   | visdrone_only | 36.0 | 77.5 | **46.5** | **38.5** | 37.7 | **40.0** | 1193 | 22.6 |

   Roughly a wash; combined is ahead on MOTA/DetA/IDs, visdrone_only slightly
   ahead on HOTA/AssA/IDF1. **Recommend the combined `retrain_aug` detector
   for VisDrone** (better MOTA, fewer ID switches, and no extra model to
   maintain).
   - Notable: **VisDrone tracking (MOTA 38.8, HOTA 38.0) is BETTER than our
     UAVDT results (MOTA 21.2, HOTA 36.1)** — the opposite of the initial
     expectation, because UAVDT's M1101 is pathological for our detector.
   - FPS 21.7 on VisDrone (paper reports 18.6), so speed is in the paper's
     range. Accuracy can't be compared to the paper's VisDrone table directly
     — that table is an image in the PDF and its values were never extracted.

9. **TWO EVAL BUGS FOUND AND FIXED** while doing the VisDrone run (commit
   `d093c2c`) — both triggered by VisDrone val's two pedestrian-only
   sequences (GT frames but zero vehicle objects -> empty prediction files):
   - **HOTA desync (severe)**: per-sequence frame offsets were computed
     independently for GT vs predictions, so a sequence whose frame range
     differed between them desynchronized everything after it. HOTA read
     **1.5** when the true value was **38.5**. Fixed via a shared offset.
   - **OVERALL MOTP = NaN**: undefined per-sequence motp propagated into the
     overall aggregate. Now recomputed as a match-weighted mean.
   - **Previously reported UAVDT numbers were re-verified and are UNCHANGED**
     (all 3 UAVDT val sequences have objects, so the desync never triggered).
   - 3 regression tests added. Lesson: metrics code needs its own edge-case
     tests — this only appeared on a dataset with degenerate sequences, and
     MOTA looked plausible the whole time while HOTA was silently broken.

**Current best configs (both val splits, tau=3):**

| Dataset | Detector | conf | MOTA | MOTP | IDF1 | HOTA | FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| UAVDT | `uavdt_only` | 0.55 | 21.2 | 77.6* | 47.4 | 36.1 | ~50 |
| VisDrone | `retrain_aug` (combined) | 0.55 | 38.8 | 77.4 | 46.2 | 38.0 | 21.7 |

\* per-sequence MOTP; see the UAVDT table above for the full breakdown.
Both use the paper's unmodified conf=0.55 and tau=3.

**Where the gap stands**: tracker/association are validated and working
across both datasets. Detector work delivered UAVDT MOTA 3.0 -> 17.0 -> 21.2;
VisDrone came in at 38.8 first try. Returns on detector tuning are now
clearly diminishing (UAVDT M1101 remains the outlier at -18.0). Remaining gap
is dominated by detector quality on hard/OOD sequences — not MSFP (~1% per
the paper's own ablation) and not tracker logic.

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
