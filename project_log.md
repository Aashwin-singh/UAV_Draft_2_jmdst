# Project Log

Running status for the JMDST reproduction, for teammates picking up work in
parallel. See `PROJECT_CONTEXT.md` for the full technical spec (Section A)
and detailed phase roadmap (Section B) — this file is just a quick "where
are we" snapshot, updated as phases complete.

## FINAL HELD-OUT TEST RESULTS (2026-07-28)

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
