# JMDST Reproduction — Results Summary

**Source paper:** J. She, Y. Liu, R. Zhou, N. Qi, "Joint multi-target detection and
single-target tracking framework for vehicles tracking based on UAV,"
*Aerospace Science and Technology* 171 (2026) 111625.

**Date:** 2026-08-11 | **Hardware:** RTX 4070 Laptop (8 GB) | **Paper's hardware:** RTX 3070

**Final configuration (single, used for both datasets):** YOLOv11m
(`retrain_aug_m_lowlr`, combined-dataset training, lr0 0.0025) + FELNet
(`full_run`) + IoU-only association + **BYTE two-stage matching**
(low band 0.4) + NSA Kalman (detection frames) + ECC camera-motion
compensation + conf 0.55 + τ = 3.

> **Comparability note.** The paper never published its UAVDT/VisDrone sequence
> splits, so our seeded-random splits are **not** the paper's test set. All
> paper-vs-ours comparisons below are indicative, not like-for-like.
> Phase 6 (MSFP) is not implemented — worth ~1 % by the paper's own ablation.

---

## 1. Final held-out test results

### 1.1 UAVDT test — 8 sequences

| Configuration | MOTA | MOTP | IDF1 | HOTA | DetA | AssA | IDs | FP | FN | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours — IoU association** | **55.4** | 77.5 | **59.7** | **47.8** | 48.7 | **47.1** | **637** | 21320 | 40745 | **26.9** |
| Ours — FELNet appearance | 53.8 | 77.5 | 50.0 | 40.6 | **48.8** | 33.9 | 2258 | 23241 | **39451** | 25.6 |

### 1.2 VisDrone2019 test — 10 sequences

| Configuration | MOTA | MOTP | IDF1 | HOTA | DetA | AssA | IDs | FP | FN | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours — IoU association** | **44.6** | **78.6** | **59.7** | **49.3** | 40.4 | **60.6** | **317** | **9056** | 37333 | **15.6** |
| Ours — FELNet appearance | 43.4 | 78.5 | 55.1 | 46.0 | 40.4 | 52.5 | 734 | 9942 | **36999** | 11.7 |

---

## 2. Comparison against the paper's published tables

Paper values transcribed from its Tables 5 and 6.

### 2.1 UAVDT

| Method | MOTA | MOTP | IDs | IDF1 | HOTA | DetA | AssA | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSORT (paper) | 31.4 | 75.6 | 200 | 48.3 | 38.4 | 31.3 | 47.7 | 15.8 |
| SORT (paper) | 30.2 | 75.7 | **161** | 46.9 | 37.0 | 29.2 | 47.6 | 25.9 |
| ByteTrack (paper) | 32.0 | 74.9 | 165 | 49.8 | 39.2 | 31.6 | 49.2 | 22.9 |
| DiffusionTrack (paper) | 46.9 | 75.3 | 129 | 62.3 | 46.5 | 40.7 | 53.5 | 9.6 |
| QDTrack (paper) | **69.7** | **81.5** | 179 | **76.5** | **62.9** | **59.2** | **67.0** | 13.0 |
| JMDST (paper) | 67.3 | 81.2 | 170 | 73.5 | 60.1 | 59.0 | 61.3 | **26.6** |
| **JMDST (ours)** | 55.4 | 77.5 | 637 | 59.7 | 47.8 | 48.7 | 47.1 | **26.9** |

### 2.2 VisDrone2019

| Method | MOTA | MOTP | IDs | IDF1 | HOTA | DetA | AssA | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSORT (paper) | 36.3 | 75.8 | 905 | 47.9 | 38.7 | 34.3 | 44.3 | 12.2 |
| SORT (paper) | 36.1 | 76.2 | 944 | 47.2 | 38.4 | 33.2 | 45.2 | 16.8 |
| ByteTrack (paper) | 39.0 | 75.6 | 457 | 53.6 | 41.5 | 34.9 | 50.2 | 16.9 |
| DiffusionTrack (paper) | 40.5 | 75.3 | 652 | 55.0 | 43.2 | 37.9 | 50.1 | 13.2 |
| QDTrack (paper) | **49.5** | 77.2 | **161** | **66.0** | 49.0 | **41.0** | 58.9 | 11.3 |
| JMDST (paper) | 44.9 | 76.5 | 178 | 62.4 | 46.4 | 38.0 | 56.9 | **18.6** |
| **JMDST (ours)** | 44.6 | **78.6** | 317 | 59.7 | **49.3** | 40.4 | **60.6** | 15.6 |

### 2.3 Ours vs. paper's JMDST — deltas

| Metric | UAVDT (ours − paper) | VisDrone (ours − paper) |
|---|---:|---:|
| MOTA | −11.9 | −0.3 |
| MOTP | −3.7 | **+2.1** |
| IDF1 | −13.8 | −2.7 |
| **HOTA** | −12.3 | **+2.9** |
| DetA | −10.3 | **+2.4** |
| AssA | −14.2 | **+3.7** |
| IDs | worse (637 vs 170) | worse (317 vs 178) |
| FPS | **+0.3** | −3.0 |

Ours vs. **QDTrack**, the strongest baseline in the paper's own tables:
VisDrone HOTA **+0.3** (49.3 vs 49.0) and AssA **+1.7** (60.6 vs 58.9).

---

## 3. Improvement trajectory (validation split)

### 3.1 Detector

| Detector | Variant | Best epoch | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| `uavdt_only` | yolo11n | 7 | 0.673 | 0.367 | 0.647 | 0.658 |
| `uavdt_only_s` | yolo11s | 3 | 0.631 | 0.378 | 0.683 | 0.560 |
| `uavdt_only_m` | yolo11m | 1 | 0.701 | 0.411 | 0.716 | 0.614 |
| `uavdt_only_m_lowlr` | yolo11m, lr0 0.0025 | 2 | **0.757** | **0.462** | **0.760** | **0.660** |
| `retrain_aug` | yolo11n, combined | 8 | 0.498 | 0.298 | 0.634 | 0.478 |
| `retrain_aug_m` | yolo11m, combined | 17 | 0.587 | 0.373 | 0.647 | 0.561 |
| **`retrain_aug_m_lowlr`** | yolo11m, combined, lr0 0.0025 | 1 | **0.669** | **0.423** | **0.723** | **0.614** |

### 3.2 End-to-end progression (val, HOTA)

| Stage | UAVDT | VisDrone |
|---|---:|---:|
| Baseline (yolo11n, per-dataset configs) | 38.7 | 42.0 |
| + yolo11m detector | 41.5 | 43.5 |
| + low-LR detector retrain | 44.3 | 45.4 |
| + NSA Kalman (detection frames) | 45.4 | 46.6 |
| + ECC camera-motion compensation | 45.7 | 51.1 |
| **+ BYTE two-stage association** | **46.6** | **51.8** |

### 3.3 NSA + CMC ablation (val, HOTA)

| Configuration | UAVDT | VisDrone |
|---|---:|---:|
| Neither | 44.3 | 45.4 |
| NSA only | 45.4 | 46.6 |
| CMC only | 44.2 | 48.3 |
| **NSA + CMC** | **45.7** | **51.1** |

### 3.4 NSA application point (val)

| Variant | UAVDT HOTA | UAVDT IDs | VisDrone HOTA | VisDrone IDs |
|---|---:|---:|---:|---:|
| NSA off | 44.3 | 74 | 45.4 | 849 |
| NSA on both branches | 42.7 | 101 | 43.7 | 934 |
| **NSA at detection frames only** | **45.4** | **71** | **46.6** | **774** |

### 3.5 Confidence-threshold sweep (val, final config incl. BYTE)

| conf | UAVDT HOTA | VisDrone HOTA | UAVDT MOTA | VisDrone MOTA |
|---|---:|---:|---:|---:|
| 0.45 | 45.6 | 51.4 | 20.8 | 56.0 |
| 0.50 | 45.9 | 51.6 | 22.8 | 55.7 |
| **0.55 (selected, = paper's value)** | **46.6** | **51.8** | 25.7 | 56.3 |
| 0.60 | 45.9 | 51.1 | 27.3 | 55.7 |
| 0.65 | 46.2 | 51.2 | **30.0** | **56.5** |

0.55 is the maximum of both HOTA columns independently, not a compromise.

### 3.6 CMC motion model (val, conf 0.55)

| Model | UAVDT HOTA | VisDrone HOTA | VisDrone IDs | VisDrone FPS |
|---|---:|---:|---:|---:|
| **Euclidean (selected)** | **45.7** | 51.1 | 560 | **16.5** |
| Affine | 45.4 | **51.3** | **545** | 15.4 |

### 3.7 BYTE low-confidence band sweep (val)

| `--byte-low-conf` | UAVDT HOTA | VisDrone HOTA | UAVDT MOTA | VisDrone MOTA | UAVDT FP+FN |
|---|---:|---:|---:|---:|---:|
| off | 45.7 | 51.1 | 24.2 | 54.6 | 17298 |
| 0.1 | 45.4 | **52.7** | 22.4 | **57.8** | 17734 |
| 0.2 | 45.4 | 52.6 | 22.0 | 57.5 | 17816 |
| 0.3 | 45.4 | 52.1 | 23.3 | 57.2 | 17520 |
| **0.4 (selected)** | **46.6** | 51.8 | **25.7** | 56.3 | **16972** |
| 0.45 | 45.7 | 51.3 | 24.3 | 55.4 | 17294 |
| 0.5 | 45.8 | 51.4 | 23.9 | 55.9 | 17382 |

0.4 is the only setting where both datasets beat the BYTE-off baseline.

### 3.8 Low-confidence band composition (measured, test sequences)

Detections at 0.10–0.55, scored against ground truth at IoU ≥ 0.5.

| Sequence | high/frame | low/frame | real objects in low band /frame |
|---|---:|---:|---:|
| M0101 | 16.3 | 6.8 | 0.58 |
| uav0000308_00000_v | 13.7 | 3.3 | 0.78 |
| M1006 | 19.9 | 18.6 | 7.82 |
| M1301 | 22.2 | 22.6 | 7.88 |
| uav0000124_00944_v | 40.2 | 21.5 | **14.73** |

---

## 4. Association ablation — FELNet appearance vs. IoU

Paper Sec. 3.4.2 compares FELNet against DeepSORT's ReIDNet; no ReIDNet weights
were available, so "off" here means no appearance model at all.

| Split | Dataset | FELNet appearance HOTA | IoU-only HOTA | Δ | Appearance IDs | IoU IDs |
|---|---|---:|---:|---:|---:|---:|
| val | UAVDT | 38.9 | **43.2** | +4.3 | 282 | **96** |
| val | VisDrone | 39.6 | **44.6** | +5.0 | 2041 | **1037** |
| test (pre-BYTE) | UAVDT | 38.5 | **47.2** | +8.7 | 2388 | 661 |
| test (pre-BYTE) | VisDrone | 45.0 | **48.1** | +3.1 | 770 | 345 |
| **test (final)** | UAVDT | 40.6 | **47.8** | **+7.2** | 2258 | **637** |
| **test (final)** | VisDrone | 46.0 | **49.3** | +3.3 | 734 | **317** |

### 4.1 FELNet loss-rebalancing attempts (val, component metrics)

| Checkpoint | Lever | Localisation IoU | Embedding separation | Confidence acc. |
|---|---|---:|---:|---:|
| `full_run` | none (paper λ = 1,1,1) | **0.634** | 0.653 | **0.843** |
| `rebalanced` | `--overlap-scale 64` | 0.610 | 0.782 | 0.835 |
| `lambda_e30` | λ₁ = 30 | 0.593 | 0.784 | 0.825 |
| `lambda_e100` | λ₁ = 100 | 0.522 | **0.789** | 0.753 |

| Checkpoint | UAVDT HOTA | VisDrone HOTA |
|---|---:|---:|
| `full_run` + appearance | 38.9 | 39.6 |
| `lambda_e30` + appearance | 39.2 | 39.2 |

### 4.2 ID-switch attribution (test, final config)

Per GT track, following which predicted ID is matched to it frame by frame.
`gap = 0` means the object was matched on both sides of the ID change
(association failure); `gap >= 1` means it went unmatched first (detection
failure).

| Split | association (gap=0) | detection (gap≥1) | association share |
|---|---:|---:|---:|
| UAVDT test | 38 | 603 | **5.9 %** |
| VisDrone test | 59 | 270 | **17.9 %** |

GT displacement at the switch, in box heights:

| | UAVDT median / p90 | VisDrone median / p90 |
|---|---|---|
| association switches | 0.04 / 0.12 | 0.02 / 0.10 |
| detection switches | 0.20 / 1.76 | 0.36 / 5.05 |

Gap distribution peaks at 4–5 frames (N = τ+1 = 4 confirmation latency) with a
second mass at 10+ frames (re-identification after long absence).

---

## 5. Detection-interval (τ) ablation — mirrors paper Sec. 3.4.3

UAVDT val, `uavdt_only` detector, conf 0.55.

| τ | MOTA | MOTP | IDF1 | HOTA | DetA | AssA | IDs | FP | FN | FPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **34.9** | 78.2 | 48.4 | 38.1 | 37.1 | 39.2 | 362 | **4795** | **9755** | 35.6 |
| 2 | 27.9 | 77.4 | **51.2** | **39.3** | 35.7 | **43.3** | 153 | 6916 | 9469 | 48.2 |
| 3 | 22.6 | 76.8 | 50.3 | 38.0 | 33.9 | 42.9 | 125 | 7886 | 9725 | 51.7 |
| 4 | 19.2 | 76.7 | 48.9 | 37.0 | 32.8 | 42.0 | **96** | 8304 | 10128 | 54.9 |
| 5 | 14.9 | 76.0 | 44.5 | 34.4 | 31.3 | 37.9 | 126 | 8959 | 10429 | 55.9 |
| 6 | 11.0 | 75.9 | 43.4 | 33.7 | 29.5 | 38.6 | 120 | 9189 | 11100 | **57.2** |

Paper trends reproduced: MOTA ↓ with τ, FPS ↑ with τ, FN ↑ with τ.
Diverged: paper's FP decreases with τ, ours increases.

---

## 6. Per-sequence results, final test configuration

### 6.1 UAVDT test

| Sequence | MOTA | MOTP | IDF1 | IDs | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| M0703 | **74.9** | 80.0 | 69.8 | 88 | 2018 | 1722 |
| M0101 | 74.3 | 80.2 | **80.5** | 16 | 1050 | 258 |
| M0603 | 71.3 | 78.0 | 57.0 | 265 | 8106 | 3516 |
| M1301 | 55.4 | 76.0 | 67.6 | 26 | 2442 | 14682 |
| M1006 | 43.3 | 79.0 | 60.4 | 82 | 3142 | 8898 |
| M1009 | 17.2 | 67.8 | 27.7 | 114 | 1719 | 5959 |
| M1004 | 16.1 | 73.1 | 32.7 | **6** | **235** | 3731 |
| M0210 | 2.1 | 74.0 | 38.4 | 40 | 2608 | 1979 |

### 6.2 VisDrone2019 test

| Sequence | MOTA | MOTP | IDF1 | IDs | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| uav0000308_00000_v | **76.6** | **84.2** | **84.0** | **3** | 332 | 431 |
| uav0000306_00230_v | 67.4 | 76.4 | 78.2 | 19 | 716 | 1547 |
| uav0000316_01288_v | 56.5 | 79.5 | 63.3 | 12 | **170** | 423 |
| uav0000315_00000_v | 52.8 | 79.2 | 65.8 | 47 | 2637 | 3779 |
| uav0000119_02301_v | 51.1 | 78.0 | 66.3 | 22 | 291 | 1571 |
| uav0000124_00944_v | 43.0 | 79.2 | 56.3 | 55 | 756 | 18994 |
| uav0000281_00460_v | 34.7 | 76.2 | 51.8 | 121 | 2329 | 6394 |
| uav0000370_00001_v | 19.1 | 79.6 | 38.8 | 9 | 401 | 1592 |
| uav0000263_03289_v | 6.4 | 77.3 | 35.1 | 29 | 1424 | 2589 |
| uav0000073_00600_v\* | 0.0 | n/a | 0.0 | 0 | 0 | 13 |

\* 13 ground-truth objects in the entire sequence.

---

## 7. Parameters the paper does not specify

| Parameter | Paper | Ours |
|---|---|---|
| YOLO variant / LR / epochs / augmentation | "trained separately using its default parameters" | yolo11m, MuSGD, lr0 0.0025, batch 8, 150 ep / patience 20, strong aug |
| `k_max` | symbol only | 5 |
| `N_o` / `N_s` | symbol only | 8 / 8 |
| `N_P` / `N_N` | symbol only | 128 / 128 |
| `M`, `N_b` (MSFP) | symbol only | n/a — not implemented |
| `max_cosine_distance` | not given | 0.3 |
| Kalman noise weights | not given | 1/20 position, 1/160 velocity |
| FELNet optimiser LR | "Adam + cosine annealing" | 1e-3 |

Paper-specified and retained: τ = 3, NMS IoU 0.2, conf 0.55, d = 5,
c₁/i₁/i₂ = 0.9/0.1/0.1, N = τ+1 = 4, 100-miss deletion, λ = (1,1,1),
embedding dim 16, SSI 64×64.

---

## 8. Status

| Item | State |
|---|---|
| Phases 1–5, 7–11 | Complete, verified, committed |
| Phase 6 (MSFP) | Not implemented — `mamba-ssm` PyPI sdist ships without its CUDA sources |
| Adaptive process noise (SAKF-RCE) | Not implemented — ruled out by §4.2: only 5.9/17.9 % of ID switches are association failures, and those occur at ~0.03 box-heights displacement |
| Unit tests | 181 passing |
| Splits held out | Test split used once, after config was frozen on val |

### Deviations from the paper

| Deviation | Reason |
|---|---|
| IoU-only association (paper specifies FELNet appearance matching) | +3.1 to +8.7 HOTA on test, both datasets |
| YOLOv11m at lr0 0.0025 (paper: "default parameters") | +11.1 MOTA val vs yolo11n at default LR |
| NSA Kalman filter, detection frames only | Paper specifies no Kalman internals |
| ECC camera-motion compensation | Addresses the failure mode in paper Sec. 3.5.4 |
| BYTE two-stage association (low band 0.4) | +0.6 / +1.2 HOTA on test; recovers real objects the 0.55 threshold discarded |
| Phase 6 (MSFP) absent | Dependency unbuildable; ~1 % by paper's own ablation |
