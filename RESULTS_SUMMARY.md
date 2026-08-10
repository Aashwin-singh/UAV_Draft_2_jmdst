# JMDST Reproduction — Results Summary

**Source paper:** J. She, Y. Liu, R. Zhou, N. Qi, "Joint multi-target detection and
single-target tracking framework for vehicles tracking based on UAV,"
*Aerospace Science and Technology* 171 (2026) 111625.

**Date:** 2026-08-10 | **Hardware:** RTX 4070 Laptop (8 GB) | **Paper's hardware:** RTX 3070

**Final configuration (single, used for both datasets):** YOLOv11m
(`retrain_aug_m_lowlr`, combined-dataset training, lr0 0.0025) + FELNet
(`full_run`) + IoU-only association + NSA Kalman (detection frames) +
ECC camera-motion compensation + conf 0.55 + τ = 3.

> **Comparability note.** The paper never published its UAVDT/VisDrone sequence
> splits, so our seeded-random splits are **not** the paper's test set. All
> paper-vs-ours comparisons below are indicative, not like-for-like.
> Phase 6 (MSFP) is not implemented — worth ~1 % by the paper's own ablation.

---

## 1. Final held-out test results

### 1.1 UAVDT test — 8 sequences

| Configuration | MOTA | MOTP | IDF1 | HOTA | DetA | AssA | IDs | FP | FN | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours — IoU association** | **54.3** | 77.6 | **59.0** | **47.2** | **48.1** | **46.5** | **661** | 21003 | 42604 | 23.6 |
| Ours — FELNet appearance | 52.2 | 77.6 | 46.9 | 38.5 | 47.9 | 31.1 | 2388 | 23057 | 41691 | 26.0 |

### 1.2 VisDrone2019 test — 10 sequences

| Configuration | MOTA | MOTP | IDF1 | HOTA | DetA | AssA | IDs | FP | FN | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours — IoU association** | **42.7** | **78.7** | **58.2** | **48.1** | 39.3 | **59.1** | **345** | 9464 | 38523 | 13.0 |
| Ours — FELNet appearance | 42.5 | 78.6 | 53.6 | 45.0 | **39.6** | 51.4 | 770 | 9551 | 38149 | 11.7 |

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
| **JMDST (ours)** | 54.3 | 77.6 | 661 | 59.0 | 47.2 | 48.1 | 46.5 | 23.6 |

### 2.2 VisDrone2019

| Method | MOTA | MOTP | IDs | IDF1 | HOTA | DetA | AssA | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSORT (paper) | 36.3 | 75.8 | 905 | 47.9 | 38.7 | 34.3 | 44.3 | 12.2 |
| SORT (paper) | 36.1 | 76.2 | 944 | 47.2 | 38.4 | 33.2 | 45.2 | 16.8 |
| ByteTrack (paper) | 39.0 | 75.6 | 457 | 53.6 | 41.5 | 34.9 | 50.2 | 16.9 |
| DiffusionTrack (paper) | 40.5 | 75.3 | 652 | 55.0 | 43.2 | 37.9 | 50.1 | 13.2 |
| QDTrack (paper) | **49.5** | 77.2 | **161** | **66.0** | 49.0 | **41.0** | 58.9 | 11.3 |
| JMDST (paper) | 44.9 | 76.5 | 178 | 62.4 | 46.4 | 38.0 | 56.9 | **18.6** |
| **JMDST (ours)** | 42.7 | **78.7** | 345 | 58.2 | **48.1** | 39.3 | **59.1** | 13.0 |

### 2.3 Ours vs. paper's JMDST — deltas

| Metric | UAVDT (ours − paper) | VisDrone (ours − paper) |
|---|---:|---:|
| MOTA | −13.0 | −2.2 |
| MOTP | −3.6 | **+2.2** |
| IDF1 | −14.5 | −4.2 |
| **HOTA** | −12.9 | **+1.7** |
| DetA | −10.9 | **+1.3** |
| AssA | −14.8 | **+2.2** |
| IDs | worse (661 vs 170) | worse (345 vs 178) |
| FPS | −3.0 | −5.6 |

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
| **+ ECC camera-motion compensation** | **45.7** | **51.1** |

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

### 3.5 Confidence-threshold sweep (val, final config)

| conf | UAVDT HOTA | VisDrone HOTA | UAVDT MOTA | VisDrone MOTA |
|---|---:|---:|---:|---:|
| 0.45 | 45.1 | **51.3** | 19.3 | **55.2** |
| **0.55 (selected, = paper's value)** | 45.7 | 51.1 | 24.2 | 54.6 |
| 0.65 | **45.9** | 49.4 | **30.7** | 52.9 |

### 3.6 CMC motion model (val, conf 0.55)

| Model | UAVDT HOTA | VisDrone HOTA | VisDrone IDs | VisDrone FPS |
|---|---:|---:|---:|---:|
| **Euclidean (selected)** | **45.7** | 51.1 | 560 | **16.5** |
| Affine | 45.4 | **51.3** | **545** | 15.4 |

---

## 4. Association ablation — FELNet appearance vs. IoU

Paper Sec. 3.4.2 compares FELNet against DeepSORT's ReIDNet; no ReIDNet weights
were available, so "off" here means no appearance model at all.

| Split | Dataset | FELNet appearance HOTA | IoU-only HOTA | Δ | Appearance IDs | IoU IDs |
|---|---|---:|---:|---:|---:|---:|
| val | UAVDT | 38.9 | **43.2** | +4.3 | 282 | **96** |
| val | VisDrone | 39.6 | **44.6** | +5.0 | 2041 | **1037** |
| test | UAVDT | 38.5 | **47.2** | **+8.7** | 2388 | **661** |
| test | VisDrone | 45.0 | **48.1** | +3.1 | 770 | **345** |

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
| M0101 | **75.3** | 80.2 | **80.9** | 16 | 977 | 280 |
| M0603 | 74.1 | 78.1 | 57.9 | 263 | 6742 | 3750 |
| M0703 | 73.8 | 80.0 | 68.5 | 98 | 1888 | 2001 |
| M1301 | 51.4 | 76.1 | 65.1 | 37 | 3328 | 15351 |
| M1006 | 39.0 | 79.1 | 58.6 | 88 | 3580 | 9358 |
| M1009 | 16.8 | 67.9 | 31.2 | 109 | 1709 | 6014 |
| M1004 | 14.3 | 73.4 | 30.5 | **6** | **247** | 3806 |
| M0210 | 2.2 | 73.8 | 34.7 | 44 | 2532 | 2044 |

### 6.2 VisDrone2019 test

| Sequence | MOTA | MOTP | IDF1 | IDs | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| uav0000308_00000_v | **74.0** | **84.2** | **80.7** | 6 | 341 | 503 |
| uav0000306_00230_v | 63.2 | 76.5 | 75.1 | 20 | 907 | 1652 |
| uav0000316_01288_v | 57.2 | 79.4 | 65.5 | 18 | 147 | 430 |
| uav0000119_02301_v | 50.6 | 78.0 | 65.4 | 23 | 269 | 1608 |
| uav0000315_00000_v | 46.2 | 79.3 | 62.8 | 51 | 3191 | 4126 |
| uav0000124_00944_v | 42.2 | 79.3 | 55.6 | 58 | 842 | 19165 |
| uav0000281_00460_v | 36.7 | 76.2 | 51.0 | 130 | 1745 | 6704 |
| uav0000370_00001_v | 18.0 | 79.1 | 37.2 | 10 | 404 | 1616 |
| uav0000263_03289_v | −0.8 | 77.6 | 33.2 | 29 | 1618 | 2706 |
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
| Unit tests | 112 passing |
| Splits held out | Test split used once, after config was frozen on val |

### Deviations from the paper

| Deviation | Reason |
|---|---|
| IoU-only association (paper specifies FELNet appearance matching) | +3.1 to +8.7 HOTA on test, both datasets |
| YOLOv11m at lr0 0.0025 (paper: "default parameters") | +11.1 MOTA val vs yolo11n at default LR |
| NSA Kalman filter, detection frames only | Paper specifies no Kalman internals |
| ECC camera-motion compensation | Addresses the failure mode in paper Sec. 3.5.4 |
| Phase 6 (MSFP) absent | Dependency unbuildable; ~1 % by paper's own ablation |
