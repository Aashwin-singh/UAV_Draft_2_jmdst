# JMDST Reproduction — Project Context

This document is the persistent reference for this capstone project. It combines
(1) the exact technical specification extracted from the source paper, and
(2) the current implementation status. Treat Section A as ground truth for
correctness — implement against it directly rather than re-deriving details
from memory or from re-reading the PDF each session.

**Source paper:** J. She, Y. Liu, R. Zhou, N. Qi, "Joint multi-target detection
and single-target tracking framework for vehicles tracking based on UAV,"
*Aerospace Science and Technology*, 171 (2026) 111625.

**Prime directive:** Reproduce the paper faithfully first. Do not simplify the
architecture or substitute easier alternatives (e.g. no YOLO+ByteTrack shortcut).
Implement and verify one module at a time. Commit to git after every verified
milestone. Do not proceed to the next phase until the current one is verified.

---

# A. Formal specification (implement exactly)

## A.1 Overall system

JMDST has two branches that alternate on a fixed schedule, controlled by
detection interval **τ = 3**:

- At `t = k·τ` (k = 0,1,2,...): **detection branch** runs.
- At all other `t`: **tracking branch** runs.

```
Detection branch (every τ frames):
    YOLOv11 → detections → SSI crop → FELNet → embeddings
    → predict existing tracklets' embeddings via MSFP, boxes via Kalman
    → cascade match (embeddings) → IoU match (remaining) → update trajectories 𝒯

Tracking branch (all other frames):
    Kalman predict box for each tracked target → crop SSI → FELNet
    (localization + embedding) → update trajectory
```

### Algorithm 1 — application routine (paper's exact pseudocode)

```
Input: frame sequence I_o^t, detection interval τ, YOLOv11, FELNet, MSFP
Output: confirmed-state tracklets updated at time t

𝒯^0 = ∅, t = 0
for each frame I_o^t:
    if t = k·τ:                                   # DETECTION BRANCH
        1. YOLOv11 detects N_D^t targets → boxes {B_j^t}
        2. Crop N_D^t SSIs from {B_j^t}
        3. FELNet encodes SSIs → embeddings {E_j^t}
        4. For each tracklet in 𝒯^{t-1}: predict Ê_{T,i}^t via MSFP (Eq. 8)
        5. For each tracklet in 𝒯^{t-1}: predict B̂_{T,i}^t via Kalman filter
        6. Cascade match: 𝒯_c^{t-1} vs D^t using {Ê_{T,i}^t} and {E_j^t}
        7. IoU assignment: 𝒯_uc^{t-1}, unmatched-from-cascade, D_um^t
        8. Update 𝒯^t and Kalman filters
    else:                                          # TRACKING BRANCH
        1. For each tracked target: predict B̂_{T,i_d}^t via Kalman filter
        2. Crop SSIs from predicted boxes
        3. FELNet encodes SSIs → embeddings {E_{i_d}^t} and overlap vectors {o_{i_d}^t}
        4. Convert overlap vectors → refined boxes {B_{i_d}^t} (Eq. 2)
        5. Update 𝒯^t and Kalman filters
    Output confirmed-state tracklets in 𝒯^t
    t = t + 1
```

## A.2 SSI (small-sized image) cropping

Fixed output size: **64 × 64**. Crop is centered on the target's (predicted)
bounding box center. Two formulas, chosen by aspect-ratio extremity:

- **Default** (near-square targets): crop side length =
  `sqrt((1.3w + 0.3h) * (1.3h + 0.3w))`
- **Asymmetric fallback** (avoids the target center landing ambiguously near
  multiple anchor-box centers when w/h or h/w is extreme):
  `min(1.9w, 1.9h, default_formula)`

Training-time robustness additions:
- Crop position randomly translated by up to **16 px**.
- SSI further augmented via shifting, flipping, scaling.

## A.3 FELNet architecture (Feature Encoding and Location Network)

Darknet-53-derived but channel-reduced and depth-reduced (target occupies
limited info within a 64×64 crop, so heavy channels are unnecessary). Input
64×64×3. Backbone output is a **2×2 grid of 4 anchor boxes**, each anchor
covering a 32×32 region of the input SSI.

| Stage | Layers | Channels | Kernel/Stride | Output |
|---|---|---|---|---|
| Stem | Conv, Conv | 16, 32 | 3×3/1, 3×3/2 | 64×64 → 32×32 |
| 1× block | Conv, Conv, Residual | 16, 32 | 3×3/1, 3×3/1 | 32×32 |
| Downsample | Conv | 64 | 3×3/2 | 16×16 |
| 2× block | Conv, Conv, Residual | 32, 64 | 3×3/1, 3×3/1 | 16×16 |
| Downsample | Conv | 128 | 3×3/2 | 8×8 |
| 6× block | Conv, Conv, Residual | 64, 128 | 3×3/1, 3×3/1 | 8×8 |
| Downsample | Conv | 128 | 3×3/2 | 4×4 |
| 6× block | Conv, Conv, Residual | 128, 128 | 3×3/1, 3×3/1 | 4×4 |
| Downsample | Conv | 128 | 3×3/2 | 2×2 |
| 4× block | Conv, Conv, Residual | 128, 128 | 3×3/1, 3×3/1 | 2×2 |

Three **fully-convolutional heads** off the final 2×2×128 feature map
(replacing the original FC layer):

| Head | Layers (1×1 convs) | Output channels | Output shape |
|---|---|---|---|
| `o` (overlap) | 64→32→16→4 | 4 | 2×2×4 |
| `E` (embedding) | 64→32→16→16 | 16 | 2×2×16 |
| `C` (confidence) | 64→32→16→1 | 1 | 2×2×1 |

Anchor assignment during training: if a GT box's center falls inside an
anchor's 32×32 region, that anchor is responsible for it. If multiple GT
boxes cover the same anchor center, use the one with smallest overlap-vector
deviation. Anchors with no covering target get overlap vector = 0 and
confidence label = 0.

**Overlap vector** given SSI box `B_S = [l1,u1,w1,h1]` and object box (relative
to the original frame) `B_H = [l2,u2,w2,h2]`:

```
o1 = max(0, l1 + w1 - l2)      # left overlap
o2 = max(0, l2 + w2 - l1)      # right overlap
o3 = max(0, u1 + h1 - u2)      # top overlap
o4 = max(0, u2 + h2 - u1)      # bottom overlap
```

Inverse (recover `B_H` from `B_S` and predicted `o`):
```
l2 = l1 + w1 - o1
u2 = u1 + h1 - o3
w2 = o1 + o2 - w1
h2 = o3 + o4 - h1
```

**Inference output selection**: FELNet always emits 4 anchor outputs. Convert
the tracking branch's predicted box into an overlap vector relative to the
crop; compute Euclidean distance to each anchor's predicted overlap vector;
among anchors with **confidence > 0.9**, select the one with smallest distance.

## A.4 FELNet training

Sampling: pick random interval `k ∈ [1, k_max]`, sample `N_o` frames at that
interval, crop `N_s` SSIs per frame from GT object labels (with
shift/flip/scale augmentation).

Losses (all RMSE-style):

```
L_o = sqrt( Σ_n Σ_a (ō_{n,a} - ô_{n,a})²  /  (4 · N_o · N_s) )      # overlap, a=1..4 anchors
L_C = sqrt( Σ_n Σ_a (C̄_{n,a} - Ĉ_{n,a})²  /  (4 · N_o · N_s) )      # confidence
L_E = sqrt( Σ (FCC(Ê_i^ta, Ê_j^tb) - T(i,j))²  /  N_pair )          # embedding
    where T(i,j) = +1 if same target (i==j), -1 otherwise
    N_pair = N_P (positive pairs) + N_N (negative pairs), sampled from
    all N_o × N_s SSIs' embeddings. FCC = fast cross-correlation similarity.

L_total = λ1·L_E + λ2·L_o + λ3·L_C     (λ1=λ2=λ3=1.0 by default)
```

### Algorithm 2 — FELNet training routine

```
Initialize FELNet
while epochs remain:
    k ~ Uniform[1, k_max]
    sample N_o frames at interval k
    for each frame: randomly crop N_s SSIs from GT object labels
    augment SSIs (shift, flip, scale)
    forward SSIs through FELNet → {Ê, ô, Ĉ}
    convert GT boxes to GT overlap vectors {ō} via Eq. A.3
    sample N_P positive / N_N negative embedding pairs
    compute T(i,j) labels for the pairs
    compute L_total = λ1·L_E + λ2·L_o + λ3·L_C
    backprop, optimizer step (Adam + cosine annealing LR)
```

## A.5 MSFP (Mamba Sequence Feature Prediction)

Purpose: predict a target's *current* feature embedding from its historical
embedding sequence, to smooth appearance noise/occlusion and reduce ID
switches — used to make cascade matching more robust at detection frames.

```
{Ê_{T,i}^t, ..., Ê_{T,i}^{t-M}} = MSFP({E_{T,i}^{t-1}, ..., E_{T,i}^{t-M-1}})
```

Structure: **3 stacked Mamba blocks**, each block = Linear → (Conv → SSM →
gating with σ) → Linear, matching the standard Mamba block layout (see
Fig. 4 of the paper — this is the vanilla Mamba block from Gu & Dao 2024,
not a custom variant).

Loss (RMSE over a batch of feature sequences):
```
L_mamba = sqrt( Σ_k Σ_m (Ē_{i_k}^{t-m} - Ê_{i_k}^{t-m})²  /  (M · N_b) )
```

**Do not implement/train MSFP until `mamba-ssm` installs successfully.**
This is Phase 6 and comes after FELNet is fully trained, since MSFP consumes
FELNet-generated embeddings.

## A.6 Trajectory association & update (modified DeepSORT)

State machine: **tentative → confirmed → deleted**.
- Confirm threshold: `N = τ + 1` consecutive detections (paper explicitly
  derives this: `N ≤ τ` raises false positives, `N > τ+1` raises false
  negatives; `N = τ+1` balances both).

At detection frames (`t = kτ`):
1. Split `𝒯^{t-1}` into confirmed `𝒯_c^{t-1}` and unconfirmed `𝒯_uc^{t-1}`.
2. **Cascade matching**: `𝒯_c^{t-1}` vs `D^t` using embedding similarity
   (predicted `Ê_{T,i}^t` from MSFP vs detected `E_j^t`) →
   matched (`𝒯_{c,m}^{t-1}`, `D_m^t`), unmatched tracklets `𝒯_{c,um}^{t-1}`,
   unmatched detections `D_um^t`.
3. **IoU assignment**: cost matrix over `𝒯_uc^{t-1} ∪ 𝒯_{c,um}^{t-1}` (only
   those with time-since-update ≤ τ+1) vs `D_um^t` →
   matched (`𝒯_m'^{t-1}`, `D_m'^t`), unmatched tracklets `𝒯_um'^{t-1}`,
   unmatched detections `D_um'^t`.
4. Update matched tracklets (`𝒯_{c,m}^{t-1}` and `𝒯_m'^{t-1}`) via Kalman filter.
5. Delete unmatched **unconfirmed** tracklets immediately.
6. Delete unmatched **confirmed** tracklets after **100** consecutive misses.
7. Initialize new tracklets from `D_um'^t`.

### Missed-detection robustness (expanded tracking set)

At non-detection frames, don't just track the last detection-frame's targets —
**also** run tracking-branch localization on confirmed tracklets from
`𝒯_um'^{t-1}` (i.e. those that went unmatched at the last detection cycle) if
they meet **all** of:
- state is confirmed,
- time since last update ≤ τ + 1,
- distance from image boundary > `d` (excludes targets that likely left frame).

### Output filtering for tracking-branch results (from previously-unmatched trajectories only)

- Discard as false positive if confidence < `c1`.
- Discard as false positive if IoU with **any** current detection box > `i1`.
- If two such outputs have IoU > `i2` with each other, keep only the
  higher-confidence one.

## A.7 Training procedure (three modules, trained separately)

1. **YOLOv11**: trained independently with its own default parameters.
2. **FELNet**: trained independently with Adam optimizer + cosine annealing LR
   schedule (Algorithm 2).
3. **MSFP**: trained *after* FELNet has converged (since it needs FELNet's
   embeddings as ground truth), with Adam + cosine annealing, loss = RMSE
   (`L_mamba`, Eq. A.5).

## A.8 Hyperparameters (from paper's implementation section, Sec. 3.3)

| Parameter | Value | Meaning |
|---|---|---|
| τ | 3 | detection interval (best accuracy/speed trade-off; τ=2 gives best raw MOTA if you need max accuracy) |
| NMS IoU threshold | 0.2 | YOLOv11 post-processing |
| Detection confidence threshold | 0.55 | YOLOv11 post-processing |
| d | 5 | min. distance from image boundary to keep tracking a missed target |
| c1 | 0.9 | confidence floor for tracking-branch outputs |
| i1 | 0.1 | IoU-with-detection ceiling for tracking-branch outputs |
| i2 | 0.1 | IoU ceiling between two unmatched-trajectory outputs before dedup |
| N (confirm threshold) | τ+1 = 4 | consecutive detections to confirm a tracklet |
| Confirmed-tracklet deletion | 100 misses | consecutive unmatched frames before deletion |
| λ1, λ2, λ3 (FELNet loss weights) | 1.0, 1.0, 1.0 | embedding, overlap, confidence |
| Embedding dimension | 16 | FELNet `E` head output |
| SSI size | 64×64 | fixed |

## A.9 Datasets, splits, and metrics

- **VisDrone2019 MOT**: classes {car, van, truck, bus} only. Split: 45 train /
  10 val / 10 test sequences.
- **UAVDT**: class {car} only. 23 of 50 available sequences selected: 12
  train / 3 val / 8 test. Frame resolution 1080×540.
- **MDMT** (used in the paper for main comparisons/ablations, not currently
  in this project's scope per the roadmap, but noted for reference): car
  class only, half of paired sequences used, 24/5/13 train/val/test.
- **Metrics**: MOTA, MOTP, IDs, IDF1, HOTA (= sqrt(DetA × AssA)), DetA, AssA,
  FPS. Formulas are in paper Sec. 3.2 if needed for the eval script (Phase 10).

---

# B. Current implementation status

**Environment** — conda env `jmdst`, Python 3.11.15, PyTorch 2.11.0+cu128,
RTX 4070 Laptop GPU, CUDA verified working via `torch.cuda.is_available()`.
Installed: torch, torchvision, ultralytics, numpy, pandas, Pillow, scipy,
PyYAML, tqdm, opencv-python, motmetrics. **Not installed**: `mamba-ssm` (fails
to install — acceptable for now, MSFP is Phase 6, don't spend time on it yet).

**Repository layout**:
```
_UAV_DRAFT_2/
    agents/
    configs/
    jmdst/
    outputs/
    scripts/
    tests/
    README.md
    requirements.txt
    VisDrone_MOT/      # raw dataset, ideally moves under datasets/
    UAVDT/             # raw dataset, ideally moves under datasets/
```
`.gitignore` excludes datasets, outputs, checkpoints, cache files. First git
commit made after Phase 1 (dataset prep). Commit after every future
milestone, only after it's verified.

**Unified dataset format** (`data/unified/{visdrone,uavdt}/<sequence>/`):
```
seqinfo.json
annotations.jsonl      # one JSON object per line, per frame:
{
    "frame_id": 1,
    "image_path": "...",
    "objects": [
        {"track_id": 1, "class_id": 0, "class_name": "car", "bbox_xywh": [x, y, w, h]}
    ]
}
```

## B.1 Already implemented and working

- **Dataset converters** for VisDrone2019 MOT and UAVDT → unified format above.
- **SSI crop generator** implementing the Sec. A.2 formula, dynamic crop size,
  resize to 64×64.
- **FELNet target generation**: overlap targets, confidence targets, anchor
  assignment (Sec. A.3 rules).
- **PyTorch dataset classes**: Detection Dataset, FELNet Dataset, Tracking
  Sequence Dataset — all PyTorch-ready.
- **Visualization script** for bounding boxes and SSI crops.
- **Synthetic dataset generator** for debugging without real data.
- **Unit tests** (`python -m unittest discover -s tests -v`).
- **Dataset verification script** (`scripts/verify_dataset.py`, added this
  session): walks the unified format, reports sequence/frame/annotation
  counts, class distribution, bbox validity + statistics (width/height/area/
  aspect ratio), missing/corrupted images, duplicate track IDs per frame, and
  simulated SSI crop-size statistics using the Sec. A.2 formula (including how
  many crops would exceed image bounds pre-padding). Outputs
  `outputs/verification/report.json` and `report.md`.

Dataset conversion has completed successfully. No learning model has been
implemented yet — everything above is data infrastructure.

## B.2 Not yet implemented (in roadmap order)

- **Current task**: run `scripts/verify_dataset.py` against the real converted
  data, review the report, fix any flagged issues (invalid boxes, missing/
  corrupted images) before touching YOLO training.
- **Phase 2 — YOLOv11**: unified→YOLO format converter, `dataset.yaml`
  generation, train/val/inference scripts, checkpoint saving, visualization.
  Train with YOLOv11's default parameters (per A.7); apply NMS 0.2 / conf 0.55
  at inference (per A.8).
- **Phase 3 — FELNet architecture**: implement per Table in A.3 exactly
  (channel counts, kernel sizes, output shapes, three-head design).
- **Phase 4 — FELNet training**: implement Algorithm 2 / losses in A.4.
- **Phase 5 — Feature extraction**: run trained FELNet over all unified data,
  save embedding sequences (input for MSFP training).
- **Phase 6 — MSFP**: blocked on `mamba-ssm` installing successfully. 3
  stacked Mamba blocks per A.5. Do not start until Mamba is resolved.
- **Phase 7 — Tracking infrastructure**: Kalman filter, track lifecycle/states
  (tentative/confirmed/deleted).
- **Phase 8 — Modified DeepSORT**: cascade matching (FELNet/MSFP embeddings)
  + IoU matching per A.6, including the missed-detection expansion and output
  filtering rules.
- **Phase 9 — Full JMDST integration**: wire YOLOv11 → SSI crop → FELNet →
  MSFP → modified DeepSORT into Algorithm 1's application routine.
- **Phase 10 — Evaluation**: MOTA/MOTP/IDF1/HOTA/FPS on VisDrone2019 and
  UAVDT test splits (A.9).
- **Phase 11 — Ablations**: without MSFP, varying τ, YOLO-only, standard
  DeepSORT ReID instead of FELNet — mirrors the paper's own ablation study
  design (Sec. 3.4 of the paper) so results are comparable.

---

# C. Working rules for this project

1. Always implement the paper faithfully first (Section A above) — no
   simplified substitutes.
2. Implement and verify one module at a time; don't jump ahead to a later
   phase before the current one works.
3. After every completed phase: verify it, then commit to git.
4. Don't work on Mamba/MSFP until `mamba-ssm` installs cleanly.
5. Code should be production-quality, modular, and well-documented — this is
   a long-term research capstone, not a throwaway script.
