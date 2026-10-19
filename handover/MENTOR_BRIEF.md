# UAV vehicle tracking — integration & testing brief

For the mentor review. One page of substance, then the demo script.

---

## 1. What the system is

Our UAV vehicle-tracking system. It builds on the **JMDST** framework (She et
al., *Aerospace Science and Technology* 171, 2026): run the heavy detector
(YOLOv11) only on every τ-th frame (τ = 3) and track the frames in between with
a light single-target network (FELNet) and a Kalman filter. Our additions: a
retrained YOLOv11m, IoU-only association, NSA Kalman, camera-motion
compensation, BYTE association, and the deployment simulation below.

It is now integrated as a **split UAV / ground-station deployment**:

```
 camera (30 fps) --> EDGE = the UAV ------------------------------------------
                     CMC -> Kalman -> FELNet -> association -> tau schedule
                       |  detection frames only (1 in 3)       ^ boxes
                       v  JPEG, UDP, ~1400-byte packets         |
                     LINK emulator: delay, jitter, bandwidth, buffer, loss
                       |                                        ^
                       v                                        |
                     GROUND station: YOLOv11 --------------------
 If a reply is lost or misses its deadline, that frame is tracked on the UAV
 instead -- the paper's own missed-detection mechanism.
```

Three separate processes on one machine, real UDP sockets, one shared clock.
The edge can be limited to a small onboard computer (CPU cap, cores, memory).

## 2. What was integrated (and where)

| Component | Code | Integrated with |
|---|---|---|
| Detector (YOLOv11m) | `jmdst/pipeline/models.py` | ground node |
| FELNet + Kalman tracker (paper Alg. 1) | `jmdst/pipeline/jmdst.py`, `jmdst/tracking/` | edge node |
| Network link (netem-equivalent) | `jmdst/net/link.py`, `relay.py` | between nodes |
| Edge/ground protocol | `jmdst/net/protocol.py`, `nodes.py` | both nodes |
| Edge hardware limits | `jmdst/sim/hardware.py` | edge process |
| Real-time camera + streaming scoring | `jmdst/sim/camera.py` | edge node |
| Telemetry | `scripts/sim_telemetry.py` | all three logs |

Key integration decision: the network sits **behind the existing `detector`
interface** of the tracker, so the tracker code is the same in the
single-process and the split system — nothing was duplicated.

## 3. How it was tested — four levels

**Level 1 — unit tests: 184, all passing** (`python -m unittest discover -s tests`).

| Area | Tests |
|---|---:|
| Tracking, association, Kalman, CMC | 67 |
| Network link model + relay over real sockets | 21 |
| Evaluation metrics (MOTA/IDF1/HOTA, edge cases) | 16 |
| FELNet model + training (incl. runaway-box regression) | 23 |
| Protocol, edge/ground nodes, detection cache | 12 |
| Dual-branch pipeline incl. lost-detection fallback | 11 |
| Camera, hardware limits, streaming scoring, telemetry | 10 |
| Data pipeline, demo, export, verification | 24 |

**Levels 2 and 3 — every component checked against a reference it must match.**

| Level | Check | Compared against | Result |
|---|---|---|---|
| 2 | Network emulator | its own configured delay, jitter, loss, burst length, throughput, buffer | **29/29 checks pass**; packets released within 1.2 ms (p95) of schedule |
| 2 | UAV CPU cap | the configured budget (1.0 core) | 4 busy threads held to 1.14 cores |
| 2 | UAV memory cap | the configured limit (1 GB) | a 2 GB allocation refused |
| 3 | Split system, perfect link, lossless frames | the original single-process tracker | **byte-identical, 13/13 val sequences** (5.35 M packets, 0 lost) |
| 3 | Modelled ground station | live YOLO on the ground | **byte-identical**, 829 detection frames, 0 cache misses |
| 3 | Real-time runs (timing-dependent) | a deterministic replay of the same run | **byte-identical**, 3/3 sequences, GPU and CPU UAVs |
| 3 | Box-size fix after the 2% loss crash | output before the fix | **identical, 13/13 val sequences** |
| 4 | Tracker accuracy | the paper's published tables | Section 4 |
| 4 | Deployment sweep | — | Section 5 |

**Testing found real defects** — each would have produced plausible but wrong
numbers, not a crash:

| Found by | Defect | Effect before the fix |
|---|---|---|
| profiling | crop copied the whole frame once per target | UAVDT FPS +29%, VisDrone ~+70% after fix |
| calibration | Windows 11 drops a background process's 1 ms timer after ~2 s | packets released up to 15.7 ms late |
| equivalence run | relay read a whole burst before forwarding | 9 ms lateness on a perfect link |
| lossy-link test | ground's first YOLO call cold (150 ms) | first frame falsely "lost" |
| pilot sweep | edge's first frame cold (2,075 ms) | camera moved on 58 frames |
| replay verifier | YOLO replayed on the wrong device | replay mismatched (verifier bug) |
| GPU clock probe | busy CPU edge down-clocks the shared laptop GPU | ground YOLO 15 -> 42 ms |
| sweep at 2% loss | a long-coasting track's box grew without bound; crop raised `MemoryError` | run crashed; box now capped at frame size, 13/13 val outputs unchanged |

## 4. Tracker results (held-out test split)

| Metric | UAVDT ours | UAVDT paper | VisDrone ours | VisDrone paper |
|---|---:|---:|---:|---:|
| HOTA | 47.8 | 60.1 | **49.3** | 46.4 |
| MOTA | 55.4 | 67.3 | 44.6 | 44.9 |
| IDF1 | 59.7 | 73.5 | 59.7 | 62.4 |
| FPS | **62.9** | 26.6 | **31.8** | 18.6 |

Takeaway: VisDrone matches the paper on MOTA and beats it on HOTA; UAVDT
trails. Splits are not the paper's (never published), so this is indicative.
FPS is on an RTX 4070 Laptop GPU (the paper used an RTX 3070), re-measured
after the crop speed fix with byte-identical tracking output.

## 5. Deployment results (simulation)

Full tables and charts: `outputs/sim/sweep_quick/report/report.md`.

**5.1 Setup**

| Setting | Value |
|---|---|
| Data | UAVDT val, 3 sequences, 2,484 frames |
| Camera | real time, 30 fps; a frame the UAV is too busy for is skipped |
| Scoring | each frame scored against the newest *finished* result (what an operator sees) |
| Ground station | modelled: live-identical detections, 0 cache misses |
| Base UAV | 4 CPU cores |
| Base link | 10 ms ±2 each way, 50 Mbit/s uplink, no loss |
| Reply deadline | 150 ms |

**5.2 Where should detection run? (HOTA)**

| UAV computer | Split: YOLO on ground | Onboard: YOLO on the UAV | Better | Margin |
|---|---:|---:|---|---:|
| GPU | 39.2 | **43.6** | onboard | +4.4 |
| 4 CPU cores | **36.8** | 22.5 | split | +14.3 |
| 1 CPU core | 32.4 | — | split | — |

Takeaway: offload detection only when the UAV has no GPU.

**5.3 How much computer does the UAV need? (split design)**

| UAV computer | HOTA | Frames processed | Frame latency p50 | Frame latency p95 |
|---|---:|---:|---:|---:|
| GPU | 39.2 | 85% | 41 ms | 74 ms |
| 4 CPU cores | 36.8 | 64% | 69 ms | 99 ms |
| 1 CPU core | 32.4 | 44% | 88 ms | 148 ms |
| *Onboard, 4 CPU cores (for contrast)* | *22.5* | *20%* | *87 ms* | *458 ms* |

Takeaway: the split design degrades gracefully, because only 1 frame in 3
leaves the aircraft.

**5.4 What does the network cost? (4-core UAV)**

| Condition | HOTA | Change vs base | Replies on time | Frames processed | Round trip p50 |
|---|---:|---:|---:|---:|---:|
| Base: 10 ms, 50 Mbit/s, no loss | 36.8 | — | 100% | 64% | 58 ms |
| 0.5% packet loss | 26.1 | −10.7 | 71% | 52% | 57 ms |
| 2% packet loss | 9.0 | −27.8 | 18% | 38% | 56 ms |
| 10 Mbit/s uplink | 23.0 | −13.8 | 60% | 46% | 120 ms |
| 50 ms one-way delay | 25.3 | −11.5 | 76% | 40% | 143 ms |

Takeaway: the network costs more than the UAV's computer. Every lost or late
reply stalls the UAV for the full 150 ms deadline, so the camera moves on ~4
frames each time.

**5.5 Why packet loss hurts so much: one lost packet loses the whole frame**

| Run | Packets per frame | Packet loss | Frames surviving, theory | Measured |
|---|---:|---:|---:|---:|
| Demo snapshot (clip) | ~158 | 0.1% | 0.999^158 = 85% | 85.5% |
| Sweep | ~75 | 0.5% | 0.995^75 = 69% | 71% |
| Lossy-link test | ~70 | 2% | 0.98^70 = 24% | 23% |

Takeaway: there is no retransmission, so frame loss grows with frame size;
theory and measurement agree.

**5.6 What real time costs (UAVDT val, HOTA)**

| Configuration | HOTA | Change |
|---|---:|---:|
| Offline, lossless frames, every frame processed | 46.6 | — |
| Offline, JPEG frames | 45.8 | −0.8 |
| Offline JPEG, shown one frame late (theory for a perfect real-time UAV) | 43.8 | −2.0 |
| Real time, onboard GPU (no network) | 43.6 | −0.2 |
| Real time, split, GPU UAV (58 ms round trip) | 39.2 | −4.4 |
| Real time, split, 4 CPU cores | 36.8 | −2.4 |
| Real time, split, 1 CPU core | 32.4 | −4.4 |

Takeaway: the onboard GPU run matches theory (43.6 vs 43.8), which confirms the
real-time scoring; then each step down in network or compute costs 2–4 HOTA.

**5.7 What the results point to next**

| Problem | Possible fix |
|---|---|
| One lost packet loses a whole frame | forward error correction, or retransmit missing packets within the deadline |
| Large frames take long to send | lower JPEG quality, or send only regions of interest |
| A lost reply stalls the UAV for the full deadline | a shorter deadline, or use late detections asynchronously (a measured departure from Algorithm 1) |
| Only UAVDT measured | run the full 24-configuration plan (`configs/sweeps/edge_ground_val.json`, ~2 h) for VisDrone too |

Caveats: one run per configuration; HOTA is the headline because UAVDT's
partial labelling makes MOTA strongly negative under real-time scoring.

## 6. Honest limitations

* All three nodes run **on one laptop**; the link is emulated, not a radio.
* The edge is a CPU-capped laptop process: a **compute budget**, not a named
  device (x86 cores are not an ARM board).
* On one laptop CPU and GPU share a power budget, so live ground timings are
  coupled to the edge's load; sweeps use the modelled ground to remove that.
* VisDrone has no published frame rate; 30 fps is assumed. MSFP (Phase 6 of
  the paper) is not implemented — about 1% by the paper's own ablation.

---

## Demo script (~10 minutes)

All commands are for **cmd.exe** in VS Code, run from `C:\ML project\UAV_Draft_2`
with the `jmdst` environment active in **every** terminal
(`conda activate jmdst`). Before the mentor arrives, open three terminals and
run the preflight check once (it must end with exit code 0):

```bash
python scripts/preflight.py --source demo_package/clips/uavdt_M0101.mp4
```

**Step 1 — Testing (Terminal 1, ~15 s).**

```bash
python -m unittest discover -s tests
```

Expect `Ran 184 tests ... OK`. Say: four levels of testing (Section 3); this is
level 1, and it includes the network relay over real sockets and the
lost-detection fallback.

**Step 2 — Component calibration (Terminal 1, ~20 s).**

```bash
python scripts/calibrate_link.py --quick
```

Expect `29/29 checks passed`. Say: every network setting is measured against
its configured value — delay, jitter, loss, burst length, throughput, buffer.

**Step 3 — The integrated system, live (three terminals, ~1 min).**
Start them in this order, each after the previous one prints the line shown.

Terminal 1 — ground station (waits for `[ground] listening`):

```bash
python scripts/ground_node.py --yolo outputs/yolo_runs/retrain_aug_m_lowlr/weights/best.pt --print-every 10
```

Terminal 2 — network link (waits for `[link] listening for edge`):

```bash
python scripts/link_relay.py --e2g "delay 15ms 5ms rate 50mbit loss 0.1% seed 1" --g2e "delay 15ms seed 2" --telemetry outputs/sim/live/link.csv
```

Terminal 3 — the UAV (runs ~15 s):

```bash
python scripts/edge_node.py --felnet outputs/felnet_runs/full_run/best.pt --source demo_package/clips/uavdt_M0101.mp4 --conf 0.55 --no-appearance --nsa --cmc --byte --byte-low-conf 0.4 --realtime --device cpu --edge-cpus 16-19 --deadline-ms 150 --output-root outputs/sim/live
```

Point at: Terminal 3's `sent N (ok, lost) | local M | skipped K` lines (1 frame
in 3 goes to the ground, the rest are tracked on the UAV; `skipped` = camera
frames the 4-core UAV was too busy for); Terminal 1's `serving frame ...` and
`never completed` lines (a frame that lost one packet on the link). Then stop
Terminals 1 and 2 with **Ctrl+C** — each prints its summary.

**Step 4 — Telemetry (Terminal 3, ~2 s).**

```bash
python scripts/sim_telemetry.py --run outputs/sim/live --rows 15
```

One row per detection frame: send time, arrival time, queue delay, bytes,
packets, loss flag and *why*, RTT — then the run summary.

**Step 5 — What the tracker sees (Terminal 3, ~15 s, a window opens; `q` closes it).**

```bash
python scripts/demo.py --source demo_package/clips/uavdt_M0101.mp4 --show
```

Point at the HUD's second line alternating `DETECT` / `TRACK`: YOLO runs on 1
frame in 3. Boxes appear after ~4 frames — a track is confirmed only after
N = tau + 1 = 4 frames (the paper's rule, not a bug).

**Step 6 — Results.** Open this file's Sections 4–5 and
`outputs/sim/sweep_quick/report/report.md` (tables + charts).

If anything fails live, the same evidence is pre-generated in
`handover/snapshots/` (4.10 terminals, 4.11 telemetry, 4.12 tests).

