# JMDST — integration & testing brief

For the mentor review. One page of substance, then the demo script.

---

## 1. What the system is

A reproduction of **JMDST** (She et al., *Aerospace Science and Technology*
171, 2026): UAV vehicle tracking that runs the heavy detector (YOLOv11) only
on every τ-th frame (τ = 3) and tracks the frames in between with a light
single-target network (FELNet) and a Kalman filter.

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

**Level 2 — component calibration (each part measured against its spec).**
* Network emulator: **29/29 checks** — configured vs measured delay, jitter,
  loss rate, burst length, throughput, buffer overflow. Release timing within
  1.2 ms (p95) of schedule. (`scripts/calibrate_link.py`)
* CPU cap: 4 busy threads capped at 1 core used 1.14; a 1 GB memory cap
  refused a 2 GB allocation.

**Level 3 — integration: the split system must equal the single process.**
* Over a perfect link with lossless frames, **all 13 val sequences are
  byte-identical** to the single-process tracker (5.35 M packets, 0 lost).
* The modelled ground station gives **byte-identical** output to live YOLO
  (829 detection frames, 0 cache misses).
* Real-time runs cannot be compared to a fixed reference (they depend on
  timing), so each is **replayed deterministically: 3/3 sequences
  byte-identical**, on GPU and CPU-limited edges.

**Level 4 — system evaluation.**
* Tracker accuracy on the held-out test split (Section 4).
* Deployment sweep: edge hardware × network conditions (Section 5).

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

| | UAVDT (ours / paper) | VisDrone (ours / paper) |
|---|---|---|
| HOTA | 47.8 / 60.1 | **49.3 / 46.4** |
| MOTA | 55.4 / 67.3 | 44.6 / 44.9 |
| IDF1 | 59.7 / 73.5 | 59.7 / 62.4 |
| FPS | 26.9 / 26.6 | 15.6 / 18.6 |

On VisDrone the reproduction **matches the paper on MOTA and exceeds it on
HOTA**; UAVDT trails. Splits are not the paper's (it never published them), so
the comparison is indicative.

## 5. Deployment results (simulation)

UAVDT val (3 sequences, 2,484 frames). Real-time camera at 30 fps, scored
the way an operator would see it (each frame shows the newest *finished*
result). Modelled ground station: live-identical detections, 0 cache misses.
Base: a **4-core CPU UAV**, link 10 ms ±2 each way, 50 Mbit/s, no loss,
150 ms reply deadline. For scale: the same tracker with no clock (every frame
processed) scores HOTA 45.8. Full tables and charts:
`outputs/sim/sweep_quick/report/report.md`.

**Where should detection run? (HOTA, frames processed)**

| UAV compute | Split: YOLO on ground | Onboard: YOLO on the UAV |
|---|---|---|
| GPU | 39.2 (85%) | **43.6** (100%) |
| 4 CPU cores | **36.8** (64%) | 22.5 (20%) |
| 1 CPU core | 32.4 (44%) | — |

* **With a GPU on the UAV, keep detection onboard** (+4.4 HOTA): it avoids a
  58 ms network round trip on every detection frame.
* **Without one, offloading is decisive: +14.3 HOTA on 4 cores.** YOLO on a
  CPU takes up to 458 ms a frame, so the UAV processes only 20% of frames.
* **The split design degrades gracefully** as the UAV gets weaker (39.2 ->
  36.8 -> 32.4), because only 1 frame in 3 ever leaves the aircraft.

**What does the network cost? (4-core UAV)**

| Condition | HOTA | Replies on time | Frames processed |
|---|---:|---:|---:|
| Base: 10 ms, 50 Mbit/s, no loss | 36.8 | 100% | 64% |
| 0.5% packet loss | 26.1 | 71% | 52% |
| 2% packet loss | 9.0 | 18% | 38% |
| 10 Mbit/s uplink | 23.0 | 60% | 46% |
| 50 ms one-way delay | 25.3 | 76% | 40% |

**The network matters more than the UAV's compute.** Two mechanisms, both
visible in the telemetry:

1. **One lost packet loses the whole frame.** A detection frame is ~75 packets
   and there is no retransmission, so 0.5% packet loss loses ~30% of frames
   (theory 0.995^75 = 69% survive; measured 71%).
2. **Every lost or late reply stalls the UAV for the full 150 ms deadline**
   (Algorithm 1 waits for frame t's detections), so the camera moves on ~4
   frames each time. A slow link (10 Mbit/s: 72 ms just to send a frame) or
   high latency (round trip 143 ms against a 150 ms deadline) does the same by
   pushing replies past the deadline.

Next steps this points to: tuning the deadline, forward error correction or
selective retransmission, smaller frames (JPEG quality, regions of interest),
and using late detections asynchronously (a measured departure from Alg. 1).

**Consistency check:** the onboard GPU UAV processes every frame and shows the
previous frame's result, so it should score like the offline output shifted
one frame — predicted 43.8, measured 43.6.

Caveats: UAVDT only (VisDrone is in the full 24-configuration plan,
`configs/sweeps/edge_ground_val.json`, ~2 h); one run per configuration. MOTA
is strongly negative under real-time scoring on UAVDT (its partial labelling
plus displayed boxes one frame stale), so HOTA is the headline, as elsewhere in
this project.

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

