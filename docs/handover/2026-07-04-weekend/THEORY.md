# Theory & Evidence — why RuView says "present" in an empty office

All numbers below are measured on **camera-ground-truthed** data
(MediaPipe pose oracle, 2026-07-03) unless marked otherwise. Scripts:
`scripts/camera_ground_truth_oracle.py`, `scripts/weekend_auto_retrain.py`,
plus scratch experiments referenced inline.

---

## 1. The phenomenon

With the office verifiably empty, the live system reports:

- `classification.presence = true` **91–93%** of scored samples (n=116+)
- `desk_occupancy.votes.desk_los = 1.0` pegged — nodes 1–3 "see" a person
- `votes.absent` *decays* toward 0 the longer the room is empty (feedback loop)
- Per-node motion classifiers: `present_moving` ~100% on **all 6 nodes**
  during confirmed-empty (node 1: 83% moving / 17% still / **0% absent**)

So the failure is systemic: empty-office multipath saturates every per-node
motion classifier, and every downstream layer (desk_los votes, room guard,
absence clamp) inherits the corruption. Phase A (commit `13dd546d`) removed the
worst latching but cannot fix garbage-in.

## 2. What we ruled out — with evidence

| Hypothesis | Test | Result | Verdict |
|---|---|---|---|
| "Add an `empty` class to the static desk discriminator" | logreg, 25 window features, held-out **recordings** | 11% empty recall (59% in-distribution) | ❌ dead |
| "Static residual vs a stored empty baseline" | June baseline vs July empty vs occupied | empty-vs-baseline dev (0.038) > person effect (0.008–0.016) | ❌ dead |
| "Static classifier on clean labels" | leave-session-out, camera-labelled | 26% empty recall on unseen session | ❌ dead |
| "It's a labeling problem, WiFi can't do this at all" | clean adjacent-pair test | **94/95% recall** with fresh baseline | ❌ (pessimism refuted) |

**Central empirical law of this office:**
> Session-to-session drift of the empty-room CSI signature is LARGER than the
> difference a human presence makes — for absolute amplitude features.
> Within a short horizon (≲10 min), a person is cleanly separable (2.4–2.5×
> deviation ratio).

## 3. The measured drift structure

Empty-vs-empty deviation by baseline age (median normalized |z|; fresh ≈ 0.2,
person ≈ 0.55–0.85):

| Baseline age | dev (Fri afternoon runs) |
|---|---|
| 0 min (self) | 0.21–0.28 |
| 3 min | 0.25–0.29 |
| 10–13 min | 0.30 … **0.60–0.69** (abrupt jumps happen) |
| 33–120 min | 0.42–0.58 (person-level) |

Two distinct drift modes:
1. **Slow drift** — gradual environment/hardware change; minutes-scale.
2. **Abrupt invalidation** — step changes within ≤15 min (door/AC/AP behaviour?),
   observed 0.30→0.84 between adjacent windows.

Suspected contributors (research agents verifying): ESP32 AGC per-packet gain,
AP transmit-power/rate adaptation, thermal drift, genuine scene changes.
**Design consequence:** baselines must be refreshed on every camera-confirmed
empty, and a stale/invalidated baseline must degrade to "uncertain" — never
assert absent from a stale reference.

## 4. Why the adjacent-pair 94% became 37% multi-session (the open problem)

Fri 17:39 honest cycle (14 empty / 42 occupied sessions, leave-newest-out):
fresh-baseline empty recall **37.2%**, occupied recall 69.3%.

Interpretation: the residual *classifier* currently learns session-specific
residual scales. Residual = (window − baseline) still carries absolute scale;
when drift changes the gain, the residual distribution shifts and the trained
boundary misses. The fix direction is **scale/drift-invariant features**
(normalize per window; shape-correlation instead of raw differences) — Phase R2
in ROADMAP.md. The weekend watcher gives the honest metric to iterate against.

## 5. Working hypotheses — status after Friday-evening kill tests

- **H1 (AGC/normalization rescues stale baselines): ❌ REFUTED (Fri 17:50).**
  L2-normalizing window profiles flattens scale, but the person's deviation
  collapses identically: separation margin vs a stale baseline is 0.70× raw and
  0.65× normalized (need >1). **Inter-session drift changes the profile SHAPE,
  not just gain.** Consequence: the camera-refresh loop is load-bearing; no
  feature trick on profile means rescues a stale baseline.
- **H1b (temporal micro-motion features transfer across sessions): ❌ REFUTED
  (Fri 17:55)** at 3-second windows. Gain-normalized diff features,
  leave-newest-session-out: occupied recall 80%, but held-out empty recall
  **27.5%** — empty-room flicker statistics are as session-specific as gain.
  Short-window "motion" does not transfer either.
- **H2 (shape stability): weakened by H1 result** — shape drifts too. Retained
  only as a *within-fresh-horizon* feature.
- **H3 (per-node voting): open** — test on weekend data.
- **H4 (breathing-band): open, PROMOTED** — with H1/H1b dead, long-window
  (≥30 s) periodicity is the main candidate for still-person detection beyond
  the fresh-baseline horizon. Needs real per-node timing (recordings'
  `nodes[].sync` field — investigate; recording `timestamp`s are batch-written
  ~fake-1 kHz and unusable for spectral work).
- **H5 (abrupt-drift detection): open** — multi-node-simultaneity signature.

### Design consequence (supersedes "train a transferable classifier")

Since NEITHER profile residuals NOR short-window temporal stats transfer across
sessions, but fresh-baseline deviation separates cleanly (2.4–2.5×), the runtime
detector should be **self-calibrating per baseline, not pre-trained**: on each
camera-confirmed empty, capture (a) the per-node reference profile AND (b) its
own noise spread; detect presence as sustained deviation > k×spread. No
training-set transfer is required at all. A trained classifier becomes a
secondary refinement inside the fresh horizon, not the foundation. This is
simpler to implement in Rust and matches every measurement we have.

## 6. Fix architecture (validated direction)

```
camera oracle ──confirms empty──▶ POST /presence/baseline/capture
                                        │
ESP32 CSI ──▶ per-node window ──▶ residual vs recent baseline ──▶ drift-invariant
                                        │                          features
                                 staleness check                     │
                                        │                      logreg (v1 artifact)
                                        ▼                            │
                              stale ⇒ "uncertain"          confident-sustained-empty
                                                             ⇒ presence clamp
```

Key properties: camera is a *labeler/refresher*, never a runtime presence
decider; WiFi remains the sensor; stale baseline degrades honestly; same
`baseline/capture` seam accepts a future mmWave oracle (ESP32-C6 + MR60BHA2,
already in the supported-hardware table) if the camera is ever retired.

## 7. Methodology rules (paid for in blood on 2026-07-03)

1. **Camera-frame absence ≠ room empty.** Only the oracle's debounced,
   sustained-absence state (12 s no-detection) labels empty; guard windows
   around transitions are discarded.
2. **Never validate within-session.** 91.5% within-session == 26% held-out.
3. **Chat-coordinated ground truth is banned.** Two "bug confirmed" calls were
   wrong because of message latency + hysteresis. The camera timeline
   (`labels.jsonl`) is the only accepted label source.
4. Every retrain cycle logs sessions used + held-out + metrics — reproducible
   from `retrain_report.jsonl`.

## 8. Research findings (three agents, Fri evening — synthesized)

### 8a. Drift: it's probably the ESP32's AGC, and there are named fixes

The receiver-effects literature says our inter-session drift is most likely
**dominated by ESP32 AGC + FFT-scale changes, not the environment**. ESP32 CSI
amplitude is scaled per-packet by automatic gain control; stock ESP-IDF does
not expose the gain, so absolute amplitude statistics are uncalibrated across
packets. "Large, infrequent step changes" from AP rate/TX-power adaptation
match our abrupt 15-min jumps.

Prioritized fixes from the literature (each with source):

| Priority | Fix | Expected effect | Source |
|---|---|---|---|
| P0 | **L1 per-packet normalization** (`Ĥ_k = |H_k| / mean_k|H_k|`) before any feature | recovers up to 75% of accuracy lost to receiver effects | [Same Signal, Different Story (arXiv 2605.26836)](https://arxiv.org/html/2605.26836) |
| P0 | **AGC/FFT gain-lock in firmware** (S3 supports `phy_force_rx_gain` / `phy_fft_scale_force`, median-of-300 at boot) | stops per-packet gain oscillation at source | [ESPectre algorithms](https://espectre.dev/documentation/algorithms/); [esp-csi #185](https://github.com/espressif/esp-csi/issues/185), [#78](https://github.com/espressif/esp-csi/issues/78) — **NOTE: reflash requires Dave's explicit approval** |
| P1 | Scale-invariant features: **CV** (σ/μ, provably gain-invariant), **lag-1 autocorrelation**; drop raw mean/percentiles | boundary stops moving with gain | ESPectre |
| P2 | dB-domain + differencing; **EWMA background, 1–5 min half-life** + change-point-triggered reset, refreshed only when gated empty | matches our measured ~10-min staleness horizon | [Optimal Preprocessing (arXiv 2307.12126)](https://arxiv.org/html/2307.12126v2); SuBSENSE pattern |
| P3 | Within-packet **subcarrier ratios** / cross-link ratios (single-antenna analog of FarSense CSI-ratio) | cancels AGC + TX-power in-shape | [FarSense (arXiv 1907.03994)](https://arxiv.org/pdf/1907.03994) |

**Attribution experiments** (cheap, on captured data): (1) L1-normalize and
re-measure the staleness matrix — whatever disappears was gain; (2) flat-vs-
frequency-selective deviation split (flat = device/TX, selective = environment);
(3) correlate amplitude steps with per-packet RSSI (which *is* exposed).

**Reconciling with our H1 kill-test (§5):** our Fri test L2-normalized
*window-mean profiles* and found person-separation collapsed too. The
literature's P0 recipe is different — normalize *per packet*, then use
**variance-structure features** (CV, autocorrelation, eigenvalue), not profile
distances. H1-as-tested stays refuted; the L1-per-packet + CV/eigenvalue recipe
is **untested on our data** and is the first Monday experiment.

### 8b. Empty-vs-still: ranked techniques from the presence literature

1. **Cross-Subcarrier CSI Ratio (CSCR)** — divide CSI between two *subcarriers
   on the same antenna*; cancels common-mode drift, restores usable phase, no
   second antenna needed. Validated on commodity ESP32, ~91% respiration
   detection to 8 m. The one method designed for exactly our hardware.
   [SA-WiSense (arXiv 2507.17623)](https://arxiv.org/abs/2507.17623)
2. **Breathing-band periodicity as the empty-vs-still arbiter** — an empty room
   has no coherent 0.1–0.5 Hz component; a still person does. Needs ≥10 Hz
   CSI and 20–30 s windows (20 s → 3 bpm resolution). Self-evidencing: needs NO
   fresh empty baseline — complements the fresh-baseline detector exactly where
   it's weakest (stale horizon).
   [DeMan (JSAC'15)](https://tns.thss.tsinghua.edu.cn/~cswu/papers/JSAC15_Deman_paper.pdf) 94.1% TP / 98.6% TN;
   [BreatheBand (ToSN'23)](https://dl.acm.org/doi/10.1145/3582079)
3. **Drift-robust structural features + multi-node fusion** — inter-subcarrier
   correlation (empty: 56 subcarriers move coherently; person: decorrelates),
   top covariance eigenvalue ("power-irrelevant yet variation-dependent"),
   majority vote across our 6 nodes (independent multipath blips rarely
   coincide). An ESP32 multi-zone occupancy study hit **F1 0.96** with
   amplitude stats + band energy + inter-subcarrier correlation.
   [PADS (ICPADS'14)](https://zhouzimu.github.io/paper/icpads14-qian.pdf);
   [R-TTWD (IEEE Access'17)](https://ieeexplore.ieee.org/document/7874163/);
   [ESP32 occupancy study (2025)](https://www.sciencedirect.com/science/article/abs/pii/S2352710225013920)

   Also noted: [FullBreathe (UbiComp'18)](https://dl.acm.org/doi/10.1145/3264958)
   proves amplitude-only sensing has structural *blind spots* for still people —
   position-dependent misses are guaranteed until we adopt a ratio/phase-like
   representation (CSCR).

### 8c. Repo survey: the fix is ~70% already built (reuse, don't rebuild)

- **`ruvsense/calibration.rs`** (34 KB, implements "Proposed" ADR-135):
  `CalibrationRecorder` + `BaselineCalibration` — per-subcarrier Welford
  mean/var, `rms_amplitude_z` deviation per live frame, drift detection
  (`drift_score>4.0` → `BaselineDrift`), TOML/NVS serialization. **This IS the
  per-node reference + deviation runtime from §6** — wire the camera gate into
  `CalibrationRecorder::record()`, done.
- **`wifi-densepose-calibration` crate** (ADR-151, implemented): `Features`
  extraction from baseline-subtracted CSI (with NaN fail-closed guard — a NaN
  poisoning bug once silently disabled presence for a bank's lifetime),
  `PresenceSpecialist`, `MixtureOfSpecialists` with STALE flags and
  presence→vitals short-circuit. Extend + retrain on camera-confirmed anchors;
  do NOT build a parallel presence path.
- **ADR-135's own words justify our architecture**: auto-recalibration was left
  off because *"there is no reliable mechanism to verify room emptiness from
  CSI alone"* — the camera oracle is precisely that missing mechanism.
- **Real per-node µs timestamps exist on the wire** (H4 unblocked): sync packet
  magic `0xC511A110` (ADR-110) every ~20 frames carries node `local_us` +
  mesh-aligned `epoch_us` + a sequence high-water mark; pair with each CSI
  frame's u32 sequence number to reconstruct true per-frame timing. Open
  thread: confirm the host aggregator ingests these (firmware emits them).
  Also: real CSI rate is **~10 Hz** — not the 20 Hz ADR-135 assumes nor the
  100 Hz `breathing.rs` defaults to; window math must use the real rate.
- **Guardrails from ADRs**: fail-closed always (NaN/stale/incoherent ⇒
  flagged-absent, never confident-present); reconcile with the authoritative
  eigenvalue `estimated_persons` counter (ADR-082) rather than inventing a
  parallel count; **ADR-152 evaluation trap: never validate presence on
  empty-heavy data** — an always-absent predictor scores ~96% there; evaluate
  on entry/exit + occupancy-balanced sets.
- Aspirational (synthetic-only so far): R12/R12.1 PABS physics-anchored
  background subtraction (1,161× drift-floor separation in simulation) —
  revisit after the statistical baseline works.

## 9. Per-node reinterpretation with the real room geometry (Fri ~23:00)

Zugg recovered Dave's room map (`data/calibration/office_room_map_2026-06-26.jpg`
+ `node_geometry_office_2026-06-26.yaml`: 4.14×5 m, router west wall, desk north,
Dave at (1.32,1.25)). Re-ran the corpus per-node against it.

### Person-effect per node (confirmed adjacent pair 16:26 empty → 16:27 occupied)

| node | map role (weight) | router-path→seat | person× (dev ratio) | catches occupied @5% FP |
|---|---|---|---|---|
| 1 | multipath, waist (0.55) | 1.00 m | **3.07** | **99.3%** |
| 3 | direct, high (0.90) | **0.06 m** | **3.14** | **98.5%** |
| 6 | multipath, waist (0.45) | 1.14 m | 2.41 | 60.9% |
| 5 | dead-angle, knee (0.35) | 1.05 m | 1.80 | 47.2% |
| 4 | multipath, chest (0.70) | 0.62 m | 1.75 | 56.4% |
| 2 | direct, head (1.00) | 0.45 m | 1.90 | **34.3%** |

**Map half-confirmed, half-contradicted:** node 3 (router path through Dave's
seat) is a top witness exactly as the geometry predicts — but node 2, the map's
"strongest seated witness" (weight 1.0), is the WEAKEST measured, and node 1
("multipath context, 0.55") is the strongest. Measured evidence must drive the
fusion weights, not the sketch priors. Per-node voting (H3) looks strong on
confirmed pairs: 1∨3 OR-vote ≈ 99–100% catch at 6–9% FP.

### The cross-pair check exposed a label-quality boundary

Across 8 empty→occupied pairs, catch rates are bimodal: ~99–100% on
camera-confirmed real walk-outs (16:11, 16:26) but 6–31% on the 16:07–17:14
micro-segments — the stretch when Dave was in-and-out and often in the room but
OUT of the desk-camera's view. Those "empty" labels are desk-view-empty, not
room-empty; their baselines likely contain Dave.

**Corpus rule going forward:** trust (a) evening/weekend empty (Dave gone —
guaranteed) and (b) transitions explicitly confirmed in-session; EXCLUDE the
Fri 16:07–17:14 micro-segments from any training/validation. Monday's
still-seated + entry/exit sessions remain the decisive tests.

### RSSI reordering = live drift evidence

Per-node RSSI ordering has flipped repeatedly: Jun-26 snapshot node 3 = −77 dBm;
Fri afternoon −51; Fri 23:00 node 5 strongest at −49 (was −73 Jun-26). Consistent
with AP TX-power/rate adaptation (§8a) — log per-packet RSSI alongside features
and treat RSSI steps as baseline-invalidation triggers (H5).

---

## 10. 2026-07-07 — the two findings that reframe everything

### 10a. Frozen CSI: nodes stop sensing when there's no WiFi traffic (ROOT CAUSE)

The single most important discovery of the project. ESP32 CSI is computed
**per received WiFi frame**; with no traffic, a node's CSI freezes at its last
value. Measured in the recordings:

| Recording | Nodes with live (varying) CSI |
|---|---|
| Empty, Mon 15:38 | **1/6** (only node 6) |
| Occupied (Dave present), Sat | 4/6 (4 & 5 frozen) |
| Empty, Fri evening | 4/6 (4 & 5 frozen) |
| **Any recording, Mon evening, Dave + devices active** | **6/6** |

No hardware is dead — liveness tracks *traffic*, not occupancy. Implications
that invalidate earlier confidence:
- The weekend corpus (and the R2 "12/12") was computed on data where 1–4 nodes
  were frozen and *which* ones changed per session — the 6-node vote was really
  a 1–4-node vote, met by luck.
- Frozen nodes silently drop out of any per-node scheme (R3 included).

### 10b. Traffic injection fixes the freeze — but is a double-edged sword

`scripts/csi_traffic_injector.py` blasts unicast UDP to all 6 node IPs
(~40 Hz/node); each frame is delivered over-air by the AP → CSI stimulus.
**Result: 6/6 nodes live through a camera-confirmed empty room (vs 1/6
without).** Decisive, reversible, and standard "active mode" CSI practice.

**But** the injected traffic bounces around the empty room and creates its own
multipath dynamics: the empty room now reads `present_moving` 41% of the time
(old classifier), and a *still* person adds little on top. So injection trades
"frozen (undetectable)" for "live but noisy empty." Empty-with-traffic and
still-person-with-traffic look similar.

### 10c. R3 live result — the full chain fired, and it clamped

First end-to-end run (18:42): camera→empty → oracle POSTed baseline capture →
gate engaged with **all 6 nodes voting** (injection) → **presence clamped false**
at 14 s sustained empty. The architecture works. But it **oscillates**: per-node
deviation from a fresh 30 s baseline during empty —

| node | median z | %>3 (votes occ) |
|---|---|---|
| 1 | 1.99 | 39% | 2 | 1.98 | 11% | 3 | 1.66 | 17% |
| 4 | 2.98 | 44% | **5** | **3.43** | **56%** | 6 | 1.93 | **0%** |

Nodes 4 & 5 (5 = the gain-locked one) sit above k=3 when empty and trip the
≥2-node vote. Threshold sweep on live empty vs a **still**-person occupied
recording found **no clean operating point** (k=4.5/vote=3: empty clean 1.4%
but occupied recall collapses to 16%) — because the occupied sample was 71%
dead-still, the genuine hard case, now sharpened by 10b.

**Conclusion:** R3 will likely handle the common **moving** in/out case well
(large deviation); the **still-person vs live-empty** case is unsolved and is
the real remaining problem. Next iteration (not a one-line fix):
1. Longer baseline capture (90–120 s) → truer per-node spread → auto-raises the
   bar for noisy nodes 4/5.
2. Quality-weighted / ≥3-node vote; down-weight the 2 noisiest nodes.
3. Measure moving-recall and still-recall **separately**, never averaged.
4. Consider steadier injection so empty-room dynamics settle; and revisit
   breathing-band (H4, per-node timing now confirmed available) as the
   principled still-person signal that survives the injection-noise floor.

---

## 11. 2026-07-08 — AGC gain-lock verdict + fleet reflash

Overnight (07/07 20:27 → 07/08 08:07), 15 clean 6-node empty captures with
injection running. Per-node cross-session drift (mean profile CV across 8
sampled sessions):

| node | RAW drift | after L1-norm | |
|---|---|---|---|
| 1 | 9.2% | 8.5% | stock |
| 2 | 9.4% | 9.2% | stock |
| 3 | 6.6% | 6.4% | stock |
| 4 | 10.1% | 9.9% | stock |
| **5** | **2.0%** | **2.0%** | **gain-locked** |
| 6 | 8.0% | 7.9% | stock |

**Verdict: the gain-lock works.** Node 5 (locked overnight) drifted 3–5× less
across sessions than every stock node — and **L1 normalization did NOT close
the gap** (stock stays 6–10% after it). So the inter-session drift is not a
simple global gain factor software can strip; it's deeper per-subcarrier
AGC/FFT-scale drift the firmware lock catches at source. This is the drift that
killed static classifiers (§2) and shortened baseline staleness (§3).

Caveat: node 5 is the weakest-angle node, so position could contribute — but
being the clear outlier (2% vs 6.6–10.1%) right after being the only locked node
is strong. Note the lock fixes CROSS-session drift, not WITHIN-session noise
(node 5 was still noisiest moment-to-moment in the §10c R3 test — environmental).

**Action taken:** all 6 nodes reflashed to `build-gainlock-v54` (2026-07-08),
verified `gain_lock: lock holding` + correct node IDs. Fleet now 3–5× more
stable cross-session → longer-lived empty baselines + session-transferable
models become viable. MAC registry + PSK caveat:
`~/.local/share/ruview/oracle/node_hardware_registry.txt`.

---

## 12. 2026-07-08 — BREAKTHROUGH: still-person IS separable (R3 iteration)

The "unsolvable" still-vs-empty case (§10c) is solved once three fixes stack:
**injection** (live CSI, not frozen — §10) + **fleet gain-lock** (no cross-session
drift — §11) + **90s fresh baseline** (not 30s; not stale). Yesterday's pessimism
was an artifact of frozen/drifting/stale data, not physics.

Clean test (same session, all 6 nodes locked, camera-confirmed empty, no drift
confound): baseline from `auto_empty_20260708_092745` head, scored against its
own tail, an 8-min **dead-still** capture, and a 2-min **moving** capture.

Per-node median deviation z (fresh 90s baseline):

| node | empty | still | move |
|---|---|---|---|
| 1 | 4.6 | 44.7 | 20.0 | 2 | 2.9 | 36.2 | 15.2 | 3 | 3.1 | 18.4 | 13.0 |
| 4 | 5.9 | 18.7 | 17.9 | 5 | 3.0 | 5.4 | 6.0 | 6 | 3.5 | 10.2 | 16.2 |

Empty floor z≈3–6; still z≈5–45; moving z≈6–20 — every node separates. Room
vote (# nodes with z>K ≥ 2) sweep — % windows reading OCCUPIED:

| K | EMPTY | STILL | MOVE |
|---|---|---|---|
| 6 | 11% | 100% | 100% |
| **8** | **0%** | **100%** | **100%** |
| 10 | 0% | 100% | 100% |
| 12 | 0% | 100% | 100% |

**Chosen: K=8, 2-node vote, 90s baseline.** EMPTY 0% / STILL 100% / MOVE 100%,
wide clean window (K=6–12). Shipped to `presence_baseline.rs` (K 3→8,
CAPTURE_SECS 30→90) + oracle BASELINE_CAPTURE_SECS 30→90.

**Caveat:** one still-session vs one empty-session, same afternoon. Needs
cross-day replication + a staleness-horizon re-measure on the locked fleet
before "bulletproof". But it is the cleanest, most honest test to date and the
margin is comfortable. Next: live verify (gate holds a still person present,
flips empty on departure); then multi-day validation.
