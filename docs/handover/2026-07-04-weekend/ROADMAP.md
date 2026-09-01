# Roadmap — RuView presence detection: from 91% false-positive to operational

**Goal:** presence that is trustworthy against ground truth (empty → absent within
seconds; still-seated person → present), reliable activity states, vitals gated on
real presence — then begin *proper* large-scale training on top of a sound base.
**Method rule (hard-won on 2026-07-03):** every claim is validated against
**camera ground truth on held-out sessions**. Within-session splits are banned —
they gave us 91.5% that was really 26%.

Legend: ☐ open · each phase lists its **exit tests** — tick them off before moving on.

---

## Phase R1 — Weekend data + honest baseline tracking *(running now, unattended)*

Infrastructure detached and running: camera oracle (labelled duty-cycled capture)
+ auto-retrain watcher (3 h honest validation cycles).

**Exit tests**
- ☐ ≥ 20 empty sessions spanning ≥ 3 different times of day exist in
  `~/.local/share/ruview/data/recordings/`
- ☐ `retrain_report.md` has ≥ 10 cycles with staleness curves
- ☐ Weekend scoreboard shows the *unchanged* live system's empty FP% (expect ~90%)
  as the "before" number, measured over hours not minutes
- ☐ Occupied data includes ≥ 30 min of *still-seated* (Dave reading/on calls,
  Monday) — hardest case for the classifier

## Phase R2 — Self-calibrating fresh-baseline detector *(next code work)*

**Revised Fri evening after H1/H1b kill tests (THEORY.md §5):** cross-session
transfer of trained classifiers is a dead end (profile shape AND short-window
temporal stats are both session-specific). The detector must be
**self-calibrating per baseline**:

1. On camera-confirmed empty: capture per-node reference profile **and its own
   noise spread** (mean + std of window deviations during the empty period).
2. Presence = sustained deviation > k×spread (per-node, then cross-node vote —
   tests H3). k tuned on weekend data; no training-set transfer involved.
3. Trained residual logreg becomes a *secondary* refinement within the fresh
   horizon (activity hints), not the presence foundation.
4. Validate with `weekend_auto_retrain.py`-style leave-session-out harness
   adapted to self-calibrating mode: for each held-out empty session, calibrate
   on its head, score its tail + the nearest occupied session.
5. Investigate H4 (breathing-band, ≥30 s windows) for still-person coverage
   beyond the fresh horizon — first step: decode `nodes[].sync` for real
   per-node timing.

**Monday experiment queue (from Fri research — THEORY §8; run before any Rust work)**
1. ☐ **L1-per-packet norm + CV/eigenvalue/inter-subcarrier-correlation features**,
   rerun the staleness matrix + leave-session-out on the weekend corpus.
   (Different from Fri's refuted H1 test — see THEORY §8a reconciliation.)
2. ☐ **CSCR prototype** (cross-subcarrier ratio, SA-WiSense): best-pair ratios →
   breathing-band energy; measure empty-vs-still separation.
3. ☑ **Sync-packet ingestion check — ANSWERED Mon 2026-07-07 (H4 feasible).**
   The server ingests sync packets (`main.rs:6949-6977` → `NodeState.latest_sync`,
   tested) and every live node carries a `sync` snapshot with offsets + leader
   flags. Real per-node CSI rates measured: **22.8–36.5 Hz** (docs' ~10 Hz guess
   was wrong; ample for the 0.1–0.5 Hz breathing band). **The one gap:**
   `mesh_aligned_us_for_csi_frame()` (`main.rs:804`) is implemented + unit-tested
   but has NO production caller — frames are not yet stamped with mesh time.
   → precise R3 item: add the call-site in the frame path + persist the mesh
   timestamp into `SensingUpdate`/recordings. Node 5 re-validated with the mesh
   leader post-reflash (`is_valid=true`).
4. ☐ Flat-vs-frequency-selective drift attribution + RSSI-step correlation
   (device vs environment split).
5. ☑ **Geometry reconciliation — first pass DONE Fri 23:00** (THEORY §9):
   measured person-effect per node on the confirmed adjacent pair says the
   fusion should weight **nodes 1 and 3** (99%/98% catch @5% FP), NOT node 2
   (34%, despite map weight 1.0). Per-node OR-vote 1∨3 ≈ 99–100% catch at
   6–9% FP on confirmed pairs. Remaining: re-verify on Monday's still-seated
   session (node 1 vs 3 may reorder for a motionless person), then wire
   evidence-based weights into `node_geometry_weight`.
6. ☐ **Corpus hygiene**: exclude the Fri 16:07–17:14 empty micro-segments from
   training/validation (desk-view-empty ≠ room-empty during Dave's in/out
   stretch — THEORY §9). Trust evening/weekend empty + confirmed walk-outs.

**Exit tests** *(evaluation sets MUST be occupancy-balanced with entry/exit
transitions — ADR-152 trap: an always-absent predictor scores ~96% on
empty-heavy data)*
- ☐ Self-calibrated empty recall ≥ **95%** on ≥ 5 held-out empty sessions (tail after head-calibration)
- ☐ Occupied recall ≥ **90%** vs nearest-in-time occupied sessions, incl. still-seated set
- ☐ **Entry/exit transitions detected within ≤ 10 s** on ≥ 10 camera-timelined transitions (the anti-trap test)
- ☐ Staleness horizon measured: recall vs baseline age curve on weekend corpus; k and horizon documented
- ☐ H3 answered: per-node voting vs pooled — number in report
- ☐ H4 feasibility answered: real per-node timing recovered from sync packets? (yes/no + evidence)
- ☐ Same numbers reproduce from a cold rerun (deterministic script)

## Phase R3 — Runtime integration (Rust, `wifi-densepose-sensing-server`)

**Repo survey finding (THEORY §8c): ~70% of this already exists. REUSE:**

1. Baseline runtime = **`ruvsense/calibration.rs`** (`CalibrationRecorder`,
   `BaselineCalibration`, `rms_amplitude_z`, `drift_score` — implemented
   ADR-135). Do NOT write a new `EmptyBaseline` struct; wire the camera gate
   into `CalibrationRecorder::record()` (accept frames only while the oracle
   confirms empty) — this fills ADR-135's own stated gap ("no reliable
   mechanism to verify room emptiness from CSI alone").
2. Presence head = **`wifi-densepose-calibration` crate** (`Features`,
   `PresenceSpecialist`, `MixtureOfSpecialists` — ADR-151): extend features
   (keep the NaN fail-closed guard), retrain on camera-confirmed anchors.
   No parallel presence path.
3. `POST /api/v1/presence/baseline/capture` — called by the camera oracle on
   confirmed-empty (same seam a future mmWave oracle could drive).
4. Decision policy: confident-sustained-empty ⇒ presence clamp (extends
   `apply_product_absence_clamp`); stale baseline (> validated horizon) or
   NaN/incoherent (reuse `coherence_gate.rs::GatePolicy`) ⇒ **degrade to
   "uncertain", never assert**. Reconcile with the eigenvalue
   `estimated_persons` counter (ADR-082) — no competing count.
5. Keep Phase A guards (no regressions on the 13 room-fusion tests).

**Exit tests**
- ☐ `cargo test -p wifi-densepose-sensing-server --no-default-features` — 0 failed,
  new unit tests for baseline capture / staleness / clamp policy
- ☐ Rust inference output == Python inference on 100 recorded windows (parity test)
- ☐ Container rebuilt + redeployed with rollback preserved
- ☐ **Live: oracle scoreboard empty FP% drops from ~91% to ≤ 10%** over ≥ 2 h of
  camera-confirmed empty
- ☐ Live: still-seated Dave stays present ≥ 95% (no over-clamp) — scoreboard FN
- ☐ Full pre-merge checklist (CLAUDE.md) incl. workspace tests + Python proof

## Phase R4 — Activity + vitals on a sound base

1. Retrain broad activity classifier (`adaptive_classifier.rs`) with
  camera-labelled data via existing `/api/v1/adaptive/train` (explicit approval
  step — never as a side effect).
2. Vitals gating: only emit breathing/HR when presence is confident;
  investigate breathing-band coherence as *additional* presence evidence
  (needs per-node timing reconstruction — recordings carry a `sync` field;
  see THEORY.md open questions).
3. Desk-occupancy refinement on top of trustworthy room-level presence.

**Exit tests**
- ☐ Activity states (still/moving/away/empty) ≥ 85% vs camera timeline over a day
- ☐ Vitals shown only when present; no vitals on empty room over 24 h
- ☐ Desk seated/near/away accuracy re-measured against camera ≥ current (0.87)

## Phase R5 — "Proper training" gate (the original goal)

Only after R3 exit tests pass does large-scale training make sense (garbage-in
ends here). Then: multi-day labelled corpus via the oracle → train richer
temporal models (the `N_TEMPORAL_FEATURES` path) → cross-environment work
(MERIDIAN/ADR-027) if desired.

**Entry criteria (== R3+R4 exits)** · **Exit:** defined with Dave once entered.

---

## Standing infrastructure asks (for Dave)

- ☑ **Physical node positions** for nodes 1–6 — Dave's room sketch and derived
  geometry now live in `data/calibration/office_room_map_2026-06-26.jpg` and
  `data/calibration/node_geometry_office_2026-06-26.yaml`. The live API may
  still emit placeholder `[2.0,0,1.5]`; R2/R3 work should wire/consume the repo
  calibration artifact for real geometry weighting.
- ☐ ~30 min still-seated session Monday (reading/calls) for the hardest-case set.
- ☐ **Approve an AGC/FFT gain-lock firmware experiment on ONE node** (ESP32-S3
  supports `phy_force_rx_gain`/`phy_fft_scale_force`; ESPectre recipe,
  THEORY §8a). Literature says AGC is the likely dominant drift source; locking
  one node and comparing empty-room stability against an unlocked node measures
  the device share directly. Reflashing requires your explicit approval per the
  standing rules — this is the ask.
- ☐ Optional: second camera or wider lens only if oracle blind spots become an
  issue (current desk-view framing has been sufficient so far).
