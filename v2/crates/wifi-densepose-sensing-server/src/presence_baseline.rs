//! Camera-anchored empty-room presence gate (ROADMAP R3, 2026-07-07).
//!
//! Implements the detector validated offline on the 2026-07-04 weekend corpus
//! (12/12 empty and 12/12 occupied sessions correct at the per-node-vote level;
//! see `docs/handover/2026-07-04-weekend/ROADMAP.md` Phase R2): per-node
//! **L1-normalized** window profiles scored against a **camera-confirmed empty
//! baseline** captured at runtime, combined by **per-node majority vote**.
//!
//! Design rules (THEORY.md §5/§8, ADR-151 fail-closed guard):
//! - The baseline is captured only when the external oracle (camera today,
//!   mmWave later) confirms the room is empty — `POST /api/v1/presence/baseline/capture`.
//! - A stale (> [`STALE_SECS`]) or missing baseline yields `Uncertain`, never a
//!   confident claim in either direction.
//! - NaN/short-window inputs are dropped, never poisoning the reference
//!   (ADR-151's NaN-poisoning regression class).
//! - The gate can only *clamp presence off* (sustained Empty verdict); it never
//!   forces presence on — the eigenvalue counter (ADR-082) stays authoritative
//!   for counts.
//!
//! Constants mirror the validated Python exactly; change them only together
//! with a rerun of `scratchpad/r2_feature_experiments.py`-style validation.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// Window length in frames (per node) — matches offline WIN=20.
pub const WIN: usize = 20;
/// Evaluate every STRIDE frames once the window is full — matches offline.
pub const STRIDE: usize = 5;
/// Deviation threshold in spread units. Retuned 2026-09-04 from 8.0 → 6.0
/// after Dave-labelled four-state baseline showed the previous threshold left
/// true occupied desk states with only one voting node while empty stayed below
/// the two-node occupied quorum. Keep the room-level quorum as the false-blip
/// guard; K should separate current geometry, not mask it. See THEORY §12.
pub const K: f64 = 6.0;
/// A node votes "occupied" when more than this fraction of its recent
/// windows exceed [`K`] — matches offline node vote rule.
pub const NODE_OCCUPIED_FRAC: f64 = 0.5;
/// Room is "occupied" when at least this many nodes vote occupied.
pub const VOTE_OCCUPIED_MIN: usize = 2;
/// Need votes from at least this many nodes to assert "empty" (quorum).
pub const VOTE_EMPTY_QUORUM: usize = 3;
/// Recent z-scores kept per node for the vote fraction.
pub const RECENT_WINDOWS: usize = 24;
/// A node abstains until it has produced this many windows.
pub const MIN_WINDOWS_TO_VOTE: usize = 6;
/// Baseline staleness horizon (measured ~10–15 min on 2026-07-03; be inside it).
pub const STALE_SECS: f64 = 720.0;
/// Default capture duration when the oracle confirms empty. Retuned 2026-07-08
/// from 30 → 90s: a 30s window underestimated the noisier nodes' spread (node 5
/// false-"occupied" 62%→4% going 30s→90s), causing the earlier oscillation.
pub const CAPTURE_SECS_DEFAULT: f64 = 90.0;
/// Minimum profiles per node for that node's baseline to be usable.
pub const MIN_CAPTURE_PROFILES: usize = 8;
/// Sustained Empty verdict required before the presence clamp may fire.
pub const EMPTY_CLAMP_SUSTAIN_SECS: f64 = 10.0;
/// Spread floor as a fraction of ‖ref‖ — guards against a degenerate capture
/// (unnaturally quiet RF) collapsing the spread to ~0 and making every
/// subsequent window read "occupied". Real captures sit well above this.
pub const REL_SPREAD_FLOOR: f64 = 0.005;

/// L1-normalize one frame: each subcarrier divided by the frame's mean |amp|.
fn l1_normalize(frame: &[f64]) -> Option<Vec<f64>> {
    if frame.is_empty() || frame.iter().any(|v| !v.is_finite()) {
        return None; // fail-closed: never ingest NaN/Inf (ADR-151 lesson)
    }
    let mean = frame.iter().map(|v| v.abs()).sum::<f64>() / frame.len() as f64;
    if mean <= 1e-12 {
        return None;
    }
    Some(frame.iter().map(|v| v / mean).collect())
}

fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Per-node frozen reference captured during a confirmed-empty period.
#[derive(Debug, Clone)]
struct NodeBaseline {
    ref_profile: Vec<f64>,
    /// Std-dev of ||profile − ref|| over the capture windows (+ε).
    spread: f64,
}

/// Per-node streaming state (window buffer + recent deviations).
#[derive(Debug, Default)]
struct NodeStream {
    frames: VecDeque<Vec<f64>>,
    since_eval: usize,
    recent_z: VecDeque<f64>,
}

/// Room-level verdict from the gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateVerdict {
    Occupied,
    Empty,
    /// Missing/stale baseline, or not enough node votes — never asserted.
    Uncertain,
}

/// Compact state surfaced in `SensingUpdate` and the status endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresenceGateInfo {
    pub verdict: GateVerdict,
    pub baseline_age_secs: Option<f64>,
    pub baseline_stale: bool,
    pub nodes_voting: usize,
    pub nodes_occupied: usize,
    pub sustained_empty_secs: f64,
    /// Per-node latest deviation (z), for the UI/oracle.
    pub node_z: HashMap<u8, f64>,
}

/// The frozen reference the gate scores against.
struct ReadyBaseline {
    per_node: HashMap<u8, NodeBaseline>,
    captured_at: Instant,
}

/// A baseline capture accumulating in the background (does NOT stop scoring).
struct CaptureInProgress {
    until: Instant,
    profiles: HashMap<u8, Vec<Vec<f64>>>,
}

/// The camera-anchored presence gate. Owned by `AppStateInner`.
///
/// `active` and `capture` are independent: the gate always scores against
/// `active` (when present) while a new baseline accumulates in `capture`, so a
/// re-baseline never blinds the gate (fixes the 30%→~0% live gap, 2026-07-08).
pub struct PresenceGate {
    active: Option<ReadyBaseline>,
    capture: Option<CaptureInProgress>,
    streams: HashMap<u8, NodeStream>,
    empty_since: Option<Instant>,
    last_info: Option<PresenceGateInfo>,
}

impl Default for PresenceGate {
    fn default() -> Self {
        Self {
            active: None,
            capture: None,
            streams: HashMap::new(),
            empty_since: None,
            last_info: None,
        }
    }
}

impl PresenceGate {
    /// Begin a baseline capture window (oracle has confirmed the room empty).
    pub fn start_capture(&mut self, now: Instant, duration_secs: f64) {
        let dur = if duration_secs.is_finite() && duration_secs > 1.0 {
            duration_secs
        } else {
            CAPTURE_SECS_DEFAULT
        };
        // Start accumulating a NEW baseline in the background. Keep `active`
        // (and the streams feeding it) so scoring continues uninterrupted — the
        // oracle only triggers capture after a debounced+stable empty, so the
        // frames arriving during the window are already person-free.
        self.capture = Some(CaptureInProgress {
            until: now + std::time::Duration::from_secs_f64(dur),
            profiles: HashMap::new(),
        });
    }

    /// True while a capture window is open.
    pub fn is_capturing(&self) -> bool {
        self.capture.is_some()
    }

    /// Feed one node frame (raw subcarrier amplitudes). Call for every node
    /// sample in the ESP32 ingest path.
    pub fn push_frame(&mut self, node_id: u8, amplitudes: &[f64], now: Instant) {
        let Some(norm) = l1_normalize(amplitudes) else {
            return;
        };
        let stream = self.streams.entry(node_id).or_default();
        stream.frames.push_back(norm);
        if stream.frames.len() > WIN {
            stream.frames.pop_front();
        }
        if stream.frames.len() < WIN {
            return;
        }
        stream.since_eval += 1;
        if stream.since_eval < STRIDE {
            return;
        }
        stream.since_eval = 0;

        let dim = stream.frames[0].len();
        if stream.frames.iter().any(|f| f.len() != dim) {
            stream.frames.clear();
            return;
        }
        let mut profile = vec![0.0; dim];
        for f in &stream.frames {
            for (p, v) in profile.iter_mut().zip(f) {
                *p += v / WIN as f64;
            }
        }
        // stream borrow ends here (profile is owned) so we can touch other fields.

        // 1. Finalize an expired background capture (swaps in the new baseline).
        if self.capture.as_ref().map_or(false, |c| now > c.until) {
            self.finalize_capture(now);
        }
        // 2. Accumulate into an ongoing background capture (scoring unaffected).
        if let Some(cap) = &mut self.capture {
            if now <= cap.until {
                cap.profiles.entry(node_id).or_default().push(profile.clone());
            }
        }
        // 3. Score against the ACTIVE baseline — always, even mid-capture.
        let z = self
            .active
            .as_ref()
            .and_then(|a| a.per_node.get(&node_id))
            .and_then(|b| {
                (b.ref_profile.len() == profile.len())
                    .then(|| l2(&profile, &b.ref_profile) / b.spread)
            });
        if let Some(z) = z {
            let s = self.streams.entry(node_id).or_default();
            s.recent_z.push_back(z);
            if s.recent_z.len() > RECENT_WINDOWS {
                s.recent_z.pop_front();
            }
        }
    }

    /// Close the capture window and freeze per-node references.
    /// Called automatically once `until` passes; harmless to call twice.
    pub fn finalize_capture(&mut self, now: Instant) {
        let Some(cap) = self.capture.take() else {
            return;
        };
        let mut per_node = HashMap::new();
        for (nid, profs) in cap.profiles {
            if profs.len() < MIN_CAPTURE_PROFILES {
                continue;
            }
            let dim = profs[0].len();
            if profs.iter().any(|p| p.len() != dim) {
                continue;
            }
            let mut reference = vec![0.0; dim];
            for p in &profs {
                for (r, v) in reference.iter_mut().zip(p) {
                    *r += v / profs.len() as f64;
                }
            }
            let dists: Vec<f64> = profs.iter().map(|p| l2(p, &reference)).collect();
            let mean_d = dists.iter().sum::<f64>() / dists.len() as f64;
            let var = dists.iter().map(|d| (d - mean_d) * (d - mean_d)).sum::<f64>()
                / dists.len() as f64;
            let ref_norm = reference.iter().map(|v| v * v).sum::<f64>().sqrt();
            let spread = var.sqrt().max(REL_SPREAD_FLOOR * ref_norm) + 1e-9;
            per_node.insert(
                nid,
                NodeBaseline {
                    ref_profile: reference,
                    spread,
                },
            );
        }
        if per_node.len() >= VOTE_EMPTY_QUORUM {
            // Atomic swap: new baseline becomes active. Intentionally do NOT
            // clear recent_z — keeping vote history means the gate never blinks
            // to `uncertain` during a re-baseline. The new baseline is a similar
            // empty reference, and stale z's age out within RECENT_WINDOWS.
            self.active = Some(ReadyBaseline {
                per_node,
                captured_at: now,
            });
        }
        // else: too few nodes — keep the PREVIOUS active baseline (fail-safe),
        // just drop the failed capture rather than going blind.
    }

    /// Evaluate the room verdict. Cheap; call once per broadcast tick.
    pub fn evaluate(&mut self, now: Instant) -> PresenceGateInfo {
        // Auto-finalize an elapsed capture even if no frames arrive.
        if self.capture.as_ref().map_or(false, |c| now > c.until) {
            self.finalize_capture(now);
        }

        let (verdict, age, stale, voting, occupied, node_z) = match &self.active {
            Some(ReadyBaseline {
                per_node,
                captured_at,
            }) => {
                let age = now.duration_since(*captured_at).as_secs_f64();
                let stale = age > STALE_SECS;
                let mut voting = 0usize;
                let mut occ = 0usize;
                let mut node_z = HashMap::new();
                for (nid, s) in &self.streams {
                    if !per_node.contains_key(nid) {
                        continue;
                    }
                    if let Some(z) = s.recent_z.back() {
                        node_z.insert(*nid, *z);
                    }
                    if s.recent_z.len() < MIN_WINDOWS_TO_VOTE {
                        continue;
                    }
                    voting += 1;
                    let frac =
                        s.recent_z.iter().filter(|z| **z > K).count() as f64
                            / s.recent_z.len() as f64;
                    if frac > NODE_OCCUPIED_FRAC {
                        occ += 1;
                    }
                }
                let verdict = if stale {
                    GateVerdict::Uncertain
                } else if occ >= VOTE_OCCUPIED_MIN {
                    GateVerdict::Occupied
                } else if voting >= VOTE_EMPTY_QUORUM {
                    GateVerdict::Empty
                } else {
                    GateVerdict::Uncertain
                };
                (verdict, Some(age), stale, voting, occ, node_z)
            }
            None => (GateVerdict::Uncertain, None, false, 0, 0, HashMap::new()),
        };

        match verdict {
            GateVerdict::Empty => {
                if self.empty_since.is_none() {
                    self.empty_since = Some(now);
                }
            }
            _ => self.empty_since = None,
        }
        let sustained = self
            .empty_since
            .map(|t| now.duration_since(t).as_secs_f64())
            .unwrap_or(0.0);

        let info = PresenceGateInfo {
            verdict,
            baseline_age_secs: age,
            baseline_stale: stale,
            nodes_voting: voting,
            nodes_occupied: occupied,
            sustained_empty_secs: sustained,
            node_z,
        };
        self.last_info = Some(info.clone());
        info
    }

    /// The clamp rule: only a sustained, quorum-backed Empty verdict may turn
    /// presence off. Never returns "force present".
    pub fn should_clamp_absent(&self) -> bool {
        self.last_info
            .as_ref()
            .map(|i| {
                i.verdict == GateVerdict::Empty
                    && i.sustained_empty_secs >= EMPTY_CLAMP_SUSTAIN_SECS
            })
            .unwrap_or(false)
    }

    pub fn last_info(&self) -> Option<&PresenceGateInfo> {
        self.last_info.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Deterministic LCG noise in [-1, 1] — honest per-frame RF jitter without
    /// a rand dependency. Distinct seeds give statistically-similar but
    /// non-identical streams (like real empty-room CSI across sessions).
    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64 / (u32::MAX as f64 / 2.0)) - 1.0
    }

    /// Empty-room frame: fixed multipath shape + 2% per-sample noise, times a
    /// global gain (AGC) factor. Same shape family across seeds/gains.
    fn empty_frame(seed: &mut u64, gain: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|k| gain * (10.0 + 0.5 * (k as f64 * 0.3).sin() + 0.2 * lcg(seed)))
            .collect()
    }

    /// A frame with a person present: a *static* body shadow (a person blocks
    /// and reflects paths, shifting the multipath profile shape — this is what
    /// survives a 20-frame time-average) plus micro-motion plus noise.
    fn person_frame(seed: &mut u64, n: usize, t: usize) -> Vec<f64> {
        (0..n)
            .map(|k| {
                10.0 + 0.5 * (k as f64 * 0.3).sin()
                    + 2.0 * (k as f64 * 0.5 + 1.3).sin()          // static body shadow
                    + 1.0 * ((k as f64 * 0.7 + t as f64 * 0.9).sin()) // micro-motion
                    + 0.2 * lcg(seed)
            })
            .collect()
    }

    fn flat_frame(level: f64, n: usize) -> Vec<f64> {
        vec![level; n]
    }

    fn drive(gate: &mut PresenceGate, now: Instant, frames: usize, mut f: impl FnMut(usize) -> Vec<f64>) {
        for t in 0..frames {
            for nid in 1..=6u8 {
                gate.push_frame(nid, &f(t), now);
            }
        }
    }

    fn captured_gate(now: Instant) -> PresenceGate {
        let mut g = PresenceGate::default();
        g.start_capture(now, 30.0);
        let mut seed = 42u64;
        for _t in 0..80 {
            for nid in 1..=6u8 {
                let f = empty_frame(&mut seed, 1.0, 56);
                g.push_frame(nid, &f, now);
            }
        }
        g.finalize_capture(now + Duration::from_secs(31));
        g
    }

    #[test]
    fn no_baseline_is_uncertain_never_empty() {
        let mut g = PresenceGate::default();
        let now = Instant::now();
        drive(&mut g, now, 60, |_| flat_frame(10.0, 56));
        assert_eq!(g.evaluate(now).verdict, GateVerdict::Uncertain);
        assert!(!g.should_clamp_absent());
    }

    #[test]
    fn empty_room_reads_empty_and_clamps_after_sustain() {
        let now = Instant::now();
        let mut g = captured_gate(now);
        let later = now + Duration::from_secs(60);
        // same empty-room shape family, DIFFERENT noise seed and a 1.7× global
        // gain change (AGC): L1 normalization must make this read EMPTY.
        let mut seed = 777u64;
        for _t in 0..80 {
            for nid in 1..=6u8 {
                let f = empty_frame(&mut seed, 1.7, 56);
                g.push_frame(nid, &f, later);
            }
        }
        assert_eq!(g.evaluate(later).verdict, GateVerdict::Empty);
        assert!(!g.should_clamp_absent(), "must not clamp before sustain");
        let sustained = later + Duration::from_secs(11);
        assert_eq!(g.evaluate(sustained).verdict, GateVerdict::Empty);
        assert!(g.should_clamp_absent(), "sustained empty must clamp");
    }

    #[test]
    fn rebaseline_does_not_blind_the_gate() {
        // The 2026-07-08 live gap: re-capturing a baseline made the gate go
        // `uncertain` for the whole window, leaking false-positives. It must now
        // keep scoring against the active baseline throughout a background capture.
        let now = Instant::now();
        let mut g = captured_gate(now);
        let t = now + Duration::from_secs(60);
        let mut seed = 1u64;
        for _ in 0..40 {
            for nid in 1..=6u8 {
                g.push_frame(nid, &empty_frame(&mut seed, 1.0, 56), t);
            }
        }
        assert_eq!(g.evaluate(t).verdict, GateVerdict::Empty, "empty before re-capture");
        // Kick off a background re-baseline and keep feeding empty frames.
        g.start_capture(t, 30.0);
        assert!(g.is_capturing());
        let mut worst = GateVerdict::Empty;
        let mut s2 = 2u64;
        for k in 0..40 {
            let tk = t + Duration::from_secs_f64(k as f64 * 0.2);
            for nid in 1..=6u8 {
                g.push_frame(nid, &empty_frame(&mut s2, 1.0, 56), tk);
            }
            let v = g.evaluate(tk).verdict;
            if v == GateVerdict::Uncertain {
                worst = GateVerdict::Uncertain;
            }
        }
        assert_ne!(worst, GateVerdict::Uncertain, "gate must NOT go uncertain during re-baseline");
    }

    #[test]
    fn person_shaped_deviation_reads_occupied() {
        let now = Instant::now();
        let mut g = captured_gate(now);
        let later = now + Duration::from_secs(60);
        let mut seed = 99u64;
        drive(&mut g, later, 80, |t| person_frame(&mut seed, 56, t));
        let info = g.evaluate(later);
        assert_eq!(info.verdict, GateVerdict::Occupied);
        assert!(!g.should_clamp_absent());
    }

    #[test]
    fn single_node_blip_does_not_flip_room_occupied() {
        let now = Instant::now();
        let mut g = captured_gate(now);
        let later = now + Duration::from_secs(60);
        let mut seed = 5150u64;
        for t in 0..80 {
            for nid in 1..=6u8 {
                let f = if nid == 2 {
                    person_frame(&mut seed, 56, t) // one noisy node
                } else {
                    empty_frame(&mut seed, 1.0, 56)
                };
                g.push_frame(nid, &f, later);
            }
        }
        let info = g.evaluate(later);
        assert_eq!(info.nodes_occupied, 1);
        assert_eq!(info.verdict, GateVerdict::Empty, "vote needs >=2 nodes");
    }

    #[test]
    fn stale_baseline_degrades_to_uncertain() {
        let now = Instant::now();
        let mut g = captured_gate(now);
        let much_later = now + Duration::from_secs_f64(STALE_SECS + 60.0);
        let mut seed = 31337u64;
        for _t in 0..80 {
            for nid in 1..=6u8 {
                let f = empty_frame(&mut seed, 1.0, 56);
                g.push_frame(nid, &f, much_later);
            }
        }
        let info = g.evaluate(much_later);
        assert!(info.baseline_stale);
        assert_eq!(info.verdict, GateVerdict::Uncertain);
        assert!(!g.should_clamp_absent(), "stale baseline must never clamp");
    }

    #[test]
    fn nan_frames_are_dropped_not_poisoning() {
        let now = Instant::now();
        let mut g = PresenceGate::default();
        g.start_capture(now, 30.0);
        let mut seed = 42u64;
        for t in 0..80 {
            for nid in 1..=6u8 {
                let mut f = empty_frame(&mut seed, 1.0, 56);
                if t % 7 == 0 {
                    f[3] = f64::NAN; // corrupt every 7th frame
                }
                g.push_frame(nid, &f, now);
            }
        }
        g.finalize_capture(now + Duration::from_secs(31));
        let later = now + Duration::from_secs(60);
        let mut seed2 = 4242u64;
        for _t in 0..80 {
            for nid in 1..=6u8 {
                let f = empty_frame(&mut seed2, 1.0, 56);
                g.push_frame(nid, &f, later);
            }
        }
        let info = g.evaluate(later);
        assert_eq!(info.verdict, GateVerdict::Empty, "NaNs dropped, baseline sound");
    }

    #[test]
    fn capture_with_too_few_nodes_stays_uncertain() {
        let now = Instant::now();
        let mut g = PresenceGate::default();
        g.start_capture(now, 30.0);
        let mut seed = 42u64;
        for _t in 0..80 {
            for nid in 1..=2u8 {
                // only 2 nodes < quorum
                let f = empty_frame(&mut seed, 1.0, 56);
                g.push_frame(nid, &f, now);
            }
        }
        g.finalize_capture(now + Duration::from_secs(31));
        assert_eq!(g.evaluate(now + Duration::from_secs(40)).verdict, GateVerdict::Uncertain);
    }
}
