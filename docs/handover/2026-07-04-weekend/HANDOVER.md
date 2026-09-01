# Weekend Handover — RuView Presence Detection (2026-07-03 → Monday)

**Audience:** Hermes/Zugg agent checking in over the weekend, and Dave on Monday.
**Branch:** `feat/desk-occupancy-evidence-gate` (local only — DO NOT push externally).
**Companion docs:** [ROADMAP.md](ROADMAP.md) · [THEORY.md](THEORY.md)

---

## 1. One-paragraph state

RuView's empty-office false-positive is **confirmed real and measured against camera
ground truth: presence=true ~91–93% while the room is verifiably empty** (n=116+
scored samples). The fix architecture is chosen and partially validated —
**camera-confirmed rolling empty baseline + residual features** — but the honest
held-out numbers are not there yet (37% empty recall cross-session as of Fri
evening; adjacent-session best case 94%). Two unattended processes are running
all weekend to close the gap: a camera ground-truth oracle capturing labelled
CSI, and an auto-retrain watcher producing honest validation reports every 3 h.

## 2. What is running unattended (check these first)

| Process | What | Check |
|---|---|---|
| `camera_ground_truth_oracle.py` (detached, 72 h) | Watches webcam (MediaPipe pose), labels room empty/occupied, records duty-cycled CSI segments via `/api/v1/recording`, scores RuView live vs ground truth | `pgrep -f camera_ground_truth_oracle`; log: `~/.local/share/ruview/oracle/weekend.log`; scoreboard: `~/.local/share/ruview/oracle/scoreboard.json` |
| `weekend_auto_retrain.py` (detached, 3 h cycles) | Retrains residual presence classifier on accumulated labelled segments; honest leave-newest-session-out validation + staleness curves | `pgrep -f weekend_auto_retrain`; log: `~/.local/share/ruview/oracle/retrain_watcher.log`; **report: `~/.local/share/ruview/oracle/retrain_report.md`** |
| `ruview-sensing-latest` container | Live sensing server (Phase A image `ruvnet/wifi-densepose:phase-a-20260703`), now `--restart unless-stopped` | `docker ps`; `curl -s http://127.0.0.1:3000/health` |

**Quick weekend health check (run all three):**
```bash
pgrep -f camera_ground_truth_oracle && pgrep -f weekend_auto_retrain && \
curl -sS -m5 http://127.0.0.1:3000/health | python3 -m json.tool
tail -5 ~/.local/share/ruview/oracle/retrain_report.md
```

If the oracle died: relaunch —
```bash
cd /home/davestanyer/RuView-upstream-main
setsid nohup .venv/bin/python scripts/camera_ground_truth_oracle.py \
  --minutes 4320 --empty-budget 21600 --occ-budget 7200 --empty-gap 2700 --occ-gap 900 \
  > ~/.local/share/ruview/oracle/weekend.log 2>&1 < /dev/null &
```
If the watcher died: relaunch —
```bash
setsid nohup .venv/bin/python scripts/weekend_auto_retrain.py --cycle-mins 180 \
  > ~/.local/share/ruview/oracle/retrain_watcher.log 2>&1 < /dev/null &
```

## 3. What was proven on Friday (all camera-ground-truthed)

| Finding | Number | Meaning |
|---|---|---|
| Empty-room false-positive | presence=true **91–93%** when camera-confirmed empty | The bug is real, not a labeling artifact |
| Occupied false-negative | RuView misses a seated person **~4–7%** | Secondary issue, Phase A over-conservatism |
| Static classifier across sessions | empty recall **26%** | Session drift > person effect ⇒ static models are dead ends |
| Residual vs *fresh* baseline (adjacent pair) | **94% empty / 95% occupied recall** | The signal exists; architecture is right |
| Honest multi-session held-out (Fri 17:39 cycle) | **37% empty recall** | Not deployable yet — drift between sessions still dominates trained residuals |
| Baseline staleness | good ≤ ~10 min; can jump to person-level dev within 13–15 min | Baseline must refresh often; abrupt invalidation happens |

**The single biggest open problem:** make residual features *scale/drift-invariant*
so a classifier trained on past sessions transfers. (Research agents were tasked
Friday on CSI drift normalization + published empty-vs-still techniques; findings
land in THEORY.md.)

## 4. Node / sensor audit (Fri)

- All 6 ESP32 nodes healthy at transport level: active, `last_seen ≤ 44 ms`,
  RSSI −51…−66 dBm. **No node needs replacing.**
- **Room map now exists in the repo.** Dave's sketch and derived node geometry
  were copied into `data/calibration/office_room_map_2026-06-26.jpg`,
  `data/calibration/node_geometry_office_2026-06-26.yaml`, and
  `data/calibration/node_geometry_office_2026-06-26.md`. The live API may still
  report placeholder positions `[2.0, 0.0, 1.5]`, and the `node_geometry_weight`
  desk/near split in `main.rs` is still hand-coded, not wired from geometry.
  **→ Next:** wire/consume this calibration artifact rather than asking Dave to redraw it.
- During camera-confirmed **empty**, per-node motion classifiers report
  `present_moving` ~100% on ALL nodes (node 1: 83% moving / 17% still / 0% absent).
  Systemic, not a single bad sensor ⇒ repositioning is NOT the first lever;
  the per-node classifier itself is saturated by empty-room multipath.
- Watch-list: nodes 3 and 6 have the highest RSSI variance (±8.5 / ±8.8 dB).

## 5. Do / Don't (standing rules, unchanged)

- ✅ Container restarts are pre-approved (report what changed).
- ✅ Reading logs, scoreboards, reports; running `--once` retrain cycles.
- ❌ Do NOT push code externally. Branch stays local.
- ❌ Do NOT reflash ESP32 nodes.
- ❌ Do NOT delete recordings or training data (archive if needed).
- ❌ Do NOT retrain the *adaptive* (server-side) model as a side effect —
  the weekend watcher writes standalone artifacts only; nothing touches the
  live server's models.
- Camera is a **local calibration oracle only** — frames never leave the box;
  only yes/no labels and snapshots under `~/.local/share/ruview/oracle/`.
- Dave's live ground truth wins over any sensor claim.

## 6. Where everything lives

| Thing | Path |
|---|---|
| Oracle + watcher scripts | `scripts/camera_ground_truth_oracle.py`, `scripts/weekend_auto_retrain.py` |
| Labelled CSI segments | `~/.local/share/ruview/data/recordings/auto_{empty,occupied}_*.jsonl` |
| Camera-truth timeline | `~/.local/share/ruview/oracle/labels.jsonl` (epoch + state, 2 s cadence) |
| Live scoreboard (FP/FN vs camera) | `~/.local/share/ruview/oracle/scoreboard.json` |
| Retrain reports | `~/.local/share/ruview/oracle/retrain_report.{md,jsonl}` |
| Trained artifacts | `~/.local/share/ruview/oracle/models/presence_residual_*.json` |
| First PoC artifact (Fri) | `data/calibration/presence_residual_logreg_20260703.json` |
| Phase A commit | `13dd546d` on `feat/desk-occupancy-evidence-gate` |

## 7. Monday morning checklist (for Dave)

1. `tail -30 ~/.local/share/ruview/oracle/retrain_report.md` — did fresh-baseline
   empty recall climb over the weekend as sessions accumulated?
2. `cat ~/.local/share/ruview/oracle/scoreboard.json` — weekend-long FP/FN.
3. Read THEORY.md research findings; pick normalization techniques to implement.
4. Give me the **physical node positions** (see §4).
5. Decide go/no-go on wiring the residual classifier into the Rust server
   (ROADMAP Phase R3) based on the weekend numbers.
