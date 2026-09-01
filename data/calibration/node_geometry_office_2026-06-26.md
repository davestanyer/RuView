# RuView room map — 2026-06-26

Source sketch: `data/calibration/office_room_map_2026-06-26.jpg`
Structured calibration file: `data/calibration/node_geometry_office_2026-06-26.yaml`

## Room setup

Coordinate assumption: origin is top-left / north-west room corner. `x` increases right/east. `y` increases down/south.

- Room: `4.14 m` wide x `5.0 m` deep.
- Router/AP: green filled circle on west/left wall, approx `(0.28, 1.72)`.
- Dave: blue smiley at desk, approx `(1.32, 1.25)`.
- Desk: green desk across the north/top area, approx rectangle `x=0.15..2.85`, `y=0.30..0.95`.

## Node placement and initial roles

| Node | Approx position | Height | Initial role | Weight | Why |
|---|---:|---|---|---:|---|
| 1 | `(0.42, 0.08)` | waist | multipath_reflection | 0.55 | West/top witness; useful context but not strongly across Dave. |
| 2 | `(1.78, 0.08)` | head | direct_path | 1.00 | Likely strong seated/desk witness; router-to-node path runs close to Dave/desk. |
| 3 | `(4.05, 0.25)` | high reach | direct_path | 0.90 | Router-to-node path passes almost through Dave's usual position; high mount should help torso/head presence. |
| 4 | `(4.05, 2.30)` | chest | multipath_reflection | 0.70 | Side/diagonal witness; useful for movement and east-side changes. |
| 5 | `(3.75, 4.95)` | knee | weak_or_dead_angle | 0.35 | Far, low, and not close to seated Dave path; use as low/floor movement corroborator. |
| 6 | `(1.95, 4.95)` | waist | multipath_reflection | 0.45 | South/back-room witness; useful for movement/background, weaker for seated presence. |

## Live status checked

Local RuView runtime: `http://127.0.0.1:3000`

- Nodes active: `1, 2, 3, 4, 5, 6`
- Fused classification at check: `absent`, presence `false`, confidence about `0.847`
- Per-node features still over-call `present_moving`/`present_still` while fused says absent.

That split means the map is high-leverage: we should not treat every node equally.

## Calibration sequence

Use short clean bursts only:

1. `absent` — Dave out of office, 2 minutes.
2. `present_still` — Dave seated at usual desk position, 2 minutes.
3. `present_moving` — Dave walking/natural movement, 2 minutes.

Before retraining, archive questionable old `train_*.jsonl` files so the active training set is current, compact, and balanced.

## Fusion intuition

- Nodes `2` and `3` should dominate seated/presence evidence if calibration confirms reliability.
- Nodes `1`, `4`, and `6` should be corroborating context/multipath witnesses.
- Node `5` should be downweighted for seated presence unless calibration proves otherwise.
- Raw per-node disagreement should remain visible; product-facing fused state should be weighted/smoothed.


## Repo artifact

This copy lives in the RuView repo next to the sketch image so agents can use the physical room geometry without depending on Dave's personal Documents path.
