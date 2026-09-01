# Desk discriminator calibration — 2026-07-02

Purpose: build a desk-specific classifier from camera-supervised CSI captures so RuView can distinguish `desk_seated`, `away`, and `standing_near` instead of relying on broad all-node motion heuristics.

## Inputs

Positive seated captures:

- `data/paired-ser8-desk-calibration/seated-direct-camera-20260701_133155.paired.jsonl`
  - 1,127 paired rows
  - original seated desk sample
- `data/paired-ser8-desk-calibration/seated-direct-camera-20260702_172704.paired.jsonl`
  - 2,314 paired rows
  - fresh live-distribution seated sample captured after first live model was under-confident

Negative captures:

- `data/paired-ser8-desk-calibration/away-from-desk-camera-20260702_110829.paired.jsonl`
  - 1,617 paired rows
  - away from desk / in room
- `data/paired-ser8-desk-calibration/standing-near-desk-camera-20260702_111207.paired.jsonl`
  - 834 paired rows
  - standing near desk hard negative

Total training rows after fresh seated sample: 5,892.

## Artifact

- Training script: `scripts/train-desk-discriminator.py`
- Replay/scoring script: `scripts/score-desk-discriminator.py`
- Exported model: `data/calibration/desk_discriminator_logreg_20260702.json`
- Live mounted model path: `/home/davestanyer/.local/share/ruview/data/calibration/desk_discriminator_logreg_20260702.json`

The exported JSON contains:

- 25 hand-crafted CSI window features
- class list: `away`, `seated`, `standing_near`
- StandardScaler mean/scale
- multinomial logistic regression coefficients/intercepts

Runtime scoring is CSI-only. The camera is only a calibration oracle.

## Current training metrics

After adding the fresh seated capture:

- 5-fold CV macro-F1 mean: 0.826
- holdout accuracy: 0.865
- holdout macro-F1: 0.829

Holdout confusion matrix, labels `[seated, away, standing_near]`:

```text
[[752,  63,  45],
 [  9, 340,  55],
 [  3,  24, 182]]
```

Replay checks:

- Fresh seated replay: 99.9% seated, mean seated confidence 0.995
- Away replay: 85.0% away, mean away confidence 0.846
- Standing-near replay: 86.5% standing_near, mean standing-near confidence 0.879

## Live deployment

Docker image:

- `ruvnet/wifi-densepose:desk-discriminator-20260702`

Live container:

- `ruview-sensing-latest`

After copying the retrained JSON to the mounted data path and restarting the container, logs showed the discriminator loaded from:

```text
data/calibration/desk_discriminator_logreg_20260702.json
```

Live seated verification after warmup:

- `desk_occupancy.state`: `desk_seated`
- `desk_occupancy.seated`: `true`
- confidence: about 0.91
- `model_seated`: about 0.66 over the rolling window
- note: `Camera-supervised CSI desk discriminator sees sustained seated-at-desk windows`

## Important guardrails

- Do not use this discriminator to product-clamp to `absent`. It distinguishes desk posture, not empty-room truth.
- Keep the broad-motion uncertain state as a fallback where model votes are not decisive.
- Do not lower thresholds just to make demos look better; add live-distribution samples when confidence is marginal.
- The camera angle remains acceptable for occupancy calibration but not ideal for full pose/skeleton training.
