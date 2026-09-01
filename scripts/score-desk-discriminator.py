#!/usr/bin/env python3
"""Apply an exported RuView desk discriminator to paired JSONL or recording JSONL.

For paired JSONL this scores each row directly. For current RuView sensing_update
recordings, it projects each update's per-node amplitudes into raw CSI frames and
scores non-overlapping 20-frame windows, matching align-ground-truth.js.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def feature_vector_from_csi(csi_values: list[float]) -> list[float]:
    csi = np.asarray(csi_values, dtype=np.float32)
    if csi.size != 20 * 56:
        raise ValueError(f"expected 1120 CSI values, got {csi.size}")
    m = csi.reshape(20, 56)
    frame_mean = m.mean(axis=1)
    sub_mean = m.mean(axis=0)
    diff = np.diff(m, axis=0)
    bands = np.array_split(m, 4, axis=1)
    feats: list[float] = [
        float(csi.mean()),
        float(csi.std()),
        float(np.median(csi)),
        float(np.percentile(csi, 25)),
        float(np.percentile(csi, 75)),
        float(csi.max()),
        float(np.mean(csi == 0)),
        float(frame_mean.mean()),
        float(frame_mean.std()),
        float(frame_mean.min()),
        float(frame_mean.max()),
        float(sub_mean.std()),
        float(np.percentile(sub_mean, 10)),
        float(np.percentile(sub_mean, 90)),
        float(np.abs(diff).mean()),
        float(diff.std()),
        float(np.percentile(np.abs(diff), 90)),
    ]
    feats.extend(float(b.mean()) for b in bands)
    feats.extend(float(b.std()) for b in bands)
    return feats


def predict_one(artifact: dict, feats: list[float]) -> dict:
    x = np.asarray(feats, dtype=np.float64)
    mean = np.asarray(artifact["standardization"]["mean"], dtype=np.float64)
    scale = np.asarray(artifact["standardization"]["scale"], dtype=np.float64)
    coef = np.asarray(artifact["model"]["coef"], dtype=np.float64)
    intercept = np.asarray(artifact["model"]["intercept"], dtype=np.float64)
    z = (x - mean) / scale
    scores = coef @ z + intercept
    probs = softmax(scores)
    classes = artifact["classes"]
    idx = int(np.argmax(probs))
    return {
        "label": classes[idx],
        "confidence": float(probs[idx]),
        "probs": {classes[i]: float(probs[i]) for i in range(len(classes))},
    }


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def score_paired(path: Path, artifact: dict) -> list[dict]:
    out = []
    for i, row in enumerate(iter_jsonl(path)):
        pred = predict_one(artifact, feature_vector_from_csi(row["csi"]))
        pred.update({"row": i, "ts_start": row.get("ts_start"), "ts_end": row.get("ts_end")})
        out.append(pred)
    return out


def score_recording(path: Path, artifact: dict) -> list[dict]:
    raw_frames: list[tuple[float, list[float]]] = []
    for row in iter_jsonl(path):
        if row.get("type") != "sensing_update" or row.get("timestamp") is None:
            continue
        ts = float(row["timestamp"])
        for node in row.get("nodes") or []:
            amp = node.get("amplitude") or []
            if len(amp) == 56:
                raw_frames.append((ts, [float(v) for v in amp]))
    out = []
    for start in range(0, len(raw_frames) - 20 + 1, 20):
        window = raw_frames[start : start + 20]
        csi = [v for _ts, amp in window for v in amp]
        pred = predict_one(artifact, feature_vector_from_csi(csi))
        pred.update({"window": len(out), "ts_start": window[0][0], "ts_end": window[-1][0]})
        out.append(pred)
    return out


def summarize(preds: list[dict]) -> dict:
    counts: dict[str, int] = {}
    confs: dict[str, list[float]] = {}
    for p in preds:
        label = p["label"]
        counts[label] = counts.get(label, 0) + 1
        confs.setdefault(label, []).append(float(p["confidence"]))
    return {
        "windows": len(preds),
        "counts": counts,
        "fractions": {k: v / len(preds) for k, v in counts.items()} if preds else {},
        "mean_confidence_by_label": {k: sum(v) / len(v) for k, v in confs.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--model", default="data/calibration/desk_discriminator_logreg_20260702.json")
    parser.add_argument("--mode", choices=["paired", "recording"], default="paired")
    parser.add_argument("--output")
    args = parser.parse_args()

    artifact = json.loads(Path(args.model).read_text(encoding="utf-8"))
    path = Path(args.input)
    preds = score_paired(path, artifact) if args.mode == "paired" else score_recording(path, artifact)
    payload = {"input": str(path), "mode": args.mode, "summary": summarize(preds), "predictions": preds}
    text = json.dumps(payload, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
