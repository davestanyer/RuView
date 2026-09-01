#!/usr/bin/env python3
"""Train a small desk-occupancy discriminator from camera-supervised paired CSI.

This is intentionally small/exportable: it uses hand-crafted window statistics and
multinomial logistic regression, then writes a JSON artifact that can be ported to
Rust without loading Python pickles.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "mean",
    "std",
    "median",
    "p25",
    "p75",
    "max",
    "zero_ratio",
    "frame_mean_mean",
    "frame_mean_std",
    "frame_mean_min",
    "frame_mean_max",
    "subcarrier_mean_std",
    "subcarrier_mean_p10",
    "subcarrier_mean_p90",
    "diff_abs_mean",
    "diff_std",
    "diff_abs_p90",
    "band0_mean",
    "band1_mean",
    "band2_mean",
    "band3_mean",
    "band0_std",
    "band1_std",
    "band2_std",
    "band3_std",
]

DEFAULT_INPUTS = {
    "seated": [
        "data/paired-ser8-desk-calibration/seated-direct-camera-20260701_133155.paired.jsonl",
        "data/paired-ser8-desk-calibration/seated-direct-camera-20260702_172704.paired.jsonl",
        "data/paired-ser8-desk-calibration/seated-current-active-tx-20260714_100506.paired.jsonl",
        "data/paired-ser8-desk-calibration/seated-auto-20260714_163353.paired.jsonl",
        "data/paired-ser8-desk-calibration/seated-auto-20260714_170856.paired.jsonl",
    ],
    "away": [
        "data/paired-ser8-desk-calibration/away-from-desk-camera-20260702_110829.paired.jsonl",
        "data/paired-ser8-desk-calibration/away-lunch-empty-20260715_035657.paired.jsonl",
    ],
    "standing_near": [
        "data/paired-ser8-desk-calibration/standing-near-desk-camera-20260702_111207.paired.jsonl",
        "data/paired-ser8-desk-calibration/standing-current-active-tx-20260714_111845.paired.jsonl",
        "data/paired-ser8-desk-calibration/standing-auto-20260714_170211.paired.jsonl",
    ],
}


def feature_vector(row: dict) -> list[float]:
    csi = np.asarray(row["csi"], dtype=np.float32)
    if row.get("csi_shape") != [20, 56] or csi.size != 20 * 56:
        raise ValueError(f"expected csi_shape [20,56] / 1120 values, got {row.get('csi_shape')} / {csi.size}")

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


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_dataset(inputs: dict[str, list[str]]) -> tuple[np.ndarray, np.ndarray, dict]:
    xs: list[list[float]] = []
    ys: list[str] = []
    stats: dict[str, dict] = {}
    for label, file_names in inputs.items():
        label_rows: list[dict] = []
        for file_name in file_names:
            path = Path(file_name)
            rows = list(load_jsonl(path))
            label_rows.extend(rows)
            for row in rows:
                xs.append(feature_vector(row))
                ys.append(label)
        confidences = np.asarray(
            [float(r["conf"]) for r in label_rows if isinstance(r.get("conf"), (int, float))],
            dtype=np.float64,
        )
        stats[label] = {
            "paths": file_names,
            "rows": len(label_rows),
            "confidence_mean": float(confidences.mean()) if confidences.size else None,
            "confidence_min": float(confidences.min()) if confidences.size else None,
            "confidence_max": float(confidences.max()) if confidences.size else None,
        }
    return np.asarray(xs, dtype=np.float64), np.asarray(ys), stats


def export_logreg_json(model, dataset_stats: dict, metrics: dict, output: Path) -> None:
    scaler: StandardScaler = model.named_steps["standardscaler"]
    clf: LogisticRegression = model.named_steps["logisticregression"]
    artifact = {
        "kind": "ruview_desk_discriminator_logreg_v1",
        "created_by": "scripts/train-desk-discriminator.py",
        "feature_names": FEATURE_NAMES,
        "classes": clf.classes_.tolist(),
        "standardization": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "model": {
            "type": "multinomial_logistic_regression",
            "coef": clf.coef_.tolist(),
            "intercept": clf.intercept_.tolist(),
            "decision": "softmax(argmax(coef * ((x - mean) / scale) + intercept))",
        },
        "dataset": dataset_stats,
        "metrics": metrics,
        "notes": [
            "Training camera is a temporary calibration oracle, not a product dependency.",
            "This artifact is for desk occupancy discrimination: seated vs away vs standing_near.",
            "Validate on live rolling windows before product-clamping any user-facing state.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/calibration/desk_discriminator_logreg_20260702.json")
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    x, y, dataset_stats = load_dataset(DEFAULT_INPUTS)
    print(f"dataset shape={x.shape} labels={dict(Counter(y))}")
    print("dataset stats:")
    print(json.dumps(dataset_stats, indent=2))

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, x, y, cv=cv, scoring="f1_macro")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=8, stratify=y
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    labels = ["seated", "away", "standing_near"]
    metrics = {
        "cv_f1_macro_scores": [float(v) for v in cv_scores],
        "cv_f1_macro_mean": float(cv_scores.mean()),
        "holdout_accuracy": float(accuracy_score(y_test, pred)),
        "holdout_f1_macro": float(f1_score(y_test, pred, average="macro")),
        "labels": labels,
        "holdout_confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
        "classification_report": classification_report(y_test, pred, digits=3, output_dict=True),
    }

    print("metrics:")
    print(json.dumps(metrics, indent=2))
    print("human report:")
    print(classification_report(y_test, pred, digits=3))
    print("confusion matrix labels", labels)
    print(confusion_matrix(y_test, pred, labels=labels))

    # Refit on all data for the exported artifact.
    model.fit(x, y)
    output = Path(args.output)
    export_logreg_json(model, dataset_stats, metrics, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
