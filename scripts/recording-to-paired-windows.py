#!/usr/bin/env python3
"""Convert a RuView recording JSONL into paired-style CSI windows for training.

This is for labelled empty/away recordings where there is no camera keypoint row
per frame. It emits the minimal fields consumed by train-desk-discriminator.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def collect_raw_frames(path: Path) -> list[tuple[float, list[float]]]:
    raw_frames: list[tuple[float, list[float]]] = []
    for row in iter_jsonl(path):
        if row.get("type") != "sensing_update" or row.get("timestamp") is None:
            continue
        ts = float(row["timestamp"])
        for node in row.get("nodes") or []:
            amp = node.get("amplitude") or []
            if len(amp) == 56:
                raw_frames.append((ts, [float(v) for v in amp]))
    return raw_frames


def make_windows(raw_frames: list[tuple[float, list[float]]], cap: int | None) -> list[dict]:
    windows: list[dict] = []
    for start in range(0, len(raw_frames) - 20 + 1, 20):
        window = raw_frames[start : start + 20]
        csi = [v for _ts, amp in window for v in amp]
        windows.append(
            {
                "csi": csi,
                "csi_shape": [20, 56],
                "kp": None,
                "conf": 1.0,
                "n_camera_frames": 0,
                "n_persons_mode": 0,
                "n_persons_max": 0,
                "ts_start": window[0][0],
                "ts_end": window[-1][0],
            }
        )
    if cap and len(windows) > cap:
        # Evenly sample over the full capture so one burst does not dominate.
        if cap <= 1:
            windows = [windows[len(windows) // 2]]
        else:
            idxs = sorted({round(i * (len(windows) - 1) / (cap - 1)) for i in range(cap)})
            windows = [windows[i] for i in idxs]
    return windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cap", type=int, default=3000)
    args = parser.parse_args()

    raw_frames = collect_raw_frames(Path(args.recording))
    windows = make_windows(raw_frames, args.cap)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in windows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps({"recording": args.recording, "raw_frames": len(raw_frames), "windows_written": len(windows), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
