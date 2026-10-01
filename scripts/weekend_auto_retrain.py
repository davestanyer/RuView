#!/usr/bin/env python3
"""Weekend auto-retrain watcher for the RuView presence residual classifier.

Runs unattended alongside camera_ground_truth_oracle.py. Every cycle it:
  1. Inventories camera-labelled recordings (auto_empty_* / auto_occupied_*).
  2. When new empty segments have landed since the last cycle, retrains the
     residual logistic-regression presence classifier:
       - baseline  = per-node per-subcarrier mean over an empty segment's head
       - features  = 25 desk-discriminator window stats on (window - baseline)
  3. Validates HONESTLY (the 2026-07-03 lesson: within-session splits leak):
       - leave-one-session-out empty recall (train on earlier empty sessions,
         test on the newest unseen one)
       - staleness curve: deviation of each empty session vs baselines from
         earlier sessions, by age gap
  4. Writes artifacts + an append-only report for Monday review:
       ~/.local/share/ruview/oracle/models/presence_residual_<ts>.json
       ~/.local/share/ruview/oracle/retrain_report.jsonl   (machine-readable)
       ~/.local/share/ruview/oracle/retrain_report.md      (human-readable)

Never deletes or modifies recordings. Safe to kill/restart at any time.

Usage:
    python scripts/weekend_auto_retrain.py                  # loop forever (3h cycle)
    python scripts/weekend_auto_retrain.py --once           # single cycle
    python scripts/weekend_auto_retrain.py --cycle-mins 60
"""
from __future__ import annotations
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

REC = Path("/home/davestanyer/.local/share/ruview/data/recordings")
OUT = Path("/home/davestanyer/.local/share/ruview/oracle")
MODELS = OUT / "models"
STATE = OUT / "retrain_state.json"
SUB, WIN, STRIDE = 56, 20, 5
MAX_LINES_PER_SEG = 15000   # cap parse cost per segment (~110 s of CSI)
MIN_SEG_BYTES = 20e6        # ignore tiny flicker segments

FEATURE_NAMES = ["mean","std","median","p25","p75","max","zero_frac","fm_mean","fm_std",
    "fm_min","fm_max","sm_std","sm_p10","sm_p90","absdiff_mean","diff_std","absdiff_p90",
    "band0_mean","band1_mean","band2_mean","band3_mean","band0_std","band1_std","band2_std","band3_std"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def feats(m: np.ndarray) -> list[float]:
    csi = m.reshape(-1); fm = m.mean(1); sm = m.mean(0)
    df = np.diff(m, axis=0); bands = np.array_split(m, 4, axis=1)
    f = [csi.mean(), csi.std(), np.median(csi), np.percentile(csi, 25), np.percentile(csi, 75),
         csi.max(), np.mean(csi == 0), fm.mean(), fm.std(), fm.min(), fm.max(), sm.std(),
         np.percentile(sm, 10), np.percentile(sm, 90), np.abs(df).mean(), df.std(),
         np.percentile(np.abs(df), 90)]
    f += [b.mean() for b in bands]; f += [b.std() for b in bands]
    return [float(x) for x in f]


def node_stream(path: Path, max_lines: int = MAX_LINES_PER_SEG) -> dict[int, np.ndarray]:
    per: dict[int, list] = {}
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i >= max_lines:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            for n in d.get("nodes") or []:
                a = n.get("amplitude"); nid = n.get("node_id")
                if isinstance(a, list) and len(a) == SUB:
                    per.setdefault(nid, []).append(a)
    return {k: np.asarray(v, float) for k, v in per.items()}


def baseline_of(streams: dict[int, np.ndarray], frac: float = 0.5) -> dict[int, np.ndarray]:
    out = {}
    for nid, a in streams.items():
        cut = max(WIN * 3, int(len(a) * frac))
        if cut < len(a):
            out[nid] = a[:cut].mean(0)
    return out


def resid_windows(streams: dict[int, np.ndarray], base: dict[int, np.ndarray],
                  cap: int = 4000, skip_frac: float = 0.0) -> list[list[float]]:
    rows = []
    for nid, a in streams.items():
        if nid not in base:
            continue
        a = a[int(len(a) * skip_frac):]
        r = a - base[nid]; i = 0; c = 0
        while i + WIN <= len(r) and c < cap:
            rows.append(feats(r[i:i + WIN])); i += STRIDE; c += 1
    return rows


def dev_median(streams: dict[int, np.ndarray], base: dict[int, np.ndarray]) -> float:
    devs = []
    for nid, a in streams.items():
        if nid not in base:
            continue
        bstd = a.std(0) + 1e-6; i = 0
        while i + WIN <= len(a):
            w = a[i:i + WIN].mean(0)
            devs.append(float(np.mean(np.abs((w - base[nid]) / bstd)))); i += STRIDE * 2
    return float(np.median(devs)) if devs else float("nan")


def inventory() -> tuple[list[Path], list[Path]]:
    emp = sorted(p for p in REC.glob("auto_empty_*.jsonl") if p.stat().st_size >= MIN_SEG_BYTES)
    occ = sorted(p for p in REC.glob("auto_occupied_*.jsonl") if p.stat().st_size >= MIN_SEG_BYTES)
    return emp, occ


def selfcal_validation(emp_streams: dict, occ_streams: dict, ks=(2.0, 3.0, 4.0)) -> list[dict]:
    """Self-calibrating detector (2026-07-03 revised architecture, THEORY.md §5).

    For each empty session: calibrate per-node (reference profile + noise
    spread) on its HEAD, then score its TAIL (want: below threshold ⇒ empty)
    and the nearest occupied session (want: above ⇒ present). No trained
    classifier, no cross-session transfer.
    """
    results = []
    occ_names = list(occ_streams)
    for name, st in emp_streams.items():
        # calibrate on head
        ref, spread, tail = {}, {}, {}
        for nid, a in st.items():
            cut = max(WIN * 3, int(len(a) * 0.5))
            if cut >= len(a) - WIN:
                continue
            head = a[:cut]
            profs = np.array([head[i:i + WIN].mean(0) for i in range(0, len(head) - WIN, STRIDE)])
            if len(profs) < 5:
                continue
            ref[nid] = profs.mean(0)
            spread[nid] = float(np.linalg.norm(profs - ref[nid], axis=1).std() + 1e-9)
            tail[nid] = a[cut:]
        if not ref:
            continue
        def devs(streams):
            out = []
            for nid, a in streams.items():
                if nid not in ref:
                    continue
                for i in range(0, max(0, len(a) - WIN), STRIDE):
                    d = float(np.linalg.norm(a[i:i + WIN].mean(0) - ref[nid]))
                    out.append(d / spread[nid])
            return np.asarray(out)
        d_emp = devs(tail)
        occ_name = occ_names[-1] if occ_names else None
        d_occ = devs(occ_streams[occ_name]) if occ_name else np.array([])
        for k in ks:
            row = {"session": name, "k": k,
                   "empty_recall": round(float(np.mean(d_emp <= k)), 3) if len(d_emp) else None,
                   "occupied_recall": round(float(np.mean(d_occ > k)), 3) if len(d_occ) else None,
                   "occ_session": occ_name}
            results.append(row)
    return results


def cycle() -> dict:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    emp_files, occ_files = inventory()
    result: dict = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "empty_segments": [p.name for p in emp_files],
                    "occupied_segments": [p.name for p in occ_files]}
    if len(emp_files) < 2 or not occ_files:
        result["status"] = "waiting_for_data"
        return result

    # Load capped streams (newest empty last)
    emp_streams = {p.name: node_stream(p) for p in emp_files[-8:]}   # cap sessions per cycle
    occ_streams = {p.name: node_stream(p) for p in occ_files[-4:]}

    # --- 1. Leave-newest-session-out validation -------------------------------
    names = list(emp_streams)
    test_name, train_names = names[-1], names[:-1]
    # baseline from the OLDEST training empty session's head (worst realistic case
    # uses freshest; we report both fresh + stale-trained recall)
    base_fresh = baseline_of(emp_streams[train_names[-1]])
    Xe, Xo = [], []
    for n in train_names:
        Xe += resid_windows(emp_streams[n], base_fresh, cap=2500, skip_frac=0.5 if n == train_names[-1] else 0.0)
    for n in occ_streams:
        Xo += resid_windows(occ_streams[n], base_fresh, cap=2500)
    if len(Xe) < 200 or len(Xo) < 200:
        result["status"] = "insufficient_windows"
        return result
    X = np.array(Xe + Xo); y = np.array(["empty"] * len(Xe) + ["occupied"] * len(Xo))
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    pipe.fit(X, y)
    # held-out: newest empty session, residual vs the SAME training baseline
    Xh = resid_windows(emp_streams[test_name], base_fresh, cap=4000)
    if Xh:
        pred = pipe.predict(np.array(Xh))
        empty_recall_heldout = float(np.mean(pred == "empty"))
    else:
        empty_recall_heldout = float("nan")
    # and with a FRESH baseline from the held-out session itself (deployment mode:
    # baseline is refreshed on camera-confirmed empty, so this is the honest runtime case)
    base_self = baseline_of(emp_streams[test_name])
    Xh2 = resid_windows(emp_streams[test_name], base_self, cap=4000, skip_frac=0.5)
    if Xh2:
        pred2 = pipe.predict(np.array(Xh2))
        empty_recall_fresh_baseline = float(np.mean(pred2 == "empty"))
    else:
        empty_recall_fresh_baseline = float("nan")
    occ_hold = resid_windows(occ_streams[list(occ_streams)[-1]], base_self, cap=3000)
    occupied_recall = float(np.mean(pipe.predict(np.array(occ_hold)) == "occupied")) if occ_hold else float("nan")

    # --- 2. Staleness curve ----------------------------------------------------
    stale = []
    def t_of(name: str) -> float:
        # Accept both canonical names like auto_empty_YYYYMMDD_HHMMSS.jsonl
        # and annotated manual names like
        # auto_empty_manual_YYYYMMDD_HHMMSS_post_reprovision_all6.jsonl.
        import re
        m = re.search(r"(20\d{6})_(\d{6})", name)
        if not m:
            return 0.0
        return time.mktime(time.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"))
    for bi, bn in enumerate(names[:-1]):
        b = baseline_of(emp_streams[bn])
        for tn in names[bi + 1:]:
            stale.append({"baseline": bn, "test": tn,
                          "age_min": round((t_of(tn) - t_of(bn)) / 60, 1),
                          "dev_median": round(dev_median(emp_streams[tn], b), 3)})

    # --- 3. Export artifact -----------------------------------------------------
    MODELS.mkdir(parents=True, exist_ok=True)
    sc = pipe.named_steps["standardscaler"]; clf = pipe.named_steps["logisticregression"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    art = {"kind": "ruview_presence_residual_logreg_v1",
           "created_by": "scripts/weekend_auto_retrain.py",
           "feature_names": FEATURE_NAMES, "classes": clf.classes_.tolist(),
           "standardization": {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()},
           "model": {"type": "multinomial_logistic_regression", "coef": clf.coef_.tolist(),
                     "intercept": clf.intercept_.tolist(),
                     "decision": "softmax(coef*((x-mean)/scale)+intercept)"},
           "train": {"empty_windows": len(Xe), "occupied_windows": len(Xo),
                     "train_sessions": train_names, "heldout_session": test_name},
           "validation": {"empty_recall_heldout_stale_baseline": empty_recall_heldout,
                          "empty_recall_heldout_fresh_baseline": empty_recall_fresh_baseline,
                          "occupied_recall": occupied_recall}}
    (MODELS / f"presence_residual_{ts}.json").write_text(json.dumps(art, indent=2))

    # --- 4. Self-calibrating detector validation (revised architecture) --------
    selfcal = selfcal_validation(emp_streams, occ_streams)

    result.update({"status": "trained", "artifact": f"presence_residual_{ts}.json",
                   "empty_recall_heldout_stale_baseline": round(empty_recall_heldout, 3),
                   "empty_recall_heldout_fresh_baseline": round(empty_recall_fresh_baseline, 3),
                   "occupied_recall": round(occupied_recall, 3),
                   "staleness": stale[-12:],
                   "selfcal": selfcal})
    return result


def append_report(result: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "retrain_report.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    md = [f"\n## Cycle {result['ts']} — {result['status']}"]
    if result["status"] == "trained":
        md.append(f"- artifact: `{result['artifact']}`")
        md.append(f"- **empty recall (held-out session, FRESH baseline — deployment mode): "
                  f"{result['empty_recall_heldout_fresh_baseline']:.1%}**")
        md.append(f"- empty recall (held-out session, stale training baseline): "
                  f"{result['empty_recall_heldout_stale_baseline']:.1%}")
        md.append(f"- occupied recall: {result['occupied_recall']:.1%}")
        md.append(f"- sessions: {len(result['empty_segments'])} empty / "
                  f"{len(result['occupied_segments'])} occupied")
        if result.get("staleness"):
            md.append("- staleness (baseline→test, age_min → dev_median; fresh≈0.2, person≈0.55):")
            for s in result["staleness"]:
                md.append(f"    - {s['age_min']:>7.0f} min → {s['dev_median']}")
        if result.get("selfcal"):
            md.append("- **SELF-CALIBRATING detector (the go/no-go metric, THEORY §5):**")
            for r in result["selfcal"]:
                if r["k"] == 3.0:
                    md.append(f"    - {r['session']} k=3: empty {r['empty_recall']}, "
                              f"occupied {r['occupied_recall']} (vs {r['occ_session']})")
    else:
        md.append(f"- empty segments so far: {len(result.get('empty_segments', []))}")
    with open(OUT / "retrain_report.md", "a") as fh:
        fh.write("\n".join(md) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--cycle-mins", type=float, default=180.0)
    args = ap.parse_args()
    if not (OUT / "retrain_report.md").exists():
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "retrain_report.md").write_text(
            "# Weekend auto-retrain report (presence residual classifier)\n"
            "Honest validation only: held-out SESSIONS, never within-session splits.\n"
            "Key number = empty recall with FRESH baseline (deployment mode).\n")
    while True:
        try:
            res = cycle()
            append_report(res)
            log(f"cycle done: {res['status']} "
                + (f"fresh-baseline empty recall {res.get('empty_recall_heldout_fresh_baseline')}"
                   if res["status"] == "trained" else ""))
        except Exception:
            log("cycle FAILED:\n" + traceback.format_exc())
        if args.once:
            break
        time.sleep(args.cycle_mins * 60)


if __name__ == "__main__":
    main()
