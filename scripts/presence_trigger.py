#!/usr/bin/env python3
"""Presence-triggered actions for RuView — fire a command when someone enters/leaves.

Turns RuView's presence detection into an event source: watches
`/api/v1/sensing/latest` `classification.presence`, debounces it, and runs a
shell command on the empty->present edge (ENTER) and the present->empty edge
(LEAVE). Wire it to anything on the SER8 — a spoken greeting, a sound clip, a
smart plug, a USB relay, a BLE robot ("C-3PO, say hello").

Examples
--------
# Default: C-3PO-ish spoken greeting on entry (espeak-ng)
python scripts/presence_trigger.py

# Play a sound clip on entry, nothing on leave
python scripts/presence_trigger.py --on-enter "aplay -q /home/davestanyer/c3po_hello.wav"

# Toggle a WiFi smart plug on entry / off on leave (Tasmota example)
python scripts/presence_trigger.py \
  --on-enter "curl -s http://192.168.195.50/cm?cmnd=Power%20On" \
  --on-leave "curl -s http://192.168.195.50/cm?cmnd=Power%20Off"

# Talk to a BLE robot (your own script)
python scripts/presence_trigger.py --on-enter "python3 /home/davestanyer/c3po_ble.py greet"

Notes
-----
- Uses the product presence flag (gate-clamped), so ENTER means RuView actually
  decided a person is present, not raw noise. Debounced to avoid flicker.
- Commands run in the background (a slow/blocking action never stalls detection).
"""
from __future__ import annotations
import argparse, subprocess, sys, time, urllib.request

BASE = "http://127.0.0.1:3000"
DEFAULT_ENTER = ('espeak-ng -v en+m3 -s 160 -p 30 '
                 '"Oh! Hello there. I did not see you come in."')
DEFAULT_LEAVE = ""  # quiet by default on leave


def get_presence():
    try:
        with urllib.request.urlopen(BASE + "/api/v1/sensing/latest", timeout=4) as r:
            import json
            d = json.load(r)
            return (d.get("classification") or {}).get("presence")
    except Exception:
        return None


def run(cmd, label):
    if not cmd:
        return
    print(f"[{time.strftime('%H:%M:%S')}] {label} -> running: {cmd}", flush=True)
    try:
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"    action failed: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on-enter", default=DEFAULT_ENTER,
                    help="shell command when someone ENTERS (empty->present)")
    ap.add_argument("--on-leave", default=DEFAULT_LEAVE,
                    help="shell command when the room goes EMPTY (present->empty)")
    ap.add_argument("--poll", type=float, default=0.8, help="poll interval seconds")
    ap.add_argument("--confirm", type=int, default=2,
                    help="consecutive matching samples to confirm an edge (debounce)")
    ap.add_argument("--cooldown", type=float, default=5.0,
                    help="min seconds between two ENTER fires (avoid re-triggering)")
    args = ap.parse_args()

    print(f"presence trigger up. poll={args.poll}s confirm={args.confirm} "
          f"cooldown={args.cooldown}s", flush=True)
    print(f"  on-enter: {args.on_enter or '(none)'}", flush=True)
    print(f"  on-leave: {args.on_leave or '(none)'}", flush=True)

    state = None            # confirmed presence state
    pending = None          # candidate new state
    pending_n = 0
    last_enter = -1e9

    while True:
        p = get_presence()
        if p is None:
            time.sleep(args.poll)
            continue
        if p == state:
            pending = None
            pending_n = 0
        else:
            if p == pending:
                pending_n += 1
            else:
                pending = p
                pending_n = 1
            if pending_n >= args.confirm:
                prev, state = state, p
                pending = None
                pending_n = 0
                if state and (prev is False or prev is None):
                    now = time.monotonic()
                    if now - last_enter >= args.cooldown:
                        last_enter = now
                        run(args.on_enter, "ENTER (someone walked in)")
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] ENTER (within cooldown, skipped)", flush=True)
                elif state is False and prev:
                    run(args.on_leave, "LEAVE (room now empty)")
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
