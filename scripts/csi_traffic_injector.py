#!/usr/bin/env python3
"""CSI traffic injector for RuView (ROADMAP R2/hardware, 2026-07-07).

ESP32 CSI is computed per RECEIVED WiFi frame. In a quiet/empty office the
nodes stop hearing packets and their CSI freezes at the last value (found
2026-07-07: 5/6 nodes frozen in empty recordings while all 6 were live when
Dave + devices were active). This injects a steady, low-rate stream of unicast
frames to every node so each one always has packets to sense — keeping CSI
live regardless of occupancy, the standard "active mode" practice for CSI
sensing (esp-csi, ESPectre).

Each UDP datagram to a node's IP is delivered over the air by the AP as a
unicast 802.11 frame; the node computes CSI from it even if the port is closed.
No root required. Fully reversible: kill the process and injection stops.

Usage:
    python scripts/csi_traffic_injector.py                 # 30 Hz/node, all 6
    python scripts/csi_traffic_injector.py --hz 50
    python scripts/csi_traffic_injector.py --nodes 192.168.195.132,...
"""
from __future__ import annotations
import argparse, socket, time, sys

DEFAULT_NODES = [f"192.168.195.{o}" for o in (132, 182, 194, 200, 224, 247)]
PORT = 5099  # arbitrary; frame arrival is what matters, not a listener
PAYLOAD = b"ruview-csi-keepalive"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default=",".join(DEFAULT_NODES),
                    help="comma-separated node IPs")
    ap.add_argument("--hz", type=float, default=30.0,
                    help="frames per second PER node (default 30, ~CSI rate)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--stats-secs", type=float, default=30.0)
    args = ap.parse_args()

    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    period = 1.0 / (args.hz * len(nodes))  # inter-packet gap across the round-robin
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)

    sent = 0
    t0 = time.monotonic()
    last_stats = t0
    print(f"[injector] {len(nodes)} nodes @ {args.hz:.0f} Hz each "
          f"({args.hz*len(nodes):.0f} pkt/s total) -> port {args.port}", flush=True)
    try:
        i = 0
        next_send = time.monotonic()
        while True:
            now = time.monotonic()
            if now >= next_send:
                node = nodes[i % len(nodes)]
                try:
                    sock.sendto(PAYLOAD, (node, args.port))
                    sent += 1
                except OSError:
                    pass
                i += 1
                next_send += period
                # avoid runaway if we fell behind
                if next_send < now - 0.5:
                    next_send = now + period
            else:
                time.sleep(min(period, next_send - now))
            if now - last_stats >= args.stats_secs:
                rate = sent / (now - t0)
                print(f"[injector] sent={sent} avg={rate:.0f} pkt/s over {now-t0:.0f}s",
                      flush=True)
                last_stats = now
    except KeyboardInterrupt:
        print(f"[injector] stopped after {sent} packets", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
