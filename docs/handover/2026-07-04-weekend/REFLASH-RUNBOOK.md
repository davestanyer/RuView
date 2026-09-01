# Reflash runbook — AGC gain-lock experiment (one node)

> **STATUS Mon 2026-07-07 ~13:00: DONE — node 5 flashed and live.**
> Identified via esptool-hold (last_seen spike), 8 MB S3, MAC `1c:db:d4:41:2f:f4`
> (LAN `.194`). Gain locked `agc=57 fft=-42`, holding. OTA PSK provisioned;
> recorded in `~/.config/wifi-densepose/esp32-provision-state/_dev_ttyUSB0.json`.
> Remounted original position (RSSI −60, matches Fri audit). Sync packets
> observed on serial with `local_us`/`epoch_us` (ADR-110 confirmed emitting).
> Compare drift after ~24 h (Tue morning).
>
> **⚠ Provisioning gotcha (hit live):** `provision.py` merges the per-PORT state
> file — which held `node_id: 6` from a previous session on this port — into
> everything it writes. A bare `--ota-psk` retry would have relabeled node 5 as
> node 6. **Always pass `--node-id <n>` explicitly when provisioning a node on
> a reused port.** (First attempt failed safe: without the IDF env,
> `nvs_partition_gen` is unavailable and provision.py exits before writing.)

**What this fixes:** ESP32 AGC rescales CSI amplitude per-packet, which is the
prime suspect for why empty-room baselines go stale in ~10–15 min (THEORY §8a).
Locking the gain makes amplitude share one scale so baselines/features hold.
**What it does NOT fix by itself:** empty-vs-person classification — it makes
that problem tractable, per the drift-research P0 recommendation.

**Status (prepared Fri night/Sat):** code merged on the branch, image built,
remote OTA ruled out (nodes have no `ota_psk` provisioned — fail-closed by
design, RuView#596; nothing in the repo ever set one). Flash is a USB job.

## What exists already

| Thing | Where |
|---|---|
| Gain-lock module | `firmware/esp32-csi-node/main/gain_lock.{c,h}` (CONFIG_CSI_GAIN_LOCK, default OFF) |
| Build profile | `firmware/esp32-csi-node/sdkconfig.gainlock` (just `CONFIG_CSI_GAIN_LOCK=y`) |
| Built image | `firmware/esp32-csi-node/build-gainlock-v54/` (IDF v5.4.2, target esp32s3) |
| `--ota-psk` provisioning | `provision.py --ota-psk <psk>` (namespace `security`, key `ota_psk`) — NEW; enables future remote flashes |
| Component dep | `espressif/esp_csi_gain_ctrl` (managed component, fetched at configure) |
| Deployed fleet | 6× S3 @ v0.7.0 (Jun 2), OTA slots ota_0/ota_1, **max app size 921,600 B** — verify the new image fits |
| Node IPs (LAN) | 192.168.195.{132,182,194,201,224,247} (MAC prefix `1c:db:d4`), OTA status: `curl http://<ip>:8032/ota/status` |

## Guinea pig: node 5

Weakest presence witness (dead-angle, weight 0.35 — lowest sensing risk) AND
the worst drifter (5.58× on Fri's staleness matrix — clearest before/after).

**Which physical box is node 5?** Unplug a candidate → within seconds
`curl -s http://127.0.0.1:3000/api/v1/nodes` shows node 5 `stale`/missing and
one LAN IP stops answering `:8032/ota/status`. That box is node 5. Plug into
the SER8 via USB-C.

## Partition-table gotcha (found Sat)

Deployed v0.7.0 nodes report **900 KB OTA slots** — an older layout matching
neither current CSV (8 MB table: 2 MB slots; 4 MB table: 1.875 MB slots). The
gain-lock image is **1.14 MB**, so:
- it can NEVER be OTA-pushed to the deployed layout — **every node's first
  upgrade must be USB** (which rewrites the partition table to current);
- after that first USB flash (+ `--ota-psk`), slots are big enough and future
  updates are remote.
USB flash offsets (0x0/0x8000/0xf000/0x20000) match the deployed layout and
NVS (0x9000) is not written, so WiFi creds/node_id survive.

## Flash procedure (Linux, SER8)

```bash
cd /home/davestanyer/RuView-upstream-main/firmware/esp32-csi-node
. /home/davestanyer/esp/esp-idf-v5.4/export.sh
ls /dev/ttyACM*            # node appears here when plugged in

# 0. Which flash size is this board? (8MB → build-gainlock-v54;
#    4MB SuperMini → build-gainlock-4mb-v54)
python -m esptool -p /dev/ttyACM0 flash_id | grep -i "flash size"

# 1. Flash the gain-lock image (app+bootloader+partition table; NVS —
#    WiFi creds, node_id — is NOT touched by `flash`)
idf.py -B build-gainlock-v54 -p /dev/ttyACM0 flash      # 8MB boards
# idf.py -B build-gl4mb-v54 -p /dev/ttyACM0 flash        # 4MB SuperMini
# (build-gl4mb-v54 uses its own isolated sdkconfig; the shared project
#  sdkconfig silently overrides SDKCONFIG_DEFAULTS otherwise — verify any
#  build with: python3 -c "import json; d=json.load(open('<dir>/config/sdkconfig.json')); print(d.get('CSI_GAIN_LOCK'), d.get('ESPTOOLPY_FLASHSIZE'))")

# 2. While it's on USB: provision an OTA PSK so FUTURE flashes are remote
python3 provision.py --port /dev/ttyACM0 --ota-psk "$(openssl rand -hex 16)"
#    → record the PSK it in the per-port state file + tell Dave/Zugg where

# 3. Watch it boot + lock
python3 -m serial.tools.miniterm /dev/ttyACM0 115200
#    EXPECT within ~10 s of joining WiFi:
#    gain_lock: GAIN LOCKED after 100 samples: agc=<X> fft=<Y>
#    then periodic "lock holding: agc=<X> fft=<Y>"
```

Remount the node in its original position (geometry matters — same spot).
Confirm rejoin: `curl -s http://127.0.0.1:3000/api/v1/nodes` → node 5 active.

## Measurement (the whole point)

Let the oracle run ≥24 h, then compare per-node stale-baseline drift
(the Fri §9 analysis, `scripts/` scratch or ask Claude to rerun):

- **Success:** node 5's empty-vs-empty deviation across sessions flattens
  toward its same-session level (~0.2) while unlocked nodes stay 2.5–5×.
- **Also watch:** "gain moved despite lock" warnings in serial (lock not
  holding), node 5 RSSI/connectivity anomalies, any drop in its occupied
  catch-rate (over-locked gain could clip).
- **If success:** roll out to remaining 5 nodes (now remotely, via OTA + PSK).
- **If no change:** drift is AP/environment-side — pivot to the software-side
  normalization + change-point invalidation (THEORY §8a P1/P2), no fleet flash.

## Rollback

Stock behavior = same source with the flag off. Reflash without the overlay:
```bash
idf.py -B build-stock -D SDKCONFIG_DEFAULTS="sdkconfig.defaults" build
idf.py -B build-stock -p /dev/ttyACM0 flash
```
(Or the v0.7.0 GH release binaries.) NVS/provisioning survives either way.
