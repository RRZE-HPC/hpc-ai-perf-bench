#!/usr/bin/env python3
# Low-overhead AMD GPU sampler at 100 ms using sysfs/hwmon (CSV output)
# amd docs: https://docs.kernel.org/gpu/amdgpu/thermal.html

import csv
import datetime as dt
import glob
import os
import time

PERIOD_S = 0.100  # 100 ms
if os.environ.get("AMD_SYSFS_LOG"):
    print("✅ Info AMD log: AMD_SYSFS_LOG set")
else:
    print("⚠️ Warning AMD log: AMD_SYSFS_LOG not set")
OUT = os.environ.get("AMD_SYSFS_LOG", "amd_sysfs_gpu_usage.csv")

print(f"Logging AMD GPU stats to {OUT} every {PERIOD_S*1000:.0f} ms")

def read_int(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return None


def read_str(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def find_one(pattern):
    xs = glob.glob(pattern)
    return xs[0] if xs else None


def pick_temp_input(hwmon_dir):
    # pick junction/hotspot/edge if labels exist, else first temp*_input
    inputs = sorted(glob.glob(os.path.join(hwmon_dir, "temp*_input")))
    if not inputs:
        return None

    labels = {}
    for lp in glob.glob(os.path.join(hwmon_dir, "temp*_label")):
        key = os.path.basename(lp).replace("_label", "")
        labels[key] = read_str(lp).lower()

    best = None
    best_rank = 10**9
    pref = ("junction", "hotspot", "edge", "gpu")

    for ip in inputs:
        key = os.path.basename(ip).replace("_input", "")
        lab = labels.get(key, "")
        rank = next((i for i, p in enumerate(pref) if p in lab), 10**6)
        if rank < best_rank:
            best_rank = rank
            best = ip

    return best or inputs[0]


def list_amd_cards():
    # Only keep cards that look like amdgpu devices with hwmon
    cards = []
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
        dev = os.path.join(card, "device")
        hw = find_one(os.path.join(dev, "hwmon", "hwmon*"))
        if not hw:
            continue
        idx = int(os.path.basename(card).replace("card", ""))
        cards.append((idx, card, dev, hw))
    return cards


def open_fd(path):
    try:
        f = open(path, "r", encoding="utf-8", buffering=1)
        return f
    except Exception:
        return None


def read_fd_int(f):
    try:
        f.seek(0)
        s = f.read().strip()
        return int(s) if s else None
    except Exception:
        return None


def main():
    cards = list_amd_cards()
    if not cards:
        raise SystemExit("No /sys/class/drm/card*/device/hwmon/hwmon* found.")

    # Prepare per-GPU file paths + keep file descriptors open (lower overhead)
    gpus = []
    for idx, card, dev, hw in cards:
        # PCI BDF: /sys/class/drm/cardX/device is a symlink to .../0000:BB:DD.F
        pci = os.path.basename(os.path.realpath(dev))

        gpu_busy = os.path.join(dev, "gpu_busy_percent")
        if not os.path.exists(gpu_busy):
            gpu_busy = None

        # hwmon files you showed exist on your node:
        # freq1_input (gfx), freq2_input (mem), power1_input, temp*_input
        gfx = os.path.join(hw, "freq1_input")
        mem = os.path.join(hw, "freq2_input")
        pwr = os.path.join(hw, "power1_input")
        tmp = pick_temp_input(hw)

        gpu = {
            "index": idx,
            "pci": pci,
            "gpu_busy_f": open_fd(gpu_busy) if gpu_busy else None,
            "gfx_f": open_fd(gfx) if os.path.exists(gfx) else None,
            "mem_f": open_fd(mem) if os.path.exists(mem) else None,
            "pwr_f": open_fd(pwr) if os.path.exists(pwr) else None,
            "tmp_f": open_fd(tmp) if tmp and os.path.exists(tmp) else None,
        }
        gpus.append(gpu)

    print(f"Found {len(gpus)} AMD GPU(s) for monitoring.")
    print("GPU list: {gpus}".format(gpus=", ".join(f"GPU{g['index']}({g['pci']})" for g in gpus)))

    with open(OUT, "w", newline="", encoding="utf-8", buffering=1024*256) as out:
        w = csv.writer(out)
        w.writerow(
            [
                "timestamp",
                "index",
                "pci.bus_id",
                "power.draw [W]",
                "clocks.gfx [MHz]",
                "clocks.mem [MHz]",
                "utilization.gpu [%]",
                "temperature.gpu [C]",
            ]
        )


        t_next = time.perf_counter()
        while True:
            ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            for g in gpus:
                util = read_fd_int(g["gpu_busy_f"]) if g["gpu_busy_f"] else None
                gfx_hz = read_fd_int(g["gfx_f"]) if g["gfx_f"] else None
                mem_hz = read_fd_int(g["mem_f"]) if g["mem_f"] else None
                pwr_raw = read_fd_int(g["pwr_f"]) if g["pwr_f"] else None
                tmp_raw = read_fd_int(g["tmp_f"]) if g["tmp_f"] else None

                # Unit normalization (common conventions):
                # - freq*_input: Hz -> MHz (divide by 1,000,000)
                # - temp*_input: millidegrees C -> degrees C
                # - power1_input: often microwatts -> watts (but varies by driver)
                gfx_mhz = f"{gfx_hz/1_000_000.0:.0f}" if gfx_hz is not None else ""
                mem_mhz = f"{mem_hz/1_000_000.0:.0f}" if mem_hz is not None else ""
                temp_c = f"{tmp_raw/1000.0:.3f}" if tmp_raw is not None else ""
                power_w = f"{pwr_raw/1_000_000.0:.6f}" if pwr_raw is not None else ""

                w.writerow(
                    [
                        ts,
                        g["index"],
                        g["pci"],
                        power_w,
                        gfx_mhz,
                        mem_mhz,
                        "" if util is None else str(util),
                        temp_c,
                    ]
                )


            # sleep until next tick (absolute schedule, low drift)
            t_next += PERIOD_S
            while True:
                rem = t_next - time.perf_counter()
                if rem <= 0:
                    break
                time.sleep(rem if rem < 0.01 else 0.01)


if __name__ == "__main__":
    main()
