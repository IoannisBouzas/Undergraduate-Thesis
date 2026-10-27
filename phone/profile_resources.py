"""
Background hardware resource profiler for Android / Termux (Samsung S23 Ultra).

Run this ON the phone, inside Termux, beside the llama.cpp inference. Running it
on-device (not over adb from the laptop) keeps its timestamps on the same clock
as the inference logs (run_memory_eval.py writes query_start_ts/query_end_ts
with the same time source), so resource samples can be integrated per question
to get joules-per-answer.

Why battery is special: on Samsung One UI the raw sysfs nodes
(/sys/class/power_supply/battery/current_now, voltage_now, temp, ...) are
root-walled (Permission denied without root). So battery metrics come from
`dumpsys battery`, which works WITHOUT root and exposes:
  - current now      (signed, mA on this device)
  - voltage          (mV)
  - charge counter   (uAh)  <-- the robust energy source (coulomb counter)
  - temperature      (tenths of degC), status, level
Fallback order: dumpsys -> termux-battery-status -> raw sysfs.
CPU / GPU / thermal / RAM are read from world-readable sysfs.

ENERGY: prefer charge-counter changes over a run window, not instantaneous
current. Pair each charge interval with its measured mean battery voltage to
obtain voltage-weighted battery-side energy. Instantaneous power_w
(current * voltage) is logged too but is noisier.

IMPORTANT: MEASURE ON BATTERY (unplugged). While charging, `current now` is the
charge current, not consumption, and the charge counter rises instead of falling.
The profiler prints a warning at startup if the phone is plugged in.
"""

import time
import argparse
import csv
import json
import os
import re
import signal
import subprocess
from pathlib import Path

NOMINAL_VOLTAGE = 3.85  # fallback for energy-from-coulombs if voltage missing


def read_sysfs(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return ""


def run_cmd(args: list[str], timeout: float = 4.0) -> str:
    """Run a command, return stdout ('' on any failure). Used for dumpsys / termux-api."""
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return out.stdout or ""
    except Exception:
        return ""


class AndroidProfiler:
    def __init__(self, output_file: Path, run_id: str = ""):
        self.output_file = output_file
        self.run_id = run_id
        self._last_cpu_times = self._read_cpu_times()
        self.battery_source = "unknown"
        self.headers = [
            "run_id",
            "timestamp",
            "cpu_busy_pct", "cpu_freq_avg_mhz", "cpu_freq_max_mhz",
            "soc_temp_c",
            "gpu_busy_pct", "gpu_freq_mhz",
            "ram_used_mb", "ram_free_mb",
            "battery_source",
            "battery_current_ma", "battery_voltage_mv", "power_w",
            "battery_charge_counter_uah", "battery_temp_c",
            "battery_status", "battery_level_pct", "battery_plugged",
        ]

        if self.output_file.exists() and self.output_file.stat().st_size > 0:
            with self.output_file.open(newline="", encoding="utf-8") as f:
                existing_header = next(csv.reader(f), [])
            if existing_header != self.headers:
                raise ValueError(
                    "resource CSV schema mismatch; write to a new file or migrate the existing header"
                )
        else:
            with self.output_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    # ------------------------------------------------------------------ CPU
    def _read_cpu_times(self) -> tuple[int, int] | None:
        stat = read_sysfs("/proc/stat")
        for line in stat.splitlines():
            if not line.startswith("cpu "):
                continue
            try:
                values = [int(v) for v in line.split()[1:]]
            except ValueError:
                return None
            if len(values) < 4:
                return None
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            return idle, sum(values)
        return None

    def get_cpu_metrics(self) -> tuple[float, float, float]:
        current_times = self._read_cpu_times()
        cpu_busy_pct = 0.0
        if current_times and self._last_cpu_times:
            prev_idle, prev_total = self._last_cpu_times
            idle, total = current_times
            delta_idle = idle - prev_idle
            delta_total = total - prev_total
            if delta_total > 0:
                cpu_busy_pct = round((1.0 - (delta_idle / delta_total)) * 100.0, 2)
        if current_times:
            self._last_cpu_times = current_times

        freqs_mhz = []
        cpu_root = Path("/sys/devices/system/cpu")
        for cpu_dir in cpu_root.glob("cpu[0-9]*"):
            raw_freq = (
                read_sysfs(str(cpu_dir / "cpufreq" / "scaling_cur_freq"))
                or read_sysfs(str(cpu_dir / "cpufreq" / "cpuinfo_cur_freq"))
            )
            if not raw_freq:
                continue
            try:
                value = float(raw_freq)
            except ValueError:
                continue
            mhz = value / 1000.0
            if mhz > 10000:
                mhz = value / 1000000.0
            freqs_mhz.append(mhz)

        if not freqs_mhz:
            return cpu_busy_pct, 0.0, 0.0
        avg_mhz = round(sum(freqs_mhz) / len(freqs_mhz), 1)
        max_mhz = round(max(freqs_mhz), 1)
        return cpu_busy_pct, avg_mhz, max_mhz

    # ------------------------------------------------------------------ GPU
    def get_gpu_metrics(self) -> tuple[float, float]:
        """Adreno (kgsl). gpubusy returns 'busy total' ticks since last read."""
        busy_data = read_sysfs("/sys/class/kgsl/kgsl-3d0/gpubusy")
        gpu_pct = 0.0
        if busy_data:
            try:
                busy, total = map(float, busy_data.split())
                if total > 0:
                    gpu_pct = round((busy / total) * 100.0, 2)
            except ValueError:
                pass

        freq_data = read_sysfs("/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq")
        if not freq_data:
            freq_data = read_sysfs("/sys/class/kgsl/kgsl-3d0/gpuclk")
        gpu_mhz = 0.0
        if freq_data:
            try:
                gpu_mhz = round(float(freq_data) / 1e6, 1)  # Hz -> MHz
            except ValueError:
                pass
        return gpu_pct, gpu_mhz

    # -------------------------------------------------------------- thermal
    def get_soc_temp(self) -> float:
        """Hottest readable thermal zone, in degC (world-readable, no root)."""
        temps = []
        for tz in Path("/sys/class/thermal").glob("thermal_zone*"):
            raw = read_sysfs(str(tz / "temp"))
            if not raw:
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            # millidegC on most zones; some report tenths/degC
            if v > 1000:
                v = v / 1000.0
            elif v > 200:
                v = v / 10.0
            temps.append(v)
        return round(max(temps), 1) if temps else 0.0

    # ------------------------------------------------------------------ RAM
    def get_ram_metrics(self) -> tuple[float, float]:
        meminfo = read_sysfs("/proc/meminfo")
        total_kb = 0
        available_kb = 0
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                available_kb = int(line.split()[1])
        used_mb = round((total_kb - available_kb) / 1024.0, 1)
        free_mb = round(available_kb / 1024.0, 1)
        return used_mb, free_mb

    # -------------------------------------------------------------- battery
    @staticmethod
    def _find(pattern: str, text: str):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1) if m else None

    def get_battery_metrics(self) -> dict:
        """current (mA), voltage (mV), charge_counter (uAh), temp (C), status, level, plugged."""
        out = {
            "source": "none", "current_ma": None, "voltage_mv": None,
            "charge_counter_uah": None, "temp_c": None, "status": "",
            "level_pct": None, "plugged": None,
        }

        # 1) dumpsys battery (no root; richest, includes coulomb counter)
        raw = run_cmd(["dumpsys", "battery"])
        if raw and "current now" in raw.lower():
            out["source"] = "dumpsys"
            # Anchor to line start: dumpsys prints "Max charging voltage: 0" and
            # "Max charging current: 0" before the real values, so unanchored
            # patterns would capture the wrong (zero) line.
            cur = self._find(r"(?im)^\s*current now:\s*(-?\d+)", raw)
            volt = self._find(r"(?im)^\s*voltage:\s*(\d+)", raw)
            cc = self._find(r"(?im)^\s*charge counter:\s*(\d+)", raw)
            temp = self._find(r"(?im)^\s*temperature:\s*(-?\d+)", raw)
            level = self._find(r"(?im)^\s*level:\s*(\d+)", raw)
            status = self._find(r"(?im)^\s*status:\s*(\d+)", raw)
            usb = self._find(r"(?im)^\s*USB powered:\s*(\w+)", raw)
            ac = self._find(r"(?im)^\s*AC powered:\s*(\w+)", raw)
            out["current_ma"] = int(cur) if cur is not None else None      # Samsung: mA
            out["voltage_mv"] = int(volt) if volt is not None else None
            out["charge_counter_uah"] = int(cc) if cc is not None else None
            out["temp_c"] = round(int(temp) / 10.0, 1) if temp is not None else None
            out["level_pct"] = int(level) if level is not None else None
            out["status"] = status or ""
            out["plugged"] = (str(usb).lower() == "true") or (str(ac).lower() == "true")
            return out

        # 2) termux-battery-status (Termux:API addon; current in uA)
        raw = run_cmd(["termux-battery-status"])
        if raw.strip():
            try:
                j = json.loads(raw)
                out["source"] = "termux-api"
                cur_ua = j.get("current")
                out["current_ma"] = round(cur_ua / 1000.0, 1) if cur_ua is not None else None  # uA -> mA
                out["voltage_mv"] = j.get("voltage")            # mV (present on this device)
                out["charge_counter_uah"] = j.get("charge_counter")  # uAh (coulomb counter)
                out["temp_c"] = j.get("temperature")            # already degC
                out["level_pct"] = j.get("percentage")
                out["status"] = str(j.get("status", ""))
                out["plugged"] = str(j.get("plugged", "")).upper() not in ("", "UNPLUGGED")
                return out
            except Exception:
                pass

        # 3) raw sysfs (only if rooted)
        cur = read_sysfs("/sys/class/power_supply/battery/current_now")
        volt = read_sysfs("/sys/class/power_supply/battery/voltage_now")
        temp = read_sysfs("/sys/class/power_supply/battery/temp")
        if cur or volt:
            out["source"] = "sysfs"
            try:
                if cur:
                    out["current_ma"] = round(float(cur) / 1000.0, 1)   # uA -> mA
                if volt:
                    out["voltage_mv"] = round(float(volt) / 1000.0, 1)  # uV -> mV
                if temp:
                    out["temp_c"] = round(float(temp) / 10.0, 1)
            except ValueError:
                pass
        return out

    @staticmethod
    def compute_power_w(current_ma, voltage_mv) -> float | None:
        # signed: discharge current is negative on this device, so power_w < 0 = consuming.
        # termux-battery-status gives no voltage, so fall back to NOMINAL_VOLTAGE.
        if current_ma is None:
            return None
        v = (voltage_mv / 1000.0) if voltage_mv is not None else NOMINAL_VOLTAGE
        return round((current_ma / 1000.0) * v, 3)

    # ------------------------------------------------------------------ poll
    def poll(self):
        timestamp = time.time()
        cpu_busy_pct, cpu_freq_avg_mhz, cpu_freq_max_mhz = self.get_cpu_metrics()
        soc_temp_c = self.get_soc_temp()
        gpu_pct, gpu_mhz = self.get_gpu_metrics()
        ram_used, ram_free = self.get_ram_metrics()
        b = self.get_battery_metrics()
        self.battery_source = b["source"]
        power_w = self.compute_power_w(b["current_ma"], b["voltage_mv"])

        row = [
            self.run_id,
            timestamp,
            cpu_busy_pct, cpu_freq_avg_mhz, cpu_freq_max_mhz,
            soc_temp_c,
            gpu_pct, gpu_mhz,
            ram_used, ram_free,
            b["source"],
            b["current_ma"], b["voltage_mv"], power_w,
            b["charge_counter_uah"], b["temp_c"],
            b["status"], b["level_pct"], b["plugged"],
        ]
        with self.output_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        return b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/resources_log.csv", help="Output CSV path")
    parser.add_argument("--interval", type=float, default=0.5, help="Sampling interval in seconds")
    parser.add_argument("--run-id", default="", help="Run identifier to join resource samples with inference outputs")
    parser.add_argument("--parent-pid", type=int, default=0,
                        help="exit automatically when this PID (the launching shell) disappears; "
                             "prevents the orphaned-sampler-runs-for-hours bug if the Termux "
                             "window is frozen/closed before phone_run.sh can kill it")
    parser.add_argument("--max-minutes", type=float, default=360.0,
                        help="hard safety cap: stop sampling after this many minutes no matter what")
    parser.add_argument("--stop-on-plug", action="store_true",
                        help="stop as soon as the phone is detected plugged in (energy runs are "
                             "unplugged-only; a plug event means the measured run is over)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profiler = AndroidProfiler(out_path, run_id=args.run_id)

    # One probe up front so the user can see the battery source and a plugged warning.
    b = profiler.get_battery_metrics()
    print(f"Battery source: {b['source']} | current_ma={b['current_ma']} "
          f"voltage_mv={b['voltage_mv']} charge_counter_uah={b['charge_counter_uah']} "
          f"level={b['level_pct']}")
    if b["source"] == "none":
        print("WARNING: no battery source available. Install Termux:API (pkg install termux-api) "
              "or check that `dumpsys battery` runs in this shell.")
    if b.get("plugged"):
        print("WARNING: phone is PLUGGED IN. `current now` is charge current, not consumption. "
              "Unplug and run on battery for valid energy numbers.")

    print(f"Profiler started. Logging to {out_path} every {args.interval}s...")
    print("Press Ctrl+C to stop.")

    # SIGTERM (what `kill $PROF` sends) should stop us cleanly, not be ignored.
    stop = {"flag": False}
    def _sigterm(_signum, _frame):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, _sigterm)

    deadline = time.time() + args.max_minutes * 60.0
    reason = "signal/interrupt"
    try:
        while not stop["flag"]:
            t0 = time.time()
            b = profiler.poll()
            # guard 1: launching shell gone (frozen/closed Termux window) -> orphaned, exit
            if args.parent_pid:
                try:
                    os.kill(args.parent_pid, 0)
                except OSError:
                    reason = "parent shell exited"
                    break
            # guard 2: plugged in -> the unplugged measurement window is over
            if args.stop_on_plug and b.get("plugged"):
                reason = "phone plugged in"
                break
            # guard 3: hard safety cap
            if t0 >= deadline:
                reason = f"max-minutes={args.max_minutes} cap"
                break
            elapsed = time.time() - t0
            time.sleep(max(0.01, args.interval - elapsed))
    except KeyboardInterrupt:
        reason = "Ctrl+C"
    print(f"\nProfiler stopped ({reason}).")


if __name__ == "__main__":
    main()
