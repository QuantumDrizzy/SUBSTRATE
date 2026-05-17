#!/usr/bin/env python3
"""Experiment 0 data logger -- persists (timestamp, dst_nT, singlet_yield) to disk.

The SUBSTRATE GUI keeps only 240 frames in RAM. This daemon runs alongside the
motor, calls magnon_layer.run() on a fixed cadence, and appends each result to
a JSONL file that survives restarts.

Usage:
    python3 tools/exp0_logger.py               # 60s cadence
    python3 tools/exp0_logger.py --interval 30
    python3 tools/exp0_logger.py --once        # single sample and exit
    python3 tools/exp0_logger.py --export-csv  # convert log to CSV for analysis

Output:
    ~/.cache/substrate/experiment_0/log.jsonl

Each line:
    {"t": 1778782109, "utc": "2026-05-14T18:08:29Z",
     "dst_nT": -42.0, "dst_source": "noaa_dst_cache",
     "singlet_yield": 0.7931, "fidelity": 0.9812, ...}
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os as _os
import signal
import sys
import time
from pathlib import Path

_ROOT     = Path(__file__).resolve().parent.parent
_LOG_DIR  = Path.home() / ".cache" / "substrate" / "experiment_0"
_LOG_FILE = _LOG_DIR / "log.jsonl"
_MARKER   = Path.home() / ".cache" / "substrate" / "geomagnetic" / "experiment_0_marker.json"


def _ensure_dirs():
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def _worker(queue: multiprocessing.Queue, root: str):
    """Runs in a fresh subprocess — all C extensions and SDR buffers die with it."""
    import sys, os, time
    sys.path.insert(0, str(root + "/engine"))
    sys.path.insert(0, str(root + "/modules/magnon"))

    t0  = time.time()
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))
    try:
        import magnon_layer  # type: ignore
        data = magnon_layer.run().get("data", {})
        queue.put({
            "t":                 int(t0),
            "utc":               utc,
            "dst_nT":            data.get("dst_nT", 0.0),
            "dst_source":        data.get("dst_source", "unknown"),
            "singlet_yield":     data.get("singlet_yield"),
            "fidelity":          data.get("fidelity"),
            "noise_source":      data.get("noise_source"),
            "noise_power_ratio": data.get("noise_power_ratio"),
            "b_earth_nT":        data.get("b_earth_nT"),
            "solve_ms":          data.get("solve_ms"),
            "error":             data.get("error"),
        })
    except Exception as exc:
        queue.put({"t": int(t0), "utc": utc, "dst_nT": None,
                   "dst_source": "logger_error", "error": str(exc)})


def _sample():
    """Spawn a subprocess for each sample — memory fully reclaimed on exit."""
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(q, str(_ROOT)), daemon=True)
    p.start()
    p.join(timeout=120)  # 2 min hard timeout
    if p.exitcode is None:
        p.terminate()
        t0 = int(time.time())
        return {"t": t0, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                "dst_nT": None, "dst_source": "logger_error", "error": "worker timeout"}
    if not q.empty():
        return q.get_nowait()
    t0 = int(time.time())
    return {"t": t0, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            "dst_nT": None, "dst_source": "logger_error", "error": "worker returned nothing"}


def _append(record):
    _ensure_dirs()
    with _LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _print_record(record):
    dst = record.get("dst_nT")
    ys  = record.get("singlet_yield")
    err = record.get("error")
    if err:
        print(f"[{record['utc']}]  ERROR: {err}")
    else:
        print(f"[{record['utc']}]  dst={dst:+.1f} nT [{record.get('dst_source','?')}]"
              f"  Y_s={ys:.8f}  noise={record.get('noise_source','?')}"
              f"  {record.get('solve_ms',0):.1f}ms")


def run_daemon(interval):
    _ensure_dirs()
    print(f"Experiment 0 logger  |  interval={interval}s  |  {_LOG_FILE}")
    print("Ctrl+C to stop.\n")

    def _stop(sig, frame):
        print("\nLogger stopped.")
        try:
            m = json.loads(_MARKER.read_text())
            m["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _MARKER.write_text(json.dumps(m, indent=2))
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    n = 0
    while True:
        rec = _sample()
        _append(rec)
        _print_record(rec)
        n += 1
        if n % 60 == 0:
            lines = sum(1 for _ in _LOG_FILE.open())
            print(f"  [{n} samples, {lines} total lines on disk]")
        time.sleep(interval)


def run_once():
    rec = _sample()
    print(json.dumps(rec, indent=2))
    _append(rec)
    print(f"\nAppended to {_LOG_FILE}")


def export_csv(out_path=None):
    import csv
    if not _LOG_FILE.exists():
        print(f"No log file at {_LOG_FILE}")
        return
    out = Path(out_path) if out_path else _LOG_DIR / "exp0_data.csv"
    fields = ["t", "utc", "dst_nT", "dst_source", "singlet_yield",
              "fidelity", "noise_power_ratio", "noise_source", "solve_ms"]
    total = kept = 0
    with _LOG_FILE.open() as fin, out.open("w", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for line in fin:
            try:
                row = json.loads(line)
                total += 1
                if row.get("dst_source") == "dst_unavailable":
                    continue
                if row.get("singlet_yield") is None:
                    continue
                w.writerow({k: row.get(k, "") for k in fields})
                kept += 1
            except json.JSONDecodeError:
                continue
    print(f"Exported {kept}/{total} rows -> {out}  ({total-kept} dropped)")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--interval",   type=int, default=60)
    p.add_argument("--once",       action="store_true")
    p.add_argument("--export-csv", action="store_true")
    p.add_argument("--out",        default=None)
    args = p.parse_args()

    if args.export_csv:
        export_csv(args.out)
    elif args.once:
        run_once()
    else:
        run_daemon(args.interval)
