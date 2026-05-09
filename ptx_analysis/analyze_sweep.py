#!/usr/bin/env python3
"""
Author: parallelArchitect / gb10-kernel-probe
Description: Analyze GB10/GX10 kernel sweep JSONL output and generate a human-readable report.
Usage:
  python3 ptx_analysis/analyze_sweep.py results/sweep_gb10_*.jsonl
  python3 ptx_analysis/analyze_sweep.py results/sweep_gb10_*.jsonl --plot
Time/Date: generated for local execution; report includes runtime timestamp.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as stats
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Number = Optional[float]


@dataclass
class SweepStats:
    path: Path
    rows: List[Dict[str, Any]]
    passed: List[Dict[str, Any]]
    failed: List[Dict[str, Any]]


def _escape_literal_newlines_inside_strings(s: str) -> str:
    """Repair common broken JSONL where a raw newline appears inside a quoted string."""
    out: List[str] = []
    in_string = False
    escaped = False

    for ch in s:
        if in_string and ch == "\n":
            out.append("\\n")
            continue
        out.append(ch)

        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            in_string = not in_string

    return "".join(out)


def _split_json_objects(text: str) -> List[str]:
    """Split JSONL-like text into object chunks, tolerant of bad embedded newlines."""
    starts = [m.start() for m in re.finditer(r"(?m)^\s*\{", text)]
    if not starts:
        return []
    chunks = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(errors="replace")
    rows: List[Dict[str, Any]] = []

    # Fast path: valid JSONL.
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except json.JSONDecodeError:
            rows = []
            break

    if rows:
        return rows

    # Repair path: objects may contain literal newlines inside fields.
    bad_chunks: List[Tuple[int, str]] = []
    for idx, chunk in enumerate(_split_json_objects(text), 1):
        fixed = _escape_literal_newlines_inside_strings(chunk)
        try:
            obj = json.loads(fixed)
            if isinstance(obj, dict):
                rows.append(obj)
        except json.JSONDecodeError as exc:
            bad_chunks.append((idx, str(exc)))

    if not rows:
        detail = f"; first parse error: {bad_chunks[0]}" if bad_chunks else ""
        raise RuntimeError(f"No valid JSON objects parsed from {path}{detail}")

    return rows


def fnum(v: Any) -> Number:
    try:
        if v is None:
            return None
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def values(rows: Sequence[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        x = fnum(row.get(key))
        if x is not None:
            out.append(x)
    return out


def pct_delta(first: float, last: float) -> float:
    if first == 0:
        return 0.0
    return ((last - first) / abs(first)) * 100.0


def mean(xs: Sequence[float]) -> float:
    return stats.fmean(xs) if xs else 0.0


def stdev(xs: Sequence[float]) -> float:
    return stats.pstdev(xs) if len(xs) > 1 else 0.0


def cv(xs: Sequence[float]) -> float:
    m = mean(xs)
    return stdev(xs) / m if m else 0.0


def slope(xs: Sequence[float]) -> float:
    """Simple least-squares slope per sample index."""
    n = len(xs)
    if n < 2:
        return 0.0
    xbar = (n - 1) / 2.0
    ybar = mean(xs)
    denom = sum((i - xbar) ** 2 for i in range(n))
    if denom == 0:
        return 0.0
    return sum((i - xbar) * (y - ybar) for i, y in enumerate(xs)) / denom


def classify_thermal(temps: Sequence[float]) -> str:
    if len(temps) < 4:
        return "INSUFFICIENT_DATA"
    start, end = temps[0], temps[-1]
    delta = end - start
    n = len(temps)
    first = temps[: max(2, n // 3)]
    last = temps[-max(2, n // 3):]
    first_slope = slope(first)
    last_slope = slope(last)

    if delta <= 2.0 and abs(last_slope) < 0.08:
        return "THERMALLY_STABLE"
    if first_slope > 0.10 and abs(last_slope) <= 0.08:
        return "RAPID_INITIAL_RAMP_TO_PLATEAU"
    if delta > 5.0 and last_slope > 0.08:
        return "GRADUAL_RISE"
    if last_slope < -0.05:
        return "COOLING_OR_LOAD_REDUCTION"
    return "MILD_RISE_WITH_STABILIZATION"


def classify_perf(tflops: Sequence[float]) -> str:
    if len(tflops) < 3:
        return "INSUFFICIENT_DATA"
    c = cv(tflops)
    if c < 0.03:
        return "STABLE"
    if c < 0.08:
        return "MODERATE_VARIANCE"
    return "HIGH_VARIANCE"


def detect_throttle(rows: Sequence[Dict[str, Any]]) -> Tuple[str, List[str]]:
    temps = values(rows, "gpu_temp_c")
    clocks = values(rows, "clk_sm_mhz")
    tflops = values(rows, "tflops")
    reasons: List[str] = []

    if len(tflops) >= 6:
        first_avg = mean(tflops[: max(3, len(tflops) // 4)])
        last_avg = mean(tflops[-max(3, len(tflops) // 4):])
        drop = pct_delta(first_avg, last_avg)
        if drop < -10.0:
            reasons.append(f"late throughput dropped {abs(drop):.1f}% from early-sweep average")

    if len(clocks) >= 6:
        first_clk = mean(clocks[: max(3, len(clocks) // 4)])
        last_clk = mean(clocks[-max(3, len(clocks) // 4):])
        clk_drop = pct_delta(first_clk, last_clk)
        if clk_drop < -10.0:
            reasons.append(f"late SM clock dropped {abs(clk_drop):.1f}% from early-sweep average")

    if temps and max(temps) >= 85.0:
        reasons.append(f"peak temperature reached {max(temps):.0f}C")

    return ("POSSIBLE" if reasons else "NONE_OBSERVED", reasons)


def best_and_worst(rows: Sequence[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    valid = [r for r in rows if fnum(r.get("tflops")) is not None]
    if not valid:
        return None, None
    return max(valid, key=lambda r: float(r["tflops"])), min(valid, key=lambda r: float(r["tflops"]))


def fmt(v: Any, digits: int = 2, suffix: str = "") -> str:
    x = fnum(v)
    if x is None:
        return "N/A"
    return f"{x:.{digits}f}{suffix}"


def config_line(r: Optional[Dict[str, Any]]) -> str:
    if not r:
        return "  N/A"
    return (
        f"  Config:      {r.get('config_name', 'unknown')}\n"
        f"  TB Shape:    {r.get('tb_shape', 'unknown')}\n"
        f"  Warp Shape:  {r.get('warp_shape', 'unknown')}\n"
        f"  Stages:      {r.get('stages', 'unknown')}\n"
        f"  Cluster:     {r.get('cluster_shape', 'unknown')}\n"
        f"  DType:       {r.get('dtype', 'unknown')}\n"
        f"  TFLOPS:      {fmt(r.get('tflops'), 4)}\n"
        f"  Temp:        {fmt(r.get('gpu_temp_c'), 0, 'C')}\n"
        f"  SM Clock:    {fmt(r.get('clk_sm_mhz'), 0, ' MHz')}"
    )


def build_report(stats_obj: SweepStats) -> str:
    rows = stats_obj.rows
    passed = stats_obj.passed
    failed = stats_obj.failed
    temps = values(passed, "gpu_temp_c")
    tflops = values(passed, "tflops")
    clocks = values(passed, "clk_sm_mhz")
    power = values(passed, "gpu_power_w")
    best, worst = best_and_worst(passed)
    throttle_state, throttle_reasons = detect_throttle(passed)

    platform = rows[0].get("platform", "unknown") if rows else "unknown"
    driver = rows[0].get("driver", "unknown") if rows else "unknown"
    cuda = rows[0].get("cuda_version", "unknown") if rows else "unknown"
    timestamp = rows[0].get("timestamp", "unknown") if rows else "unknown"

    temp_start = temps[0] if temps else None
    temp_peak = max(temps) if temps else None
    temp_delta = (temps[-1] - temps[0]) if len(temps) >= 2 else None

    perf_state = classify_perf(tflops)
    thermal_state = classify_thermal(temps)

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("GB10 KERNEL SWEEP ANALYZER")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Input File: {stats_obj.path}")
    lines.append(f"Report UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")
    lines.append("Platform:")
    lines.append(f"  Platform:     {platform}")
    lines.append(f"  Driver:       {driver}")
    lines.append(f"  CUDA:         {cuda}")
    lines.append(f"  Sweep Stamp:  {timestamp}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("SWEEP INTEGRITY")
    lines.append("=" * 60)
    lines.append(f"Total Rows:       {len(rows)}")
    lines.append(f"Passed Rows:      {len(passed)}")
    lines.append(f"Failed Rows:      {len(failed)}")
    lines.append(f"Sweep Integrity:  {'COMPLETE' if failed == [] else 'HAS_FAILURES'}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("THERMAL SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Start Temp:       {fmt(temp_start, 0, 'C')}")
    lines.append(f"Peak Temp:        {fmt(temp_peak, 0, 'C')}")
    lines.append(f"Final Temp:       {fmt(temps[-1] if temps else None, 0, 'C')}")
    lines.append(f"Thermal Delta:    {fmt(temp_delta, 0, 'C')}")
    lines.append(f"Average Temp:     {fmt(mean(temps) if temps else None, 1, 'C')}")
    lines.append(f"Thermal State:    {thermal_state}")
    lines.append("")
    lines.append("Interpretation:")
    if thermal_state == "RAPID_INITIAL_RAMP_TO_PLATEAU":
        lines.append("  Temperature increased early, then temperature growth flattened")
        lines.append("  as the system approached thermal steady-state.")
    elif thermal_state == "GRADUAL_RISE":
        lines.append("  Temperature rose progressively across the sustained sweep window.")
    elif thermal_state == "THERMALLY_STABLE":
        lines.append("  Temperature remained nearly flat across the sweep window.")
    else:
        lines.append("  Thermal behavior remained within observed operating range.")
    lines.append("")
    lines.append("=" * 60)
    lines.append("PERFORMANCE SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Peak TFLOPS:      {fmt(max(tflops) if tflops else None, 4)}")
    lines.append(f"Average TFLOPS:   {fmt(mean(tflops) if tflops else None, 4)}")
    lines.append(f"Minimum TFLOPS:   {fmt(min(tflops) if tflops else None, 4)}")
    lines.append(f"TFLOPS StdDev:    {fmt(stdev(tflops) if tflops else None, 4)}")
    lines.append(f"TFLOPS CV:        {fmt(cv(tflops) if tflops else None, 4)}")
    lines.append(f"Performance State:{' ' if len(perf_state) < 12 else ''}{perf_state}")
    lines.append("")
    lines.append("Clock / Power:")
    lines.append(f"  Avg SM Clock:   {fmt(mean(clocks) if clocks else None, 0, ' MHz')}")
    lines.append(f"  Min SM Clock:   {fmt(min(clocks) if clocks else None, 0, ' MHz')}")
    lines.append(f"  Max SM Clock:   {fmt(max(clocks) if clocks else None, 0, ' MHz')}")
    lines.append(f"  Avg Power:      {fmt(mean(power) if power else None, 2, ' W')}")
    lines.append(f"  Peak Power:     {fmt(max(power) if power else None, 2, ' W')}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("THROTTLE DETECTION")
    lines.append("=" * 60)
    lines.append(f"Throttle State:   {throttle_state}")
    if throttle_reasons:
        lines.append("Reasons:")
        for reason in throttle_reasons:
            lines.append(f"  - {reason}")
    else:
        lines.append("Reasons:")
        lines.append("  - No sustained throughput collapse detected")
        lines.append("  - No major SM clock collapse detected")
        lines.append("  - Peak temperature stayed below thermal-limit warning band")
    lines.append("")
    lines.append("=" * 60)
    lines.append("BEST CONFIGURATION")
    lines.append("=" * 60)
    lines.append(config_line(best))
    lines.append("")
    lines.append("=" * 60)
    lines.append("LOWEST TFLOPS CONFIGURATION")
    lines.append("=" * 60)
    lines.append(config_line(worst))
    lines.append("")
    lines.append("=" * 60)
    lines.append("FINAL INTERPRETATION")
    lines.append("=" * 60)
    lines.append(f"The system completed {len(passed)} passing sweep configurations")
    lines.append("with no observed thermal throttle signature." if throttle_state == "NONE_OBSERVED" else "with possible throttle indicators requiring review.")
    lines.append("")
    lines.append("Observed result:")
    lines.append(f"  Thermal behavior:     {thermal_state}")
    lines.append(f"  Throughput behavior:  {perf_state}")
    lines.append(f"  Throttle behavior:    {throttle_state}")
    lines.append("")
    lines.append("This report is derived from the raw JSONL sweep data. The raw JSONL")
    lines.append("remains the ground truth; this file is the human-readable analyzer layer.")
    lines.append("=" * 60)
    lines.append("")
    return "\n".join(lines)


def write_plots(stats_obj: SweepStats, outdir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("--plot requires matplotlib: python3 -m pip install matplotlib") from exc

    rows = stats_obj.passed
    x = list(range(1, len(rows) + 1))
    temps = values(rows, "gpu_temp_c")
    tflops = values(rows, "tflops")
    clocks = values(rows, "clk_sm_mhz")

    if temps:
        plt.figure()
        plt.plot(x[: len(temps)], temps)
        plt.xlabel("Sweep Configuration Index")
        plt.ylabel("GPU Temperature (C)")
        plt.title("Thermal Curve")
        plt.tight_layout()
        plt.savefig(outdir / "thermal_curve.png", dpi=160)
        plt.close()

    if tflops:
        plt.figure()
        plt.plot(x[: len(tflops)], tflops)
        plt.xlabel("Sweep Configuration Index")
        plt.ylabel("TFLOPS")
        plt.title("TFLOPS Curve")
        plt.tight_layout()
        plt.savefig(outdir / "tflops_curve.png", dpi=160)
        plt.close()

    if clocks:
        plt.figure()
        plt.plot(x[: len(clocks)], clocks)
        plt.xlabel("Sweep Configuration Index")
        plt.ylabel("SM Clock (MHz)")
        plt.title("SM Clock Stability")
        plt.tight_layout()
        plt.savefig(outdir / "sm_clock_curve.png", dpi=160)
        plt.close()

    # Lightweight heatmap: tb_shape x stages, average TFLOPS.
    buckets: Dict[Tuple[str, str], List[float]] = {}
    tb_labels: List[str] = []
    stage_labels: List[str] = []
    for r in rows:
        tb = str(r.get("tb_shape", "unknown"))
        st = str(r.get("stages", "unknown"))
        val = fnum(r.get("tflops"))
        if val is None:
            continue
        buckets.setdefault((tb, st), []).append(val)
        if tb not in tb_labels:
            tb_labels.append(tb)
        if st not in stage_labels:
            stage_labels.append(st)

    if buckets:
        matrix = []
        for tb in tb_labels:
            row = []
            for st in stage_labels:
                vals = buckets.get((tb, st), [])
                row.append(mean(vals) if vals else float("nan"))
            matrix.append(row)

        plt.figure()
        plt.imshow(matrix, aspect="auto")
        plt.xticks(range(len(stage_labels)), stage_labels)
        plt.yticks(range(len(tb_labels)), tb_labels)
        plt.xlabel("K Stages")
        plt.ylabel("Threadblock Shape")
        plt.title("Average TFLOPS Heatmap")
        plt.colorbar(label="TFLOPS")
        plt.tight_layout()
        plt.savefig(outdir / "tflops_heatmap.png", dpi=160)
        plt.close()


def analyze_file(path: Path, outdir: Optional[Path], plot: bool) -> Path:
    rows = load_jsonl(path)
    passed = [r for r in rows if str(r.get("run_status", "pass")).lower() == "pass"]
    failed = [r for r in rows if str(r.get("run_status", "pass")).lower() != "pass"]
    stats_obj = SweepStats(path=path, rows=rows, passed=passed, failed=failed)

    if outdir is None:
        outdir = path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    report = build_report(stats_obj)
    report_path = outdir / "sweep_analysis.txt"
    if len(list(Path(outdir).glob("sweep_analysis.txt"))) and report_path.exists():
        # Keep the requested default name; overwrite is intentional for latest run.
        pass
    report_path.write_text(report)

    if plot:
        write_plots(stats_obj, outdir)

    print(report)
    print(f"\nWrote report: {report_path}")
    if plot:
        print(f"Wrote plots:  {outdir / 'thermal_curve.png'}")
        print(f"              {outdir / 'tflops_curve.png'}")
        print(f"              {outdir / 'sm_clock_curve.png'}")
        print(f"              {outdir / 'tflops_heatmap.png'}")

    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze GB10/GX10 kernel sweep JSONL output.")
    parser.add_argument("jsonl", nargs="+", help="Sweep JSONL file(s)")
    parser.add_argument("--output-dir", "-o", type=Path, default=None, help="Directory for report/plots")
    parser.add_argument("--plot", action="store_true", help="Generate PNG plots")
    args = parser.parse_args()

    for file_arg in args.jsonl:
        path = Path(file_arg)
        if not path.exists():
            raise FileNotFoundError(path)
        analyze_file(path, args.output_dir, args.plot)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
