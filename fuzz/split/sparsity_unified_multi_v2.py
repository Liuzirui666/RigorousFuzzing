#!/usr/bin/env python3
"""Unified sparsity-based split-time annotator (non-split baseline).

Goal
----
Given a *non-split* FuzzBench CSV, produce:
  1) one plot per (benchmark, fuzzer): average bugs_covered vs time (hours)
     with candidate split times marked.
  2) a summary CSV with the computed split times.

This is meant to *validate a sparsity metric* by showing where the metric
would trigger splits on the non-split baseline.

Mathematical definition (what this script implements)
-----------------------------------------------------
For a fixed (benchmark b, fuzzer f), let N_i(t) be the bugs_covered value of
trial i at time t (seconds). If multiple rows share the same
(b,f,trial_id,time), we set N_i(t) := max bugs_covered among them.

Define the *mean curve* over m trials:
    \bar N(t) = (1/m) * sum_{i=1..m} N_i(t)
(implemented by averaging at each timestamp and then linear interpolation).

Let T be the analysis end time (hours), \bar N(T) the endpoint value.
Define the total-average rate:
    r_tot = \bar N(T) / T

Define a post-t rate r_after(t) in one of two ways:
  - tail mode:
        r_after(t) = (\bar N(T) - \bar N(t)) / (T - t)
  - window mode (window length W hours):
        r_after(t) = (\bar N(min(t+W, T)) - \bar N(t)) / min(W, T-t)

Define the *relative sparsity ratio*:
    rho(t) = r_after(t) / r_tot

Persistence (anti-spike): given persistence H hours, define
    rho_H(t) = max_{u in [t, t+H]} rho(u)

Sparsity-zone entry time for threshold theta (and start time S):
    z_theta = min{ t >= S : rho_H(t) <= theta }

Split time for threshold theta:
  - if BUG_TRIGGER_AFTER_ZONE = True:
        s_theta = first time > z_theta where \bar N(t) increases
        (i.e., first detected new bug after entering sparsity zone)
        If no increase exists, s_theta = None (or fallback as configured).
  - else:
        s_theta = z_theta

Unifying across benchmarks
-------------------------
We avoid hand-labelling "bug-rich" vs "bug-sparse".
Instead, we compute a *richness score* B for each (b,f) from the same data,
then map B continuously to a pair-specific earliest-split start time
S_{b,f} in [START_HOUR_MIN, START_HOUR_MAX].

Default richness score:
    B = log1p( \bar N(T) )

Within each fuzzer f, compute two quantiles of B:
    q_low = Q_{Q_LOW}(B), q_high = Q_{Q_HIGH}(B)

Normalize and clamp:
    u = clip((B - q_low) / (q_high - q_low), 0, 1)

Continuous start-time mapping (power family):
    S_{b,f} = START_HOUR_MIN + (START_HOUR_MAX - START_HOUR_MIN) * u^p
where p = START_HOUR_POWER.

Optionally, AUTO_FIT_POWER can choose p so that a pivot quantile (e.g. median)
lands at a desired pivot hour (e.g. 12h).

Configuration
-------------
Edit the CONFIG section below. The script also accepts an optional input path
as argv[1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG (edit here)
# =========================

INPUT_CSV = Path("data.csv")

# Optional filters (set to None for "no filter")
FILTER_FUZZERS: Optional[List[str]] = ["afl","aflfast","aflsmart","aflplusplus","fairfuzz","mopt", "honggfuzz","libfuzzer"]

# FILTER_BENCHMARKS: Optional[List[str]] = None
FILTER_BENCHMARKS = [
    "arrow_parquet-arrow-fuzz",
    "libhevc_hevc_dec_fuzzer",
    "stb_stbi_read_fuzzer",
    "poppler_pdf_fuzzer",
    "ffmpeg_ffmpeg_demuxer_fuzzer",
    "matio_matio_fuzzer",
    "openh264_decoder_fuzzer",
    "grok_grk_decompress_fuzzer",
    "php_php-fuzz-parser-2020-07-25",
    "libhtp_fuzz_htp",
]  

# Output directory
OUT_DIR = Path("plots")

# Time handling
MAX_TIME_SECONDS: Optional[int] = 23 * 3600  # None = use max available per pair
SNAPSHOT_SECONDS = 900  # used only to set default min gaps/window discretization

# Sparsity ratio definition
RATE_MODE = "window"  # "tail" or "window"
WINDOW_HOURS = 1.0  # only used if RATE_MODE == "window"
PERSIST_HOURS = 0.0

# Threshold list (you can change length; the script will compute split1..splitK)
THRESHOLDS: List[float] = [0.5, 0.3, 0.15]

# Whether to convert zone-entry time -> first bug after zone
BUG_TRIGGER_AFTER_ZONE = True

# If no new bug appears after zone-entry:
#   - "none": keep split time as None
#   - "zone": fallback to zone-entry time
#   - "end": fallback to end time T
NO_BUG_AFTER_ZONE_FALLBACK = "none"  # "none" | "zone" | "end"

# Minimum gap between split points (hours) when producing multiple split marks
MIN_GAP_HOURS = 1.5

# Do not perform any split within the last NO_SPLIT_LAST_HOURS of the run (hours).
# Example: with T=23h and NO_SPLIT_LAST_HOURS=2, we disallow splits after 21h.
NO_SPLIT_LAST_HOURS = 2.0

# Produce at most MAX_SPLITS split marks (typically 2 or 3). If a later split is infeasible
# under the constraints (MIN_GAP_HOURS, NO_SPLIT_LAST_HOURS), it will be set to None.
MAX_SPLITS = 3
MIN_SPLITS = 2  # informational; we do not force feasibility

# ---------- Unified mapping: richness -> earliest allowed split time S_{b,f} ----------

# Richness score: based on endpoint mean \bar N(T)
RICHNESS_METRIC = "bugs_end"  # "bugs_end" or "total_rate"
RICHNESS_TRANSFORM = "none"  # "log1p" or "none"

# Quantiles (editable!) used for normalization within each fuzzer
Q_LOW = 0.2
Q_HIGH = 0.6

# Continuous start-hour range
START_HOUR_MIN = 8.0
START_HOUR_MAX = 16.0

# Shape of mapping: S = min + (max-min) * u^p
START_HOUR_POWER = 1.0

# Auto-fit p so that pivot quantile maps to pivot hour (uses the *same* dataset)
AUTO_FIT_POWER = True
PIVOT_Q = 0.50          # which quantile of richness to pin (e.g., median)
PIVOT_HOUR = 12.0       # desired start hour at that pivot quantile

# If you want to *disable* auto-fit and manually tune p, set AUTO_FIT_POWER=False.

# Save format
SAVE_PNG = True
SAVE_PDF = False

# Plot aesthetics
PLOT_DPI = 200

# Marker style for zone/split points (per fuzzer curve)
MARKER_SIZE = 120  # scatter marker area
ZONE_MARKER = 'o'
SPLIT_MARKER = 'X'

# =========================
# END CONFIG
# =========================


@dataclass
class PairSummary:
    benchmark: str
    fuzzer: str
    n_trials: int
    t_end_h: float
    bugs_end_avg: float
    richness_raw: float
    richness_used: float
    q_low: float
    q_high: float
    u_norm: float
    start_h: float
    start_power: float
    rate_mode: str
    window_h: float
    persist_h: float
    thresholds: List[float]
    zone_times_h: List[Optional[float]]
    split_times_h: List[Optional[float]]


def _read_csv_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path.resolve()}")
    if path.suffix == ".gz":
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def _ensure_cols(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Have: {list(df.columns)}")


def _to_hours(seconds: np.ndarray) -> np.ndarray:
    return seconds.astype(float) / 3600.0


def _interp_at(times: np.ndarray, values: np.ndarray, t: float) -> float:
    """Linear interpolation on (times, values). times must be sorted."""
    if len(times) == 0:
        return 0.0
    if t <= times[0]:
        return float(values[0])
    if t >= times[-1]:
        return float(values[-1])
    return float(np.interp(t, times, values))


def _compute_ratio_series(
    times_h: np.ndarray,
    n_h: np.ndarray,
    t_end_h: float,
    n_end: float,
    rate_mode: str,
    window_h: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute rho(t) on each grid time."""
    if t_end_h <= 0:
        return np.zeros_like(times_h, dtype=float)
    if n_end <= 0:
        return np.zeros_like(times_h, dtype=float)

    total_rate = n_end / max(t_end_h, eps)
    ratios = np.zeros_like(times_h, dtype=float)

    for i, t in enumerate(times_h):
        if t < 0:
            ratios[i] = 0.0
            continue

        if rate_mode == "tail":
            dt = max(t_end_h - t, eps)
            rate = (n_end - n_h[i]) / dt
        elif rate_mode == "window":
            t2 = min(t + window_h, t_end_h)
            n2 = _interp_at(times_h, n_h, t2)
            dt = max(t2 - t, eps)
            rate = (n2 - n_h[i]) / dt
        else:
            raise ValueError(f"Unknown RATE_MODE: {rate_mode}")

        ratios[i] = rate / max(total_rate, eps)

    return np.clip(ratios, 0.0, 10.0)


def _future_window_max(x: np.ndarray, window_len: int) -> np.ndarray:
    """For each i, return max(x[i:i+window_len]) (clamped at end)."""
    n = len(x)
    out = np.empty_like(x)
    wl = max(window_len, 1)
    for i in range(n):
        j2 = min(n, i + wl)
        out[i] = float(np.max(x[i:j2]))
    return out


def _find_zone_and_split_times(
    times_h: np.ndarray,
    ratios: np.ndarray,
    thresholds: Sequence[float],
    start_h: float,
    persist_h: float,
    min_gap_h: float,
    t_end_h: float,
    no_split_last_h: float,
    n_h: np.ndarray,
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Compute (zone_times, split_times) sequentially.

    Key constraints enforced on *final split times*:
      1) Each split must be at least min_gap_h after the previous split.
      2) No splits are allowed in the last no_split_last_h hours of the run.

    Note: We search zone times starting from max(start_h, prev_split+min_gap_h), so that
    the eventual split time (which is >= zone time) also satisfies the gap.
    """
    if len(times_h) == 0:
        return ([None for _ in thresholds], [None for _ in thresholds])

    # Cutoff: splits must happen at or before this time.
    cutoff_h = float(t_end_h) - float(no_split_last_h)

    dt = float(np.median(np.diff(times_h))) if len(times_h) > 1 else (SNAPSHOT_SECONDS / 3600.0)
    win_len = max(1, int(math.ceil(persist_h / max(dt, 1e-6))))
    future_max = _future_window_max(ratios, win_len)

    zone_times: List[Optional[float]] = []
    split_times: List[Optional[float]] = []

    prev_split: Optional[float] = None

    for theta in thresholds:
        # Earliest allowed zone search time for this split index.
        earliest = float(start_h)
        if prev_split is not None:
            earliest = max(earliest, float(prev_split) + float(min_gap_h))

        # If we are already beyond cutoff, no more splits.
        if earliest > cutoff_h + 1e-12:
            zone_times.append(None)
            split_times.append(None)
            continue

        # Find zone entry z (only searching up to cutoff since split must occur <= cutoff).
        z: Optional[float] = None
        for i, t in enumerate(times_h):
            if t < earliest:
                continue
            if t > cutoff_h + 1e-12:
                break
            if future_max[i] <= theta:
                z = float(t)
                break

        zone_times.append(z)

        if z is None:
            split_times.append(None)
            continue

        # Convert zone-entry to split time.
        if BUG_TRIGGER_AFTER_ZONE:
            s = _first_increase_after(times_h, n_h, float(z))
            if s is None:
                if NO_BUG_AFTER_ZONE_FALLBACK == 'none':
                    s = None
                elif NO_BUG_AFTER_ZONE_FALLBACK == 'zone':
                    s = float(z)
                elif NO_BUG_AFTER_ZONE_FALLBACK == 'end':
                    s = float(t_end_h)
                else:
                    raise ValueError(f"Unknown NO_BUG_AFTER_ZONE_FALLBACK: {NO_BUG_AFTER_ZONE_FALLBACK}")
        else:
            s = float(z)

        # Enforce cutoff and min-gap on the final split time.
        if s is None:
            split_times.append(None)
            continue

        if float(s) > cutoff_h + 1e-12:
            # Disallow splits in last no_split_last_h hours.
            split_times.append(None)
            continue

        if prev_split is not None and float(s) < float(prev_split) + float(min_gap_h) - 1e-12:
            # Should be rare because we constrain z via prev_split, but keep as safety.
            split_times.append(None)
            continue

        split_times.append(float(s))
        prev_split = float(s)

    return zone_times, split_times


def _first_increase_after(times_h: np.ndarray, n_h: np.ndarray, after_h: float) -> Optional[float]:
    """First time > after_h where bugs_covered increases (on the mean curve)."""
    if len(times_h) < 2:
        return None

    n_mon = np.maximum.accumulate(n_h)  # guard
    for i in range(1, len(times_h)):
        if times_h[i] <= after_h:
            continue
        if n_mon[i] > n_mon[i - 1]:
            return float(times_h[i])
    return None


def _apply_richness_transform(x: float, transform: str) -> float:
    if transform == "none":
        return float(x)
    if transform == "log1p":
        return float(np.log1p(max(x, 0.0)))
    raise ValueError(f"Unknown RICHNESS_TRANSFORM: {transform}")


def _compute_richness_used(n_end: float, t_end_h: float) -> Tuple[float, float]:
    """Return (raw, used) richness."""
    raw = float(n_end)
    if RICHNESS_METRIC == "bugs_end":
        used_raw = raw
    elif RICHNESS_METRIC == "total_rate":
        used_raw = raw / max(t_end_h, 1e-12)
    else:
        raise ValueError(f"Unknown RICHNESS_METRIC: {RICHNESS_METRIC}")

    used = _apply_richness_transform(used_raw, RICHNESS_TRANSFORM)
    return used_raw, used


def _auto_fit_power(u_pivot: float, min_h: float, max_h: float, pivot_h: float) -> Optional[float]:
    """Solve for p in min + (max-min)*u^p = pivot_h. Returns None if ill-posed."""
    if not (0.0 < u_pivot < 1.0):
        return None
    y = (pivot_h - min_h) / max(max_h - min_h, 1e-12)
    if not (0.0 < y < 1.0):
        return None
    try:
        return float(math.log(y) / math.log(u_pivot))
    except (ValueError, ZeroDivisionError):
        return None


def _compute_start_hours_per_fuzzer(richness_used_by_pair: Dict[Tuple[str, str], float]) -> Tuple[Dict[Tuple[str, str], float], Dict[str, Dict[str, float]]]:
    """Compute start_h per (bench,fuzzer) with continuous mapping in [min,max].

    Returns:
      - start_h_by_pair
      - stats_by_fuzzer: {fuzzer: {q_low, q_high, pivot_u, power_used}}
    """
    # group by fuzzer
    by_fuzzer: Dict[str, List[Tuple[Tuple[str, str], float]]] = {}
    for key, val in richness_used_by_pair.items():
        _, f = key
        by_fuzzer.setdefault(f, []).append((key, float(val)))

    start_h_by_pair: Dict[Tuple[str, str], float] = {}
    stats_by_fuzzer: Dict[str, Dict[str, float]] = {}

    for fuzzer, items in by_fuzzer.items():
        vals = np.array([v for _, v in items], dtype=float)
        if len(vals) == 0:
            continue

        q_low = float(np.quantile(vals, Q_LOW))
        q_high = float(np.quantile(vals, Q_HIGH))
        denom = max(q_high - q_low, 1e-12)

        # Optional auto-fit power using pivot quantile
        p_use = float(START_HOUR_POWER)
        pivot_u = float("nan")
        if AUTO_FIT_POWER and len(vals) >= 3:
            v_pivot = float(np.quantile(vals, PIVOT_Q))
            pivot_u = float(np.clip((v_pivot - q_low) / denom, 0.0, 1.0))
            p_fit = _auto_fit_power(pivot_u, START_HOUR_MIN, START_HOUR_MAX, PIVOT_HOUR)
            if p_fit is not None and math.isfinite(p_fit) and p_fit > 0:
                p_use = p_fit

        stats_by_fuzzer[fuzzer] = {
            "q_low": q_low,
            "q_high": q_high,
            "pivot_u": pivot_u,
            "power_used": p_use,
        }

        for key, v in items:
            u = float(np.clip((v - q_low) / denom, 0.0, 1.0))
            start_h = START_HOUR_MIN + (START_HOUR_MAX - START_HOUR_MIN) * (u ** p_use)
            start_h_by_pair[key] = float(np.clip(start_h, START_HOUR_MIN, START_HOUR_MAX))

    return start_h_by_pair, stats_by_fuzzer


def main() -> None:
    import sys

    in_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else INPUT_CSV

    df = _read_csv_any(in_path)
    _ensure_cols(df, ["benchmark", "fuzzer", "trial_id", "time", "bugs_covered"])

    # Normalize
    df = df.copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["bugs_covered"] = pd.to_numeric(df["bugs_covered"], errors="coerce")
    df = df.dropna(subset=["time", "bugs_covered", "benchmark", "fuzzer", "trial_id"])

    if FILTER_FUZZERS is not None:
        df = df[df["fuzzer"].isin(FILTER_FUZZERS)]
    if FILTER_BENCHMARKS is not None:
        df = df[df["benchmark"].isin(FILTER_BENCHMARKS)]

    if MAX_TIME_SECONDS is not None:
        df = df[df["time"] <= MAX_TIME_SECONDS]

    # Dedup within trial-time
    df = (
        df.groupby(["benchmark", "fuzzer", "trial_id", "time"], as_index=False)["bugs_covered"]
        .max()
        .sort_values(["benchmark", "fuzzer", "trial_id", "time"])
    )

    # Mean curve per (benchmark,fuzzer,time)
    mean_df = (
        df.groupby(["benchmark", "fuzzer", "time"], as_index=False)["bugs_covered"]
        .mean()
        .sort_values(["benchmark", "fuzzer", "time"])
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Precompute richness per pair (needs t_end and n_end)
    pair_info: Dict[Tuple[str, str], Dict[str, float]] = {}

    for (bench, fuzzer), g in mean_df.groupby(["benchmark", "fuzzer"], sort=True):
        g = g.sort_values("time")
        times_s = g["time"].to_numpy(dtype=float)
        times_h = _to_hours(times_s)
        n_h = g["bugs_covered"].to_numpy(dtype=float)

        if MAX_TIME_SECONDS is not None:
            t_end_h = MAX_TIME_SECONDS / 3600.0
        else:
            t_end_h = float(times_h[-1])

        n_end = _interp_at(times_h, n_h, t_end_h)
        rich_raw, rich_used = _compute_richness_used(float(n_end), float(t_end_h))

        pair_info[(bench, fuzzer)] = {
            "t_end_h": float(t_end_h),
            "n_end": float(n_end),
            "rich_raw": float(rich_raw),
            "rich_used": float(rich_used),
        }

    richness_used_by_pair = {k: v["rich_used"] for k, v in pair_info.items()}
    start_h_by_pair, stats_by_fuzzer = _compute_start_hours_per_fuzzer(richness_used_by_pair)

    # Summaries
    summaries: List[PairSummary] = []

    # For per-benchmark plots: store per (benchmark,fuzzer) curves + marks
    pair_plot_data: Dict[Tuple[str, str], Dict[str, object]] = {}

    for (bench, fuzzer), g in mean_df.groupby(["benchmark", "fuzzer"], sort=True):
        g = g.sort_values("time")
        times_s = g["time"].to_numpy(dtype=float)
        times_h = _to_hours(times_s)
        n_h = g["bugs_covered"].to_numpy(dtype=float)

        info = pair_info[(bench, fuzzer)]
        t_end_h = float(info["t_end_h"])
        n_end = float(info["n_end"])
        rich_raw = float(info["rich_raw"])
        rich_used = float(info["rich_used"])

        # Count trials
        n_trials = int(df[(df["benchmark"] == bench) & (df["fuzzer"] == fuzzer)]["trial_id"].nunique())

        # Start hour computed from richness mapping
        start_h = float(start_h_by_pair.get((bench, fuzzer), START_HOUR_MIN))

        thresholds_use = list(THRESHOLDS)[: int(MAX_SPLITS)]

        # Ratios + zones
        ratios = _compute_ratio_series(
            times_h=times_h,
            n_h=n_h,
            t_end_h=t_end_h,
            n_end=n_end,
            rate_mode=RATE_MODE,
            window_h=WINDOW_HOURS,
        )
        zone_times, split_times = _find_zone_and_split_times(
            times_h=times_h,
            ratios=ratios,
            thresholds=thresholds_use,
            start_h=start_h,
            persist_h=PERSIST_HOURS,
            min_gap_h=MIN_GAP_HOURS,
            t_end_h=t_end_h,
            no_split_last_h=NO_SPLIT_LAST_HOURS,
            n_h=n_h,
        )

        # Store data for per-benchmark plots (overlay all fuzzers)
        pair_plot_data[(bench, fuzzer)] = {
            'times_h': times_h,
            'n_h': n_h,
            'thresholds': thresholds_use,
            'zone_times': zone_times,
            'split_times': split_times,
            'start_h': start_h,
            'n_trials': n_trials,
        }

        # fuzzer stats for q_low/q_high/power
        fstats = stats_by_fuzzer.get(fuzzer, {"q_low": float("nan"), "q_high": float("nan"), "pivot_u": float("nan"), "power_used": float(START_HOUR_POWER)})
        q_low = float(fstats["q_low"])
        q_high = float(fstats["q_high"])
        p_use = float(fstats["power_used"])
        denom = max(q_high - q_low, 1e-12)
        u_norm = float(np.clip((rich_used - q_low) / denom, 0.0, 1.0)) if math.isfinite(q_low) and math.isfinite(q_high) else float("nan")

        summaries.append(
            PairSummary(
                benchmark=bench,
                fuzzer=fuzzer,
                n_trials=n_trials,
                t_end_h=t_end_h,
                bugs_end_avg=float(n_end),
                richness_raw=float(rich_raw),
                richness_used=float(rich_used),
                q_low=q_low,
                q_high=q_high,
                u_norm=u_norm,
                start_h=float(start_h),
                start_power=p_use,
                rate_mode=RATE_MODE,
                window_h=(WINDOW_HOURS if RATE_MODE == "window" else 0.0),
                persist_h=PERSIST_HOURS,
                thresholds=list(thresholds_use),
                zone_times_h=zone_times,
                split_times_h=split_times,
            )
        )


    # --- Plots per benchmark (overlay all selected fuzzers) ---
    def _safe_bench_name(s: str) -> str:
        return s.replace("/", "_")

    benchmarks = sorted({b for (b, _) in pair_plot_data.keys()})
    for bench in benchmarks:
        fig = plt.figure()
        fuzzers = sorted([f for (b, f) in pair_plot_data.keys() if b == bench])

        for fuzzer in fuzzers:
            pdct = pair_plot_data[(bench, fuzzer)]
            times_h = pdct["times_h"]
            n_h = pdct["n_h"]
            zone_times = pdct["zone_times"]
            split_times = pdct["split_times"]
            start_h = float(pdct["start_h"])
            n_trials = int(pdct["n_trials"])

            label = f"{fuzzer} (S={start_h:.2f}h, trials={n_trials})"
            (line,) = plt.plot(times_h, n_h, linewidth=2, label=label)
            color = line.get_color()

            # Mark zone/split as large points on the curve and label with time
            thresholds_use = pdct.get('thresholds', THRESHOLDS)
            for k, (theta, z, s) in enumerate(zip(thresholds_use, zone_times, split_times), start=1):
                # Zone point
                if z is not None:
                    zf = float(z)
                    yz = _interp_at(times_h, n_h, zf)
                    plt.scatter([zf], [yz], s=MARKER_SIZE, marker=ZONE_MARKER, color=color, zorder=6)
                    plt.annotate(
                        f"z{k}@{zf:.2f}h",
                        (zf, yz),
                        textcoords="offset points",
                        xytext=(4, 6 + 10 * k),
                        ha="left",
                        va="bottom",
                        fontsize=8,
                        color=color,
                    )

                # Split point
                if s is not None:
                    sf = float(s)
                    ys = _interp_at(times_h, n_h, sf)
                    plt.scatter([sf], [ys], s=MARKER_SIZE, marker=SPLIT_MARKER, color=color, zorder=7)
                    # If split==zone, avoid duplicating text by slightly different label
                    tag = "s" if (z is None or abs(sf - float(z)) > 1e-6) else "zs"
                    plt.annotate(
                        f"{tag}{k}@{sf:.2f}h",
                        (sf, ys),
                        textcoords="offset points",
                        xytext=(4, -10 - 10 * k),
                        ha="left",
                        va="top",
                        fontsize=8,
                        color=color,
                    )

        plt.xlabel("Time (hours)")
        plt.ylabel("Average bugs_covered")
        plt.title(f"{bench} | avg bugs_covered (non-split) | RATE_MODE={RATE_MODE}")
        plt.legend(fontsize=8)
        plt.tight_layout()

        safe_bench = _safe_bench_name(bench)
        if SAVE_PNG:
            fig.savefig(OUT_DIR / f"{safe_bench}.png", dpi=PLOT_DPI)
        if SAVE_PDF:
            fig.savefig(OUT_DIR / f"{safe_bench}.pdf")
        plt.close(fig)

    # Summary CSV
    rows: List[Dict[str, object]] = []
    for s in summaries:
        row: Dict[str, object] = {
            "benchmark": s.benchmark,
            "fuzzer": s.fuzzer,
            "n_trials": s.n_trials,
            "t_end_h": s.t_end_h,
            "bugs_end_avg": s.bugs_end_avg,
            "richness_raw": s.richness_raw,
            "richness_used": s.richness_used,
            "q_low": s.q_low,
            "q_high": s.q_high,
            "u_norm": s.u_norm,
            "start_h": s.start_h,
            "start_power": s.start_power,
            "rate_mode": s.rate_mode,
            "window_h": s.window_h,
            "persist_h": s.persist_h,
            "thresholds": ",".join(str(x) for x in s.thresholds),
            "bug_trigger_after_zone": BUG_TRIGGER_AFTER_ZONE,
            "no_bug_fallback": NO_BUG_AFTER_ZONE_FALLBACK,
            "richness_metric": RICHNESS_METRIC,
            "richness_transform": RICHNESS_TRANSFORM,
            "Q_LOW": Q_LOW,
            "Q_HIGH": Q_HIGH,
            "START_HOUR_MIN": START_HOUR_MIN,
            "START_HOUR_MAX": START_HOUR_MAX,
            "AUTO_FIT_POWER": AUTO_FIT_POWER,
            "PIVOT_Q": PIVOT_Q,
            "PIVOT_HOUR": PIVOT_HOUR,
            "MIN_GAP_HOURS": MIN_GAP_HOURS,
            "NO_SPLIT_LAST_HOURS": NO_SPLIT_LAST_HOURS,
            "MAX_SPLITS": MAX_SPLITS,
            "MIN_SPLITS": MIN_SPLITS,
        }
        for i, z in enumerate(s.zone_times_h, start=1):
            row[f"zone{i}_h"] = z
        for i, t in enumerate(s.split_times_h, start=1):
            row[f"split{i}_h"] = t
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values(["fuzzer", "benchmark"])
    summary_path = OUT_DIR / "sparsity_split_times_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Wrote plots to: {OUT_DIR.resolve()}")
    print(f"Wrote summary to: {summary_path.resolve()}")

    # Also print per-fuzzer mapping stats (helpful for debugging)
    if stats_by_fuzzer:
        print("\nPer-fuzzer mapping stats:")
        for f, st in sorted(stats_by_fuzzer.items()):
            print(f"  {f}: q_low={st['q_low']:.6g}, q_high={st['q_high']:.6g}, power_used={st['power_used']:.4g}, pivot_u={st['pivot_u']:.4g}")


if __name__ == "__main__":
    main()
