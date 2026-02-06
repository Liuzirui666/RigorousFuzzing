# /home/fuzz/split_union/split_strategy.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Stage:
    stage_idx: int         # 1..4
    start_hour: float
    end_hour: float

    @property
    def duration_seconds(self) -> int:
        return int(round((self.end_hour - self.start_hour) * 3600))

    @property
    def trials(self) -> int:
        # Route B: stage1=2, stage2=4, stage3=8, stage4=16
        return 2 ** self.stage_idx


@dataclass(frozen=True)
class FixedTimeSplitPlan:
    checkpoints_hours: List[float]  # e.g. [16, 17.5, 19, 20.5, 24]

    def stages(self) -> List[Stage]:
        cps = self.checkpoints_hours
        if len(cps) < 2:
            raise ValueError("Need >=2 checkpoints.")
        out: List[Stage] = []
        for i in range(len(cps) - 1):
            out.append(Stage(stage_idx=i + 1, start_hour=cps[i], end_hour=cps[i + 1]))
        return out


def default_4split_plan() -> FixedTimeSplitPlan:
    # split at 13.5h, 17h; run until 23h
    # Do not use. Only for debugging.
    return FixedTimeSplitPlan(checkpoints_hours=[13.5, 17, 23.0])


# ---------------------------------------------------------------------------
# Automatic split plan from sparsity_split_times_summary.csv
# ---------------------------------------------------------------------------

# Modes:
#   - "sparsity_csv": build plan from sparsity_split_times_summary.csv (zone1_h/zone2_h/zone3_h)
#   - "fixed_test":   use default_4split_plan() for validation
SPLIT_PLAN_MODE = "sparsity_csv"   # change to "fixed_test" for the legacy plan

# Expected to be produced by sparsity_unified_multi_v2.py and placed next to this file.
# If you keep the summary under a subdirectory (e.g., plots_test/), edit this path accordingly.
SPARSITY_SUMMARY_CSV = Path(__file__).resolve().parent / "plots" / "sparsity_split_times_summary.csv"

_ZONE_COLS = ("zone1_h", "zone2_h", "zone3_h")

# Cache to avoid re-reading CSV repeatedly
_SPARSE_CACHE: Optional[Dict[Tuple[str, str], Dict[str, str]]] = None


def _to_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    t = str(s).strip()
    if t == "" or t.lower() in ("none", "nan"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _load_summary(csv_path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"[split-plan] missing summary CSV: {csv_path}")

    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bm = (row.get("benchmark") or "").strip()
            fz = (row.get("fuzzer") or "").strip()
            if not bm or not fz:
                continue
            out[(fz, bm)] = {k: (v if v is not None else "") for k, v in row.items()}
    return out


def _get_row(fuzzer: str, benchmark: str) -> Dict[str, str]:
    global _SPARSE_CACHE
    if _SPARSE_CACHE is None:
        _SPARSE_CACHE = _load_summary(SPARSITY_SUMMARY_CSV)

    key = (fuzzer, benchmark)
    row = _SPARSE_CACHE.get(key)
    if row is None:
        avail = sorted({fz for (fz, bm) in _SPARSE_CACHE.keys() if bm == benchmark})
        raise KeyError(
            f"[split-plan] (fuzzer={fuzzer}, benchmark={benchmark}) not found in {SPARSITY_SUMMARY_CSV}. "
            f"Available fuzzers for this benchmark: {avail}"
        )
    return row


def _build_plan_for_pair(fuzzer: str, benchmark: str) -> FixedTimeSplitPlan:
    """
    Requested rule:
      - read zone1_h/zone2_h/zone3_h
      - if all empty -> definitely no split (single stage)
      - otherwise use the non-empty zone times as split checkpoints
    Always append t_end_h.
    """
    row = _get_row(fuzzer, benchmark)

    zones: List[float] = []
    for c in _ZONE_COLS:
        v = _to_float(row.get(c))
        if v is not None:
            zones.append(v)

    t_end = _to_float(row.get("t_end_h"))
    if t_end is None:
        raise ValueError(f"[split-plan] missing/invalid t_end_h for (fuzzer={fuzzer}, benchmark={benchmark})")

    if not zones:
        start_h = _to_float(row.get("start_h"))
        if start_h is None:
            raise ValueError(
                f"[split-plan] no zones and missing/invalid start_h for (fuzzer={fuzzer}, benchmark={benchmark})"
            )
        if t_end <= start_h:
            raise ValueError(
                f"[split-plan] invalid range: start_h={start_h} t_end_h={t_end} for (fuzzer={fuzzer}, benchmark={benchmark})"
            )
        return FixedTimeSplitPlan(checkpoints_hours=[start_h, t_end])

    checkpoints = list(zones)
    if abs(checkpoints[-1] - t_end) > 1e-9:
        checkpoints.append(t_end)

    for i in range(1, len(checkpoints)):
        if checkpoints[i] <= checkpoints[i - 1]:
            raise ValueError(
                f"[split-plan] non-increasing checkpoints for (fuzzer={fuzzer}, benchmark={benchmark}): {checkpoints}"
            )

    return FixedTimeSplitPlan(checkpoints_hours=checkpoints)


def _same_checkpoints(a: List[float], b: List[float], eps: float = 1e-9) -> bool:
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= eps for x, y in zip(a, b))


def get_split_plan(fuzzer: str, benchmarks: Sequence[str]) -> FixedTimeSplitPlan:
    """
    Select the split plan used by run_split_union.py.

    Note: a single run_split_union invocation runs all benchmarks together per stage,
    so all benchmarks must share identical checkpoints for that invocation.
    """
    if SPLIT_PLAN_MODE == "fixed_test":
        return default_4split_plan()
    if SPLIT_PLAN_MODE != "sparsity_csv":
        raise ValueError(f"[split-plan] unknown SPLIT_PLAN_MODE={SPLIT_PLAN_MODE}")

    if not benchmarks:
        raise ValueError("[split-plan] empty benchmarks list")

    per_bm: List[Tuple[str, FixedTimeSplitPlan]] = []
    for bm in benchmarks:
        per_bm.append((bm, _build_plan_for_pair(fuzzer, bm)))

    ref_bm, ref_plan = per_bm[0]
    mismatched = []
    for bm, plan in per_bm[1:]:
        if not _same_checkpoints(plan.checkpoints_hours, ref_plan.checkpoints_hours):
            mismatched.append((bm, plan.checkpoints_hours))

    if mismatched:
        detail = "; ".join([f"{ref_bm}={ref_plan.checkpoints_hours}"] + [f"{bm}={cps}" for bm, cps in mismatched])
        raise ValueError(
            "[split-plan] mismatched checkpoints across benchmarks in one run_split_union invocation. "
            f"Details: {detail}. "
            "Run separate invocations per benchmark (or ensure identical split times)."
        )

    return ref_plan
