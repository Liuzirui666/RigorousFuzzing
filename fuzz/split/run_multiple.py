#!/usr/bin/env python3
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class RunVariant:
    split_id: str
    corpus_time: str
    experiment_filestore: Path
    report_filestore: Path
    cpu_offset: int


def _this_dir() -> Path:
    return Path(__file__).resolve().parent


def _pump_output(prefix: str, pipe) -> None:
    """Read a child process' stdout and print with a stable prefix (no log files)."""
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            # Keep it minimal but readable
            sys.stdout.write(f"[{prefix}] {line}")
            sys.stdout.flush()
    except Exception:
        pass


def _run_one_variant(
    run_split_union_py: Path,
    root: Path,
    fuzzer: str,
    benchmarks: List[str],
    concurrent_builds: int,
    runners_cpus: int,
    measurers_cpus: int,
    allow_uncommitted: bool,
    docker_registry: str,
    tmp_config_root: Path,
    v: RunVariant,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(run_split_union_py),
        "--split-id", v.split_id,
        "--fuzzer", fuzzer,
        "--benchmarks", *benchmarks,
        "--root", str(root),
        "--corpus-time", v.corpus_time,
        "--experiment-filestore", str(v.experiment_filestore),
        "--report-filestore", str(v.report_filestore),
        "--tmp-config-root", str(tmp_config_root),
        "--concurrent-builds", str(concurrent_builds),
        "--runners-cpus", str(runners_cpus),
        "--measurers-cpus", str(measurers_cpus),
        "--cpu-offset", str(v.cpu_offset),
        "--docker-registry", docker_registry,
    ]
    if allow_uncommitted:
        cmd.append("--allow-uncommitted")

    env = os.environ.copy()
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    t = threading.Thread(target=_pump_output, args=(v.split_id, p.stdout), daemon=True)
    t.start()

    print(f"[main] spawned {v.split_id} (cpu_offset={v.cpu_offset}, corpus_time={v.corpus_time})", flush=True)
    return p


def main() -> int:
    # =========================
    # EDIT HERE (only place you change)
    # =========================
    ROOT = Path("/home/fuzz")

    MULTIPLE_ROOT = ROOT / "aflpp1"
    MULTIPLE_ROOT.mkdir(parents=True, exist_ok=True)

    FUZZER = "aflplusplus"
    BENCHMARKS = [
        "arrow_parquet-arrow-fuzz",
        # "libhevc_hevc_dec_fuzzer",
        #"stb_stbi_read_fuzzer",
        #"ffmpeg_ffmpeg_demuxer_fuzzer",
        # "matio_matio_fuzzer",
        # "openh264_decoder_fuzzer",
        # "grok_grk_decompress_fuzzer",
        #"php_php-fuzz-parser-2020-07-25",
        # "libhtp_fuzz_htp",
        #"poppler_pdf_fuzzer",
    ]

    CONCURRENT_BUILDS = 6
    RUNNERS_CPUS = 16
    MEASURERS_CPUS = 6
    ALLOW_UNCOMMITTED = True

    CPU_OFFSET_BASE = 0
    # CPU offsets: 0, 30, 60, ...
    CPU_OFFSET_STEP = 25

    BASE_NAME = "aflpp1"
    N = 2  # trial1..trialN

    # All generated tmp configs also live under multiple-data (no extra top-level dirs)
    TMP_CONFIG_ROOT = MULTIPLE_ROOT / "tmp-configs"
    TMP_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    variants: List[RunVariant] = []
    for i in range(1, N + 1):
        variants.append(
            RunVariant(
                split_id=f"{BASE_NAME}-trial{i}",
                corpus_time=f"h16_0_{i}",
                experiment_filestore=MULTIPLE_ROOT / f"experiment-data-{i}",
                report_filestore=MULTIPLE_ROOT / f"report-data-{i}",
                cpu_offset=(CPU_OFFSET_BASE + (i - 1) * CPU_OFFSET_STEP),
            )
        )

    # =========================
    # Do NOT edit below
    # =========================
    run_split_union_py = _this_dir() / "run_split_union.py"
    if not run_split_union_py.exists():
        print(f"[FATAL] Missing {run_split_union_py}. Put run_multiple.py next to run_split_union.py.", file=sys.stderr)
        return 2

    # create filestores
    for v in variants:
        v.experiment_filestore.mkdir(parents=True, exist_ok=True)
        v.report_filestore.mkdir(parents=True, exist_ok=True)

    procs: List[subprocess.Popen] = []

    def _terminate_all(sig_name: str) -> None:
        print(f"\n[main] received {sig_name}; terminating all runs...", flush=True)
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass

    def _handle_sigint(signum, frame):
        _terminate_all("SIGINT")

    def _handle_sigterm(signum, frame):
        _terminate_all("SIGTERM")

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        for v in variants:
            procs.append(
                _run_one_variant(
                    run_split_union_py=run_split_union_py,
                    root=ROOT,
                    fuzzer=FUZZER,
                    benchmarks=BENCHMARKS,
                    concurrent_builds=CONCURRENT_BUILDS,
                    runners_cpus=RUNNERS_CPUS,
                    measurers_cpus=MEASURERS_CPUS,
                    allow_uncommitted=ALLOW_UNCOMMITTED,
                    docker_registry="gcr.io/fuzzbench",
                    tmp_config_root=TMP_CONFIG_ROOT,
                    v=v,
                )
            )

        exit_code = 0
        alive = set(procs)

        while alive:
            finished = []
            for p in list(alive):
                rc = p.poll()
                if rc is None:
                    continue
                finished.append((p, rc))

            for p, rc in finished:
                alive.remove(p)
                if rc != 0 and exit_code == 0:
                    exit_code = rc
                    print(f"[main] A run failed (exit={rc}). Terminating others...", flush=True)
                    _terminate_all("failure")

            if alive:
                time.sleep(1)

        print(f"[main] all done. exit={exit_code}", flush=True)
        return exit_code

    except KeyboardInterrupt:
        _terminate_all("KeyboardInterrupt")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
