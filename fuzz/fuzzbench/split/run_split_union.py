#!/usr/bin/env python3
# /home/fuzz/split_union/run_split_union.py
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Optional

import yaml

from split_strategy_multi import Stage, get_split_plan
# from split_strategy_multi import FixedTimeSplitPlan, Stage, default_4split_plan, plan_for_fuzzer_and_benchmarks
from seed_utils import SeedPool


@dataclass(frozen=True)
class RunSplitUnionConfig:
    split_id: str
    fuzzer: str
    benchmarks: List[str]

    root: Path = Path("/home/fuzz")
    custom_corpus_root: Optional[Path] = None
    corpus_time: str = "h9_0_1"

    experiment_filestore: Path = Path("/home/fuzz/experiment-data")
    report_filestore: Path = Path("/home/fuzz/report-data")

    tmp_config_root: Optional[Path] = None  # default: root/tmp-configs

    concurrent_builds: int = 5
    runners_cpus: int = 20
    measurers_cpus: int = 8
    cpu_offset: int = 0
    allow_uncommitted: bool = True

    docker_registry: str = "gcr.io/fuzzbench"

    # Resume knobs (CLI-controlled; default preserves original behavior)
    start_from_stage: int = 1
    stop_after_stage: Optional[int] = None

    @property
    def fuzzbench_dir(self) -> Path:
        return self.root / "fuzzbench"

    @property
    def run_experiment_py(self) -> Path:
        return self.fuzzbench_dir / "experiment" / "run_experiment.py"

    @property
    def _custom_root(self) -> Path:
        return self.custom_corpus_root or (self.root / "custom_corpus")

    @property
    def basic_corpus_root(self) -> Path:
        return self._custom_root / "basic-corpus" / self.fuzzer / self.corpus_time

    @property
    def split_corpus_root(self) -> Path:
        return self._custom_root / self.split_id / self.fuzzer

    @property
    def stage_seeds_dir(self) -> Path:
        return self.split_corpus_root / "stage-seeds"

    @property
    def seed_store_dir(self) -> Path:
        return self.split_corpus_root / "seed-store"

    @property
    def state_dir(self) -> Path:
        return self.split_corpus_root / "state"

    @property
    def tmp_extract_dir(self) -> Path:
        return self.split_corpus_root / "tmp-extract"

    @property
    def tmp_cfg_dir(self) -> Path:
        base = self.tmp_config_root or (self.root / "tmp-configs")
        return base / self.split_id

    def run_experiment_workdir(self, experiment_name: str) -> Path:
        # Critical for parallel runs: fuzzbench creates src.tar.gz and a CONFIG_DIR in CWD.
        # If multiple run_experiment processes share the same CWD, they race and you get:
        #   cp: cannot stat 'src.tar.gz'
        return self.tmp_extract_dir / "_runexp_work" / experiment_name


def _write_config_for_stage(cfg: RunSplitUnionConfig, stage: Stage) -> Path:
    cfg.tmp_cfg_dir.mkdir(parents=True, exist_ok=True)
    stage_cfg = {
        "trials": int(stage.trials),
        "max_total_time": int(stage.duration_seconds),
        "docker_registry": cfg.docker_registry,
        "experiment_filestore": str(cfg.experiment_filestore),
        "report_filestore": str(cfg.report_filestore),
        "local_experiment": True,
    }
    out = cfg.tmp_cfg_dir / f"{cfg.split_id}-stage{stage.stage_idx}-trials{stage.trials}.yaml"
    out.write_text(yaml.safe_dump(stage_cfg, sort_keys=False))
    return out


def _list_trial_corpus_dirs(cfg: RunSplitUnionConfig, experiment_name: str, benchmark: str) -> List[Path]:
    bench_root = (
        cfg.experiment_filestore
        / experiment_name
        / "experiment-folders"
        / f"{benchmark}-{cfg.fuzzer}"
    )
    if not bench_root.exists():
        raise FileNotFoundError(f"Benchmark folder not found: {bench_root}")

    trials = []
    for p in bench_root.glob("trial-*"):
        if not p.is_dir():
            continue
        try:
            tid = int(p.name.split("-", 1)[1])
        except Exception:
            continue
        cdir = p / "corpus"
        if cdir.exists():
            trials.append((tid, cdir))

    trials.sort(key=lambda x: x[0])
    return [cdir for _, cdir in trials]


def _run_one_stage(cfg: RunSplitUnionConfig, stage: Stage, experiment_name: str, seed_root: Path) -> None:
    cfg_path = _write_config_for_stage(cfg, stage)

    for b in cfg.benchmarks:
        p = seed_root / b
        if not p.exists():
            raise FileNotFoundError(f"Missing benchmark dir under seed_root: {p}")
        if not any(p.iterdir()):
            raise RuntimeError(f"Seed dir empty: {p}")

    workdir = cfg.run_experiment_workdir(experiment_name)
    workdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3",
        str(cfg.run_experiment_py),  # absolute path, so cwd can be isolated
        "--experiment-config",
        str(cfg_path),
        "--experiment-name",
        experiment_name,
        "--fuzzers",
        cfg.fuzzer,
        "--benchmarks",
        *cfg.benchmarks,
        "--concurrent-builds",
        str(cfg.concurrent_builds),
        "--runners-cpus",
        str(cfg.runners_cpus),
        "--measurers-cpus",
        str(cfg.measurers_cpus),
        "--cpu-offset",
        str(cfg.cpu_offset),
        "--custom-seed-corpus-dir",
        str(seed_root),
    ]
    if cfg.allow_uncommitted:
        cmd.append("--allow-uncommitted-changes")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(cfg.fuzzbench_dir)

    print(f"\n=== [{cfg.split_id}] STAGE {stage.stage_idx} RUN ===")
    print(f"experiment_name={experiment_name}")
    print(f"workdir={workdir}  (isolates src.tar.gz / config dir to avoid parallel collisions)")
    print(f"seed_root={seed_root}")
    print(f"cpu_offset={cfg.cpu_offset}  runners={cfg.runners_cpus}  measurers={cfg.measurers_cpus}")
    print(f"experiment_filestore={cfg.experiment_filestore}")
    print(f"report_filestore={cfg.report_filestore}")

    subprocess.run(cmd, cwd=str(workdir), check=True, env=env)


def _harvest_for_next_stage(
    cfg: RunSplitUnionConfig,
    prev_stage: Stage,
    prev_experiment_name: str,
    pool: SeedPool,
) -> None:
    """Harvest corpuses from |prev_experiment_name| and materialize seeds for next stage."""
    print(f"\n--- [{cfg.split_id}] Harvesting corpuses for next stage from {prev_experiment_name} ---")
    cfg.tmp_extract_dir.mkdir(parents=True, exist_ok=True)

    for b in cfg.benchmarks:
        union_hashes: Set[str] = set()

        trial_cdirs = _list_trial_corpus_dirs(cfg, prev_experiment_name, b)
        if len(trial_cdirs) < prev_stage.trials:
            raise FileNotFoundError(
                f"[{cfg.split_id}] Not enough trials for {b}: need {prev_stage.trials}, found {len(trial_cdirs)} under "
                f"{cfg.experiment_filestore/prev_experiment_name/'experiment-folders'/f'{b}-{cfg.fuzzer}'}"
            )

        for cdir in trial_cdirs[-prev_stage.trials:]:
            discovered = pool.ingest_trial_archive_seeds(
                benchmark=b,
                fuzzer=cfg.fuzzer,
                trial_corpus_dir=cdir,
                tmp_root=cfg.tmp_extract_dir,
                max_backtrack=80,
            )
            union_hashes |= discovered

        pool.update_union_and_trim(b, union_hashes)
        pool.save_state(b)

    next_seed_root = cfg.stage_seeds_dir / f"stage-{prev_stage.stage_idx + 1}"
    pool.materialize_stage_seed_root(next_seed_root, cfg.benchmarks)
    print(f"[{cfg.split_id}] Next stage seeds materialized at: {next_seed_root}")


def run_split_union(cfg: RunSplitUnionConfig) -> None:
    cfg.split_corpus_root.mkdir(parents=True, exist_ok=True)
    cfg.stage_seeds_dir.mkdir(parents=True, exist_ok=True)
    cfg.seed_store_dir.mkdir(parents=True, exist_ok=True)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.tmp_extract_dir.mkdir(parents=True, exist_ok=True)

    pool = SeedPool(seed_store_root=cfg.seed_store_dir, state_root=cfg.state_dir)
    for b in cfg.benchmarks:
        basic_b_dir = cfg.basic_corpus_root / b
        if not basic_b_dir.exists():
            raise FileNotFoundError(
                f"[{cfg.split_id}] basic-corpus missing for benchmark: {basic_b_dir}\n"
                f"Expected layout: {cfg.basic_corpus_root}/<benchmark>/*"
            )
        pool.load_or_init_benchmark(b, basic_b_dir)

    # plan = plan_for_fuzzer_and_benchmarks(cfg.fuzzer, cfg.benchmarks)
    plan = get_split_plan(cfg.fuzzer, cfg.benchmarks)
    stages = plan.stages()

    if cfg.start_from_stage > len(stages):
        raise ValueError(
            f"start_from_stage={cfg.start_from_stage} but this split plan has only {len(stages)} stages"
        )
    if cfg.stop_after_stage is not None:
        if cfg.stop_after_stage < cfg.start_from_stage:
            raise ValueError(
                f"stop_after_stage={cfg.stop_after_stage} < start_from_stage={cfg.start_from_stage}"
            )
        if cfg.stop_after_stage > len(stages):
            raise ValueError(
                f"stop_after_stage={cfg.stop_after_stage} but this split plan has only {len(stages)} stages"
            )

    stage1_seed_root = cfg.stage_seeds_dir / "stage-1"
    pool.materialize_stage_seed_root(stage1_seed_root, cfg.benchmarks)

    # ===== Resume preparation (for continue mode) =====
    if cfg.start_from_stage < 1:
        raise ValueError(f"start_from_stage must be >= 1, got {cfg.start_from_stage}")

    if cfg.start_from_stage > 1:
        # Ensure stage-k seed roots exist for k=start_from_stage.
        # If missing, reconstruct by harvesting outputs from earlier stages.
        def _has_any_file(d: Path) -> bool:
            try:
                return any(p.is_file() for p in d.rglob("*"))
            except Exception:
                return False

        for prev_idx in range(1, min(cfg.start_from_stage, len(stages) + 1)):
            need_seed_root = cfg.stage_seeds_dir / f"stage-{prev_idx + 1}"
            if need_seed_root.exists() and _has_any_file(need_seed_root):
                continue

            prev_stage = stages[prev_idx - 1]
            prev_exp_name = f"{cfg.split_id}-split{prev_idx}"
            _harvest_for_next_stage(cfg, prev_stage, prev_exp_name, pool)

    for stage in stages:
        if stage.stage_idx < cfg.start_from_stage:
            continue
        if cfg.stop_after_stage is not None and stage.stage_idx > cfg.stop_after_stage:
            break

        exp_name = f"{cfg.split_id}-split{stage.stage_idx}"

        seed_root = cfg.stage_seeds_dir / f"stage-{stage.stage_idx}"
        if not seed_root.exists():
            pool.materialize_stage_seed_root(seed_root, cfg.benchmarks)

        _run_one_stage(cfg, stage, exp_name, seed_root)

        if stage.stage_idx < len(stages):
            _harvest_for_next_stage(cfg, stage, exp_name, pool)

    print(f"\n[{cfg.split_id}] ALL DONE.")


def _parse_args(argv: List[str]) -> RunSplitUnionConfig:
    p = argparse.ArgumentParser(description="Run one split-union experiment (multi-stage).")

    p.add_argument("--split-id", required=True)
    p.add_argument("--fuzzer", required=True)
    p.add_argument("--benchmarks", nargs="+", required=True)

    p.add_argument("--root", default="/home/fuzz")
    p.add_argument("--custom-corpus-root", default=None)
    p.add_argument("--corpus-time", default="h16_5_1")

    p.add_argument("--experiment-filestore", required=True)
    p.add_argument("--report-filestore", required=True)
    p.add_argument("--tmp-config-root", default=None)

    p.add_argument("--concurrent-builds", type=int, default=5)
    p.add_argument("--runners-cpus", type=int, default=20)
    p.add_argument("--measurers-cpus", type=int, default=8)
    p.add_argument("--cpu-offset", type=int, default=0)
    p.add_argument("--allow-uncommitted", action="store_true")
    p.add_argument("--docker-registry", default="gcr.io/fuzzbench")

    # Continue/resume controls
    p.add_argument("--start-from-stage", type=int, default=1)
    p.add_argument("--stop-after-stage", type=int, default=None)

    a = p.parse_args(argv)

    return RunSplitUnionConfig(
        split_id=a.split_id,
        fuzzer=a.fuzzer,
        benchmarks=list(a.benchmarks),
        root=Path(a.root),
        custom_corpus_root=(Path(a.custom_corpus_root) if a.custom_corpus_root else None),
        corpus_time=a.corpus_time,
        experiment_filestore=Path(a.experiment_filestore),
        report_filestore=Path(a.report_filestore),
        tmp_config_root=(Path(a.tmp_config_root) if a.tmp_config_root else None),
        concurrent_builds=a.concurrent_builds,
        runners_cpus=a.runners_cpus,
        measurers_cpus=a.measurers_cpus,
        cpu_offset=a.cpu_offset,
        allow_uncommitted=bool(a.allow_uncommitted),
        docker_registry=a.docker_registry,
        start_from_stage=int(a.start_from_stage),
        stop_after_stage=(int(a.stop_after_stage) if a.stop_after_stage is not None else None),
    )


def main(argv: Optional[List[str]] = None) -> None:
    cfg = _parse_args(sys.argv[1:] if argv is None else argv)
    run_split_union(cfg)


if __name__ == "__main__":
    main()
