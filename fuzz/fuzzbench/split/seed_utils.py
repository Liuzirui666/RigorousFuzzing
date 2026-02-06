# /home/fuzz/split_union/seed_utils.py
from __future__ import annotations

import hashlib
import os
import re
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


_ARCH_RE = re.compile(r"^corpus-archive-(\d{4})\.tar\.gz$")


def sha256_file(p: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_extract_tar_gz(tar_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, mode="r:gz") as tf:
        for m in tf.getmembers():
            # block path traversal
            target = (dest_dir / m.name).resolve()
            if not str(target).startswith(str(dest_dir.resolve()) + os.sep) and target != dest_dir.resolve():
                raise RuntimeError(f"Unsafe path in tar: {m.name}")
        tf.extractall(dest_dir)


def flatten_if_single_topdir(dest_dir: Path) -> None:
    entries = list(dest_dir.iterdir())
    if not entries:
        return
    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]
    if len(dirs) == 1 and len(files) == 0:
        top = dirs[0]
        for child in top.iterdir():
            shutil.move(str(child), str(dest_dir / child.name))
        top.rmdir()


def list_archives_desc(corpus_dir: Path) -> List[Tuple[int, Path]]:
    found: List[Tuple[int, Path]] = []
    for p in corpus_dir.iterdir():
        if not p.is_file():
            continue
        m = _ARCH_RE.match(p.name)
        if not m:
            continue
        found.append((int(m.group(1)), p))
    found.sort(key=lambda x: x[0], reverse=True)
    return found


def seed_files_from_extracted_root(extracted_root: Path, fuzzer: str) -> List[Path]:
    """
    Normalize "where seeds live" by fuzzer:
      - afl/aflplusplus: queue/
      - libfuzzer: corpus/
      - honggfuzz: corpus/
    Then return *files only* under that directory.
    """
    f = fuzzer.lower()

    # pick candidate directory
    if f.startswith("afl"):
        cand = extracted_root / "queue"
        if not cand.exists():
            cand = extracted_root  # fallback
        # AFL queue has many real inputs named like id:000000,...
        files = [p for p in cand.rglob("*") if p.is_file()]
        files = [p for p in files if p.name.startswith("id:")]  # avoid stats/plot/etc
        return files

    if f in {"libfuzzer", "honggfuzz"}:
        cand = extracted_root / "corpus"
        if not cand.exists():
            cand = extracted_root
        files = [p for p in cand.rglob("*") if p.is_file()]
        return files

    # unknown fuzzer: best effort (treat all files as seeds)
    return [p for p in extracted_root.rglob("*") if p.is_file()]


@dataclass
class BenchSeedState:
    base_hashes: Set[str]
    selected_new_hashes: Set[str]  
    base_count: int

    @property
    def cap_total(self) -> int:
        return 2 * self.base_count

    @property
    def cap_new(self) -> int:
        # total <= 2*base => new <= base
        return self.base_count


class SeedPool:
    """
    For each benchmark:
      - Keep ALL basic seeds (base_hashes).
      - Maintain selected_new_hashes (<= base_count).
      - Persist every unique seed's content in seed_store/<benchmark>/<sha256>.
      - Next-stage seed dir = base + selected_new (flat files).
    """

    def __init__(self, seed_store_root: Path, state_root: Path):
        self.seed_store_root = seed_store_root
        self.state_root = state_root
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.seed_store_root.mkdir(parents=True, exist_ok=True)
        self.bench_states: Dict[str, BenchSeedState] = {}

    def _state_path(self, benchmark: str) -> Path:
        return self.state_root / f"{benchmark}.txt"

    def load_or_init_benchmark(self, benchmark: str, basic_bench_dir: Path) -> None:
        """
        Initialize base_hashes from basic-corpus (flat seeds).
        Persist base seeds into seed_store.
        Load selected_new_hashes if state exists; else empty.
        """
        basic_files = [p for p in basic_bench_dir.iterdir() if p.is_file()]
        if not basic_files:
            raise FileNotFoundError(f"basic-corpus for benchmark is empty: {basic_bench_dir}")

        base_hashes: Set[str] = set()
        store_dir = self.seed_store_root / benchmark
        store_dir.mkdir(parents=True, exist_ok=True)

        for f in basic_files:
            h = sha256_file(f)
            base_hashes.add(h)
            dst = store_dir / h
            if not dst.exists():
                shutil.copy2(f, dst)

        base_count = len(base_hashes)

        selected_new: Set[str] = set()
        sp = self._state_path(benchmark)
        if sp.exists():
            for line in sp.read_text().splitlines():
                s = line.strip()
                if s:
                    selected_new.add(s)
            # enforce cap if someone edited state
            # selected_new = set(sorted(selected_new)[:base_count])

        self.bench_states[benchmark] = BenchSeedState(
            base_hashes=base_hashes,
            selected_new_hashes=selected_new,
            base_count=base_count,
        )

    def save_state(self, benchmark: str) -> None:
        st = self.bench_states[benchmark]
        sp = self._state_path(benchmark)
        sp.write_text("\n".join(sorted(st.selected_new_hashes)) + ("\n" if st.selected_new_hashes else ""))

    def ingest_trial_archive_seeds(
        self,
        benchmark: str,
        fuzzer: str,
        trial_corpus_dir: Path,
        tmp_root: Path,
        max_backtrack: int = 50,
    ) -> Set[str]:
        """
        Find latest non-empty corpus archive for this trial (backtrack if empty),
        extract seeds, store them in seed_store, and return discovered hashes.
        """
        archives = list_archives_desc(trial_corpus_dir)
        if not archives:
            raise FileNotFoundError(f"No corpus-archive-XXXX.tar.gz in {trial_corpus_dir}")

        store_dir = self.seed_store_root / benchmark
        store_dir.mkdir(parents=True, exist_ok=True)

        tried = 0
        for idx, ap in archives:
            tried += 1
            if tried > max_backtrack:
                break

            work = tmp_root / f"{benchmark}-try-{idx}"
            if work.exists():
                shutil.rmtree(work)
            work.mkdir(parents=True, exist_ok=True)

            safe_extract_tar_gz(ap, work)
            flatten_if_single_topdir(work)

            seed_files = seed_files_from_extracted_root(work, fuzzer)
            if not seed_files:
                # empty -> backtrack
                shutil.rmtree(work, ignore_errors=True)
                continue

            discovered: Set[str] = set()
            for sf in seed_files:
                h = sha256_file(sf)
                discovered.add(h)
                dst = store_dir / h
                if not dst.exists():
                    shutil.copy2(sf, dst)

            shutil.rmtree(work, ignore_errors=True)
            return discovered

        raise RuntimeError(
            f"All recent archives seem empty for trial_corpus_dir={trial_corpus_dir} "
            f"(backtracked up to {max_backtrack})."
        )

    def update_union_and_trim(self, benchmark: str, union_hashes_from_all_trials: Set[str]) -> None:
        """
        Apply: keep base always; new = union - base; selected_new = deterministic(hash-sort) first N.
        """
        st = self.bench_states[benchmark]
        candidate_new = union_hashes_from_all_trials - st.base_hashes

        # union semantics: consider ALL discovered new seeds so far
        merged_new = st.selected_new_hashes | candidate_new

        trimmed = set(sorted(merged_new)[:])
        st.selected_new_hashes = trimmed

    def materialize_stage_seed_root(self, stage_seed_root: Path, benchmarks: List[str]) -> None:
        """
        Create stage_seed_root/<benchmark>/ with flat seed files:
          base_hashes + selected_new_hashes
        Filenames are their sha256 (deterministic, dedup-friendly).
        """
        if stage_seed_root.exists():
            shutil.rmtree(stage_seed_root)
        stage_seed_root.mkdir(parents=True, exist_ok=True)

        for b in benchmarks:
            st = self.bench_states[b]
            out_b = stage_seed_root / b
            out_b.mkdir(parents=True, exist_ok=True)

            store_b = self.seed_store_root / b

            # copy base
            for h in st.base_hashes:
                src = store_b / h
                dst = out_b / h
                if not dst.exists():
                    shutil.copy2(src, dst)

            # copy selected new
            for h in st.selected_new_hashes:
                src = store_b / h
                dst = out_b / h
                if not dst.exists():
                    shutil.copy2(src, dst)
