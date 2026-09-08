from __future__ import annotations

from pathlib import Path
from typing import Iterable


CODING_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODING_DIR.parent

DATA_DIR = CODING_DIR / "data"
ARTIFACTS_DIR = CODING_DIR / "artifacts"
DOCS_DIR = CODING_DIR / "docs"
EXPERIMENTS_DIR = CODING_DIR / "experiments"
SNAPSHOTS_DIR = CODING_DIR / "snapshots"
TESTS_DIR = CODING_DIR / "tests"

QUERY_DATA_DIR = DATA_DIR / "query"
QUERY_STRIPPED_DIR = QUERY_DATA_DIR / "stripped"
QUERY_DEBUG_DIR = QUERY_DATA_DIR / "debug"
QUERY_PARTIAL_INLINE_SLICE_DIR = QUERY_DATA_DIR / "partial_inline_slice"
POOL_DIR = DATA_DIR / "pool"
INLINE_DATA_DIR = DATA_DIR / "inline"
INLINE_DEBUG_DIR = INLINE_DATA_DIR / "debug"
INLINE_STRIP_DIR = INLINE_DATA_DIR / "strip"
INLINE_NOINLINE_DIR = INLINE_DATA_DIR / "noinline"

BENCHMARK_ARTIFACTS_DIR = ARTIFACTS_DIR / "benchmark"
ANALYSIS_ARTIFACTS_DIR = ARTIFACTS_DIR / "analysis"
TMP_ARTIFACTS_DIR = ARTIFACTS_DIR / "tmp"
LOG_ARTIFACTS_DIR = ARTIFACTS_DIR / "logs"

METHOD_DOCS_DIR = DOCS_DIR / "method"
WORKFLOW_DOCS_DIR = DOCS_DIR / "workflow"
REPORT_DOCS_DIR = DOCS_DIR / "reports"

MODEL_DIR = EXPERIMENTS_DIR / "model"
MODEL_EVAL_DIR = EXPERIMENTS_DIR / "model_eval"

TESTDATA_DIR = TESTS_DIR / "testdata"
TEST_TOOLS_DIR = TESTS_DIR / "tools"

LEGACY_QUERY_STRIPPED_DIR = CODING_DIR / "query_binary_stripped"
LEGACY_QUERY_DEBUG_DIR = CODING_DIR / "query_binary_debug"
LEGACY_QUERY_PARTIAL_INLINE_SLICE_DIR = CODING_DIR / "partial_query_inline_slice"
LEGACY_POOL_DIR = CODING_DIR / "pool_binary"
LEGACY_LOG_DIR = CODING_DIR / "log"
LEGACY_MODEL_DIR = CODING_DIR / "model"


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _resolve_read_path(new_path: Path, legacy_path: Path) -> Path:
    return _first_existing((new_path, legacy_path)) or new_path


def _resolve_read_dir(new_dir: Path, legacy_dir: Path) -> Path:
    return _resolve_read_path(new_dir, legacy_dir)


def _prepare_write_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def query_stripped_dir(*, read: bool = True) -> Path:
    return _resolve_read_dir(QUERY_STRIPPED_DIR, LEGACY_QUERY_STRIPPED_DIR) if read else QUERY_STRIPPED_DIR


def query_debug_dir(*, read: bool = True) -> Path:
    return _resolve_read_dir(QUERY_DEBUG_DIR, LEGACY_QUERY_DEBUG_DIR) if read else QUERY_DEBUG_DIR


def query_partial_inline_slice_dir(*, read: bool = True) -> Path:
    if read:
        return _resolve_read_dir(QUERY_PARTIAL_INLINE_SLICE_DIR, LEGACY_QUERY_PARTIAL_INLINE_SLICE_DIR)
    return QUERY_PARTIAL_INLINE_SLICE_DIR


def pool_dir(*, read: bool = True) -> Path:
    return _resolve_read_dir(POOL_DIR, LEGACY_POOL_DIR) if read else POOL_DIR


def logs_dir(*, read: bool = True) -> Path:
    return _resolve_read_dir(LOG_ARTIFACTS_DIR, LEGACY_LOG_DIR) if read else LOG_ARTIFACTS_DIR


def feature_path(base_dir: Path, binary_name: str) -> Path:
    return Path(base_dir) / f"{binary_name}_bb_slice_feature.json"


def candidates_path(base_dir: Path, binary_name: str) -> Path:
    return Path(base_dir) / f"{binary_name}_inline_slice_candidates.json"


def embedding_path(base_dir: Path, binary_name: str) -> Path:
    return Path(base_dir) / f"{binary_name}_bb_slice_embeddings.pkl"


def inline_info_path(base_dir: Path, binary_name: str) -> Path:
    return Path(base_dir) / f"{binary_name}_inline_functions.json"


def benchmark_cases_path(name: str, *, read: bool = True) -> Path:
    new_path = BENCHMARK_ARTIFACTS_DIR / f"slice_benchmark_cases_{name}.json"
    legacy_path = CODING_DIR / f"slice_benchmark_cases_{name}.json"
    return _resolve_read_path(new_path, legacy_path) if read else _prepare_write_path(new_path)


def benchmark_summary_path(name: str, *, read: bool = True) -> Path:
    new_path = BENCHMARK_ARTIFACTS_DIR / f"slice_benchmark_summary_{name}.json"
    legacy_path = CODING_DIR / f"slice_benchmark_summary_{name}.json"
    return _resolve_read_path(new_path, legacy_path) if read else _prepare_write_path(new_path)


def tmp_benchmark_path(filename: str, *, read: bool = True) -> Path:
    new_path = BENCHMARK_ARTIFACTS_DIR / filename
    legacy_path = CODING_DIR / filename
    return _resolve_read_path(new_path, legacy_path) if read else _prepare_write_path(new_path)


def analysis_artifact_path(filename: str, *, read: bool = True) -> Path:
    new_path = ANALYSIS_ARTIFACTS_DIR / filename
    legacy_path = CODING_DIR / filename
    return _resolve_read_path(new_path, legacy_path) if read else _prepare_write_path(new_path)


def tmp_artifact_path(filename: str, *, read: bool = True) -> Path:
    new_path = TMP_ARTIFACTS_DIR / filename
    legacy_path = CODING_DIR / filename
    return _resolve_read_path(new_path, legacy_path) if read else _prepare_write_path(new_path)


def testdata_path(filename: str) -> Path:
    return TESTDATA_DIR / filename


def ensure_layout_dirs() -> None:
    for directory in (
        DATA_DIR,
        QUERY_DATA_DIR,
        QUERY_STRIPPED_DIR,
        QUERY_DEBUG_DIR,
        QUERY_PARTIAL_INLINE_SLICE_DIR,
        POOL_DIR,
        INLINE_DATA_DIR,
        INLINE_DEBUG_DIR,
        INLINE_STRIP_DIR,
        INLINE_NOINLINE_DIR,
        ARTIFACTS_DIR,
        BENCHMARK_ARTIFACTS_DIR,
        ANALYSIS_ARTIFACTS_DIR,
        TMP_ARTIFACTS_DIR,
        LOG_ARTIFACTS_DIR,
        DOCS_DIR,
        METHOD_DOCS_DIR,
        WORKFLOW_DOCS_DIR,
        REPORT_DOCS_DIR,
        EXPERIMENTS_DIR,
        MODEL_DIR,
        MODEL_EVAL_DIR,
        SNAPSHOTS_DIR,
        TESTS_DIR,
        TESTDATA_DIR,
        TEST_TOOLS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
