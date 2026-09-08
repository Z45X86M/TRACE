import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import faiss
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
for path in (str(ROOT_DIR), str(CURRENT_DIR)):
    if path not in sys.path:
        sys.path.append(path)


DEFAULT_EMBEDDING_SUFFIX = "_bb_slice_embeddings.pkl"
LEGACY_EMBEDDING_SUFFIX = "_inline_slice_embeddings.pkl"


def discover_embedding_pickles(
    input_dirs: Sequence[str],
    embedding_suffix: str = DEFAULT_EMBEDDING_SUFFIX,
) -> List[str]:
    paths: List[str] = []
    for input_dir in input_dirs:
        base = Path(input_dir)
        if not base.exists():
            continue
        for path in sorted(base.glob(f"*{embedding_suffix}")):
            paths.append(str(path))
    return paths


def iter_partial_rows(embedding_paths: Sequence[str]) -> Iterable[np.ndarray]:
    for path in embedding_paths:
        data = pickle.load(open(path, "rb"))
        for value in data.values():
            partial = np.asarray(value.get("partial", []), dtype=np.float32)
            if partial.size == 0:
                continue
            if partial.ndim == 1:
                partial = partial.reshape(1, -1)
            if partial.ndim != 2 or partial.shape[1] == 0:
                continue
            yield np.ascontiguousarray(partial, dtype=np.float32)


def iter_partial_rows_with_binary_name(
    embedding_paths: Sequence[str],
    embedding_suffix: str = DEFAULT_EMBEDDING_SUFFIX,
) -> Iterable[Tuple[str, np.ndarray]]:
    for path in embedding_paths:
        binary_name = Path(path).name.replace(embedding_suffix, "")
        data = pickle.load(open(path, "rb"))
        for value in data.values():
            partial = np.asarray(value.get("partial", []), dtype=np.float32)
            if partial.size == 0:
                continue
            if partial.ndim == 1:
                partial = partial.reshape(1, -1)
            if partial.ndim != 2 or partial.shape[1] == 0:
                continue
            yield binary_name, np.ascontiguousarray(partial, dtype=np.float32)


def iter_function_partials(
    embedding_paths: Sequence[str],
    embedding_suffix: str = DEFAULT_EMBEDDING_SUFFIX,
) -> Iterator[Tuple[str, np.ndarray]]:
    for path in embedding_paths:
        binary_name = Path(path).name.replace(embedding_suffix, "")
        data = pickle.load(open(path, "rb"))
        for func_addr, value in data.items():
            partial = np.asarray(value.get("partial", []), dtype=np.float32)
            if partial.size == 0:
                continue
            if partial.ndim == 1:
                partial = partial.reshape(1, -1)
            if partial.ndim != 2 or partial.shape[1] == 0:
                continue
            yield f"{binary_name}::{func_addr}", np.ascontiguousarray(partial, dtype=np.float32)


def load_partial_matrix(embedding_paths: Sequence[str]) -> np.ndarray:
    rows = list(iter_partial_rows(embedding_paths))
    if not rows:
        raise ValueError("no partial embeddings found")
    return np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.float32)


def load_partial_matrix_with_binary_names(
    embedding_paths: Sequence[str],
    embedding_suffix: str = DEFAULT_EMBEDDING_SUFFIX,
) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[np.ndarray] = []
    binary_names: List[str] = []
    for binary_name, partial in iter_partial_rows_with_binary_name(
        embedding_paths, embedding_suffix=embedding_suffix
    ):
        rows.append(partial)
        binary_names.extend([binary_name] * int(partial.shape[0]))
    if not rows:
        raise ValueError("no partial embeddings found")
    matrix = np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.float32)
    binary_name_array = np.asarray(binary_names, dtype=object)
    return matrix, binary_name_array


def choose_nlist(num_vectors: int) -> int:
    """Pick a sane IVF coarse-cluster count for `num_vectors` partial rows.

    Auto-scales continuously with corpus size: targets ~25 slice rows per
    bucket (≈5 functions/bucket at ~5 slices/function), rounded up to the
    nearest power of two, floored at 256 and capped at 262144. The intent is
    that each `quantizer.search(topk=5)` recall stays in the
    low-hundreds-of-candidates range across pool sizes 10K → 100M+, instead
    of suffering the coarse-tier jumps the previous piecewise table had at
    the 200K / 1M / 5M boundaries.

    Empirically (BB+MaxSim 2026-05): the 100K-function pool (≈500K slices)
    now resolves to nlist=32768 instead of 16384, halving avg bucket
    occupancy and tightening Stage-A candidate quality, which is the
    dominant source of @10 decay between 10K and 100K pools.

    Power users with very different bucket-cost tradeoffs should pass
    `--nlist` explicitly.
    """
    if num_vectors <= 0:
        return 256
    target = num_vectors // 25
    if target < 256:
        return 256
    nlist = 1
    while nlist < target:
        nlist <<= 1
    return min(nlist, 262144)


def choose_train_size(num_vectors: int, nlist: int) -> int:
    target = max(16 * nlist, 30_000)
    return min(num_vectors, target)


def build_signature_projection(dim: int, signature_bits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((dim, signature_bits), dtype=np.float32)
    projection /= np.maximum(np.linalg.norm(projection, axis=0, keepdims=True), 1e-12)
    return np.ascontiguousarray(projection, dtype=np.float32)


def compute_signature_keys(
    X: np.ndarray,
    projection: np.ndarray,
) -> List[bytes]:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    normalized = X / np.maximum(norms, 1e-12)
    bits = np.matmul(normalized, projection) >= 0
    packed = np.packbits(bits.astype(np.uint8), axis=1)
    return [row.tobytes() for row in packed]


def select_frequency_balanced_indices(
    signature_keys: Sequence[bytes],
    train_size: int,
    seed: int,
    per_signature_cap: Optional[int] = None,
) -> np.ndarray:
    if train_size >= len(signature_keys):
        return np.arange(len(signature_keys), dtype=np.int64)
    groups: Dict[bytes, List[int]] = {}
    for idx, key in enumerate(signature_keys):
        groups.setdefault(key, []).append(idx)
    cap = per_signature_cap or max(1, int(np.ceil(train_size / max(1, len(groups)))))
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    leftovers: List[int] = []
    for key in sorted(groups):
        members = np.asarray(groups[key], dtype=np.int64)
        permuted = rng.permutation(members)
        keep = permuted[:cap]
        selected.extend(int(v) for v in keep)
        if permuted.size > cap:
            leftovers.extend(int(v) for v in permuted[cap:])
    if len(selected) > train_size:
        selected = [int(v) for v in rng.permutation(np.asarray(selected, dtype=np.int64))[:train_size]]
    elif len(selected) < train_size and leftovers:
        needed = min(train_size - len(selected), len(leftovers))
        refill = rng.permutation(np.asarray(leftovers, dtype=np.int64))[:needed]
        selected.extend(int(v) for v in refill)
    return np.asarray(selected[:train_size], dtype=np.int64)


def apply_per_binary_cap(
    binary_names: Sequence[str],
    per_binary_cap: int,
    seed: int,
) -> np.ndarray:
    binary_names = np.asarray(binary_names, dtype=object)
    if per_binary_cap <= 0 or binary_names.size == 0:
        return np.arange(binary_names.size, dtype=np.int64)
    groups: Dict[str, List[int]] = {}
    for idx, binary_name in enumerate(binary_names.tolist()):
        groups.setdefault(str(binary_name), []).append(idx)
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for binary_name in sorted(groups):
        members = np.asarray(groups[binary_name], dtype=np.int64)
        if members.size <= per_binary_cap:
            chosen = members
        else:
            chosen = rng.permutation(members)[:per_binary_cap]
        selected.extend(int(v) for v in chosen.tolist())
    return np.asarray(selected, dtype=np.int64)


def _stable_path_seed(path: str, seed: int) -> int:
    """Deterministic 64-bit per-path seed (stable across runs / processes)."""
    return (int(seed) * 0x9E3779B97F4A7C15 + zlib.adler32(path.encode("utf-8"))) & ((1 << 64) - 1)


def _load_partial_pickle_for_sampling(args):
    """Worker: load one pickle, balance within-binary, return capped rows + stats.

    Per-binary balancing semantics:
      1. compute 24-bit signatures of all partial rows in the binary (if
         signature_bits > 0);
      2. cap per signature to `per_signature_cap` (random subset within group);
      3. if still over `per_binary_cap`, random-trim to that cap.

    This is the streaming-friendly variant of the legacy global
    `sample_rows_frequency_balanced`: balancing is bounded per binary so the
    main process never has to materialize the full corpus matrix.
    """
    (
        path,
        embedding_suffix,
        per_binary_cap,
        signature_bits,
        per_signature_cap,
        seed,
    ) = args
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        return path, None, {
            "binary_name": Path(path).name.replace(embedding_suffix, ""),
            "raw_rows": 0,
            "kept_rows": 0,
            "norm_sum": 0.0,
            "norm_min": float("inf"),
            "norm_max": float("-inf"),
            "unique_signatures": np.zeros((0,), dtype=object),
            "error": str(exc)[:200],
        }
    rows: List[np.ndarray] = []
    for value in data.values():
        partial = np.asarray(value.get("partial", []), dtype=np.float32)
        if partial.size == 0:
            continue
        if partial.ndim == 1:
            partial = partial.reshape(1, -1)
        if partial.ndim != 2 or partial.shape[1] == 0:
            continue
        rows.append(np.ascontiguousarray(partial, dtype=np.float32))
    binary_name = Path(path).name.replace(embedding_suffix, "")
    if not rows:
        return path, None, {
            "binary_name": binary_name,
            "raw_rows": 0,
            "kept_rows": 0,
            "norm_sum": 0.0,
            "norm_min": float("inf"),
            "norm_max": float("-inf"),
            "unique_signatures": np.zeros((0,), dtype=object),
        }
    X = np.concatenate(rows, axis=0)
    raw_rows = int(X.shape[0])
    norms = np.linalg.norm(X, axis=1)
    norm_sum = float(norms.sum())
    norm_min_val = float(norms.min())
    norm_max_val = float(norms.max())

    unique_signatures = np.zeros((0,), dtype=object)
    if signature_bits > 0 and X.shape[0] > 0:
        projection = build_signature_projection(X.shape[1], signature_bits, seed)
        sigs = compute_signature_keys(X, projection)
        rng = np.random.default_rng(_stable_path_seed(path, seed))
        if per_signature_cap > 0:
            groups: Dict[bytes, List[int]] = {}
            for i, s in enumerate(sigs):
                groups.setdefault(s, []).append(i)
            keep: List[int] = []
            for key, members in groups.items():
                if len(members) <= per_signature_cap:
                    keep.extend(members)
                else:
                    idx = rng.permutation(np.asarray(members, dtype=np.int64))[:per_signature_cap]
                    keep.extend(int(v) for v in idx.tolist())
            unique_signatures = np.asarray(sorted(set(sigs)), dtype=object)
            if keep:
                X = np.ascontiguousarray(X[np.asarray(keep, dtype=np.int64)], dtype=np.float32)
            else:
                X = X[:0]
        else:
            unique_signatures = np.asarray(sorted(set(sigs)), dtype=object)

    if per_binary_cap > 0 and X.shape[0] > per_binary_cap:
        rng = np.random.default_rng(_stable_path_seed(path, seed) ^ 0xA5A5A5A5)
        idx = rng.permutation(X.shape[0])[:per_binary_cap]
        X = np.ascontiguousarray(X[idx], dtype=np.float32)

    return path, X, {
        "binary_name": binary_name,
        "raw_rows": raw_rows,
        "kept_rows": int(X.shape[0]),
        "norm_sum": norm_sum,
        "norm_min": norm_min_val,
        "norm_max": norm_max_val,
        "unique_signatures": unique_signatures,
    }


def stream_training_sample(
    embedding_paths: Sequence[str],
    train_size: int,
    seed: int,
    signature_bits: int,
    per_binary_cap: int,
    per_signature_cap: int,
    embedding_suffix: str,
    workers: int,
) -> Tuple[np.ndarray, Dict[str, object], Dict[str, float]]:
    """Stream-load partial embeddings and build a bounded training sample.

    Memory bound is `train_size * dim * 4` bytes regardless of corpus size.

    Returns
    -------
    sample : np.ndarray
        Training rows for FAISS-IVF coarse training, shape (≤train_size, dim).
    sample_stats : dict
        Bookkeeping for the meta.json `sample_stats` section.
    vector_stats : dict
        Streaming-aggregated norms/counts mirroring `compute_stats` output.
    """
    if workers <= 0:
        workers = max(1, min(8, mp.cpu_count() or 1))
    workers = min(workers, max(1, len(embedding_paths)))

    task_args = [
        (path, embedding_suffix, per_binary_cap, signature_bits, per_signature_cap, seed)
        for path in embedding_paths
    ]
    print(
        f"      streaming {len(task_args)} pickle files with workers={workers} "
        f"(per_binary_cap={per_binary_cap}, per_signature_cap={per_signature_cap}, "
        f"signature_bits={signature_bits})"
    )

    rng = np.random.default_rng(seed)
    sample_buf: Optional[np.ndarray] = None
    fill_count = 0
    stream_pos = 0
    raw_total = 0
    norm_sum = 0.0
    norm_min = float("inf")
    norm_max = float("-inf")
    binary_count = 0
    failed_files: List[Tuple[str, str]] = []
    global_signatures: Set[bytes] = set()
    t_start = time.time()

    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for path, X, file_stats in pool.imap_unordered(
            _load_partial_pickle_for_sampling, task_args, chunksize=1
        ):
            binary_count += 1
            if file_stats.get("error"):
                failed_files.append((path, file_stats["error"]))
            raw_total += int(file_stats.get("raw_rows", 0))
            if file_stats.get("raw_rows", 0):
                norm_sum += float(file_stats.get("norm_sum", 0.0))
                norm_min = min(norm_min, float(file_stats.get("norm_min", float("inf"))))
                norm_max = max(norm_max, float(file_stats.get("norm_max", float("-inf"))))
            uniq = file_stats.get("unique_signatures")
            if uniq is not None and len(uniq) > 0:
                global_signatures.update(uniq.tolist())
            if X is None or X.shape[0] == 0:
                if binary_count % 32 == 0 or binary_count == len(task_args):
                    elapsed = max(time.time() - t_start, 1e-6)
                    print(
                        f"      streamed {binary_count}/{len(task_args)} binaries "
                        f"kept={stream_pos} fill={fill_count}/{train_size} "
                        f"rate={binary_count / elapsed:.1f} bin/s elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                continue

            n = int(X.shape[0])
            dim = int(X.shape[1])
            if sample_buf is None:
                sample_buf = np.zeros((train_size, dim), dtype=np.float32)
            elif sample_buf.shape[1] != dim:
                raise ValueError(
                    f"embedding dim mismatch: expected {sample_buf.shape[1]}, got {dim} from {path}"
                )

            free_slots = train_size - fill_count
            if free_slots > 0:
                fill_n = min(free_slots, n)
                sample_buf[fill_count:fill_count + fill_n] = X[:fill_n]
                fill_count += fill_n
                stream_pos += fill_n
                X = X[fill_n:]
                n = X.shape[0]
            if n > 0:
                positions = stream_pos + 1 + np.arange(n, dtype=np.int64)
                probs = float(train_size) / positions.astype(np.float64)
                keep_mask = rng.random(n) < probs
                keep_idx = np.nonzero(keep_mask)[0]
                if keep_idx.size > 0:
                    slot_idx = rng.integers(train_size, size=keep_idx.size)
                    sample_buf[slot_idx] = X[keep_idx]
                stream_pos += n

            if binary_count % 32 == 0 or binary_count == len(task_args):
                elapsed = max(time.time() - t_start, 1e-6)
                print(
                    f"      streamed {binary_count}/{len(task_args)} binaries "
                    f"kept={stream_pos} fill={fill_count}/{train_size} "
                    f"rate={binary_count / elapsed:.1f} bin/s elapsed={elapsed:.1f}s",
                    flush=True,
                )

    if sample_buf is None or fill_count == 0:
        raise ValueError("no partial embeddings collected from input dirs")
    sample = np.ascontiguousarray(sample_buf[:fill_count], dtype=np.float32)

    elapsed = time.time() - t_start
    print(
        f"      stream done: {binary_count} binaries, raw_rows={raw_total}, "
        f"post_balance_rows={stream_pos}, sample_rows={sample.shape[0]}, "
        f"distinct_signatures≈{len(global_signatures)}, elapsed={elapsed:.1f}s",
        flush=True,
    )
    if failed_files:
        print(f"      WARN: {len(failed_files)} files failed to load", flush=True)
        for fp, err in failed_files[:8]:
            print(f"        {Path(fp).name}: {err}", flush=True)

    norm_min_out = float(norm_min) if norm_min != float("inf") else 0.0
    norm_max_out = float(norm_max) if norm_max != float("-inf") else 0.0
    vector_stats = {
        "num_vectors": int(raw_total),
        "dim": int(sample.shape[1]),
        "norm_mean": float(norm_sum / max(raw_total, 1)),
        "norm_min": norm_min_out,
        "norm_max": norm_max_out,
    }
    sample_stats: Dict[str, object] = {
        "sample_mode": "balanced",
        "signature_bits": int(signature_bits),
        "signature_count": int(len(global_signatures)),
        "train_size": int(train_size),
        "selected_rows": int(sample.shape[0]),
        "eligible_rows": int(stream_pos),
        "per_binary_cap": int(per_binary_cap),
        "per_signature_cap": int(per_signature_cap),
        "raw_rows_seen": int(raw_total),
        "stream_workers": int(workers),
        "stream_elapsed_sec": float(elapsed),
        "stream_failed_files": int(len(failed_files)),
        "stream_semantic": (
            "per-binary balanced (signature-cap + binary-cap applied inside each pickle); "
            "global reservoir (Algorithm R) merges the per-binary samples to train_size"
        ),
    }
    return sample, sample_stats, vector_stats


def _load_partial_pickle_raw(args):
    """Worker: load one pickle, return raw concatenated partials + per-func offsets.

    Used by `--add-all` add path and the bucket-distribution diagnostic. Returns
    the full embedding matrix for a single binary (no balancing applied).
    """
    path, embedding_suffix = args
    binary_name = Path(path).name.replace(embedding_suffix, "")
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        return path, binary_name, None, np.zeros((1,), dtype=np.int64), 0, str(exc)[:200]
    func_arrays: List[np.ndarray] = []
    for value in data.values():
        partial = np.asarray(value.get("partial", []), dtype=np.float32)
        if partial.size == 0:
            continue
        if partial.ndim == 1:
            partial = partial.reshape(1, -1)
        if partial.ndim != 2 or partial.shape[1] == 0:
            continue
        func_arrays.append(np.ascontiguousarray(partial, dtype=np.float32))
    if not func_arrays:
        return path, binary_name, None, np.zeros((1,), dtype=np.int64), 0, ""
    func_sizes = np.asarray([fa.shape[0] for fa in func_arrays], dtype=np.int64)
    func_offsets = np.concatenate([[0], np.cumsum(func_sizes)])
    X = np.concatenate(func_arrays, axis=0)
    return path, binary_name, X, func_offsets, len(func_arrays), ""


def stream_diagnose_bucket_function_distribution(
    embedding_paths: Sequence[str],
    transform: faiss.VectorTransform,
    quantizer: faiss.Index,
    bucket_topk: int,
    embedding_suffix: str,
    workers: int,
) -> Dict[str, float]:
    """Parallel-load variant of `diagnose_bucket_function_distribution`.

    Workers load pickles concurrently; the main process (which holds the
    quantizer, possibly on GPU) does the transform + search per file.
    """
    if workers <= 0:
        workers = max(1, min(8, mp.cpu_count() or 1))
    workers = min(workers, max(1, len(embedding_paths)))

    function_bucket_hits: List[np.ndarray] = []
    total_functions = 0
    total_slices = 0
    bucket_count = int(quantizer.ntotal)
    t_start = time.time()
    binary_count = 0

    task_args = [(p, embedding_suffix) for p in embedding_paths]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for path, _binary_name, X, func_offsets, n_funcs, err in pool.imap_unordered(
            _load_partial_pickle_raw, task_args, chunksize=1
        ):
            binary_count += 1
            if err:
                print(f"      diag: failed {Path(path).name}: {err}", flush=True)
                continue
            if X is None or X.shape[0] == 0:
                continue
            transformed = np.ascontiguousarray(transform.apply_py(X), dtype=np.float32)
            _scores, bucket_ids = quantizer.search(transformed, bucket_topk)
            for i in range(n_funcs):
                start = int(func_offsets[i])
                end = int(func_offsets[i + 1])
                if end <= start:
                    continue
                hits = bucket_ids[start:end]
                function_bucket_hits.append(np.unique(hits[hits >= 0]).astype(np.int32))
            total_functions += n_funcs
            total_slices += int(X.shape[0])
            if binary_count % 32 == 0 or binary_count == len(task_args):
                elapsed = max(time.time() - t_start, 1e-6)
                print(
                    f"      diag {binary_count}/{len(task_args)} functions={total_functions} "
                    f"slices={total_slices} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    bucket_function_counts = compute_bucket_function_counts(function_bucket_hits, bucket_count)
    summary = summarize_bucket_function_counts(bucket_function_counts)
    summary.update(
        {
            "total_functions": int(total_functions),
            "total_slices": int(total_slices),
            "bucket_topk": int(bucket_topk),
        }
    )
    return summary


def stream_add_all_to_index(
    embedding_paths: Sequence[str],
    index: faiss.Index,
    embedding_suffix: str,
    workers: int,
) -> int:
    """Stream all per-binary partials into the FAISS index (used by --add-all)."""
    if workers <= 0:
        workers = max(1, min(8, mp.cpu_count() or 1))
    workers = min(workers, max(1, len(embedding_paths)))
    task_args = [(p, embedding_suffix) for p in embedding_paths]
    t_start = time.time()
    binary_count = 0
    total_added = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for path, _binary_name, X, _func_offsets, _n_funcs, err in pool.imap_unordered(
            _load_partial_pickle_raw, task_args, chunksize=1
        ):
            binary_count += 1
            if err:
                print(f"      add-all: failed {Path(path).name}: {err}", flush=True)
                continue
            if X is None or X.shape[0] == 0:
                continue
            index.add(np.ascontiguousarray(X, dtype=np.float32))
            total_added += int(X.shape[0])
            if binary_count % 32 == 0 or binary_count == len(task_args):
                elapsed = max(time.time() - t_start, 1e-6)
                print(
                    f"      add-all {binary_count}/{len(task_args)} added={total_added} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    return total_added


def sample_rows_frequency_balanced(
    X: np.ndarray,
    train_size: int,
    seed: int,
    signature_bits: int,
    binary_names: Optional[Sequence[str]] = None,
    per_binary_cap: int = 0,
    per_signature_cap: int = 0,
    return_indices: bool = False,
) -> Tuple[np.ndarray, Dict[str, int]]:
    eligible_indices = np.arange(X.shape[0], dtype=np.int64)
    if binary_names is not None and per_binary_cap > 0:
        eligible_indices = apply_per_binary_cap(binary_names, per_binary_cap, seed)
        X = np.ascontiguousarray(X[eligible_indices], dtype=np.float32)
    effective_train_size = min(train_size, X.shape[0])
    if effective_train_size >= X.shape[0]:
        sampled = np.ascontiguousarray(X, dtype=np.float32)
        stats = {
            "sample_mode": "balanced",
            "signature_bits": int(signature_bits),
            "signature_count": int(X.shape[0]),
            "train_size": int(effective_train_size),
            "selected_rows": int(sampled.shape[0]),
            "eligible_rows": int(X.shape[0]),
            "per_binary_cap": int(per_binary_cap),
            "per_signature_cap": int(per_signature_cap),
        }
        if return_indices:
            stats["selected_global_indices"] = [int(v) for v in eligible_indices.tolist()]
        return sampled, stats
    projection = build_signature_projection(X.shape[1], signature_bits, seed)
    signature_keys = compute_signature_keys(X, projection)
    indices = select_frequency_balanced_indices(
        signature_keys,
        effective_train_size,
        seed,
        per_signature_cap=per_signature_cap or None,
    )
    signature_count = len(set(signature_keys))
    selected_global_indices = eligible_indices[indices]
    stats = {
        "sample_mode": "balanced",
        "signature_bits": int(signature_bits),
        "signature_count": int(signature_count),
        "train_size": int(effective_train_size),
        "selected_rows": int(indices.size),
        "eligible_rows": int(X.shape[0]),
        "per_binary_cap": int(per_binary_cap),
        "per_signature_cap": int(per_signature_cap),
    }
    sampled = np.ascontiguousarray(X[indices], dtype=np.float32)
    if return_indices:
        stats["selected_global_indices"] = [int(v) for v in selected_global_indices.tolist()]
    return sampled, stats


def sample_rows(X: np.ndarray, train_size: int, seed: int) -> np.ndarray:
    if train_size >= X.shape[0]:
        return np.ascontiguousarray(X, dtype=np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=train_size, replace=False)
    return np.ascontiguousarray(X[idx], dtype=np.float32)


def build_identity_transform(d: int) -> faiss.LinearTransform:
    transform = faiss.LinearTransform(d, d)
    faiss.copy_array_to_vector(np.eye(d, dtype=np.float32).ravel(), transform.A)
    faiss.copy_array_to_vector(np.zeros(d, dtype=np.float32), transform.b)
    transform.is_trained = True
    return transform


def train_transform(X_train: np.ndarray, input_dim: int, pca_dim: int) -> Tuple[faiss.VectorTransform, np.ndarray, int]:
    if pca_dim <= 0 or pca_dim >= input_dim:
        transform = build_identity_transform(input_dim)
        return transform, np.ascontiguousarray(X_train, dtype=np.float32), input_dim
    transform = faiss.PCAMatrix(input_dim, pca_dim, eigen_power=0.0)
    transform.train(X_train)
    X_t = np.ascontiguousarray(transform.apply_py(X_train), dtype=np.float32)
    return transform, X_t, pca_dim


def metric_from_name(metric_name: str) -> int:
    if metric_name == "ip":
        return faiss.METRIC_INNER_PRODUCT
    if metric_name == "l2":
        return faiss.METRIC_L2
    raise ValueError(f"unsupported metric: {metric_name}")


def make_flat_index(dim: int, metric_name: str, use_gpu: bool, gpu_id: int):
    metric = metric_from_name(metric_name)
    if metric == faiss.METRIC_INNER_PRODUCT:
        cpu_index = faiss.IndexFlatIP(dim)
        if use_gpu:
            res = faiss.StandardGpuResources()
            return cpu_index, faiss.index_cpu_to_gpu(res, gpu_id, cpu_index), res
        return cpu_index, cpu_index, None
    cpu_index = faiss.IndexFlatL2(dim)
    if use_gpu:
        res = faiss.StandardGpuResources()
        return cpu_index, faiss.index_cpu_to_gpu(res, gpu_id, cpu_index), res
    return cpu_index, cpu_index, None


def train_ivf_centroids(
    X_train: np.ndarray,
    dim: int,
    nlist: int,
    metric_name: str,
    niter: int,
    max_points_per_centroid: int,
    use_gpu: bool,
    gpu_id: int,
) -> np.ndarray:
    quantizer_cpu, trainer_index, _res = make_flat_index(dim, metric_name, use_gpu, gpu_id)
    clustering = faiss.Clustering(dim, nlist)
    clustering.niter = niter
    clustering.max_points_per_centroid = max_points_per_centroid
    clustering.verbose = True
    if metric_name == "ip":
        clustering.spherical = True
    clustering.train(X_train, trainer_index)
    centroids = faiss.vector_to_array(clustering.centroids).reshape(nlist, dim)
    return np.ascontiguousarray(centroids, dtype=np.float32)


def add_vectors(index: faiss.IndexPreTransform, X: np.ndarray, batch_size: int) -> None:
    for start in range(0, X.shape[0], batch_size):
        end = min(X.shape[0], start + batch_size)
        index.add(np.ascontiguousarray(X[start:end], dtype=np.float32))


def compute_stats(X: np.ndarray) -> Dict[str, float]:
    norms = np.linalg.norm(X, axis=1)
    return {
        "num_vectors": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "norm_mean": float(norms.mean()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
    }


def compute_bucket_function_counts(
    function_bucket_hits: Sequence[np.ndarray],
    bucket_count: int,
) -> np.ndarray:
    bucket_sets: List[set] = [set() for _ in range(bucket_count)]
    for func_idx, bucket_ids in enumerate(function_bucket_hits):
        bucket_ids = np.asarray(bucket_ids, dtype=np.int32)
        if bucket_ids.size == 0:
            continue
        for bucket_id in np.unique(bucket_ids[bucket_ids >= 0]):
            bucket_sets[int(bucket_id)].add(func_idx)
    return np.asarray([len(bucket_sets[bucket_id]) for bucket_id in range(bucket_count)], dtype=np.int32)


def summarize_bucket_function_counts(bucket_function_counts: np.ndarray) -> Dict[str, float]:
    counts = np.asarray(bucket_function_counts, dtype=np.int32)
    nonempty = counts[counts > 0]
    return {
        "bucket_count": int(counts.size),
        "nonempty_bucket_count": int(nonempty.size),
        "mean_functions_per_bucket": float(counts.mean()) if counts.size else 0.0,
        "mean_functions_per_nonempty_bucket": float(nonempty.mean()) if nonempty.size else 0.0,
        "p50_functions_per_nonempty_bucket": float(np.percentile(nonempty, 50)) if nonempty.size else 0.0,
        "p90_functions_per_nonempty_bucket": float(np.percentile(nonempty, 90)) if nonempty.size else 0.0,
        "p95_functions_per_nonempty_bucket": float(np.percentile(nonempty, 95)) if nonempty.size else 0.0,
        "p99_functions_per_nonempty_bucket": float(np.percentile(nonempty, 99)) if nonempty.size else 0.0,
        "max_functions_per_bucket": int(nonempty.max()) if nonempty.size else 0,
    }


def diagnose_bucket_function_distribution(
    embedding_paths: Sequence[str],
    transform: faiss.VectorTransform,
    quantizer: faiss.Index,
    bucket_topk: int,
    embedding_suffix: str = DEFAULT_EMBEDDING_SUFFIX,
) -> Dict[str, float]:
    function_bucket_hits: List[np.ndarray] = []
    total_functions = 0
    total_slices = 0
    for _func_key, partial in iter_function_partials(
        embedding_paths, embedding_suffix=embedding_suffix
    ):
        transformed = np.ascontiguousarray(transform.apply_py(partial), dtype=np.float32)
        _scores, bucket_ids = quantizer.search(transformed, bucket_topk)
        function_bucket_hits.append(np.unique(bucket_ids[bucket_ids >= 0]).astype(np.int32))
        total_functions += 1
        total_slices += int(partial.shape[0])
    bucket_count = int(quantizer.ntotal)
    bucket_function_counts = compute_bucket_function_counts(function_bucket_hits, bucket_count)
    summary = summarize_bucket_function_counts(bucket_function_counts)
    summary.update(
        {
            "total_functions": int(total_functions),
            "total_slices": int(total_slices),
            "bucket_topk": int(bucket_topk),
        }
    )
    return summary


DEFAULT_INPUT_DIRS = [str(ROOT_DIR / "Coding" / "data" / "inline" / "pool")]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "Coding" / "artifacts" / "tmp"
DEFAULT_OUTPUT_INDEX = DEFAULT_OUTPUT_DIR / "inline_slice_pool.index"


def main():
    parser = argparse.ArgumentParser(description="Train a FAISS IVF index for Coding inline-slice partial embeddings.")
    parser.add_argument("--input-dir", action="append", default=None,
                        help=f"Directory containing *<embedding-suffix> pickle files. Can be specified multiple times. "
                             f"Default: {DEFAULT_INPUT_DIRS}")
    parser.add_argument(
        "--embedding-suffix",
        default=DEFAULT_EMBEDDING_SUFFIX,
        help=f"Filename suffix of the per-binary embedding pickles to scan "
             f"(default: {DEFAULT_EMBEDDING_SUFFIX}). Use "
             f"`{LEGACY_EMBEDDING_SUFFIX}` to target the legacy varchain output.",
    )
    parser.add_argument("--output-index", default=str(DEFAULT_OUTPUT_INDEX))
    parser.add_argument("--output-meta", default=None)
    parser.add_argument("--metric", choices=("ip", "l2"), default="ip")
    parser.add_argument("--pca-dim", type=int, default=0, help="Set 0 to disable dimensionality reduction.")
    parser.add_argument("--nlist", type=int, default=0, help="0 means auto-select based on partial vector count.")
    parser.add_argument(
        "--auto-nlist-cap",
        type=int,
        default=65536,
        help="Upper bound for auto-selected nlist (only used when --nlist=0). "
             "Bounds the streaming reservoir size to ~16 * cap rows. "
             "Set higher (e.g. 262144) for >20M-vector pools when --nlist is left auto.",
    )
    parser.add_argument("--train-size", type=int, default=0, help="0 means auto-select based on nlist.")
    parser.add_argument("--niter", type=int, default=25)
    parser.add_argument("--max-points-per-centroid", type=int, default=256)
    parser.add_argument("--add-batch", type=int, default=100_000)
    parser.add_argument("--add-all", action="store_true", help="Also add all vectors into IVF. Current bucket-only retrieval does not require this.")
    parser.add_argument("--sample-mode", choices=("balanced", "uniform"), default="balanced")
    parser.add_argument("--signature-bits", type=int, default=24)
    parser.add_argument("--per-signature-cap", type=int, default=0, help="0 means auto cap based on train_size/signature_count.")
    parser.add_argument("--per-binary-train-cap", type=int, default=0, help="0 disables per-binary row cap before balanced sampling.")
    parser.add_argument("--bucket-topk-diagnostic", type=int, default=24)
    parser.add_argument("--bucket-stats-output", default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Pickle-loader worker processes. 0 = auto (min(8, cpu_count)). "
             "Streaming reservoir keeps RAM bounded to train_size * dim * 4 bytes.",
    )
    args = parser.parse_args()

    if not args.input_dir:
        args.input_dir = list(DEFAULT_INPUT_DIRS)

    embedding_paths = discover_embedding_pickles(
        args.input_dir, embedding_suffix=args.embedding_suffix
    )
    if not embedding_paths:
        raise FileNotFoundError(
            f"no *{args.embedding_suffix} found under input dirs: {args.input_dir}"
        )

    nlist_estimate = args.nlist if args.nlist > 0 else max(args.auto_nlist_cap, 1)
    if args.train_size > 0:
        streaming_train_size = int(args.train_size)
    else:
        streaming_train_size = max(16 * nlist_estimate, 30_000)
    print(
        f"[1/7] streaming partial embeddings from {len(embedding_paths)} pickle files "
        f"(suffix={args.embedding_suffix})"
    )
    t0 = time.time()
    if args.sample_mode == "balanced":
        sample, sample_stats, vector_stats = stream_training_sample(
            embedding_paths,
            train_size=streaming_train_size,
            seed=args.seed,
            signature_bits=args.signature_bits,
            per_binary_cap=args.per_binary_train_cap,
            per_signature_cap=args.per_signature_cap,
            embedding_suffix=args.embedding_suffix,
            workers=args.workers,
        )
    else:
        sample, sample_stats, vector_stats = stream_training_sample(
            embedding_paths,
            train_size=streaming_train_size,
            seed=args.seed,
            signature_bits=0,
            per_binary_cap=0,
            per_signature_cap=0,
            embedding_suffix=args.embedding_suffix,
            workers=args.workers,
        )
        sample_stats["sample_mode"] = "uniform"
    num_vectors = int(vector_stats["num_vectors"])
    print(
        f"      streamed {num_vectors} raw vectors, sample_rows={sample.shape[0]}, "
        f"dim={vector_stats['dim']} in {time.time() - t0:.2f}s"
    )

    nlist = args.nlist or choose_nlist(num_vectors)
    final_train_size = args.train_size or choose_train_size(num_vectors, nlist)
    final_train_size = min(final_train_size, sample.shape[0])
    if final_train_size < nlist:
        raise ValueError(
            f"train_size ({final_train_size}) must be >= nlist ({nlist}); "
            f"streaming sample rows = {sample.shape[0]}"
        )
    if final_train_size < sample.shape[0]:
        sample = np.ascontiguousarray(sample[:final_train_size], dtype=np.float32)
    sample_stats["train_size"] = int(final_train_size)
    sample_stats["selected_rows"] = int(sample.shape[0])
    train_size = final_train_size

    print(f"[2/7] sampling done: train_size={train_size}, nlist={nlist}, mode={sample_stats['sample_mode']}")

    print(f"[3/7] training transform: {vector_stats['dim']} -> {args.pca_dim if args.pca_dim > 0 else vector_stats['dim']}")
    transform, X_train_t, out_dim = train_transform(sample, int(vector_stats["dim"]), args.pca_dim)

    print(f"[4/7] training coarse centroids: metric={args.metric}, dim={out_dim}, nlist={nlist}, niter={args.niter}")
    centroids = train_ivf_centroids(
        X_train_t,
        out_dim,
        nlist,
        args.metric,
        args.niter,
        args.max_points_per_centroid,
        not args.cpu,
        args.gpu_id,
    )

    print("[5/7] building FAISS index")
    metric = metric_from_name(args.metric)
    if metric == faiss.METRIC_INNER_PRODUCT:
        quantizer = faiss.IndexFlatIP(out_dim)
    else:
        quantizer = faiss.IndexFlatL2(out_dim)
    quantizer.add(centroids)
    ivf = faiss.IndexIVFFlat(quantizer, out_dim, nlist, metric)
    ivf.is_trained = True
    ivf.nprobe = min(8, nlist)
    index = faiss.IndexPreTransform(transform, ivf)
    if args.add_all:
        print(f"      streaming add of all {num_vectors} vectors into IVF (workers={args.workers or 'auto'})")
        added = stream_add_all_to_index(
            embedding_paths, index, args.embedding_suffix, args.workers
        )
        print(f"      index.ntotal={index.ntotal} (streamed={added})")
    else:
        print("      skip add_all; current bucket-only retrieval only needs PCA + quantizer centroids")

    print(f"[6/7] saving index -> {args.output_index}")
    os.makedirs(os.path.dirname(args.output_index) or ".", exist_ok=True)
    faiss.write_index(index, args.output_index)

    if args.bucket_topk_diagnostic > 0:
        diag_quantizer = quantizer
        diag_res = None
        if not args.cpu:
            try:
                diag_res = faiss.StandardGpuResources()
                diag_quantizer = faiss.index_cpu_to_gpu(diag_res, args.gpu_id, quantizer)
                print(
                    f"[7/7] diagnosing bucket distribution with bucket_topk={args.bucket_topk_diagnostic} "
                    f"(quantizer on GPU {args.gpu_id})"
                )
            except Exception as exc:
                print(f"[7/7] GPU quantizer unavailable ({exc}); falling back to CPU diagnostic")
                diag_quantizer = quantizer
                diag_res = None
        else:
            print(f"[7/7] diagnosing bucket distribution with bucket_topk={args.bucket_topk_diagnostic} (CPU)")
        bucket_stats = stream_diagnose_bucket_function_distribution(
            embedding_paths,
            transform,
            diag_quantizer,
            args.bucket_topk_diagnostic,
            embedding_suffix=args.embedding_suffix,
            workers=args.workers,
        )
        del diag_quantizer
        del diag_res
    else:
        print("[7/7] skipping bucket distribution diagnosis")
        bucket_stats = {
            "bucket_count": int(nlist),
            "nonempty_bucket_count": 0,
            "mean_functions_per_bucket": 0.0,
            "mean_functions_per_nonempty_bucket": 0.0,
            "p50_functions_per_nonempty_bucket": 0.0,
            "p90_functions_per_nonempty_bucket": 0.0,
            "p95_functions_per_nonempty_bucket": 0.0,
            "p99_functions_per_nonempty_bucket": 0.0,
            "max_functions_per_bucket": 0,
            "total_functions": 0,
            "total_slices": 0,
            "bucket_topk": 0,
            "diagnostic_skipped": True,
        }

    meta = {
        "input_dirs": list(args.input_dir),
        "embedding_suffix": args.embedding_suffix,
        "embedding_files": embedding_paths,
        "num_vectors": int(num_vectors),
        "input_dim": int(vector_stats["dim"]),
        "metric": args.metric,
        "pca_dim": int(out_dim),
        "nlist": int(nlist),
        "train_size": int(train_size),
        "niter": int(args.niter),
        "max_points_per_centroid": int(args.max_points_per_centroid),
        "add_all": bool(args.add_all),
        "sample_stats": sample_stats,
        "bucket_distribution": bucket_stats,
        "bucket_topk_note": "Current Coding/pool_inline_eval.py uses quantizer.search(topk), not ivf.search(nprobe).",
        "vector_stats": vector_stats,
    }
    meta_path = args.output_meta or f"{args.output_index}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    bucket_stats_path = args.bucket_stats_output or f"{args.output_index}.bucket_stats.json"
    with open(bucket_stats_path, "w", encoding="utf-8") as f:
        json.dump(bucket_stats, f, indent=2, ensure_ascii=False)
    print(f"[done] meta -> {meta_path}")
    print(f"[done] bucket stats -> {bucket_stats_path}")


if __name__ == "__main__":
    main()
