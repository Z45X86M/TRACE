"""Train the FAISS IVF **centroids** for whole-function global embeddings.

This is step 1 of the global-ANN pipeline, mirroring the slice-side split:

    Step 1 (this script):
        Read ``<binary>_feature_global_embeddings.pkl`` from the **source pool**
        (Coding/data/inline/pool) and train coarse IVF centroids over the
        whole-function global vectors. Output is a centroids-only index file
        (no pool corpus baked in by default).

    Step 2 (build_global_ann_artifacts.py):
        Load the centroids from step 1 + the **materialized pool** (whichever
        variant: tmp/pool, tmp/pool_noinline, tmp/pool_lossless, etc.), add
        those pool vectors into the index, and write the populated database
        that downstream eval will search against.

    Step 3 (global_ann_inline_eval.py):
        Load step 2's populated index, encode queries, IVF search.

Defaults are wired so that re-running step 1 is rare (only when the source
pool distribution changes); step 2 runs once per pool variant.

Typical usage::

    python3 Coding/train_faiss_global_ann.py        # uses all defaults

Scaling: the pickle/json loading step uses multi-process I/O with a streaming
reservoir, so peak RAM is ``train_size * dim * 4`` bytes regardless of the
total pool size. This is the same pattern as
``train_faiss_inline_slice.py``; tune ``--workers`` for I/O concurrency and
``--auto-nlist-cap`` to bound the upper-bound reservoir when ``--nlist=0``.
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import faiss
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
for path in (str(ROOT_DIR), str(CURRENT_DIR), str(ROOT_DIR / "V3")):
    if path not in sys.path:
        sys.path.append(path)

from utils import discover_binaries
from utils import read_json, read_pickle

from train_faiss_inline_slice import (
    _stable_path_seed,
    add_vectors,
    build_signature_projection,
    choose_nlist,
    choose_train_size,
    compute_signature_keys,
    compute_stats,
    metric_from_name,
    sample_rows,
    sample_rows_frequency_balanced,
    train_ivf_centroids,
    train_transform,
)

POOL_EMB_SUFFIX = "_feature_global_embeddings.pkl"
POOL_FEATURE_SUFFIX = "_bb_slice_feature.json"
GLOBAL_EMB_SUFFIX = "_feature_global_embeddings.pkl"


def iter_pool_globals(
    pool_dir: str,
) -> Iterator[Tuple[str, str, str, np.ndarray]]:
    """Legacy single-process iterator over (binary_name, addr, func_name, vec).

    Kept for backwards compatibility and ad-hoc inspection. The current
    centroid-training path uses the streaming worker pool instead — see
    ``stream_training_sample_globals``.
    """
    for binary in discover_binaries(pool_dir):
        binary_name = os.path.basename(binary)
        feature_path = f"{binary}{POOL_FEATURE_SUFFIX}"
        if not os.path.exists(feature_path):
            continue
        feature_info = read_json(feature_path)

        global_cache_path = f"{binary}{GLOBAL_EMB_SUFFIX}"
        legacy_embedding_path = f"{binary}{POOL_EMB_SUFFIX}"
        if os.path.exists(global_cache_path):
            cache = read_pickle(global_cache_path)
        elif os.path.exists(legacy_embedding_path):
            cache = read_pickle(legacy_embedding_path)
        else:
            continue

        for addr, item in feature_info.items():
            if addr not in cache:
                continue
            bundle = cache[addr]
            if not isinstance(bundle, dict):
                continue
            vec = bundle.get("global")
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            func_name = str(item.get("func_name") or "")
            yield binary_name, addr, func_name, arr


def load_pool_matrix(pool_dir: str) -> Tuple[np.ndarray, List[Tuple[str, str, str]], np.ndarray]:
    """Legacy in-memory pool matrix loader; uses the iterator above.

    Memory-bounded by total pool size. Kept for back-compat — use the streaming
    pipeline (``_resolve_pool_binary_inputs`` + ``stream_training_sample_globals``)
    for 10M+-row pools.
    """
    rows: List[np.ndarray] = []
    meta: List[Tuple[str, str, str]] = []
    binary_names: List[str] = []
    for binary_name, addr, func_name, vec in iter_pool_globals(pool_dir):
        rows.append(vec)
        meta.append((binary_name, addr, func_name))
        binary_names.append(binary_name)
    if not rows:
        raise ValueError(f"no globals found under pool dir {pool_dir}")
    matrix = np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    return matrix, meta, np.asarray(binary_names, dtype=object)


def _normalize_pool_dirs(pool_dir) -> List[str]:
    """Accept a single dir (str / Path) or sequence of dirs. Returns a list."""
    if isinstance(pool_dir, (str, os.PathLike)):
        return [str(pool_dir)]
    out = [str(p) for p in pool_dir if p]
    if not out:
        raise ValueError("pool_dir is empty")
    return out


def _resolve_pool_binary_inputs(pool_dir) -> List[Tuple[str, str, str]]:
    """Discover binaries with a (feature.json, global-cache pickle) pair.

    ``pool_dir`` accepts either a single dir or a sequence of dirs. Across
    multiple roots, on basename collision the first listed root wins.

    Returns sorted ``(binary_path, feature_path, cache_path)`` triples.
    ``cache_path`` prefers the new ``*_feature_global_embeddings.pkl`` and
    falls back to legacy ``*_bb_slice_embeddings.pkl``.
    """
    pool_dirs = _normalize_pool_dirs(pool_dir)
    triples: List[Tuple[str, str, str]] = []
    seen_basenames: Set[str] = set()
    collisions = 0
    for root in pool_dirs:
        for binary in discover_binaries(root):
            basename = os.path.basename(binary)
            if basename in seen_basenames:
                collisions += 1
                continue
            feature_path = f"{binary}{POOL_FEATURE_SUFFIX}"
            if not os.path.exists(feature_path):
                continue
            global_cache_path = f"{binary}{GLOBAL_EMB_SUFFIX}"
            legacy_cache_path = f"{binary}{POOL_EMB_SUFFIX}"
            if os.path.exists(global_cache_path):
                triples.append((binary, feature_path, global_cache_path))
                seen_basenames.add(basename)
            elif os.path.exists(legacy_cache_path):
                triples.append((binary, feature_path, legacy_cache_path))
                seen_basenames.add(basename)
    if collisions:
        print(
            f"[train-faiss-global-ann] skipped {collisions} duplicate-basename "
            f"binaries across pool-dir roots (first-root-wins)",
            flush=True,
        )
    return triples


def _read_globals_for_binary(
    feature_path: str,
    cache_path: str,
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """Return (X, meta_rows) for one binary. Both are aligned 1:1.

    Each row is L2-normalized to match what ``load_pool_matrix`` does.
    """
    with open(feature_path, "r", encoding="utf-8") as f:
        feature_info = json.load(f)
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    rows: List[np.ndarray] = []
    meta_rows: List[Tuple[str, str]] = []
    for addr, item in feature_info.items():
        if addr not in cache:
            continue
        bundle = cache[addr]
        if not isinstance(bundle, dict):
            continue
        vec = bundle.get("global")
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            continue
        rows.append(arr)
        meta_rows.append((addr, str(item.get("func_name") or "")))
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), []
    X = np.stack(rows, axis=0).astype(np.float32, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-12)
    return np.ascontiguousarray(X, dtype=np.float32), meta_rows


def _load_pool_globals_for_sampling(args):
    """Worker: load one binary's globals, apply per-binary balancing.

    Same balancing semantics as the slice-side worker:
      1. compute signatures and cap per signature (signature_bits > 0);
      2. random-trim to per_binary_cap.
    """
    (binary_path, feature_path, cache_path, per_binary_cap,
     signature_bits, per_signature_cap, seed) = args
    binary_name = os.path.basename(binary_path)
    try:
        X, _meta_rows = _read_globals_for_binary(feature_path, cache_path)
    except Exception as exc:
        return binary_path, None, {
            "binary_name": binary_name,
            "raw_rows": 0,
            "kept_rows": 0,
            "norm_sum": 0.0,
            "norm_min": float("inf"),
            "norm_max": float("-inf"),
            "unique_signatures": np.zeros((0,), dtype=object),
            "error": str(exc)[:200],
        }
    if X.size == 0 or X.shape[0] == 0:
        return binary_path, None, {
            "binary_name": binary_name,
            "raw_rows": 0,
            "kept_rows": 0,
            "norm_sum": 0.0,
            "norm_min": float("inf"),
            "norm_max": float("-inf"),
            "unique_signatures": np.zeros((0,), dtype=object),
        }
    raw_rows = int(X.shape[0])
    flat_norms = np.linalg.norm(X, axis=1)
    norm_sum = float(flat_norms.sum())
    norm_min_val = float(flat_norms.min())
    norm_max_val = float(flat_norms.max())

    unique_signatures = np.zeros((0,), dtype=object)
    if signature_bits > 0 and X.shape[0] > 0:
        projection = build_signature_projection(X.shape[1], signature_bits, seed)
        sigs = compute_signature_keys(X, projection)
        rng = np.random.default_rng(_stable_path_seed(binary_path, seed))
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
        rng = np.random.default_rng(_stable_path_seed(binary_path, seed) ^ 0xA5A5A5A5)
        idx = rng.permutation(X.shape[0])[:per_binary_cap]
        X = np.ascontiguousarray(X[idx], dtype=np.float32)

    return binary_path, X, {
        "binary_name": binary_name,
        "raw_rows": raw_rows,
        "kept_rows": int(X.shape[0]),
        "norm_sum": norm_sum,
        "norm_min": norm_min_val,
        "norm_max": norm_max_val,
        "unique_signatures": unique_signatures,
    }


def _load_pool_globals_raw(args):
    """Worker: load one binary's globals + aligned meta, no balancing.

    Used by ``stream_add_all_globals_to_index`` and the bucket diagnostic.
    Returns ``(binary_path, binary_name, X, meta_rows, err)`` where
    ``meta_rows`` is a list of ``(addr, func_name)`` aligned 1:1 with X rows.
    """
    binary_path, feature_path, cache_path = args
    binary_name = os.path.basename(binary_path)
    try:
        X, meta_rows = _read_globals_for_binary(feature_path, cache_path)
    except Exception as exc:
        return binary_path, binary_name, None, [], str(exc)[:200]
    if X.size == 0 or X.shape[0] == 0:
        return binary_path, binary_name, None, [], ""
    return binary_path, binary_name, X, meta_rows, ""


def stream_training_sample_globals(
    binary_inputs: Sequence[Tuple[str, str, str]],
    train_size: int,
    seed: int,
    signature_bits: int,
    per_binary_cap: int,
    per_signature_cap: int,
    workers: int,
) -> Tuple[np.ndarray, Dict[str, object], Dict[str, float]]:
    """Stream-load global embeddings and build a bounded reservoir sample.

    Peak memory is ``train_size * dim * 4`` bytes regardless of total pool size.
    """
    if workers <= 0:
        workers = max(1, min(8, mp.cpu_count() or 1))
    workers = min(workers, max(1, len(binary_inputs)))

    task_args = [
        (binary, feature_path, cache_path, per_binary_cap,
         signature_bits, per_signature_cap, seed)
        for binary, feature_path, cache_path in binary_inputs
    ]
    print(
        f"      streaming {len(task_args)} binaries with workers={workers} "
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
            _load_pool_globals_for_sampling, task_args, chunksize=1
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
        raise ValueError("no global embeddings collected from pool dir")
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
            print(f"        {os.path.basename(fp)}: {err}", flush=True)

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
            "per-binary balanced (signature-cap + binary-cap applied inside each binary); "
            "global reservoir (Algorithm R) merges the per-binary samples to train_size"
        ),
    }
    return sample, sample_stats, vector_stats


def stream_diagnose_bucket_function_distribution_globals(
    binary_inputs: Sequence[Tuple[str, str, str]],
    transform: faiss.VectorTransform,
    quantizer: faiss.Index,
    bucket_topk: int,
    workers: int,
) -> Dict[str, float]:
    """Parallel-load variant of the bucket distribution diagnostic for globals.

    Workers load (feature, cache) pairs in parallel; the main process holds the
    (possibly GPU-resident) quantizer and does the transform + search per file.
    """
    bucket_count = int(quantizer.ntotal)
    if bucket_topk <= 0 or not binary_inputs:
        return {
            "bucket_count": bucket_count,
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
            "bucket_topk": int(bucket_topk),
            "diagnostic_skipped": True,
        }
    if workers <= 0:
        workers = max(1, min(8, mp.cpu_count() or 1))
    workers = min(workers, max(1, len(binary_inputs)))

    counts = np.zeros(bucket_count, dtype=np.int64)
    total_functions = 0
    binary_count = 0
    t_start = time.time()
    task_args = list(binary_inputs)

    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for path, _binary_name, X, _meta_rows, err in pool.imap_unordered(
            _load_pool_globals_raw, task_args, chunksize=1
        ):
            binary_count += 1
            if err:
                print(f"      diag: failed {os.path.basename(path)}: {err}", flush=True)
                continue
            if X is None or X.shape[0] == 0:
                continue
            transformed = np.ascontiguousarray(transform.apply_py(X), dtype=np.float32)
            _, bucket_ids = quantizer.search(transformed, bucket_topk)
            for row in bucket_ids:
                unique_ids = np.unique(row[row >= 0])
                if unique_ids.size > 0:
                    counts[unique_ids.astype(np.int64)] += 1
            total_functions += int(X.shape[0])
            if binary_count % 32 == 0 or binary_count == len(task_args):
                elapsed = max(time.time() - t_start, 1e-6)
                print(
                    f"      diag {binary_count}/{len(task_args)} functions={total_functions} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    nonempty = counts[counts > 0]
    return {
        "bucket_count": bucket_count,
        "nonempty_bucket_count": int(nonempty.size),
        "mean_functions_per_bucket": float(counts.mean()) if counts.size else 0.0,
        "mean_functions_per_nonempty_bucket": float(nonempty.mean()) if nonempty.size else 0.0,
        "p50_functions_per_nonempty_bucket": float(np.percentile(nonempty, 50)) if nonempty.size else 0.0,
        "p90_functions_per_nonempty_bucket": float(np.percentile(nonempty, 90)) if nonempty.size else 0.0,
        "p95_functions_per_nonempty_bucket": float(np.percentile(nonempty, 95)) if nonempty.size else 0.0,
        "p99_functions_per_nonempty_bucket": float(np.percentile(nonempty, 99)) if nonempty.size else 0.0,
        "max_functions_per_bucket": int(nonempty.max()) if nonempty.size else 0,
        "total_functions": int(total_functions),
        "total_slices": int(total_functions),
        "bucket_topk": int(bucket_topk),
    }


def stream_add_all_globals_to_index(
    binary_inputs: Sequence[Tuple[str, str, str]],
    index: faiss.Index,
    workers: int,
) -> Tuple[int, List[Tuple[str, str, str]]]:
    """Stream globals into the IVF in input order; return (n_added, pool_meta).

    Ordered ``imap`` is used so FAISS internal ids map deterministically to
    ``pool_meta[i] = (binary_name, func_addr, func_name)`` for the contract
    consumed by ``global_ann_inline_eval._load_pretrained_index``.
    """
    if workers <= 0:
        workers = max(1, min(8, mp.cpu_count() or 1))
    workers = min(workers, max(1, len(binary_inputs)))
    task_args = list(binary_inputs)
    t_start = time.time()
    binary_count = 0
    total_added = 0
    pool_meta: List[Tuple[str, str, str]] = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for path, binary_name, X, meta_rows, err in pool.imap(
            _load_pool_globals_raw, task_args, chunksize=1
        ):
            binary_count += 1
            if err:
                print(f"      add-all: failed {os.path.basename(path)}: {err}", flush=True)
                continue
            if X is None or X.shape[0] == 0:
                continue
            index.add(np.ascontiguousarray(X, dtype=np.float32))
            total_added += int(X.shape[0])
            for addr, func_name in meta_rows:
                pool_meta.append((binary_name, addr, func_name))
            if binary_count % 32 == 0 or binary_count == len(task_args):
                elapsed = max(time.time() - t_start, 1e-6)
                print(
                    f"      add-all {binary_count}/{len(task_args)} added={total_added} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    return total_added, pool_meta


def diagnose_bucket_function_distribution_globals(
    matrix: np.ndarray,
    transform: faiss.VectorTransform,
    quantizer: faiss.Index,
    bucket_topk: int,
) -> dict:
    """Legacy in-memory bucket diagnostic. Use the streaming variant for large pools."""
    if matrix.shape[0] == 0 or bucket_topk <= 0:
        return {
            "bucket_count": int(quantizer.ntotal),
            "nonempty_bucket_count": 0,
            "mean_functions_per_bucket": 0.0,
            "mean_functions_per_nonempty_bucket": 0.0,
            "p50_functions_per_nonempty_bucket": 0.0,
            "p90_functions_per_nonempty_bucket": 0.0,
            "p95_functions_per_nonempty_bucket": 0.0,
            "p99_functions_per_nonempty_bucket": 0.0,
            "max_functions_per_bucket": 0,
            "total_functions": int(matrix.shape[0]),
            "total_slices": int(matrix.shape[0]),
            "bucket_topk": int(bucket_topk),
            "diagnostic_skipped": True,
        }
    transformed = np.ascontiguousarray(transform.apply_py(matrix), dtype=np.float32)
    _, bucket_ids = quantizer.search(transformed, bucket_topk)
    bucket_count = int(quantizer.ntotal)
    counts = np.zeros(bucket_count, dtype=np.int64)
    for row in bucket_ids:
        unique_ids = np.unique(row[row >= 0])
        for b in unique_ids:
            counts[int(b)] += 1
    nonempty = counts[counts > 0]
    return {
        "bucket_count": int(bucket_count),
        "nonempty_bucket_count": int(nonempty.size),
        "mean_functions_per_bucket": float(counts.mean()) if counts.size else 0.0,
        "mean_functions_per_nonempty_bucket": float(nonempty.mean()) if nonempty.size else 0.0,
        "p50_functions_per_nonempty_bucket": float(np.percentile(nonempty, 50)) if nonempty.size else 0.0,
        "p90_functions_per_nonempty_bucket": float(np.percentile(nonempty, 90)) if nonempty.size else 0.0,
        "p95_functions_per_nonempty_bucket": float(np.percentile(nonempty, 95)) if nonempty.size else 0.0,
        "p99_functions_per_nonempty_bucket": float(np.percentile(nonempty, 99)) if nonempty.size else 0.0,
        "max_functions_per_bucket": int(nonempty.max()) if nonempty.size else 0,
        "total_functions": int(matrix.shape[0]),
        "total_slices": int(matrix.shape[0]),
        "bucket_topk": int(bucket_topk),
    }


DEFAULT_POOL_DIR = str((ROOT_DIR / "Coding" / "data" / "inline" / "pool").resolve())
DEFAULT_POOL_DIRS = [DEFAULT_POOL_DIR]
DEFAULT_OUTPUT_INDEX = str((ROOT_DIR / "Coding" / "artifacts" / "faiss" / "global_ann_centroids.index").resolve())


def main():
    parser = argparse.ArgumentParser(
        description="Train a FAISS index for whole-function global embeddings (global ANN baseline).",
    )
    parser.add_argument("--pool-dir",
                        nargs="+",
                        default=list(DEFAULT_POOL_DIRS),
                        help=f"One or more source pool dirs holding <binary>{GLOBAL_EMB_SUFFIX} and "
                             f"<binary>{POOL_FEATURE_SUFFIX} — used for centroid training. "
                             f"On basename collision across roots, the first listed root wins. "
                             f"Default: {DEFAULT_POOL_DIRS}")
    parser.add_argument("--output-index", default=DEFAULT_OUTPUT_INDEX)
    parser.add_argument("--output-meta", default=None)
    parser.add_argument("--metric", choices=("ip", "l2"), default="ip")
    parser.add_argument("--pca-dim", type=int, default=0, help="Set 0 to disable dimensionality reduction.")
    parser.add_argument("--nlist", type=int, default=0, help="0 = auto-select from vector count.")
    parser.add_argument(
        "--auto-nlist-cap",
        type=int,
        default=65536,
        help="Upper bound for auto-selected nlist (only used when --nlist=0). "
             "Bounds the streaming reservoir size to ~16 * cap rows. "
             "Set higher (e.g. 262144) for >20M-vector pools when --nlist is left auto.",
    )
    parser.add_argument("--train-size", type=int, default=0, help="0 = auto-select from nlist.")
    parser.add_argument("--niter", type=int, default=25)
    parser.add_argument("--max-points-per-centroid", type=int, default=256)
    parser.add_argument("--add-batch", type=int, default=100_000)
    parser.add_argument("--add-all", action="store_true",
                        help="Also bake all source-pool vectors into the IVF (legacy single-step "
                             "behavior). Default OFF — step 2 (build_global_ann_artifacts.py) "
                             "populates the index with the per-experiment materialized pool.")
    parser.add_argument("--sample-mode", choices=("balanced", "uniform"), default="balanced")
    parser.add_argument("--signature-bits", type=int, default=24)
    parser.add_argument("--per-signature-cap", type=int, default=0)
    parser.add_argument("--per-binary-train-cap", type=int, default=0)
    parser.add_argument("--bucket-topk-diagnostic", type=int, default=8)
    parser.add_argument("--bucket-stats-output", default=None)
    parser.add_argument("--nprobe", type=int, default=8, help="Default nprobe baked into the index (eval can override).")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Pickle/json loader worker processes. 0 = auto (min(8, cpu_count)). "
             "Streaming reservoir keeps RAM bounded to train_size * dim * 4 bytes.",
    )
    args = parser.parse_args()

    add_all = bool(args.add_all)

    pool_dirs = _normalize_pool_dirs(args.pool_dir)
    print(f"[1/7] discovering binaries under pool={pool_dirs}")
    t0 = time.time()
    binary_inputs = _resolve_pool_binary_inputs(pool_dirs)
    if not binary_inputs:
        raise ValueError(f"no globals found under pool dirs {pool_dirs}")
    print(f"      found {len(binary_inputs)} binaries with global embeddings in {time.time() - t0:.2f}s")

    nlist_estimate = args.nlist if args.nlist > 0 else max(args.auto_nlist_cap, 1)
    if args.train_size > 0:
        streaming_train_size = int(args.train_size)
    else:
        streaming_train_size = max(16 * nlist_estimate, 30_000)

    print(
        f"[2/7] streaming global embeddings (reservoir target={streaming_train_size}, "
        f"workers={args.workers or 'auto'})"
    )
    t_stream = time.time()
    if args.sample_mode == "balanced":
        sample, sample_stats, vector_stats = stream_training_sample_globals(
            binary_inputs,
            train_size=streaming_train_size,
            seed=args.seed,
            signature_bits=args.signature_bits,
            per_binary_cap=args.per_binary_train_cap,
            per_signature_cap=args.per_signature_cap,
            workers=args.workers,
        )
    else:
        sample, sample_stats, vector_stats = stream_training_sample_globals(
            binary_inputs,
            train_size=streaming_train_size,
            seed=args.seed,
            signature_bits=0,
            per_binary_cap=0,
            per_signature_cap=0,
            workers=args.workers,
        )
        sample_stats["sample_mode"] = "uniform"
    num_vectors = int(vector_stats["num_vectors"])
    print(
        f"      streamed {num_vectors} raw vectors, sample_rows={sample.shape[0]}, "
        f"dim={vector_stats['dim']} in {time.time() - t_stream:.2f}s"
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

    print(f"[3/7] sampling done: train_size={train_size}, nlist={nlist}, mode={sample_stats['sample_mode']}")

    print(f"[4/7] training transform: {vector_stats['dim']} -> {args.pca_dim if args.pca_dim > 0 else vector_stats['dim']}")
    transform, X_train_t, out_dim = train_transform(sample, int(vector_stats["dim"]), args.pca_dim)

    print(f"[5/7] training coarse centroids: metric={args.metric}, dim={out_dim}, nlist={nlist}, niter={args.niter}")
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

    print("[6/7] building FAISS index")
    metric = metric_from_name(args.metric)
    if metric == faiss.METRIC_INNER_PRODUCT:
        quantizer = faiss.IndexFlatIP(out_dim)
    else:
        quantizer = faiss.IndexFlatL2(out_dim)
    quantizer.add(centroids)
    ivf = faiss.IndexIVFFlat(quantizer, out_dim, nlist, metric)
    ivf.is_trained = True
    ivf.nprobe = max(1, min(args.nprobe, nlist))
    index = faiss.IndexPreTransform(transform, ivf)

    pool_meta_rows: List[Tuple[str, str, str]] = []
    if add_all:
        print(f"      streaming add of all {num_vectors} vectors into IVF (workers={args.workers or 'auto'})")
        added, pool_meta_rows = stream_add_all_globals_to_index(
            binary_inputs, index, args.workers
        )
        print(f"      index.ntotal={index.ntotal} (streamed={added})")
    else:
        print("      skip add_all (downstream must not rely on index.search)")

    print(f"[7/7] saving index -> {args.output_index}")
    os.makedirs(os.path.dirname(args.output_index) or ".", exist_ok=True)
    faiss.write_index(index, args.output_index)

    if args.bucket_topk_diagnostic > 0:
        diag_quantizer = quantizer
        diag_res = None
        if not args.cpu:
            try:
                diag_res = faiss.StandardGpuResources()
                diag_quantizer = faiss.index_cpu_to_gpu(diag_res, args.gpu_id, quantizer)
            except Exception as exc:
                print(f"[diag] GPU quantizer unavailable ({exc}); falling back to CPU")
                diag_quantizer = quantizer
                diag_res = None
        bucket_stats = stream_diagnose_bucket_function_distribution_globals(
            binary_inputs,
            transform,
            diag_quantizer,
            args.bucket_topk_diagnostic,
            workers=args.workers,
        )
        del diag_quantizer
        del diag_res
    else:
        bucket_stats = {"diagnostic_skipped": True, "bucket_count": int(nlist)}

    meta_payload = {
        "pool_dir": [os.path.abspath(p) for p in pool_dirs],
        "num_vectors": int(num_vectors),
        "input_dim": int(vector_stats["dim"]),
        "metric": args.metric,
        "pca_dim": int(out_dim),
        "nlist": int(nlist),
        "nprobe_default": int(ivf.nprobe),
        "train_size": int(train_size),
        "niter": int(args.niter),
        "max_points_per_centroid": int(args.max_points_per_centroid),
        "add_all": bool(add_all),
        "sample_stats": sample_stats,
        "bucket_distribution": bucket_stats,
        "vector_stats": vector_stats,
        "pool_meta": [
            {"binary_name": b, "func_addr": a, "func_name": f}
            for b, a, f in pool_meta_rows
        ],
    }
    meta_path = args.output_meta or f"{args.output_index}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta_payload, fp, indent=2, ensure_ascii=False)
    bucket_stats_path = args.bucket_stats_output or f"{args.output_index}.bucket_stats.json"
    with open(bucket_stats_path, "w", encoding="utf-8") as fp:
        json.dump(bucket_stats, fp, indent=2, ensure_ascii=False)
    print(f"[done] meta -> {meta_path}")
    print(f"[done] bucket stats -> {bucket_stats_path}")


if __name__ == "__main__":
    main()
