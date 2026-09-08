"""Train a FAISS IVF index for **HermesSim** whole-function embeddings.

Baseline-comparison sibling of ``train_faiss_global_ann.py``. HermesSim
(`Chat/HermesSim.pdf`) represents each function as a single graph-semantics
(SOG → GGNN) vector. This script trains coarse IVF centroids over the
HermesSim **pool** embeddings, exactly mirroring the global-ANN training step:

    * Input  : ``<binary>_HermesSim_embeddings.pkl`` ({addr: {'global': vec}})
               + ``<binary>_HermesSim_feature.json`` (addr -> {'func_name': ...})
               under the source pool (Coding/data/inline/pool).
    * Output : a single centroids-only IVF index
               ``Coding/artifacts/faiss/hermessim_centroids.index`` (+ .meta.json).

Only the file suffixes differ from ``train_faiss_global_ann.py``; the streaming
reservoir / centroid training / bucket-diagnostic machinery is imported and
reused verbatim, so RAM stays bounded to ``train_size * dim * 4`` bytes. The
HermesSim vectors are 384-d (auto-detected; no hard-coded dim).

By default the index is centroids-only (``--add-all`` off), matching the
global-ANN split where a separate populate step bakes in the materialized pool.
Pass ``--add-all`` to also stream every pool vector into the IVF and emit the
per-row ``pool_meta`` contract.

Typical usage::

    python3 Coding/train_faiss_hermessim.py        # uses all defaults
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Set, Tuple

import faiss
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
for path in (str(ROOT_DIR), str(CURRENT_DIR), str(ROOT_DIR / "V3")):
    if path not in sys.path:
        sys.path.append(path)

from utils import discover_binaries

from train_faiss_inline_slice import (
    choose_nlist,
    choose_train_size,
    metric_from_name,
    train_ivf_centroids,
    train_transform,
)
from train_faiss_global_ann import (
    _normalize_pool_dirs,
    stream_add_all_globals_to_index,
    stream_diagnose_bucket_function_distribution_globals,
    stream_training_sample_globals,
)

POOL_FEATURE_SUFFIX = "_HermesSim_feature.json"
POOL_EMB_SUFFIX = "_HermesSim_embeddings.pkl"

DEFAULT_POOL_DIR = str((ROOT_DIR / "Coding" / "data" / "inline" / "pool").resolve())
DEFAULT_POOL_DIRS = [DEFAULT_POOL_DIR]
DEFAULT_OUTPUT_INDEX = str(
    (ROOT_DIR / "Coding" / "artifacts" / "faiss" / "hermessim_centroids.index").resolve()
)
DEFAULT_GLOBAL_META = str(
    (ROOT_DIR / "Coding" / "artifacts" / "faiss" / "global_ann_centroids.index.meta.json").resolve()
)


def resolve_target_nlist(explicit_nlist: int, global_meta_path: str):
    """Decide the nlist to use, mirroring global-ANN for a controlled comparison.

    Priority: explicit ``--nlist`` > nlist read from the global-ANN meta >
    ``None`` (defer to ``choose_nlist(num_vectors)`` after streaming). Returns
    ``(nlist_or_None, reason_str)``.
    """
    if explicit_nlist and explicit_nlist > 0:
        return int(explicit_nlist), "explicit --nlist"
    if global_meta_path and os.path.exists(global_meta_path):
        try:
            with open(global_meta_path, "r", encoding="utf-8") as f:
                gn = int(json.load(f).get("nlist") or 0)
            if gn > 0:
                return gn, f"matched global nlist ({os.path.basename(global_meta_path)})"
        except Exception as exc:
            print(f"      WARN: could not read global nlist from {global_meta_path}: {exc}", flush=True)
    return None, "auto choose_nlist (global meta unavailable)"


def resolve_pool_binary_inputs(
    pool_dir,
    feature_suffix: str,
    emb_suffix: str,
) -> List[Tuple[str, str, str]]:
    """Discover binaries with a (feature.json, embeddings.pkl) pair.

    Mirrors ``train_faiss_global_ann._resolve_pool_binary_inputs`` but is
    parameterized by suffix (no legacy fallback). ``pool_dir`` accepts a single
    dir or a sequence; on basename collision the first listed root wins. Returns
    sorted ``(binary_path, feature_path, cache_path)`` triples consumable by the
    imported streaming workers (which take explicit paths).
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
            feature_path = f"{binary}{feature_suffix}"
            cache_path = f"{binary}{emb_suffix}"
            if os.path.exists(feature_path) and os.path.exists(cache_path):
                triples.append((binary, feature_path, cache_path))
                seen_basenames.add(basename)
    if collisions:
        print(
            f"[train-faiss-hermessim] skipped {collisions} duplicate-basename "
            f"binaries across pool-dir roots (first-root-wins)",
            flush=True,
        )
    return triples


def main():
    parser = argparse.ArgumentParser(
        description="Train a FAISS index for HermesSim whole-function embeddings (baseline).",
    )
    parser.add_argument(
        "--pool-dir",
        nargs="+",
        default=list(DEFAULT_POOL_DIRS),
        help=f"One or more source pool dirs holding <binary>{POOL_EMB_SUFFIX} and "
             f"<binary>{POOL_FEATURE_SUFFIX} — used for centroid training. "
             f"On basename collision across roots, the first listed root wins. "
             f"Default: {DEFAULT_POOL_DIRS}",
    )
    parser.add_argument("--output-index", default=DEFAULT_OUTPUT_INDEX)
    parser.add_argument("--output-meta", default=None)
    parser.add_argument("--metric", choices=("ip", "l2"), default="ip")
    parser.add_argument("--pca-dim", type=int, default=0, help="Set 0 to disable dimensionality reduction.")
    parser.add_argument(
        "--nlist",
        type=int,
        default=0,
        help="0 = match the global-ANN nlist (see --global-meta) for a controlled "
             "comparison; falls back to auto-select from vector count if that meta "
             "is missing. Pass a positive value to override.",
    )
    parser.add_argument(
        "--global-meta",
        default=DEFAULT_GLOBAL_META,
        help="Global-ANN centroids meta.json whose nlist is matched when --nlist=0. "
             f"Default: {DEFAULT_GLOBAL_META}",
    )
    parser.add_argument(
        "--auto-nlist-cap",
        type=int,
        default=65536,
        help="Upper bound for auto-selected nlist (only used when --nlist=0). "
             "Bounds the streaming reservoir size to ~16 * cap rows.",
    )
    parser.add_argument("--train-size", type=int, default=0, help="0 = auto-select from nlist.")
    parser.add_argument("--niter", type=int, default=25)
    parser.add_argument("--max-points-per-centroid", type=int, default=256)
    parser.add_argument("--add-batch", type=int, default=100_000)
    parser.add_argument(
        "--add-all",
        action="store_true",
        help="Also bake all source-pool vectors into the IVF (emits per-row pool_meta). "
             "Default OFF — centroids-only, matching the global-ANN split.",
    )
    parser.add_argument("--sample-mode", choices=("balanced", "uniform"), default="balanced")
    parser.add_argument("--signature-bits", type=int, default=24)
    parser.add_argument("--per-signature-cap", type=int, default=0)
    parser.add_argument("--per-binary-train-cap", type=int, default=0)
    parser.add_argument("--bucket-topk-diagnostic", type=int, default=8)
    parser.add_argument("--bucket-stats-output", default=None)
    parser.add_argument("--nprobe", type=int, default=8, help="Default nprobe baked into the index (eval can override).")
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Pickle/json loader worker processes. 0 = auto (min(8, cpu_count)).",
    )
    args = parser.parse_args()

    add_all = bool(args.add_all)

    pool_dirs = _normalize_pool_dirs(args.pool_dir)
    print(f"[1/7] discovering binaries under pool={pool_dirs}")
    t0 = time.time()
    binary_inputs = resolve_pool_binary_inputs(pool_dirs, POOL_FEATURE_SUFFIX, POOL_EMB_SUFFIX)
    if not binary_inputs:
        raise ValueError(
            f"no HermesSim embeddings found under pool dirs {pool_dirs} "
            f"(looking for <binary>{POOL_EMB_SUFFIX} + <binary>{POOL_FEATURE_SUFFIX})"
        )
    print(f"      found {len(binary_inputs)} binaries with HermesSim embeddings in {time.time() - t0:.2f}s")

    target_nlist, nlist_src = resolve_target_nlist(args.nlist, args.global_meta)
    print(f"      target nlist = {target_nlist if target_nlist else 'auto-after-stream'} ({nlist_src})")
    nlist_estimate = target_nlist if target_nlist else max(args.auto_nlist_cap, 1)
    if args.train_size > 0:
        streaming_train_size = int(args.train_size)
    else:
        streaming_train_size = max(16 * nlist_estimate, 30_000)

    print(
        f"[2/7] streaming HermesSim embeddings (reservoir target={streaming_train_size}, "
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

    nlist = target_nlist or choose_nlist(num_vectors)
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
        "method": "HermesSim",
        "pool_dir": [os.path.abspath(p) for p in pool_dirs],
        "feature_suffix": POOL_FEATURE_SUFFIX,
        "embedding_suffix": POOL_EMB_SUFFIX,
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
