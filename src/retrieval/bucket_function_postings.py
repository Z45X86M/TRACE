import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
for path in (str(ROOT_DIR), str(CURRENT_DIR)):
    if path not in sys.path:
        sys.path.append(path)


def support_saturation(value: float, scale: float = 0.85) -> float:
    if value <= 0.0:
        return 0.0
    return 1.0 - math.exp(-scale * value)


def build_bucket_function_postings(
    bucket_hits_per_func: List[List[List[Tuple[int, float]]]],
    slice_priors_per_func: List[List[float]],
    slice_kinds_per_func: List[List[str]],
) -> Dict[int, Dict[str, List[float]]]:
    bucket_func_stats: Dict[int, Dict[int, List[float]]] = {}
    for func_id, (bucket_hits, slice_priors, slice_kinds) in enumerate(
        zip(bucket_hits_per_func, slice_priors_per_func, slice_kinds_per_func)
    ):
        for row_hits, slice_prior, slice_kind in zip(bucket_hits, slice_priors, slice_kinds):
            slice_prior = float(slice_prior)
            if slice_prior <= 0.0:
                continue
            slice_kind = str(slice_kind or "unknown")
            for bucket_id, bucket_affinity in row_hits:
                bucket_id = int(bucket_id)
                contribution = slice_prior * float(bucket_affinity)
                func_stats = bucket_func_stats.setdefault(bucket_id, {}).setdefault(
                    func_id,
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                )
                func_stats[0] = max(func_stats[0], contribution)
                func_stats[1] += contribution
                if slice_kind == "var_chain":
                    func_stats[2] = max(func_stats[2], contribution)
                    func_stats[3] += contribution
                elif slice_kind == "coverage_chain":
                    func_stats[4] = max(func_stats[4], contribution)
                    func_stats[5] += contribution

    posting: Dict[int, Dict[str, List[float]]] = {}
    for bucket_id, func_stats in bucket_func_stats.items():
        ranked = sorted(
            func_stats.items(),
            key=lambda item: (
                -(0.70 * float(item[1][0]) + 0.30 * support_saturation(float(item[1][1]))),
                -float(item[1][0]),
                -float(item[1][1]),
                item[0],
            ),
        )
        posting[bucket_id] = {
            "func_ids": [int(func_id) for func_id, _stats in ranked],
            "best": [float(stats[0]) for _func_id, stats in ranked],
            "support": [float(stats[1]) for _func_id, stats in ranked],
            "var_best": [float(stats[2]) for _func_id, stats in ranked],
            "var_support": [float(stats[3]) for _func_id, stats in ranked],
            "cov_best": [float(stats[4]) for _func_id, stats in ranked],
            "cov_support": [float(stats[5]) for _func_id, stats in ranked],
        }
    return posting


def summarize_bucket_hotspots(
    posting: Dict[int, Dict[str, List[float]]],
    total_funcs: int,
    top_n: int = 20,
) -> Dict[str, object]:
    if posting:
        bucket_count = int(max(posting.keys())) + 1
    else:
        bucket_count = 0
    counts = np.zeros(bucket_count, dtype=np.int32)
    for bucket_id, item in posting.items():
        counts[int(bucket_id)] = int(len(item.get("func_ids", [])))
    nonempty = counts[counts > 0]

    top_hot_buckets = []
    for bucket_id, func_count in sorted(
        ((int(bucket_id), int(counts[int(bucket_id)])) for bucket_id in posting.keys()),
        key=lambda item: (-item[1], item[0]),
    )[:top_n]:
        top_hot_buckets.append(
            {
                "bucket_id": int(bucket_id),
                "func_count": int(func_count),
                "df_ratio": float(func_count / max(1, total_funcs)),
            }
        )

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
        "hot_bucket_count_ge_5": int(np.sum(counts >= 5)),
        "hot_bucket_count_ge_10": int(np.sum(counts >= 10)),
        "hot_bucket_count_ge_50": int(np.sum(counts >= 50)),
        "hot_bucket_count_ge_100": int(np.sum(counts >= 100)),
        "hot_bucket_count_ge_500": int(np.sum(counts >= 500)),
        "top_hot_buckets": top_hot_buckets,
    }


def analyze_pool_bucket_hotspots(pool_dir: str, model: str, top_n: int = 20) -> Dict[str, object]:
    from pool_inline_storage import InlinePool

    with tempfile.TemporaryDirectory() as tmp_dir:
        runtime = InlinePool(pool_dir, model, tmp_dir)
        bucket_hits_per_func: List[List[List[Tuple[int, float]]]] = []
        slice_priors_per_func: List[List[float]] = []
        slice_kinds_per_func: List[List[str]] = []

        for _binary_name, cleaned, feature_map in runtime._iter_pool_binaries():
            for (addr, func_name), (_emb_g, partial) in sorted(cleaned.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
                row_hits = runtime.assign_bucket_hits(partial, topk=runtime.pool_bucket_topk)
                blocks = feature_map.get((addr, func_name), {}).get("blocks_pseudocode", [])
                priors, kinds = [], []
                for idx in range(int(partial.shape[0])):
                    b = blocks[idx] if idx < len(blocks) else {}
                    priors.append(runtime._coarse_meta_weight(b))
                    kinds.append(str(b.get("slice_kind", "unknown")))
                bucket_hits_per_func.append(row_hits)
                slice_priors_per_func.append(priors)
                slice_kinds_per_func.append(kinds)

        posting = build_bucket_function_postings(
            bucket_hits_per_func=bucket_hits_per_func,
            slice_priors_per_func=slice_priors_per_func,
            slice_kinds_per_func=slice_kinds_per_func,
        )
        summary = summarize_bucket_hotspots(posting, total_funcs=len(bucket_hits_per_func), top_n=top_n)
        summary["total_functions"] = int(len(bucket_hits_per_func))
        summary["pool_dir"] = os.path.abspath(pool_dir)
        summary["model"] = os.path.abspath(model)
        summary["pool_bucket_topk"] = int(runtime.pool_bucket_topk)
        return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze bucket->func hotspot distribution for a built pool.")
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    summary = analyze_pool_bucket_hotspots(
        pool_dir=args.pool_dir,
        model=args.model,
        top_n=args.top_n,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
