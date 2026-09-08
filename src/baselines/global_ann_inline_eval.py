import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
for path in (str(ROOT_DIR), str(CURRENT_DIR)):
    if path not in sys.path:
        sys.path.append(path)

from utils import discover_binaries
from pool_inline_search import load_query_pairs
from utils import read_json, read_pickle, write_pickle, type_name


def _normalize_rows(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    if emb.size == 0:
        return emb.reshape(0, emb.shape[-1] if emb.ndim == 2 else 0)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (emb / norms).astype(np.float32)


def _load_pool_globals(pool_dir: str,
                       feature_suffix: str = "_bb_slice_feature.json",
                       embedding_suffix: str = "_bb_slice_embeddings.pkl",
                       ) -> Tuple[np.ndarray, List[Tuple[str, Any, str]]]:
    globals_list: List[np.ndarray] = []
    meta: List[Tuple[str, Any, str]] = []
    for binary in discover_binaries(pool_dir):
        binary_name = os.path.basename(binary)
        feature_path = f"{binary}{feature_suffix}"
        embedding_path = f"{binary}{embedding_suffix}"
        if not os.path.exists(feature_path) or not os.path.exists(embedding_path):
            continue
        feature_info = read_json(feature_path)
        embedding_info = read_pickle(embedding_path)
        for addr, item in feature_info.items():
            if addr not in embedding_info:
                continue
            func_name = item["func_name"]
            global_emb = np.asarray(embedding_info[addr]["global"], dtype=np.float32).reshape(-1)
            if global_emb.size == 0:
                continue
            globals_list.append(global_emb)
            meta.append((binary_name, addr, func_name))
    if not globals_list:
        return np.empty((0, 0), dtype=np.float32), []
    matrix = _normalize_rows(np.asarray(globals_list, dtype=np.float32))
    return matrix, meta


def _build_faiss_index(pool_globals: np.ndarray) -> faiss.Index:
    dim = int(pool_globals.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(np.asarray(pool_globals, dtype=np.float32))
    return index


def _extract_ivf(index: faiss.Index) -> Optional[faiss.Index]:
    """Return the inner IVF index inside a potentially wrapped index
    (IndexPreTransform / IndexRefine / etc). Uses faiss.downcast_index so the
    Python wrapper actually exposes the IVF-level attributes."""
    try:
        ivf = faiss.extract_index_ivf(index)
    except Exception:
        ivf = None
    if ivf is not None:
        return ivf
    target = index
    visited = set()
    while target is not None and id(target) not in visited:
        visited.add(id(target))
        try:
            down = faiss.downcast_index(target)
        except Exception:
            down = target
        if hasattr(down, "nprobe") and not isinstance(down, faiss.IndexPreTransform):
            return down
        inner = getattr(down, "index", None)
        if inner is None or inner is down:
            return None
        target = inner
    return None


def _set_index_nprobe(index: faiss.Index, nprobe: int) -> int:
    """Apply nprobe to any IVF inside the index. Returns the nprobe actually
    applied, or 0 if the index has no nprobe concept (pure flat)."""
    if nprobe <= 0:
        return 0
    try:
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", int(nprobe))
    except Exception:
        pass
    ivf = _extract_ivf(index)
    if ivf is not None and hasattr(ivf, "nprobe"):
        ivf.nprobe = int(nprobe)
        return int(ivf.nprobe)
    return 0


def _load_pretrained_index(
    model_path: str,
    meta_path: Optional[str],
    nprobe: int,
) -> Tuple[faiss.Index, List[Tuple[str, Any, str]], Dict[str, Any]]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"faiss index not found: {model_path}")
    index = faiss.read_index(model_path)
    resolved_meta = meta_path or f"{model_path}.meta.json"
    if not os.path.exists(resolved_meta):
        raise FileNotFoundError(
            f"meta file not found: {resolved_meta} (run train_faiss_global_ann.py to produce it)"
        )
    meta_payload = read_json(resolved_meta)
    rows = meta_payload.get("pool_meta") or []
    pool_meta: List[Tuple[str, Any, str]] = [
        (row["binary_name"], row["func_addr"], row["func_name"]) for row in rows
    ]
    if not pool_meta:
        raise RuntimeError(f"meta file {resolved_meta} has empty pool_meta")
    if index.ntotal != len(pool_meta):
        raise RuntimeError(
            f"index.ntotal ({index.ntotal}) does not match meta rows ({len(pool_meta)}). "
            "Re-run train_faiss_global_ann.py with add_all=True, or update the meta file."
        )
    applied_nprobe = _set_index_nprobe(index, nprobe)
    return index, pool_meta, {
        "meta_path": resolved_meta,
        "requested_nprobe": int(nprobe),
        "applied_nprobe": int(applied_nprobe),
        "nlist": int(meta_payload.get("nlist") or 0),
        "metric": meta_payload.get("metric"),
        "pca_dim": int(meta_payload.get("pca_dim") or 0),
    }


def _evaluate(
    index: faiss.Index,
    pool_meta: List[Tuple[str, Any, str]],
    query_pairs,
    ks: List[int],
    search_batch: int = 512,
    include_details: bool = True,
    collect_rankings: bool = False,
) -> Dict[str, Any]:
    """Batched evaluation: stack all query globals, run one `index.search` per
    batch instead of per query. For a 10k pool on IP this cuts eval time
    from O(Nq * call overhead) to O(Nq / batch * overhead + one GEMM per batch);
    on a 1M-pool IVF this is the difference between minutes and seconds.
    """
    max_k = max(ks)
    if not query_pairs:
        return {"queries": 0}

    query_vecs: List[np.ndarray] = []
    gt_types: List[int] = []
    gt_binaries: List[Any] = []
    gt_addrs: List[Any] = []
    for q, gq, gt_type, gt_binary, gt_addr, query_row_meta in query_pairs:
        del q, query_row_meta
        query_vecs.append(np.asarray(gq, dtype=np.float32).reshape(-1))
        gt_types.append(int(gt_type))
        gt_binaries.append(gt_binary)
        gt_addrs.append(gt_addr)

    Q = _normalize_rows(np.asarray(query_vecs, dtype=np.float32))
    top_k = min(max_k, len(pool_meta))

    ids_full = np.empty((Q.shape[0], top_k), dtype=np.int64)
    sims_full = np.empty((Q.shape[0], top_k), dtype=np.float32)
    for start in range(0, Q.shape[0], search_batch):
        end = min(Q.shape[0], start + search_batch)
        sims_batch, ids_batch = index.search(np.ascontiguousarray(Q[start:end]), top_k)
        sims_full[start:end] = sims_batch
        ids_full[start:end] = ids_batch

    name_to_keys: Dict[str, set] = {}
    keys_to_name: Dict[Tuple[str, str], str] = {}
    for _b, _a, _fn in pool_meta:
        if not _fn:
            continue
        _key = (str(_b), str(_a))
        keys_to_name[_key] = str(_fn)
        name_to_keys.setdefault(str(_fn), set()).add(_key)

    recalls = {k: 0 for k in ks}
    type_recalls: Dict[int, Dict[int, int]] = {0: {k: 0 for k in ks}, 1: {k: 0 for k in ks}, 2: {k: 0 for k in ks}}
    type_counts = {0: 0, 1: 0, 2: 0}
    details: List[Dict[str, Any]] = []
    rankings: Dict[Tuple[Any, Any, Any, Any], List[Tuple[Any, Any, float, int]]] = {}

    for i in range(Q.shape[0]):
        gt_type = gt_types[i]
        type_counts[gt_type] = type_counts.get(gt_type, 0) + 1
        row_ids = ids_full[i]
        row_sims = sims_full[i]
        gt_binary = gt_binaries[i]
        gt_addr = gt_addrs[i]

        gt_rank: Optional[int] = None
        strict_key = (str(gt_binary), str(gt_addr))
        hit_keys = {strict_key}
        gt_name = keys_to_name.get(strict_key)
        if gt_name:
            hit_keys |= name_to_keys.get(gt_name, set())
        for rank_idx in range(row_ids.shape[0]):
            fid = int(row_ids[rank_idx])
            if fid < 0:
                continue
            binary_name, func_addr, _fn = pool_meta[fid]
            if (str(binary_name), str(func_addr)) in hit_keys:
                gt_rank = rank_idx + 1
                break

        for k in ks:
            if gt_rank is not None and gt_rank <= k:
                recalls[k] += 1
                type_recalls[gt_type][k] += 1

        if include_details:
            ranked = []
            for rank_idx in range(min(10, row_ids.shape[0])):
                fid = int(row_ids[rank_idx])
                if fid < 0:
                    continue
                binary_name, func_addr, func_name = pool_meta[fid]
                ranked.append(
                    {
                        "rank": int(rank_idx + 1),
                        "binary_name": binary_name,
                        "func_addr": func_addr,
                        "func_name": func_name,
                        "score": float(row_sims[rank_idx]),
                    }
                )
            details.append(
                {
                    "gt_type": type_name(gt_type),
                    "gt_binary": gt_binary,
                    "gt_func_addr": gt_addr,
                    "gt_rank": gt_rank,
                    "top10": ranked,
                }
            )

        if collect_rankings:
            full_ranked: List[Tuple[Any, Any, float, int]] = []
            for rank_idx in range(row_ids.shape[0]):
                fid = int(row_ids[rank_idx])
                if fid < 0:
                    continue
                binary_name, func_addr, _func_name = pool_meta[fid]
                full_ranked.append(
                    (binary_name, func_addr, float(row_sims[rank_idx]), int(rank_idx + 1))
                )
            key = (i, gt_binary, gt_addr)
            rankings[key] = full_ranked

    total = Q.shape[0]
    metrics: Dict[str, Any] = {"queries": int(total)}
    for k in ks:
        metrics[f"ann@{k}"] = recalls[k] / total if total else 0.0
    for t in sorted(type_counts):
        metrics[f"{type_name(t)}_queries"] = type_counts[t]
        for k in ks:
            metrics[f"{type_name(t)}_ann@{k}"] = (
                type_recalls[t][k] / type_counts[t] if type_counts[t] else 0.0
            )
    if include_details:
        metrics["details"] = details
    if collect_rankings:
        metrics["rankings"] = rankings
    return metrics


DEFAULT_POOL_DIR = (CURRENT_DIR / "artifacts" / "tmp" / "pool_global_ann").resolve()
DEFAULT_QUERY_DIR = (CURRENT_DIR / "artifacts" / "tmp" / "query_global_ann").resolve()
DEFAULT_OUTPUT = (CURRENT_DIR / "artifacts" / "tmp" / "eval" / "global_ann_metrics.json").resolve()
DEFAULT_MODEL_INDEX = (CURRENT_DIR / "artifacts" / "faiss" / "global_ann_pool.index").resolve()
DEFAULT_TOP_K = 1000
_FULL_KS = (1, 5, 10, 25, 50, 100, 200, 500, 1000)


def _resolve_ks(top_k: int) -> List[int]:
    """Keep every canonical cutoff ≤ top_k, and always include top_k itself
    so the caller can read ann@top_k directly."""
    ks = [k for k in _FULL_KS if k <= top_k]
    if top_k not in ks:
        ks.append(int(top_k))
    return sorted(set(ks))


def main():
    parser = argparse.ArgumentParser(description="Evaluate simple global-vector ANN retrieval baseline.")
    parser.add_argument("--pool-dir", default=str(DEFAULT_POOL_DIR))
    parser.add_argument("--query-dir", default=str(DEFAULT_QUERY_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit-per-file", type=int, default=None)
    parser.add_argument("--limit-total", type=int, default=None)
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_INDEX),
        help="Path to the populated FAISS DB produced by "
             "build_global_ann_artifacts.py (step 2). Loads the index + its "
             ".meta.json sibling. Pass empty string to fall back to building a "
             "flat index on-the-fly from --pool-dir (slower, debug only).",
    )
    parser.add_argument(
        "--meta",
        default=None,
        help="Override meta.json path (defaults to '<--model>.meta.json').",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=131072,
        help="Override nprobe on the pre-trained IVF index (0 = keep whatever "
             "the index was trained with).",
    )
    parser.add_argument(
        "--search-batch",
        type=int,
        default=512,
        help="Queries per index.search call. Larger = faster on GPU/BLAS but more memory.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Max retrieval depth — kept at {DEFAULT_TOP_K} to match the slice "
             f"pipeline's CAP. FAISS searches top-K per query and ann@K "
             f"is reported for every canonical cutoff ≤ top-K. Default: {DEFAULT_TOP_K}.",
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Skip writing per-query details (top10 + gt_rank). Use for large-scale runs.",
    )
    parser.add_argument(
        "--emit-rankings",
        default=None,
        help="If set, also pickle the full per-query top-K rankings to this path. "
             "Consumed by pool_inline_eval.py --enable-rrf as the global channel input.",
    )
    parser.add_argument(
        "--feature-suffix",
        default="_bb_slice_feature.json",
        help="Filename suffix of the pool-side feature.json when --model is not "
             "given and the script builds a flat index from --pool-dir "
             "(default: _bb_slice_feature.json, the BB-bounded F output). Pass "
             "_inline_slice_feature.json for the legacy varchain output.",
    )
    parser.add_argument(
        "--embedding-suffix",
        default=None,
        help="Filename suffix of the pool-side slice embedding pkl. If omitted, "
             "derived from --feature-suffix by replacing trailing _feature.json "
             "with _embeddings.pkl.",
    )
    args = parser.parse_args()
    if args.embedding_suffix is None:
        from utils import derive_embedding_suffix
        args.embedding_suffix = derive_embedding_suffix(args.feature_suffix)

    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    ks = _resolve_ks(args.top_k)
    index_meta: Dict[str, Any] = {"mode": "flat_from_pool_dir"}
    if args.model:
        index, pool_meta, ann_info = _load_pretrained_index(args.model, args.meta, args.nprobe)
        index_meta = {"mode": "pretrained_faiss", "model_path": os.path.abspath(args.model), **ann_info}
        print(
            f"[index] loaded {args.model} ntotal={index.ntotal} "
            f"nprobe_applied={ann_info['applied_nprobe']} nlist={ann_info['nlist']}",
            flush=True,
        )
    else:
        pool_globals, pool_meta = _load_pool_globals(
            args.pool_dir,
            feature_suffix=args.feature_suffix,
            embedding_suffix=args.embedding_suffix,
        )
        if pool_globals.size == 0 or not pool_meta:
            raise RuntimeError("pool globals are empty")
        index = _build_faiss_index(pool_globals)
        index_meta["pool_dir"] = os.path.abspath(args.pool_dir)
        index_meta["pool_size"] = int(len(pool_meta))
    query_pairs = load_query_pairs(args.query_dir, limit_per_file=args.limit_per_file, limit_total=args.limit_total)
    include_details = not args.no_details
    collect_rankings = args.emit_rankings is not None
    metrics = _evaluate(
        index,
        pool_meta,
        query_pairs,
        ks,
        search_batch=args.search_batch,
        include_details=include_details,
        collect_rankings=collect_rankings,
    )
    metrics["index"] = index_meta

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    details = metrics.pop("details", None)
    rankings = metrics.pop("rankings", None)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if details is not None:
        output_path.with_suffix(".details.json").write_text(
            json.dumps(details, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if rankings is not None:
        rankings_path = Path(args.emit_rankings)
        rankings_path.parent.mkdir(parents=True, exist_ok=True)
        write_pickle(rankings, str(rankings_path))
        print(f"[rankings] wrote {len(rankings)} query-record rankings to {rankings_path}")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
