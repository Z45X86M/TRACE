"""Efficiency micro-benchmark for the single-vector global ANN baseline (DEJINA).

Does NOT call global_ann_inline_eval.main() and uses NO multiprocessing / batching:
each query is searched on its own in a timed loop, mirroring how the slice pipeline
measures per-query latency, so the two latency columns are directly comparable.

The reported ANN recall (artifacts/model_eval/partial_ann_*) was produced at
nprobe = nlist (131072) on an IVFFlat index, i.e. an *exhaustive* probe = exact
search. We therefore time the recall-identical operation — exact flat-IP search
over the same pool vectors (reconstructed from the index). Flat-exact is also the
fastest correct single-vector retrieval, so this is the most generous fair baseline
(no one can claim the ANN was crippled by a bad nlist to make Ours look good).

Recall is read from the existing *_ann_* metrics files, not recomputed here.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from pool_inline_search import load_query_pairs  # noqa: E402


def _normalize(emb):
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    norms = np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    return (emb / norms).astype(np.float32)


def _pct(xs, p):
    if not xs:
        return 0.0
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p))


def main():
    ap = argparse.ArgumentParser(description="Global-ANN single-vector latency micro-benchmark (per-query, no batching).")
    ap.add_argument("--index", required=True, help="e.g. artifacts/faiss/global_partial_pool_10K.index")
    ap.add_argument("--query-dir", required=True)
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--topk", type=int, default=1000, help="match slice CAP-style depth")
    ap.add_argument("--limit-total", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    index = faiss.read_index(args.index)
    ntotal = int(index.ntotal)
    dim = int(index.d)
    ivf = faiss.extract_index_ivf(index)
    ivf.make_direct_map()
    pool = index.reconstruct_n(0, ntotal)
    pool = _normalize(pool)

    flat = faiss.IndexFlatIP(dim)
    flat.add(np.ascontiguousarray(pool))
    on_gpu = args.device == "gpu"
    if on_gpu:
        res = faiss.StandardGpuResources()
        flat = faiss.index_cpu_to_gpu(res, int(args.gpu_id), flat)

    pairs = load_query_pairs(args.query_dir, limit_total=args.limit_total)
    Q = _normalize(np.asarray([np.asarray(gq, dtype=np.float32).reshape(-1)
                               for _q, gq, *_rest in pairs], dtype=np.float32))
    topk = min(args.topk, ntotal)
    print(f"[bench] method=global_ann device={args.device} ntotal={ntotal} "
          f"queries={Q.shape[0]} topk={topk} index={os.path.basename(args.index)}", flush=True)

    for i in range(min(args.warmup, Q.shape[0])):
        flat.search(np.ascontiguousarray(Q[i:i + 1]), topk)

    tot_ms = []
    for i in range(Q.shape[0]):
        t0 = time.perf_counter()
        flat.search(np.ascontiguousarray(Q[i:i + 1]), topk)
        tot_ms.append((time.perf_counter() - t0) * 1000.0)

    res_obj = {
        "method": "global_ann",
        "device": args.device,
        "pool_size": ntotal,
        "topk": int(topk),
        "queries": len(tot_ms),
        "total_ms": {"mean": float(np.mean(tot_ms)), "p50": _pct(tot_ms, 50), "p95": _pct(tot_ms, 95)},
        "search_mode": "exact_flat_ip (== reported nprobe=nlist exhaustive)",
        "index": os.path.basename(args.index),
    }
    print(json.dumps(res_obj, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res_obj, indent=2), encoding="utf-8")
        print(f"[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
