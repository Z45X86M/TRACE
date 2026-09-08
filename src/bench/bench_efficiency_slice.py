"""Efficiency micro-benchmark for the slice multi-vector pipeline (Ours).

Reuses the *already materialized* eval pool runtime (model_eval/<run>) — does NOT
rebuild the pool and does NOT call pool_inline_search.main(). It reconstructs the
EmbeddingRetrievalDB exactly as main() does, then replays each query through the
public stage methods with a Stage A / Stage B latency split.

Stage A = _build_query_profile + _build_query_slice_sets + _score_stage_a
          (candidate generation: query slice -> FAISS bucket -> postings -> aggregate -> shortlist)
Stage B = _dense_rerank (MaxSim-style dense rerank + base/slice/global fusion)

--device gpu : FAISS coarse quantizer on GPU + rerank tensors on cuda:<gpu-id>
--device cpu : forces the CPU quantizer AND cpu rerank (a clean accelerator-free number).

Recall is NOT recomputed here — read it from the run's shortlist_eval_metrics.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from pool_inline_search import EmbeddingRetrievalDB, load_query_pairs  # noqa: E402
from utils import derive_embedding_suffix      # noqa: E402


def _pct(xs, p):
    if not xs:
        return 0.0
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p))


def main():
    ap = argparse.ArgumentParser(description="Slice-pipeline latency micro-benchmark (Stage A / Stage B split).")
    ap.add_argument("--pool-dir", required=True, help="materialized pool dir, e.g. artifacts/tmp/pool_partial_10K")
    ap.add_argument("--save-db", required=True, help="run dir holding cached runtime npy, e.g. artifacts/model_eval/partial_10K")
    ap.add_argument("--query-dir", required=True)
    ap.add_argument("--model", default=os.path.join(CURRENT_DIR, "artifacts/tmp/inline_slice_pool.index"))
    ap.add_argument("--pool-size", type=int, required=True, help="drives recommended_cap(N)")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--feature-suffix", default="_bb_slice_feature.json")
    ap.add_argument("--embedding-suffix", default=None)
    ap.add_argument("--limit-total", type=int, default=None, help="subsample queries (None = all)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    emb_suffix = args.embedding_suffix or derive_embedding_suffix(args.feature_suffix)
    rerank_device = f"cuda:{args.gpu_id}" if args.device == "gpu" else "cpu"

    db = EmbeddingRetrievalDB(
        args.pool_dir, args.model, args.save_db,
        feature_suffix=args.feature_suffix, embedding_suffix=emb_suffix,
        faiss_gpu_id=args.gpu_id,
    )
    db.CAP = int(args.cap) if args.cap is not None else EmbeddingRetrievalDB.recommended_cap(int(args.pool_size))
    db.RERANK_DEVICE = rerank_device
    db.build_pool()

    if args.device == "cpu":
        db.gpu_quantizer = db.ivf.quantizer
        db.use_gpu_quantizer = False
        db.res = None

    on_gpu = args.device == "gpu"
    if on_gpu:
        import torch
    db._ensure_gpu_state()

    pairs = load_query_pairs(args.query_dir, limit_total=args.limit_total)
    print(f"[bench] device={args.device} CAP={db.CAP} pool_size={args.pool_size} "
          f"queries={len(pairs)} model={os.path.basename(args.model)}", flush=True)

    def run_one(q, gq, row_meta):
        if on_gpu:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        profile = db._build_query_profile(q, row_meta, global_query=gq)
        cand, agg, _dbg = db._build_query_slice_sets(profile)
        stage_a = db._score_stage_a(cand, agg, profile)
        if on_gpu:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        _ranked = db._dense_rerank(stage_a, profile)
        if on_gpu:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        return (t1 - t0), (t2 - t1)

    for q, gq, _t, _gb, _ga, rm in pairs[: max(0, args.warmup)]:
        run_one(q, gq, rm)

    a_ms, b_ms, tot_ms = [], [], []
    for q, gq, _t, _gb, _ga, rm in pairs:
        da, dbt = run_one(q, gq, rm)
        a_ms.append(da * 1000.0)
        b_ms.append(dbt * 1000.0)
        tot_ms.append((da + dbt) * 1000.0)

    res = {
        "method": "slice_multivec",
        "device": args.device,
        "pool_size": int(args.pool_size),
        "cap": int(db.CAP),
        "btopk": int(db.BTOPK),
        "queries": len(tot_ms),
        "stage_a_ms": {"mean": float(np.mean(a_ms)), "p50": _pct(a_ms, 50), "p95": _pct(a_ms, 95)},
        "stage_b_ms": {"mean": float(np.mean(b_ms)), "p50": _pct(b_ms, 50), "p95": _pct(b_ms, 95)},
        "total_ms":   {"mean": float(np.mean(tot_ms)), "p50": _pct(tot_ms, 50), "p95": _pct(tot_ms, 95)},
        "model_index": os.path.basename(args.model),
    }
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
