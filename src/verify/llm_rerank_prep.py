"""Prep step for the optional LLM verifier experiment.

Decoupled from the (slow) LLM scoring so we run db.search ONCE and can re-run /
re-prompt the LLM offline. For a representative stride-sample of queries we dump,
per query's top-K shortlist:
  qt, score[K], is_gt[K], q_text, cand_texts[K]
where *_text is the full decompiled pseudocode of the function (query side from
strip, candidate side from the materialized pool). FULL text — no truncation.

The query text = join of all blocks_pseudocode of the func at query_addr (we
verified len(q) == n_bbs, i.e. the query IS that whole strip function).

Usage:
  PYTHONPATH=.. python3 llm_rerank_prep.py \
    --query-dir artifacts/tmp/query_partial \
    --pool-dir  artifacts/tmp/pool_partial_100K \
    --save-db   artifacts/model_eval/partial_100K \
    --pool-size 100000 --topk 50 --max-queries 450 \
    --out artifacts/diagnostics/llm_rerank_prep_partial_100K.pkl
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
for p in (str(ROOT_DIR), str(CURRENT_DIR)):
    if p not in sys.path:
        sys.path.append(p)

from pool_inline_search import (
    EmbeddingRetrievalDB, _build_pool_name_index, prepare_eval_pool_dir, build_arg_parser,
)
from utils import derive_embedding_suffix
from utils import read_pickle, write_pickle


def _norm_addr(a):
    if isinstance(a, int):
        return a
    s = str(a).strip()
    try:
        return int(s, 16) if s.lower().startswith("0x") else int(s)
    except ValueError:
        try:
            return int(s, 16)
        except ValueError:
            return s


def func_text(func_obj):
    """Full decompiled text — NO truncation (API context dwarfs our functions)."""
    parts = [(b.get("pseudos") or "") for b in func_obj.get("blocks_pseudocode", [])]
    return "\n".join(p for p in parts if p)


def build_text_map(feat_dir, suffix="_bb_slice_feature.json"):
    amap = {}
    files = [f for f in sorted(os.listdir(feat_dir)) if f.endswith(suffix)]
    for f in tqdm(files, desc="cand text map", dynamic_ncols=True):
        stem = f[: -len(suffix)]
        try:
            d = json.load(open(os.path.join(feat_dir, f)))
        except Exception:
            continue
        for k, v in d.items():
            amap[(stem, _norm_addr(v.get("func_addr", k)))] = func_text(v)
    return amap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-dir", default="artifacts/tmp/query_partial")
    ap.add_argument("--pool-dir", default="artifacts/tmp/pool_partial_100K")
    ap.add_argument("--save-db", default="artifacts/model_eval/partial_100K")
    ap.add_argument("--pool-size", type=int, default=100000)
    ap.add_argument("--model", default=str(CURRENT_DIR / "artifacts" / "tmp" / "inline_slice_pool.index"))
    ap.add_argument("--rerank-device", default="cuda:1")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--max-queries", type=int, default=450)
    ap.add_argument("--strip-feat-dir", default="data/inline/strip")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = build_arg_parser().parse_args([
        "--query-dir", args.query_dir, "--pool-dir", args.pool_dir,
        "--save-db", args.save_db, "--pool-size", str(args.pool_size),
        "--model", args.model, "--rerank-device", args.rerank_device,
    ])
    base.embedding_suffix = derive_embedding_suffix(base.feature_suffix)
    built = prepare_eval_pool_dir(base)
    faiss_gpu_id = int(args.rerank_device.split(":", 1)[1]) if args.rerank_device.startswith("cuda:") else 0
    db = EmbeddingRetrievalDB(built, base.model, base.save_db,
                              feature_suffix=base.feature_suffix,
                              embedding_suffix=base.embedding_suffix,
                              faiss_gpu_id=faiss_gpu_id)
    db.RERANK_DEVICE = str(args.rerank_device)
    db.CAP = (int(base.cap) if getattr(base, "cap", None) is not None
              else EmbeddingRetrievalDB.recommended_cap(int(args.pool_size)))
    if getattr(base, "btopk", None) is not None:
        db.BTOPK = int(base.btopk)
    db.VERIFIER_ENABLED = False
    print("[prep] CAP=%d BTOPK=%d (pool_size=%d)" % (db.CAP, db.BTOPK, args.pool_size), flush=True)
    db.build_pool()
    db._ensure_gpu_state()
    name_to_keys, keys_to_name = _build_pool_name_index(built)

    print("[prep] building candidate text map ...", flush=True)
    cmap = build_text_map(built)
    print("[prep] cand funcs = %d" % len(cmap), flush=True)

    qcache = {}

    def qtext(binary, addr):
        if binary not in qcache:
            p = os.path.join(args.strip_feat_dir, binary + "_bb_slice_feature.json")
            m = {}
            if os.path.exists(p):
                try:
                    for k, v in json.load(open(p)).items():
                        m[_norm_addr(v.get("func_addr", k))] = func_text(v)
                except Exception:
                    pass
            qcache[binary] = m
        return qcache[binary].get(_norm_addr(addr), "")

    allq = []
    for name in sorted(os.listdir(args.query_dir)):
        if not name.endswith(".pkl"):
            continue
        for r in read_pickle(os.path.join(args.query_dir, name)):
            allq.append(r)
    stride = max(1, len(allq) // args.max_queries)
    sample = allq[::stride][: args.max_queries]
    print("[prep] total q=%d  stride=%d  sampled=%d" % (len(allq), stride, len(sample)), flush=True)

    K = int(args.topk)
    records = []
    t0 = time.time()
    for r in tqdm(sample, desc="search", dynamic_ncols=True):
        q = np.asarray(r["q"], dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1) if q.size else q.reshape(0, 0)
        if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] == 0:
            continue
        t = int(r.get("type", 2))
        gq = np.asarray(r["gq"], dtype=np.float32)
        qrm = r.get("query_row_meta") or []
        qt_text = qtext(r["binary"], r["query_addr"])
        strict = (r["gt_binary"], r["addr"])
        hit = {strict}
        nm = keys_to_name.get(strict, "")
        if nm:
            hit |= name_to_keys.get(nm, set())
        res = db.search(q, global_query=gq, query_row_meta=qrm)
        sl = res["recall_results"][:K]
        n = len(sl)
        score = np.empty(n, dtype=np.float32)
        is_gt = np.zeros(n, dtype=bool)
        ctexts = []
        cand_bin = []
        cand_name = []
        for i, it in enumerate(sl):
            score[i] = float(it["score"])
            key = (it["binary_name"], it["func_addr"])
            is_gt[i] = key in hit
            ctexts.append(cmap.get((it["binary_name"], _norm_addr(it["func_addr"])), ""))
            cand_bin.append(it["binary_name"])
            cand_name.append(keys_to_name.get(key, ""))
        records.append({"qt": t, "score": score, "is_gt": is_gt,
                        "q_text": qt_text, "cand_texts": ctexts,
                        "q_name": r.get("query_func_name", ""),
                        "cand_bin": cand_bin, "cand_name": cand_name,
                        "q_binary": r["binary"], "gt_binary": r["gt_binary"],
                        "target_name": nm})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    write_pickle({"records": records, "topk": K}, args.out)
    from collections import Counter
    ct = Counter(r["qt"] for r in records)
    have_gt = sum(1 for r in records if r["is_gt"].any())
    print("[done] %d records (type dist=%s, GT-in-topK=%d) -> %s  (%.1fs)"
          % (len(records), dict(ct), have_gt, args.out, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
