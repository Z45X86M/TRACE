"""Ablation driver for slice-level BCSD retrieval.

Mirrors ``pool_inline_search.py`` *exactly* on the CLI side — same args, same
pool prep, same FAISS index, same encoder, same query loading — but instead of
emitting one ``recall_eval_metrics.json`` it runs THREE ablation studies on
the SAME run and prints all of them:

  A1  Stage-A candidate generation     IVF (slice->bucket->func) vs
                                        no-IVF (slice->slice ANN). Validates the
                                        bucket-quantization constraint itself.
  A2  Rerank channel leave-one-out      base / slice / global fusion channels.
  A3  Slice-similarity direction        slice_channel formula: q2f / f2q /
                                        symmetric / max / current rectifier.

------------------------------------------------------------------------------
A1 (Stage A — candidate generation mechanism)
------------------------------------------------------------------------------
Two variants, BOTH stopped at Stage A (no Stage B rerank). The headline metric
is ``recall@1000`` == whether the GT made it into the 1000-candidate set
(the recall ceiling of the candidate-generation mechanism); ``recall@{1..}``
shows how the base ranking places the GT inside that set.

  ours   : IVF  slice -> top-K bucket -> bucket->func posting -> aggregate
           -> top-1000 by Stage-A slice_score. Slices are quantized to coarse
           buckets and candidates are pulled via the bucket->func inverted
           index (slice maps to FUNCTIONS through a quantization layer).
  no-IVF : slice -> slice exact ANN. Each query slice retrieves its top-1000
           most-similar POOL SLICES (slice-to-slice, the core difference: no
           bucket quantization, no slice->func shortcut). The functions owning
           those slices, unioned across all query slices, form the candidate
           set; each candidate's base_score = mean of its hit-slice
           similarities; top-1000 by base_score.

The contrast isolates the bucket-quantization constraint: does adding the IVF
coarse layer (which keeps memory ~independent of pool size, the key to scaling
to 10M) cost recall vs. exact slice-to-slice nearest neighbour?

------------------------------------------------------------------------------
A2 / A3 (Stage B — rerank) — cheap & exact
------------------------------------------------------------------------------
Stage A (IVF) is run once per query; ``_dense_rerank`` already stores every raw
signal per candidate (q2f, f2q, q2f_idf, f2q_idf, slice_channel, global_sim,
alignment, base_score). Each A2/A3 variant is a re-fusion + re-sort over that
frozen top-1000 set, so every number is EXACT (recall_k <= 1000 == cap).

Usage is byte-for-byte identical to pool_inline_search.py, e.g.:

  python3 pool_inline_search_ablation.py \
      --query-dir artifacts/tmp/query_noinline \
      --save-db   artifacts/model_eval/noinline_10K \
      --pool-dir  artifacts/tmp/pool_noinline_10K \
      --pool-size 10000
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
for _p in (ROOT_DIR, CURRENT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from pool_inline_search import (  # noqa: E402
    EmbeddingRetrievalDB,
    build_arg_parser,
    prepare_eval_pool_dir,
    _build_pool_name_index,
    load_query_pairs,
)
from utils import type_name  # noqa: E402

SLICE_ANN_PER_SLICE_TOPK = 1000
SLICE_ANN_POOL_CHUNK = 2_000_000


class AblationRetrievalDB(EmbeddingRetrievalDB):
    def _ensure_slice2func(self):
        """slice_index -> owning func_id, length P (= total pool slices)."""
        cached = getattr(self, "_slice2func_np", None)
        if cached is not None:
            return cached
        offsets = np.asarray(self.func_partial_offsets, dtype=np.int64)
        counts = np.diff(offsets)
        num_funcs = counts.shape[0]
        s2f = np.repeat(np.arange(num_funcs, dtype=np.int64), counts)
        self._slice2func_np = s2f
        return s2f

    def _slice_ann_recall(self, query_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """no-IVF candidate set: per query slice take top-1000 pool slices by
        exact cosine, union their owning functions, score each function by the
        mean of its hit-slice similarities, return top-cap by that base_score.
        Slice-to-slice (no bucket quantization). Stops at Stage A.
        """
        cap = int(self.CAP)
        norm_emb_np = np.asarray(query_profile["norm_emb"], dtype=np.float32)
        if norm_emb_np.ndim != 2 or norm_emb_np.shape[0] == 0:
            return []
        self._ensure_gpu_state()
        if getattr(self, "_func_partials_gpu", None) is not None:
            funcs, base = self._slice_ann_recall_gpu(norm_emb_np)
        else:
            funcs, base = self._slice_ann_recall_cpu(norm_emb_np)
        if funcs.size == 0:
            return []
        order = np.argsort(-base, kind="stable")[:cap]
        out: List[Dict[str, Any]] = []
        for rank_pos, j in enumerate(order.tolist()):
            fid = int(funcs[j])
            binary_name, func_addr, func_name = self.index_to_meta[fid]
            sc = float(base[j])
            out.append({
                "func_id": fid, "binary_name": binary_name, "func_addr": func_addr,
                "func_name": func_name, "base_score": sc, "score": sc,
            })
        return out

    def _slice_ann_recall_gpu(self, norm_emb_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import torch
        device = self._rerank_device
        partials = self._func_partials_gpu
        P = int(partials.shape[0])
        if P == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        s2f_np = self._ensure_slice2func()
        s2f = getattr(self, "_slice2func_gpu", None)
        if s2f is None:
            s2f = torch.from_numpy(s2f_np).to(device)
            self._slice2func_gpu = s2f
        ne = torch.from_numpy(norm_emb_np).to(device).half()
        R = ne.shape[0]
        k = min(SLICE_ANN_PER_SLICE_TOPK, P)
        best_v = best_i = None
        for s in range(0, P, SLICE_ANN_POOL_CHUNK):
            e = min(P, s + SLICE_ANN_POOL_CHUNK)
            block = partials[s:e]
            sims = (ne @ block.T).float().clamp_(0.0, 1.0)
            kk = min(k, e - s)
            v, i = torch.topk(sims, kk, dim=1)
            i = i + s
            if best_v is None:
                best_v, best_i = v, i
            else:
                cat_v = torch.cat([best_v, v], dim=1)
                cat_i = torch.cat([best_i, i], dim=1)
                kkk = min(k, cat_v.shape[1])
                sv, si = torch.topk(cat_v, kkk, dim=1)
                best_v = sv
                best_i = torch.gather(cat_i, 1, si)
        hit_funcs = s2f[best_i.reshape(-1)]
        hit_sims = best_v.reshape(-1)
        unique_func, inverse = torch.unique(hit_funcs, return_inverse=True)
        N = int(unique_func.numel())
        sum_sims = torch.zeros(N, device=device)
        sum_sims.scatter_add_(0, inverse, hit_sims)
        cnt = torch.zeros(N, device=device)
        cnt.scatter_add_(0, inverse, torch.ones_like(hit_sims))
        base = (sum_sims / cnt.clamp(min=1.0))
        return unique_func.cpu().numpy().astype(np.int64), base.cpu().numpy().astype(np.float32)

    def _slice_ann_recall_cpu(self, norm_emb_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        flat = self.func_partials_flat
        if flat is None or flat.shape[0] == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        P = int(flat.shape[0])
        s2f = self._ensure_slice2func()
        k = min(SLICE_ANN_PER_SLICE_TOPK, P)
        hit_func_parts: List[np.ndarray] = []
        hit_sim_parts: List[np.ndarray] = []
        for r in range(norm_emb_np.shape[0]):
            q = norm_emb_np[r]
            sims = np.clip(np.asarray(flat, dtype=np.float32) @ q, 0.0, 1.0)
            if k < P:
                top = np.argpartition(-sims, k - 1)[:k]
            else:
                top = np.arange(P)
            hit_func_parts.append(s2f[top])
            hit_sim_parts.append(sims[top])
        hit_funcs = np.concatenate(hit_func_parts)
        hit_sims = np.concatenate(hit_sim_parts)
        order = np.argsort(hit_funcs, kind="mergesort")
        uf, start = np.unique(hit_funcs[order], return_index=True)
        sum_sims = np.add.reduceat(hit_sims[order], start)
        cnt = np.add.reduceat(np.ones_like(hit_sims), start)
        base = sum_sims / np.maximum(cnt, 1.0)
        return uf.astype(np.int64), base.astype(np.float32)


def _block_to_records(db, block):
    """numpy stage block -> list[dict], same schema as search() (already top-CAP order)."""
    fids = None if block is None else block.get("fids")
    if fids is None or int(fids.shape[0]) == 0:
        return []
    n = int(fids.shape[0])
    meta = db.index_to_meta
    numeric_keys = [k for k in (
        "hit_count", "slice_score", "support", "base_score", "slice_sim", "f2q_sim",
        "q2f_idf", "f2q_idf", "slice_channel", "global_sim", "alignment", "score",
        "verify_conn", "anchor_overlap", "anchor_dice") if k in block]
    recs = []
    for i in range(n):
        fid = int(fids[i])
        b, a, nm = meta[fid]
        row = {"func_id": fid, "binary_name": b, "func_addr": a, "func_name": nm}
        for k in numeric_keys:
            v = block[k][i]
            row[k] = int(v) if k == "hit_count" else float(v)
        row["rank"] = i + 1
        recs.append(row)
    return recs


def _dyn_w(it: Dict[str, Any], db: EmbeddingRetrievalDB) -> Tuple[float, float]:
    """Production alignment-dynamic (w_slice, w_global). alignment = min(q2f,
    f2q) is fixed across A2/A3 variants."""
    a = float(it.get("alignment", 0.0))
    wg = db.RERANK_W_GLOBAL_LOW + (db.RERANK_W_GLOBAL_HIGH - db.RERANK_W_GLOBAL_LOW) * a
    ws = db.RERANK_W_SLICE_HIGH - (db.RERANK_W_SLICE_HIGH - db.RERANK_W_SLICE_LOW) * a
    return ws, wg


def _make_a2(kind):
    def f(it, db):
        ws, wg = _dyn_w(it, db)
        wb = db.RERANK_W_BASE_CONST
        base = float(it.get("base_score", 0.0))
        sc = float(it.get("slice_channel", 0.0))
        gl = float(it.get("global_sim", 0.0))
        if kind == "all":         return wb * base + ws * sc + wg * gl
        if kind == "no_base":     return ws * sc + wg * gl
        if kind == "no_slice":    return wb * base + wg * gl
        if kind == "no_global":   return wb * base + ws * sc
        if kind == "slice_only":  return sc
        if kind == "global_only": return gl
        return wb * base + ws * sc + wg * gl
    return f


A2 = {
    "all_channels (base+slice+global)*": _make_a2("all"),
    "no_base":     _make_a2("no_base"),
    "no_slice":    _make_a2("no_slice"),
    "no_global":   _make_a2("no_global"),
    "slice_only":  _make_a2("slice_only"),
    "global_only": _make_a2("global_only"),
}


def _slice_channel_variant(it, kind, db=None):
    q2f = float(it.get("slice_sim", 0.0))
    f2q = float(it.get("f2q_sim", 0.0))
    f2q_idf = float(it.get("f2q_idf", 0.0))
    excess = float(getattr(db, "RERANK_EXCESS_COEF", 0.80)) if db is not None else 0.80
    if kind == "q2f_plus_excess": return q2f + excess * max(0.0, f2q_idf - q2f)
    if kind == "q2f_only":        return q2f
    if kind == "f2q_only":        return f2q
    if kind == "symmetric_mean":  return 0.5 * (q2f + f2q)
    if kind == "max_dir":         return max(q2f, f2q)
    return q2f


def _make_a3(kind):
    def f(it, db):
        ws, wg = _dyn_w(it, db)
        wb = db.RERANK_W_BASE_CONST
        base = float(it.get("base_score", 0.0))
        gl = float(it.get("global_sim", 0.0))
        sc = _slice_channel_variant(it, kind, db)
        return wb * base + ws * sc + wg * gl
    return f


A3 = {
    "q2f+0.80*excess (current)*": _make_a3("q2f_plus_excess"),
    "q2f_only":                  _make_a3("q2f_only"),
    "f2q_only":                  _make_a3("f2q_only"),
    "symmetric_mean":            _make_a3("symmetric_mean"),
    "max(q2f,f2q)":              _make_a3("max_dir"),
}

A1_NAME = "A1 - Stage A candidate generation (IVF slice->bucket->func vs no-IVF slice->slice ANN)"
A2_NAME = "A2 - Rerank channel leave-one-out (base / slice / global) [Stage B]"
A3_NAME = "A3 - Slice-similarity directionality (slice_channel formula) [Stage B]"

A1_VARIANTS = ("ours: IVF (slice->bucket->func)*", "no-IVF: slice->slice ANN (top-1000/slice)")



def _gt_rank(ranked_list, hit_keys) -> Optional[int]:
    for i, it in enumerate(ranked_list, 1):
        if (it["binary_name"], it["func_addr"]) in hit_keys:
            return i
    return None


def evaluate_ablations(
    db: AblationRetrievalDB,
    query_pairs,
    recall_ks=(1, 5, 10, 50, 100, 200, 500, 1000),
    name_to_keys: Optional[Dict[str, Set[Tuple[str, str]]]] = None,
    keys_to_name: Optional[Dict[Tuple[str, str], str]] = None,
) -> Dict[str, Any]:
    recall_ks = tuple(sorted(set(int(k) for k in recall_ks)))
    loose_enabled = name_to_keys is not None and keys_to_name is not None

    variant_order: Dict[str, List[str]] = {
        A1_NAME: list(A1_VARIANTS),
        A2_NAME: list(A2.keys()),
        A3_NAME: list(A3.keys()),
    }
    cells: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for abl, vs in variant_order.items():
        for var in vs:
            cells[(abl, var)] = {
                "hits": {k: 0 for k in recall_ks},
                "type_hits": {t: {k: 0 for k in recall_ks} for t in (0, 1, 2)},
            }

    type_counts = {0: 0, 1: 0, 2: 0}
    total = 0

    for q, gq, qt, gt_binary, gt_addr, qrm in tqdm(
        query_pairs, desc="ablation: queries", total=len(query_pairs), dynamic_ncols=True,
    ):
        strict = (gt_binary, gt_addr)
        hit_keys: Set[Tuple[str, str]] = {strict}
        if loose_enabled:
            gt_name = keys_to_name.get(strict, "")
            if gt_name:
                extra = name_to_keys.get(gt_name, set())
                if extra:
                    hit_keys = hit_keys | extra

        qp = db._build_query_profile(q, qrm, global_query=gq)
        cand_ids, aggregates, _dbg = db._build_query_slice_sets(qp)
        stage_a_block = db._score_stage_a(cand_ids, aggregates, qp)
        reranked_block = db._dense_rerank(stage_a_block, qp)
        stage_a = _block_to_records(db, stage_a_block)
        reranked = _block_to_records(db, reranked_block)
        noivf = db._slice_ann_recall(qp)

        total += 1
        type_counts[qt] = type_counts.get(qt, 0) + 1

        variant_lists: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        variant_lists[(A1_NAME, A1_VARIANTS[0])] = stage_a
        variant_lists[(A1_NAME, A1_VARIANTS[1])] = noivf
        for var, fn in A2.items():
            variant_lists[(A2_NAME, var)] = sorted(
                reranked, key=lambda it, fn=fn: (-fn(it, db), it["binary_name"], str(it["func_addr"])))
        for var, fn in A3.items():
            variant_lists[(A3_NAME, var)] = sorted(
                reranked, key=lambda it, fn=fn: (-fn(it, db), it["binary_name"], str(it["func_addr"])))

        for key, ranked_list in variant_lists.items():
            gr = _gt_rank(ranked_list, hit_keys)
            if gr is None:
                continue
            cell = cells[key]
            for k in recall_ks:
                if gr <= k:
                    cell["hits"][k] += 1
                    cell["type_hits"][qt][k] += 1

    out: Dict[str, Any] = {
        "queries": total,
        "root_queries": type_counts.get(0, 0),
        "internal_queries": type_counts.get(1, 0),
        "leaf_queries": type_counts.get(2, 0),
        "recall_ks": list(recall_ks),
        "note": "A1 stops at Stage A: recall@1000 == candidate-set recall ceiling "
                "of the mechanism. A2/A3 run on the IVF+StageB set.",
        "ablations": {},
    }
    for abl, vs in variant_order.items():
        out["ablations"][abl] = {}
        for var in vs:
            cell = cells[(abl, var)]
            entry: Dict[str, float] = {}
            for k in recall_ks:
                entry[f"recall@{k}"] = (cell["hits"][k] / total) if total else 0.0
            for t in (0, 1, 2):
                tc = type_counts.get(t, 0)
                for k in recall_ks:
                    entry[f"{type_name(t)}_recall@{k}"] = (cell["type_hits"][t][k] / tc) if tc else 0.0
            out["ablations"][abl][var] = entry
    return out


def _print_tables(out: Dict[str, Any]) -> None:
    show_k = [k for k in (1, 5, 10, 50, 100, 1000) if k in out["recall_ks"]]
    namew = 38
    line = "=" * 96
    print("\n" + line)
    print(f"ABLATION RESULTS  (queries={out['queries']}  "
          f"root={out['root_queries']} internal={out['internal_queries']} leaf={out['leaf_queries']})")
    print(out["note"])
    print(line)

    for abl, variants in out["ablations"].items():
        print("\n" + abl)
        header = f"  {'variant':<{namew}}" + "".join(f"{'@'+str(k):>9}" for k in show_k)
        print(header)
        print("  " + "-" * (namew + 9 * len(show_k)))
        for var, entry in variants.items():
            row = f"  {var:<{namew}}" + "".join(f"{entry[f'recall@{k}']:>9.4f}" for k in show_k)
            print(row)
        pt_k = 1000 if abl == A1_NAME else 10
        if pt_k not in out["recall_ks"]:
            pt_k = max(out["recall_ks"])
        print(f"  {('per-type @' + str(pt_k)):<{namew}}{'root':>9}{'internal':>9}{'leaf':>9}")
        for var, entry in variants.items():
            row = (f"  {var:<{namew}}"
                   f"{entry[f'root_recall@{pt_k}']:>9.4f}"
                   f"{entry[f'internal_recall@{pt_k}']:>9.4f}"
                   f"{entry[f'leaf_recall@{pt_k}']:>9.4f}")
            print(row)
    print("\n  (* = production reference; A1 @1000 = candidate-set recall ceiling)")
    print(line + "\n")


def main():
    args = build_arg_parser().parse_args()
    if args.embedding_suffix is None:
        from utils import derive_embedding_suffix
        args.embedding_suffix = derive_embedding_suffix(args.feature_suffix)
    built = prepare_eval_pool_dir(args)

    faiss_gpu_id = 0
    if args.rerank_device and args.rerank_device.startswith("cuda:"):
        try:
            faiss_gpu_id = int(args.rerank_device.split(":", 1)[1])
        except ValueError:
            faiss_gpu_id = 0

    db = AblationRetrievalDB(
        built, args.model, args.save_db,
        feature_suffix=args.feature_suffix, embedding_suffix=args.embedding_suffix,
        faiss_gpu_id=faiss_gpu_id,
    )
    if args.cap is not None:
        db.CAP = int(args.cap)
    else:
        db.CAP = EmbeddingRetrievalDB.recommended_cap(int(args.pool_size))
    if args.btopk is not None:
        db.BTOPK = int(args.btopk)
    if args.rerank_device is not None:
        db.RERANK_DEVICE = str(args.rerank_device)
    db.build_pool()
    db._ensure_gpu_state()

    name_to_keys, keys_to_name = _build_pool_name_index(built)
    out = evaluate_ablations(
        db,
        load_query_pairs(args.query_dir, limit_per_file=args.limit_per_file, limit_total=args.limit_total),
        recall_ks=(1, 5, 10, 50, 100, 200, 500, 1000),
        name_to_keys=name_to_keys,
        keys_to_name=keys_to_name,
    )
    out["built_pool_dir"] = built
    out["pool_size"] = int(args.pool_size)

    save_dir = Path(args.save_db)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "ablation_eval_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_tables(out)
    print(f"[saved] {save_dir / 'ablation_eval_results.json'}")


if __name__ == "__main__":
    main()
