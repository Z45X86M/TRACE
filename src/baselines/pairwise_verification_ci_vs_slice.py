"""Pairwise verification: CI-Detector vs our slice (2-channel) — ROC-AUC.

CI-Detector's native regime is *pairwise verification* (margin-loss embedding,
trained/evaluated on pos/neg pairs), NOT pool retrieval. This script puts both
methods in that regime: can they separate matching (query, GT) pairs from
non-matching (query, non-GT) pairs? Both are type-agnostic at scoring; pairs are
built from root/internal/leaf queries (we only do recall — the model can't judge
inline type), which mirrors how CI is meant to be used.

No FAISS / no retrieval here — a pairwise comparison is computed directly from
the per-function vectors:

OURS (2-channel, pure pairwise):
    q2f = mean_i max_j cos(q_i, p_j)        # query slices -> cand slices
    f2q = mean_j max_i cos(p_j, q_i)        # cand slices  -> query slices
    slice_channel = q2f + 0.58 * max(0, f2q - q2f)
    alignment     = min(q2f, f2q)
    w_slice = 0.85 - 0.55*alignment ;  w_global = 0.05 + 0.60*alignment
    global_sim = cos(gq, p_global)
    score = w_slice*slice_channel + w_global*global_sim
  This is the production fusion (pool_inline_search.RERANK_*, post 2026-05-27
  retune) stripped of the two retrieval-stage, pool-coupled terms that have
  no meaning in pairwise matching: the Stage-A ``base`` channel, and the
  pool-IDF reweighting of f2q. In production the slice channel uses f2q_idf
  (per-slice IDF from the bucket postings); in pure pairwise we have no pool,
  so it degrades to raw f2q. The 0.58 coefficient and 0.05 global floor still
  match production.

CI (type-agnostic ensemble, paper-style):
    combine_n cos( ci_n(query), ci_n(cand) )  over the 3 per-pattern encoders
    (root/internal/leaf), default combine = max ("match under ANY detector",
    since the pattern is unknown at inference).

Metric: ROC-AUC (Mann-Whitney), overall + per query-type, CI vs OURS.
Data:   query_partial + pool_partial_10K (10K scale) by default.
"""

import argparse
import glob
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

W_GLOBAL_LOW, W_GLOBAL_HIGH = 0.05, 0.65
W_SLICE_LOW, W_SLICE_HIGH = 0.30, 0.85
EXCESS_COEF = 0.58

CI_EMB_SUFFIX_TMPL = "_cross_inlining_embeddings_type%d.pkl"
CI_TYPES = (0, 1, 2)


def roc_auc(pos: List[float], neg: List[float]) -> float:
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    sorted_v = allv[order]
    ranks = np.empty(len(allv), dtype=np.float64)
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rank_pos_sum = ranks[:n_pos].sum()
    return float((rank_pos_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_curve_pts(pos: List[float], neg: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fpr, tpr) points of the ROC curve. Threshold = score >= t (higher=match)."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-scores, kind="mergesort")
    s, l = scores[order], labels[order]
    tp = np.cumsum(l)
    fp = np.cumsum(1.0 - l)
    P, N = float(l.sum()), float(len(l) - l.sum())
    keep = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tpr = np.r_[0.0, tp[keep] / max(P, 1.0)]
    fpr = np.r_[0.0, fp[keep] / max(N, 1.0)]
    return fpr, tpr


def _interp_tpr(fpr: np.ndarray, tpr: np.ndarray, grid: np.ndarray) -> List[float]:
    return [float(v) for v in np.interp(grid, fpr, tpr)]


def _l2norm_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


def load_pool(pool_dir: str, feature_suffix: str, embedding_suffix: str):
    keys: List[Tuple[str, str]] = []
    func_names: List[str] = []
    partials: List[np.ndarray] = []
    globals_: List[np.ndarray] = []
    for emb_path in sorted(glob.glob(os.path.join(pool_dir, f"*{embedding_suffix}"))):
        binary = os.path.basename(emb_path)[: -len(embedding_suffix)]
        feat_path = emb_path[: -len(embedding_suffix)] + feature_suffix
        feat = json.load(open(feat_path, encoding="utf-8")) if os.path.exists(feat_path) else {}
        d = pickle.load(open(emb_path, "rb"))
        for addr, item in d.items():
            if not isinstance(item, dict):
                continue
            glob_v = item.get("global")
            if glob_v is None:
                continue
            g = _l2norm_rows(np.asarray(glob_v, dtype=np.float32))[0]
            part = item.get("partial")
            P = _l2norm_rows(np.asarray(part, dtype=np.float32)) if part is not None else np.zeros((0, 0), np.float32)
            if P.ndim != 2 or P.shape[0] == 0 or P.shape[1] != g.shape[0]:
                P = np.zeros((0, g.shape[0]), dtype=np.float32)
            keys.append((binary, str(addr)))
            func_names.append(str(feat.get(addr, {}).get("func_name") or ""))
            partials.append(P)
            globals_.append(_l2norm_rows(np.asarray(glob_v, dtype=np.float32))[0])
    return keys, func_names, partials, np.stack(globals_, axis=0) if globals_ else np.zeros((0, 0), np.float32)


class CIEmbStore:
    def __init__(self, root: str):
        self.root = root
        self._cache: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}

    def _load(self, binary: str, type_id: int) -> Dict[str, np.ndarray]:
        key = (binary, type_id)
        if key not in self._cache:
            path = os.path.join(self.root, f"{binary}{CI_EMB_SUFFIX_TMPL % type_id}")
            d: Dict[str, np.ndarray] = {}
            if os.path.exists(path):
                raw = pickle.load(open(path, "rb"))
                for addr, item in raw.items():
                    vec = item.get("global") if isinstance(item, dict) else None
                    if vec is not None:
                        a = np.asarray(vec, dtype=np.float32)
                        nrm = np.linalg.norm(a)
                        d[str(addr)] = a / nrm if nrm > 1e-12 else a
            self._cache[key] = d
        return self._cache[key]

    def get(self, binary: str, addr: str) -> Optional[np.ndarray]:
        rows = []
        for t in CI_TYPES:
            v = self._load(binary, t).get(str(addr))
            if v is None:
                return None
            rows.append(v)
        return np.stack(rows, axis=0)


def ours_pair_scores(Q: np.ndarray, gq: np.ndarray,
                     cand_partials: List[np.ndarray], cand_globals: np.ndarray) -> np.ndarray:
    """Q: (R, D) normalized query slices; gq: (D,) normalized query global.
    cand_partials[m]: (S_m, D) normalized; cand_globals: (M, D) normalized.
    Returns (M,) 2-channel scores."""
    M = len(cand_partials)
    out = np.zeros(M, dtype=np.float32)
    global_sims = np.clip(cand_globals @ gq, 0.0, 1.0)
    for m in range(M):
        P = cand_partials[m]
        if P.shape[0] == 0 or P.shape[1] != Q.shape[1]:
            q2f = f2q = 0.0
        else:
            sim = np.clip(Q @ P.T, 0.0, 1.0)
            q2f = float(sim.max(axis=1).mean())
            f2q = float(sim.max(axis=0).mean())
        slice_channel = q2f + EXCESS_COEF * max(0.0, f2q - q2f)
        align = min(q2f, f2q)
        w_global = W_GLOBAL_LOW + (W_GLOBAL_HIGH - W_GLOBAL_LOW) * align
        w_slice = W_SLICE_HIGH - (W_SLICE_HIGH - W_SLICE_LOW) * align
        out[m] = w_slice * slice_channel + w_global * float(global_sims[m])
    return out


def ci_pair_scores(query_vec: np.ndarray, cand_vecs: np.ndarray, combine: str) -> np.ndarray:
    """query_vec: (T, D); cand_vecs: (M, T, D) -> (M,) combined similarity."""
    per_type = np.einsum("td,mtd->mt", query_vec, cand_vecs)
    return per_type.mean(axis=1) if combine == "mean" else per_type.max(axis=1)


def load_query_records(query_dir: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(query_dir)):
        if not name.endswith(".pkl"):
            continue
        recs = pickle.load(open(os.path.join(query_dir, name), "rb"))
        if isinstance(recs, list):
            out.extend(r for r in recs if isinstance(r, dict))
    return out


def main():
    ap = argparse.ArgumentParser(description="Pairwise verification ROC-AUC: CI-Detector vs slice (2-channel). No FAISS.")
    ap.add_argument("--pool-dir", default="artifacts/tmp/pool_partial_10K")
    ap.add_argument("--query-dir", default="artifacts/tmp/query_partial")
    ap.add_argument("--ci-pool-dir", default="data/inline/pool")
    ap.add_argument("--ci-query-dir", default="data/inline/strip")
    ap.add_argument("--neg-per-query", type=int, default=100)
    ap.add_argument("--ci-combine", choices=("max", "mean"), default="max")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--feature-suffix", default="_bb_slice_feature.json")
    ap.add_argument("--embedding-suffix", default="_bb_slice_embeddings.pkl")
    ap.add_argument("--output", default="artifacts/model_eval/pairwise_ci_vs_slice.json")
    ap.add_argument("--plot", default="artifacts/model_eval/pairwise_ci_vs_slice_roc.png",
                    help="ROC curve figure (CI vs OURS, overall + per type). Empty string to skip.")
    ap.add_argument("--limit-queries", type=int, default=0, help="0 = all queries.")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print(f"[pool] loading slices from {args.pool_dir}")
    keys, func_names, partials, pool_globals = load_pool(args.pool_dir, args.feature_suffix, args.embedding_suffix)
    n_funcs = len(keys)
    if n_funcs == 0:
        raise SystemExit("empty pool")
    key2idx = {k: i for i, k in enumerate(keys)}
    keys_to_name = {keys[i]: func_names[i] for i in range(n_funcs) if func_names[i]}
    name_to_keys: Dict[str, set] = {}
    for i in range(n_funcs):
        if func_names[i]:
            name_to_keys.setdefault(func_names[i], set()).add(keys[i])
    print(f"[pool] {n_funcs} functions")

    print(f"[ci] preloading pool CI vectors from {args.ci_pool_dir}")
    pool_ci_store = CIEmbStore(args.ci_pool_dir)
    pool_ci: Optional[np.ndarray] = None
    pool_ci_ok = np.zeros(n_funcs, dtype=bool)
    for i, (b, a) in enumerate(keys):
        v = pool_ci_store.get(b, a)
        if v is None:
            continue
        if pool_ci is None:
            pool_ci = np.zeros((n_funcs, v.shape[0], v.shape[1]), dtype=np.float32)
        pool_ci[i] = v
        pool_ci_ok[i] = True
    if pool_ci is None:
        raise SystemExit("no pool CI vectors found")
    print(f"[ci] pool CI coverage: {int(pool_ci_ok.sum())}/{n_funcs}")

    query_ci_store = CIEmbStore(args.ci_query_dir)
    records = load_query_records(args.query_dir)
    if args.limit_queries > 0:
        records = records[:args.limit_queries]
    print(f"[query] {len(records)} records")

    pos = {"ci": {0: [], 1: [], 2: []}, "ours": {0: [], 1: [], 2: []}}
    neg = {"ci": {0: [], 1: [], 2: []}, "ours": {0: [], 1: [], 2: []}}
    stats = {"queries_used": 0, "skipped_no_pos": 0, "skipped_no_query_ci": 0,
             "skipped_no_qslices": 0, "pos_pairs": 0, "neg_pairs": 0}

    for ri, rec in enumerate(records):
        if ri % 200 == 0:
            print(f"  ... {ri}/{len(records)}", flush=True)
        qbin, qaddr = str(rec.get("binary")), str(rec.get("query_addr"))
        qtype = int(rec.get("type", 2))
        gt_key = (str(rec.get("gt_binary")), str(rec.get("addr")))
        gt_name = keys_to_name.get(gt_key)

        pos_keys = set(name_to_keys.get(gt_name, set())) if gt_name else set()
        if gt_key in key2idx:
            pos_keys.add(gt_key)
        pos_idx = [key2idx[k] for k in pos_keys if k in key2idx and pool_ci_ok[key2idx[k]]]
        if not pos_idx:
            stats["skipped_no_pos"] += 1
            continue

        qci = query_ci_store.get(qbin, qaddr)
        if qci is None:
            stats["skipped_no_query_ci"] += 1
            continue

        Q = _l2norm_rows(np.asarray(rec.get("q"), dtype=np.float32))
        if Q.ndim != 2 or Q.shape[0] == 0:
            stats["skipped_no_qslices"] += 1
            continue
        gq = _l2norm_rows(np.asarray(rec.get("gq"), dtype=np.float32))[0]

        exclude = {gt_name} if gt_name else set()
        neg_idx: List[int] = []
        tries = 0
        while len(neg_idx) < args.neg_per_query and tries < args.neg_per_query * 20:
            tries += 1
            f = rng.randrange(n_funcs)
            if not pool_ci_ok[f]:
                continue
            if func_names[f] and func_names[f] in exclude:
                continue
            neg_idx.append(f)

        cand = pos_idx + neg_idx
        labels = [1] * len(pos_idx) + [0] * len(neg_idx)
        cand_partials = [partials[i] for i in cand]
        cand_globals = pool_globals[np.asarray(cand, dtype=np.int64)]

        ours_scores = ours_pair_scores(Q, gq, cand_partials, cand_globals)
        ci_scores = ci_pair_scores(qci, pool_ci[np.asarray(cand, dtype=np.int64)], args.ci_combine)

        for s_ci, s_ours, lab in zip(ci_scores, ours_scores, labels):
            bank = pos if lab == 1 else neg
            bank["ci"][qtype].append(float(s_ci))
            bank["ours"][qtype].append(float(s_ours))
        stats["queries_used"] += 1
        stats["pos_pairs"] += len(pos_idx)
        stats["neg_pairs"] += len(neg_idx)

    def auc_for(method: str, types: List[int]) -> float:
        p = [v for t in types for v in pos[method][t]]
        n = [v for t in types for v in neg[method][t]]
        return roc_auc(p, n)

    result = {
        "config": {k: getattr(args, k) for k in
                   ("pool_dir", "query_dir", "ci_pool_dir", "ci_query_dir",
                    "neg_per_query", "ci_combine", "seed")},
        "stats": stats,
        "roc_auc": {
            "CI_%s" % args.ci_combine: {"overall": auc_for("ci", [0, 1, 2]),
                                        "root": auc_for("ci", [0]), "internal": auc_for("ci", [1]), "leaf": auc_for("ci", [2])},
            "OURS_2ch": {"overall": auc_for("ours", [0, 1, 2]),
                         "root": auc_for("ours", [0]), "internal": auc_for("ours", [1]), "leaf": auc_for("ours", [2])},
        },
        "type_pair_counts": {name: {"pos": len(pos["ci"][t]), "neg": len(neg["ci"][t])}
                             for t, name in ((0, "root"), (1, "internal"), (2, "leaf"))},
    }

    buckets = {"overall": [0, 1, 2], "root": [0], "internal": [1], "leaf": [2]}
    grid = np.linspace(0.0, 1.0, 101)
    curves: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    result["roc_curve_grid_fpr"] = [float(v) for v in grid]
    result["roc_curve_tpr"] = {}
    for bname, tps in buckets.items():
        curves[bname] = {}
        result["roc_curve_tpr"][bname] = {}
        for method in ("ci", "ours"):
            p = [v for t in tps for v in pos[method][t]]
            n = [v for t in tps for v in neg[method][t]]
            fpr, tpr = roc_curve_pts(p, n)
            curves[bname][method] = (fpr, tpr)
            result["roc_curve_tpr"][bname][method] = _interp_tpr(fpr, tpr, grid)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.plot:
        ci_key = "CI_%s" % args.ci_combine
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        for ax, bname in zip(axes.ravel(), ("overall", "root", "internal", "leaf")):
            auc_ci = result["roc_auc"][ci_key][bname]
            auc_ours = result["roc_auc"]["OURS_2ch"][bname]
            f_ci, t_ci = curves[bname]["ci"]
            f_ou, t_ou = curves[bname]["ours"]
            ax.plot(f_ou, t_ou, color="#d62728", lw=2.2, label=f"OURS (2ch)  AUC={auc_ours:.3f}")
            ax.plot(f_ci, t_ci, color="#1f77b4", lw=2.2, label=f"CI ({args.ci_combine})  AUC={auc_ci:.3f}")
            ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="random  AUC=0.500")
            npos = len(pos["ci"][0] if bname == "root" else pos["ci"][1] if bname == "internal"
                       else pos["ci"][2] if bname == "leaf" else [v for t in (0, 1, 2) for v in pos["ci"][t]])
            ax.set_title(f"{bname}  (pos pairs={npos})")
            ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.001); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)
        fig.suptitle("Pairwise verification ROC — CI-Detector vs slice (2-channel), partial 10K",
                     fontsize=13, y=0.995)
        fig.tight_layout()
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
        print(f"[plot] {plot_path}")

    print("\n=== pairwise verification ROC-AUC (CI vs OURS, no FAISS) ===")
    print(f"queries_used={stats['queries_used']}  pos_pairs={stats['pos_pairs']}  neg_pairs={stats['neg_pairs']}  "
          f"(neg/query={args.neg_per_query}, ci_combine={args.ci_combine})")
    hdr = f"{'method':16s} {'overall':>9s} {'root':>9s} {'internal':>9s} {'leaf':>9s}"
    print(hdr); print("-" * len(hdr))
    for label, key in (("CI (%s)" % args.ci_combine, "CI_%s" % args.ci_combine), ("OURS (2ch)", "OURS_2ch")):
        r = result["roc_auc"][key]
        print(f"{label:16s} {r['overall']:9.4f} {r['root']:9.4f} {r['internal']:9.4f} {r['leaf']:9.4f}")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
