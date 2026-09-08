"""Evaluate the LLM verifier scores: does it recover recall the deterministic
verifier could not? Reports, on the prepped sample (top-K rerank window):
  * baseline recall@{1,5,10} from Stage-B score
  * pure-LLM rerank (sort by P(Yes), tiebreak Stage-B)
  * promote-only fusion: score' = stageB_norm + lam*llm  (best lam per metric)
  * HARD-NEG AUC: does P(Yes) separate GT from the negs ranked ABOVE it?
    (directly comparable to the deterministic 0.49)

Usage:
  PYTHONPATH=.. python3 llm_rerank_eval.py \
    --scores artifacts/diagnostics/llm_rerank_scores_partial_100K.pkl
"""
import argparse
import sys
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
for p in (str(ROOT_DIR), str(CURRENT_DIR)):
    if p not in sys.path:
        sys.path.append(p)
from utils import read_pickle

KS = (1, 5, 10)


def gt_rank(score, is_gt):
    order = np.argsort(-score, kind="stable")
    pos = np.flatnonzero(is_gt[order])
    return int(pos[0]) if pos.size else None


def recall(records, key):
    agg = {t: {k: 0 for k in KS} for t in (0, 1, 2, "all")}
    nn = {t: 0 for t in (0, 1, 2, "all")}
    for r in records:
        gr = key(r)
        for tt in (r["qt"], "all"):
            nn[tt] += 1
            if gr is not None:
                for k in KS:
                    if gr < k:
                        agg[tt][k] += 1
    out = {}
    for tt in (0, 1, 2, "all"):
        n = nn[tt] or 1
        out[tt] = {k: agg[tt][k] / n for k in KS}
        out[tt]["n"] = nn[tt]
    return out


def norm01(v):
    v = v.astype(np.float64)
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < 1e-9:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def show(tag, m, base):
    print("  %-26s @1=%.4f(%+.4f) @5=%.4f(%+.4f) @10=%.4f(%+.4f) | t1@10=%+.4f t2@10=%+.4f"
          % (tag, m["all"][1], m["all"][1] - base["all"][1],
             m["all"][5], m["all"][5] - base["all"][5],
             m["all"][10], m["all"][10] - base["all"][10],
             m[1][10] - base[1][10], m[2][10] - base[2][10]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    args = ap.parse_args()
    d = read_pickle(args.scores)
    records, llm, K = d["records"], d["llm"], d["rerank_k"]
    print("[eval] %d records, rerank window top-%d | LLM run: %d pairs %.1fs %.2f pairs/s"
          % (len(records), K, d.get("n_pairs", 0), d.get("elapsed_s", 0),
             d.get("n_pairs", 0) / max(d.get("elapsed_s", 1), 1e-9)))

    for ri, r in enumerate(records):
        n = len(llm[ri])
        r["_sc"] = r["score"][:n].astype(np.float64)
        r["_gt"] = r["is_gt"][:n]
        r["_llm"] = np.nan_to_num(llm[ri], nan=0.0).astype(np.float64)

    base = recall(records, lambda r: gt_rank(r["_sc"], r["_gt"]))
    print("\n[baseline within top-%d window]" % K)
    for tt in (0, 1, 2, "all"):
        b = base[tt]
        print("  type%-3s n=%-4d @1=%.4f @5=%.4f @10=%.4f" % (str(tt), b["n"], b[1], b[5], b[10]))

    mp = recall(records, lambda r: gt_rank(r["_llm"] + 1e-6 * norm01(r["_sc"]), r["_gt"]))
    print("\n==== pure-LLM rerank (sort by P(Yes), tiebreak StageB) ====")
    show("pure-LLM", mp, base)

    print("\n==== promote-only fusion: StageB_norm + lam*P(Yes) ====")
    best = None
    for lam in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        m = recall(records, lambda r, l=lam: gt_rank(norm01(r["_sc"]) + l * r["_llm"], r["_gt"]))
        g = (m["all"][1] - base["all"][1]) + (m["all"][10] - base["all"][10])
        show("lam=%.2f" % lam, m, base)
        if best is None or g > best[0]:
            best = (g, lam, m)
    print("  -> best fusion lam=%.2f" % best[1])

    print("\n==== HARD-NEG AUC: does P(Yes) separate GT from negs ranked above it? ====")
    for tt in (0, 1, 2, "all"):
        pos, neg = [], []
        for r in records:
            if tt != "all" and r["qt"] != tt:
                continue
            gr = gt_rank(r["_sc"], r["_gt"])
            if gr is None or gr == 0:
                continue
            order = np.argsort(-r["_sc"], kind="stable")
            gti = order[gr]
            pos.append(r["_llm"][gti])
            for ai in order[:gr]:
                neg.append(r["_llm"][ai])
        if pos and neg:
            P = np.array(pos)[:, None]
            N = np.array(neg)[None, :]
            auc = float((P > N).mean() + 0.5 * (P == N).mean())
            print("  type%-3s headroom-q=%d  neg-above=%d  AUC(P(Yes): GT>neg)=%.3f"
                  % (str(tt), len(pos), len(neg), auc))
        else:
            print("  type%-3s (no headroom)" % str(tt))


if __name__ == "__main__":
    main()
