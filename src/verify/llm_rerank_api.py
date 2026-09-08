"""Concurrent LLM verifier via cloud API (DeepSeek / DashScope) — CoT-Pro, API-only.

We call the cloud API CONCURRENTLY (thread pool): a query's top-K candidate pairs
are sent in parallel and received in parallel, overlapping across queries up to
--concurrency in flight. The prompt is CoT-Pro (llm_verifier_prompt.build_messages):
the model reasons through guiding questions then emits a final `ANSWER: <LABEL>`.

We do NOT ask the LLM for a score. classify() parses the DISCRETE label and maps it
to a rule-based match flag (1.0 for HOST/INLINE, 0.0 for NO_MATCH). The rerank is a
label-driven stable promotion downstream (llm_rerank_eval.py).

NO TEXT TRUNCATION: full decompiled function text is sent both sides. The model
context (128K+) dwarfs our functions (~a few k tokens); the old char caps were a
local-V100 artifact and are gone. Thinking is disabled (extra_body) so all CoT is
in the response content and parseable.

Usage:
  DEEPSEEK_API_KEY=sk-... PYTHONPATH=.. python3 llm_rerank_api.py \
    --prep artifacts/diagnostics/llm_rerank_prep_CVE_1M_top50.pkl \
    --model deepseek-v4-flash --rerank-k 50 --concurrency 400 \
    --out artifacts/diagnostics/llm_rerank_scores_api.pkl
"""
import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
for p in (str(ROOT_DIR), str(CURRENT_DIR)):
    if p not in sys.path:
        sys.path.append(p)
from utils import read_pickle, write_pickle
from llm_verifier_prompt import build_messages, LABELS, MATCH_LABEL_IDX, parse_label

PROVIDERS = {
    "dashscope": {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                  "extra": {"enable_thinking": False}},
    "deepseek":  {"base_url": "https://api.deepseek.com",
                  "extra": {"thinking": {"type": "disabled"}}},
}


def classify(client, model, q, c, extra, retries=5, max_tokens=512):
    """CoT-Pro: the model reasons through the guiding questions then emits a final
    `ANSWER: <LABEL>` line. We parse the DISCRETE label only (no score) and map it
    to a rule-based match flag: 1.0 for HOST/INLINE, 0.0 for NO_MATCH / unparseable.
    Full function text both sides — NO truncation.
    -> (match_flag, pred_label, total_tokens, cached_tokens) or ('ERR', msg, 0, 0)."""
    msgs = build_messages(q, c)
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=msgs,
                extra_body=extra,
                max_tokens=max_tokens, temperature=0,
                timeout=120)
            ch = r.choices[0]
            pred = parse_label(ch.message.content or "")
            if pred is None:
                pred = -1
            match = 1.0 if pred in MATCH_LABEL_IDX else 0.0
            u = r.usage
            tot = u.total_tokens if u else 0
            ptd = getattr(u, "prompt_tokens_details", None) if u else None
            cached = (getattr(ptd, "cached_tokens", 0) or 0) if ptd else 0
            return match, int(pred), tot, cached
        except Exception as e:
            if attempt == retries - 1:
                return ("ERR", str(e)[:120], 0, 0)
            time.sleep(1.5 * (attempt + 1))


def verify_pairs(tasks, *, model="deepseek-v4-flash", provider="deepseek",
                 concurrency=400, group_batch=16, api_key=None, base_url=None,
                 max_tokens=512, progress_label="llm-verify"):
    """Concurrent CoT verification of (query, candidate) pairs — the reusable core
    shared by this CLI and pool_inline_search's inline LLM-verify path.

    tasks: list of (qid, cid, q_text, c_text). Pairs are grouped by qid so each
    query's [few-shot+query] prefix is warmed once (first candidate) then its rest
    fired immediately → two-level prefix cache (few-shot cached the whole run).
    Returns (results, stats) where results[(qid,cid)] = (match_flag, pred_label):
      match_flag = 1.0 for HOST/INLINE else 0.0 (rule on the discrete label, NOT a
      score); pred_label in {0,1,2} or -1 on error."""
    assert api_key, "verify_pairs needs api_key"
    prov = PROVIDERS[provider]
    base_url = base_url or prov["base_url"]
    extra = prov["extra"]
    if concurrency > 200:
        try:
            threading.stack_size(1024 * 1024)
        except (ValueError, RuntimeError):
            pass
    import httpx
    from openai import OpenAI
    lim = concurrency + 64
    http_client = httpx.Client(
        limits=httpx.Limits(max_connections=lim, max_keepalive_connections=lim),
        timeout=httpx.Timeout(90.0, connect=20.0), trust_env=True)
    client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client, max_retries=0)

    results = {}
    tot = [0]; cached = [0]; n_err = [0]; done = [0]
    lock = threading.Lock()
    total = len(tasks)
    pbar = tqdm(total=total, desc=progress_label, dynamic_ncols=True)

    def work(t):
        qid, cid, q, c = t
        return qid, cid, classify(client, model, q, c, extra, max_tokens=max_tokens)

    def run_batch(batch):
        if not batch:
            return
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for f in as_completed([ex.submit(work, t) for t in batch]):
                qid, cid, res = f.result()
                if res is None or (isinstance(res, tuple) and res[0] == "ERR"):
                    results[(qid, cid)] = (0.0, -1)
                    with lock:
                        n_err[0] += 1
                else:
                    match, pred, tk, cc = res
                    results[(qid, cid)] = (float(match), int(pred))
                    with lock:
                        tot[0] += tk; cached[0] += cc
                with lock:
                    done[0] += 1
                    if done[0] % 50 == 0 or done[0] == total:
                        pbar.set_postfix(err=n_err[0],
                                         cached="%.0f%%" % (100.0 * cached[0] / max(tot[0], 1)),
                                         refresh=False)
                    pbar.update(1)

    from collections import defaultdict
    by_q = defaultdict(list)
    for t in tasks:
        by_q[t[0]].append(t)
    qids = sorted(by_q)
    B = max(1, group_batch)
    t0 = time.time()
    for s in range(0, len(qids), B):
        chunk = qids[s:s + B]
        run_batch([by_q[ri][0] for ri in chunk])
        run_batch([t for ri in chunk for t in by_q[ri][1:]])
    pbar.close()
    stats = {"n_pairs": total, "errors": n_err[0], "tokens": tot[0],
             "cached": cached[0], "elapsed_s": time.time() - t0,
             "cached_pct": 100.0 * cached[0] / max(tot[0], 1)}
    return results, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--provider", choices=list(PROVIDERS), default="deepseek")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--base-url", default=None, help="override provider base_url")
    ap.add_argument("--rerank-k", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--group-batch", type=int, default=16,
                    help="#queries processed per warm+rest cycle. Small => each query's "
                         "[few-shot+query] prefix is reused immediately (no cache eviction); "
                         "few-shot prefix stays globally cached the whole run regardless.")
    ap.add_argument("--limit-pairs", type=int, default=None)
    ap.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY") or os.getenv("DASHSCOPE_API_KEY"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.api_key, "set DEEPSEEK_API_KEY / DASHSCOPE_API_KEY or pass --api-key"

    d = read_pickle(args.prep)
    records = d["records"]
    K = min(args.rerank_k, d["topk"])

    tasks = []
    for ri, r in enumerate(records):
        for ci in range(min(K, len(r["cand_texts"]))):
            tasks.append((ri, ci, r["q_text"], r["cand_texts"][ci]))
            if args.limit_pairs and len(tasks) >= args.limit_pairs:
                break
        if args.limit_pairs and len(tasks) >= args.limit_pairs:
            break
    print("[api] %d records, %d pairs (top-%d), provider=%s model=%s, concurrency=%d"
          % (len(records), len(tasks), K, args.provider, args.model, args.concurrency), flush=True)

    results, stats = verify_pairs(
        tasks, model=args.model, provider=args.provider, concurrency=args.concurrency,
        group_batch=args.group_batch, api_key=args.api_key, base_url=args.base_url,
        progress_label="progress")

    llm = [np.full(min(K, len(r["cand_texts"])), np.nan, dtype=np.float32) for r in records]
    lab = [np.full(min(K, len(r["cand_texts"])), -1, dtype=np.int8) for r in records]
    for (ri, ci), (match, pred) in results.items():
        llm[ri][ci] = match
        lab[ri][ci] = pred
    dt = stats["elapsed_s"]; tot_tokens = [stats["tokens"]]
    cached_tokens = [stats["cached"]]; n_err = [stats["errors"]]

    write_pickle({"records": records, "llm": llm, "label": lab, "labels": LABELS,
                  "rerank_k": K, "n_pairs": len(tasks), "elapsed_s": dt,
                  "model": args.model, "total_tokens": tot_tokens[0],
                  "cached_tokens": cached_tokens[0], "n_err": n_err[0]},
                 args.out)
    from collections import Counter
    pc = Counter(int(x) for r in lab for x in r if x >= 0)
    print("\n[done] %d pairs in %.1fs = %.2f pairs/s | concurrency=%d | errors=%d | "
          "tokens=%d (cached=%d, %.0f%%) | pred=%s -> %s"
          % (len(tasks), dt, len(tasks) / max(dt, 1e-9), args.concurrency, n_err[0],
             tot_tokens[0], cached_tokens[0], 100.0 * cached_tokens[0] / max(tot_tokens[0], 1),
             {LABELS[k]: v for k, v in pc.items()}, args.out), flush=True)


if __name__ == "__main__":
    main()
