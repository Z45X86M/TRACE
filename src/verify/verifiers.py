"""Post-retrieval verifiers for pool_inline_search, kept out of the main file.

Two independent, default-OFF stages that run AFTER Stage-B dense rerank:

  1. Symbolic-anchor promote (non-LLM): over the WHOLE CAP, add a Dice-normalized
     idf-weighted shared-symbol (const / string / imported-call) bonus to each
     candidate's score and re-sort. Reads only the stripped query's own symbols vs
     pool candidates' symbols — no gt_*, no same-binary, no ground-truth-type signal.

  2. LLM verify + rerank (API-only): for each query's top-K shortlist, fetch
     query+candidate pseudocode and run the DeepSeek CoT verifier (llm_rerank_api),
     stable-promoting matched candidates. API-based only — no local model path.

Chaining (both on): anchor reshapes the full CAP inside search(); the LLM then
operates on the anchor-reranked top-K. Either can run alone.

This module must NOT import pool_inline_search (one-directional dependency).
"""
import json
import math
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from const_anchor import extract_consts
from utils import type_name


def norm_addr_key(a):
    """Canonical hashable address key: int when parseable as hex/dec, else str."""
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


def addr_hex(a):
    """Canonical address key string (hex) for the func-text index / lookups."""
    na = norm_addr_key(a)
    return hex(na) if isinstance(na, int) else str(na)


def func_text(func_obj):
    """Full decompiled text of a function (no truncation) from its feature entry."""
    parts = [(b.get("pseudos") or "") for b in func_obj.get("blocks_pseudocode", [])]
    return "\n".join(p for p in parts if p)


_ANCHOR_CALL_RE = re.compile(r"\b([A-Za-z_]\w{2,})\s*\(")
_ANCHOR_CALL_KW = {"if", "for", "while", "switch", "return", "sizeof", "else", "do",
                   "goto", "__fastcall", "__cdecl", "__int64", "unsigned", "int",
                   "char", "void"}

INCLUDE_CALL_CHANNEL = os.environ.get("ANCHOR_DROP_CALL", "0") not in ("1", "true", "True", "yes", "on")


def anchor_func_anchors(func_obj):
    """(CONST_NUM, STRING_LIT, CALL) three frozensets.

    Channel 0 is now decimal+hex distinctive constants via const_anchor.extract_consts
    (canonicalizes top-window unsigned wraparound, drops common magnitudes), read
    straight from the pseudocode. This supersedes the hex-only stored CONST_HEX:
    decimal magic numbers (57600, mod primes, table sizes) are equally invariant under
    inline / optimization / cross-arch, and the hex magics are still captured (the
    extractor reads dec+hex alike, dropping only ubiquitous values like 0xff/255).
    STRING_LIT reads the stored anchors; CALL is regexed from pseudos (drop keywords /
    sub_ stubs / own name) — imported call names survive stripping."""
    consts, strs, calls = set(), set(), set()
    own = func_obj.get("func_name", "")
    for b in func_obj.get("blocks_pseudocode", []):
        anc = b.get("anchors") or {}
        strs.update(anc.get("STRING_LIT") or [])
        ps = b.get("pseudos") or ""
        consts |= extract_consts(ps)
        for m in _ANCHOR_CALL_RE.findall(ps):
            if (m not in _ANCHOR_CALL_KW
                    and not m.startswith(("sub_", "loc_", "nullsub", "j_"))
                    and m != own):
                calls.add(m)
    if not INCLUDE_CALL_CHANNEL:
        calls = set()
    return frozenset(consts), frozenset(strs), frozenset(calls)



def delimit_matched_blocks(sim, tau=0.5, floor=0.20):
    """From a query×candidate slice-similarity matrix, return the participating block index sets on
    each side: (S_Q, S_C).

      sim[i, j] = cos(query_slice_i, candidate_slice_j)   shape [R x P]
      q_best[i] = max_j sim[i, j]   (per query row, best candidate slice)
      p_best[j] = max_i sim[i, j]   (per candidate slice, best query row)
      S_Q = { i : q_best[i] >= tau * max_i q_best  AND q_best[i] > floor }   query blocks in the match
      S_C = { j : p_best[j] >= tau * max_j p_best  AND p_best[j] > floor }   candidate blocks in the match

    Relative cutoff `tau` so it adapts to root (whole-host) vs leaf (small inlined fragment)
    without knowing the ground-truth type; absolute `floor` drops near-zero spurious matches. Uses only the
    embedding similarity matrix — no GT / no type / no debug signal. Returns two int index arrays."""
    sim = np.clip(np.asarray(sim, dtype=np.float32), 0.0, 1.0)
    if sim.ndim != 2 or sim.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    q_best = sim.max(axis=1)
    p_best = sim.max(axis=0)
    s_q = np.where((q_best >= tau * q_best.max()) & (q_best > floor))[0]
    s_c = np.where((p_best >= tau * p_best.max()) & (p_best > floor))[0]
    return s_q, s_c


def build_anchor_pool_map(feat_dir, suffix="_bb_slice_feature.json"):
    """(binary_name, norm_addr) -> (hex, str, call) over every func in the pool dir."""
    amap = {}
    files = [f for f in sorted(os.listdir(feat_dir)) if f.endswith(suffix)]
    for f in tqdm(files, desc="anchor: pool map", dynamic_ncols=True):
        binstem = f[: -len(suffix)]
        try:
            d = json.load(open(os.path.join(feat_dir, f)))
        except Exception:
            continue
        for k, v in d.items():
            amap[(binstem, norm_addr_key(v.get("func_addr", k)))] = anchor_func_anchors(v)
    return amap


def build_anchor_idf(amap):
    """token -> idf over pool funcs, separate hex/str/call namespaces (rarer => higher)."""
    N = len(amap) or 1
    df = [defaultdict(int), defaultdict(int), defaultdict(int)]
    for anchs in amap.values():
        for j in range(3):
            for t in anchs[j]:
                df[j][t] += 1
    return [{t: math.log((N + 1.0) / (c + 1.0)) for t, c in df[j].items()} for j in range(3)]


def anchor_overlap(qa, ca, idf):
    """IDF-weighted sum over symbols shared between query qa and candidate ca.
    anchor_overlap(a, a, idf) gives a's own idf mass (used for Dice normalization)."""
    tot = 0.0
    for j in range(3):
        sh = qa[j] & ca[j]
        if sh:
            tot += sum(idf[j].get(t, 0.0) for t in sh)
    return tot


def default_anchor_index_path(feat_dir, suffix="_bb_slice_feature.json"):
    """Default location of a prebuilt anchor index for a pool dir."""
    tag = suffix[:-len(".json")] if suffix.endswith(".json") else suffix
    tag = tag.strip("_") or "anchor"
    return os.path.join(feat_dir, "anchor_index_%s.pkl" % tag)


def save_anchor_index(path, pool, idf, pmass, suffix):
    """Serialize (pool anchor map, idf, per-candidate Dice mass) so retrieval can skip the
    online pool scan. pool: {(binary,addr)->(hex,str,call)}; idf: [dict]*3; pmass: {key->float}."""
    import pickle
    with open(path, "wb") as f:
        pickle.dump({"pool": pool, "idf": idf, "pmass": pmass, "suffix": suffix, "n": len(pool)},
                    f, protocol=pickle.HIGHEST_PROTOCOL)


def load_anchor_index(path):
    """Load a prebuilt anchor index -> (pool, idf, pmass). Extra keys (e.g. a legacy 'ctrl' map
    from the removed CONTROL experiment) are ignored, so old indexes still load fine."""
    import pickle
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["pool"], d["idf"], d["pmass"]


def build_anchor_resources(feat_dir, suffix="_bb_slice_feature.json", index_path=None, save=True):
    """Pool anchor map + idf + per-candidate idf mass (for Dice). If index_path exists, load the
    prebuilt index and skip the (slow) pool scan; if index_path is given but missing and save=True,
    build it once and write it there for next time."""
    if index_path and os.path.exists(index_path):
        return load_anchor_index(index_path)
    pool = build_anchor_pool_map(feat_dir, suffix)
    idf = build_anchor_idf(pool)
    pmass = {key: anchor_overlap(a, a, idf) for key, a in pool.items()}
    if index_path and save:
        try:
            save_anchor_index(index_path, pool, idf, pmass, suffix)
        except Exception:
            pass
    return pool, idf, pmass


def load_query_anchors(query_dirs, binary, addr, suffix, cache):
    """Query (inline-host, strip build) anchors. First listed dir that contains the
    binary's feature json wins per-addr (strip preferred, noinline fallback). `cache`
    is a caller-owned dict {binary -> {norm_addr -> anchors}}."""
    if binary not in cache:
        m: Dict[Any, Tuple] = {}
        for d in query_dirs:
            p = os.path.join(d, str(binary) + suffix)
            if not os.path.exists(p):
                continue
            try:
                jd = json.load(open(p))
            except Exception:
                continue
            for k, v in jd.items():
                key = norm_addr_key(v.get("func_addr", k))
                if key not in m:
                    m[key] = anchor_func_anchors(v)
        cache[binary] = m
    return cache[binary].get(norm_addr_key(addr), (frozenset(), frozenset(), frozenset()))


def apply_anchor_promote(ranked, qa, idf, pool, pmass, lam):
    """Promote-only over the full candidate list: score' = score + lam*Dice, re-sort.
    Dice = 2*ov/(q_mass + cand_mass) normalizes by the candidate's own anchor mass so
    large distractors that share many common constants can't override the root GT."""
    q_mass = anchor_overlap(qa, qa, idf)
    empty = (frozenset(), frozenset(), frozenset())
    for it in ranked:
        key = (it["binary_name"], norm_addr_key(it["func_addr"]))
        ca = pool.get(key, empty)
        ov = anchor_overlap(qa, ca, idf)
        cand_mass = pmass.get(key, 0.0)
        dice = (2.0 * ov / (q_mass + cand_mass + 1e-6)) if ov > 0.0 else 0.0
        it["anchor_overlap"] = float(ov)
        it["anchor_dice"] = float(dice)
        it["score"] = float(it.get("score", 0.0)) + lam * dice
    ranked.sort(key=lambda it: (
        -float(it["score"]), -float(it.get("slice_channel", 0.0)),
        -float(it.get("global_sim", 0.0)), -float(it.get("slice_score", 0.0)),
        it["binary_name"], str(it["func_addr"]),
    ))
    return ranked


def default_functext_sqlite_path(src_dir):
    return os.path.join(src_dir, "pool_func_text.sqlite")


def read_functext_file(args):
    """Worker (picklable, module-level): parse one feature JSON -> list of
    (binary, addr_hex, text) rows. Heavy JSON parse runs in a child process."""
    path, suffix = args
    stem = os.path.basename(path)[: -len(suffix)]
    try:
        d = json.load(open(path))
    except Exception:
        return []
    return [(stem, addr_hex(v.get("func_addr", k)), func_text(v)) for k, v in d.items()]


def build_func_text_index(src_dir, feature_suffix="_bb_slice_feature.json",
                          sqlite_path=None, workers=8, force=False):
    """Build an on-disk (binary, addr)->pseudocode sqlite index DIRECTLY from a pool
    source dir (e.g. data/inline/pool) — NO runtime rebuild needed. The LLM verifier
    then fetches candidate text BY KEY (primary-key index), without re-parsing JSON
    or holding pool text in RAM. Feature JSONs are parsed across `workers` PROCESSES
    (real parallelism; JSON parse is CPU-bound) while the parent streams inserts.
    Idempotent: skips if the index exists (unless force). Returns the sqlite path."""
    import sqlite3
    from concurrent.futures import ProcessPoolExecutor
    path = sqlite_path or default_functext_sqlite_path(src_dir)
    if os.path.exists(path) and not force:
        return path
    files = [os.path.join(src_dir, f) for f in sorted(os.listdir(src_dir))
             if f.endswith(feature_suffix)]
    if not files:
        raise SystemExit("no %s under %s" % (feature_suffix, src_dir))
    tmp = path + ".building"
    if os.path.exists(tmp):
        os.remove(tmp)
    con = sqlite3.connect(tmp)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("CREATE TABLE functext (binary TEXT, addr TEXT, text TEXT, "
                "PRIMARY KEY(binary, addr)) WITHOUT ROWID")
    t0 = time.time(); n = 0
    args_list = [(f, feature_suffix) for f in files]
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as ex:
        for rows in tqdm(ex.map(read_functext_file, args_list, chunksize=2),
                         total=len(files), desc="build func-text index", dynamic_ncols=True):
            if rows:
                con.executemany("INSERT OR REPLACE INTO functext VALUES (?,?,?)", rows)
                n += len(rows)
    con.commit(); con.close()
    os.replace(tmp, path)
    print("[func-text-index] %d funcs from %d files (%d workers) in %.0fs -> %s"
          % (n, len(files), workers, time.time() - t0, path), flush=True)
    return path


def fetch_texts_sqlite(sqlite_path, keys):
    """keys: iterable of (binary, addr_hex). Returns {(binary, addr_hex): text} for
    the keys present. Opens the index READ-ONLY and does one primary-key lookup per
    key — `functext` is WITHOUT ROWID, so (binary, addr) IS the clustered index →
    O(log n) random access; only the needed rows are touched (no scan, no RAM load)."""
    import sqlite3
    keys = list(keys)
    con = sqlite3.connect("file:%s?mode=ro" % sqlite_path, uri=True)
    cur = con.cursor()
    out = {}
    for b, a in tqdm(keys, desc="fetch cand text (sqlite)", dynamic_ncols=True):
        r = cur.execute("SELECT text FROM functext WHERE binary=? AND addr=?", (b, a)).fetchone()
        if r is not None:
            out[(b, a)] = r[0]
    con.close()
    return out


def build_texts_for_keys(needed_by_binary, feat_dir, suffix, workers=12):
    """needed_by_binary: {binary_name -> set(norm_addr)}. Read ONLY those binaries'
    feature JSONs (in parallel), extract ONLY the needed funcs' text. Fast: touches
    just the binaries that actually surfaced in some query's top-K, not the pool."""
    out: Dict[Tuple[str, Any], str] = {}
    from concurrent.futures import ThreadPoolExecutor, as_completed as _ac

    def _one(binary):
        path = os.path.join(feat_dir, binary + suffix)
        if not os.path.exists(path):
            return binary, {}
        try:
            d = json.load(open(path))
        except Exception:
            return binary, {}
        need = needed_by_binary[binary]
        sub = {}
        for k, v in d.items():
            na = norm_addr_key(v.get("func_addr", k))
            if na in need:
                sub[na] = func_text(v)
        return binary, sub

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in _ac([ex.submit(_one, b) for b in needed_by_binary]):
            binary, sub = fut.result()
            for na, txt in sub.items():
                out[(binary, na)] = txt
    return out


def llm_verify_and_rerank(collected, *, text_index_path, index_src_dir, strip_feat_dir,
                          feature_suffix, recall_ks, model, concurrency, api_key, topk,
                          index_workers=8):
    """Optional LLM-verify path: for each query's top-K shortlist (already gathered
    by evaluate_metrics — and already anchor-reranked when --anchor is on), fetch
    query+candidate pseudocode, run the DeepSeek CoT verifier, STABLE-promote matched
    candidates to the front (preserving prior order), and recompute recall@k. Baseline
    output untouched.

    Candidate text: random-access by KEY from the on-disk sqlite func-text index
    (built once from `index_src_dir`, e.g. data/inline/pool — NO pool/runtime
    rebuild, NOT held in RAM). Query text: prefer the in-record `q_text` (stored at
    query-build time, in memory); fall back to reading only the query binaries'
    feature JSONs when older query pkls lack it. Returns (llm_metrics, llm_misses, stats).

    API-ONLY: the actual model call is llm_rerank_api.verify_pairs (DeepSeek). There is
    no local-model path."""
    t_build = time.time()

    if not os.path.exists(text_index_path):
        print("[llm-verify] func-text index missing -> building from %s (%d workers)"
              % (index_src_dir, index_workers), flush=True)
        build_func_text_index(index_src_dir, feature_suffix, sqlite_path=text_index_path,
                              workers=index_workers)
    cand_keys = set()
    for cr in collected:
        for (b, a, _is_gt) in cr["topk"]:
            cand_keys.add((b, addr_hex(a)))
    cand_text = fetch_texts_sqlite(text_index_path, cand_keys)

    q_text = {}
    q_need = defaultdict(set)
    for cr in collected:
        if cr.get("q_text"):
            q_text[(cr["q_binary"], addr_hex(cr["q_addr"]))] = cr["q_text"]
        elif cr["q_binary"] is not None and cr["q_addr"] is not None:
            q_need[cr["q_binary"]].add(norm_addr_key(cr["q_addr"]))
    if q_need:
        fallback = build_texts_for_keys(q_need, strip_feat_dir, feature_suffix)
        for (b, na), txt in fallback.items():
            q_text[(b, addr_hex(na))] = txt
    print("[llm-verify] texts: %d cand (of %d keys) | %d query (%d from record, %d read) in %.1fs"
          % (len(cand_text), len(cand_keys), len(q_text),
             sum(1 for cr in collected if cr.get("q_text")), len(q_need), time.time() - t_build), flush=True)

    tasks = []
    for qi, cr in enumerate(collected):
        qt_txt = q_text.get((cr["q_binary"], addr_hex(cr["q_addr"])), "")
        if not qt_txt:
            continue
        for ci, (b, a, _is_gt) in enumerate(cr["topk"]):
            ct = cand_text.get((b, addr_hex(a)), "")
            if ct:
                tasks.append((qi, ci, qt_txt, ct))
    print("[llm-verify] %d queries, %d pairs (top-%d), model=%s, concurrency=%d"
          % (len(collected), len(tasks), topk, model, concurrency), flush=True)

    from llm_rerank_api import verify_pairs
    results, stats = verify_pairs(tasks, model=model, provider="deepseek",
                                  concurrency=concurrency, api_key=api_key,
                                  progress_label="llm-verify")
    print("[llm-verify] done: %d pairs in %.0fs | errors=%d | cached=%.0f%%"
          % (stats["n_pairs"], stats["elapsed_s"], stats["errors"], stats["cached_pct"]), flush=True)

    recall_ks = tuple(sorted(set(int(k) for k in recall_ks)))
    total = len(collected)
    recall_hits = {k: 0 for k in recall_ks}
    type_counts = {0: 0, 1: 0, 2: 0}
    type_hits = {t: {k: 0 for k in recall_ks} for t in type_counts}
    llm_misses = []
    for qi, cr in enumerate(collected):
        qt = cr["qt"]; type_counts[qt] = type_counts.get(qt, 0) + 1
        n = len(cr["topk"])
        matched = [results.get((qi, ci), (0.0, -1))[0] == 1.0 for ci in range(n)]
        order = sorted(range(n), key=lambda ci: (0 if matched[ci] else 1, ci))
        new_gt_rank = None
        for pos, ci in enumerate(order):
            if cr["topk"][ci][2]:
                new_gt_rank = pos + 1
                break
        if new_gt_rank is None:
            new_gt_rank = cr["gt_rank"]
        for k in recall_ks:
            if new_gt_rank is not None and new_gt_rank <= k:
                recall_hits[k] += 1
                type_hits[qt][k] += 1
        if new_gt_rank is None or new_gt_rank > max(recall_ks):
            llm_misses.append({"gt_type": type_name(qt), "gt_rank_llm": new_gt_rank,
                               "gt_rank_baseline": cr["gt_rank"]})

    m: Dict[str, float] = {"queries": float(total)}
    if total:
        for k in recall_ks:
            m[f"recall@{k}"] = recall_hits[k] / total
        for qt, count in type_counts.items():
            m[f"{type_name(qt)}_queries"] = float(count)
            for k in recall_ks:
                m[f"{type_name(qt)}_recall@{k}"] = type_hits[qt][k] / count if count else 0.0
    m["llm_pairs"] = int(stats["n_pairs"])
    m["llm_errors"] = int(stats["errors"])
    m["llm_cached_pct"] = round(stats["cached_pct"], 1)
    m["llm_seconds"] = round(stats["elapsed_s"], 1)
    return m, llm_misses, stats
