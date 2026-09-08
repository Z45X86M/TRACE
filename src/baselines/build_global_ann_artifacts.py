"""Step 2 of the global-ANN pipeline: assemble the searchable vector DB.

Two responsibilities, both wired by default:

  (a) **Swap globals into the per-experiment slice pool/query**:
      Read ``<binary>{POOL_EMB_SUFFIX}`` from ``--source-pool-dir`` and
      replace each per-addr ``global`` field with the encoder-of-record's
      vector from ``<binary>{GLOBAL_EMB_SUFFIX}`` under ``--pool-global-source-dir``
      (default: ``data/inline/pool``). Likewise for the query side
      (``query_record['gq']``). Output written to ``--output-pool-dir`` /
      ``--output-query-dir``.

  (b) **Populate the FAISS DB** for global ANN retrieval:
      Load the centroids-only index from ``--centroids-index`` (produced by
      step 1, ``train_faiss_global_ann.py``), gather every pool global vector
      from ``--output-pool-dir``, and ``index.add(...)`` them. Persist the
      populated index to ``--output-index`` along with ``.meta.json``
      mapping FAISS positions → (binary, addr, func_name) — this is what
      ``global_ann_inline_eval.py --model`` reads.

Run order::

    Step 1 (once per source-pool distribution change):
        python3 Coding/train_faiss_global_ann.py

    Step 2 (per experiment / per pool variant):
        python3 Coding/build_global_ann_artifacts.py

    Step 3 (per experiment):
        python3 Coding/global_ann_inline_eval.py

All defaults are wired — naked CLI invocations work for the active default
(noinline) experiment. Re-target by overriding ``--source-pool-dir`` /
``--source-query-dir`` / ``--output-index``.
"""

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
for path in (str(ROOT_DIR), str(CURRENT_DIR), str(ROOT_DIR / "V3")):
    if path not in sys.path:
        sys.path.append(path)

from utils import read_json, read_pickle, write_pickle
from extract_global_embedding_from_feature import (
    DEFAULT_BATCH_SIZE as ON_DEMAND_DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_SEQ_LENGTH as ON_DEMAND_DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_PATH as ON_DEMAND_DEFAULT_MODEL_PATH,
    GlobalEncoder,
    _pick_source_feature,
    extract_global_text,
    mean_pooling,
)

POOL_EMB_SUFFIX = "_bb_slice_embeddings.pkl"
POOL_FEATURE_SUFFIX = "_bb_slice_feature.json"
GLOBAL_EMB_SUFFIX = "_feature_global_embeddings.pkl"
POOL_MANIFEST_NAME = "pool_manifest.json"


def derive_pool_embed_suffix(feature_suffix: str) -> str:
    """Mirror of extract_embedding_inline_slice.derive_embedding_suffix; kept
    local to avoid an extra import in this script's typical CLI use."""
    tail = "_feature.json"
    if feature_suffix.endswith(tail):
        return feature_suffix[: -len(tail)] + "_embeddings.pkl"
    base = feature_suffix
    if base.endswith(".json"):
        base = base[: -len(".json")]
    return base + ".embeddings.pkl"


def _load_global_cache(
    source_dir: Path,
    binary_name: str,
    global_suffix: str = GLOBAL_EMB_SUFFIX,
) -> Optional[Dict[str, list]]:
    """Read the per-binary global-embedding cache. Returns ``None`` if the
    file is missing OR unreadable — corrupt pkls (truncated, ``\\x00`` header,
    etc.) are downgraded to a cache miss so the on-demand encoder can rebuild
    them; ``_persist_subset_cache`` then atomically overwrites the bad file.
    """
    path = source_dir / f"{binary_name}{global_suffix}"
    if not path.exists():
        return None
    try:
        pkl = read_pickle(str(path))
    except (pickle.UnpicklingError, EOFError, ValueError, OSError, AttributeError) as e:
        tqdm.write(
            f"[global-cache] CORRUPT {path.name} ({type(e).__name__}: {e}); "
            "treating as cache miss — will re-encode + overwrite."
        )
        return None
    if not isinstance(pkl, dict):
        tqdm.write(f"[global-cache] UNEXPECTED type for {path.name} (got {type(pkl).__name__}); "
                   "treating as cache miss.")
        return None
    out: Dict[str, list] = {}
    for addr, item in pkl.items():
        vec = item.get("global") if isinstance(item, dict) else None
        if vec is not None:
            out[addr] = vec
    return out


class _OnDemandEncoder:
    """Lazy holder for the global encoder used by on-demand subset encoding.

    Loading the encoder costs a few seconds + GPU memory; we only want to pay
    that once per process and only if at least one binary actually needs it.
    """

    def __init__(self, model_path: str, gpu_id: int, max_seq_length: int, batch_size: int):
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self._encoder: Optional[GlobalEncoder] = None

    def get(self) -> GlobalEncoder:
        if self._encoder is None:
            print(
                f"[on-demand] loading encoder model={self.model_path} gpu={self.gpu_id} "
                f"max_len={self.max_seq_length} batch={self.batch_size}",
                flush=True,
            )
            self._encoder = GlobalEncoder(
                self.model_path, self.gpu_id, self.max_seq_length, self.batch_size,
            )
        return self._encoder


def _locate_feature_path(binary_name: str, search_dirs: List[Path]) -> Optional[str]:
    for d in search_dirs:
        candidate, _ = _pick_source_feature(str(d / binary_name))
        if candidate is not None:
            return candidate
    return None


def _persist_subset_cache(
    binary_name: str,
    addr_to_emb: Dict[str, list],
    persist_dir: Path,
    global_suffix: str = GLOBAL_EMB_SUFFIX,
) -> None:
    """Merge a newly-encoded {addr: emb} into <binary>{global_suffix}."""
    out_path = persist_dir / f"{binary_name}{global_suffix}"
    existing: Dict[str, dict] = {}
    if out_path.exists():
        try:
            loaded = read_pickle(str(out_path))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}
    for addr, emb in addr_to_emb.items():
        existing[addr] = {"global": emb}
    write_pickle(existing, str(out_path))
    print(f"[on-demand] cached -> {out_path} ({len(addr_to_emb)} addrs)", flush=True)


def _encode_one_batch(encoder: GlobalEncoder, texts: List[str]) -> List[list]:
    """Run a single forward pass over ``texts`` — no internal tqdm, no batching.

    Caller is responsible for keeping ``len(texts) <= encoder.batch_size``.
    """
    device = torch.device(f"cuda:{encoder.gpu_id}")
    res = encoder.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=encoder.max_seq_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        output = encoder.model(res["input_ids"].long(), attention_mask=res["attention_mask"].int())
        emb = mean_pooling(output, res["attention_mask"].int())
        emb = F.normalize(emb, p=2, dim=1)
    return emb.detach().cpu().numpy().tolist()


def _batch_encode_subset(
    on_demand_specs: List[Tuple[str, List[str], List[Path]]],
    encoder_holder: "_OnDemandEncoder",
    persist_dir: Optional[Path],
    desc_tag: str,
    global_suffix: str = GLOBAL_EMB_SUFFIX,
) -> Dict[str, Dict[str, list]]:
    """Stream-encode globals for ``(binary, addrs, feature_search_dirs)`` specs.

    Reads each binary's feature.json lazily, accumulates ``(binary, addr, text)``
    until ``batch_size`` is reached, then immediately fires a single encoder
    forward pass. A binary's persistence to ``persist_dir/<binary>{global_suffix}``
    happens as soon as all its addrs are encoded — partial progress is durable
    even if the process is later killed.

    Progress is reported via a single tqdm bar whose total is the *upper bound*
    of addrs to encode (= sum of ``needed_addrs`` across specs). The bar's
    postfix shows the current binary so the user can see which one is being
    processed right now.
    """
    located: List[Tuple[str, str, List[str]]] = []
    upper_bound = 0
    for binary_name, needed_addrs, search_dirs in on_demand_specs:
        if not needed_addrs:
            continue
        source_path = _locate_feature_path(binary_name, search_dirs)
        if source_path is None:
            continue
        located.append((binary_name, source_path, needed_addrs))
        upper_bound += len(needed_addrs)

    if not located:
        return {}

    print(
        f"[on-demand] {desc_tag}: stream-encoding up to {upper_bound} addrs across "
        f"{len(located)} binaries (batch={encoder_holder.batch_size})",
        flush=True,
    )

    encoder = encoder_holder.get()
    batch_size = max(1, encoder.batch_size)

    per_binary: Dict[str, Dict[str, list]] = {}
    binary_pending: Dict[str, int] = {}
    persisted: set = set()
    pending: List[Tuple[str, str, str]] = []

    pbar = tqdm(
        total=upper_bound,
        desc=f"on-demand-{desc_tag}",
        unit="func",
        dynamic_ncols=True,
        mininterval=0.5,
    )

    def flush() -> None:
        if not pending:
            return
        texts = [t for _, _, t in pending]
        embs = _encode_one_batch(encoder, texts)
        for (bn, addr, _), emb in zip(pending, embs):
            per_binary.setdefault(bn, {})[addr] = emb
            if bn in binary_pending:
                binary_pending[bn] -= 1
        pbar.update(len(pending))
        if persist_dir is not None:
            for bn, remaining in list(binary_pending.items()):
                if remaining <= 0 and bn not in persisted:
                    _persist_subset_cache(bn, per_binary[bn], persist_dir, global_suffix)
                    persisted.add(bn)
                    del binary_pending[bn]
        pending.clear()

    for binary_name, source_path, needed_addrs in located:
        pbar.set_postfix_str(f"binary={binary_name}", refresh=True)
        feature_dict = read_json(source_path)
        addrs_in_feature = [a for a in needed_addrs if a in feature_dict]
        if not addrs_in_feature:
            pbar.total = max(0, pbar.total - len(needed_addrs))
            pbar.refresh()
            continue
        missing = len(needed_addrs) - len(addrs_in_feature)
        if missing:
            pbar.total = max(0, pbar.total - missing)
            pbar.refresh()
        binary_pending[binary_name] = len(addrs_in_feature)
        for addr in addrs_in_feature:
            pending.append((binary_name, addr, extract_global_text(feature_dict[addr])))
            if len(pending) >= batch_size:
                flush()

    flush()
    pbar.close()

    if persist_dir is not None:
        for bn, addr_to_emb in per_binary.items():
            if bn not in persisted:
                _persist_subset_cache(bn, addr_to_emb, persist_dir, global_suffix)
                persisted.add(bn)

    return per_binary


def _replicate_sidecar(src_path: Path, dst_path: Path) -> None:
    if dst_path.is_symlink() or dst_path.exists():
        try:
            dst_path.unlink()
        except IsADirectoryError:
            shutil.rmtree(str(dst_path))
    if src_path.is_symlink():
        target = os.readlink(str(src_path))
        os.symlink(target, str(dst_path))
    else:
        shutil.copy2(str(src_path), str(dst_path))


def _wipe_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(str(path))
    path.mkdir(parents=True, exist_ok=True)


def _stream_encode_one_binary(
    binary_name: str,
    needed_addrs: List[str],
    feature_search_dirs: List[Path],
    encoder_holder: "_OnDemandEncoder",
    persist_dir: Optional[Path],
    global_suffix: str = GLOBAL_EMB_SUFFIX,
    outer_pbar: Optional[tqdm] = None,
    tag: str = "pool",
) -> Optional[Dict[str, list]]:
    """Stream-encode globals for ONE binary's needed addrs.

    Reads the binary's feature.json, fills a pending buffer; whenever the
    buffer hits ``encoder.batch_size`` we fire a forward pass immediately.
    Persistence happens once at the end of the binary (atomic-ish: a Ctrl-C
    mid-binary loses only the in-flight buffer, never partially writes the
    on-disk cache pkl).
    """
    source_path = _locate_feature_path(binary_name, feature_search_dirs)
    if source_path is None:
        return None
    feature_dict = read_json(source_path)
    addrs_in_feature = [a for a in needed_addrs if a in feature_dict]
    if not addrs_in_feature:
        return None

    encoder = encoder_holder.get()
    batch_size = max(1, encoder.batch_size)

    result: Dict[str, list] = {}
    pending: List[Tuple[str, str]] = []

    def flush_buf() -> None:
        if not pending:
            return
        embs = _encode_one_batch(encoder, [t for _, t in pending])
        for (addr, _), emb in zip(pending, embs):
            result[addr] = emb
        pending.clear()

    total = len(addrs_in_feature)
    if outer_pbar is not None:
        outer_pbar.set_postfix_str(f"binary={binary_name} encode 0/{total}", refresh=True)

    for i, addr in enumerate(addrs_in_feature, start=1):
        pending.append((addr, extract_global_text(feature_dict[addr])))
        if len(pending) >= batch_size:
            flush_buf()
            if outer_pbar is not None:
                outer_pbar.set_postfix_str(
                    f"binary={binary_name} encode {len(result)}/{total}", refresh=True
                )
    flush_buf()
    if outer_pbar is not None:
        outer_pbar.set_postfix_str(
            f"binary={binary_name} encoded {len(result)}/{total}", refresh=True
        )

    if persist_dir is not None and result:
        _persist_subset_cache(binary_name, result, persist_dir, global_suffix)
    return result


def patch_pool(
    source_pool_dir: Path,
    global_source_dir: Path,
    output_pool_dir: Path,
    global_suffix: str = GLOBAL_EMB_SUFFIX,
    pool_feature_suffix: str = POOL_FEATURE_SUFFIX,
    pool_embed_suffix: str = POOL_EMB_SUFFIX,
    encoder_holder: Optional["_OnDemandEncoder"] = None,
    persist_on_demand_cache: bool = True,
) -> dict:
    _wipe_output_dir(output_pool_dir)
    stats = {
        "pool_binaries": 0,
        "pool_binaries_with_cache": 0,
        "pool_binaries_missing_cache": 0,
        "pool_binaries_on_demand_encoded": 0,
        "pool_addrs_total": 0,
        "pool_addrs_replaced": 0,
        "pool_addrs_missing_in_cache": 0,
    }

    emb_paths = sorted(source_pool_dir.glob(f"*{pool_embed_suffix}"))
    pbar = tqdm(emb_paths, desc="pool", unit="bin", dynamic_ncols=True, mininterval=0.3)
    for emb_path in pbar:
        binary_name = emb_path.name[: -len(pool_embed_suffix)]
        stats["pool_binaries"] += 1
        pbar.set_postfix_str(f"binary={binary_name}", refresh=True)
        try:
            emb_dict = read_pickle(str(emb_path))
        except (pickle.UnpicklingError, EOFError, ValueError, OSError) as e:
            tqdm.write(
                f"[pool] CORRUPT slice embeddings {emb_path.name} ({type(e).__name__}: {e}); "
                "skipping binary — re-run extract_embedding_inline_slice.py for it."
            )
            stats["pool_binaries_missing_cache"] += 1
            continue

        cache = _load_global_cache(global_source_dir, binary_name, global_suffix)
        if cache is None and encoder_holder is not None:
            needed = [a for a, item in emb_dict.items() if isinstance(item, dict)]
            cache = _stream_encode_one_binary(
                binary_name,
                needed,
                feature_search_dirs=[global_source_dir, source_pool_dir],
                encoder_holder=encoder_holder,
                persist_dir=global_source_dir if persist_on_demand_cache else None,
                global_suffix=global_suffix,
                outer_pbar=pbar,
                tag="pool",
            )
            if cache is not None:
                stats["pool_binaries_on_demand_encoded"] += 1

        if cache is None:
            stats["pool_binaries_missing_cache"] += 1
            tqdm.write(f"[pool] WARN missing global cache for {binary_name} — keeping original globals")
        else:
            stats["pool_binaries_with_cache"] += 1
            for addr, item in emb_dict.items():
                stats["pool_addrs_total"] += 1
                if not isinstance(item, dict):
                    continue
                new_vec = cache.get(addr)
                if new_vec is None:
                    stats["pool_addrs_missing_in_cache"] += 1
                    continue
                item["global"] = new_vec
                stats["pool_addrs_replaced"] += 1

        write_pickle(emb_dict, str(output_pool_dir / emb_path.name))

        feature_src = emb_path.with_name(f"{binary_name}{pool_feature_suffix}")
        if feature_src.exists():
            _replicate_sidecar(feature_src, output_pool_dir / feature_src.name)

        binary_src = emb_path.with_name(binary_name)
        if binary_src.exists() or binary_src.is_symlink():
            _replicate_sidecar(binary_src, output_pool_dir / binary_name)

        del emb_dict
    pbar.close()

    manifest_src = source_pool_dir / POOL_MANIFEST_NAME
    if manifest_src.exists():
        shutil.copy2(str(manifest_src), str(output_pool_dir / POOL_MANIFEST_NAME))

    return stats


def patch_query(
    source_query_dir: Path,
    global_source_dir: Path,
    output_query_dir: Path,
    global_suffix: str = GLOBAL_EMB_SUFFIX,
    encoder_holder: Optional["_OnDemandEncoder"] = None,
    persist_on_demand_cache: bool = True,
) -> dict:
    _wipe_output_dir(output_query_dir)
    stats = {
        "query_files": 0,
        "query_records_total": 0,
        "query_gq_replaced": 0,
        "query_missing_binary_cache": 0,
        "query_missing_addr_in_cache": 0,
        "query_missing_fields": 0,
        "query_binaries_on_demand_encoded": 0,
    }
    all_records: List[Tuple[Path, list]] = []
    binary_to_addrs: Dict[str, List[str]] = {}
    binary_addr_seen: Dict[str, set] = {}
    for qpath in sorted(source_query_dir.glob("*_query*.pkl")):
        records = read_pickle(str(qpath))
        all_records.append((qpath, records))
        if not isinstance(records, list):
            continue
        for rec in records:
            if not isinstance(rec, dict):
                continue
            binary_name = rec.get("binary")
            addr = rec.get("query_addr")
            if binary_name is None or addr is None:
                continue
            seen = binary_addr_seen.setdefault(binary_name, set())
            if addr in seen:
                continue
            seen.add(addr)
            binary_to_addrs.setdefault(binary_name, []).append(addr)

    caches: Dict[str, Dict[str, list]] = {}
    cache_misses: set = set()
    missing_binaries: List[str] = []
    for binary_name in binary_to_addrs:
        loaded = _load_global_cache(global_source_dir, binary_name, global_suffix)
        if loaded is not None:
            caches[binary_name] = loaded
        else:
            missing_binaries.append(binary_name)

    if missing_binaries and encoder_holder is not None:
        specs = [
            (binary_name, binary_to_addrs[binary_name], [global_source_dir, source_query_dir])
            for binary_name in missing_binaries
        ]
        encoded = _batch_encode_subset(
            specs,
            encoder_holder=encoder_holder,
            persist_dir=global_source_dir if persist_on_demand_cache else None,
            desc_tag="query",
            global_suffix=global_suffix,
        )
        for binary_name, sub in encoded.items():
            caches[binary_name] = sub
            stats["query_binaries_on_demand_encoded"] += 1

    for binary_name in missing_binaries:
        if binary_name not in caches:
            cache_misses.add(binary_name)
            print(f"[query] WARN missing global cache for {binary_name}")

    for qpath, records in all_records:
        stats["query_files"] += 1
        if not isinstance(records, list):
            write_pickle(records, str(output_query_dir / qpath.name))
            continue
        for rec in records:
            stats["query_records_total"] += 1
            if not isinstance(rec, dict):
                continue
            binary_name = rec.get("binary")
            addr = rec.get("query_addr")
            if binary_name is None or addr is None:
                stats["query_missing_fields"] += 1
                continue
            if binary_name in cache_misses:
                stats["query_missing_binary_cache"] += 1
                continue
            cache = caches.get(binary_name)
            if cache is None:
                stats["query_missing_binary_cache"] += 1
                continue
            new_vec = cache.get(addr)
            if new_vec is None:
                stats["query_missing_addr_in_cache"] += 1
                continue
            rec["gq"] = new_vec
            stats["query_gq_replaced"] += 1
        write_pickle(records, str(output_query_dir / qpath.name))

    return stats


def _detect_target_global_dim(
    global_source_dir: Path,
    global_suffix: str,
) -> Optional[int]:
    """Peek the first available cache file under ``global_source_dir`` to
    determine the expected per-vector dim for the chosen ``global_suffix``.

    Used by ``repair_missing_pool_globals`` to identify pool rows whose vector
    still carries the *original* (e.g. 768-d slice) embedding because the
    substitution cache had no entry for them — those rows need a borrowed
    same-dim vector or ``populate_index`` will crash with a shape mismatch.
    """
    for path in sorted(global_source_dir.glob(f"*{global_suffix}")):
        try:
            d = read_pickle(str(path))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for _addr, item in d.items():
            if not isinstance(item, dict):
                continue
            vec = item.get("global")
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size > 0:
                return int(arr.size)
    return None


def _load_query_gt_func_names(source_query_dir: Path) -> set:
    """Collect the set of ``noinline_func_name`` values across every query record.

    These are the function names that some query expects to find in the pool
    (loose-GT match is by ``func_name``). A missing pool row whose
    ``func_name`` is in this set is "case 1" — it must be repaired with a
    same-name substitute so the query can still find its GT semantically.
    """
    names: set = set()
    for qpath in sorted(source_query_dir.glob("*_query*.pkl")):
        try:
            records = read_pickle(str(qpath))
        except Exception:
            continue
        if not isinstance(records, list):
            continue
        for rec in records:
            if not isinstance(rec, dict):
                continue
            name = (
                rec.get("noinline_func_name")
                or rec.get("debug_func_name")
                or rec.get("gt_func_name")
            )
            if name:
                names.add(name)
    return names


def _derive_feature_suffix_candidates(global_suffix: str, pool_feature_suffix: str) -> List[str]:
    """Prefer the feature.json that matches the global cache suffix; fall back
    to the slice-pipeline default. e.g. ``_HermesSim_embeddings.pkl`` →
    [``_HermesSim_feature.json``, ``_bb_slice_feature.json``].
    """
    matched: List[str] = []
    tail = "_embeddings.pkl"
    if global_suffix.endswith(tail):
        matched.append(global_suffix[: -len(tail)] + "_feature.json")
    if pool_feature_suffix not in matched:
        matched.append(pool_feature_suffix)
    return matched


def _build_substitute_index(
    global_source_dir: Path,
    global_suffix: str,
    feature_suffix_candidates: List[str],
    target_dim: int,
) -> Tuple[Dict[str, List[Tuple[str, str]]], List[Tuple[str, str]]]:
    """Scan ``global_source_dir`` for every (binary, addr) whose cached global
    vec has the target dim AND has a resolvable ``func_name``. Returns two
    handles ready for substitute selection:

      * ``by_funcname``: ``{func_name -> [(binary, addr), ...]}`` for case-1
        same-name lookups.
      * ``all_subs``: flat list of ``(binary, addr)`` for case-2 random picks.

    Vectors are *not* held in memory — they're loaded lazily via
    ``_load_substitute_vec_lru`` when actually picked, keeping this index small
    (~tens of MB for 600+ binaries × 10k addrs) instead of ~10 GB.
    """
    by_funcname: Dict[str, List[Tuple[str, str]]] = {}
    all_subs: List[Tuple[str, str]] = []

    cache_files = sorted(global_source_dir.glob(f"*{global_suffix}"))
    pbar = tqdm(cache_files, desc="scan-subs", unit="bin", dynamic_ncols=True, mininterval=0.3)
    for cache_path in pbar:
        binary_name = cache_path.name[: -len(global_suffix)]
        feat_path: Optional[Path] = None
        for fs in feature_suffix_candidates:
            cand = global_source_dir / f"{binary_name}{fs}"
            if cand.exists():
                feat_path = cand
                break
        if feat_path is None:
            continue
        try:
            cache = read_pickle(str(cache_path))
            feature = json.load(open(str(feat_path), encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(cache, dict) or not isinstance(feature, dict):
            continue
        sample_ok = False
        for _a, _it in cache.items():
            if isinstance(_it, dict) and _it.get("global") is not None:
                arr = np.asarray(_it["global"], dtype=np.float32).reshape(-1)
                sample_ok = (arr.size == target_dim)
                break
        if not sample_ok:
            continue
        for addr, item in cache.items():
            if not isinstance(item, dict) or item.get("global") is None:
                continue
            feat_item = feature.get(addr)
            if not isinstance(feat_item, dict):
                if isinstance(addr, str) and addr.startswith("0x"):
                    try:
                        feat_item = feature.get(int(addr, 16))
                    except ValueError:
                        feat_item = None
                elif isinstance(addr, int):
                    feat_item = feature.get(hex(addr))
            if not isinstance(feat_item, dict):
                continue
            func_name = feat_item.get("func_name") or ""
            if not func_name:
                continue
            entry = (binary_name, addr)
            by_funcname.setdefault(func_name, []).append(entry)
            all_subs.append(entry)
    pbar.close()
    return by_funcname, all_subs


def _load_substitute_vec_lru(
    global_source_dir: Path,
    binary_name: str,
    addr: str,
    global_suffix: str,
    cache_lru: "collections.OrderedDict",
    lru_cap: int = 64,
) -> Optional[list]:
    """Lazily load a substitute's vector; keep a tiny LRU of recently-opened
    cache files so multiple substitutes from the same binary don't re-pay disk."""
    cache = cache_lru.get(binary_name)
    if cache is None:
        cache_path = global_source_dir / f"{binary_name}{global_suffix}"
        try:
            cache = read_pickle(str(cache_path))
        except Exception:
            return None
        if not isinstance(cache, dict):
            return None
        cache_lru[binary_name] = cache
        while len(cache_lru) > lru_cap:
            cache_lru.popitem(last=False)
    else:
        cache_lru.move_to_end(binary_name)
    item = cache.get(addr)
    if not isinstance(item, dict):
        return None
    return item.get("global")


def repair_missing_pool_globals(
    output_pool_dir: Path,
    source_query_dir: Path,
    global_source_dir: Path,
    global_suffix: str,
    pool_feature_suffix: str,
    pool_embed_suffix: str,
    seed: int = 1234,
) -> dict:
    """Substitute missing-cache pool rows with vectors borrowed from elsewhere.

    Walks ``output_pool_dir`` looking for entries whose ``global`` vec dim
    disagrees with the modal cache dim under ``global_source_dir`` — these are
    rows that ``patch_pool`` left at the original (e.g. 768-d slice) vec
    because the per-method cache had no entry for that ``(binary, addr)``. They
    would otherwise crash ``populate_index`` with a stack-shape error.

    Substitution preserves the row's *identity* ``(binary_name, addr,
    func_name)`` — only the vector content changes. So ``pool_meta`` written
    by ``populate_index`` matches the source pool exactly; other methods
    reading the same pool layout see the same row layout. Two cases:

      * **Case 1** — row's ``func_name`` is in the query GT name set
        (``noinline_func_name`` across all ``*_query*.pkl``). Borrow a vector
        from another ``(bin, addr)`` of the **same** ``func_name`` so the
        GT-by-name match preserves *both* identity AND vector-space proximity
        (the row stays semantically a GT-class vector for that name).
      * **Case 2** — row's ``func_name`` is not a query GT. Borrow any random
        vector; only distractor behavior matters and identity is preserved.

    A case-1 lookup with no same-name candidate falls back to case-2 (logged).
    Unrepairable rows (no substitute pool available at all) are surfaced via
    ``missing_addrs_unrepaired`` and should make ``populate_index`` fail loudly
    so the caller is forced to fix coverage.
    """
    import collections
    import random

    target_dim = _detect_target_global_dim(global_source_dir, global_suffix)
    if target_dim is None:
        print(
            f"[repair] could not detect target dim under {global_source_dir} "
            f"(suffix={global_suffix}); skipping repair",
            flush=True,
        )
        return {"repair_skipped": True}
    print(f"[repair] target global dim = {target_dim}")

    query_gt_names = _load_query_gt_func_names(source_query_dir)
    print(f"[repair] {len(query_gt_names)} query GT func_names loaded from {source_query_dir}")

    feature_candidates = _derive_feature_suffix_candidates(global_suffix, pool_feature_suffix)
    print(f"[repair] feature.json candidates: {feature_candidates}")

    by_funcname, all_subs = _build_substitute_index(
        global_source_dir, global_suffix, feature_candidates, target_dim,
    )
    print(
        f"[repair] substitute pool: {len(all_subs)} (bin, addr) entries across "
        f"{len(by_funcname)} unique func_names"
    )

    rng = random.Random(seed)
    cache_lru: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
    stats = {
        "repair_target_dim": int(target_dim),
        "repair_substitute_pool_entries": len(all_subs),
        "missing_addrs_detected": 0,
        "missing_addrs_case1": 0,
        "missing_addrs_case1_repaired": 0,
        "missing_addrs_case1_fallback_to_case2": 0,
        "missing_addrs_case2": 0,
        "missing_addrs_case2_repaired": 0,
        "missing_addrs_unrepaired": 0,
    }

    emb_paths = sorted(output_pool_dir.glob(f"*{pool_embed_suffix}"))
    pbar = tqdm(emb_paths, desc="repair-pool", unit="bin", dynamic_ncols=True, mininterval=0.3)
    for emb_path in pbar:
        binary_name = emb_path.name[: -len(pool_embed_suffix)]
        feat_path = emb_path.with_name(f"{binary_name}{pool_feature_suffix}")
        if not feat_path.exists():
            continue
        try:
            emb_dict = read_pickle(str(emb_path))
            feat_dict = json.load(open(str(feat_path), encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(emb_dict, dict) or not isinstance(feat_dict, dict):
            continue

        mutated = False
        for addr, item in emb_dict.items():
            if not isinstance(item, dict):
                continue
            vec = item.get("global")
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size == target_dim:
                continue

            stats["missing_addrs_detected"] += 1
            feat_item = feat_dict.get(addr) if isinstance(feat_dict, dict) else None
            orig_func_name = (
                feat_item.get("func_name") if isinstance(feat_item, dict) else None
            ) or ""
            is_query_gt = orig_func_name in query_gt_names

            sub_vec: Optional[list] = None
            if is_query_gt:
                stats["missing_addrs_case1"] += 1
                candidates = [
                    c for c in by_funcname.get(orig_func_name, [])
                    if not (c[0] == binary_name and c[1] == addr)
                ]
                while candidates and sub_vec is None:
                    sub_bin, sub_addr = rng.choice(candidates)
                    sub_vec = _load_substitute_vec_lru(
                        global_source_dir, sub_bin, sub_addr, global_suffix, cache_lru,
                    )
                    if sub_vec is None:
                        candidates = [c for c in candidates if c != (sub_bin, sub_addr)]
                if sub_vec is not None:
                    stats["missing_addrs_case1_repaired"] += 1
                elif all_subs:
                    while all_subs and sub_vec is None:
                        sub_bin, sub_addr = rng.choice(all_subs)
                        sub_vec = _load_substitute_vec_lru(
                            global_source_dir, sub_bin, sub_addr, global_suffix, cache_lru,
                        )
                        if sub_vec is None:
                            all_subs = [c for c in all_subs if c != (sub_bin, sub_addr)]
                    if sub_vec is not None:
                        stats["missing_addrs_case1_fallback_to_case2"] += 1
            else:
                stats["missing_addrs_case2"] += 1
                while all_subs and sub_vec is None:
                    sub_bin, sub_addr = rng.choice(all_subs)
                    sub_vec = _load_substitute_vec_lru(
                        global_source_dir, sub_bin, sub_addr, global_suffix, cache_lru,
                    )
                    if sub_vec is None:
                        all_subs = [c for c in all_subs if c != (sub_bin, sub_addr)]
                if sub_vec is not None:
                    stats["missing_addrs_case2_repaired"] += 1

            if sub_vec is None:
                stats["missing_addrs_unrepaired"] += 1
                continue
            item["global"] = sub_vec
            mutated = True

        if mutated:
            write_pickle(emb_dict, str(emb_path))
    pbar.close()
    return stats


def _collect_pool_globals(
    pool_dir: Path,
    pool_embed_suffix: str = POOL_EMB_SUFFIX,
    pool_feature_suffix: str = POOL_FEATURE_SUFFIX,
    expected_dim: Optional[int] = None,
) -> Tuple[np.ndarray, List[Tuple[str, str, str]]]:
    """Walk the populated pool dir, return (matrix, meta) ready for index.add().

    When `expected_dim` is set (the centroids index dim), any global whose
    dimensionality differs is skipped. A cross-encoder replacement (e.g. 384-d
    HermesSim over a 768-d slice pool) leaves a few pool addrs carrying their
    original-dim vector: a binary with no replacement cache, or a function the
    encoder could not embed. Those cannot enter a fixed-dim index and would
    crash the stack, so we drop them and report the count.
    """
    rows: List[np.ndarray] = []
    meta: List[Tuple[str, str, str]] = []
    dropped_dim = 0
    for emb_path in sorted(pool_dir.glob(f"*{pool_embed_suffix}")):
        binary_name = emb_path.name[: -len(pool_embed_suffix)]
        feature_path = emb_path.with_name(f"{binary_name}{pool_feature_suffix}")
        if not feature_path.exists():
            continue
        feature_info = json.load(open(feature_path, encoding="utf-8"))
        embedding_info = read_pickle(str(emb_path))
        for addr, feat_item in feature_info.items():
            if addr not in embedding_info:
                continue
            bundle = embedding_info[addr]
            if not isinstance(bundle, dict):
                continue
            vec = bundle.get("global")
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            if expected_dim is not None and arr.size != expected_dim:
                dropped_dim += 1
                continue
            rows.append(arr)
            meta.append((binary_name, addr, str(feat_item.get("func_name") or "")))
    if dropped_dim:
        print(f"[index] WARN dropped {dropped_dim} pool globals whose dim != "
              f"{expected_dim} (no same-dim replacement; excluded from index)")
    if not rows:
        return np.empty((0, 0), dtype=np.float32), []
    matrix = np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    return np.ascontiguousarray(matrix, dtype=np.float32), meta


def populate_index(
    centroids_index_path: Path,
    pool_dir: Path,
    output_index_path: Path,
    pool_embed_suffix: str = POOL_EMB_SUFFIX,
    pool_feature_suffix: str = POOL_FEATURE_SUFFIX,
    add_batch: int = 100_000,
    nprobe: int = 16,
) -> dict:
    """Load centroids-only index, add all pool vectors, write populated DB."""
    if not centroids_index_path.exists():
        raise SystemExit(
            f"centroids index not found at {centroids_index_path} — "
            "run train_faiss_global_ann.py first (step 1)."
        )
    print(f"[index] load centroids from {centroids_index_path}")
    index = faiss.read_index(str(centroids_index_path))

    print(f"[index] collect pool globals from {pool_dir}")
    matrix, meta = _collect_pool_globals(
        pool_dir, pool_embed_suffix, pool_feature_suffix, expected_dim=index.d)
    if matrix.size == 0:
        raise SystemExit(f"no global vectors collected from {pool_dir}; nothing to populate")
    print(f"[index] adding {matrix.shape[0]} vectors (dim={matrix.shape[1]}) in batches of {add_batch}")

    try:
        index.reset()
    except Exception:
        pass

    for start in range(0, matrix.shape[0], add_batch):
        end = min(start + add_batch, matrix.shape[0])
        index.add(np.ascontiguousarray(matrix[start:end], dtype=np.float32))
    print(f"[index] index.ntotal={index.ntotal}")

    try:
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", int(max(1, nprobe)))
    except Exception:
        pass

    output_index_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[index] write populated DB -> {output_index_path}")
    faiss.write_index(index, str(output_index_path))

    meta_payload = {
        "centroids_index": str(centroids_index_path),
        "pool_dir": str(pool_dir),
        "num_vectors": int(matrix.shape[0]),
        "dim": int(matrix.shape[1]),
        "nprobe": int(max(1, nprobe)),
        "pool_meta": [
            {"binary_name": b, "func_addr": a, "func_name": f}
            for b, a, f in meta
        ],
    }
    meta_path = output_index_path.with_suffix(output_index_path.suffix + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta_payload, fp, indent=2, ensure_ascii=False)
    print(f"[index] meta -> {meta_path}")
    return {
        "centroids_index": str(centroids_index_path),
        "output_index": str(output_index_path),
        "meta_path": str(meta_path),
        "num_vectors": int(matrix.shape[0]),
    }


DEFAULT_SOURCE_POOL_DIR = ROOT_DIR / "Coding" / "artifacts" / "tmp" / "pool_noinline"
DEFAULT_SOURCE_QUERY_DIR = ROOT_DIR / "Coding" / "artifacts" / "tmp" / "query_noinline"
DEFAULT_POOL_GLOBAL_SOURCE = ROOT_DIR / "Coding" / "data" / "inline" / "pool"
DEFAULT_QUERY_GLOBAL_SOURCE = ROOT_DIR / "Coding" / "data" / "inline" / "strip"
DEFAULT_OUTPUT_POOL_DIR = ROOT_DIR / "Coding" / "artifacts" / "tmp" / "pool_global_ann"
DEFAULT_OUTPUT_QUERY_DIR = ROOT_DIR / "Coding" / "artifacts" / "tmp" / "query_global_ann"
DEFAULT_CENTROIDS_INDEX = ROOT_DIR / "Coding" / "artifacts" / "faiss" / "global_ann_centroids.index"
DEFAULT_OUTPUT_INDEX = ROOT_DIR / "Coding" / "artifacts" / "faiss" / "global_ann_pool.index"


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: swap pool/query globals + populate the FAISS DB for "
        "global_ann_inline_eval.py.",
    )
    parser.add_argument("--source-pool-dir", default=str(DEFAULT_SOURCE_POOL_DIR))
    parser.add_argument("--source-query-dir", default=str(DEFAULT_SOURCE_QUERY_DIR))
    parser.add_argument("--pool-global-source-dir", default=str(DEFAULT_POOL_GLOBAL_SOURCE),
                        help="Directory holding <binary>_feature_global_embeddings.pkl for pool binaries.")
    parser.add_argument("--query-global-source-dir", default=str(DEFAULT_QUERY_GLOBAL_SOURCE),
                        help="Directory holding <binary>_feature_global_embeddings.pkl for query binaries.")
    parser.add_argument("--output-pool-dir", default=str(DEFAULT_OUTPUT_POOL_DIR))
    parser.add_argument("--output-query-dir", default=str(DEFAULT_OUTPUT_QUERY_DIR))
    parser.add_argument("--centroids-index", default=str(DEFAULT_CENTROIDS_INDEX),
                        help="Path to centroids-only index produced by step 1 "
                             "(train_faiss_global_ann.py).")
    parser.add_argument("--output-index", default=str(DEFAULT_OUTPUT_INDEX),
                        help="Path where the populated FAISS DB will be written. "
                             "This is what global_ann_inline_eval.py --model defaults to.")
    parser.add_argument("--nprobe", type=int, default=16,
                        help="nprobe baked into the populated index. Eval can still override.")
    parser.add_argument("--add-batch", type=int, default=100_000)
    parser.add_argument("--skip-populate", action="store_true",
                        help="Only swap pkl globals; do NOT populate the FAISS DB. Use this "
                             "when you only need fresh pkl artifacts and will rebuild the index "
                             "by another path.")
    parser.add_argument("--global-suffix", default=GLOBAL_EMB_SUFFIX,
                        help=f"Suffix of the per-binary global-embedding cache (default: {GLOBAL_EMB_SUFFIX}).")
    parser.add_argument("--pool-feature-suffix", default=POOL_FEATURE_SUFFIX)
    parser.add_argument("--pool-embed-suffix", default=None,
                        help="Filename suffix of the pool-side slice embedding pkl. If omitted, "
                             "derived from --pool-feature-suffix.")
    parser.add_argument("--no-on-demand-encode", action="store_true",
                        help="Disable the on-the-fly encoder fallback. When a binary has no "
                             f"{GLOBAL_EMB_SUFFIX} cache, just WARN and keep original globals "
                             "(legacy behavior).")
    parser.add_argument("--on-demand-model-path", default=ON_DEMAND_DEFAULT_MODEL_PATH,
                        help="Encoder model path used for on-demand subset encoding "
                             f"(default: {ON_DEMAND_DEFAULT_MODEL_PATH}).")
    parser.add_argument("--on-demand-gpu-id", type=int, default=1)
    parser.add_argument("--on-demand-max-length", type=int, default=ON_DEMAND_DEFAULT_MAX_LENGTH)
    parser.add_argument("--on-demand-batch-size", type=int, default=ON_DEMAND_DEFAULT_BATCH_SIZE)
    parser.add_argument("--no-on-demand-persist", action="store_true",
                        help="Don't write the freshly-encoded subset back to the global-source "
                             f"dir as {GLOBAL_EMB_SUFFIX}. Default: persist (so future runs hit "
                             "the cache).")
    parser.add_argument(
        "--substitute-missing-globals", action="store_true",
        help="After patch_pool, find pool rows whose 'global' still carries the "
             "original (mismatched-dim) vector because the source cache had no "
             "entry for that (binary, addr), and replace them with a borrowed "
             "vector. Case 1 (row's func_name is in the query GT set): borrow "
             "from a same-func_name (bin, addr) — preserves both row identity "
             "AND vector semantics so loose-GT matching still resolves. "
             "Case 2 (not a query GT): borrow any random vector. Row identity "
             "(binary_name, addr, func_name) is preserved either way, so "
             "pool_meta stays aligned with the source pool. Required when the "
             "chosen --global-suffix has coverage gaps (e.g. HermesSim "
             "decompile failures) that would otherwise break populate_index.")
    parser.add_argument("--substitute-seed", type=int, default=1234,
                        help="RNG seed used for substitute selection — change "
                             "to perturb which (bin, addr) gets borrowed.")
    args = parser.parse_args()
    if args.pool_embed_suffix is None:
        args.pool_embed_suffix = derive_pool_embed_suffix(args.pool_feature_suffix)

    source_pool = Path(args.source_pool_dir).resolve()
    source_query = Path(args.source_query_dir).resolve()
    pool_global_src = Path(args.pool_global_source_dir).resolve()
    query_global_src = Path(args.query_global_source_dir).resolve()
    out_pool = Path(args.output_pool_dir).resolve()
    out_query = Path(args.output_query_dir).resolve()
    global_suffix = args.global_suffix

    if out_pool == source_pool:
        raise SystemExit("output-pool-dir must differ from source-pool-dir")
    if out_query == source_query:
        raise SystemExit("output-query-dir must differ from source-query-dir")

    encoder_holder: Optional[_OnDemandEncoder] = None
    if not args.no_on_demand_encode:
        encoder_holder = _OnDemandEncoder(
            model_path=args.on_demand_model_path,
            gpu_id=args.on_demand_gpu_id,
            max_seq_length=args.on_demand_max_length,
            batch_size=args.on_demand_batch_size,
        )
    persist_on_demand = not args.no_on_demand_persist

    print(f"[pool] source={source_pool}")
    print(f"[pool] global-source={pool_global_src}")
    print(f"[pool] global-suffix={global_suffix}")
    print(f"[pool] feature-suffix={args.pool_feature_suffix}  embed-suffix={args.pool_embed_suffix}")
    print(f"[pool] output={out_pool}")
    print(f"[pool] on-demand-encode={'on' if encoder_holder is not None else 'off'} "
          f"persist={'on' if persist_on_demand else 'off'}")
    pool_stats = patch_pool(
        source_pool, pool_global_src, out_pool, global_suffix,
        pool_feature_suffix=args.pool_feature_suffix,
        pool_embed_suffix=args.pool_embed_suffix,
        encoder_holder=encoder_holder,
        persist_on_demand_cache=persist_on_demand,
    )
    for k, v in pool_stats.items():
        print(f"  {k}={v}")

    print(f"[query] source={source_query}")
    print(f"[query] global-source={query_global_src}")
    print(f"[query] global-suffix={global_suffix}")
    print(f"[query] output={out_query}")
    print(f"[query] on-demand-encode={'on' if encoder_holder is not None else 'off'} "
          f"persist={'on' if persist_on_demand else 'off'}")
    query_stats = patch_query(
        source_query, query_global_src, out_query, global_suffix,
        encoder_holder=encoder_holder,
        persist_on_demand_cache=persist_on_demand,
    )
    for k, v in query_stats.items():
        print(f"  {k}={v}")

    if pool_stats["pool_addrs_missing_in_cache"] > 0 or query_stats["query_missing_addr_in_cache"] > 0:
        print(
            "[warn] some addresses were not covered by the new global cache — those vectors "
            "were left untouched. If this is unexpected, re-run extract_global_embedding_from_feature.py "
            "with --force over the source dir that feeds those binaries.",
        )

    if args.substitute_missing_globals:
        missing_total = (
            pool_stats["pool_addrs_missing_in_cache"]
            + pool_stats["pool_binaries_missing_cache"]
        )
        if missing_total == 0:
            print("[repair] no missing globals reported by patch_pool — skipping substitution.")
        else:
            print(
                f"[repair] substitute-missing-globals=on: borrowing vectors for "
                f"{pool_stats['pool_addrs_missing_in_cache']} missing addrs + "
                f"{pool_stats['pool_binaries_missing_cache']} fully-uncovered binaries"
            )
            repair_stats = repair_missing_pool_globals(
                output_pool_dir=out_pool,
                source_query_dir=source_query,
                global_source_dir=pool_global_src,
                global_suffix=global_suffix,
                pool_feature_suffix=args.pool_feature_suffix,
                pool_embed_suffix=args.pool_embed_suffix,
                seed=args.substitute_seed,
            )
            for k, v in repair_stats.items():
                print(f"  {k}={v}")
            if repair_stats.get("missing_addrs_unrepaired", 0) > 0:
                raise SystemExit(
                    f"[repair] {repair_stats['missing_addrs_unrepaired']} addrs could "
                    f"not be substituted (no candidates in {pool_global_src} with "
                    f"suffix {global_suffix}). populate_index would crash on the "
                    f"residual dim mismatch — fix coverage and re-run."
                )

    if args.skip_populate:
        print("[index] skip-populate set — leaving FAISS DB untouched.")
        return

    populate_stats = populate_index(
        centroids_index_path=Path(args.centroids_index).resolve(),
        pool_dir=out_pool,
        output_index_path=Path(args.output_index).resolve(),
        pool_embed_suffix=args.pool_embed_suffix,
        pool_feature_suffix=args.pool_feature_suffix,
        add_batch=args.add_batch,
        nprobe=args.nprobe,
    )
    for k, v in populate_stats.items():
        print(f"  {k}={v}")


if __name__ == "__main__":
    main()
