import argparse
import multiprocessing
import os
import shutil
import sys
import time
from collections import defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple, Union

PoolSourceArg = Union[str, "os.PathLike[str]", Sequence[Union[str, "os.PathLike[str]"]]]

from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
for path in (str(ROOT_DIR), str(CURRENT_DIR), str(ROOT_DIR / "V3")):
    if path not in sys.path:
        sys.path.append(path)

import re

from utils import read_json, read_pickle, write_json, write_pickle
from cve_pool_identity import cve_source_signature, is_cve_entry

_BINARY_FAMILY_PREFIX_RE = re.compile(r"^[^-]+-[^-]+-[^-]+-[^-]+-(.+)$")


def normalize_binary_key(binary_name):
    raw = os.path.basename(str(binary_name or ""))
    match = _BINARY_FAMILY_PREFIX_RE.match(raw)
    if match:
        return match.group(1)
    parts = raw.split("-", 4)
    if len(parts) == 5:
        return parts[4]
    return raw


INLINE_FEATURE_SUFFIX = "_bb_slice_feature.json"
INLINE_EMBED_SUFFIX = "_bb_slice_embeddings.pkl"
MIN_PSEUDOCODE_LINES = 8
DEFAULT_BUILD_WORKERS = 4


def _addr_sort_key(addr: str):
    raw = str(addr)
    try:
        return (0, int(raw, 0))
    except (TypeError, ValueError):
        return (1, raw)


def _normalize_pool_source_dirs(pool_source_dir: PoolSourceArg) -> List[str]:
    """Accept either a single dir (str / Path) or a sequence of dirs.

    Returned as a list of stringified absolute-ish paths (caller-supplied,
    not necessarily resolved). Empty / falsy entries are dropped.
    """
    if isinstance(pool_source_dir, (str, os.PathLike)):
        items: Sequence = [pool_source_dir]
    else:
        items = list(pool_source_dir)
    out: List[str] = []
    for item in items:
        if not item:
            continue
        out.append(str(item))
    if not out:
        raise ValueError("pool_source_dir is empty")
    return out


def _iter_pool_binaries(pool_source_dir: PoolSourceArg,
                        feature_suffix: str = INLINE_FEATURE_SUFFIX,
                        embedding_suffix: str = INLINE_EMBED_SUFFIX,
                        ) -> Iterable[Tuple[str, Path, Path, Path]]:
    """Walk all feature files across one or more pool source roots.

    On binary_name collisions across roots, the first root wins — later
    duplicates are skipped (logged once per dup name).
    """
    source_dirs = _normalize_pool_source_dirs(pool_source_dir)
    seen: Set[str] = set()
    collisions: List[Tuple[str, str, str]] = []
    for source_dir in source_dirs:
        source_root = Path(source_dir)
        for feature_path in sorted(source_root.rglob(f"*{feature_suffix}")):
            binary_name = feature_path.name[: -len(feature_suffix)]
            binary_path = feature_path.with_name(binary_name)
            embedding_path = feature_path.with_name(f"{binary_name}{embedding_suffix}")
            if not binary_path.exists():
                continue
            if binary_name in seen:
                collisions.append((binary_name, str(feature_path.parent), source_dir))
                continue
            seen.add(binary_name)
            yield binary_name, binary_path, feature_path, embedding_path
    if collisions:
        print(
            f"[pool-builder] skipped {len(collisions)} duplicate binary names "
            f"appearing in multiple --pool-source-dir roots (first-root-wins). "
            f"Sample: {collisions[:3]}",
            flush=True,
        )


def load_query_pool_constraints(query_dir: str) -> Dict[str, object]:
    gt_entries: "OrderedDict[Tuple[str, str], Dict[str, str]]" = OrderedDict()
    related_groups: Dict[Tuple[str, str], Set[Tuple[str, str, str]]] = defaultdict(set)
    query_paths = sorted(Path(query_dir).glob("*.pkl"))
    for query_path in tqdm(query_paths, desc="pool-builder: scan query", dynamic_ncols=True):
        infos = read_pickle(query_path)
        for info in infos:
            gt_binary = os.path.basename(str(info.get("gt_binary") or ""))
            gt_addr = str(info.get("addr"))
            func_name = str(info.get("noinline_func_name") or info.get("debug_func_name") or "")
            if not gt_binary or gt_addr in {"", "None"}:
                continue
            key = (gt_binary, gt_addr)
            if key in gt_entries:
                pass
            else:
                gt_entries[key] = {
                    "binary_name": gt_binary,
                    "func_addr": gt_addr,
                    "func_name": func_name,
                }
            query_binary = os.path.basename(str(info.get("binary") or ""))
            query_addr = str(info.get("query_addr"))
            if query_binary and query_addr not in {"", "None"} and func_name:
                related_groups[(query_binary, query_addr)].add((gt_binary, gt_addr, func_name))

    excluded_related_signatures: Set[Tuple[str, str]] = set()
    multi_target_query_count = 0
    for group_items in related_groups.values():
        unique_pairs = {(binary_name, func_addr) for binary_name, func_addr, _ in group_items}
        if len(unique_pairs) <= 1:
            continue
        multi_target_query_count += 1
        for binary_name, _func_addr, func_name in group_items:
            if binary_name and func_name:
                excluded_related_signatures.add((binary_name, func_name))

    return {
        "gt_functions": list(gt_entries.values()),
        "gt_entries": gt_entries,
        "excluded_related_signatures": excluded_related_signatures,
        "multi_target_query_count": int(multi_target_query_count),
    }


def _load_one_feature_slim(
    args: Tuple[str, str, str, str],
) -> Optional[Tuple[str, str, str, str, List[Tuple[str, Dict[str, object]]]]]:
    """Worker: read one feature.json, return slim per-addr index.

    Slim shape: ``{addr: {"func_name": str, "lines_count": int}}``.
    Full feature dicts are NOT carried back — Phase B workers re-read the
    source json for the small subset they actually need.
    """
    binary_name, binary_path, feature_path, embedding_path = args
    feature_info = read_json(feature_path)
    items: List[Tuple[str, Dict[str, object]]] = []
    for addr in sorted(feature_info.keys(), key=_addr_sort_key):
        feat = feature_info[addr]
        items.append((
            str(addr),
            {
                "func_name": str(feat.get("func_name") or ""),
                "lines_count": int(len(feat.get("lines") or [])),
            },
        ))
    if not items:
        return None
    return binary_name, binary_path, feature_path, embedding_path, items


def _absorb_inventory_result(
    inventory: Dict[str, Dict[str, object]],
    result: Optional[Tuple[str, str, str, str, List[Tuple[str, Dict[str, object]]]]],
) -> None:
    if result is None:
        return
    binary_name, binary_path, feature_path, embedding_path, items = result
    functions: "OrderedDict[str, Dict[str, object]]" = OrderedDict()
    for addr, slim in items:
        functions[addr] = slim
    inventory[binary_name] = {
        "binary_path": binary_path,
        "feature_path": feature_path,
        "embedding_path": embedding_path,
        "functions": functions,
    }


def load_pool_inventory(pool_source_dir: PoolSourceArg,
                         feature_suffix: str = INLINE_FEATURE_SUFFIX,
                         embedding_suffix: str = INLINE_EMBED_SUFFIX,
                         workers: int = DEFAULT_BUILD_WORKERS,
                         ) -> Dict[str, Dict[str, object]]:
    """Scan feature files only. Embeddings are produced on demand later
    (or pulled from a pre-encoded pkl at materialize time, if present).

    Inventory rows hold a slim per-addr index (``func_name`` + ``lines_count``).
    The full feature dict is intentionally NOT retained in RAM — at 3M-funcs
    pool scale the full retention costs 30-50 GB and provides no value to
    ``select_pool_functions`` / sibling+CVE-clone analyzers, which only read
    ``func_name`` and the pseudocode line count. ``materialize_selected_pool``
    re-reads each binary's feature.json from disk to assemble its subset.
    """
    pool_binaries = list(_iter_pool_binaries(
        pool_source_dir, feature_suffix=feature_suffix,
        embedding_suffix=embedding_suffix,
    ))
    args_list = [
        (binary_name, str(binary_path), str(feature_path), str(embedding_path))
        for binary_name, binary_path, feature_path, embedding_path in pool_binaries
    ]

    inventory: Dict[str, Dict[str, object]] = {}
    desc = "pool-builder: scan source pool"
    if workers <= 1 or len(args_list) <= 1:
        for args in tqdm(args_list, desc=desc, dynamic_ncols=True):
            _absorb_inventory_result(inventory, _load_one_feature_slim(args))
        return inventory

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        for result in tqdm(
            ex.map(_load_one_feature_slim, args_list, chunksize=4),
            total=len(args_list), desc=desc, dynamic_ncols=True,
        ):
            _absorb_inventory_result(inventory, result)
    return inventory


def select_pool_functions(
    gt_functions: List[Dict[str, str]],
    inventory: Dict[str, Dict[str, object]],
    pool_size: int,
    excluded_related_signatures: Optional[Set[Tuple[str, str]]] = None,
) -> Dict[str, List[str]]:
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")

    selected_by_binary: Dict[str, List[str]] = defaultdict(list)
    selected_set = set()
    excluded_related_signatures = excluded_related_signatures or set()
    for gt in gt_functions:
        binary_name = gt["binary_name"]
        func_addr = gt["func_addr"]
        binary_item = inventory.get(binary_name)
        if binary_item is None:
            raise ValueError(f"GT binary missing from pool source: {binary_name}")
        functions = binary_item["functions"]
        if func_addr not in functions:
            raise ValueError(f"GT function missing from pool source: {binary_name}:{func_addr}")
        key = (binary_name, func_addr)
        if key in selected_set:
            continue
        selected_set.add(key)
        selected_by_binary[binary_name].append(func_addr)

    gt_count = len(selected_set)
    if gt_count > pool_size:
        raise ValueError(f"pool_size={pool_size} is smaller than unique GT count={gt_count}")

    remaining_by_binary: Dict[str, List[str]] = {}
    for binary_name, binary_item in inventory.items():
        functions = binary_item["functions"]
        remaining_by_binary[binary_name] = [
            addr
            for addr in functions.keys()
            if (binary_name, addr) not in selected_set
            and (
                (binary_name, str(functions[addr].get("func_name") or ""))
                not in excluded_related_signatures
            )
            and int(functions[addr].get("lines_count") or 0) > MIN_PSEUDOCODE_LINES
        ]

    remaining_capacity = pool_size - gt_count

    for binary_name in tqdm(
        sorted(inventory.keys()),
        desc="pool-builder: cover binaries",
        dynamic_ncols=True,
    ):
        if remaining_capacity <= 0:
            break
        if selected_by_binary.get(binary_name):
            continue
        remaining = remaining_by_binary[binary_name]
        if not remaining:
            continue
        func_addr = remaining.pop(0)
        selected_by_binary[binary_name].append(func_addr)
        selected_set.add((binary_name, func_addr))
        remaining_capacity -= 1

    fill_bar = tqdm(total=remaining_capacity, desc="pool-builder: fill pool", dynamic_ncols=True)
    while remaining_capacity > 0:
        candidate_binaries = [
            binary_name
            for binary_name, remaining in remaining_by_binary.items()
            if remaining
        ]
        if not candidate_binaries:
            break
        candidate_binaries.sort(key=lambda name: (len(selected_by_binary.get(name, [])), name))
        progressed = False
        for binary_name in candidate_binaries:
            if remaining_capacity <= 0:
                break
            remaining = remaining_by_binary[binary_name]
            if not remaining:
                continue
            func_addr = remaining.pop(0)
            selected_by_binary[binary_name].append(func_addr)
            selected_set.add((binary_name, func_addr))
            remaining_capacity -= 1
            fill_bar.update(1)
            progressed = True
        if not progressed:
            break
    fill_bar.close()

    if remaining_capacity > 0:
        raise ValueError(
            f"pool source does not have enough functions to fill pool_size={pool_size}"
        )

    normalized: Dict[str, List[str]] = {}
    for binary_name, addrs in selected_by_binary.items():
        unique_addrs = sorted(set(str(addr) for addr in addrs), key=_addr_sort_key)
        if unique_addrs:
            normalized[binary_name] = unique_addrs
    return normalized


def _prepare_output_dir(output_dir: str, clean: bool = False,
                         embedding_suffix: str = INLINE_EMBED_SUFFIX):
    """Prepare ``output_dir``.

    By default, preserve any existing ``*<embedding_suffix>`` files so a
    rebuild with the same selection can reuse the encoded vectors. Pass
    ``clean=True`` to wipe the whole directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if clean:
        for item in output_path.iterdir():
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink()
        return
    for item in output_path.iterdir():
        if item.is_file() and item.name.endswith(embedding_suffix):
            continue
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()


def _link_binary(src_binary: Path, dst_binary: Path):
    if dst_binary.exists() or dst_binary.is_symlink():
        dst_binary.unlink()
    os.symlink(src_binary, dst_binary)


def _materialize_one_binary(
    args: Tuple[str, str, str, str, str, List[str], FrozenSet[str], bool, str, str],
) -> Dict[str, object]:
    """Worker: reads source feature.json + embedding pkl, writes subset
    feature.json + embedding pkl, returns metadata.

    Does NOT use the embedder — workers can't share GPU state. If the
    source pkl is missing or incomplete, the worker signals via
    ``needs_embedder=True`` and the main process handles those binaries
    sequentially.
    """
    (
        binary_name,
        src_binary_path,
        src_feature_path,
        src_embedding_path,
        output_dir,
        addrs,
        gt_addr_set,
        embed_force,
        feature_suffix,
        embedding_suffix,
    ) = args
    output_path = Path(output_dir)
    dst_binary = output_path / binary_name
    _link_binary(Path(src_binary_path), dst_binary)

    feature_info = read_json(src_feature_path)
    subset_feature: "OrderedDict[str, object]" = OrderedDict()
    selected_records: List[Dict[str, object]] = []
    for addr in addrs:
        feat = feature_info[addr]
        subset_feature[addr] = feat
        selected_records.append({
            "binary_name": binary_name,
            "func_addr": addr,
            "func_name": str(feat.get("func_name") or ""),
            "is_gt": addr in gt_addr_set,
        })
    subset_feature_path = f"{dst_binary}{feature_suffix}"
    subset_embedding_path = f"{dst_binary}{embedding_suffix}"
    write_json(subset_feature, subset_feature_path)

    candidate = None
    if (
        not embed_force
        and src_embedding_path
        and Path(src_embedding_path).exists()
    ):
        try:
            candidate = read_pickle(src_embedding_path)
        except Exception as exc:
            print(
                f"[pool-builder] failed to read {src_embedding_path}: {exc}; "
                "will signal re-encode",
                flush=True,
            )
            candidate = None

    if isinstance(candidate, dict) and all(addr in candidate for addr in addrs):
        subset_embedding: "OrderedDict[str, object]" = OrderedDict()
        for addr in addrs:
            subset_embedding[addr] = candidate[addr]
        write_pickle(subset_embedding, subset_embedding_path)
        return {
            "binary_name": binary_name,
            "selected_records": selected_records,
            "reused": len(addrs),
            "needs_embedder": False,
        }

    return {
        "binary_name": binary_name,
        "selected_records": selected_records,
        "reused": 0,
        "needs_embedder": True,
    }


def materialize_selected_pool(
    selected_by_binary: Dict[str, List[str]],
    gt_functions: List[Dict[str, str]],
    inventory: Dict[str, Dict[str, object]],
    output_dir: str,
    pool_size: int,
    query_dir: str,
    pool_source_dir: PoolSourceArg,
    embedder=None,
    embed_force: bool = False,
    clean_output: bool = False,
    feature_suffix: str = INLINE_FEATURE_SUFFIX,
    embedding_suffix: str = INLINE_EMBED_SUFFIX,
    workers: int = DEFAULT_BUILD_WORKERS,
) -> Dict[str, object]:
    """Write subset feature files + symlink binaries.

    Embedding strategy:
      * If ``embedder`` is provided, call ``embedder.encode_and_persist``
        on the subset feature file — only the selected addrs are encoded
        and the resulting pickle is written into ``output_dir`` for reuse.
      * Otherwise fall back to slicing a pre-encoded pkl from the source
        directory. Binaries without a pre-encoded pkl raise in this mode.

    By default, previously-written embedding pkls in ``output_dir`` are
    preserved so a rebuild reuses the cached vectors. Set
    ``clean_output=True`` to wipe the directory first.
    """
    _prepare_output_dir(output_dir, clean=clean_output, embedding_suffix=embedding_suffix)
    output_path = Path(output_dir)
    selected_functions: List[Dict[str, str]] = []
    embedded_function_count = 0
    reused_function_count = 0
    embed_seconds = 0.0
    any_reused = False
    any_encoded = False

    binary_names = sorted(selected_by_binary.keys())
    gt_addr_by_binary: Dict[str, Set[str]] = defaultdict(set)
    for gt in gt_functions:
        gt_addr_by_binary[gt["binary_name"]].add(str(gt["func_addr"]))

    worker_args: List[
        Tuple[str, str, str, str, str, List[str], FrozenSet[str], bool, str, str]
    ] = []
    for binary_name in binary_names:
        binary_item = inventory[binary_name]
        worker_args.append((
            binary_name,
            str(binary_item["binary_path"]),
            str(binary_item["feature_path"]),
            str(binary_item.get("embedding_path") or ""),
            str(output_path),
            list(selected_by_binary[binary_name]),
            frozenset(gt_addr_by_binary.get(binary_name, set())),
            embed_force,
            feature_suffix,
            embedding_suffix,
        ))

    needs_embedder_binaries: List[str] = []
    desc = "pool-builder: write pool"

    def _consume(result: Dict[str, object]) -> None:
        nonlocal reused_function_count, any_reused
        selected_functions.extend(result["selected_records"])
        if result["needs_embedder"]:
            needs_embedder_binaries.append(str(result["binary_name"]))
        else:
            reused_function_count += int(result["reused"])
            any_reused = True

    if workers <= 1 or len(worker_args) <= 1:
        for args in tqdm(worker_args, desc=desc, dynamic_ncols=True):
            _consume(_materialize_one_binary(args))
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for result in tqdm(
                ex.map(_materialize_one_binary, worker_args, chunksize=2),
                total=len(worker_args), desc=desc, dynamic_ncols=True,
            ):
                _consume(result)

    if needs_embedder_binaries:
        if embedder is None:
            first = needs_embedder_binaries[0]
            src_path = inventory[first].get("embedding_path")
            raise ValueError(
                f"pre-encoded embedding missing or incomplete for {first} at "
                f"{src_path}; pass embedder for on-demand encoding "
                f"({len(needs_embedder_binaries)} affected)"
            )
        for binary_name in tqdm(
            needs_embedder_binaries,
            desc="pool-builder: on-demand encode",
            dynamic_ncols=True,
        ):
            dst_binary = output_path / binary_name
            subset_feature_path = f"{dst_binary}{feature_suffix}"
            subset_embedding_path = f"{dst_binary}{embedding_suffix}"
            addrs = list(selected_by_binary[binary_name])
            t0 = time.time()
            embedder.encode_and_persist(
                subset_feature_path,
                subset_embedding_path,
                addrs,
                force=embed_force,
                desc_prefix=f"pool:{binary_name}",
            )
            embed_seconds += time.time() - t0
            embedded_function_count += len(addrs)
            any_encoded = True

    pool_source_dirs_abs = [
        os.path.abspath(p) for p in _normalize_pool_source_dirs(pool_source_dir)
    ]
    manifest = {
        "query_dir": os.path.abspath(query_dir),
        "pool_source_dir": pool_source_dirs_abs,
        "pool_size": int(pool_size),
        "gt_function_count": int(len({(gt['binary_name'], gt['func_addr']) for gt in gt_functions})),
        "selected_function_count": int(len(selected_functions)),
        "selected_binary_count": int(len(selected_by_binary)),
        "embedding_strategy": (
            "mixed" if (any_reused and any_encoded)
            else ("reused" if any_reused else ("on_demand" if any_encoded else "empty"))
        ),
        "embedded_function_count": int(embedded_function_count),
        "reused_function_count": int(reused_function_count),
        "embed_seconds": float(round(embed_seconds, 3)),
        "gt_functions": sorted(gt_functions, key=lambda item: (item["binary_name"], _addr_sort_key(item["func_addr"]))),
        "selected_functions": sorted(
            selected_functions,
            key=lambda item: (item["binary_name"], _addr_sort_key(item["func_addr"])),
        ),
        "selected_counts_by_binary": {
            binary_name: len(addrs)
            for binary_name, addrs in sorted(selected_by_binary.items())
        },
    }
    write_json(manifest, output_path / "pool_manifest.json")
    return manifest


def _identify_short_gt_keys(
    gt_entries: Dict[Tuple[str, str], Dict[str, str]],
    inventory: Dict[str, Dict[str, object]],
    min_pseudocode_lines: int = MIN_PSEUDOCODE_LINES,
) -> Set[Tuple[str, str]]:
    """Return the set of (gt_binary, gt_addr) whose pool-side function has
    <= min_pseudocode_lines, i.e. queries that should be dropped."""
    short: Set[Tuple[str, str]] = set()
    for key, gt_info in gt_entries.items():
        binary_name = gt_info["binary_name"]
        func_addr = str(gt_info["func_addr"])
        entry = (inventory.get(binary_name, {}) or {}).get("functions", {}).get(func_addr)
        if entry is None:
            continue
        if int(entry.get("lines_count") or 0) <= min_pseudocode_lines:
            short.add((binary_name, func_addr))
    return short


def _identify_multi_config_sibling_signatures(
    gt_entries: Dict[Tuple[str, str], Dict[str, str]],
    inventory: Dict[str, Dict[str, object]],
) -> Set[Tuple[str, str]]:
    """For each GT, find every (binary_name, func_name) in the same source
    family that exposes the GT's func_name. These are "sibling builds" of
    the same source-level function (e.g. libcrypto.so.3 compiled with
    different gcc options) plus any within-binary duplicates from static
    linkage. Returning them lets the caller extend
    ``excluded_related_signatures`` so random pool sampling can't pull a
    sibling in beside the GT — without that exclusion the model can rank
    the sibling above the GT and the (gt_binary, gt_addr) tuple match
    fails even though retrieval was semantically correct.

    The GT itself is always force-included via `selected_set` in
    `select_pool_functions`, so including (gt_binary, gt_name) in the
    exclusion set is safe — it only filters OTHER addrs/binaries with the
    same name, not the GT.
    """
    family_to_binaries: Dict[str, List[str]] = defaultdict(list)
    for binary_name in inventory.keys():
        family_to_binaries[normalize_binary_key(binary_name)].append(binary_name)
    binary_name_index: Set[Tuple[str, str]] = set()
    for binary_name, bin_data in inventory.items():
        for fn_data in (bin_data.get("functions") or {}).values():
            nm = fn_data.get("func_name")
            if nm:
                binary_name_index.add((binary_name, nm))
    siblings: Set[Tuple[str, str]] = set()
    for gt_info in gt_entries.values():
        gt_binary = gt_info["binary_name"]
        gt_name = gt_info.get("func_name") or ""
        if not gt_name:
            continue
        for fam_binary in family_to_binaries.get(normalize_binary_key(gt_binary), []):
            if (fam_binary, gt_name) in binary_name_index:
                siblings.add((fam_binary, gt_name))
    return siblings


def _identify_cve_source_clone_signatures(
    gt_entries: Dict[Tuple[str, str], Dict[str, str]],
    inventory: Dict[str, Dict[str, object]],
) -> Tuple[Set[Tuple[str, str]], int]:
    """Source-level dedup for CVE-mirrored pool entries.

    CVE noinline binaries are mirrored into the pool under flat
    ``<binname>-<sha10>`` names (see ``copy_cve_noinline_to_pool.py``). The
    same upstream tool can show up under N different sha checkouts, so a
    non-vuln helper like ``rtp_print`` ends up replicated across tens of
    near-identical embeddings. Strict (gt_binary, gt_addr) eval matching
    then can't pick GT out of its own clones, and ``recall@k`` collapses.

    We group every CVE inventory entry by ``cve_source_signature`` =
    ``(case_binname, func_name)``. For groups with > 1 entry:
      * if any entry is a GT case for some query → keep the GT entries,
        exclude every non-GT clone from random sampling.
      * otherwise → keep the lex-first entry, exclude the rest.

    Returns (excluded_signatures, dropped_clone_count).
    """
    gt_signatures: Set[Tuple[str, str]] = set()
    for (gt_binary, _gt_addr), gt_info in gt_entries.items():
        fn = str(gt_info.get("func_name") or "")
        if fn:
            gt_signatures.add((gt_binary, fn))
    signature_to_entries: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
    for binary_name, bin_data in inventory.items():
        if not is_cve_entry(binary_name):
            continue
        seen_names: Set[str] = set()
        for fn_data in (bin_data.get("functions") or {}).values():
            func_name = str(fn_data.get("func_name") or "")
            sig = cve_source_signature(binary_name, func_name)
            if sig is None:
                continue
            if func_name in seen_names:
                continue
            seen_names.add(func_name)
            signature_to_entries[sig].append((binary_name, func_name))

    excluded: Set[Tuple[str, str]] = set()
    for entries in signature_to_entries.values():
        if len(entries) <= 1:
            continue
        gt_in_group = {entry for entry in entries if entry in gt_signatures}
        if gt_in_group:
            keep = gt_in_group
        else:
            keep = {min(entries)}
        for entry in entries:
            if entry not in keep:
                excluded.add(entry)
    return excluded, len(excluded)


def _prune_query_pkls(
    query_dir: str,
    short_gt_keys: Set[Tuple[str, str]],
) -> int:
    """Remove records from each *.pkl under query_dir whose
    (gt_binary, addr) is in short_gt_keys. Re-write each pkl in place.
    Returns the total number of records dropped across all pkls."""
    dropped = 0
    for query_path in sorted(Path(query_dir).glob("*.pkl")):
        infos = read_pickle(query_path)
        kept = []
        for info in infos:
            gt_binary = os.path.basename(str(info.get("gt_binary") or ""))
            gt_addr = str(info.get("addr"))
            if (gt_binary, gt_addr) in short_gt_keys:
                dropped += 1
                continue
            kept.append(info)
        if len(kept) != len(infos):
            write_pickle(kept, query_path)
    return dropped


def build_eval_pool(
    query_dir: str,
    pool_source_dir: PoolSourceArg,
    pool_size: int,
    output_dir: str,
    embedder=None,
    embed_force: bool = False,
    clean_output: bool = False,
    feature_suffix: str = INLINE_FEATURE_SUFFIX,
    embedding_suffix: str = INLINE_EMBED_SUFFIX,
    workers: int = DEFAULT_BUILD_WORKERS,
) -> Dict[str, object]:
    query_constraints = load_query_pool_constraints(query_dir)
    inventory = load_pool_inventory(
        pool_source_dir,
        feature_suffix=feature_suffix,
        embedding_suffix=embedding_suffix,
        workers=workers,
    )
    short_gt_keys = _identify_short_gt_keys(
        query_constraints["gt_entries"], inventory, MIN_PSEUDOCODE_LINES,
    )
    short_gt_dropped_records = 0
    if short_gt_keys:
        short_gt_dropped_records = _prune_query_pkls(query_dir, short_gt_keys)
        print(
            f"[pool-builder] pruned {short_gt_dropped_records} query records "
            f"from {query_dir} (GT pool func has <= {MIN_PSEUDOCODE_LINES} "
            f"pseudocode lines; {len(short_gt_keys)} distinct short GTs)",
            flush=True,
        )
        query_constraints = load_query_pool_constraints(query_dir)
    sibling_signatures = _identify_multi_config_sibling_signatures(
        query_constraints["gt_entries"], inventory,
    )
    existing = query_constraints["excluded_related_signatures"]
    new_from_siblings = sibling_signatures - existing
    query_constraints["excluded_related_signatures"] = existing | sibling_signatures
    if new_from_siblings:
        print(
            f"[pool-builder] added {len(new_from_siblings)} sibling "
            f"(binary, func_name) signatures to exclusion set "
            f"(same-family same-name duplicates that would interfere with GT-tuple match)",
            flush=True,
        )
    cve_clone_signatures, cve_clone_dropped = _identify_cve_source_clone_signatures(
        query_constraints["gt_entries"], inventory,
    )
    new_from_cve = cve_clone_signatures - query_constraints["excluded_related_signatures"]
    query_constraints["excluded_related_signatures"] = (
        query_constraints["excluded_related_signatures"] | cve_clone_signatures
    )
    if new_from_cve:
        unique_groups = {
            cve_source_signature(binary_name, func_name)
            for binary_name, func_name in cve_clone_signatures
        }
        unique_groups.discard(None)
        print(
            f"[pool-builder] added {len(new_from_cve)} CVE source-level clone "
            f"(binary, func_name) signatures to exclusion set "
            f"(dropped {cve_clone_dropped} non-GT clones across "
            f"{len(unique_groups)} (case_binname, func_name) groups)",
            flush=True,
        )
    gt_functions = query_constraints["gt_functions"]
    selected_by_binary = select_pool_functions(
        gt_functions,
        inventory,
        pool_size,
        excluded_related_signatures=query_constraints["excluded_related_signatures"],
    )
    manifest = materialize_selected_pool(
        selected_by_binary=selected_by_binary,
        gt_functions=gt_functions,
        inventory=inventory,
        output_dir=output_dir,
        pool_size=pool_size,
        query_dir=query_dir,
        pool_source_dir=pool_source_dir,
        embedder=embedder,
        embed_force=embed_force,
        clean_output=clean_output,
        feature_suffix=feature_suffix,
        embedding_suffix=embedding_suffix,
        workers=workers,
    )
    manifest["excluded_related_signature_count"] = int(
        len(query_constraints["excluded_related_signatures"])
    )
    manifest["multi_target_query_count"] = int(query_constraints["multi_target_query_count"])
    manifest["short_gt_min_pseudocode_lines"] = int(MIN_PSEUDOCODE_LINES)
    manifest["short_gt_pruned_query_records"] = int(short_gt_dropped_records)
    manifest["short_gt_distinct_count"] = int(len(short_gt_keys))
    manifest["cve_clone_signature_count"] = int(len(cve_clone_signatures))
    manifest["cve_clone_dropped_count"] = int(cve_clone_dropped)
    write_json(manifest, Path(output_dir) / "pool_manifest.json")
    return manifest


DEFAULT_QUERY_DIR = ROOT_DIR / "Coding" / "artifacts" / "tmp" / "query"
DEFAULT_POOL_SOURCE_DIR = ROOT_DIR / "Coding" / "data" / "inline" / "pool"
DEFAULT_POOL_SOURCE_DIRS = [str(DEFAULT_POOL_SOURCE_DIR)]
DEFAULT_POOL_SIZE = 50000
DEFAULT_OUTPUT_DIR = ROOT_DIR / "Coding" / "artifacts" / "tmp" / "pool"
DEFAULT_EMBED_MODEL = os.getenv("TRACE_ENCODER_PATH", "")


def main():
    parser = argparse.ArgumentParser(description="Build an eval pool that includes all GT functions first.")
    parser.add_argument("--query-dir", default=str(DEFAULT_QUERY_DIR))
    parser.add_argument(
        "--pool-source-dir",
        nargs="+",
        default=list(DEFAULT_POOL_SOURCE_DIRS),
        help="One or more pool source roots. Each is rglob-scanned for "
             "*_bb_slice_feature.json (or --feature-suffix). On binary_name "
             "collision across roots, the first listed root wins.",
    )
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--embed-model", default=str(DEFAULT_EMBED_MODEL),
                        help="Path to embedding model checkpoint. If set, selected pool functions are encoded on demand.")
    parser.add_argument("--embed-gpu-id", type=int, default=0)
    parser.add_argument("--embed-global-max-length", type=int, default=None)
    parser.add_argument("--embed-partial-max-length", type=int, default=None)
    parser.add_argument("--embed-global-batch-size", type=int, default=None)
    parser.add_argument("--embed-partial-batch-size", type=int, default=None)
    parser.add_argument("--embed-force", action="store_true",
                        help="Re-encode even if a cached pickle already holds the addr.")
    parser.add_argument("--clean-output", action="store_true",
                        help="Wipe output-dir first. By default embedding pkls are preserved for reuse.")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_BUILD_WORKERS,
        help=f"Parallel workers for inventory scan + materialize "
             f"(default {DEFAULT_BUILD_WORKERS}). Set to 1 to disable parallelism.",
    )
    parser.add_argument(
        "--feature-suffix",
        default=INLINE_FEATURE_SUFFIX,
        help=f"Filename suffix of feature.json (default: {INLINE_FEATURE_SUFFIX}, "
             "the BB-bounded F output). Pass `_inline_slice_feature.json` for the "
             "legacy varchain output.",
    )
    parser.add_argument(
        "--embedding-suffix",
        default=None,
        help="Filename suffix of the embedding pkl. If omitted, derived from "
             "--feature-suffix by replacing trailing `_feature.json` with "
             "`_embeddings.pkl`.",
    )
    args = parser.parse_args()
    if args.embedding_suffix is None:
        from utils import derive_embedding_suffix
        args.embedding_suffix = derive_embedding_suffix(args.feature_suffix)

    embedder = None
    if args.embed_model:
        raise SystemExit(
            "On-the-fly encoding needs the TRACE encoder, which is not "
            "distributed. Use the released pre-embedded pools instead.")

    manifest = build_eval_pool(
        query_dir=args.query_dir,
        pool_source_dir=args.pool_source_dir,
        pool_size=args.pool_size,
        output_dir=args.output_dir,
        embedder=embedder,
        embed_force=args.embed_force,
        clean_output=args.clean_output,
        feature_suffix=args.feature_suffix,
        embedding_suffix=args.embedding_suffix,
        workers=args.workers,
    )
    print(
        f"[pool] built {manifest['selected_function_count']} functions across "
        f"{manifest['selected_binary_count']} binaries at {args.output_dir} "
        f"(embedding_strategy={manifest.get('embedding_strategy')}, "
        f"embed_seconds={manifest.get('embed_seconds')})"
    )


if __name__ == "__main__":
    main()
