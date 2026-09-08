"""BB-bounded slice extractor (the new F).

Reads `*_feature.json` files (output of the IDA decompile stage), partitions
each function's pseudocode into basic blocks bounded by control-flow
keywords, and attaches 4 BB-local anchor channels + 1 function-level
global fingerprint.

Output format: `*_bb_slice_feature.json` next to the input file. Existing
slice consumers can read `blocks_pseudocode` as before (list of slices,
each with `line_index` and `pseudos`); new fields (`anchors`, `slice_kind`,
function-level `global_fp`) are additive.

Per-anchor evidence (see Coding/docs/research/F_definition.md):
  - CONTROL_KEYWORDS  : INLINE/IP Jaccard 0.60, BB-cov 88.3%
  - CONST_HEX         : INLINE/IP Jaccard 0.79, BB-cov  8.8%
  - DEREF_PATTERN     : INLINE/IP Jaccard 0.64, BB-cov 10.6%
  - STRING_LIT        : INLINE/IP Jaccard 0.53, BB-cov  7.6%
  - CFG_SHAPE (global): INLINE/IP Jaccard 0.37, function-level only

Multi-process (one binary per worker) with tqdm progress bar.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:
    print("ERROR: tqdm is required. pip install tqdm", file=sys.stderr)
    sys.exit(2)



_HEX_NOISE = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "f", "ff", "fff",
              "ffff"}

_HEX_RE = re.compile(r"\b0[xX]([0-9a-fA-F]+)(?:[uULl]+)?\b")
_STR_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')
_DEREF_TYPE_RE = re.compile(
    r"\*\s*\(\s*(_?(?:BYTE|WORD|DWORD|QWORD|OWORD|TBYTE))\s*\*?\s*\*?\s*\)"
)
_CTRL_KW_RE = re.compile(
    r"\b(if|for|while|switch|goto|return|break|continue)\b"
)
_LABEL_RE = re.compile(r"^\s*LABEL_\d+\s*:\s*$")

_PURE_BRACE_LINE_RE = re.compile(r"^[\s{};]+$")


def _is_inert_brace_line(ln: dict) -> bool:
    text = (ln.get("line") or "").strip()
    if not text or not _PURE_BRACE_LINE_RE.match(text):
        return False
    return not ln.get("line_ea")


def _bucket_count(n: int) -> str:
    if n == 0: return "0"
    if n == 1: return "1"
    if n <= 3: return "2-3"
    if n <= 7: return "4-7"
    if n <= 15: return "8-15"
    if n <= 31: return "16-31"
    if n <= 63: return "32-63"
    return "64+"



def split_into_bbs(parent_lines: List[dict]) -> List[List[dict]]:
    """Split a function's lines into basic blocks at control-flow boundaries.

    Boundary rules (deterministic, text-only):
      - A line containing any of {if, for, while, switch, goto, return,
        break, continue} CLOSES the BB it belongs to.
      - A `LABEL_n:` line STARTS a new BB (and is its first line).
      - The function entry starts a BB; the function end closes the last BB.

    Empty / whitespace-only lines are kept inside their containing BB
    (so the BB text stays readable when joined).
    """
    bbs: List[List[dict]] = []
    cur: List[dict] = []
    for ln in parent_lines or []:
        if _is_inert_brace_line(ln):
            continue
        text = (ln.get("line") or "").strip()
        if _LABEL_RE.match(text):
            if cur:
                bbs.append(cur)
            cur = [ln]
            continue
        cur.append(ln)
        if text and _CTRL_KW_RE.search(text):
            bbs.append(cur)
            cur = []
    if cur:
        bbs.append(cur)
    return [bb for bb in bbs if bb]


def extract_bb_anchors(bb_lines: List[dict]) -> Dict[str, List[str]]:
    """Extract the 4 BB-local anchor channels (sorted lists for JSON stability)."""
    out_sets = {
        "CONTROL_KEYWORDS": set(),
        "CONST_HEX": set(),
        "DEREF_PATTERN": set(),
        "STRING_LIT": set(),
    }
    for ln in bb_lines or []:
        text = ln.get("line") or ""
        if not text:
            continue
        for m in _CTRL_KW_RE.finditer(text):
            out_sets["CONTROL_KEYWORDS"].add(m.group(1))
        for m in _HEX_RE.finditer(text):
            h = m.group(1).lower().lstrip("0") or "0"
            if h not in _HEX_NOISE:
                out_sets["CONST_HEX"].add("0x" + h)
        for m in _DEREF_TYPE_RE.finditer(text):
            out_sets["DEREF_PATTERN"].add(m.group(1).strip())
        for m in _STR_RE.finditer(text):
            s = m.group(1)
            if s and len(s) >= 4:
                out_sets["STRING_LIT"].add(s)
    return {k: sorted(v) for k, v in out_sets.items()}


def extract_function_global_fp(parent_lines: List[dict]) -> Dict[str, str]:
    """Function-level CFG_SHAPE fingerprint (bucketed cf-keyword counts)."""
    n_branches = 0
    n_loops = 0
    n_returns = 0
    for ln in parent_lines or []:
        text = ln.get("line") or ""
        if not text:
            continue
        for m in _CTRL_KW_RE.finditer(text):
            kw = m.group(1)
            if kw in ("if", "switch"):
                n_branches += 1
            elif kw in ("for", "while"):
                n_loops += 1
            elif kw == "return":
                n_returns += 1
    return {
        "branches": _bucket_count(n_branches),
        "loops": _bucket_count(n_loops),
        "returns": _bucket_count(n_returns),
        "n_branches_raw": n_branches,
        "n_loops_raw": n_loops,
        "n_returns_raw": n_returns,
    }


def process_function(rec: dict) -> dict:
    """Build the new BB-based slice record from a feature.json function record."""
    parent_lines = rec.get("lines") or []
    bbs = split_into_bbs(parent_lines)
    blocks_pseudocode = []
    for bb_id, bb in enumerate(bbs):
        line_indices = [int(ln.get("line_index"))
                        for ln in bb if ln.get("line_index") is not None]
        text_lines = [ln.get("line", "") for ln in bb]
        eas: List[str] = []
        for ln in bb:
            ea = ln.get("line_ea")
            if ea:
                eas.append(str(ea))
        anchors = extract_bb_anchors(bb)
        blocks_pseudocode.append({
            "slice_kind": "bb",
            "bb_id": bb_id,
            "line_index": line_indices,
            "pseudos": "\n".join(text_lines),
            "anchors": anchors,
            "n_lines": len(bb),
            "line_eas": eas,
        })
    out = {
        "func_addr": rec.get("func_addr"),
        "func_name": rec.get("func_name"),
        "debug_func_name": rec.get("debug_func_name") or rec.get("func_name"),
        "lines": parent_lines,
        "blocks_pseudocode": blocks_pseudocode,
        "global_fp": extract_function_global_fp(parent_lines),
        "n_bbs": len(bbs),
        "n_lines_total": len(parent_lines),
    }
    return out


def process_binary(args: Tuple[Path, Path, bool]) -> Dict:
    """Worker: process a single <bn>_feature.json into <bn>_bb_slice_feature.json.

    Returns a stat dict for tqdm aggregation.
    """
    in_path, out_path, force = args
    stats = {"in": str(in_path), "out": str(out_path),
             "n_funcs": 0, "n_bbs": 0, "n_lines": 0,
             "skipped": False, "error": None, "elapsed": 0.0}
    t0 = time.time()
    try:
        if out_path.exists() and not force:
            stats["skipped"] = True
            return stats
        with in_path.open() as f:
            features = json.load(f)
        if isinstance(features, list):
            features = {(rec.get("func_addr") or f"idx{i}"): rec
                        for i, rec in enumerate(features) if isinstance(rec, dict)}
        if not isinstance(features, dict):
            stats["error"] = f"unexpected top-level type {type(features).__name__}"
            return stats

        out: Dict[str, dict] = {}
        for key, rec in features.items():
            if not isinstance(rec, dict):
                continue
            try:
                processed = process_function(rec)
            except Exception as e:
                continue
            out[key] = processed
            stats["n_funcs"] += 1
            stats["n_bbs"] += processed["n_bbs"]
            stats["n_lines"] += processed["n_lines_total"]

        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w") as f:
            json.dump(out, f, indent=4)
        tmp_path.replace(out_path)
    except Exception as e:
        stats["error"] = f"{type(e).__name__}: {e}"
        stats["traceback"] = traceback.format_exc()
    finally:
        stats["elapsed"] = time.time() - t0
    return stats



_RAW_SUFFIX = "_feature.json"
_SLICE_SUFFIX = "_inline_slice_feature.json"
_BB_SUFFIX = "_bb_slice_feature.json"
_CANDIDATES_SUFFIX = "_inline_slice_candidates.json"


def _input_base(p: Path) -> Optional[Tuple[str, str]]:
    """Return (parent, base) where base = filename minus a known input suffix.

    Returns None if the file is not a recognized input.
    """
    name = p.name
    if name.endswith(_BB_SUFFIX) or name.endswith(_CANDIDATES_SUFFIX):
        return None
    if name.endswith(_SLICE_SUFFIX):
        return (str(p.parent), name[:-len(_SLICE_SUFFIX)])
    if name.endswith(_RAW_SUFFIX):
        return (str(p.parent), name[:-len(_RAW_SUFFIX)])
    return None


def find_feature_files(input_paths: List[Path],
                        allow_slice_fallback: bool = True) -> List[Path]:
    """Resolve --input args into a list of input feature files.

    Default behavior: for each (parent, base), prefer `<base>_feature.json`
    (raw IDA decompile output). If raw is missing AND allow_slice_fallback,
    use `<base>_inline_slice_feature.json` instead (still has `lines` field).
    """
    raw_by_base: Dict[Tuple[str, str], Path] = {}
    slice_by_base: Dict[Tuple[str, str], Path] = {}

    def _scan_file(f: Path) -> None:
        base = _input_base(f)
        if base is None:
            return
        if f.name.endswith(_RAW_SUFFIX) and not f.name.endswith(_SLICE_SUFFIX):
            raw_by_base[base] = f
        elif f.name.endswith(_SLICE_SUFFIX):
            slice_by_base[base] = f

    for p in input_paths:
        if p.is_file():
            _scan_file(p)
        elif p.is_dir():
            for pat in (f"*{_RAW_SUFFIX}", f"*{_SLICE_SUFFIX}"):
                for f in sorted(p.rglob(pat)):
                    _scan_file(f)

    out: List[Path] = []
    seen: Set[Tuple[str, str]] = set()
    for base, fp in raw_by_base.items():
        out.append(fp)
        seen.add(base)
    if allow_slice_fallback:
        for base, fp in slice_by_base.items():
            if base not in seen:
                out.append(fp)
    return sorted(out)


def _derive_output_path(in_path: Path, out_dir: Optional[Path]) -> Path:
    """Replace the input suffix with _bb_slice_feature.json."""
    name = in_path.name
    if name.endswith(_SLICE_SUFFIX):
        out_name = name[:-len(_SLICE_SUFFIX)] + _BB_SUFFIX
    elif name.endswith(_RAW_SUFFIX):
        out_name = name[:-len(_RAW_SUFFIX)] + _BB_SUFFIX
    else:
        out_name = name.rsplit(".", 1)[0] + _BB_SUFFIX
    return (out_dir / out_name) if out_dir else in_path.with_name(out_name)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract BB-bounded slices from *_feature.json files.")
    ap.add_argument("--input", nargs="+", required=True, type=Path,
                    help="One or more feature.json files OR directories "
                         "containing them (recursive search).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="If provided, write all outputs to this dir; "
                         "otherwise write next to each input.")
    ap.add_argument("--no-slice-fallback", action="store_true",
                    help="Do NOT fall back to *_inline_slice_feature.json when "
                         "*_feature.json is missing for the same basename.")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 1),
                    help="Worker processes (default: cpu_count-1)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing outputs (default: skip)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N files (debug)")
    args = ap.parse_args()

    feature_files = find_feature_files(
        args.input, allow_slice_fallback=not args.no_slice_fallback)
    if args.limit:
        feature_files = feature_files[:args.limit]
    if not feature_files:
        print("No input files found.", file=sys.stderr)
        return 1
    n_raw = sum(1 for f in feature_files if f.name.endswith(_RAW_SUFFIX)
                 and not f.name.endswith(_SLICE_SUFFIX))
    n_slice = sum(1 for f in feature_files if f.name.endswith(_SLICE_SUFFIX))
    print(f"Found {len(feature_files)} input files "
          f"(raw={n_raw}, slice-fallback={n_slice}); workers={args.workers}; "
          f"force={args.force}", flush=True)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks: List[Tuple[Path, Path, bool]] = [
        (fp, _derive_output_path(fp, args.out_dir), args.force)
        for fp in feature_files
    ]

    n_done = 0
    n_skipped = 0
    n_errored = 0
    n_funcs_total = 0
    n_bbs_total = 0
    n_lines_total = 0
    errors: List[Dict] = []

    if args.workers <= 1:
        iterator = (process_binary(t) for t in tasks)
        results_iter = iterator
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(args.workers)
        results_iter = pool.imap_unordered(process_binary, tasks, chunksize=1)

    try:
        with tqdm(total=len(tasks), desc="binaries", unit="bin", smoothing=0.1) as pbar:
            for stats in results_iter:
                if stats.get("error"):
                    n_errored += 1
                    errors.append(stats)
                elif stats.get("skipped"):
                    n_skipped += 1
                else:
                    n_done += 1
                    n_funcs_total += stats["n_funcs"]
                    n_bbs_total += stats["n_bbs"]
                    n_lines_total += stats["n_lines"]
                pbar.update(1)
                pbar.set_postfix(
                    done=n_done, skip=n_skipped, err=n_errored,
                    funcs=n_funcs_total, bbs=n_bbs_total,
                )
    finally:
        if args.workers > 1:
            pool.close()
            pool.join()

    print()
    print(f"Summary:")
    print(f"  processed = {n_done}")
    print(f"  skipped (already exists)  = {n_skipped}")
    print(f"  errored = {n_errored}")
    print(f"  total functions = {n_funcs_total}")
    print(f"  total BBs       = {n_bbs_total}")
    print(f"  total lines     = {n_lines_total}")
    if n_funcs_total:
        print(f"  avg BBs/func   = {n_bbs_total/n_funcs_total:.2f}")
        print(f"  avg lines/func = {n_lines_total/n_funcs_total:.1f}")
    if errors:
        print(f"\nFirst {min(5, len(errors))} errors:")
        for e in errors[:5]:
            print(f"  {e['in']}: {e['error']}")
    return 0 if n_errored == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
