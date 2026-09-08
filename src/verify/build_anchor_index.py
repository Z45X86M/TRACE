"""Build an offline anchor index for the symbolic-anchor verifier.

Scans a pool of *_bb_slice_feature.json once, extracts the 3-channel anchors
(CONST_HEX / STRING_LIT / CALL), computes per-channel IDF and per-candidate Dice
mass, and pickles {pool, idf, pmass} so that pool_inline_search.py --anchor can
load it instantly via --anchor-index instead of rebuilding online (which scans
the whole pool and regexes every function — minutes on a 1M pool).

Build the index over the SAME pool you will search (its IDF and candidate keys
must match). Example:

  python3 Coding/build_anchor_index.py --pool-dir Coding/data/inline/pool
  # -> Coding/data/inline/pool/anchor_index_bb_slice_feature.pkl

  # then at retrieval time:
  python3 Coding/pool_inline_search.py --anchor \
      --anchor-pool-dir Coding/data/inline/pool \
      --anchor-index   Coding/data/inline/pool/anchor_index_bb_slice_feature.pkl ...
"""
import argparse
import os
import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
for _p in (str(CURRENT_DIR), str(CURRENT_DIR.parent)):
    if _p not in sys.path:
        sys.path.append(_p)

import verifiers  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-dir", default=str(CURRENT_DIR / "data" / "inline" / "pool"),
                    help="Dir holding the pool *_bb_slice_feature.json (default data/inline/pool).")
    ap.add_argument("--feature-suffix", default="_bb_slice_feature.json")
    ap.add_argument("--out", default=None,
                    help="Output pickle path (default <pool-dir>/anchor_index_<suffix>.pkl).")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild even if the output already exists.")
    args = ap.parse_args()

    out = args.out or verifiers.default_anchor_index_path(args.pool_dir, args.feature_suffix)
    if os.path.exists(out) and not args.force:
        print("[anchor-index] %s already exists; use --force to rebuild." % out)
        return 0

    t0 = time.time()
    print("[anchor-index] scanning pool %s (suffix=%s) ..." % (args.pool_dir, args.feature_suffix),
          flush=True)
    pool = verifiers.build_anchor_pool_map(args.pool_dir, args.feature_suffix)
    print("[anchor-index] pool funcs=%d  building idf + Dice mass ..." % len(pool), flush=True)
    idf = verifiers.build_anchor_idf(pool)
    pmass = {k: verifiers.anchor_overlap(a, a, idf) for k, a in pool.items()}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    verifiers.save_anchor_index(out, pool, idf, pmass, args.feature_suffix)
    sz = os.path.getsize(out) / 1e6
    print("[anchor-index] wrote %s  (%d funcs, vocab hex=%d str=%d call=%d, %.1f MB, %.1fs)"
          % (out, len(pool), len(idf[0]), len(idf[1]), len(idf[2]), sz, time.time() - t0),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
