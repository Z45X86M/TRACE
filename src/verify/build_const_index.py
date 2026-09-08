"""Offline: per-function distinctive-constant sets (dec+hex) + global DF, pickled.

  python3 Coding/build_const_index.py -t Coding/data/inline/pool -w 12
  -> <dir>/const_index.pkl = {"consts": {(binary, addr_int): [v,...]}, "df": {v: count}, "n": N}
"""
import argparse, json, os, pickle, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
for p in (str(CURRENT_DIR), str(CURRENT_DIR.parent)):
    if p not in sys.path: sys.path.append(p)
from const_anchor import func_consts

SUF = "_bb_slice_feature.json"

def _na(a):
    if isinstance(a, int): return a
    s = str(a).strip()
    try: return int(s, 16) if s.lower().startswith("0x") else int(s)
    except ValueError:
        try: return int(s, 16)
        except ValueError: return s

def one(args):
    path, stem = args
    out = {}
    try:
        d = json.load(open(path))
    except Exception:
        return out
    for k, v in d.items():
        fa = _na(v.get("func_addr", k))
        cs = func_consts([b.get("pseudos", "") for b in v.get("blocks_pseudocode", [])])
        out[(stem, fa)] = sorted(cs)
    return out

def main(target, workers):
    files = [f for f in sorted(os.listdir(target)) if f.endswith(SUF)]
    tasks = [(os.path.join(target, f), f[:-len(SUF)]) for f in files]
    print("[const-index] %d binaries under %s" % (len(tasks), target), flush=True)
    consts = {}; df = defaultdict(int); t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(one, t) for t in tasks]):
            r = fut.result(); consts.update(r)
            for cs in r.values():
                for v in cs: df[v] += 1
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print("[const-index] %d/%d, funcs=%d vocab=%d %.0fs"
                      % (done, len(tasks), len(consts), len(df), time.time()-t0), flush=True)
    out = os.path.join(target, "const_index.pkl")
    with open(out, "wb") as f:
        pickle.dump({"consts": consts, "df": dict(df), "n": len(consts)}, f, protocol=pickle.HIGHEST_PROTOCOL)
    nonempty = sum(1 for cs in consts.values() if cs)
    print("[const-index] wrote %s funcs=%d with-const=%d (%.1f%%) vocab=%d %.1fMB %.0fs"
          % (out, len(consts), nonempty, 100.0*nonempty/max(1,len(consts)), len(df),
             os.path.getsize(out)/1e6, time.time()-t0), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("-t","--target",required=True)
    ap.add_argument("-w","--workers",type=int,default=12); a=ap.parse_args()
    main(a.target, a.workers)
