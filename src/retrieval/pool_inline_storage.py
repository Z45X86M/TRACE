"""Inline slice pool — storage + I/O.

Reads per-binary `*_bb_slice_feature.json` + `*_bb_slice_embeddings.pkl`,
quantizes partials via FAISS IVF, builds bucket → (func_id, overall_signal)
inverted index + bucket IDF/hot/df stats + per-function flat partials/globals
mmaps. Everything mmap-loaded → RAM peak = hot pages only, not pool size;
10M pool fits the same code path. Recall lives in pool_inline_search.py.
"""

import math
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
for path in (ROOT_DIR, CURRENT_DIR):
    if path not in sys.path:
        sys.path.append(path)

from utils import discover_binaries
from utils import read_json, read_pickle, write_pickle


DEFAULT_POOL_READ_WORKERS = 12


def _read_binary_for_pool(args: Tuple[str, str, str]) -> Optional[
    Tuple[str, Dict[Tuple[Any, str], Tuple[np.ndarray, np.ndarray]],
          Dict[Tuple[Any, str], Dict[str, Any]], int]
]:
    """Worker: read one binary's feature.json + embedding.pkl, return cleaned dicts.

    Mirrors ``InlinePool._read_one_binary`` but without instance state — picklable
    for use with ``ProcessPoolExecutor``. The caller validates ``embedding_dim``
    consistency across binaries at result-receipt time.
    """
    binary, feature_suffix, embedding_suffix = args
    fp = f"{binary}{feature_suffix}"
    ep = f"{binary}{embedding_suffix}"
    if not (os.path.exists(fp) and os.path.exists(ep)):
        return None
    feat = read_json(fp)
    emb = read_pickle(ep)
    cleaned: Dict[Tuple[Any, str], Tuple[np.ndarray, np.ndarray]] = {}
    fmap: Dict[Tuple[Any, str], Dict[str, Any]] = {}
    dim: Optional[int] = None
    for addr in feat:
        if addr not in emb:
            continue
        fn = feat[addr]["func_name"]
        eg = np.asarray(emb[addr]["global"], dtype=np.float32).reshape(-1)
        ep_arr = np.asarray(emb[addr]["partial"], dtype=np.float32)
        if ep_arr.size == 0:
            continue
        if ep_arr.ndim == 1:
            ep_arr = ep_arr.reshape(1, -1)
        if ep_arr.ndim != 2 or eg.ndim != 1 or ep_arr.shape[1] == 0:
            continue
        cleaned[(addr, fn)] = (eg, ep_arr)
        fmap[(addr, fn)] = feat[addr]
        if dim is None:
            dim = int(eg.shape[0])
    if not cleaned:
        return None
    return os.path.basename(binary), cleaned, fmap, int(dim or 0)


class InlinePool:
    DB_VERSION = 26
    DEFAULT_BTOPK = 4
    DEFAULT_POOL_BUCKET_TOPK = 24
    HOT_BUCKET_QUERY_WEIGHT_CAP = 0.55
    HOT_BUCKET_MIN_WEIGHT = 0.35

    def __init__(
        self,
        pool_dir: Optional[str],
        model: str,
        save_db: Optional[str],
        feature_suffix: str = "_bb_slice_feature.json",
        embedding_suffix: str = "_bb_slice_embeddings.pkl",
        faiss_gpu_id: int = 0,
    ):
        self.binaries: List[str] = discover_binaries(pool_dir) if pool_dir else []
        self.save_db = save_db
        self.feature_suffix = feature_suffix
        self.embedding_suffix = embedding_suffix
        self.index_to_meta: List[Tuple[str, Any, str]] = []
        self.meta_to_index: Dict[Tuple[str, Any, str], int] = {}
        self.embedding_dim: Optional[int] = None
        self.bucket_topk = self.DEFAULT_BTOPK
        self.pool_bucket_topk = self.DEFAULT_POOL_BUCKET_TOPK
        self.stop_bucket_df_ratio = 0.18
        self.stop_bucket_min_posting = 128
        self.hot_bucket_posting_threshold = 0
        for name in (
            "bucket_func_offsets", "bucket_func_ids", "bucket_func_overall_signal",
            "bucket_idf", "bucket_posting_len", "bucket_df_ratio", "bucket_hot_weight",
            "func_globals", "func_slice_count", "func_total_bucket_support",
            "func_partials_flat", "func_partial_offsets",
        ):
            setattr(self, name, None)
        self.index = faiss.read_index(model)
        self.pca = faiss.downcast_VectorTransform(self.index.chain.at(0))
        self.ivf = faiss.downcast_index(self.index.index)
        self.res: Optional[Any] = None
        self.gpu_quantizer = self.ivf.quantizer
        self.use_gpu_quantizer = False
        try:
            if hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu"):
                self.res = faiss.StandardGpuResources()
                self.gpu_quantizer = faiss.index_cpu_to_gpu(self.res, int(faiss_gpu_id), self.ivf.quantizer)
                self.use_gpu_quantizer = True
        except Exception:
            pass

    def _assign_bucket_hits_arrays(
        self, q: np.ndarray, topk: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized core. Returns (ids, affinity, valid) each shape (N, topk).

        - One FAISS search call regardless of N — batch this when caller has
          many slices, not slice-by-slice.
        - The legacy Python dedup over duplicate bucket ids per row is dropped:
          the IVF quantizer returns distinct centroid ids in its top-k, so the
          dedup was a no-op safety net.
        """
        q = np.asarray(q, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if topk is None:
            topk = self.bucket_topk
        if self.embedding_dim is not None and q.shape[1] != self.embedding_dim:
            raise ValueError(f"dim mismatch: {q.shape[1]} vs {self.embedding_dim}")
        if q.shape[0] == 0:
            return (np.empty((0, topk), dtype=np.int32),
                    np.empty((0, topk), dtype=np.float32),
                    np.empty((0, topk), dtype=bool))
        distances, ids = self.gpu_quantizer.search(self.pca.apply_py(q), topk)
        ids = np.asarray(ids, dtype=np.int64)
        dists = np.asarray(distances, dtype=np.float32)
        valid = ids >= 0
        is_l2 = self.ivf.metric_type == faiss.METRIC_L2
        dists_for_min = np.where(valid, dists, np.float32(np.inf))
        best_d = dists_for_min.min(axis=1, keepdims=True)
        best_d = np.where(np.isfinite(best_d), best_d, np.float32(0.0))
        if is_l2:
            delta = np.maximum(0.0, dists - best_d)
            scale = np.maximum(best_d * 2.5, 1e-4)
        else:
            delta = np.maximum(0.0, best_d - dists)
            scale = np.maximum(np.abs(best_d) * 0.15, 1e-4)
        rank = np.arange(ids.shape[1], dtype=np.float32).reshape(1, -1)
        affinity = (np.exp(-delta / scale) / np.sqrt(rank + 1.0)).astype(np.float32)
        affinity = np.where(valid, affinity, np.float32(0.0))
        return ids.astype(np.int32, copy=False), affinity, valid

    def assign_bucket_hits(self, q: np.ndarray, topk: Optional[int] = None) -> List[List[Tuple[int, float]]]:
        """Each row of q -> top-K FAISS buckets as (bucket_id, affinity); affinity
        in (0,1] = distance softmax x rank decay."""
        ids, affinity, valid = self._assign_bucket_hits_arrays(q, topk)
        rows: List[List[Tuple[int, float]]] = []
        for ids_row, aff_row, val_row in zip(ids, affinity, valid):
            rows.append([
                (int(b), float(a))
                for b, a, v in zip(ids_row.tolist(), aff_row.tolist(), val_row.tolist())
                if v
            ])
        return rows

    _KIND_WEIGHT = {
        "var_chain": 1.25, "coverage_chain": 0.70, "cover": 0.80, "guarded_dispatch": 1.35,
        "predicate_accept": 1.20, "postprocess_tail": 1.10, "iterator_backbone": 0.95,
        "anchor_bridge": 1.25, "context": 1.15, "anchor": 1.05, "api_hybrid": 0.95,
        "ret_forward": 0.90, "straight": 0.85, "skeleton": 0.75, "anchor_micro": 0.70,
    }

    @classmethod
    def _row_kind_weight(cls, kind: str) -> float:
        return cls._KIND_WEIGHT.get(kind or "unknown", 1.0)

    @classmethod
    def _coarse_meta_weight(cls, meta: Dict[str, Any]) -> float:
        kind_w = cls._row_kind_weight(meta.get("slice_kind", "unknown"))
        quality = min(1.50, max(0.35, float(meta.get("slice_quality", 1.0) or 1.0)))
        role_w = 1.0 if (meta.get("semantic_role", "primary") or "primary") == "primary" else 0.25
        gp = float(meta.get("generic_penalty", 0.0) or 0.0)
        generic_w = max(0.45, 1.0 - 0.50 * min(1.0, max(0.0, gp)))
        return kind_w * quality * role_w * generic_w

    @staticmethod
    def _normalize_embedding_rows(emb: np.ndarray) -> np.ndarray:
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if emb.size == 0:
            return emb.reshape(0, emb.shape[-1] if emb.ndim == 2 else 0)
        norms = np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        return (emb / norms).astype(np.float32)

    @classmethod
    def _normalize_vector(cls, vec: np.ndarray) -> np.ndarray:
        return cls._normalize_embedding_rows(np.asarray(vec, dtype=np.float32).reshape(1, -1))[0]

    @classmethod
    def _weighted_mean_embedding(cls, emb: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[0] == 0:
            return np.empty((emb.shape[1] if emb.ndim == 2 else 0,), dtype=np.float32)
        if weights is None:
            mean = np.mean(emb, axis=0)
        else:
            weights = np.asarray(weights, dtype=np.float32).reshape(-1)
            if weights.shape[0] != emb.shape[0] or float(np.sum(weights)) <= 0.0:
                mean = np.mean(emb, axis=0)
            else:
                mean = np.average(emb, axis=0, weights=weights)
        return cls._normalize_vector(mean)

    @staticmethod
    def _compose_bucket_func_signal(
        best: np.ndarray, support: np.ndarray, bucket_count: np.ndarray,
    ) -> np.ndarray:
        """Per-(bucket, func) signal = 0.58·best_sat + 0.30·support_sat + 0.12·bucket_count_score."""
        best = np.asarray(best, dtype=np.float32)
        support = np.asarray(support, dtype=np.float32)
        bucket_count = np.asarray(bucket_count, dtype=np.float32)
        best_sat = 1.0 - np.exp(-1.6 * np.clip(best, 0.0, None))
        support_sat = 1.0 - np.exp(-0.85 * np.clip(support, 0.0, None))
        bucket_count_score = 1.0 / np.sqrt(np.maximum(1.0, bucket_count))
        signal = 0.58 * best_sat + 0.30 * support_sat + 0.12 * bucket_count_score
        signal = np.clip(signal, 0.0, 1.0).astype(np.float32, copy=False)
        signal[(best <= 0.0) & (support <= 0.0)] = 0.0
        return signal

    def _runtime_path(self, name: str) -> str:
        return os.path.join(self.save_db, f"{name}.npy")

    def _save_runtime_array(self, name: str, array: np.ndarray):
        np.save(self._runtime_path(name), np.asarray(array))

    def _load_runtime_array(self, name: str, mmap: bool = False, cast: Optional[np.dtype] = None) -> np.ndarray:
        array = np.load(self._runtime_path(name), mmap_mode="r" if mmap else None)
        return np.asarray(array, dtype=cast) if cast is not None else array

    def _open_runtime_memmap(self, name: str, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """Writable mmap .npy; empty shape (any dim==0) → on-disk zeros placeholder."""
        if any(int(d) <= 0 for d in shape):
            placeholder = np.zeros(tuple(max(0, int(d)) for d in shape), dtype=dtype)
            np.save(self._runtime_path(name), placeholder)
            return placeholder
        return np.lib.format.open_memmap(self._runtime_path(name), mode="w+", dtype=dtype, shape=tuple(int(d) for d in shape))

    def _bucket_func_slice(self, bucket_id: int) -> slice:
        if self.bucket_func_offsets is None or bucket_id < 0 or bucket_id + 1 >= self.bucket_func_offsets.shape[0]:
            return slice(0, 0)
        return slice(int(self.bucket_func_offsets[bucket_id]), int(self.bucket_func_offsets[bucket_id + 1]))

    def _bucket_func_posting(self, bucket_id: int, limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (func_ids, overall_signal) for one bucket; ordered at build time
        by (0.70·best + 0.30·support_sat) DESC + tie-breaks."""
        if self.bucket_func_ids is None or self.bucket_func_overall_signal is None:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        s = self._bucket_func_slice(bucket_id)
        func_ids = np.asarray(self.bucket_func_ids[s], dtype=np.int32)
        signal = np.asarray(self.bucket_func_overall_signal[s], dtype=np.float32)
        if limit is not None:
            func_ids, signal = func_ids[:limit], signal[:limit]
        return func_ids, signal

    def _refresh_hot_bucket_threshold(self):
        """Re-derive bucket_hot_weight from bucket_posting_len: posting_len > p99
        gets a 1/sqrt(ratio) penalty floored at HOT_BUCKET_MIN_WEIGHT."""
        if self.bucket_posting_len is None or self.bucket_posting_len.size == 0:
            self.hot_bucket_posting_threshold = 0
            self.bucket_hot_weight = None
            return
        nonempty = np.asarray(self.bucket_posting_len[self.bucket_posting_len > 0], dtype=np.int32)
        if nonempty.size == 0:
            self.hot_bucket_posting_threshold = 0
            self.bucket_hot_weight = np.ones_like(self.bucket_posting_len, dtype=np.float32)
            return
        p99 = int(max(1, math.ceil(float(np.percentile(nonempty, 99)))))
        self.hot_bucket_posting_threshold = p99
        ratios = np.asarray(self.bucket_posting_len, dtype=np.float32) / float(max(1, p99))
        hot_weight = np.ones_like(ratios, dtype=np.float32)
        hot_mask = ratios > 1.0
        hot_weight[hot_mask] = np.maximum(
            self.HOT_BUCKET_MIN_WEIGHT, 1.0 / np.sqrt(ratios[hot_mask]),
        ).astype(np.float32)
        self.bucket_hot_weight = hot_weight.astype(np.float32, copy=False)

    def _bucket_query_weight(self, bucket_id: int) -> float:
        """weight = idf × min(soft_df_penalty × soft_posting_penalty)
                  × stop_bucket_penalty × hot_weight,  capped at HOT_BUCKET_QUERY_WEIGHT_CAP."""
        if self.bucket_idf is None or self.bucket_posting_len is None or self.bucket_df_ratio is None:
            return 1.0
        if bucket_id < 0 or bucket_id >= self.bucket_idf.shape[0]:
            return 1.0
        base = float(self.bucket_idf[bucket_id])
        posting_len = int(self.bucket_posting_len[bucket_id])
        df_ratio = float(self.bucket_df_ratio[bucket_id])
        soft_df = max(0.15, 1.0 - 2.5 * df_ratio)
        soft_post = 1.0 / math.sqrt(max(1.0, posting_len / 32.0))
        weight = base * max(0.18, soft_df * soft_post)
        if posting_len >= self.stop_bucket_min_posting and df_ratio >= self.stop_bucket_df_ratio:
            df_pen = self.stop_bucket_df_ratio / max(df_ratio, self.stop_bucket_df_ratio)
            post_pen = self.stop_bucket_min_posting / max(posting_len, self.stop_bucket_min_posting)
            weight *= max(0.18, math.sqrt(df_pen * post_pen))
        hot_w = 1.0
        if self.bucket_hot_weight is not None and bucket_id < self.bucket_hot_weight.shape[0]:
            hot_w = float(self.bucket_hot_weight[bucket_id])
        return float(min(weight, self.HOT_BUCKET_QUERY_WEIGHT_CAP) * hot_w)

    def _read_one_binary(self, binary: str) -> Tuple[
        Dict[Tuple[Any, str], Tuple[np.ndarray, np.ndarray]],
        Dict[Tuple[Any, str], Dict[str, Any]],
    ]:
        fp = f"{binary}{self.feature_suffix}"
        ep = f"{binary}{self.embedding_suffix}"
        if not (os.path.exists(fp) and os.path.exists(ep)):
            return {}, {}
        feat, emb = read_json(fp), read_pickle(ep)
        cleaned: Dict[Tuple[Any, str], Tuple[np.ndarray, np.ndarray]] = {}
        fmap: Dict[Tuple[Any, str], Dict[str, Any]] = {}
        for addr in feat:
            if addr not in emb:
                continue
            fn = feat[addr]["func_name"]
            eg = np.asarray(emb[addr]["global"], dtype=np.float32).reshape(-1)
            ep_arr = np.asarray(emb[addr]["partial"], dtype=np.float32)
            if ep_arr.size == 0:
                continue
            if ep_arr.ndim == 1:
                ep_arr = ep_arr.reshape(1, -1)
            if ep_arr.ndim != 2 or eg.ndim != 1 or ep_arr.shape[1] == 0:
                continue
            cleaned[(addr, fn)] = (eg, ep_arr)
            fmap[(addr, fn)] = feat[addr]
        if cleaned:
            dim = next(iter(cleaned.values()))[0].shape[0]
            if self.embedding_dim is None:
                self.embedding_dim = dim
            elif self.embedding_dim != dim:
                raise ValueError(f"embedding dim mismatch in {binary}")
        return cleaned, fmap

    def _iter_pool_binaries(self):
        """Stream binaries one at a time so RAM peak = one binary."""
        for binary in self.binaries:
            cleaned, feature_map = self._read_one_binary(binary)
            if cleaned:
                yield os.path.basename(binary), cleaned, feature_map

    def build_pool(self):
        """3-pass build, RAM peak ≈ all per-binary cleaned arrays cached between P1/P2.
          P1 discovery: scan binaries → func_ids, partial offsets; cache (cleaned, fmap)
            so P2 doesn't re-read the same json+pkl. At pool_size=100K this is ~2GB;
            for >1M-func pools the caller should split the pool into shards.
          P2 materialize: per-binary BATCHED FAISS search over all slices at once,
            numpy aggregation of (bucket, func) → (best, support), spill to 256
            bucket-sharded files.
          P3 postings: read each shard, lexsort by (bid, composite DESC, …),
             materialize bucket_func_ids + overall_signal, save bucket meta.
        """
        pool_path = os.path.join(self.save_db, "pool_inline_slice.pkl")
        if os.path.exists(pool_path):
            try:
                self.load_db()
                return
            except (ValueError, FileNotFoundError):
                pass
        os.makedirs(self.save_db, exist_ok=True)
        self.index_to_meta = []
        self.meta_to_index = {}

        binary_entry_keys: Dict[str, List[Tuple[Any, str]]] = {}
        binary_iteration_order: List[str] = []
        binary_cache: Dict[str, Tuple[
            Dict[Tuple[Any, str], Tuple[np.ndarray, np.ndarray]],
            Dict[Tuple[Any, str], Dict[str, Any]],
        ]] = {}
        offsets_l: List[int] = [0]
        slice_counts: List[int] = []

        def _absorb(binary_name, cleaned, feature_map, dim):
            if self.embedding_dim is None:
                self.embedding_dim = int(dim)
            elif self.embedding_dim != int(dim):
                raise ValueError(f"embedding dim mismatch in {binary_name}")
            keys: List[Tuple[Any, str]] = []
            for (addr, func_name), (_emb_g, partial) in sorted(
                cleaned.items(), key=lambda kv: (str(kv[0][0]), kv[0][1]),
            ):
                func_id = len(self.index_to_meta)
                self.meta_to_index[(binary_name, addr, func_name)] = func_id
                self.index_to_meta.append((binary_name, addr, func_name))
                slice_counts.append(int(partial.shape[0]))
                offsets_l.append(offsets_l[-1] + int(partial.shape[0]))
                keys.append((addr, func_name))
            binary_entry_keys[binary_name] = keys
            binary_iteration_order.append(binary_name)
            binary_cache[binary_name] = (cleaned, feature_map)

        workers = int(getattr(self, "p1_read_workers", DEFAULT_POOL_READ_WORKERS) or 1)
        bar = tqdm(
            total=len(self.binaries), desc="build pool: discover funcs",
            dynamic_ncols=True,
        )
        if workers <= 1 or len(self.binaries) <= 1:
            for binary in self.binaries:
                result = _read_binary_for_pool((
                    binary, self.feature_suffix, self.embedding_suffix,
                ))
                if result is not None:
                    _absorb(*result)
                bar.update(1)
        else:
            ctx = multiprocessing.get_context("spawn")
            args_list = [
                (binary, self.feature_suffix, self.embedding_suffix)
                for binary in self.binaries
            ]
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                for result in ex.map(_read_binary_for_pool, args_list, chunksize=2):
                    if result is not None:
                        _absorb(*result)
                    bar.update(1)
        bar.close()

        total_funcs = len(self.index_to_meta)
        total_partials = int(offsets_l[-1])
        dim = int(self.embedding_dim or 0)
        func_partial_offsets = np.asarray(offsets_l, dtype=np.int64)
        func_slice_count = np.asarray(slice_counts, dtype=np.int32)
        print(f"[build pool] total_funcs={total_funcs} total_partials={total_partials} dim={dim}", flush=True)

        flat_mm = self._open_runtime_memmap(
            "func_partials_flat",
            (max(1, total_partials), max(1, dim)) if total_partials > 0 and dim > 0 else (0, max(1, dim)),
            dtype=np.float32,
        )
        globals_mm = self._open_runtime_memmap("func_globals", (total_funcs, max(1, dim)), dtype=np.float32)

        SHARD_COUNT = 256
        shard_dtype = np.dtype([("bucket_id", "<i4"), ("func_id", "<i4"),
                                ("best", "<f4"), ("support", "<f4")])
        func_bucket_count = np.zeros(total_funcs, dtype=np.int32)
        max_bucket_id = -1

        shard_dir = os.path.join(self.save_db, "_posting_shards")
        _wipe_dir(shard_dir)
        os.makedirs(shard_dir, exist_ok=True)
        shard_paths = [os.path.join(shard_dir, f"shard_{i:04d}.bin") for i in range(SHARD_COUNT)]
        shard_writers = [open(p, "wb") for p in shard_paths]

        pool_topk = int(self.pool_bucket_topk)
        slice_bucket_top1 = np.full(max(1, total_partials), -1, dtype=np.int32)
        bar = tqdm(total=total_funcs, desc="build pool: materialize funcs", dynamic_ncols=True)
        try:
            for binary_name in binary_iteration_order:
                cleaned, feature_map = binary_cache.pop(binary_name, ({}, {}))
                bin_partials: List[np.ndarray] = []
                bin_slice_priors: List[float] = []
                bin_slice_fids: List[int] = []
                slice_offset = 0
                func_slice_ranges: List[Tuple[int, int, int]] = []
                for key in binary_entry_keys.get(binary_name, []):
                    if key not in cleaned:
                        continue
                    func_id = self.meta_to_index[(binary_name, key[0], key[1])]
                    emb_global, partial = cleaned[key]
                    start = int(func_partial_offsets[func_id])
                    end = int(func_partial_offsets[func_id + 1])
                    n_slices = int(partial.shape[0])
                    if n_slices != (end - start):
                        bar.update(1)
                        continue
                    norm_partial = self._normalize_embedding_rows(partial)
                    if isinstance(flat_mm, np.memmap) and end > start:
                        flat_mm[start:end] = norm_partial.astype(np.float32, copy=False)
                    if isinstance(globals_mm, np.memmap):
                        globals_mm[func_id] = self._normalize_embedding_rows(emb_global.reshape(1, -1))[0]
                    blocks = feature_map.get(key, {}).get("blocks_pseudocode", [])
                    for idx in range(n_slices):
                        bin_slice_priors.append(self._coarse_meta_weight(
                            blocks[idx] if idx < len(blocks) else {}
                        ))
                        bin_slice_fids.append(func_id)
                    bin_partials.append(partial)
                    func_slice_ranges.append((func_id, slice_offset, slice_offset + n_slices))
                    slice_offset += n_slices
                    bar.update(1)

                if not bin_partials:
                    continue

                binary_partials = np.vstack(bin_partials).astype(np.float32, copy=False)
                ids_mat, aff_mat, valid_mat = self._assign_bucket_hits_arrays(
                    binary_partials, topk=pool_topk,
                )

                ids_top1_batch = ids_mat[:, 0].astype(np.int32, copy=False)
                for fid, b_start, b_end in func_slice_ranges:
                    gs = int(func_partial_offsets[fid])
                    ge = int(func_partial_offsets[fid + 1])
                    if ge - gs == b_end - b_start and ge > gs:
                        slice_bucket_top1[gs:ge] = ids_top1_batch[b_start:b_end]

                slice_fids_arr = np.asarray(bin_slice_fids, dtype=np.int32)
                slice_priors_arr = np.asarray(bin_slice_priors, dtype=np.float32)
                bid_flat = ids_mat.reshape(-1)
                aff_flat = aff_mat.reshape(-1)
                valid_flat = valid_mat.reshape(-1)
                fid_flat = np.repeat(slice_fids_arr, pool_topk)
                sp_flat = np.repeat(slice_priors_arr, pool_topk)

                if valid_flat.any():
                    cnt_fid = fid_flat[valid_flat]
                    cnt_bid = bid_flat[valid_flat].astype(np.int64)
                    if cnt_bid.size:
                        local_max = int(cnt_bid.max())
                        if local_max > max_bucket_id:
                            max_bucket_id = local_max
                    pair_keys = (cnt_fid.astype(np.int64) << 32) | (cnt_bid & 0xFFFFFFFF)
                    pair_unique = np.unique(pair_keys)
                    fids_from_pairs = (pair_unique >> 32).astype(np.int32)
                    unique_fids, counts_per_fid = np.unique(fids_from_pairs, return_counts=True)
                    func_bucket_count[unique_fids] = counts_per_fid.astype(np.int32, copy=False)

                agg_mask = valid_flat & (sp_flat > 0.0)
                if not agg_mask.any():
                    continue
                agg_bid = bid_flat[agg_mask].astype(np.int64)
                agg_fid = fid_flat[agg_mask].astype(np.int64)
                agg_c = (sp_flat[agg_mask] * aff_flat[agg_mask]).astype(np.float32, copy=False)

                pair_keys2 = (agg_bid << 32) | (agg_fid & 0xFFFFFFFF)
                order = np.argsort(pair_keys2, kind="stable")
                keys_s = pair_keys2[order]
                c_s = agg_c[order]
                unique_keys, start_idx = np.unique(keys_s, return_index=True)
                best_per = np.maximum.reduceat(c_s, start_idx)
                support_per = np.add.reduceat(c_s, start_idx)
                bid_unique = (unique_keys >> 32).astype(np.int32)
                fid_unique = (unique_keys & 0xFFFFFFFF).astype(np.int32)

                records = np.empty(unique_keys.size, dtype=shard_dtype)
                records["bucket_id"] = bid_unique
                records["func_id"] = fid_unique
                records["best"] = best_per.astype(np.float32, copy=False)
                records["support"] = support_per.astype(np.float32, copy=False)

                sids = (records["bucket_id"].astype(np.int64) % SHARD_COUNT).astype(np.int32)
                shard_order = np.argsort(sids, kind="stable")
                sorted_records = records[shard_order]
                sorted_sids = sids[shard_order]
                shard_starts = np.searchsorted(sorted_sids, np.arange(SHARD_COUNT), side="left")
                shard_ends = np.searchsorted(sorted_sids, np.arange(SHARD_COUNT), side="right")
                for sid_val in range(SHARD_COUNT):
                    lo, hi = int(shard_starts[sid_val]), int(shard_ends[sid_val])
                    if hi > lo:
                        sorted_records[lo:hi].tofile(shard_writers[sid_val])
        finally:
            bar.close()
            for w in shard_writers:
                try:
                    w.close()
                except OSError:
                    pass

        for arr in (flat_mm, globals_mm):
            if isinstance(arr, np.memmap):
                arr.flush()
        del flat_mm, globals_mm
        np.save(self._runtime_path("func_partial_offsets"), func_partial_offsets)
        np.save(self._runtime_path("slice_bucket_top1"), slice_bucket_top1)

        bucket_count = max_bucket_id + 1 if max_bucket_id >= 0 else 0
        bucket_posting_len = np.zeros(bucket_count, dtype=np.int32)
        bar = tqdm(total=SHARD_COUNT, desc="build pool: shard size", dynamic_ncols=True)
        for p in shard_paths:
            try:
                arr = np.fromfile(p, dtype=shard_dtype)
            except OSError:
                arr = np.empty(0, dtype=shard_dtype)
            bar.update(1)
            if arr.size:
                bucket_posting_len += np.bincount(arr["bucket_id"].astype(np.int64), minlength=bucket_count).astype(np.int32, copy=False)
        bar.close()

        bucket_offsets = np.zeros(bucket_count + 1, dtype=np.int64)
        if bucket_count > 0:
            np.cumsum(bucket_posting_len.astype(np.int64), out=bucket_offsets[1:])
        total_postings = int(bucket_offsets[-1]) if bucket_offsets.size else 0
        bucket_df_ratio = np.zeros(bucket_count, dtype=np.float32)
        bucket_idf = np.ones(bucket_count, dtype=np.float32)
        if total_funcs > 0 and bucket_count > 0:
            pl_f = bucket_posting_len.astype(np.float32)
            bucket_df_ratio = (pl_f / float(total_funcs)).astype(np.float32, copy=False)
            bucket_idf = (np.log((float(total_funcs) + 1.0) / (pl_f + 1.0)) + 1.0).astype(np.float32, copy=False)
        self.bucket_idf = bucket_idf
        self.bucket_posting_len = bucket_posting_len
        self.bucket_df_ratio = bucket_df_ratio
        self._refresh_hot_bucket_threshold()

        ids_mm = self._open_runtime_memmap("bucket_func_ids", (max(1, total_postings),), dtype=np.int32)
        signal_mm = self._open_runtime_memmap("bucket_func_overall_signal", (max(1, total_postings),), dtype=np.float32)
        write_cursor = bucket_offsets[:-1].astype(np.int64, copy=True) if bucket_count > 0 else np.zeros(0, dtype=np.int64)
        func_total_bucket_support = np.zeros(total_funcs, dtype=np.float32)
        SAT_SCALE = 0.85
        bar = tqdm(total=SHARD_COUNT, desc="build pool: posting shards", dynamic_ncols=True)
        try:
            for p in shard_paths:
                try:
                    arr = np.fromfile(p, dtype=shard_dtype)
                except OSError:
                    arr = np.empty(0, dtype=shard_dtype)
                bar.update(1)
                if arr.size == 0:
                    _safe_unlink(p)
                    continue
                bid_col, fid_col, best_col, support_col = arr["bucket_id"], arr["func_id"], arr["best"], arr["support"]
                composite = (
                    0.70 * best_col.astype(np.float32, copy=False)
                    + 0.30 * np.where(support_col > 0.0, 1.0 - np.exp(-SAT_SCALE * support_col), 0.0).astype(np.float32, copy=False)
                )
                order = np.lexsort((fid_col, -support_col, -best_col, -composite, bid_col))
                sorted_bid, sorted_fid = bid_col[order], fid_col[order]
                sorted_best, sorted_support = best_col[order], support_col[order]
                unique_buckets, start_idx, counts = np.unique(sorted_bid, return_index=True, return_counts=True)
                for bid_int, sidx, cnt in zip(unique_buckets.tolist(), start_idx.tolist(), counts.tolist()):
                    sidx_int, cnt_int = int(sidx), int(cnt)
                    write_start = int(write_cursor[bid_int])
                    write_end = write_start + cnt_int
                    fids_sub = sorted_fid[sidx_int:sidx_int + cnt_int]
                    best_sub = sorted_best[sidx_int:sidx_int + cnt_int]
                    support_sub = sorted_support[sidx_int:sidx_int + cnt_int]
                    ids_mm[write_start:write_end] = fids_sub
                    signal_mm[write_start:write_end] = self._compose_bucket_func_signal(
                        best_sub, support_sub, func_bucket_count[fids_sub]
                    )
                    write_cursor[bid_int] = write_end
                    np.add.at(func_total_bucket_support, fids_sub.astype(np.int64, copy=False), support_sub)
                del arr, order
                _safe_unlink(p)
        finally:
            bar.close()
            _wipe_dir(shard_dir)
            try:
                os.rmdir(shard_dir)
            except OSError:
                pass
        for arr in (ids_mm, signal_mm):
            if isinstance(arr, np.memmap):
                arr.flush()
        del ids_mm, signal_mm

        for name, array in {
            "bucket_func_offsets": bucket_offsets,
            "bucket_idf": bucket_idf,
            "bucket_posting_len": bucket_posting_len,
            "bucket_df_ratio": bucket_df_ratio,
            "func_slice_count": func_slice_count,
            "func_total_bucket_support": func_total_bucket_support,
        }.items():
            self._save_runtime_array(name, array)
        write_pickle({
            "meta_to_index": self.meta_to_index,
            "index_to_meta": self.index_to_meta,
            "embedding_dim": self.embedding_dim,
            "bucket_topk": self.bucket_topk,
            "pool_bucket_topk": self.pool_bucket_topk,
            "db_version": self.DB_VERSION,
        }, pool_path)
        self.load_db()

    def load_db(self):
        """mmap every runtime array; RAM = hot pages only (10M: ~30GB postings + ~30GB partials)."""
        pool = read_pickle(os.path.join(self.save_db, "pool_inline_slice.pkl"))
        if pool.get("db_version") != self.DB_VERSION:
            raise ValueError("stale pool db version")
        self.meta_to_index = pool["meta_to_index"]
        self.index_to_meta = pool["index_to_meta"]
        self.embedding_dim = pool["embedding_dim"]
        self.bucket_topk = int(pool.get("bucket_topk", self.bucket_topk))
        self.pool_bucket_topk = int(pool.get("pool_bucket_topk", self.pool_bucket_topk))
        self.bucket_func_offsets = self._load_runtime_array("bucket_func_offsets")
        for name in ("bucket_func_ids", "bucket_func_overall_signal", "func_globals"):
            setattr(self, name, self._load_runtime_array(name, mmap=True))
        for name, dt in (("bucket_idf", np.float32), ("bucket_posting_len", np.int32),
                         ("bucket_df_ratio", np.float32), ("func_slice_count", np.int32),
                         ("func_total_bucket_support", np.float32)):
            setattr(self, name, self._load_runtime_array(name, cast=dt))
        self._refresh_hot_bucket_threshold()
        fp, op = self._runtime_path("func_partials_flat"), self._runtime_path("func_partial_offsets")
        self.func_partials_flat = np.load(fp, mmap_mode="r") if os.path.exists(fp) else None
        self.func_partial_offsets = np.load(op).astype(np.int64, copy=False) if os.path.exists(op) else None


def _wipe_dir(path: str) -> None:
    if not os.path.isdir(path):
        return
    for entry in os.listdir(path):
        _safe_unlink(os.path.join(path, entry))


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
