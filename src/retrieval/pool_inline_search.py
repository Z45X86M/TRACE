"""Semantic-unit BCSD retrieval: recall algorithm + CLI.

Entry point: ``EmbeddingRetrievalDB.search``. Four stages, GPU-preferred:
  Stage 0  _build_query_profile     row-normalize + coherence rescale
  Stage A1 _build_query_slice_sets  per row -> top-K bucket -> posting -> aggregate
  Stage A2 _score_stage_a           slice_score, then cap the candidate set
  Stage B  _dense_rerank            MaxSim + IDF weighting + base/slice/global fusion

Peak memory scales with the number of query units, not with pool size.
"""

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
for path in (ROOT_DIR, CURRENT_DIR):
    if path not in sys.path:
        sys.path.append(path)

from build_inline_eval_pool import build_eval_pool
from pool_inline_storage import InlinePool
from utils import read_pickle, type_name
import verifiers


class EmbeddingRetrievalDB(InlinePool):
    CAP = 500
    BTOPK = 16
    SPECIFICITY_MULTIPLIER_MIN = 0.6
    SPECIFICITY_MULTIPLIER_MAX = 1.4
    STAGE_A_MAXB_WEIGHT: float = 0.8
    STAGE_A_MAXB_CLIP: float = 1.5
    RERANK_W_BASE_CONST: float = 0.05
    RERANK_W_GLOBAL_LOW: float = 0.05
    RERANK_W_GLOBAL_HIGH: float = 0.65
    RERANK_W_SLICE_LOW: float = 0.30
    RERANK_W_SLICE_HIGH: float = 0.85
    RERANK_EXCESS_COEF: float = 0.80
    HUGE_POOL_THRESHOLD: int = 500_000
    HUGE_BTOPK: int = 16
    HUGE_CAP: int = 6000
    RERANK_HUGE_W_BASE: float = 0.05
    RERANK_HUGE_W_SLICE: float = 0.70
    RERANK_HUGE_W_GLOBAL: float = 0.05
    RERANK_HUGE_EXCESS_COEF: float = 0.80
    RERANK_HUGE_PENALTY_COEF: float = 0.30
    RERANK_DEVICE: str = "cuda:1"
    RERANK_STORAGE_DTYPE: str = "fp16"

    VERIFIER_ENABLED: bool = False
    VERIFIER_WEIGHT: float = 0.4
    VERIFIER_TAU: float = 0.85
    VERIFIER_FLOOR: float = 1e-3

    ANCHOR_ENABLED: bool = False
    ANCHOR_LAMBDA: float = 0.5

    @staticmethod
    def recommended_cap(pool_size: int) -> int:
        """Empirical shortlist cap: CAP(N) = max(500, ceil(6*sqrt(N))),
        N = number of pool functions (sub-linear coverage knee)."""
        import math
        return max(500, math.ceil(6.0 * math.sqrt(int(pool_size))))

    def _is_huge_pool(self) -> bool:
        """Huge pool (real-world CVE scale) -> slice-dominant, low-global fusion.
        Decided by the materialized pool function count, not --pool-size."""
        try:
            n = len(self.index_to_meta)
        except (AttributeError, TypeError):
            n = 0
        return n >= int(self.HUGE_POOL_THRESHOLD)


    def prepare_anchors(self, pool_feat_dir: str, query_dirs: List[str]) -> None:
        """Build the pool anchor table + IDF once and register the query anchor
        dirs. See verifiers.build_anchor_resources."""
        idx = getattr(self, "_anchor_index_path", None)
        loaded = bool(idx and os.path.exists(idx))
        self._anchor_pool, self._anchor_idf, self._anchor_pool_mass = \
            verifiers.build_anchor_resources(pool_feat_dir, self.feature_suffix, index_path=idx)
        self._anchor_qdirs = list(query_dirs)
        self._anchor_qcache: Dict[str, Dict[Any, Tuple]] = {}
        print("[anchor] %s pool funcs=%d  vocab const=%d str=%d call=%d  (Dice, lambda=%.3f, full-CAP promote)"
              % ("loaded index," if loaded else "built,", len(self._anchor_pool),
                 len(self._anchor_idf[0]), len(self._anchor_idf[1]), len(self._anchor_idf[2]),
                 float(self.ANCHOR_LAMBDA)), flush=True)

    def _apply_anchor_promote(self, block, query_loc):
        """Dice promote-only rerank over the whole CAP candidate set (off by
        default; a no-op when disabled or when the query has no anchors)."""
        if not (self.ANCHOR_ENABLED and getattr(self, "_anchor_pool", None) is not None
                and query_loc and query_loc[0] is not None):
            return block
        if block is None or len(block.get("fids", ())) == 0 or "score" not in block:
            return block
        cache = getattr(self, "_anchor_qcache", None)
        if cache is None:
            self._anchor_qcache = cache = {}
        qa = verifiers.load_query_anchors(getattr(self, "_anchor_qdirs", []),
                                          query_loc[0], query_loc[1], self.feature_suffix, cache)
        if not (qa[0] or qa[1] or qa[2]):
            return block
        idf = self._anchor_idf
        pool = self._anchor_pool
        pmass = getattr(self, "_anchor_pool_mass", {})
        lam = float(self.ANCHOR_LAMBDA)
        q_mass = verifiers.anchor_overlap(qa, qa, idf)
        empty = (frozenset(), frozenset(), frozenset())
        meta = self.index_to_meta
        fids = block["fids"]
        n = int(fids.shape[0])
        ovs = np.zeros(n, dtype=np.float32)
        dices = np.zeros(n, dtype=np.float32)
        for i in range(n):
            b, a, _nm = meta[int(fids[i])]
            key = (b, verifiers.norm_addr_key(a))
            ov = verifiers.anchor_overlap(qa, pool.get(key, empty), idf)
            if ov > 0.0:
                ovs[i] = ov
                dices[i] = 2.0 * ov / (q_mass + pmass.get(key, 0.0) + 1e-6)
        score = np.asarray(block["score"], dtype=np.float64) + float(lam) * dices.astype(np.float64)
        mr = self._ensure_meta_rank()[fids]
        order = np.lexsort((mr, -np.asarray(block["slice_score"], dtype=np.float32),
                            -np.asarray(block["global_sim"], dtype=np.float32),
                            -np.asarray(block["slice_channel"], dtype=np.float32), -score))
        out = {k: np.asarray(v)[order] for k, v in block.items()}
        out["score"] = score[order]
        out["anchor_overlap"] = ovs[order]
        out["anchor_dice"] = dices[order]
        return out


    def _resolve_rerank_device(self):
        if not self.RERANK_DEVICE or self.RERANK_DEVICE == "cpu":
            return None
        try:
            import torch
        except ImportError:
            return None
        if not (self.RERANK_DEVICE.startswith("cuda") and torch.cuda.is_available()):
            return None
        return torch.device(self.RERANK_DEVICE)

    def _ensure_gpu_state(self) -> bool:
        """Upload pool arrays to GPU and precompute per-bucket weights. Idempotent.

        First call on a freshly built pool reads tens of GB of mmap pages and
        moves them to GPU; without progress this looks like a hang. Wrap each
        big tensor upload in a tqdm tick so the user sees forward motion.
        """
        if getattr(self, "_gpu_state_ready", None) is not None:
            return bool(self._gpu_state_ready)
        device = self._resolve_rerank_device()
        if device is None:
            self._gpu_state_ready = False
            return False
        import torch
        storage_dt = torch.float16 if self.RERANK_STORAGE_DTYPE == "fp16" else torch.float32

        upload_specs: List[Tuple[str, Any]] = [
            ("func_partials_flat", lambda: (self.func_partials_flat, np.float32, storage_dt)),
            ("func_globals", lambda: (self.func_globals, np.float32, storage_dt)),
            ("func_partial_offsets", lambda: (self.func_partial_offsets, np.int64, None)),
            ("pool_slice_idf", lambda: (self._ensure_pool_slice_idf(), np.float32, None)),
        ]
        stage_a_specs: List[Tuple[str, Any]] = []
        if self.bucket_func_offsets is not None and self.bucket_func_ids is not None:
            stage_a_specs = [
                ("bucket_func_offsets", lambda: (self.bucket_func_offsets, np.int64, None)),
                ("bucket_func_ids", lambda: (self.bucket_func_ids, np.int64, None)),
                ("bucket_func_overall_signal", lambda: (self.bucket_func_overall_signal, np.float32, None)),
                ("bucket_idf", lambda: (self.bucket_idf, np.float32, None)),
                ("func_slice_count", lambda: (self.func_slice_count, np.int64, None)),
                ("func_total_bucket_support", lambda: (self.func_total_bucket_support, np.float32, None)),
            ]
        all_specs = upload_specs + stage_a_specs

        def _up(
            arr: Optional[np.ndarray], dtype, target_dt: Optional["torch.dtype"] = None,
        ) -> Optional["torch.Tensor"]:
            """Upload `arr` to GPU, optionally casting to `target_dt`.

            Two paths:
              - Same-dtype: wrap mmap zero-copy, single `.to(device)`. Fault-in
                happens during transfer. Saves ~30 GB RSS peak vs. old code.
              - fp32 → fp16 (big arrays at multi-million pool): pre-allocate dst
                in fp16 and copy chunk by chunk. Avoids the previous OOM where
                `.to(fp16)` on a 27 GB fp32 GPU tensor needed an extra 13.6 GB
                of headroom (peak = src + dst).
            """
            if arr is None or arr.shape[0] == 0:
                return None
            src_np_dtype = np.dtype(dtype)
            cpu_torch_dtype = (
                torch.float32 if src_np_dtype == np.float32 else
                torch.float16 if src_np_dtype == np.float16 else
                torch.int64 if src_np_dtype == np.int64 else
                torch.int32 if src_np_dtype == np.int32 else
                None
            )
            same_dtype = (target_dt is None or target_dt == cpu_torch_dtype)
            if same_dtype:
                src = arr if arr.dtype == src_np_dtype else np.ascontiguousarray(arr, dtype=src_np_dtype)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="The given NumPy array is not writable.*"
                    )
                    cpu_t = torch.from_numpy(src)
                return cpu_t.to(device)
            shape = tuple(arr.shape)
            dst = torch.empty(shape, dtype=target_dt, device=device)
            row_bytes = max(1, int(np.prod(shape[1:])) * src_np_dtype.itemsize) if len(shape) > 1 else src_np_dtype.itemsize
            CHUNK_BYTES = 1 << 30
            chunk_rows = max(1, CHUNK_BYTES // row_bytes)
            N = shape[0]
            for i in range(0, N, chunk_rows):
                end = min(N, i + chunk_rows)
                sub = arr[i:end]
                if sub.dtype != src_np_dtype:
                    sub = np.ascontiguousarray(sub, dtype=src_np_dtype)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="The given NumPy array is not writable.*"
                    )
                    cpu_t = torch.from_numpy(sub)
                dst[i:end] = cpu_t.to(device).to(dtype=target_dt)
            return dst

        uploaded: Dict[str, Any] = {}
        bar = tqdm(total=len(all_specs), desc="warm pool: upload to gpu", dynamic_ncols=True)
        try:
            for label, spec in all_specs:
                arr, src_dtype, post_dtype = spec()
                uploaded[label] = _up(arr, src_dtype, target_dt=post_dtype)
                bar.update(1)
        finally:
            bar.close()

        self._func_partials_gpu = uploaded.get("func_partials_flat")
        self._func_globals_gpu = uploaded.get("func_globals")
        self._func_partial_offsets_gpu = uploaded.get("func_partial_offsets")
        self._pool_slice_idf_gpu = uploaded.get("pool_slice_idf")

        self._stage_a_gpu_ready = False
        if stage_a_specs:
            self._bucket_func_offsets_gpu = uploaded.get("bucket_func_offsets")
            self._bucket_func_ids_gpu = uploaded.get("bucket_func_ids")
            self._bucket_func_signal_gpu = uploaded.get("bucket_func_overall_signal")
            self._bucket_idf_gpu = uploaded.get("bucket_idf")
            self._bucket_query_weight_gpu = torch.from_numpy(
                self._compute_bucket_query_weights_np()
            ).to(device)
            self._idf_pivot = float(np.median(self.bucket_idf)) if self.bucket_idf.size > 0 else 0.0
            import faiss as _faiss
            self._faiss_is_l2 = (self.ivf.metric_type == _faiss.METRIC_L2)
            self._func_slice_count_gpu = uploaded.get("func_slice_count")
            self._func_total_bucket_support_gpu = uploaded.get("func_total_bucket_support")
            self._stage_a_gpu_ready = True
        self._rerank_device = device
        self._gpu_state_ready = True
        return True

    def _compute_bucket_query_weights_np(self) -> np.ndarray:
        """Vectorized InlinePool._bucket_query_weight; returns (nlist,) fp32."""
        idf = np.asarray(self.bucket_idf, dtype=np.float32)
        pl = np.asarray(self.bucket_posting_len, dtype=np.float32)
        df = np.asarray(self.bucket_df_ratio, dtype=np.float32)
        soft_df = np.maximum(0.15, 1.0 - 2.5 * df)
        soft_post = 1.0 / np.sqrt(np.maximum(1.0, pl / 32.0))
        weight = idf * np.maximum(0.18, soft_df * soft_post)
        stop_mask = (pl >= self.stop_bucket_min_posting) & (df >= self.stop_bucket_df_ratio)
        if stop_mask.any():
            df_pen = self.stop_bucket_df_ratio / np.maximum(df, self.stop_bucket_df_ratio)
            post_pen = self.stop_bucket_min_posting / np.maximum(pl, self.stop_bucket_min_posting)
            weight = np.where(stop_mask, weight * np.maximum(0.18, np.sqrt(df_pen * post_pen)), weight)
        weight = np.minimum(weight, self.HOT_BUCKET_QUERY_WEIGHT_CAP)
        if self.bucket_hot_weight is not None and self.bucket_hot_weight.shape[0] == idf.shape[0]:
            weight = weight * np.asarray(self.bucket_hot_weight, dtype=np.float32)
        return weight.astype(np.float32, copy=False)

    def _ensure_pool_slice_idf(self) -> np.ndarray:
        """Per-pool-unit IDF weights (mean-normalized to 1). Three-level cache:
          1. <save_db>/pool_slice_idf.npy      — prebuilt array (fastest)
          2. <save_db>/slice_bucket_top1.npy   — top-1 bucket ids, O(N) lookup
          3. fallback: recompute via the FAISS quantizer (slowest)
        """
        cached = getattr(self, "_pool_slice_idf_cache", None)
        if cached is not None:
            return cached
        cache_path = self._runtime_path("pool_slice_idf") if hasattr(self, "_runtime_path") else None
        if cache_path and os.path.exists(cache_path):
            arr = np.load(cache_path).astype(np.float32, copy=False)
            self._pool_slice_idf_cache = arr
            return arr
        bucket_idf = self.bucket_idf
        top1_path = self._runtime_path("slice_bucket_top1") if hasattr(self, "_runtime_path") else None
        if top1_path and os.path.exists(top1_path) and bucket_idf is not None:
            bucket_ids = np.load(top1_path).astype(np.int32, copy=False)
            weights = self._slice_idf_from_bucket_ids(bucket_ids, bucket_idf)
            if cache_path:
                try:
                    np.save(cache_path, weights)
                except Exception:
                    pass
            self._pool_slice_idf_cache = weights
            return weights
        flat = self.func_partials_flat
        if flat is None or bucket_idf is None:
            arr = np.ones(0 if flat is None else flat.shape[0], dtype=np.float32)
            self._pool_slice_idf_cache = arr
            return arr
        N = int(flat.shape[0])
        bucket_ids = np.empty(N, dtype=np.int32)
        BATCH = 65536
        n_batches = (N + BATCH - 1) // BATCH
        bar = tqdm(total=n_batches, desc="warm pool: slice idf", dynamic_ncols=True)
        try:
            for i in range(0, N, BATCH):
                chunk = np.asarray(flat[i: i + BATCH], dtype=np.float32)
                chunk = self.pca.apply_py(chunk) if self.pca is not None else chunk
                _, ids = self.gpu_quantizer.search(chunk, 1)
                bucket_ids[i: i + BATCH] = ids[:, 0]
                bar.update(1)
        finally:
            bar.close()
        weights = self._slice_idf_from_bucket_ids(bucket_ids, bucket_idf)
        if cache_path:
            try:
                np.save(cache_path, weights)
            except Exception:
                pass
        self._pool_slice_idf_cache = weights
        return weights

    def _slice_idf_from_bucket_ids(
        self, bucket_ids: np.ndarray, bucket_idf: np.ndarray,
    ) -> np.ndarray:
        N = int(bucket_ids.shape[0])
        valid = bucket_ids >= 0
        idf = np.empty(N, dtype=np.float32)
        if valid.any():
            idf[valid] = bucket_idf[np.clip(bucket_ids[valid], 0, bucket_idf.shape[0] - 1)]
        if (~valid).any():
            idf[~valid] = float(bucket_idf.mean()) if bucket_idf.size else 1.0
        return (idf / max(float(idf.mean()) if idf.size else 1.0, 1e-6)).astype(np.float32)


    def search(
        self,
        query_emb: np.ndarray,
        global_query: Optional[np.ndarray] = None,
        query_profile: Optional[Dict[str, Any]] = None,
        query_row_meta: Optional[List[Dict[str, Any]]] = None,
        return_debug: bool = False,
        query_loc: Optional[Tuple[Any, Any]] = None,
    ):
        """One query -> a candidate list of up to CAP entries. query_loc=(binary,
        addr) lets the anchor verifier read query-side symbols (ANCHOR only)."""
        t0 = time.perf_counter()
        if query_profile is None:
            query_profile = self._build_query_profile(query_emb, query_row_meta, global_query=global_query)
        candidate_func_ids, aggregates, debug = self._build_query_slice_sets(query_profile)
        debug["touched_fids"] = np.asarray(candidate_func_ids)
        block = self._score_stage_a(candidate_func_ids, aggregates, query_profile)
        block = self._dense_rerank(block, query_profile)
        block = self._apply_anchor_promote(block, query_loc)

        fids = block.get("fids")
        n = 0 if fids is None else min(int(fids.shape[0]), int(self.CAP))
        meta = self.index_to_meta
        cols = [(k, block[k]) for k in (
            "hit_count", "slice_score", "base_score", "slice_sim", "f2q_sim", "q2f_idf",
            "f2q_idf", "slice_channel", "global_sim", "alignment", "score",
            "verify_conn", "anchor_overlap", "anchor_dice") if k in block]
        recall: List[Dict[str, Any]] = []
        for i in range(n):
            fid = int(fids[i])
            b, a, nm = meta[fid]
            row = {"func_id": fid, "binary_name": b, "func_addr": a, "func_name": nm}
            for k, arr in cols:
                row[k] = int(arr[i]) if k == "hit_count" else float(arr[i])
            row["rank"] = i + 1
            recall.append(row)
        debug["total_search_time_s"] = float(time.perf_counter() - t0)
        result = {"recall_results": recall}
        return (result, debug) if return_debug else result

    def _build_query_profile(
        self,
        query_emb: np.ndarray,
        query_row_meta: Optional[List[Dict[str, Any]]],
        global_query: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """row_weights = coarse_meta x coherence rescale (a unit more consistent
        with the function-level vector gets a higher weight)."""
        query_emb = np.asarray(query_emb, dtype=np.float32)
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        R, D = query_emb.shape

        row_meta = list(query_row_meta or [])
        while len(row_meta) < R:
            row_meta.append({"slice_kind": "coverage_chain", "slice_quality": 1.0, "semantic_role": "primary"})
        row_meta = row_meta[:R]

        row_weights = np.asarray([self._coarse_meta_weight(m) for m in row_meta], dtype=np.float32)
        if row_weights.size == 0 or float(row_weights.sum()) <= 0.0:
            row_weights = np.ones(R, dtype=np.float32)

        norm_emb = self._normalize_embedding_rows(query_emb)

        gq_norm = None
        if global_query is not None:
            gq_arr = np.asarray(global_query, dtype=np.float32).reshape(-1)
            if gq_arr.size == D:
                gq_norm = self._normalize_embedding_rows(gq_arr.reshape(1, -1))[0]
                agrees = np.maximum(0.0, (norm_emb @ gq_norm).astype(np.float32, copy=False))
                mean_a = float(agrees.mean()) if agrees.size > 0 else 0.0
                if mean_a > 1e-6:
                    row_weights = (row_weights * (agrees / mean_a)).astype(np.float32, copy=False)

        query_centroid = self._weighted_mean_embedding(norm_emb, row_weights)
        return {
            "norm_emb": norm_emb,
            "row_weights": row_weights,
            "row_count": R,
            "total_query_weight": float(max(1e-6, row_weights.sum())),
            "query_centroid": query_centroid,
            "gq_or_centroid": gq_norm if gq_norm is not None else query_centroid,
        }


    def _ensure_meta_rank(self) -> np.ndarray:
        mr = getattr(self, "_meta_rank", None)
        if mr is not None and mr.shape[0] == len(self.index_to_meta):
            return mr
        keys = [(b, str(a)) for b, a, _n in self.index_to_meta]
        order = sorted(range(len(keys)), key=keys.__getitem__)
        mr = np.empty(len(keys), dtype=np.int64)
        mr[np.asarray(order, dtype=np.int64)] = np.arange(len(keys), dtype=np.int64)
        self._meta_rank = mr
        return mr

    @staticmethod
    def _empty_stage_block() -> Dict[str, np.ndarray]:
        return {"fids": np.empty(0, dtype=np.int64),
                "slice_score": np.empty(0, dtype=np.float32),
                "support": np.empty(0, dtype=np.float32),
                "hit_count": np.empty(0, dtype=np.int32)}

    def _build_query_slice_sets(self, query_profile):
        self._ensure_gpu_state()
        if getattr(self, "_stage_a_gpu_ready", False):
            return self._build_query_slice_sets_gpu(query_profile)
        return self._build_query_slice_sets_cpu(query_profile)

    def _score_stage_a(self, candidate_func_ids, aggregates, query_profile):
        if getattr(self, "_stage_a_gpu_ready", False):
            return self._score_stage_a_gpu(candidate_func_ids, aggregates, query_profile)
        return self._score_stage_a_cpu(candidate_func_ids, aggregates, query_profile)

    def _dense_rerank(self, candidates, query_profile):
        if candidates is None or len(candidates.get("fids", ())) == 0:
            return candidates if candidates is not None else self._empty_stage_block()
        if self._ensure_gpu_state():
            return self._dense_rerank_gpu(candidates, query_profile)
        return self._dense_rerank_cpu(candidates, query_profile)


    def _build_query_slice_sets_gpu(
        self, query_profile: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
        """Stage A1: batch FAISS -> vectorized affinity/specificity -> posting
        gather -> two-level within-row + cross-row scatter_add. Matches the CPU path."""
        import torch
        device = self._rerank_device
        norm_emb_np = np.asarray(query_profile["norm_emb"], dtype=np.float32)
        row_weights_np = np.asarray(query_profile["row_weights"], dtype=np.float32)
        R = int(norm_emb_np.shape[0])
        K = int(self.BTOPK)
        spec_lo, spec_hi = float(self.SPECIFICITY_MULTIPLIER_MIN), float(self.SPECIFICITY_MULTIPLIER_MAX)
        idf_pivot = float(self._idf_pivot)

        debug = {"query_bucket_count": 0, "touched_func_count": 0}
        empty_i, empty_f = np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        empty_agg = {"hit_count": empty_i, "hit_weight": empty_f, "support_mass": empty_f,
                     "selectivity_mass": empty_f, "max_bucket_evidence": empty_f}
        if R == 0:
            return empty_i, empty_agg, debug

        pca_in = self.pca.apply_py(norm_emb_np) if self.pca is not None else norm_emb_np
        distances_np, ids_np = self.gpu_quantizer.search(pca_in, K + 1)
        bucket_ids = torch.from_numpy(np.asarray(ids_np[:, :K], dtype=np.int64).copy()).to(device)
        bucket_d = torch.from_numpy(np.asarray(distances_np[:, :K], dtype=np.float32).copy()).to(device)

        best_d = bucket_d.amin(dim=1, keepdim=True)
        if self._faiss_is_l2:
            delta = (bucket_d - best_d).clamp_min(0.0)
            scale = (best_d * 2.5).clamp_min(1e-4)
        else:
            delta = (best_d - bucket_d).clamp_min(0.0)
            scale = (best_d.abs() * 0.15).clamp_min(1e-4)
        ranks = torch.arange(K, device=device, dtype=torch.float32).unsqueeze(0).expand(R, K)
        affinities = torch.exp(-delta / scale) / torch.sqrt(ranks + 1.0)

        valid_rk = bucket_ids >= 0
        nlist = self._bucket_idf_gpu.shape[0]
        safe_bid = bucket_ids.clamp(0, nlist - 1)
        bucket_idf_per = torch.where(valid_rk, self._bucket_idf_gpu[safe_bid], torch.zeros_like(self._bucket_idf_gpu[safe_bid]))
        row_max_idf = bucket_idf_per.amax(dim=1)
        spec_mult = (row_max_idf / idf_pivot).clamp(min=spec_lo, max=spec_hi) if idf_pivot > 0 else torch.ones(R, device=device)
        row_w_eff = torch.from_numpy(row_weights_np[:R].copy()).to(device) * spec_mult

        bucket_qw_per = torch.where(valid_rk, self._bucket_query_weight_gpu[safe_bid], torch.zeros_like(self._bucket_query_weight_gpu[safe_bid]))

        flat_bid = bucket_ids.reshape(-1)
        flat_aff = affinities.reshape(-1)
        flat_qw = bucket_qw_per.reshape(-1)
        flat_valid = valid_rk.reshape(-1)
        flat_row = torch.arange(R, device=device).unsqueeze(1).expand(R, K).reshape(-1)
        flat_idf = bucket_idf_per.reshape(-1)

        valid_slots = flat_valid.nonzero(as_tuple=False).reshape(-1)
        if valid_slots.numel() == 0:
            return empty_i, empty_agg, debug
        uniq_bid, slot2u = torch.unique(flat_bid[valid_slots], return_inverse=True, sorted=True)
        debug["query_bucket_count"] = int(uniq_bid.numel())
        u_starts = self._bucket_func_offsets_gpu[uniq_bid]
        u_lens = self._bucket_func_offsets_gpu[uniq_bid + 1] - u_starts
        sum_u = int(u_lens.sum().item())
        if sum_u == 0:
            return empty_i, empty_agg, debug
        useg = torch.repeat_interleave(torch.arange(int(uniq_bid.numel()), device=device), u_lens)
        u_base = torch.cat([torch.zeros(1, device=device, dtype=torch.long), u_lens.cumsum(0)])
        u_within = torch.arange(sum_u, device=device, dtype=torch.long) - u_base[useg]
        u_pidx = u_starts[useg] + u_within
        u_fid = self._bucket_func_ids_gpu[u_pidx]
        u_sig = self._bucket_func_signal_gpu[u_pidx]

        slot_lens = u_lens[slot2u]
        sum_p = int(slot_lens.sum().item())
        seg_of_g = torch.repeat_interleave(valid_slots, slot_lens)
        slot_b = torch.cat([torch.zeros(1, device=device, dtype=torch.long), slot_lens.cumsum(0)])
        seg_of_valid = torch.repeat_interleave(torch.arange(int(valid_slots.numel()), device=device), slot_lens)
        within = torch.arange(sum_p, device=device, dtype=torch.long) - slot_b[seg_of_valid]
        u_entry = u_base[slot2u[seg_of_valid]] + within
        entry_fid = u_fid[u_entry]
        entry_signal = flat_aff[seg_of_g] * flat_qw[seg_of_g] * u_sig[u_entry]
        entry_row = flat_row[seg_of_g]
        entry_sig_idf = entry_signal * (flat_idf[seg_of_g] / max(idf_pivot, 1e-6)).clamp(min=0.0)

        n_funcs_total = int(len(self.index_to_meta))
        composite = entry_row.long() * (n_funcs_total + 1) + entry_fid
        unique_composite, inverse_idx = torch.unique(composite, return_inverse=True, sorted=True)
        M = int(unique_composite.numel())
        summed_signal = torch.zeros(M, device=device, dtype=torch.float32)
        summed_signal.scatter_add_(0, inverse_idx, entry_signal)
        unique_row = (unique_composite // (n_funcs_total + 1)).long()
        unique_fid = (unique_composite % (n_funcs_total + 1)).long()

        count_per_row = torch.zeros(R, device=device, dtype=torch.float32)
        count_per_row.scatter_add_(0, unique_row, torch.ones(M, device=device, dtype=torch.float32))

        row_w_pair = row_w_eff[unique_row]
        contrib_hc = torch.ones(M, device=device, dtype=torch.float32)
        contrib_hw = row_w_pair
        contrib_sm = row_w_pair * summed_signal
        contrib_sel = row_w_pair / torch.sqrt(count_per_row[unique_row].clamp_min(1.0))

        unique_func, inverse_func = torch.unique(unique_fid, return_inverse=True, sorted=True)
        N_cand = int(unique_func.numel())
        aggs_gpu = []
        for src in (contrib_hc, contrib_hw, contrib_sm, contrib_sel):
            t = torch.zeros(N_cand, device=device, dtype=torch.float32)
            t.scatter_add_(0, inverse_func, src)
            aggs_gpu.append(t)

        fid_pos = torch.searchsorted(unique_func, entry_fid.long())
        maxb = torch.zeros(N_cand, device=device, dtype=torch.float32)
        maxb.scatter_reduce_(0, fid_pos, entry_sig_idf, reduce="amax", include_self=True)

        cand_ids_np = unique_func.to(torch.int32).cpu().numpy()
        aggregates = {
            "hit_count":        aggs_gpu[0].to(torch.int32).cpu().numpy(),
            "hit_weight":       aggs_gpu[1].cpu().numpy(),
            "support_mass":     aggs_gpu[2].cpu().numpy(),
            "selectivity_mass": aggs_gpu[3].cpu().numpy(),
            "max_bucket_evidence": maxb.cpu().numpy(),
        }
        debug["touched_func_count"] = int(N_cand)
        return cand_ids_np, aggregates, debug

    def _score_stage_a_gpu(
        self, candidate_func_ids: np.ndarray, aggregates: Dict[str, np.ndarray],
        query_profile: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Stage A2: 6-component slice_score + MAXB + top-k prefilter + tie-break
        sort. Matches the CPU path."""
        import torch
        device = self._rerank_device
        M = int(candidate_func_ids.shape[0])
        if M == 0:
            return []
        cap = int(self.CAP)
        cand_ids_np = np.asarray(candidate_func_ids, dtype=np.int64)
        cand_ids = torch.from_numpy(cand_ids_np.copy()).to(device)

        hit_count = torch.from_numpy(np.asarray(aggregates["hit_count"], dtype=np.int64).copy()).to(device)
        hit_weight = torch.from_numpy(np.asarray(aggregates["hit_weight"], dtype=np.float32).copy()).to(device)
        support_mass = torch.from_numpy(np.asarray(aggregates["support_mass"], dtype=np.float32).copy()).to(device)
        selectivity_mass = torch.from_numpy(np.asarray(aggregates["selectivity_mass"], dtype=np.float32).copy()).to(device)

        fsc = (self._func_slice_count_gpu[cand_ids].clamp(min=1).float()
               if self._func_slice_count_gpu is not None else torch.ones(M, device=device))
        ftbs = (self._func_total_bucket_support_gpu[cand_ids].clamp(min=1e-6)
                if self._func_total_bucket_support_gpu is not None else torch.full((M,), 1e-6, device=device))

        R = int(query_profile["row_count"])
        tqw = max(1e-6, float(query_profile["total_query_weight"]))
        sym_n = torch.minimum(torch.full_like(fsc, float(R)), fsc).clamp(min=1.0)
        sym_w_denom = (tqw * sym_n / max(1.0, float(R))).clamp(min=1e-6)

        hc_f = hit_count.float()
        QWR = (hit_weight / sym_w_denom).clamp(max=1.0)
        PHR = (hc_f / fsc).clamp(max=1.0)
        MHR = torch.minimum(QWR, PHR)
        SUP = 1.0 - torch.exp(-support_mass / tqw)
        SEL = selectivity_mass / tqw
        SPEC = (support_mass / ftbs).clamp(max=1.0)
        slice_score = 0.10 * QWR + 0.28 * PHR + 0.20 * SUP + 0.18 * SPEC + 0.14 * MHR + 0.10 * SEL
        maxb_np = aggregates.get("max_bucket_evidence")
        if maxb_np is not None and len(maxb_np) == M:
            maxb = torch.from_numpy(np.asarray(maxb_np, dtype=np.float32).copy()).to(device)
            slice_score = slice_score + self.STAGE_A_MAXB_WEIGHT * maxb.clamp(0.0, self.STAGE_A_MAXB_CLIP)

        keep_n = min(cap, M)
        top_idx = torch.topk(slice_score, k=keep_n).indices

        out = torch.stack([slice_score[top_idx], SUP[top_idx], hc_f[top_idx]], dim=0).cpu().numpy()
        top_fids = cand_ids_np[top_idx.cpu().numpy()]

        mr = self._ensure_meta_rank()[top_fids]
        order = np.lexsort((mr, -out[2], -out[1], -out[0]))
        return {
            "fids": top_fids[order].astype(np.int64, copy=False),
            "slice_score": out[0][order].astype(np.float32, copy=False),
            "support": out[1][order].astype(np.float32, copy=False),
            "hit_count": out[2][order].astype(np.int32),
        }

    def _dense_rerank_gpu(
        self, candidates: List[Dict[str, Any]], query_profile: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Stage B: MaxSim (q2f, f2q, q2f_idf, f2q_idf) + weighted fusion, via
        scatter_reduce."""
        import torch
        device = self._rerank_device
        norm_emb_np = np.asarray(query_profile["norm_emb"], dtype=np.float32)
        row_weights_np = np.asarray(query_profile["row_weights"], dtype=np.float32)
        row_w_sum = max(1e-6, float(row_weights_np.sum()))
        q_centroid_np = np.asarray(query_profile["gq_or_centroid"], dtype=np.float32)
        cand_ids_np = np.asarray(candidates["fids"], dtype=np.int64)
        base_scores_np = np.asarray(candidates["slice_score"], dtype=np.float32)
        N = int(cand_ids_np.shape[0])
        R = int(norm_emb_np.shape[0]) if norm_emb_np.ndim == 2 else 0

        norm_emb = torch.from_numpy(norm_emb_np).to(device)
        row_weights = torch.from_numpy(row_weights_np[:R]).to(device)
        q_centroid = torch.from_numpy(q_centroid_np).to(device)
        cand_ids = torch.from_numpy(cand_ids_np).to(device)
        base_scores_t = torch.from_numpy(base_scores_np).to(device)

        global_sims_t = torch.zeros(N, device=device)
        if self._func_globals_gpu is not None and N > 0:
            global_sims_t = (self._func_globals_gpu[cand_ids].float() @ q_centroid).clamp(0.0, 1.0)

        slice_sims_t = torch.zeros(N, device=device)
        f2q_sims_t = torch.zeros(N, device=device)
        slice_sims_idf_t = torch.zeros(N, device=device)
        f2q_sims_idf_t = torch.zeros(N, device=device)
        conn_t = torch.ones(N, device=device)

        offsets = self._func_partial_offsets_gpu
        if self._func_partials_gpu is not None and offsets is not None and N > 0 and R > 0:
            starts = offsets[cand_ids]
            counts = offsets[cand_ids + 1] - starts
            sum_partials = int(counts.sum().item())
            if sum_partials > 0:
                seg_of_g = torch.repeat_interleave(torch.arange(N, device=device), counts)
                group_b = torch.cat([torch.zeros(1, device=device, dtype=torch.long), counts.cumsum(0)])
                within = torch.arange(sum_partials, device=device, dtype=torch.long) - group_b[seg_of_g]
                partial_idx = starts[seg_of_g] + within
                cand_sims = (self._func_partials_gpu[partial_idx].float() @ norm_emb.T).clamp(0.0, 1.0)
                idx_2d = seg_of_g.unsqueeze(1).expand(-1, R)

                q_best = torch.zeros(N, R, device=device)
                q_best.scatter_reduce_(0, idx_2d, cand_sims, reduce="amax", include_self=False)
                slice_sims_t = ((q_best @ row_weights) / row_w_sum).clamp_min(0.0)

                if self.VERIFIER_ENABLED and R > 1:
                    row_idx = torch.arange(R, device=device).unsqueeze(0)
                    mx = q_best.amax(dim=1, keepdim=True)
                    high = (q_best >= (self.VERIFIER_TAU * mx)) & (mx > self.VERIFIER_FLOOR)
                    cnt = high.sum(dim=1).float()
                    big = float(R + 1)
                    imin = torch.where(high, row_idx, torch.full_like(row_idx, R + 1)).amin(dim=1).float()
                    imax = torch.where(high, row_idx, torch.full_like(row_idx, -1)).amax(dim=1).float()
                    span = (imax - imin + 1.0).clamp_min(1.0)
                    conn = (cnt / span).clamp(0.0, 1.0)
                    conn_t = torch.where(cnt <= 1.0, torch.ones_like(conn), conn)
                p_best = cand_sims.amax(dim=1)
                seg_sums = torch.zeros(N, device=device); seg_sums.scatter_add_(0, seg_of_g, p_best)
                f2q_sims_t = (seg_sums / counts.clamp(min=1).float()).clamp_min(0.0)

                if self._pool_slice_idf_gpu is not None:
                    w_col = self._pool_slice_idf_gpu[partial_idx]
                    weighted_cs = cand_sims * w_col.unsqueeze(1)
                    q_best_idf = torch.zeros(N, R, device=device)
                    q_best_idf.scatter_reduce_(0, idx_2d, weighted_cs, reduce="amax", include_self=False)
                    slice_sims_idf_t = ((q_best_idf @ row_weights) / row_w_sum).clamp_min(0.0)
                    seg_sums_w = torch.zeros(N, device=device); seg_sums_w.scatter_add_(0, seg_of_g, p_best * w_col)
                    w_norms = torch.zeros(N, device=device); w_norms.scatter_add_(0, seg_of_g, w_col)
                    f2q_sims_idf_t = (seg_sums_w / w_norms.clamp(min=1e-6)).clamp_min(0.0)

        alignment = torch.minimum(slice_sims_t, f2q_sims_t)
        if self._is_huge_pool():
            slice_channel = (slice_sims_t
                             + self.RERANK_HUGE_EXCESS_COEF * (f2q_sims_idf_t - slice_sims_t).clamp_min(0.0)
                             - self.RERANK_HUGE_PENALTY_COEF * (slice_sims_t - f2q_sims_idf_t).clamp_min(0.0))
            final_scores = (self.RERANK_HUGE_W_BASE * base_scores_t
                            + self.RERANK_HUGE_W_SLICE * slice_channel
                            + self.RERANK_HUGE_W_GLOBAL * global_sims_t)
        else:
            slice_channel = slice_sims_t + self.RERANK_EXCESS_COEF * (f2q_sims_idf_t - slice_sims_t).clamp_min(0.0)
            w_global = self.RERANK_W_GLOBAL_LOW + (self.RERANK_W_GLOBAL_HIGH - self.RERANK_W_GLOBAL_LOW) * alignment
            w_slice = self.RERANK_W_SLICE_HIGH - (self.RERANK_W_SLICE_HIGH - self.RERANK_W_SLICE_LOW) * alignment
            final_scores = self.RERANK_W_BASE_CONST * base_scores_t + w_slice * slice_channel + w_global * global_sims_t

        if self.VERIFIER_ENABLED:
            final_scores = final_scores * (1.0 - self.VERIFIER_WEIGHT * (1.0 - conn_t))

        out = torch.stack([
            slice_sims_t, f2q_sims_t, slice_sims_idf_t, f2q_sims_idf_t,
            slice_channel, global_sims_t, alignment, final_scores, conn_t,
        ], dim=0).cpu().numpy()

        mr = self._ensure_meta_rank()[cand_ids_np]
        order = np.lexsort((mr, -base_scores_np, -out[5], -out[4], -out[7]))
        block = {k: np.asarray(v)[order] for k, v in candidates.items()}
        block.update({
            "base_score": base_scores_np[order],
            "slice_sim": out[0][order], "f2q_sim": out[1][order],
            "q2f_idf": out[2][order], "f2q_idf": out[3][order],
            "slice_channel": out[4][order], "global_sim": out[5][order],
            "alignment": out[6][order], "score": out[7][order],
            "verify_conn": out[8][order],
        })
        return block


    def _build_query_slice_sets_cpu(
        self, query_profile: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
        """CPU fallback for Stage A1; equivalent to the GPU path (per-row FAISS
        + numpy aggregation)."""
        norm_emb = np.asarray(query_profile["norm_emb"], dtype=np.float32)
        row_weights = np.asarray(query_profile["row_weights"], dtype=np.float32)
        R = norm_emb.shape[0]
        idf_pivot = float(np.median(self.bucket_idf)) if (self.bucket_idf is not None and self.bucket_idf.size > 0) else 0.0
        spec_lo, spec_hi = float(self.SPECIFICITY_MULTIPLIER_MIN), float(self.SPECIFICITY_MULTIPLIER_MAX)

        chunks_func_ids: List[np.ndarray] = []
        chunks_hit_count: List[np.ndarray] = []
        chunks_hit_weight: List[np.ndarray] = []
        chunks_support_mass: List[np.ndarray] = []
        chunks_selectivity_mass: List[np.ndarray] = []
        chunks_maxb: List[np.ndarray] = []
        queried_bucket_ids: Set[int] = set()

        for row_idx in range(R):
            bucket_hits = self.assign_bucket_hits(norm_emb[row_idx].reshape(1, -1), topk=self.BTOPK)
            row_buckets = [int(bid) for bid, _aff in (bucket_hits[0] if bucket_hits else [])]
            queried_bucket_ids.update(row_buckets)
            specificity_multiplier = 1.0
            if idf_pivot > 0.0 and row_buckets and self.bucket_idf.size > 0:
                valid = [b for b in row_buckets if 0 <= b < self.bucket_idf.shape[0]]
                if valid:
                    row_max_idf = float(self.bucket_idf[np.asarray(valid, dtype=np.int64)].max())
                    if row_max_idf > 0.0:
                        specificity_multiplier = max(spec_lo, min(spec_hi, row_max_idf / idf_pivot))

            row_func_id_parts: List[np.ndarray] = []
            row_support_parts: List[np.ndarray] = []
            row_sig_idf_parts: List[np.ndarray] = []
            for bucket_id, bucket_affinity in (bucket_hits[0] if bucket_hits else []):
                bucket_id = int(bucket_id)
                func_ids, per_func_signal = self._bucket_func_posting(bucket_id, limit=None)
                if func_ids.size == 0:
                    continue
                row_signal = (
                    np.float32(bucket_affinity) * np.float32(self._bucket_query_weight(bucket_id)) * per_func_signal
                ).astype(np.float32, copy=False)
                row_func_id_parts.append(func_ids)
                row_support_parts.append(row_signal)
                idf_n = 0.0
                if idf_pivot > 0.0 and self.bucket_idf is not None and 0 <= bucket_id < self.bucket_idf.shape[0]:
                    idf_n = max(0.0, float(self.bucket_idf[bucket_id]) / max(idf_pivot, 1e-6))
                row_sig_idf_parts.append(row_signal * np.float32(idf_n))

            if not row_func_id_parts:
                continue
            flat_ids = np.concatenate(row_func_id_parts)
            flat_sup = np.concatenate(row_support_parts)
            flat_sig_idf = np.concatenate(row_sig_idf_parts)
            order = np.argsort(flat_ids, kind="mergesort")
            unique_ids, start = np.unique(flat_ids[order], return_index=True)
            summed = np.add.reduceat(flat_sup[order], start).astype(np.float32, copy=False)
            row_maxb = np.maximum.reduceat(flat_sig_idf[order], start).astype(np.float32, copy=False)

            base_w = float(row_weights[row_idx]) if row_idx < row_weights.shape[0] else 1.0
            row_w = np.float32(base_w * specificity_multiplier)
            n = unique_ids.shape[0]
            chunks_func_ids.append(unique_ids.astype(np.int32, copy=False))
            chunks_hit_count.append(np.ones(n, dtype=np.int32))
            chunks_hit_weight.append(np.full(n, row_w, dtype=np.float32))
            chunks_support_mass.append(row_w * summed)
            chunks_selectivity_mass.append(np.full(n, row_w / np.float32(np.sqrt(n)), dtype=np.float32))
            chunks_maxb.append(row_maxb)

        debug: Dict[str, Any] = {"query_bucket_count": len(queried_bucket_ids)}
        if not chunks_func_ids:
            empty_i, empty_f = np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
            debug["touched_func_count"] = 0
            return empty_i, {"hit_count": empty_i, "hit_weight": empty_f,
                              "support_mass": empty_f, "selectivity_mass": empty_f,
                              "max_bucket_evidence": empty_f}, debug

        flat_ids = np.concatenate(chunks_func_ids)
        order = np.argsort(flat_ids, kind="mergesort")
        unique_ids, group_start = np.unique(flat_ids[order], return_index=True)

        def _reduce(chunks: List[np.ndarray], dtype) -> np.ndarray:
            return np.add.reduceat(np.concatenate(chunks)[order], group_start).astype(dtype, copy=False)

        candidate_func_ids = unique_ids.astype(np.int32, copy=False)
        aggregates = {
            "hit_count": _reduce(chunks_hit_count, np.int32),
            "hit_weight": _reduce(chunks_hit_weight, np.float32),
            "support_mass": _reduce(chunks_support_mass, np.float32),
            "selectivity_mass": _reduce(chunks_selectivity_mass, np.float32),
            "max_bucket_evidence": np.maximum.reduceat(
                np.concatenate(chunks_maxb)[order], group_start).astype(np.float32, copy=False),
        }
        debug["touched_func_count"] = int(candidate_func_ids.shape[0])
        return candidate_func_ids, aggregates, debug

    def _score_stage_a_cpu(
        self, candidate_func_ids: np.ndarray, aggregates: Dict[str, np.ndarray],
        query_profile: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """CPU Stage A2: 6 components + STAGE_A_MAXB_WEIGHT*clip(maxb). Matches GPU."""
        M = int(candidate_func_ids.shape[0])
        if M == 0:
            return []
        cand_int = candidate_func_ids.astype(np.int64, copy=False)
        cap = int(self.CAP)

        hit_count = aggregates["hit_count"].astype(np.int64, copy=False)
        hit_weight = aggregates["hit_weight"].astype(np.float64, copy=False)
        support_mass = aggregates["support_mass"].astype(np.float64, copy=False)
        selectivity_mass = aggregates["selectivity_mass"].astype(np.float64, copy=False)

        fsc_arr = self.func_slice_count
        ftbs_arr = self.func_total_bucket_support
        func_slice_count = (
            np.maximum(1, np.asarray(fsc_arr, dtype=np.int64)[cand_int])
            if fsc_arr is not None and fsc_arr.shape[0] > 0 else np.ones(M, dtype=np.int64)
        )
        total_bucket_support = (
            np.maximum(1e-6, np.asarray(ftbs_arr, dtype=np.float64)[cand_int])
            if ftbs_arr is not None and ftbs_arr.shape[0] > 0 else np.full(M, 1e-6, dtype=np.float64)
        )

        R = int(query_profile["row_count"])
        tqw = max(1e-6, float(query_profile["total_query_weight"]))
        sym_n = np.maximum(1, np.minimum(R, func_slice_count)).astype(np.float64)
        sym_w_denom = np.maximum(1e-6, tqw * (sym_n / float(max(1, R))))
        hc_f = hit_count.astype(np.float64)

        QWR = np.minimum(1.0, hit_weight / sym_w_denom)
        PHR = np.minimum(1.0, hc_f / func_slice_count.astype(np.float64))
        MHR = np.minimum(QWR, PHR)
        SUP = 1.0 - np.exp(-support_mass / tqw)
        SEL = selectivity_mass / tqw
        SPEC = np.minimum(1.0, support_mass / total_bucket_support)
        slice_score = 0.10 * QWR + 0.28 * PHR + 0.20 * SUP + 0.18 * SPEC + 0.14 * MHR + 0.10 * SEL
        maxb = aggregates.get("max_bucket_evidence")
        if maxb is not None and len(maxb) == M:
            slice_score = slice_score + self.STAGE_A_MAXB_WEIGHT * np.clip(
                maxb.astype(np.float64, copy=False), 0.0, self.STAGE_A_MAXB_CLIP)

        order = np.lexsort((-hit_count, -SUP, -slice_score))
        if M > max(cap * 5, 10000):
            order = order[: cap * 5]

        fids = cand_int[order]
        ss = slice_score[order]
        sup = SUP[order]
        hc = hit_count[order]
        mr = self._ensure_meta_rank()[fids]
        o2 = np.lexsort((mr, -hc, -sup, -ss))[:cap]
        return {
            "fids": fids[o2].astype(np.int64, copy=False),
            "slice_score": ss[o2].astype(np.float32),
            "support": sup[o2].astype(np.float32),
            "hit_count": hc[o2].astype(np.int32),
        }

    def _dense_rerank_cpu(
        self, candidates: List[Dict[str, Any]], query_profile: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """CPU fallback for Stage B; equivalent to the GPU path (numpy matmul +
        reduceat)."""
        norm_emb = np.asarray(query_profile["norm_emb"], dtype=np.float32)
        row_weights = np.asarray(query_profile["row_weights"], dtype=np.float32)
        row_w_sum = max(1e-6, float(row_weights.sum()))
        q_centroid = np.asarray(query_profile["gq_or_centroid"], dtype=np.float32)
        cand_ids = np.asarray(candidates["fids"], dtype=np.int64)
        N = cand_ids.shape[0]

        global_sims = np.zeros(N, dtype=np.float32)
        if self.func_globals is not None and cand_ids.size:
            global_sims = np.clip(
                np.asarray(self.func_globals[cand_ids], dtype=np.float32) @ q_centroid, 0.0, 1.0
            ).astype(np.float32, copy=False)

        slice_sims = np.zeros(N, dtype=np.float32)
        f2q_sims = np.zeros(N, dtype=np.float32)
        slice_sims_idf = np.zeros(N, dtype=np.float32)
        f2q_sims_idf = np.zeros(N, dtype=np.float32)
        flat, offsets = self.func_partials_flat, self.func_partial_offsets
        if flat is not None and offsets is not None and flat.shape[0] > 0 and norm_emb.shape[0] > 0 and N > 0:
            pool_slice_w = self._ensure_pool_slice_idf()
            starts = offsets[cand_ids]
            counts = (offsets[cand_ids + 1] - starts).astype(np.int64, copy=False)
            ne_mask = counts > 0
            if np.any(ne_mask):
                ne_pos = np.flatnonzero(ne_mask)
                ne_starts, ne_counts = starts[ne_pos], counts[ne_pos]
                group_b = np.empty(ne_counts.shape[0] + 1, dtype=np.int64)
                group_b[0] = 0
                np.cumsum(ne_counts, out=group_b[1:])
                shift = ne_starts - group_b[:-1]
                partial_idx = np.arange(int(ne_counts.sum()), dtype=np.int64) + np.repeat(shift, ne_counts)
                cand_sims = np.clip(flat[partial_idx] @ norm_emb.T, 0.0, 1.0)
                starts_at = group_b[:-1].astype(np.intp, copy=False)
                q_best = np.maximum.reduceat(cand_sims, starts_at, axis=0)
                slice_sims[ne_pos] = np.maximum(0.0, (q_best @ row_weights[: norm_emb.shape[0]]) / row_w_sum)
                p_best = cand_sims.max(axis=1)
                seg_sums = np.add.reduceat(p_best, starts_at, axis=0)
                f2q_sims[ne_pos] = np.maximum(0.0, seg_sums / np.maximum(ne_counts.astype(np.float32), 1.0))
                w_col = pool_slice_w[partial_idx].astype(np.float32, copy=False)
                weighted_cs = cand_sims * w_col[:, None]
                q_best_idf = np.maximum.reduceat(weighted_cs, starts_at, axis=0)
                slice_sims_idf[ne_pos] = np.maximum(0.0, (q_best_idf @ row_weights[: norm_emb.shape[0]]) / row_w_sum)
                seg_sums_w = np.add.reduceat(p_best * w_col, starts_at, axis=0)
                w_norms = np.add.reduceat(w_col, starts_at, axis=0)
                f2q_sims_idf[ne_pos] = np.maximum(0.0, seg_sums_w / np.maximum(w_norms, 1e-6))

        base_scores = np.asarray(candidates["slice_score"], dtype=np.float32)
        alignment = np.minimum(slice_sims, f2q_sims).astype(np.float32, copy=False)
        if self._is_huge_pool():
            slice_channel = (slice_sims
                             + self.RERANK_HUGE_EXCESS_COEF * np.maximum(0.0, f2q_sims_idf - slice_sims)
                             - self.RERANK_HUGE_PENALTY_COEF * np.maximum(0.0, slice_sims - f2q_sims_idf)
                             ).astype(np.float32, copy=False)
            final_scores = (self.RERANK_HUGE_W_BASE * base_scores
                            + self.RERANK_HUGE_W_SLICE * slice_channel
                            + self.RERANK_HUGE_W_GLOBAL * global_sims).astype(np.float32, copy=False)
        else:
            slice_channel = (slice_sims + self.RERANK_EXCESS_COEF * np.maximum(0.0, f2q_sims_idf - slice_sims)).astype(np.float32, copy=False)
            w_global = self.RERANK_W_GLOBAL_LOW + (self.RERANK_W_GLOBAL_HIGH - self.RERANK_W_GLOBAL_LOW) * alignment
            w_slice = self.RERANK_W_SLICE_HIGH - (self.RERANK_W_SLICE_HIGH - self.RERANK_W_SLICE_LOW) * alignment
            final_scores = (self.RERANK_W_BASE_CONST * base_scores + w_slice * slice_channel + w_global * global_sims).astype(np.float32, copy=False)

        mr = self._ensure_meta_rank()[cand_ids]
        order = np.lexsort((mr, -base_scores, -global_sims, -slice_channel, -final_scores))
        block = {k: np.asarray(v)[order] for k, v in candidates.items()}
        block.update({
            "base_score": base_scores[order],
            "slice_sim": slice_sims[order], "f2q_sim": f2q_sims[order],
            "q2f_idf": slice_sims_idf[order], "f2q_idf": f2q_sims_idf[order],
            "slice_channel": slice_channel[order], "global_sim": global_sims[order],
            "alignment": alignment[order], "score": final_scores[order],
        })
        return block



def load_query_pairs(
    query_dir: str,
    limit_per_file: Optional[int] = None,
    limit_total: Optional[int] = None,
    with_query_loc: bool = False,
) -> List[Tuple]:
    """Read *.pkl; each query -> (q, gq, type, gt_binary, gt_addr, query_row_meta).
    with_query_loc=True appends (query_binary, query_addr) for the LLM verifier to
    fetch query pseudocode by location; default False keeps the 6-tuple."""
    pairs: List[Tuple] = []
    for name in tqdm(sorted(os.listdir(query_dir)), desc="eval: load queries", dynamic_ncols=True):
        if not name.endswith(".pkl"):
            continue
        infos = read_pickle(os.path.join(query_dir, name))
        if limit_per_file is not None:
            infos = infos[:limit_per_file]
        for info in infos:
            q = np.asarray(info["q"], dtype=np.float32)
            if q.ndim == 1:
                q = q.reshape(1, -1) if q.size else q.reshape(0, 0)
            if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] == 0:
                continue
            rec = [
                q, np.asarray(info["gq"], dtype=np.float32),
                int(info.get("type", 2)), info["gt_binary"], info["addr"],
                info.get("query_row_meta") or [],
            ]
            if with_query_loc:
                rec += [info.get("binary"), info.get("query_addr"), info.get("q_text")]
            pairs.append(tuple(rec))
            if limit_total is not None and len(pairs) >= limit_total:
                return pairs
    return pairs


def evaluate_metrics(
    db: EmbeddingRetrievalDB,
    query_pairs,
    recall_ks=(1, 5, 10, 50),
    emit_details: bool = False,
    name_to_keys: Optional[Dict[str, Set[Tuple[str, str]]]] = None,
    keys_to_name: Optional[Dict[Tuple[str, str], str]] = None,
    collect_topk: int = 0,
):
    """Run db.search once per query; aggregate recall@k with a per-type split.
    emit_details=True also returns per-query gt_rank records.

    Loose match: when ``name_to_keys`` and ``keys_to_name`` are provided, the
    set of "valid GT" pool entries for each query is expanded from the strict
    ``{(gt_binary, gt_addr)}`` to every ``(binary, addr)`` in the pool sharing
    the GT's ``func_name``. This aligns with ``resolve_gt``'s "any of them is
    a valid GT" cross-config semantics and avoids penalising the retriever
    when same-source clones (e.g. cross-CVE-case copies of a library helper)
    occupy ranks the strict GT instance would otherwise take.

    Both maps are derived from the materialised pool. When omitted, eval
    behaves identically to the strict-tuple legacy.
    """
    recall_ks = tuple(sorted(set(int(k) for k in recall_ks)))
    recall_hits = {k: 0 for k in recall_ks}
    type_counts = {0: 0, 1: 0, 2: 0}
    type_hits = {t: {k: 0 for k in recall_ks} for t in type_counts}
    query_seconds: List[float] = []
    misses: List[Dict[str, Any]] = []
    details: Optional[List[Dict[str, Any]]] = [] if emit_details else None
    total = 0
    gt_recall_hits = 0
    loose_enabled = name_to_keys is not None and keys_to_name is not None
    collected: List[Dict[str, Any]] = [] if collect_topk else []
    key2fid: Optional[Dict[Tuple[str, Any], int]] = None

    for rec in tqdm(
        query_pairs, desc="eval: queries", total=len(query_pairs), dynamic_ncols=True,
    ):
        q, gq, qt, gt_binary, gt_addr, query_row_meta = rec[:6]
        q_binary, q_addr = (rec[6], rec[7]) if len(rec) >= 8 else (None, None)
        q_text = rec[8] if len(rec) >= 9 else None
        strict_key = (gt_binary, gt_addr)
        hit_keys: Set[Tuple[str, str]] = {strict_key}
        loose_count = 0
        if loose_enabled:
            gt_func_name = keys_to_name.get(strict_key, "")
            if gt_func_name:
                loose_extra = name_to_keys.get(gt_func_name, set())
                if loose_extra:
                    hit_keys = hit_keys | loose_extra
                    loose_count = len(hit_keys) - 1

        results, debug = db.search(q, global_query=gq, query_row_meta=query_row_meta,
                                   return_debug=True, query_loc=(q_binary, q_addr))
        ranked = results["recall_results"]
        gt_rank = next(
            (int(it["rank"]) for it in ranked
             if (it["binary_name"], it["func_addr"]) in hit_keys),
            None,
        )
        total += 1
        type_counts[qt] = type_counts.get(qt, 0) + 1
        recall_hit = (gt_rank is not None) or bool(hit_keys & debug.get("recalled_func_keys", set()))
        touched_fids = debug.get("touched_fids")
        if touched_fids is not None:
            if key2fid is None:
                key2fid = {(b, a): i for i, (b, a, _nm) in enumerate(db.index_to_meta)}
            hf = [key2fid[k] for k in hit_keys if k in key2fid]
            ta = np.asarray(touched_fids, dtype=np.int64)
            if ta.size and hf:
                hfa = np.asarray(hf, dtype=np.int64)
                pos = np.clip(np.searchsorted(ta, hfa), 0, ta.size - 1)
                gt_touched = bool(np.any(ta[pos] == hfa))
            else:
                gt_touched = False
        else:
            gt_touched = bool(hit_keys & debug.get("touched_func_keys", set()))
        if recall_hit:
            gt_recall_hits += 1
        for k in recall_ks:
            if gt_rank is not None and gt_rank <= k:
                recall_hits[k] += 1
                type_hits[qt][k] += 1
        query_seconds.append(float(debug.get("total_search_time_s", 0.0)))
        if not recall_hit:
            misses.append({
                "gt_type": type_name(qt), "gt_binary": gt_binary, "gt_func_addr": gt_addr,
                "gt_rank": int(gt_rank) if gt_rank is not None else None,
                "miss_reason": "ranked_out" if gt_touched else "not_touched",
                "loose_alt_keys": int(loose_count),
            })
        if details is not None:
            details.append({
                "gt_type": type_name(qt), "gt_binary": gt_binary, "gt_func_addr": gt_addr,
                "gt_rank": int(gt_rank) if gt_rank is not None else None,
                "query_R": int(q.shape[0]) if hasattr(q, "shape") and q.ndim == 2 else 0,
                "recall_hit": bool(recall_hit),
                "gt_touched": gt_touched,
                "loose_alt_keys": int(loose_count),
            })
        if collect_topk:
            topk = []
            for it in ranked[:collect_topk]:
                key = (it["binary_name"], it["func_addr"])
                topk.append((it["binary_name"], it["func_addr"], key in hit_keys))
            collected.append({
                "qt": int(qt), "q_binary": q_binary, "q_addr": q_addr, "q_text": q_text,
                "gt_rank": int(gt_rank) if gt_rank is not None else None,
                "topk": topk,
            })

    metrics: Dict[str, float] = {"queries": float(total)}
    if total:
        metrics["gt_recall_rate"] = gt_recall_hits / total
        for k in recall_ks:
            metrics[f"recall@{k}"] = recall_hits[k] / total
        metrics["avg_query_seconds"] = float(np.mean(query_seconds))
        metrics["p95_query_seconds"] = float(np.percentile(query_seconds, 95))
        for qt, count in type_counts.items():
            metrics[f"{type_name(qt)}_queries"] = float(count)
            for k in recall_ks:
                metrics[f"{type_name(qt)}_recall@{k}"] = type_hits[qt][k] / count if count else 0.0
    if collect_topk:
        return metrics, misses, details, collected
    return metrics, misses, details


_ART = Path(CURRENT_DIR) / "artifacts" / "tmp"
DEFAULT_POOL_DIR = (_ART / "pool").resolve()
DEFAULT_QUERY_DIR = (_ART / "query").resolve()
DEFAULT_FAISS_MODEL = (_ART / "inline_slice_pool.index").resolve()
DEFAULT_SAVE_DB = (_ART / "eval").resolve()
DEFAULT_POOL_SOURCE_DIR = (Path(CURRENT_DIR) / "data" / "inline" / "pool").resolve()
DEFAULT_POOL_SOURCE_DIRS = [str(DEFAULT_POOL_SOURCE_DIR)]
DEFAULT_EMBED_MODEL = os.getenv("TRACE_ENCODER_PATH", "")

def _manifest_pool_source_dirs(m: Dict[str, object]) -> List[str]:
    """Read ``pool_source_dir`` from a pool manifest as a list of abspath.

    The field may be a single string (legacy manifests) or a list (current).
    """
    raw = m.get("pool_source_dir")
    if isinstance(raw, str):
        if not raw:
            return []
        return [os.path.abspath(raw)]
    if isinstance(raw, (list, tuple)):
        return [os.path.abspath(str(p)) for p in raw if p]
    return []


def prepare_eval_pool_dir(args) -> str:
    """Build the eval pool (skip if manifest matches)."""
    built = args.pool_dir or str((Path(args.save_db) / "built_pool").resolve())
    manifest = Path(built) / "pool_manifest.json"
    arg_sources = [os.path.abspath(p) for p in args.pool_source_dir]
    if manifest.exists():
        try:
            m = json.loads(manifest.read_text(encoding="utf-8"))
            if getattr(args, "trust_pool", False) and int(m.get("pool_size", -1)) == int(args.pool_size):
                return built
            if (os.path.abspath(str(m.get("query_dir", ""))) == os.path.abspath(args.query_dir)
                    and _manifest_pool_source_dirs(m) == arg_sources
                    and int(m.get("pool_size", -1)) == int(args.pool_size)):
                return built
        except (json.JSONDecodeError, OSError):
            pass
    embedder = None
    if getattr(args, "embed_model", None):
        raise SystemExit(
            "On-the-fly encoding needs the TRACE encoder, which is not "
            "distributed. Use the released pre-embedded pools instead.")
    build_eval_pool(
        query_dir=args.query_dir, pool_source_dir=args.pool_source_dir,
        pool_size=int(args.pool_size), output_dir=built, embedder=embedder,
        feature_suffix=args.feature_suffix, embedding_suffix=args.embedding_suffix,
    )
    return built


def _build_pool_name_index(
    built_pool_dir: str,
) -> Tuple[Dict[str, Set[Tuple[str, str]]], Dict[Tuple[str, str], str]]:
    """Build (name_to_keys, keys_to_name) from the materialized pool manifest.

    Used by ``evaluate_metrics`` to enable loose match: any pool entry that
    shares the strict GT's ``func_name`` becomes a valid alt-GT key. This is
    the BCSD-correct semantics that ``resolve_gt`` already implements on the
    query-construction side ("any of them is a valid GT for BCSD eval").
    """
    manifest_path = Path(built_pool_dir) / "pool_manifest.json"
    if not manifest_path.exists():
        return {}, {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}, {}
    name_to_keys: Dict[str, Set[Tuple[str, str]]] = {}
    keys_to_name: Dict[Tuple[str, str], str] = {}
    for item in manifest.get("selected_functions") or []:
        binary_name = str(item.get("binary_name") or "")
        func_addr = str(item.get("func_addr") or "")
        func_name = str(item.get("func_name") or "")
        if not binary_name or not func_addr or not func_name:
            continue
        key = (binary_name, func_addr)
        keys_to_name[key] = func_name
        name_to_keys.setdefault(func_name, set()).add(key)
    return name_to_keys, keys_to_name


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate bucket->func recall retrieval.")
    p.add_argument("--pool-dir", default=str(DEFAULT_POOL_DIR))
    p.add_argument(
        "--pool-source-dir",
        nargs="+",
        default=list(DEFAULT_POOL_SOURCE_DIRS),
        help="One or more pool source roots. Each is rglob-scanned for "
             "*_bb_slice_feature.json. On binary_name collision across roots, "
             "the first listed root wins.",
    )
    p.add_argument("--pool-size", type=int, default=100000)
    p.add_argument("--trust-pool", action="store_true",
                   help="Accept an existing materialized pool (pool_manifest.json) whose "
                        "pool_size matches, without re-checking the absolute build paths "
                        "recorded in the manifest. Use with downloaded artifact pools.")
    p.add_argument("--query-dir", default=str(DEFAULT_QUERY_DIR))
    p.add_argument("--model", default=str(DEFAULT_FAISS_MODEL))
    p.add_argument("--embed-model", default=str(DEFAULT_EMBED_MODEL))
    p.add_argument("--embed-gpu-id", type=int, default=1)
    p.add_argument("--save-db", default=str(DEFAULT_SAVE_DB))
    p.add_argument("--feature-suffix", default="_bb_slice_feature.json")
    p.add_argument("--embedding-suffix", default=None)
    p.add_argument("--limit-per-file", type=int, default=None)
    p.add_argument("--limit-total", type=int, default=None)
    p.add_argument("--emit-details", action="store_true",
                   help="Also write recall_eval_details.json with per-query gt_rank.")
    p.add_argument("--cap", type=int, default=None,
                   help="Override CAP. Default = recommended_cap(--pool-size) = "
                        "max(500, ceil(6*sqrt(N))) per RQ_topk_cap_empirical_formula_v2.md.")
    p.add_argument("--btopk", type=int, default=None,
                   help="Override BTOPK (default 16, per RQ_topk_cap_empirical_formula_v2.md).")
    p.add_argument("--rerank-device", default='cuda:1',
                   help="Override EmbeddingRetrievalDB.RERANK_DEVICE (default cuda:0). "
                        "Use e.g. 'cuda:1' to avoid contention with a busy GPU 0.")
    p.add_argument("--verifier", action="store_true",
                   help="Enable the post-Stage-B connectivity verifier: demote candidates "
                        "whose high-similarity query slices are scattered (not contiguous) "
                        "in pseudocode order. Off by default.")
    p.add_argument("--verifier-weight", type=float, default=None,
                   help="Verifier demotion strength w (default %.2f). final*=(1-w*(1-conn))."
                        % EmbeddingRetrievalDB.VERIFIER_WEIGHT)
    p.add_argument("--verifier-tau", type=float, default=None,
                   help="High-similarity threshold tau (default %.2f): slice r is 'hit' if "
                        "s_r >= tau*max_r(s_r)." % EmbeddingRetrievalDB.VERIFIER_TAU)
    p.add_argument("--anchor", action="store_true",
                   help="Enable the symbolic-anchor promote verifier: after Stage B sorts the "
                        "whole CAP, add lambda*idf-weighted shared-symbol(const/string/import) "
                        "overlap to each candidate's score and re-sort. Off by default.")
    p.add_argument("--anchor-lambda", type=float, default=None,
                   help="Anchor promote strength (default %.3f). score' = score + lambda*overlap."
                        % EmbeddingRetrievalDB.ANCHOR_LAMBDA)
    p.add_argument("--anchor-query-dir", nargs="+", default=["data/inline/strip", "data/inline/noinline"],
                   help="Dirs holding the QUERY-side *_bb_slice_feature.json for query-anchor "
                        "lookup (strip build first, noinline fallback). The pool dir is the "
                        "NOINLINE build and misses inline-host EAs, so it can't be the query source.")
    p.add_argument("--anchor-pool-dir", default='data/inline/pool',
                   help="Dir of POOL-side *_bb_slice_feature.json used to build/load the pool "
                        "anchors. Default: the materialized eval pool. Point at data/inline/pool "
                        "(or the pool you search) to reuse one prebuilt index across runs.")
    p.add_argument("--anchor-index", default=None,
                   help="Path to a prebuilt anchor index (pool map + idf + Dice mass) from "
                        "build_anchor_index.py. If present it is loaded directly, skipping the "
                        "online pool scan; if given but missing it is built once and written there. "
                        "Default: <anchor-pool-dir>/anchor_index_<suffix>.pkl.")
    p.add_argument("--llm-verify", action="store_true",
                   help="After Stage A+B, run the deepseek CoT verifier on each query's "
                        "top-K candidates and emit a SECOND, LLM-reranked metric set "
                        "(recall_eval_metrics_llm.json). The baseline output is unchanged.")
    p.add_argument("--llm-verify-topk", type=int, default=25,
                   help="Verify the top-K candidates per query (default 50).")
    p.add_argument("--llm-concurrency", type=int, default=400,
                   help="Concurrent LLM requests (deepseek default 400).")
    p.add_argument("--llm-model", default="deepseek-v4-flash",
                   help="LLM verifier model (deepseek-v4-flash; qwen excluded — low cache/slow).")
    p.add_argument("--llm-strip-feat-dir", default="data/inline/strip",
                   help="Dir with the QUERY-side *_bb_slice_feature.json (fallback for query "
                        "pseudocode when query pkls predate the in-record q_text field).")
    p.add_argument("--llm-text-index", default=None,
                   help="sqlite (binary,addr)->pseudocode index for candidate text. Default "
                        "<pool-source-dir>/pool_func_text.sqlite; built once if missing.")
    p.add_argument("--llm-index-workers", type=int, default=8,
                   help="Processes for building the func-text index (parallel JSON parse).")
    p.add_argument("--llm-api-key", default=None,
                   help="DeepSeek API key (default $DEEPSEEK_API_KEY).")
    return p


def _print_metrics(tag, metrics):
    """Print overall recall plus the per-type (root/internal/leaf) breakdown."""
    def _kk(s):
        try:
            return int(str(s).split("@", 1)[1])
        except Exception:
            return 1 << 30
    ks = sorted((k for k in metrics if str(k).startswith("recall@")), key=_kk)
    overall = {"queries": metrics.get("queries"), "gt_recall_rate": metrics.get("gt_recall_rate")}
    overall.update({k: metrics[k] for k in ks})
    print("[%s]" % tag, overall, flush=True)
    for t in (0, 1, 2):
        name = type_name(t)
        nq = metrics.get("%s_queries" % name)
        if not nq:
            continue
        row = {"queries": nq}
        row.update({k: metrics.get("%s_%s" % (name, k)) for k in ks})
        print("[%s:%s]" % (tag, name), row, flush=True)


def main():
    args = build_arg_parser().parse_args()
    if args.embedding_suffix is None:
        from utils import derive_embedding_suffix
        args.embedding_suffix = derive_embedding_suffix(args.feature_suffix)
    built = prepare_eval_pool_dir(args)
    faiss_gpu_id = 1
    if args.rerank_device and args.rerank_device.startswith("cuda:"):
        try:
            faiss_gpu_id = int(args.rerank_device.split(":", 1)[1])
        except ValueError:
            faiss_gpu_id = 0
    db = EmbeddingRetrievalDB(built, args.model, args.save_db,
                              feature_suffix=args.feature_suffix, embedding_suffix=args.embedding_suffix,
                              faiss_gpu_id=faiss_gpu_id)
    db.CAP = int(args.cap) if args.cap is not None else EmbeddingRetrievalDB.recommended_cap(int(args.pool_size))
    if args.btopk is not None:
        db.BTOPK = int(args.btopk)
    print("[budget] BTOPK=%d CAP=%d (empirical: BTOPK=16, CAP=max(500,ceil(6*sqrt(N))); pool_size=%d%s)"
          % (db.BTOPK, db.CAP, int(args.pool_size),
             ", CLI override" if (args.cap is not None or args.btopk is not None) else ""), flush=True)
    if args.rerank_device is not None:
        db.RERANK_DEVICE = str(args.rerank_device)
    db.VERIFIER_ENABLED = bool(args.verifier)
    if args.verifier_weight is not None:
        db.VERIFIER_WEIGHT = float(args.verifier_weight)
    if args.verifier_tau is not None:
        db.VERIFIER_TAU = float(args.verifier_tau)
    if db.VERIFIER_ENABLED:
        print("[verifier] ON  weight=%.2f tau=%.2f (connectivity demotion, GPU rerank path)"
              % (db.VERIFIER_WEIGHT, db.VERIFIER_TAU), flush=True)
    db.build_pool()
    if args.anchor:
        if args.anchor_lambda is not None:
            db.ANCHOR_LAMBDA = float(args.anchor_lambda)
        anchor_pool_dir = args.anchor_pool_dir or built
        db._anchor_index_path = (args.anchor_index
                                 or verifiers.default_anchor_index_path(anchor_pool_dir, args.feature_suffix))
        if os.path.exists(db._anchor_index_path):
            print("[anchor] ON  loading prebuilt index %s" % db._anchor_index_path, flush=True)
        else:
            print("[anchor] ON  building pool anchor map from %s (a few min on a 1M pool; "
                  "caching to %s)..." % (anchor_pool_dir, db._anchor_index_path), flush=True)
        db.prepare_anchors(anchor_pool_dir, args.anchor_query_dir)
    db._ensure_gpu_state()
    name_to_keys, keys_to_name = _build_pool_name_index(built)
    recall_ks = (1, 5, 10, 25, 50, 100, 200, 500, 1000)
    qp = load_query_pairs(args.query_dir, limit_per_file=args.limit_per_file,
                          limit_total=args.limit_total,
                          with_query_loc=bool(args.llm_verify or args.anchor))
    save_dir = Path(args.save_db); save_dir.mkdir(parents=True, exist_ok=True)
    topk = int(args.llm_verify_topk)

    def _eval(collect):
        return evaluate_metrics(db, qp, recall_ks=recall_ks, emit_details=bool(args.emit_details),
                                name_to_keys=name_to_keys, keys_to_name=keys_to_name,
                                collect_topk=collect)

    def _finalize(m):
        m["queries"] = int(m["queries"]); m["built_pool_dir"] = built
        m["pool_size"] = int(args.pool_size)
        return m

    def _write(name, metrics, misses):
        (save_dir / ("recall_eval_metrics%s.json" % name)).write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        (save_dir / ("recall_eval_misses%s.json" % name)).write_text(
            json.dumps(misses, indent=2, ensure_ascii=False), encoding="utf-8")


    db.ANCHOR_ENABLED = False
    base_collect = topk if (args.llm_verify and not args.anchor) else 0
    ev = _eval(base_collect)
    if base_collect:
        metrics, misses, details, collected = ev
    else:
        metrics, misses, details = ev; collected = None
    metrics = _finalize(metrics)
    _write("", metrics, misses)
    if details is not None:
        (save_dir / "recall_eval_details.json").write_text(
            json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_metrics("baseline", metrics)
    gt_rate = metrics.get("gt_recall_rate")

    if args.anchor:
        db.ANCHOR_ENABLED = True
        anchor_collect = topk if args.llm_verify else 0
        ev = _eval(anchor_collect)
        if anchor_collect:
            a_metrics, a_misses, _ad, collected = ev
        else:
            a_metrics, a_misses, _ad = ev
        a_metrics = _finalize(a_metrics)
        a_metrics["anchor_lambda"] = float(db.ANCHOR_LAMBDA)
        _write("_anchor", a_metrics, a_misses)
        _print_metrics("anchor", a_metrics)

    if args.llm_verify:
        api_key = args.llm_api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit("--llm-verify needs a DeepSeek key: set $DEEPSEEK_API_KEY or --llm-api-key")
        index_src_dir = args.pool_source_dir[0] if args.pool_source_dir else built
        text_index_path = args.llm_text_index or verifiers.default_functext_sqlite_path(index_src_dir)
        llm_metrics, llm_misses, _ = verifiers.llm_verify_and_rerank(
            collected, text_index_path=text_index_path, index_src_dir=index_src_dir,
            strip_feat_dir=args.llm_strip_feat_dir, feature_suffix=args.feature_suffix,
            recall_ks=recall_ks, model=args.llm_model, concurrency=int(args.llm_concurrency),
            api_key=api_key, topk=topk, index_workers=int(args.llm_index_workers),
        )
        llm_metrics["queries"] = int(llm_metrics["queries"])
        if gt_rate is not None:
            llm_metrics["gt_recall_rate"] = gt_rate
        llm_metrics["built_pool_dir"] = built
        llm_metrics["pool_size"] = int(args.pool_size)
        llm_metrics["llm_verify_topk"] = topk
        llm_metrics["llm_chained_on"] = "anchor" if args.anchor else "baseline"
        _write("_llm", llm_metrics, llm_misses)
        _print_metrics("llm-verify", llm_metrics)


if __name__ == "__main__":
    main()
