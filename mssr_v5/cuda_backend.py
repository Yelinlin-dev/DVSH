"""CUDA encoding and exact integer scoring for the existing MSSR v5 protocol.

LSH lookup/grouping, relation graph construction, and sorting remain on CPU.
Semantic vectors and NumPy row thresholds are shared with the reference engine.
"""
from pathlib import Path
import time

import numpy as np
import torch

from mssr_v5.aligned_fast import ASMSSREngine, pack_uint64


class CUDAASMSSREngine(ASMSSREngine):
    def __init__(self, *, device="cuda:0", profile_cache_dir: Path, **kwargs):
        started = time.perf_counter()
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("A CUDA device is required")
        torch.cuda.set_device(self.device)
        torch.empty(1, device=self.device)
        torch.cuda.synchronize(self.device)
        self.accelerator_initialization_ms = (time.perf_counter() - started) * 1000.0
        self.profile_cache_dir = Path(profile_cache_dir)
        self.gpu = {}
        super().__init__(**kwargs)

    def _aligned_fast_cache_path(self, config):
        return self.profile_cache_dir / super()._aligned_fast_cache_path(config).name

    def _prepare_semantics(self):
        graph_ids, embeddings = self._load_semantic_arrays()
        positions = {str(gid): row for row, gid in enumerate(graph_ids)}
        db_rows = np.fromiter((positions[gid] for gid in self.ids), dtype=np.intp, count=len(self.ids))
        query_rows = np.fromiter((positions[gid] for gid in self.task.query_ids), dtype=np.intp,
                                 count=len(self.task.query_ids))
        # Advanced indexing creates writable C-contiguous matrices in precisely
        # the same task order as the reference's per-vector lookup and stack.
        db_sem = embeddings[db_rows].astype(np.float32, copy=False)
        query_sem = embeddings[query_rows].astype(np.float32, copy=False)
        db_sem /= np.maximum(np.linalg.norm(db_sem, axis=1, keepdims=True), 1e-8)
        query_sem /= np.maximum(np.linalg.norm(query_sem, axis=1, keepdims=True), 1e-8)
        return db_sem, query_sem

    def _encode_binary(self, db_sem, query_sem):
        with torch.no_grad():
            for side, matrix in (("db", db_sem), ("query", query_sem)):
                values = torch.as_tensor(matrix, device=self.device)
                # Reuse reference thresholds to preserve binary codes exactly.
                threshold = torch.as_tensor(matrix.mean(axis=1, keepdims=True), device=self.device)
                for kind, mask in (("sign", values >= 0.0), ("row", values >= threshold)):
                    self.gpu[f"{side}_{kind}"] = mask
                    bits = mask.cpu().numpy()
                    setattr(self, f"{side}_{kind}_bits", bits)
                    setattr(self, f"{side}_{kind}64", pack_uint64(bits))
        torch.cuda.synchronize(self.device)

    def _prepare_accelerator(self):
        started = time.perf_counter()
        self.gpu["db_counts"] = torch.as_tensor(self.aligned_fast_db_counts, device=self.device)
        self.gpu["query_counts"] = torch.as_tensor(self.aligned_fast_query_counts, device=self.device)
        torch.cuda.synchronize(self.device)
        return (time.perf_counter() - started) * 1000.0

    def _gpu_lsh_keys(self, lsh):
        key_dtype = np.uint16 if lsh.key_bits <= 16 else np.uint32 if lsh.key_bits <= 32 else np.uint64
        arithmetic = torch.int32 if lsh.key_bits <= 31 else torch.int64
        weights_np = lsh.weights.astype(np.int32) if arithmetic == torch.int32 else lsh.weights.view(np.int64)
        weights = torch.as_tensor(weights_np, device=self.device)
        keys = np.empty((lsh.tables, lsh.sign_bits.shape[0]), dtype=key_dtype)
        with torch.no_grad():
            for indices, kind in ((lsh.global_indices, "sign"), (lsh.relative_indices, "row")):
                if not indices.size:
                    continue
                positions = torch.as_tensor(lsh.table_positions[indices], dtype=torch.long, device=self.device)
                selected = self.gpu[f"db_{kind}"][:, positions]
                values = (selected.to(arithmetic) * weights).sum(dim=-1, dtype=arithmetic)
                host = values.cpu().numpy()
                # Signed int64 accumulation has the same bit pattern as uint64,
                # including a selected high bit when key_bits == 64.
                if arithmetic == torch.int64:
                    host = host.view(np.uint64)
                keys[indices] = host.T.astype(key_dtype, copy=False)
        torch.cuda.synchronize(self.device)
        return keys

    def _build_lsh(self):
        self.lsh.build(key_builder=self._gpu_lsh_keys)

    def _binary_distances(self, query_no, candidates):
        with torch.no_grad():
            index = torch.as_tensor(candidates, dtype=torch.long, device=self.device)
            distances = [
                (self.gpu[f"db_{kind}"][index] != self.gpu[f"query_{kind}"][query_no])
                .sum(dim=1).float().cpu().numpy()
                for kind in ("sign", "row")
            ]
        return tuple(distances)

    def _count_similarity(self, query_no, head, timings):
        started = time.perf_counter()
        with torch.no_grad():
            index = torch.as_tensor(head, dtype=torch.long, device=self.device)
            candidates = self.gpu["db_counts"][index]
            query = self.gpu["query_counts"][query_no]
            torch.cuda.synchronize(self.device)
            timings["aligned_profile_gather_time_ms"] += (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            numerator = torch.minimum(candidates, query).sum(dim=1, dtype=torch.int32).float()
            denominator = torch.maximum(candidates, query).sum(dim=1, dtype=torch.int32).float()
            score = torch.where(denominator > 0, numerator / denominator.clamp_min(1), 0.0)
            result = score.cpu().numpy()
            timings["aligned_similarity_time_ms"] += (time.perf_counter() - started) * 1000.0
        return result

    def prepare(self, config, **kwargs):
        torch.cuda.reset_peak_memory_stats(self.device)
        super().prepare(config, **kwargs)
        self.stats.update(
            execution_device=str(self.device), score_backend="torch_cuda",
            binary_encoding_backend="torch_cuda_reference_thresholds",
            relation_profile_backend="numpy_cpu_sparse_accumulation", lsh_backend="torch_cuda_keys_numpy_stable_grouping",
            accelerator_initialization_time_ms=self.accelerator_initialization_ms,
            peak_gpu_memory_mb=torch.cuda.max_memory_allocated(self.device) / 1024**2,
            gpu_index_size_mb=sum(t.numel() * t.element_size() for t in self.gpu.values()) / 1024**2,
            relation_profile_cache_policy="fresh_run_directory",
        )

    def run(self, config):
        report = super().run(config)
        report["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated(self.device) / 1024**2
        return report
