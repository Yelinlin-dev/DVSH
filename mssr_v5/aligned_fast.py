"""MSSR v5 retrieval: semantic binary LSH + aligned simplicial fast rerank."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import time

import numpy as np

from mssr_v5.data import load_task_corpus
from mssr_v5.task import RetrievalTaskV2, load_task
from mssr_v5.semantic_cache_io import load_semantic_arrays
from mssr_v5.offline_timing import DEFAULT_CLIP_TIMINGS, load_fastest_clip_timing, account_clip_time


ROOT = Path(__file__).resolve().parents[1]
POPCOUNT8 = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint16)


@dataclass(frozen=True)
class ASMSSRConfig:
    method: str = "semantic_binary_lsh_simplicial_aligned_fast"
    seed: int = 3407
    semantic_lsh_candidates: int = 1024
    fixed_candidates: int = 1024
    adaptive_budget: bool = False
    dtype: str = "float32"
    lsh_tables: int = 64
    lsh_key_bits: int = 10
    lsh_bucket_size: int = 0
    lsh_probe_radius: int = 1
    lsh_limited_probe_m: int = 5
    lsh_probe_strategy: str = "uncertainty_guided"
    lsh_dedup_strategy: str = "sort_unique"
    candidate_backfill: bool = True
    lsh_query_flip_bits: int = 0
    semantic_score_noise_std: float = 0.0
    binary_sign_weight: float = 1.0
    binary_rowmean_weight: float = 2.0
    topk: int = 50
    simplicial_rerank_topr: int = 128
    simplicial_bits: int = 128
    simplicial_lambda: float = 0.2
    exact_relation_weight: float = 1.0
    exact_pair_weight: float = 0.0
    relation_alignment_mode: str = "offline_top1_db_space"
    relation_embedding_dim: int = 256
    relation_align_threshold: float = 0.6
    fast_profile: str = "count"
    use_adaptive_simplicial: bool = True
    adaptive_margin_bits: int = 8


class BinarySemanticLSH:
    def __init__(
        self,
        sign_bits: np.ndarray,
        row_bits: np.ndarray,
        *,
        seed: int,
        tables: int,
        key_bits: int,
        bucket_size: int,
        probe_radius: int,
        limited_probe_m: int = 4,
        probe_strategy: str = "uncertainty_guided",
        dedup_strategy: str = "sort_unique",
    ) -> None:
        self.sign_bits = np.asarray(sign_bits, dtype=bool)
        self.row_bits = np.asarray(row_bits, dtype=bool)
        self.seed = seed
        self.tables = tables
        self.key_bits = key_bits
        self.bucket_size = bucket_size
        self.probe_radius = probe_radius
        self.limited_probe_m = int(limited_probe_m)
        self.probe_strategy = probe_strategy
        self.dedup_strategy = dedup_strategy
        self.weights = (1 << np.arange(self.key_bits - 1, -1, -1, dtype=np.uint64)).astype(np.uint64)
        self.xor_masks = self.weights.copy()
        self.specs: list[tuple[np.ndarray, bool]] = []
        self.table_positions: np.ndarray | None = None
        self.use_row_mask: np.ndarray | None = None
        self.global_indices: np.ndarray | None = None
        self.relative_indices: np.ndarray | None = None
        self.buckets: list[dict[int, np.ndarray]] = []

    def _compact_keys(self, bits: np.ndarray) -> np.ndarray:
        # The default ten-bit keys fit exactly in uint16. Its stable sort uses
        # a much cheaper radix path than sorting the same values as uint64.
        dtype = np.uint16 if self.key_bits <= 16 else np.uint32 if self.key_bits <= 32 else np.uint64
        return bits.astype(dtype, copy=False) @ self.weights.astype(dtype, copy=False)

    def _group_bucket_indices(self, keys: np.ndarray) -> dict[int, np.ndarray]:
        if keys.size == 0:
            return {}
        # Stable grouping keeps database order within a bucket, including when
        # only the first bucket_size entries are retained. Order the groups by
        # their first database entry to preserve the original dictionary order.
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
        ends = np.r_[starts[1:], keys.size]
        group_order = np.argsort(order[starts], kind="stable")
        rows = {}
        for group in group_order:
            start, end = int(starts[group]), int(ends[group])
            if self.bucket_size > 0:
                end = min(end, start + self.bucket_size)
            rows[int(sorted_keys[start])] = order[start:end].copy()
        return rows

    def build(self, key_builder=None) -> None:
        rng = np.random.default_rng(self.seed + 44021)
        width = self.sign_bits.shape[1]
        specs: list[np.ndarray] = []
        use_rows: list[bool] = []
        for table in range(self.tables):
            bit_index = np.sort(rng.choice(width, size=min(self.key_bits, width), replace=False))
            use_row = bool(table % 2)
            bit_index = bit_index.astype(np.intp, copy=False)
            self.specs.append((bit_index, use_row))
            specs.append(bit_index)
            use_rows.append(use_row)
        self.table_positions = np.vstack(specs)
        self.use_row_mask = np.asarray(use_rows, dtype=bool)
        self.global_indices = np.flatnonzero(~self.use_row_mask)
        self.relative_indices = np.flatnonzero(self.use_row_mask)
        all_keys = key_builder(self) if key_builder is not None else None
        for table, (bit_index, use_row) in enumerate(zip(specs, use_rows)):
            if all_keys is None:
                bits = self.row_bits[:, bit_index] if use_row else self.sign_bits[:, bit_index]
                keys = self._compact_keys(bits)
            else:
                keys = all_keys[table]
            self.buckets.append(self._group_bucket_indices(keys))

    @staticmethod
    def _sort_unique_ranked(chunks: list[np.ndarray]) -> np.ndarray:
        if not chunks:
            return np.empty(0, dtype=np.intp)
        merged = np.concatenate(chunks)
        if merged.size == 0:
            return np.empty(0, dtype=np.intp)
        merged.sort(kind="quicksort")
        starts = np.r_[0, np.flatnonzero(merged[1:] != merged[:-1]) + 1]
        values = merged[starts]
        ends = np.r_[starts[1:], merged.size]
        counts = ends - starts
        order = np.lexsort((values, -counts))
        return values[order].astype(np.intp, copy=False)

    def candidates(
        self,
        query_sign_bits: np.ndarray,
        query_row_bits: np.ndarray,
        query_sem: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.table_positions is None or self.global_indices is None or self.relative_indices is None:
            raise RuntimeError("BinarySemanticLSH.build() must be called before candidates().")
        chunks = []
        g_pos = self.table_positions[self.global_indices]
        r_pos = self.table_positions[self.relative_indices]
        g_bits = query_sign_bits[g_pos]
        r_bits = query_row_bits[r_pos]
        g_exact = self._compact_keys(g_bits)
        r_exact = self._compact_keys(r_bits)
        exact_keys = np.empty(self.tables, dtype=np.uint64)
        exact_keys[self.global_indices] = g_exact
        exact_keys[self.relative_indices] = r_exact
        if self.probe_radius >= 1 and self.key_bits:
            if self.probe_strategy == "uncertainty_guided" and query_sem is not None:
                query_mean = float(query_sem.mean())
                g_unc_matrix = np.abs(query_sem)[g_pos]
                r_unc_matrix = np.abs(query_sem - query_mean)[r_pos]
                m = min(max(self.limited_probe_m, 0), self.key_bits)
                g_offsets = np.argsort(g_unc_matrix, axis=1, kind="stable")[:, :m]
                r_offsets = np.argsort(r_unc_matrix, axis=1, kind="stable")[:, :m]
            elif self.limited_probe_m > 0:
                m = min(self.limited_probe_m, self.key_bits)
                offsets = np.arange(m, dtype=np.intp)
                g_offsets = np.tile(offsets, (self.global_indices.size, 1))
                r_offsets = np.tile(offsets, (self.relative_indices.size, 1))
            else:
                m = self.key_bits
                offsets = np.arange(m, dtype=np.intp)
                g_offsets = np.tile(offsets, (self.global_indices.size, 1))
                r_offsets = np.tile(offsets, (self.relative_indices.size, 1))
            keys_by_table = np.empty((self.tables, m + 1), dtype=np.uint64)
            keys_by_table[:, 0] = exact_keys
            if m:
                keys_by_table[self.global_indices, 1:] = g_exact[:, None] ^ self.xor_masks[g_offsets]
                keys_by_table[self.relative_indices, 1:] = r_exact[:, None] ^ self.xor_masks[r_offsets]
        else:
            keys_by_table = exact_keys.reshape(self.tables, 1)
        for table_no, table_keys in enumerate(keys_by_table):
            buckets = self.buckets[table_no]
            for key in table_keys:
                chunk = buckets.get(int(key))
                if chunk is not None and chunk.size:
                    chunks.append(chunk)
        if self.dedup_strategy == "sort_unique":
            return self._sort_unique_ranked(chunks)
        if not chunks:
            return np.empty(0, dtype=np.intp)
        merged = np.concatenate(chunks)
        counts = np.bincount(merged, minlength=self.sign_bits.shape[0])
        values = np.flatnonzero(counts)
        order = np.lexsort((values, -counts[values]))
        return values[order]

    def size_bytes(self) -> int:
        return int(
            sum(bit_index.nbytes for bit_index, _ in self.specs)
            + self.weights.nbytes
            + self.xor_masks.nbytes
            + sum(value.nbytes for table in self.buckets for value in table.values())
        )


def rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024**2))
    except ImportError:
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def pack_uint64(bits: np.ndarray) -> np.ndarray:
    packed8 = np.packbits(np.ascontiguousarray(bits, dtype=np.uint8), axis=1, bitorder="little")
    return np.ascontiguousarray(packed8).view(np.uint64)


def pack_count_presence_uint64(counts: np.ndarray) -> np.ndarray:
    # packbits treats every nonzero count as present. Pad the packed bytes,
    # avoiding full-size boolean, padded boolean, and converted uint8 arrays.
    packed8 = np.packbits(counts, axis=1, bitorder="little")
    pad = (-packed8.shape[1]) % 8
    if pad:
        packed8 = np.pad(packed8, ((0, 0), (0, pad)), constant_values=0)
    return np.ascontiguousarray(packed8).view(np.uint64)


def hamming_distance(packed_db: np.ndarray, packed_query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(packed_db[candidates], packed_query.reshape(1, -1))
    return POPCOUNT8[xor.view(np.uint8).reshape(candidates.size, -1)].sum(axis=1).astype(np.float32)


def top_order(scores: np.ndarray, candidates: np.ndarray, limit: int) -> np.ndarray:
    if scores.size <= limit:
        return np.lexsort((candidates, -scores))
    keep = np.argpartition(scores, -limit)[-limit:]
    local = np.lexsort((candidates[keep], -scores[keep]))
    return keep[local]


def relation_text_terms(value: str) -> list[str]:
    text = str(value).strip("<>")
    text = re.sub(r"https?://", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.lower()
    raw = re.split(r"[^a-z0-9]+", text)
    stop = {"www", "org", "com", "http", "https", "dbpedia", "ontology", "freebase"}
    terms = [part for part in raw if len(part) >= 2 and part not in stop]
    if not terms:
        return ["relation"]
    expanded: list[str] = []
    for term in terms:
        expanded.append("w:" + term)
        padded = f"^{term}$"
        for n in (3, 4, 5):
            if len(padded) >= n:
                expanded.extend(f"c:{padded[i:i+n]}" for i in range(len(padded) - n + 1))
    return expanded


def hashed_text_embedding(text: str, dim: int, seed: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    for term in relation_text_terms(text):
        digest = hashlib.blake2b(f"rel-align:{seed}:{term}".encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little", signed=False)
        vector[value % dim] += 1.0 if ((value >> 16) & 1) else -1.0
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


def unique_metrics(task: RetrievalTaskV2, rankings: dict[str, list[str]]) -> dict[str, float]:
    ranks = []
    for qid in task.query_ids:
        positive = task.positives[qid][0]
        ranking = rankings.get(qid, [])
        try:
            rank = ranking.index(positive) + 1
        except ValueError:
            rank = len(task.database_ids) + 1
        ranks.append(rank)
    arr = np.asarray(ranks, dtype=np.float64)
    out = {f"hits_at_{k}": float(np.mean(arr <= k)) for k in (1, 5, 10, 20, 50)}
    out["mrr"] = float(np.mean(1.0 / arr))
    return out


class ASMSSREngine:
    def __init__(
        self,
        *,
        schema_root: Path = ROOT / "data" / "processed_schema_v2",
        cache_root: Path = ROOT / "cache",
        dataset: str = "fb15k_db15k",
        language_pair: str | None = None,
        seed: int = 3407,
        max_queries: int | None = None,
        max_database: int | None = None,
        clip_timing_policy: str = "fastest",
        clip_timing_manifest: Path = DEFAULT_CLIP_TIMINGS,
    ) -> None:
        if clip_timing_policy not in ("fastest", "algorithm-only"):
            raise ValueError("clip_timing_policy must be fastest or algorithm-only")
        self.clip_timing_policy = clip_timing_policy
        self.clip_timing_manifest = Path(clip_timing_manifest)
        self.cache_root = Path(cache_root)
        self.schema_root = schema_root
        self.dataset = dataset
        self.language_pair = language_pair
        self.seed = seed
        self.task = load_task(schema_root, dataset, "test", language_pair)
        if max_queries:
            kept = self.task.query_ids[:max_queries]
            self.task = replace(self.task, query_ids=kept, positives={qid: self.task.positives[qid] for qid in kept})
        if max_database:
            required = {gid for qid in self.task.query_ids for gid in self.task.positives.get(qid, [])}
            database = sorted(required) + [gid for gid in self.task.database_ids if gid not in required][
                : max(max_database - len(required), 0)
            ]
            self.task = replace(self.task, database_ids=database)
        self.corpus = None
        self.ids = list(self.task.database_ids)
        self.id_to_pos = {gid: idx for idx, gid in enumerate(self.ids)}
        self.stats: dict[str, float | str | bool | dict] = {}

    def _encode_binary(self, db_sem: np.ndarray, query_sem: np.ndarray) -> None:
        self.db_sign_bits = db_sem >= 0.0
        self.db_row_bits = db_sem >= db_sem.mean(axis=1, keepdims=True)
        self.query_sign_bits = query_sem >= 0.0
        self.query_row_bits = query_sem >= query_sem.mean(axis=1, keepdims=True)
        self.db_sign64 = pack_uint64(self.db_sign_bits)
        self.db_row64 = pack_uint64(self.db_row_bits)
        self.query_sign64 = pack_uint64(self.query_sign_bits)
        self.query_row64 = pack_uint64(self.query_row_bits)

    def _prepare_accelerator(self) -> float:
        return 0.0

    def _build_lsh(self) -> None:
        self.lsh.build()

    def _binary_distances(self, query_no: int, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (hamming_distance(self.db_sign64, self.query_sign64[query_no], candidates),
                hamming_distance(self.db_row64, self.query_row64[query_no], candidates))

    def _count_similarity(self, query_no: int, head: np.ndarray, timings: dict) -> np.ndarray:
        started = time.perf_counter()
        q_counts = self.aligned_fast_query_counts[query_no].astype(np.float32, copy=False)
        cand_counts = self.aligned_fast_db_counts[head].astype(np.float32, copy=False)
        timings["aligned_profile_gather_time_ms"] += (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        numerator = np.minimum(cand_counts, q_counts.reshape(1, -1)).sum(axis=1)
        denominator = np.maximum(cand_counts, q_counts.reshape(1, -1)).sum(axis=1)
        result = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float32),
                           where=denominator > 0).astype(np.float32, copy=False)
        timings["aligned_similarity_time_ms"] += (time.perf_counter() - started) * 1000.0
        return result

    def _semantic_cache_path(self):
        suffix = f"_{self.task.metadata.get('subset')}" if self.task.metadata.get("subset") else ""
        return self.cache_root / "mssr_semantic" / f"{self.task.dataset}{suffix}.npz"

    def _load_semantic_arrays(self):
        path = self._semantic_cache_path()
        if not path.exists():
            raise FileNotFoundError(path)
        graph_ids, embeddings, self.semantic_cache_io_backend = load_semantic_arrays(path)
        return graph_ids, embeddings

    def _load_semantic_cache(self) -> dict[str, np.ndarray]:
        graph_ids, embeddings = self._load_semantic_arrays()
        return {str(gid): row.astype(np.float32, copy=False) for gid, row in zip(graph_ids, embeddings)}

    def _prepare_semantics(self):
        sem_lookup = self._load_semantic_cache()
        db_sem = np.stack([sem_lookup[gid] for gid in self.ids]).astype(np.float32, copy=False)
        query_sem = np.stack([sem_lookup[gid] for gid in self.task.query_ids]).astype(np.float32, copy=False)
        db_sem /= np.maximum(np.linalg.norm(db_sem, axis=1, keepdims=True), 1e-8)
        query_sem /= np.maximum(np.linalg.norm(query_sem, axis=1, keepdims=True), 1e-8)
        return db_sem, query_sem

    def _relation_vocab_by_side(self) -> tuple[dict[int, str], dict[int, str]]:
        query_vocab: dict[int, str] = {}
        db_vocab: dict[int, str] = {}
        assert self.corpus is not None

        def add_graph(gid: str, target: dict[int, str]) -> None:
            graph = self.corpus[gid]
            for rel_id, rel_name in zip(graph.edge_relation_ids, graph.edge_relations):
                rel_id = int(rel_id)
                if rel_id not in target:
                    target[rel_id] = str(rel_name)

        for gid in self.task.query_ids:
            add_graph(gid, query_vocab)
        for gid in self.ids:
            add_graph(gid, db_vocab)
        return query_vocab, db_vocab

    def _task_cache_label(self) -> str:
        subset = self.task.metadata.get("subset") if self.task.metadata else None
        return f"{self.task.dataset}_{subset}" if subset else self.task.dataset

    def _aligned_fast_cache_path(self, config: ASMSSRConfig) -> Path:
        suffix = f"q{len(self.task.query_ids)}_d{len(self.ids)}"
        threshold = int(round(config.relation_align_threshold * 1000))
        return (
            self.cache_root
            / "mssr_v3_relation_alignment"
            / (
                f"{self._task_cache_label()}_{suffix}_seed{config.seed}_dim{config.relation_embedding_dim}"
                f"_thr{threshold}_directed_fast.npz"
            )
        )

    def _encode_relation_counts(self, graph_ids, relation_positions, num_db_relations, dim):
        assert self.corpus is not None
        counts = np.zeros((len(graph_ids), dim), dtype=np.uint8)
        for row, gid in enumerate(graph_ids):
            graph = self.corpus[gid]
            try:
                center = int(graph.meta.get("center_node_index", 0))
            except (TypeError, ValueError):
                center = 0
            sparse_counts = {}
            for src, dst, rel_id in zip(graph.edge_src, graph.edge_dst, graph.edge_relation_ids):
                if src == center:
                    offset = num_db_relations
                elif dst == center:
                    offset = 0
                else:
                    continue
                position = relation_positions.get(int(rel_id))
                if position is None:
                    continue
                index = offset + position
                sparse_counts[index] = sparse_counts.get(index, 0) + 1
            for index, value in sparse_counts.items():
                counts[row, index] = min(value, 255)
        return counts

    def _build_relation_aligned_fast_profiles(self, config: ASMSSRConfig) -> tuple[dict, dict]:
        cache_path = self._aligned_fast_cache_path(config)
        load_started = time.perf_counter()
        if cache_path.exists():
            payload = np.load(cache_path, allow_pickle=True)
            diagnostics = json.loads(str(payload["diagnostics"].item()))
            diagnostics["relation_alignment_mode"] = "offline_top1_db_space"
            diagnostics["fast_relation_alignment_mode"] = "offline_top1_db_space"
            diagnostics["fast_cache_load_time_ms"] = (time.perf_counter() - load_started) * 1000.0
            diagnostics["fast_cache_build_time_ms"] = 0.0
            diagnostics["fast_cache_write_time_ms"] = 0.0
            return {
                "db_bits64": payload["db_bits64"],
                "query_bits64": payload["query_bits64"],
                "db_counts": payload["db_counts"],
                "query_counts": payload["query_counts"],
                "dim": int(payload["dim"]),
            }, diagnostics

        build_started = time.perf_counter()
        query_vocab, db_vocab = self._relation_vocab_by_side()
        query_ids = sorted(query_vocab)
        db_ids = sorted(db_vocab)
        query_index = {rel_id: idx for idx, rel_id in enumerate(query_ids)}
        db_index = {rel_id: idx for idx, rel_id in enumerate(db_ids)}
        align_started = time.perf_counter()
        query_emb = np.stack([hashed_text_embedding(query_vocab[rel_id], config.relation_embedding_dim, config.seed) for rel_id in query_ids]).astype(np.float32, copy=False)
        db_emb = np.stack([hashed_text_embedding(db_vocab[rel_id], config.relation_embedding_dim, config.seed) for rel_id in db_ids]).astype(np.float32, copy=False)
        relation_similarity = np.maximum(query_emb @ db_emb.T, 0.0).astype(np.float32, copy=False)
        align_ms = (time.perf_counter() - align_started) * 1000.0
        best_db = relation_similarity.argmax(axis=1).astype(np.int32) if relation_similarity.size else np.empty(0, dtype=np.int32)
        best_sim = relation_similarity.max(axis=1) if relation_similarity.size else np.empty(0, dtype=np.float32)
        valid_query = best_sim >= float(config.relation_align_threshold)
        dim = 2 * max(len(db_ids), 1)

        query_positions = {rel_id: int(best_db[position]) for rel_id, position in query_index.items()
                           if bool(valid_query[position])}
        counts_started = time.perf_counter()
        db_counts = self._encode_relation_counts(self.ids, db_index, len(db_ids), dim)
        query_counts = self._encode_relation_counts(self.task.query_ids, query_positions, len(db_ids), dim)
        counts_ms = (time.perf_counter() - counts_started) * 1000.0
        presence_started = time.perf_counter()
        db_bits64 = pack_count_presence_uint64(db_counts)
        query_bits64 = pack_count_presence_uint64(query_counts)
        presence_ms = (time.perf_counter() - presence_started) * 1000.0
        diagnostics = {
            "fast_relation_alignment_mode": "offline_top1_db_space",
            "relation_alignment_mode": "offline_top1_db_space",
            "relation_alignment_build_time_ms": align_ms,
            "relation_count_encoding_time_ms": counts_ms,
            "relation_presence_packing_time_ms": presence_ms,
            "relation_similarity_matrix_size_mb": relation_similarity.nbytes / (1024**2),
            "avg_best_fb_to_db_relation_similarity": float(best_sim.mean()) if best_sim.size else 0.0,
            "median_best_fb_to_db_relation_similarity": float(np.median(best_sim)) if best_sim.size else 0.0,
            "p90_best_fb_to_db_relation_similarity": float(np.percentile(best_sim, 90)) if best_sim.size else 0.0,
            "avg_best_db_to_fb_relation_similarity": float(relation_similarity.max(axis=0).mean()) if relation_similarity.size else 0.0,
            "cache_build_time_ms": (time.perf_counter() - build_started) * 1000.0,
            "cache_load_time_ms": 0.0,
            "relation_align_threshold": float(config.relation_align_threshold),
            "mapped_fb_relation_count": int(valid_query.sum()),
            "unknown_fb_relation_count": int(valid_query.size - valid_query.sum()),
            "directed_relation_profile_dim": int(dim),
            "aligned_relation_bitset_size_mb": float((db_bits64.nbytes + query_bits64.nbytes) / (1024**2)),
            "aligned_relation_count_size_mb": float((db_counts.nbytes + query_counts.nbytes) / (1024**2)),
            "fast_cache_build_time_ms": (time.perf_counter() - build_started) * 1000.0,
            "fast_cache_load_time_ms": 0.0,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        write_started = time.perf_counter()
        np.savez_compressed(
            cache_path,
            db_bits64=db_bits64,
            query_bits64=query_bits64,
            db_counts=db_counts,
            query_counts=query_counts,
            dim=np.asarray(dim, dtype=np.int32),
            diagnostics=np.asarray(json.dumps(diagnostics), dtype=object),
        )
        diagnostics["fast_cache_write_time_ms"] = (time.perf_counter() - write_started) * 1000.0
        return {
            "db_bits64": db_bits64,
            "query_bits64": query_bits64,
            "db_counts": db_counts,
            "query_counts": query_counts,
            "dim": dim,
        }, diagnostics

    def prepare(
        self,
        config: ASMSSRConfig,
        *,
        include_mssr: bool = False,
        include_simplicial: bool = False,
        include_simplicial_exact: bool = False,
        include_relation_aligned: bool = True,
        include_relation_aligned_fast: bool = True,
    ) -> None:
        del include_mssr, include_simplicial, include_simplicial_exact, include_relation_aligned
        # Validate recorded timing outside the measured algorithm stages.
        clip_timing = None
        if getattr(self, "clip_timing_policy", "algorithm-only") == "fastest":
            clip_timing = load_fastest_clip_timing(
                self.task, self._semantic_cache_path(), language_pair=self.language_pair,
                manifest_path=self.clip_timing_manifest)
        before = rss_mb()
        started = time.perf_counter()
        if self.corpus is None:
            self.corpus = load_task_corpus(self.task)

        semantic_started = time.perf_counter()
        db_sem, query_sem = self._prepare_semantics()
        semantic_ms = (time.perf_counter() - semantic_started) * 1000.0

        binary_started = time.perf_counter()
        self._encode_binary(db_sem, query_sem)
        binary_ms = (time.perf_counter() - binary_started) * 1000.0

        aligned_started = time.perf_counter()
        if not include_relation_aligned_fast:
            raise ValueError("mssr_v5 requires include_relation_aligned_fast=True")
        fast_profiles, relation_diagnostics = self._build_relation_aligned_fast_profiles(config)
        self.aligned_fast_db_bits64 = fast_profiles["db_bits64"]
        self.aligned_fast_query_bits64 = fast_profiles["query_bits64"]
        self.aligned_fast_db_counts = fast_profiles["db_counts"]
        self.aligned_fast_query_counts = fast_profiles["query_counts"]
        aligned_total_ms = (time.perf_counter() - aligned_started) * 1000.0
        cache_write_ms = float(relation_diagnostics.get("fast_cache_write_time_ms", 0.0))
        # Relation cache persistence is an auxiliary artifact. Other in-memory
        # index methods do not persist one, so expose this cost separately.
        aligned_ms = max(0.0, aligned_total_ms - cache_write_ms)
        print(
            "[relation_alignment_fast] "
            f"mode={relation_diagnostics.get('fast_relation_alignment_mode')} "
            f"thr={relation_diagnostics.get('relation_align_threshold')} "
            f"mapped_fb={relation_diagnostics.get('mapped_fb_relation_count')} "
            f"unknown_fb={relation_diagnostics.get('unknown_fb_relation_count')} "
            f"bitset_mb={relation_diagnostics.get('aligned_relation_bitset_size_mb'):.3f} "
            f"count_mb={relation_diagnostics.get('aligned_relation_count_size_mb'):.3f} "
            f"cache_build_ms={relation_diagnostics.get('fast_cache_build_time_ms'):.3f} "
            f"cache_load_ms={relation_diagnostics.get('fast_cache_load_time_ms'):.3f}",
            flush=True,
        )

        self.db_sem = np.ascontiguousarray(db_sem)
        self.query_sem = np.ascontiguousarray(query_sem)
        lsh_started = time.perf_counter()
        self.lsh = BinarySemanticLSH(
            self.db_sign_bits,
            self.db_row_bits,
            seed=config.seed,
            tables=config.lsh_tables,
            key_bits=config.lsh_key_bits,
            bucket_size=config.lsh_bucket_size,
            probe_radius=config.lsh_probe_radius,
            limited_probe_m=config.lsh_limited_probe_m,
            probe_strategy=config.lsh_probe_strategy,
            dedup_strategy=config.lsh_dedup_strategy,
        )
        self._build_lsh()
        lsh_ms = (time.perf_counter() - lsh_started) * 1000.0
        accelerator_upload_ms = self._prepare_accelerator()
        encode_ms = semantic_ms + binary_ms + aligned_ms + accelerator_upload_ms
        index_mb = self.lsh.size_bytes() / (1024**2)
        sem_mb = (self.db_sem.nbytes + self.query_sem.nbytes + self.db_sign64.nbytes + self.db_row64.nbytes) / (1024**2)
        self.stats = {
            "training_required": False,
            "training_time_ms": 0.0,
            "semantic_cache_precomputed": True,
            "semantic_cache_io_backend": getattr(self, "semantic_cache_io_backend", "injected_or_custom"),
            "semantic_cache_generation_time": "unavailable",
            "semantic_encoding_time_ms": semantic_ms,
            "binary_encoding_time_ms": binary_ms,
            "accelerator_upload_time_ms": accelerator_upload_ms,
            "relation_alignment_encoding_time_ms": aligned_ms,
            "relation_alignment_build_time_ms": float(relation_diagnostics.get("relation_alignment_build_time_ms", 0.0)),
            "relation_similarity_matrix_size_mb": float(relation_diagnostics.get("relation_similarity_matrix_size_mb", 0.0)),
            "cache_build_time_ms": float(relation_diagnostics.get("cache_build_time_ms", 0.0)),
            "cache_load_time_ms": float(relation_diagnostics.get("cache_load_time_ms", 0.0)),
            "fast_cache_build_time_ms": float(relation_diagnostics.get("fast_cache_build_time_ms", 0.0)),
            "fast_cache_load_time_ms": float(relation_diagnostics.get("fast_cache_load_time_ms", 0.0)),
            "relation_profile_cache_write_time_ms": cache_write_ms,
            "encoding_time_ms": encode_ms,
            "lsh_build_time_ms": lsh_ms,
            "index_build_time_ms": lsh_ms,
            "total_offline_time_ms": encode_ms + lsh_ms,
            "total_offline_including_cache_write_time_ms": encode_ms + lsh_ms + cache_write_ms,
            "offline_timing_scope": "semantic_preparation_plus_query_database_encoding_plus_index_and_upload_excluding_auxiliary_cache_write",
            "index_size_mb": index_mb,
            "semantic_embedding_size_mb": sem_mb,
            "aligned_relation_bitset_size_mb": float(relation_diagnostics.get("aligned_relation_bitset_size_mb", 0.0)),
            "aligned_relation_count_size_mb": float(relation_diagnostics.get("aligned_relation_count_size_mb", 0.0)),
            "peak_memory_mb": max(rss_mb() - before, 0.0),
            "peak_cpu_memory_mb": rss_mb(),
            "peak_gpu_memory_mb": 0.0,
            "prepare_wall_time_ms": (time.perf_counter() - started) * 1000.0,
            "relation_alignment_mode": relation_diagnostics.get("relation_alignment_mode", config.relation_alignment_mode),
            "relation_diagnostics": relation_diagnostics,
        }

        account_clip_time(self.stats, clip_timing)

    def _lsh_candidates(self, query_no: int, qid: str, limit: int, cfg: ASMSSRConfig) -> np.ndarray:
        query_sign_bits = self.query_sign_bits[query_no]
        query_row_bits = self.query_row_bits[query_no]
        query_sem = self.query_sem[query_no]
        if cfg.lsh_query_flip_bits > 0:
            stable = sum((i + 1) * byte for i, byte in enumerate(qid.encode("utf-8")))
            rng = np.random.default_rng((cfg.seed * 1000003 + stable + 99173) & 0xFFFFFFFF)
            flip_count = min(int(cfg.lsh_query_flip_bits), query_sign_bits.size)
            query_sign_bits = query_sign_bits.copy()
            query_row_bits = query_row_bits.copy()
            sign_flip = rng.choice(query_sign_bits.size, size=flip_count, replace=False)
            row_flip = rng.choice(query_row_bits.size, size=flip_count, replace=False)
            query_sign_bits[sign_flip] = ~query_sign_bits[sign_flip]
            query_row_bits[row_flip] = ~query_row_bits[row_flip]
        candidates = self.lsh.candidates(query_sign_bits, query_row_bits, query_sem)
        own = self.id_to_pos.get(qid)
        if own is not None:
            candidates = candidates[candidates != own]
        if cfg.candidate_backfill and candidates.size < limit:
            stable = sum((i + 1) * byte for i, byte in enumerate(qid.encode("utf-8")))
            rng = np.random.default_rng((cfg.seed + stable) & 0xFFFFFFFF)
            filler = rng.integers(0, len(self.ids), size=(limit - candidates.size) * 4 + 16, dtype=np.intp)
            if own is not None:
                filler = filler[filler != own]
            seen = np.zeros(len(self.ids), dtype=bool)
            seen[candidates] = True
            keep: list[int] = []
            for value in filler:
                item = int(value)
                if not seen[item]:
                    keep.append(item)
                    seen[item] = True
                    if candidates.size + len(keep) >= limit:
                        break
            if keep:
                candidates = np.concatenate((candidates, np.asarray(keep, dtype=np.intp)))
        return candidates[: min(limit, candidates.size)]

    def run(self, config: ASMSSRConfig) -> dict:
        if config.method != "semantic_binary_lsh_simplicial_aligned_fast":
            raise ValueError("mssr_v5 only supports semantic_binary_lsh_simplicial_aligned_fast")
        timings = {name: 0.0 for name in (
            "binary_time_ms",
            "lsh_lookup_time_ms",
            "candidate_gather_time_ms",
            "simplicial_rerank_time_ms",
            "aligned_profile_gather_time_ms",
            "aligned_similarity_time_ms",
            "residual_rerank_time_ms",
        )}
        rankings: dict[str, list[str]] = {}
        latencies = []
        c1_sizes = []
        c2_sizes = []
        visited_sizes = []
        difficulty = {"easy": 0, "hard": 0}
        recall_hits = {128: 0, 256: 0, 512: 0, 1024: 0, 2048: 0, 4096: 0}
        max_c = max(max(recall_hits), config.semantic_lsh_candidates)

        for query_no, qid in enumerate(self.task.query_ids):
            q_start = time.perf_counter()
            gt = self.task.positives[qid][0]
            gt_pos = self.id_to_pos.get(gt)
            q_sign64 = self.query_sign64[query_no]
            q_row64 = self.query_row64[query_no]

            section = time.perf_counter()
            raw = self._lsh_candidates(query_no, qid, max_c, config)
            timings["lsh_lookup_time_ms"] += (time.perf_counter() - section) * 1000.0
            c1 = raw[: min(config.semantic_lsh_candidates, raw.size)]
            c1_sizes.append(int(c1.size))
            if gt_pos is not None:
                for limit in recall_hits:
                    if gt_pos in raw[: min(limit, raw.size)]:
                        recall_hits[limit] += 1

            section = time.perf_counter()
            sign_dist, row_dist = self._binary_distances(query_no, c1)
            semantic_distance = config.binary_sign_weight * sign_dist + config.binary_rowmean_weight * row_dist
            if config.semantic_score_noise_std > 0.0:
                stable = sum((i + 1) * byte for i, byte in enumerate(qid.encode("utf-8")))
                rng = np.random.default_rng((config.seed * 1000003 + stable + 55537) & 0xFFFFFFFF)
                semantic_distance = semantic_distance + rng.normal(
                    0.0,
                    float(config.semantic_score_noise_std),
                    size=semantic_distance.size,
                ).astype(np.float32)
            binary_scores = -semantic_distance
            timings["candidate_gather_time_ms"] += (time.perf_counter() - section) * 1000.0

            section = time.perf_counter()
            order = top_order(binary_scores, c1, c1.size)
            semantic_ranked = c1[order]
            semantic_dist = -binary_scores[order]
            top_r = min(config.simplicial_rerank_topr, semantic_ranked.size)
            use_simplicial = top_r > 1
            if config.use_adaptive_simplicial and semantic_dist.size >= 2:
                margin = float(semantic_dist[1] - semantic_dist[0])
                use_simplicial = margin <= float(config.adaptive_margin_bits)
                difficulty["hard" if use_simplicial else "easy"] += 1
            else:
                difficulty["hard"] += 1

            if use_simplicial:
                head = semantic_ranked[:top_r]
                head_sem_dist = semantic_dist[:top_r].astype(np.float32, copy=False)
                sem_norm = head_sem_dist / (512.0 * (config.binary_sign_weight + config.binary_rowmean_weight))
                gather_started = time.perf_counter()
                if config.fast_profile == "count":
                    simp_score = self._count_similarity(query_no, head, timings)
                else:
                    q_bits = self.aligned_fast_query_bits64[query_no]
                    cand_bits = self.aligned_fast_db_bits64[head]
                    timings["aligned_profile_gather_time_ms"] += (time.perf_counter() - gather_started) * 1000.0
                    sim_started = time.perf_counter()
                    inter = np.bitwise_and(cand_bits, q_bits.reshape(1, -1))
                    union = np.bitwise_or(cand_bits, q_bits.reshape(1, -1))
                    inter_count = POPCOUNT8[inter.view(np.uint8).reshape(head.size, -1)].sum(axis=1).astype(np.float32)
                    union_count = POPCOUNT8[union.view(np.uint8).reshape(head.size, -1)].sum(axis=1).astype(np.float32)
                    simp_score = np.divide(
                        inter_count,
                        union_count,
                        out=np.zeros_like(inter_count, dtype=np.float32),
                        where=union_count > 0,
                    )
                    timings["aligned_similarity_time_ms"] += (time.perf_counter() - sim_started) * 1000.0
                rerank_started = time.perf_counter()
                semantic_similarity = 1.0 - sem_norm
                residual = semantic_similarity + config.simplicial_lambda * simp_score
                original_order = np.arange(top_r, dtype=np.intp)
                local = np.lexsort((original_order, -residual))
                final = np.concatenate((head[local], semantic_ranked[top_r:]))
                timings["residual_rerank_time_ms"] += (time.perf_counter() - rerank_started) * 1000.0
            else:
                final = semantic_ranked
            timings["simplicial_rerank_time_ms"] += (time.perf_counter() - section) * 1000.0
            c2_sizes.append(int(top_r))
            visited_sizes.append(int(c1.size))
            rankings[qid] = [self.ids[int(pos)] for pos in final]
            latencies.append((time.perf_counter() - q_start) * 1000.0)

        metrics = unique_metrics(self.task, rankings)
        query_count = max(len(self.task.query_ids), 1)
        latency = np.asarray(latencies, dtype=np.float64)
        avg_visited = float(np.mean(visited_sizes))
        report = {
            "method": config.method,
            "dataset": self.dataset,
            "language_pair": self.language_pair,
            "config": asdict(config),
            **metrics,
            **self.stats,
            "latency_mean_ms": float(latency.mean()),
            "latency_p50_ms": float(np.percentile(latency, 50)),
            "latency_p95_ms": float(np.percentile(latency, 95)),
            "latency_p99_ms": float(np.percentile(latency, 99)),
            "query_time_ms": float(latency.mean()),
            "latency_ms_per_query": float(latency.mean()),
            "qps": float(1000.0 / latency.mean()) if latency.mean() > 0 else 0.0,
            "visited_ratio": avg_visited / max(len(self.ids), 1),
            "avg_visited_entities": avg_visited,
            "avg_c1": float(np.mean(c1_sizes)),
            "avg_c2": float(np.mean(c2_sizes)),
            "easy_query_ratio": difficulty["easy"] / query_count,
            "hard_query_ratio": difficulty["hard"] / query_count,
            "num_queries": len(self.task.query_ids),
            "database_size": len(self.ids),
            "candidate_recall": {str(k): recall_hits[k] / query_count for k in sorted(recall_hits)},
        }
        for key, value in report["candidate_recall"].items():
            report[f"candidate_recall_{key}"] = value
        for key, value in timings.items():
            report[key] = float(value / query_count)
        return report
