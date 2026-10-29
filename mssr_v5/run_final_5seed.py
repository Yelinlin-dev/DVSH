"""Run MSSR v5 evaluation and five-seed summaries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mssr_v5.aligned_fast import ASMSSRConfig, ASMSSREngine, ROOT
from mssr_v5.offline_timing import DEFAULT_CLIP_TIMINGS, OFFLINE_METRICS, add_clip_timing_arguments


TASKS = [
    ("fb15k_db15k", "fb15k_db15k", None),
    ("fb15k_yago15k", "fb15k_yago15k", None),
    ("dbp15k_fr_en", "dbp15k", "fr_en"),
    ("dbp15k_ja_en", "dbp15k", "ja_en"),
    ("dbp15k_zh_en", "dbp15k", "zh_en"),
    ("dbp15k_all", "dbp15k_all", None),
]

METHODS = [
    "semantic_binary_lsh_simplicial_aligned_fast",
]

SEEDS = [3407, 3408, 3409, 3410, 3411]

METRICS = [
    ("hits_at_1", "Hits@1"),
    ("hits_at_5", "Hits@5"),
    ("hits_at_10", "Hits@10"),
    ("hits_at_20", "Hits@20"),
    ("hits_at_50", "Hits@50"),
    ("mrr", "MRR"),
    ("encoding_time_ms", "Encoding Time(ms)"),
    ("index_build_time_ms", "Index Build Time(ms)"),
    ("index_size_mb", "Index Size(mb)"),
    ("peak_memory_mb", "Peak Memory(mb)"),
    ("latency_mean_ms", "Latency (ms/q)"),
    ("qps", "QPS"),
    ("training_time_ms", "Training Time"),
    ("visited_ratio", "Visited Ratio"),
] + OFFLINE_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_clip_timing_arguments(parser)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "mssr_logs_v5" / time.strftime("final_5seed_aligned_fast_%Y%m%d_%H%M%S"))
    parser.add_argument("--schema-root", type=Path, default=ROOT / "data" / "processed_schema_v2")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    parser.add_argument("--device", default="cpu", help="cpu or a CUDA device, e.g. cuda:0")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads for the CUDA backend")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--datasets", nargs="*", default=[label for label, _, _ in TASKS])
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--max-database", type=int)
    return parser.parse_args()


def configs(seed: int, dataset_label: str | None = None) -> list[ASMSSRConfig]:
    base = dict(
        seed=seed,
        dtype="float32",
        semantic_lsh_candidates=1024,
        fixed_candidates=1024,
        adaptive_budget=False,
        lsh_key_bits=10,
        lsh_tables=64,
        lsh_limited_probe_m=5,
        lsh_probe_strategy="uncertainty_guided",
        lsh_dedup_strategy="sort_unique",
        lsh_bucket_size=0,
    )
    return [
        ASMSSRConfig(
            method="semantic_binary_lsh_simplicial_aligned_fast",
            simplicial_rerank_topr=128,
            simplicial_lambda=0.2,
            exact_relation_weight=1.0,
            exact_pair_weight=0.0,
            relation_alignment_mode="offline_top1_db_space",
            relation_align_threshold=0.6,
            fast_profile="count",
            use_adaptive_simplicial=True,
            adaptive_margin_bits=8,
            **base,
        ),
    ]


def cache_path_for(engine: ASMSSREngine) -> Path:
    subset = engine.task.metadata.get("subset") if engine.task.metadata else None
    suffix = f"_{subset}" if subset else ""
    return engine.cache_root / "mssr_semantic" / f"{engine.task.dataset}{suffix}.npz"


def paper_row(row: dict[str, Any], dataset_label: str, seed: int) -> dict[str, Any]:
    out = {
        "dataset_label": dataset_label,
        "dataset": row.get("dataset", ""),
        "language_pair": row.get("language_pair") or "",
        "seed": seed,
        "method": row["method"],
        "num_queries": row.get("num_queries", 0),
        "database_size": row.get("database_size", 0),
    }
    for key, _ in METRICS:
        out[key] = float(row.get(key, 0.0) or 0.0)
    out.update({k: v for k, v in row.items() if k.startswith("clip_timing_") or k == "semantic_model_generation_included"})
    return out


def run_one(dataset_label: str, dataset: str, language_pair: str | None, seed: int, output_dir: Path, max_queries: int | None, max_database: int | None, schema_root: Path, cache_root: Path, device: str = "cpu", threads: int = 4, clip_timing_policy: str = "fastest", clip_timing_manifest: Path = DEFAULT_CLIP_TIMINGS) -> list[dict[str, Any]]:
    engine_class = ASMSSREngine
    backend_args = {}
    if device != "cpu":
        import torch
        from mssr_v5.cuda_backend import CUDAASMSSREngine
        if not device.startswith("cuda"):
            raise ValueError("device must be cpu or cuda:N")
        torch.set_num_threads(threads)
        engine_class = CUDAASMSSREngine
        backend_args = dict(device=device, profile_cache_dir=output_dir / "relation_profiles" / dataset_label / f"seed_{seed}")
    print(f"[start] {dataset_label} seed={seed} device={device}", flush=True)
    engine = engine_class(
        **backend_args,
        clip_timing_policy=clip_timing_policy,
        clip_timing_manifest=clip_timing_manifest,
        schema_root=schema_root,
        cache_root=cache_root,
        dataset=dataset,
        language_pair=language_pair,
        seed=seed,
        max_queries=max_queries,
        max_database=max_database,
    )
    cache_path = cache_path_for(engine)
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    cfgs = configs(seed, dataset_label)
    if device != "cpu" and engine._aligned_fast_cache_path(cfgs[0]).exists():
        raise RuntimeError("Use a new output directory for fresh CUDA relation-profile measurements")
    engine.prepare(
        cfgs[0],
        include_mssr=False,
        include_simplicial=False,
        include_simplicial_exact=False,
        include_relation_aligned=True,
        include_relation_aligned_fast=True,
    )
    print(f"[prepared] {dataset_label} queries={len(engine.task.query_ids)} database={len(engine.ids)}", flush=True)
    reports = [engine.run(cfg) for cfg in cfgs]
    rows = [paper_row(report, dataset_label, seed) for report in reports]
    run_dir = output_dir / "raw" / dataset_label / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reports.json").write_text(json.dumps(reports, indent=2))
    with (run_dir / "paper_rows.json").open("w") as handle:
        json.dump(rows, handle, indent=2)
    return rows


def add_dbp15k_subsets_average(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep language-pair averages separate from actual merged-corpus runs."""
    out = list(rows)
    if any(r.get("clip_timing_policy") == "fastest" for r in rows):
        return out  # Subset averages are not a measured merged-corpus CLIP run.
    for seed in sorted({int(row["seed"]) for row in rows}):
        for method in METHODS:
            group = [
                row for row in rows
                if int(row["seed"]) == seed
                and row["method"] == method
                and row["dataset_label"] in {"dbp15k_fr_en", "dbp15k_ja_en", "dbp15k_zh_en"}
            ]
            if len(group) != 3:
                continue
            total_q = sum(float(row["num_queries"]) for row in group)
            aggregate: dict[str, Any] = {
                "dataset_label": "dbp15k_subsets_average",
                "dataset": "dbp15k",
                "language_pair": "all",
                "seed": seed,
                "method": method,
                "num_queries": total_q,
                "database_size": sum(float(row["database_size"]) for row in group) / len(group),
            }
            for key, _ in METRICS:
                aggregate[key] = sum(float(row[key]) * float(row["num_queries"]) for row in group) / total_q
            out.append(aggregate)
    return out


def write_raw_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["dataset_label", "dataset", "language_pair", "seed", "method", "num_queries", "database_size"] + [key for key, _ in METRICS]
    with (output_dir / "all_seed_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    with (output_dir / "all_seed_rows.json").open("w") as handle:
        json.dump(rows, handle, indent=2)


def write_method_tables(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        datasets = [label for label, _, _ in TASKS] + ["dbp15k_subsets_average"]
        columns = ["Dataset", "Seeds"]
        for _, label in METRICS:
            columns.extend([f"{label} Mean", f"{label} Std"])
        table_rows = []
        for dataset_label in datasets:
            group = [row for row in method_rows if row["dataset_label"] == dataset_label]
            if not group:
                continue
            summary: dict[str, Any] = {"Dataset": dataset_label, "Seeds": " ".join(str(int(row["seed"])) for row in sorted(group, key=lambda item: int(item["seed"])))}
            for key, label in METRICS:
                values = np.asarray([float(row[key]) for row in group], dtype=np.float64)
                summary[f"{label} Mean"] = float(values.mean())
                summary[f"{label} Std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
            table_rows.append(summary)
        path = output_dir / f"{method}_5seed_summary.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in table_rows:
                writer.writerow(row)


def write_method_pm_tables(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        datasets = [label for label, _, _ in TASKS] + ["dbp15k_subsets_average"]
        columns = ["Dataset", "Seeds"] + [label for _, label in METRICS]
        table_rows = []
        for dataset_label in datasets:
            group = [row for row in method_rows if row["dataset_label"] == dataset_label]
            if not group:
                continue
            group = sorted(group, key=lambda item: int(item["seed"]))
            summary: dict[str, Any] = {
                "Dataset": dataset_label,
                "Seeds": " ".join(str(int(row["seed"])) for row in group),
            }
            for key, label in METRICS:
                values = np.asarray([float(row[key]) for row in group], dtype=np.float64)
                mean = float(values.mean())
                std = float(values.std(ddof=1)) if values.size > 1 else 0.0
                summary[label] = f"{mean:.4f}±{std:.4f}"
            table_rows.append(summary)
        path = output_dir / f"{method}_5seed_mean_std_4dp.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in table_rows:
                writer.writerow(row)


def main() -> None:
    args = parse_args()
    unknown = set(args.datasets) - {task[0] for task in TASKS}
    if unknown or not args.datasets or not args.seeds:
        raise SystemExit(f"Provide valid datasets and at least one seed. Unknown datasets: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = {task[0]: task for task in TASKS if task[0] in set(args.datasets)}
    jobs = [(label, dataset, language_pair, seed) for label, dataset, language_pair in selected.values() for seed in args.seeds]
    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(int(args.workers), 1)) as executor:
        futures = {
            executor.submit(run_one, label, dataset, language_pair, seed, args.output_dir, args.max_queries, args.max_database, args.schema_root, args.cache_root, args.device, args.threads, args.clip_timing_policy, args.clip_timing_manifest): (label, seed)
            for label, dataset, language_pair, seed in jobs
        }
        for future in as_completed(futures):
            label, seed = futures[future]
            try:
                rows = future.result()
                all_rows.extend(rows)
                print(f"[done] {label} seed={seed}", flush=True)
            except Exception as exc:
                failure = {"dataset_label": label, "seed": seed, "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                print(f"[failed] {label} seed={seed} {failure['error']}", flush=True)
    all_rows = add_dbp15k_subsets_average(all_rows)
    write_raw_summary(args.output_dir, all_rows)
    write_method_tables(args.output_dir, all_rows)
    write_method_pm_tables(args.output_dir, all_rows)
    with (args.output_dir / "failures.json").open("w") as handle:
        json.dump(failures, handle, indent=2)
    print(f"saved: {args.output_dir}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
