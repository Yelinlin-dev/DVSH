"""Materialised V2 task loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RetrievalTaskV2:
    dataset: str
    task_type: str
    split: str
    query_ids: list[str]
    database_ids: list[str]
    positives: dict[str, list[str]] = field(default_factory=dict)
    distances: dict[str, dict[str, float]] = field(default_factory=dict)
    views: dict[str, dict[str, Any]] = field(default_factory=dict)
    graph_store: Path = Path()
    metadata: dict[str, Any] = field(default_factory=dict)


def load_task(
    schema_root: Path,
    dataset: str,
    split: str = "test",
    language_pair: str | None = None,
) -> RetrievalTaskV2:
    root = schema_root / dataset
    with (root / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if dataset == "dbp15k":
        if language_pair not in {"zh_en", "ja_en", "fr_en"}:
            raise ValueError("DBP15K requires --language-pair zh_en, ja_en, or fr_en")
        protocol_path = root / "protocols" / f"unique_match_{language_pair}.json"
    else:
        protocol_path = root / str(manifest["protocol"])
    with protocol_path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    if split not in protocol["splits"]:
        raise ValueError(f"split {split!r} unavailable in {protocol_path}")
    selected = protocol["splits"][split]
    if "query_ids" not in selected:
        raise ValueError(f"split {split!r} is training-only and not a retrieval task")
    # Portable manifests resolve relative graph stores against the dataset directory.
    graph_store = Path(manifest["graph_store"])
    if not graph_store.is_absolute():
        graph_store = (root / graph_store).resolve()
    if not graph_store.exists():
        local_graph_store = schema_root / "assets" / dataset
        if local_graph_store.exists():
            graph_store = local_graph_store
        else:
            raise FileNotFoundError(f"Graph store not found: {graph_store}")
    return RetrievalTaskV2(
        dataset=dataset,
        task_type=protocol["task_type"],
        split=split,
        query_ids=selected["query_ids"],
        database_ids=selected["database_ids"],
        positives=selected.get("positives", {}),
        distances=selected.get("distances", {}),
        views=selected.get("views", {}),
        graph_store=graph_store,
        metadata={key: value for key, value in protocol.items() if key != "splits"},
    )
