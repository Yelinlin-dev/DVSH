"""GraphSample 加载与图结构工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np



@dataclass
class LoadedGraph:
    """运行时图对象，用于 MSSR 检索。"""

    graph_id: str
    source_dataset: str
    task_family: str
    num_nodes: int
    num_edges: int
    node_labels: list[str]
    node_label_ids: list[int]
    edge_src: list[int]
    edge_dst: list[int]
    edge_relations: list[str]
    edge_relation_ids: list[int]
    node_text: np.ndarray | None = None
    edge_text: np.ndarray | None = None
    node_visual: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def load_graph_json(path: Path, dataset_root: Path) -> LoadedGraph:
    """从 GraphSample JSON 加载图，并解析外置 npz 特征。"""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    node_text = None
    edge_text = None
    node_visual = None
    feature_files = payload.get("meta", {}).get("feature_files")
    if feature_files and "bundle" in feature_files:
        npz_path = dataset_root / feature_files["bundle"]
        if npz_path.exists():
            with np.load(npz_path) as data:
                if "node_text" in data:
                    node_text = data["node_text"].astype(np.float32)
                if "edge_text" in data:
                    edge_text = data["edge_text"].astype(np.float32)
                if "node_visual" in data:
                    node_visual = data["node_visual"].astype(np.float32)

    nodes = payload["nodes"]
    edges = payload["edges"]
    meta = dict(payload.get("meta", {}))
    # Entity-centric V2 corpora identify the retrieval entity in the node
    # records.  Preserve its local index for pair models (e.g. GELATO/GEDIOT)
    # that require an explicit supervised anchor across two positive graphs.
    center_indices = [int(node["node_id"]) for node in nodes if node.get("is_center")]
    if len(center_indices) == 1:
        meta["center_node_index"] = center_indices[0]
    return LoadedGraph(
        graph_id=payload["graph_id"],
        source_dataset=payload["source_dataset"],
        task_family=payload["task_family"],
        num_nodes=payload["num_nodes"],
        num_edges=payload["num_edges"],
        node_labels=[n["label"] for n in nodes],
        node_label_ids=[n["label_id"] for n in nodes],
        edge_src=[e["src"] for e in edges],
        edge_dst=[e["dst"] for e in edges],
        edge_relations=[e["relation"] for e in edges],
        edge_relation_ids=[e["relation_id"] for e in edges],
        node_text=node_text,
        edge_text=edge_text,
        node_visual=node_visual,
        meta=meta,
    )


def load_graph_corpus(
    dataset_root: Path,
    graph_ids: list[str] | None = None,
    max_graphs: int | None = None,
) -> dict[str, LoadedGraph]:
    """批量加载图库。"""
    graphs_dir = dataset_root / "graphs"
    if graph_ids is None:
        paths = sorted(graphs_dir.glob("*.json"))
    else:
        paths = [graphs_dir / f"{gid}.json" for gid in graph_ids]
    if max_graphs is not None:
        paths = paths[:max_graphs]

    corpus: dict[str, LoadedGraph] = {}
    for path in paths:
        if not path.exists():
            continue
        graph = load_graph_json(path, dataset_root)
        corpus[graph.graph_id] = graph
    return corpus

