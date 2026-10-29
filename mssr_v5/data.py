"""V2 graph loader, including deterministic SG paired views."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from mssr_v5.graph import LoadedGraph, load_graph_corpus
from mssr_v5.task import RetrievalTaskV2


def _rng(view_id: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{view_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _sg_view(base: LoadedGraph, view_id: str, recipe: dict) -> LoadedGraph:
    """Create an identity-preserving graph view without leaking its parent ID."""
    rng = _rng(view_id, int(recipe["seed"]))
    n = base.num_nodes
    keep = rng.random(n) >= float(recipe["node_dropout"])
    minimum = min(2, n)
    if int(keep.sum()) < minimum:
        keep[rng.choice(n, size=minimum, replace=False)] = True
    old_ids = np.flatnonzero(keep).tolist()
    remap = {old: new for new, old in enumerate(old_ids)}
    edge_rows = []
    for idx, (src, dst) in enumerate(zip(base.edge_src, base.edge_dst)):
        if src in remap and dst in remap and rng.random() >= float(recipe["edge_dropout"]):
            edge_rows.append((idx, remap[src], remap[dst]))

    def node_array(value: np.ndarray | None) -> np.ndarray | None:
        if value is None:
            return None
        out = value[old_ids].copy()
        mask = rng.random(out.shape[0]) < float(recipe["feature_dropout"])
        out[mask] = 0
        return out

    edge_idx = [row[0] for row in edge_rows]
    edge_text = base.edge_text[edge_idx].copy() if base.edge_text is not None else None
    return replace(
        base,
        graph_id=view_id,
        num_nodes=len(old_ids),
        num_edges=len(edge_rows),
        node_labels=[base.node_labels[i] for i in old_ids],
        node_label_ids=[base.node_label_ids[i] for i in old_ids],
        edge_src=[row[1] for row in edge_rows],
        edge_dst=[row[2] for row in edge_rows],
        edge_relations=[base.edge_relations[i] for i in edge_idx],
        edge_relation_ids=[base.edge_relation_ids[i] for i in edge_idx],
        node_text=node_array(base.node_text),
        edge_text=edge_text,
        node_visual=node_array(base.node_visual),
        # Keep the base-node indices only as runtime supervision metadata.  They
        # are not consumed by retrieval, but let matching models learn the true
        # correspondence between two independently dropped SG views.
        meta={"v2_view": recipe["view"], "base_node_ids": old_ids,
              "has_text": base.node_text is not None, "has_visual": base.node_visual is not None},
    )


def load_task_corpus(task: RetrievalTaskV2) -> dict[str, LoadedGraph]:
    ids = sorted(set(task.query_ids) | set(task.database_ids))
    if not task.views:
        return load_graph_corpus(task.graph_store, graph_ids=ids)
    base_ids = sorted({recipe["base_graph_id"] for recipe in task.views.values()})
    bases = load_graph_corpus(task.graph_store, graph_ids=base_ids)
    return {
        view_id: _sg_view(bases[recipe["base_graph_id"]], view_id, recipe)
        for view_id, recipe in task.views.items()
    }
