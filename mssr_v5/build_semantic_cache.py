"""Build training-free multilingual CLIP-space center-entity embeddings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from mssr_v5.task import load_task


MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


def entity_text(entity: str, aliases: dict[str, dict[str, str]]) -> str:
    value = entity.strip().strip("<>")
    if value.startswith("/m/"):
        row = aliases.get(value, {})
        label, description = row.get("label", ""), row.get("description", "")
        if label and description:
            return f"{label}. {description}"
        return label or value
    name = unquote(value.rstrip("/").rsplit("/", 1)[-1]).replace("_", " ")
    return name or value


def main() -> None:
    parser = argparse.ArgumentParser()
    project = Path(__file__).resolve().parents[1]
    parser.add_argument("--dataset", required=True,
                        choices=["fb15k_db15k", "fb15k_yago15k", "dbp15k"])
    parser.add_argument("--language-pair", choices=["zh_en", "ja_en", "fr_en"])
    parser.add_argument("--schema-root", type=Path, default=project / "data" / "processed_schema_v2")
    parser.add_argument("--aliases", type=Path, default=project / "data" / "external" / "freebase_wikidata_aliases.json")
    parser.add_argument("--output-dir", type=Path, default=project / "cache" / "mssr_semantic")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.dataset == "dbp15k" and not args.language_pair:
        parser.error("DBP15K requires --language-pair")
    os.environ.setdefault("HF_HOME", str(project / "cache" / "huggingface"))
    from sentence_transformers import SentenceTransformer

    aliases = {}
    if args.aliases.exists():
        with args.aliases.open("r", encoding="utf-8") as handle:
            aliases = json.load(handle)
    task = load_task(args.schema_root, args.dataset, "test", args.language_pair)
    graph_ids = sorted(set(task.query_ids) | set(task.database_ids))
    texts = []
    for offset, graph_id in enumerate(graph_ids, 1):
        path = task.graph_store / "graphs" / f"{graph_id}.json"
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        center = str(payload.get("meta", {}).get("center_entity", ""))
        texts.append(entity_text(center, aliases))
        if offset % 5000 == 0:
            print(f"metadata {offset}/{len(graph_ids)}", flush=True)
    model = SentenceTransformer(MODEL, device=args.device)
    embeddings = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.language_pair}" if args.language_pair else ""
    output = args.output_dir / f"{args.dataset}{suffix}.npz"
    np.savez_compressed(output, graph_ids=np.asarray(graph_ids), embeddings=embeddings,
                        texts=np.asarray(texts), model=np.asarray(MODEL))
    print(f"saved {embeddings.shape} to {output}")


if __name__ == "__main__":
    main()
