# DVSH

Source code for DVSH:: ual-View Semantic Hashing for Efficient Graph Retrieval.

## Environment Setup

### 1. Create and activate a conda environment

```bash
conda create --name mssr python=3.10
conda activate mssr
```

### 2. Install required dependencies

Run from the repository directory:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The evaluation supports six datasets: FB15K–DB15K, FB15K–YAGO15K, DBP15K fr–en, DBP15K ja–en, DBP15K zh–en and DBP15K_all. 

Prepare the processed graph files, retrieval protocols and precomputed semantic embeddings using the following layout:

```text
data/processed_schema_v2/
├── fb15k_db15k/        # manifest.json and protocols/
├── fb15k_yago15k/      # manifest.json and protocols/
├── dbp15k/            # manifest.json and protocols/unique_match_<pair>.json
├── dbp15k_all/        # manifest.json and protocols/unique_match_all.json
└── assets/
    ├── fb15k_db15k/graphs/
    ├── fb15k_yago15k/graphs/
    └── dbp15k/graphs/

cache/mssr_semantic/
├── fb15k_db15k.npz
├── fb15k_yago15k.npz
├── dbp15k_fr_en.npz
├── dbp15k_ja_en.npz
├── dbp15k_zh_en.npz
└── dbp15k_all.npz
```


## Running the Code

Create a log directory first:

```bash
mkdir -p logs
```

### FB15K–DB15K and FB15K–YAGO15K

```bash
nohup python -u -m mssr_v5.run_final_5seed --datasets fb15k_db15k --workers 1 --output-dir results/fb15k_db15k > logs/fb15k_db15k.log 2>&1 &
nohup python -u -m mssr_v5.run_final_5seed --datasets fb15k_yago15k --workers 1 --output-dir results/fb15k_yago15k > logs/fb15k_yago15k.log 2>&1 &
```

### DBP15K fr–en, ja–en and zh–en

```bash
nohup python -u -m mssr_v5.run_final_5seed --datasets dbp15k_fr_en --workers 1 --output-dir results/dbp15k_fr_en > logs/dbp15k_fr_en.log 2>&1 &
nohup python -u -m mssr_v5.run_final_5seed --datasets dbp15k_ja_en --workers 1 --output-dir results/dbp15k_ja_en > logs/dbp15k_ja_en.log 2>&1 &
nohup python -u -m mssr_v5.run_final_5seed --datasets dbp15k_zh_en --workers 1 --output-dir results/dbp15k_zh_en > logs/dbp15k_zh_en.log 2>&1 &
```

### DBP15K_all

```bash
nohup python -u -m mssr_v5.run_final_5seed --datasets dbp15k_all --workers 1 --output-dir results/dbp15k_all > logs/dbp15k_all.log 2>&1 &
```


For GPU execution, install a CUDA-compatible PyTorch build and append `--device cuda:0 --threads 4`. To run one fixed seed, append `--seeds 3407`. Use a new output directory for each GPU run.

## Offline time including recorded CLIP generation

By default, both CPU and CUDA runs report:

```text
Total Offline = recorded CLIP generation minimum + Training + Encoding + Index Build
```

The package includes [the complete five-repeat records](mssr_v5/clip_embedding_timings.json)
and [a six-dataset CSV summary](mssr_v5/clip_embedding_times.csv). These files are
also included in the installed wheel. No original audit directory or CLIP model
installation is needed to load the recorded cost; graph data and compatible
precomputed semantic caches are still required to run retrieval.

| Dataset | Recorded CLIP minimum (s) |
|---|---:|
| FB15K-DB15K | 3.691238 |
| FB15K-YAGO15K | 3.740416 |
| dbp15k_fr_en | 5.021570 |
| dbp15k_ja_en | 6.201558 |
| dbp15k_zh_en | 5.964062 |
| dbp15k_all | 18.389396 |

Measurements used `sentence-transformers/clip-ViT-B-32-multilingual-v1`, revision
`58edf8cada9e398793dca955574a48cbb7f18be2`, a Tesla V100-SXM2-32GB and batch size
256. Each value is the minimum of five fresh repeats. This is text embedding
inference, covering the unique graph IDs in the test query and database sets.
It includes tokenization, encoder inference, normalization and CPU result transfer;
it excludes model loading, input preparation, warmup and output persistence.

The same fixed CLIP cost is added to every algorithm seed. It is a **reference
machine measurement**, not a measurement of CLIP speed on the reader's hardware.
The new total is a sum of separately measured stages, not a continuous end-to-end
wall-clock measurement. Its across-seed standard deviation does not include CLIP
measurement variability. Training remains zero; inference is not training.

Normal commands above include this cost automatically. Reports and summary CSVs expose:

- `clip_embedding_generation_time_ms`: recorded minimum;
- `total_offline_without_clip_time_ms`: original algorithm preparation time;
- `total_offline_time_ms`: their sum.

Full reports and row JSON also retain `clip_timing_*` provenance. Encoding and
index fields retain their previous definitions; auxiliary relation-cache writes
remain separate. Do not add the CLIP cost a second time when processing the total.

The loader validates the full graph ID set and semantic-cache SHA-256 against the
measurement. Paths inside the JSON are provenance labels only and are not opened.
Only the verified original cache or the verified measured-output cache is accepted.
A newly generated/repacked cache can have a different hash even if its values are
similar; a mismatch is rejected rather than silently applying an unrelated cost.

For custom or truncated tasks, or for reporting only algorithm preparation, use:

```bash
python -m mssr_v5.run_final_5seed --datasets fb15k_db15k --seeds 3407 --workers 1 \
  --clip-timing-policy algorithm-only --output-dir results/algorithm_only
```

For new measurements, supply `--clip-timing-manifest /path/to/clip_embedding_timings.json`
using the bundled manifest schema, actual five-repeat measurements and verified
input/cache identity. Do not change the identity fields merely to bypass validation.
To inspect the pre-recorded values without loading datasets:

```python
import json
from mssr_v5.offline_timing import DEFAULT_CLIP_TIMINGS

records = json.loads(DEFAULT_CLIP_TIMINGS.read_text())["datasets"]
for dataset, record in records.items():
    fastest_ms = min(run["embedding_generation_time_ms"] for run in record["runs"])
    print(dataset, fastest_ms / 1000, "seconds")
```

Run offline accounting regression checks from the repository directory:

```bash
python -m unittest discover -s tests
```
