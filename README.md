# DVSH

Source code for DVSH: Dual-View Semantic Hashing for Efficient Graph Retrieval.

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

### 1. FB15K–DB15K and FB15K–YAGO15K

* Download the datasets from https://github.com/mniepert/mmkb.

### 2. DBP15K fr–en, ja–en， zh–en and DBP15K_all

* Download the datasets from https://github.com/Jingkang50/OpenPSG.




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
