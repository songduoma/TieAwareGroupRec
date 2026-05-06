# RecSys 2026 Reproducibility Artifact

**Paper title:** *Are We Really Making Progress in Group Recommendation? Unmasking the Tie-Breaking Illusion*  
**Track:** RecSys 2026 Reproducibility

This repository is the artifact package for the paper above.  
It contains all code, datasets, and archived run logs needed to reproduce the reported findings on:

- evaluation inflation under deterministic tie-breaking,
- tie-aware evaluation and tie statistics,
- removing additional sigmoid,
- temperature-scaled BPR (τ-BPR) mitigation.

## 1) Artifact Checklist (Track Requirement Mapping)

| Required item | Where in this artifact |
|---|---|
| Source code | `AlignGroup/`, `DHMAE/`, `ITR/`, `DGGVAE/`, `WWW2023ConsRec/`, `Baseline/` |
| Data | `*/data/` (each method folder includes dataset files it uses) |
| Installation instructions | Section **4) Installation** |
| Hardware configuration | Section **5) Hardware Configuration** |
| Reproduction instructions | Section **6) Reproduction** |

## 2) Repository Structure

- `WWW2023ConsRec/`: ConsRec (WWW 2023), plus τ-BPR experiments.
- `AlignGroup/`: AlignGroup (CIKM 2024).
- `DHMAE/`: DHMAE (SIGIR 2024).
- `ITR/`: ITR (NeurIPS 2024).
- `DGGVAE/`: DGGVAE (TOIS 2026).
- `Baseline/`: AGREE, GroupIM, HCR, HyperGroup, HHGR, CubeRec.

Result artifacts included in this artifact:

- `log_original`: archived runs under the original evaluation/code setting.
- `log_revised`: archived runs under the revised/corrected setting.
- `log_tau_*`: archived temperature sweep runs for ConsRec.
- `output_*_{original,revised}_seed*.log`: archived ITR per-seed stdout logs.
- `metrics_tie_aware_*`: archived DHMAE per-seed tie-aware summaries.

## 3) Data

### 3.1 Datasets in the paper

- `Mafengwo`
- `CAMRa2011`

Main files per dataset (same naming across methods):

- `userRatingTrain.txt`, `userRatingTest.txt`, `userRatingNegative.txt`
- `groupRatingTrain.txt`, `groupRatingTest.txt`, `groupRatingNegative.txt`
- `groupMember.txt`

### 3.2 Dataset statistics (paper-reported)

| Dataset | Users | Items | Groups | User-item interactions | Group-item interactions |
|---|---:|---:|---:|---:|---:|
| Mafengwo | 5,275 | 1,513 | 995 | 39,761 | 3,595 |
| CAMRa2011 | 602 | 7,710 | 290 | 116,344 | 145,068 |

### 3.3 Notes

- Train/test splits follow original codebases and are already prepared in the repository.
- No external data download is required for the two main datasets.

## 4) Installation

### 4.1 Exact environment used for paper experiments

```bash
conda create -n gr_reproducibility python=3.9.25 -y
conda activate gr_reproducibility

# PyTorch stack used in experiments
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# Core dependencies used by this artifact
pip install numpy==1.26.4 scipy==1.13.1 scikit-learn==1.6.1 tqdm==4.67.1 tensorboardX==2.6.4

# DGGVAE dependency
pip install torch-geometric==2.6.1
```

The package versions listed in this section are the ones used for the reported experiments.

### 4.2 Quick sanity checks

```bash
python -V
python -c "import torch, numpy, scipy, sklearn, tensorboardX; print(torch.__version__)"
python -c "import torch_geometric; print(torch_geometric.__version__)"
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.device_count())"
```

## 5) Hardware Configuration

This section reports the actual machine used for experiments.

### 5.1 Host and OS

- Host type: local GPU server
- OS: `Ubuntu 22.04.4 LTS (Jammy Jellyfish)`
- Kernel: `Linux 5.15.0-107-generic`

### 5.2 CPU and memory

- CPU: `AMD EPYC 7313 16-Core Processor`
- Sockets: `2`
- Cores per socket: `16`
- Threads per core: `2`
- Total logical CPUs: `64`
- System memory: `188 GiB`

### 5.3 GPU

- Driver version: `535.171.04`
- CUDA (driver): `12.2`
- CUDA used by PyTorch: `12.1`
- Number of GPUs: `8`
- GPU inventory:

| GPU type | Count | Memory per GPU |
|---|---:|---:|
| NVIDIA GeForce RTX 3090 | 2 | 24 GB |
| NVIDIA GeForce RTX 2080 Ti | 2 | 11 GB |
| NVIDIA GeForce RTX 4090 | 1 | 24 GB |
| NVIDIA RTX A5000 | 3 | 24 GB |

Primary training GPUs for this paper were RTX 3090 and RTX A5000.

### 5.4 Python runtime

- Python executable: Conda environment Python
- Python version: `3.9.25`
- Conda environment: artifact environment
- PyTorch: `2.5.1+cu121`

## 6) Reproduction

To inspect archived results without retraining, use each method's `print.py` and `print_aware.py` on the bundled log folders or use text files directly.

## 6.1 Full rerun: Retrain and regenerate logs

This mode is slower but regenerates experiment logs from code.

### Common practice

- Run each method from its own directory.
- Use three seeds and average.
- Store outputs separately for `original` vs `revised` runs.

### 6.1.1 ConsRec (WWW2023ConsRec)

```bash
cd WWW2023ConsRec

# Mafengwo
for s in 0 1 2; do
  python -u main.py --dataset=Mafengwo --predictor=MLP --loss_type=BPR \
    --learning_rate=0.0001 --device=cuda:0 --num_negatives=8 --layers=3 \
    --epoch=200 --tau=1 --seed=$s
done

# CAMRa2011
for s in 0 1 2; do
  python -u main.py --dataset=CAMRa2011 --predictor=DOT --loss_type=BPR \
    --learning_rate=0.001 --device=cuda:0 --num_negatives=2 --layers=2 \
    --epoch=30 --tau=1 --seed=$s
done
```

### 6.1.2 ITR

```bash
cd ITR

for s in 0 1 2; do
  python -u main.py --dataset=CAMRa2011 --predictor=DOT --loss_type=BPR \
    --learning_rate=0.001 --device=cuda:0 --num_negatives=2 --layers=2 \
    --epoch=30 --seed=$s > output_camra2011_revised_seed${s}.log 2>&1
done

for s in 0 1 2; do
  python -u main.py --dataset=Mafengwo --predictor=MLP --loss_type=BPR \
    --learning_rate=0.0001 --device=cuda:0 --num_negatives=8 --layers=3 \
    --epoch=2000 --seed=$s > output_mafengwo_revised_seed${s}.log 2>&1
done
```

### 6.1.3 AlignGroup

```bash
cd AlignGroup

# Mafengwo (temp=[0.2])
for s in 0 1 2; do
  python -u main.py --dataset=Mafengwo --device=cuda:0 --seed=$s
done

# CAMRa2011 (temp=[0.8])
for s in 0 1 2; do
  python -u main.py --dataset=CAMRa2011 --device=cuda:0 --seed=$s
done
```

### 6.1.4 DGGVAE

```bash
cd DGGVAE

# Mafengwo default: k=[50], temp=[0.2]
for s in 0 1 2; do
  python -u main.py --dataset=Mafengwo --device=cuda:0 --seed=$s
done

# CAMRa2011 default: k=[60], temp=[0.4]
# Set these defaults in main.py before running CAMRa2011.
for s in 0 1 2; do
  python -u main.py --dataset=CAMRa2011 --device=cuda:0 --seed=$s
done
```

### 6.1.5 DHMAE

```bash
cd DHMAE

# `run.sh` provides the base hyperparameter templates.
# For paper-style reporting, run multiple seeds and average.
for s in 0 1 2; do
  python -u main.py --dataset=CAMRa2011 --num_negatives=6 --num_enc_layers=1 \
    --num_dec_layers=3 --sce_alpha=1 --drop_ratio=0.0 --epoch=30 \
    --seed=$s --device=cuda:0
done

for s in 0 1 2; do
  python -u main.py --dataset=Mafengwo --num_negatives=10 --num_enc_layers=2 \
    --num_dec_layers=3 --sce_alpha=2 --drop_ratio=0.1 --epoch=200 \
    --seed=$s --device=cuda:0
done
```

### 6.1.6 Baselines

```bash
# Run baseline methods with three seeds.
# Logs are saved under each method's `log_rerun/` directory.

SEEDS=(0 1 2)
HHGR_SEEDS=(1111 1112 1113)

# AGREE
cd Baseline/AGREE
mkdir -p log_rerun
for s in "${SEEDS[@]}"; do
  python -u main.py --dataset=Mafengwo  --seed=$s --device=cuda:0 > log_rerun/mafengwo_seed${s}.log 2>&1
  python -u main.py --dataset=CAMRa2011 --seed=$s --device=cuda:0 > log_rerun/camra2011_seed${s}.log 2>&1
done

# GroupIM
cd ../GroupIM
mkdir -p log_rerun
for s in "${SEEDS[@]}"; do
  python -u main.py --dataset=Mafengwo  --seed=$s --device=cuda:0 > log_rerun/mafengwo_seed${s}.log 2>&1
  python -u main.py --dataset=CAMRa2011 --seed=$s --device=cuda:0 > log_rerun/camra2011_seed${s}.log 2>&1
done

# HyperGroup
cd ../HyperGroup
mkdir -p log_rerun
for s in "${SEEDS[@]}"; do
  python -u main.py --dataset=Mafengwo  --seed=$s --device=cuda:0 > log_rerun/mafengwo_seed${s}.log 2>&1
  python -u main.py --dataset=CAMRa2011 --seed=$s --device=cuda:0 > log_rerun/camra2011_seed${s}.log 2>&1
done

# HHGR
cd ../HHGR
mkdir -p log_rerun
for s in "${HHGR_SEEDS[@]}"; do
  python -u main.py --dataset=Mafengwo  --seed=$s --device=cuda:0 > log_rerun/mafengwo_seed${s}.log 2>&1
  python -u main.py --dataset=CAMRa2011 --seed=$s --device=cuda:0 > log_rerun/camra2011_seed${s}.log 2>&1
done

# CubeRec
cd ../CubeRec
mkdir -p log_rerun
for s in "${SEEDS[@]}"; do
  python -u main.py --dataset=Mafengwo  --seed=$s --device=cuda:0 --epoch=100 > log_rerun/mafengwo_seed${s}.log 2>&1
  python -u main.py --dataset=CAMRa2011 --seed=$s --device=cuda:0 --epoch=30 > log_rerun/camra2011_seed${s}.log 2>&1
done

cd ../..
```

`HCR` reads hyperparameters from `Baseline/HCR/config.py` (not argparse).  
Switch dataset by editing `self.path` in `config.py` before `python main.py`.

## 6.2 Original vs revised model switch (for methods with additional sigmoid issue)

In this repository, the revised code path is active by default for the affected methods.

Here, “revised” refers to retraining/evaluating models after removing the additional sigmoid applied to item scores before the BPR objective. It is not merely a post-hoc evaluation change.

To rerun the “original” behavior, re-enable the sigmoid at these locations:

- `AlignGroup/model.py`
- `WWW2023ConsRec/model.py`
- `ITR/model.py`
- `DGGVAE/model.py`
- `DHMAE/model.py` 
- `Baseline/AGREE/model.py`
- `Baseline/HCR/model.py`
- `Baseline/HyperGroup/model.py`

`GroupIM`, `HHGR`, and `CubeRec` are not part of this pre-BPR sigmoid switch.

Archived original logs are already bundled, so this manual switch is optional unless you need full reruns from scratch.

## 6.3 Temperature-scaled BPR (τ-BPR) experiment (Table 6 / Figure 2)

```bash
cd WWW2023ConsRec
for tau in 1 2 4 8 16 32 64; do
  for seed in 0 1 2; do
    python -u main.py --dataset=Mafengwo --predictor=MLP --loss_type=BPR \
      --learning_rate=0.0001 --device=cuda:0 --num_negatives=8 --layers=3 \
      --epoch=200 --tau=$tau --seed=$seed
  done
done
```

Archived sweep logs are in `WWW2023ConsRec/log_tau_*`.

## 6.4 Mapping to paper claims

- **Inflation under original protocol (Table 3):** compare `print.py` vs `print_aware.py` on `log_original` (or `output_*_original_*` / DHMAE original txt).
- **Tie size vs drop correlation (Table 4, Figure 1):** use `metrics_tie_aware` log lines (`num of top-score tie`, `top-score tie ratio`, `samples with tied top`).
- **Removing sigmoid / corrected comparison (Table 5):** evaluate `log_revised` (or equivalent revised outputs).
- **τ-BPR mitigation (Table 6, Figure 2):** use `WWW2023ConsRec/log_tau_*` or rerun Section 6.3.

## 7) Expected Output Format

Parser scripts report best-epoch metrics in this form:

- Group/User `Hit@[1,5,10]`
- Group/User `NDCG@[1,5,10]`
- For tie-aware logs: top-score tie statistics are printed during evaluation.

These outputs are sufficient to reconstruct the tables/figures discussed above.

## 8) Known Notes

- Several scripts use `argparse(..., type=list, ...)` for hyperparameter lists.
  Dataset-specific list settings in paper runs were set directly in code defaults.
- `HCR` uses `config.py` instead of command-line args.
- `ITR` writes to stdout by default; redirect to `.log` files for easier bookkeeping.
- `DHMAE` archived original/revised seed results are already provided as text summaries.

## 9) Citation

If you use this artifact, please cite the main paper and corresponding base model papers (AlignGroup, DHMAE, ITR, DGGVAE, ConsRec, and baselines) as appropriate.
