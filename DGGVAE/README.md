# DGGVAE: Dual-Granularity Graph Variational Auto-Encoder for Group Recommendation

## Introduction

## This is the Pytorch implementation for our DGGVAE paper:

>DGGVAE: Dual-Granularity Graph Variational Auto-Encoder for Group Recommendation

## Environment Requirement

- python 3.9
- Pytorch 2.1.0

## Datasets

We use two public experimental datasets: **Mafengwo**, **CAMRa2011**, and **Weeplaces**.
These three datasets' contents are in the `data/` folder.


## Run

```
# For Mafengwo
python main.py --dataset=Mafengwo 
(--k [50] --g_layers [3] --cl_weight [0.01] --temp [0.2])

# For CAMRa2011
python main.py --dataset=CAMRa2011 
(--k [60] --g_layers [3] --cl_weight [0.01] --temp [0.4])

# For Weeplaces
python main.py --dataset=Weeplaces
```

For more running options, please refer to `main.py`


