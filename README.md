# Beyond Perfect Scores: Proof-by-Contradiction for Trustworthy Machine Learning

Official code for the paper *Beyond Perfect Scores: Proof-by-Contradiction for Trustworthy Machine Learning* ([arXiv:2601.06704](https://arxiv.org/abs/2601.06704)).

**Authors:** Dushan N. Wadduwage\*, Dineth Jayakody, Leonidas Zimianitis — Old Dominion University, Norfolk, VA, USA

\*Corresponding author: dwadduwa@odu.edu

## Overview

High test accuracy alone is not enough to trust a biomedical machine-learning model. In many life-science datasets, thousands of **data units** (cells, spectra, image patches) come from only tens of **data buckets** (patients, isolates, specimens). That *many-units–few-buckets* structure violates the usual IID assumption and can inflate accuracy through shortcut learning, leakage, or spurious bucket-level correlations.

This repository implements a **stochastic proof-by-contradiction** test grounded in the Rubin causal model:

1. Train a classifier on the observed bucket-level labels.
2. Repeat training under many **random bucket-level label permutations** to build a null distribution of test accuracy.
3. Report a one-sided **Fisher-style p-value**: the fraction of permuted runs that match or exceed the observed accuracy.

A **small p-value** means the model’s performance is unlikely under the causal null (no label-relevant signal) and is more trustworthy. A **high p-value** despite strong accuracy warns that the model may be exploiting hierarchical artifacts rather than genuine biology.

## Repository structure

```
Beyond-Perfect-Scores/
├── rotation_mnist/          # Controlled benchmark: Rotated MNIST (LightCNN & LightViT)
│   ├── cnn/
│   └── vit/
├── Fashion-Mnist/           # Controlled benchmark: Colored Fashion-MNIST
│   ├── Cnn/
│   └── Vit/
├── Bacteria-QPM/            # QPM-WGS-AMR-21 quantitative phase microscopy tasks
│   ├── WT-NWT/
│   ├── GP-GN/
│   └── AB-KP/
├── Bacteria-id/             # Raman Bacteria-ID spectroscopy tasks
│   ├── Peniciline_and_Meropenem/
│   └── MRSA-MSSA/
└── pvalue_sampling/         # Permutation convergence ablation (Section 5.2)
```

Each subdirectory contains runnable scripts or notebooks for one paper experiment. Subfolder READMEs document flags, defaults, and reproduction steps where applicable.

## Requirements

- Python 3.8+
- PyTorch 2.0+ (paper experiments used PyTorch 2.0.1, CUDA 12.2, mixed precision)
- `torchvision`, `numpy`, `scikit-learn`, `matplotlib`
- Jupyter (for notebook-based experiments)
- NVIDIA GPU recommended for full permutation sweeps

## Data setup

**Raw datasets are not included in this repository.** Download each dataset from its source, place the files in the paths below, and **replace the sample paths hard-coded in the notebooks/scripts** with your local paths before running.

### 1. MNIST & Fashion-MNIST (auto-download)

No manual download needed. These experiments fetch data automatically on first run:

| Experiment | Default cache path in code |
|------------|----------------------------|
| `rotation_mnist/cnn/train.py` | `--data-dir ./data` |
| `rotation_mnist/vit/train.py` | `--data-dir ./data` |
| `Fashion-Mnist/Cnn/cnn.ipynb` | `root='./data'` |
| `Fashion-Mnist/Vit/vit.ipynb` | `root='./data'` |

### 2. Raman Bacteria-ID (~0.5 GB)

**Source:** [Ho et al., 2019 — bacteria-ID](https://github.com/csho33/bacteria-ID)

Download and preprocess the public Raman spectra, then place these four `.npy` files:

```
Beyond-Perfect-Scores/
  data/                          # repo-root data/ (used by Bacteria-id notebooks)
    X_reference.npy
    y_reference.npy
    X_test.npy
    y_test.npy
  pvalue_sampling/
    data/                        # same four files (used by pvalue_sampling scripts)
      X_reference.npy
      y_reference.npy
      X_test.npy
      y_test.npy
```

You can copy or symlink one folder into the other:

```bash
mkdir -p data pvalue_sampling/data
# after downloading, copy the four .npy files into data/
cp /path/to/downloaded/*.npy data/
ln -s ../data pvalue_sampling/data   # optional symlink
```

**Paths to update in code** (sample values currently in the repo):

| File | Variable / path in code | Set to |
|------|------------------------|--------|
| `pvalue_sampling/penicillin_meropenem.py` | `DATA_DIR = "./data"` | folder containing the four `.npy` files |
| `pvalue_sampling/mrsa_mssa.ipynb` | `DATA_DIR = ""` | e.g. `"./data"` |
| `Bacteria-id/MRSA-MSSA/MRSA_MSSA-simple-MLP.ipynb` | `DATA_DIR = "data"` | e.g. `"../../data"` or absolute path |
| `Bacteria-id/*/` (other notebooks) | `DATA_DIR = os.path.join(..., '..', 'data')` | resolves to repo-root `data/` when run from subfolders |

### 3. QPM-WGS-AMR-21 (quantitative phase microscopy)

**Source:** [Ahmad et al., 2023](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2023.1154620/full) — request or obtain the QPM `.mat` files from the authors / supplementary material.

**Recommended local layout:**

```
Beyond-Perfect-Scores/
  Bacteria-QPM/
    data/
      bacteria_train_full_data.mat
      bacteria_val_full_data.mat      # required for AB-KP only
      bacteria_test_full_data.mat
    WT-NWT/
      embeddings/                     # pretraining ablation only
        train_embeddings_mae_512.npy
        test_embeddings_mae_512.npy
```

**Sample paths currently in the notebooks — replace these with your paths:**

| Notebook | Sample path in code | Replace with |
|----------|---------------------|--------------|
| `Bacteria-QPM/WT-NWT/WT_NWT.ipynb` | `/home/dineth/bacteria_QPM/data/bacteria_train_full_data.mat` | `Bacteria-QPM/data/bacteria_train_full_data.mat` |
| `Bacteria-QPM/WT-NWT/WT_NWT.ipynb` | `/home/dineth/bacteria_QPM/data/bacteria_test_full_data.mat` | `Bacteria-QPM/data/bacteria_test_full_data.mat` |
| `Bacteria-QPM/GP-GN/GP_GN.ipynb` | same `/home/dineth/bacteria_QPM/data/...` paths | same `Bacteria-QPM/data/...` paths |
| `Bacteria-QPM/AB-KP/KP_AB.ipynb` | `bacteria_train/val/test_full_data.mat` under `/home/dineth/bacteria_QPM/data/` | `Bacteria-QPM/data/bacteria_{train,val,test}_full_data.mat` |
| `Bacteria-QPM/WT-NWT/WT_NWT-pretrained-version.ipynb` | `/home/dineth/bacteria_QPM/WT_NWT/Pre-train/embeddings/train_embeddings_mae_512.npy` | `Bacteria-QPM/WT-NWT/embeddings/train_embeddings_mae_512.npy` |
| `Bacteria-QPM/GP-GN/GP_GN.ipynb` | same embedding paths under `/home/dineth/bacteria_QPM/...` | `Bacteria-QPM/WT-NWT/embeddings/...` |

For the **pretraining ablation** (`WT_NWT-pretrained-version.ipynb`, `GP_GN.ipynb`): either load precomputed embeddings from the paths above, or uncomment the MAE embedding-extraction cells in the notebook to generate them from the `.mat` files.

### Quick checklist before running

1. Download Raman `.npy` files → place in `data/` and `pvalue_sampling/data/`.
2. Download QPM `.mat` files → place in `Bacteria-QPM/data/`.
3. Search the repo for `/home/dineth/` and update every match to your local path.
4. Set `DATA_DIR` in `mrsa_mssa.ipynb` (currently empty) to your Raman data folder.

## Interpreting results

| Observation | Interpretation |
|-------------|----------------|
| High accuracy, **low p-value** | Performance unlikely under label permutation → stronger causal evidence |
| High accuracy, **high p-value** | Accuracy compatible with spurious bucket structure → **not trustworthy** |
| Low accuracy, high p-value | May reflect weak model capacity rather than absent signal (see Improved MLP / RamanNet on Penicillin vs. Meropenem) |

Minimum attainable p-value: `p_min = 1/M` where `M` is the number of unique bucket-label permutations (reported per experiment in the paper).

## Citation

If you use this code or method, please cite:

```bibtex
@article{wadduwage2026beyond,
  title={Beyond Perfect Scores: Proof-by-Contradiction for Trustworthy Machine Learning},
  author={Wadduwage, Dushan N and Jayakody, Dineth and Zimianitis, Leonidas},
  journal={arXiv preprint arXiv:2601.06704},
  year={2026}
}
```
