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

## Data

| Dataset | Source | Local layout |
|---------|--------|--------------|
| MNIST / Fashion-MNIST | Auto-download via `torchvision` | `./data` |
| Raman Bacteria-ID | [Ho et al., 2019](https://github.com/csho33/bacteria-ID) | `pvalue_sampling/data/X_reference.npy`, `y_reference.npy`, `X_test.npy`, `y_test.npy` |
| QPM-WGS-AMR-21 | [Ahmad et al., 2023](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2023.1154620/full) | Place `.mat` / image arrays as paths specified in each QPM notebook |

Raw Raman spectra (~0.5 GB) are **not** bundled; download from the Bacteria-ID repository and place files under `pvalue_sampling/data/`.

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
