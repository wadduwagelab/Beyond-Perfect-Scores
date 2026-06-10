# Raman Bacteria-ID — Permutation Test & Convergence Ablation

This experiment applies the trustworthiness / permutation-test framework to the
**Bacteria-ID** Raman spectroscopy dataset (Ho et al., *Nature Communications*,
2019), testing whether classifiers exploit clinically meaningful structure on
two binary tasks:

- **Penicillin vs. Meropenem** empiric-treatment groups (`6,435` permutations,
  `p_min` sampled at `1,000` permutations ≈ `0.0002`).
- **MRSA vs. MSSA** (methicillin-resistant vs. -susceptible *S. aureus*).

It also contains the **"Number of Permutations for Stable Causal Inference"**
convergence ablation (p-value vs. number of permutations *k*).

## Files

| File | Purpose |
|------|---------|
| `config.py` | Strain indices, ATCC empiric-treatment groupings, antibiotic labels. |
| `penicillin_meropenem.py` | Main permutation test (Penicillin vs. Meropenem); writes `results/perm_results.csv`. |
| `mrsa_mssa.ipynb` | MRSA vs. MSSA permutation test. |
| `convergence_pvalues_vs_k.ipynb` | Generates the p-value-vs-*k* convergence figure from `results/perm_results.csv`. |
| `results/perm_results.csv` | Pre-computed per-permutation accuracies (so the convergence notebook runs without a multi-hour rerun). |
| `results/pvalues_by_k_clean.png` | The convergence figure used in the paper. |

## Data

The raw spectra are **not** included in this repository (≈0.5 GB and not ours to
redistribute). Download the public Bacteria-ID dataset and place the arrays in a
local `data/` folder:

```
raman_bacteria_id/
  data/
    X_reference.npy
    y_reference.npy
    X_test.npy
    y_test.npy
```

Dataset and preprocessing: <https://github.com/csho33/bacteria-ID>
(Ho et al., 2019). The scripts read these arrays from `./data` (`DATA_DIR`).

## Usage

```bash
# 1. Run the Penicillin vs. Meropenem permutation test (regenerates perm_results.csv)
python penicillin_meropenem.py

# 2. Build the convergence figure (uses results/perm_results.csv by default)
jupyter notebook convergence_pvalues_vs_k.ipynb

# 3. MRSA vs. MSSA task
jupyter notebook mrsa_mssa.ipynb
```

## Models

Three lightweight spectral classifiers are compared: a Simple MLP, an Improved
MLP, and **RamanNet** (1-D CNN), following the original Bacteria-ID setup.
