# ViT Experiment — Rotated MNIST Permutation Test (LightViT)

Permutation test of the trustworthiness framework on a controlled **Rotated
MNIST** benchmark using **LightViT** (`ViT_Tiny_P4_28`: patch 4, embed dim 96,
depth 6, 3 heads, learnable positional embeddings + CLS token).

- **Treated digits** `T = {0,1,2,3,4}` are rotated by a configurable mean angle
  (truncated Gaussian, `sd = 2°`); **untreated digits** `F = {5,6,7,8,9}` are
  centered at `0°`.
- The model is trained on the **true** treatment labels and then on **all
  `C(10,5) = 252` unique label permutations**, yielding the permutation null and
  the one-sided Fisher-style p-value (`p_min = 1/252 ≈ 0.0040`).

## Files

| File | Purpose |
|------|---------|
| `train.py` | Full experiment entry point. |
| `models.py` | `ViT_Tiny_P4_28` / `LightVisionTransformer` (LightViT). |

## Usage

```bash
# Default 45-degree treated rotation (paper setting)
python train.py --angle 45
```

### Reproduce the angle sweep

The paper reports `{0, 5, 10, 15, 20, 25, 30, 45, 60, 90}`. **Change `--angle`
to reproduce any other row** — no other edits needed:

```bash
for a in 0 5 10 15 20 25 30 45 60 90; do
    python train.py --angle $a
done
```

| Flag | Default | Notes |
|------|---------|-------|
| `--angle` | `45` | Mean treated-class rotation (degrees). |
| `--gpu` | `0` | CUDA device id. |
| `--iterations` | `5` | Outer repetitions. |
| `--epochs` / `--perm-epochs` | `5` / `5` | Training length (true / permuted models). |
| `--data-dir` | `./data` | MNIST download/cache location (downloaded automatically). |

## Reproducibility note

The **LightViT** p-values reported in the paper were produced on a separate
Harvard workstation (different hardware/driver/seed state); a local re-run may
yield slightly different p-values and is **not** bit-for-bit identical. The
252-permutation framework itself is shared with the CNN experiment.
