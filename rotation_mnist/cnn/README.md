# CNN Experiment — Rotated MNIST Permutation Test (LightCNN)

Permutation test of the trustworthiness framework on a controlled **Rotated
MNIST** benchmark using **LightCNN** (`SimpleCNN_MNIST`).

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
| `models.py` | `SimpleCNN_MNIST` (LightCNN). |

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
| `--gpu` | `1` | CUDA device id. |
| `--iterations` | `5` | Outer repetitions. |
| `--epochs` / `--perm-epochs` | `5` / `5` | Training length (true / permuted models). |
| `--data-dir` | `./data` | MNIST download/cache location (downloaded automatically). |

The LightCNN numbers reproduce locally with the seeds in this script.
