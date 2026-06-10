"""
Rotated-MNIST permutation test — LightViT (ViT_Tiny_P4_28).

Treated digits {0,1,2,3,4} are rotated around a configurable mean angle
(`--angle`, degrees) drawn from a truncated Gaussian (sd=2°); untreated digits
{5,6,7,8,9} are centered at 0°. The model is trained on the true treatment
labels and then on all C(10,5) = 252 unique label permutations to build the
permutation null distribution and the one-sided Fisher-style p-value
(p_min = 1/252 ≈ 0.0040).

Reproduce a single row of the paper's angle-sweep table by changing --angle.
The paper sweeps {0, 5, 10, 15, 20, 25, 30, 45, 60, 90}.

Reproducibility: the LightViT p-values reported in the paper were produced on a
separate Harvard workstation; a local re-run may differ slightly and is not
bit-for-bit identical (see README).

Example
-------
    python train.py --angle 45
"""

import argparse
import itertools
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from models import LightVisionTransformer


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_rng_state():
    return {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def load_rng_state(state):
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def sample_trunc_normal(mean, sd, low, high, rng):
    x = rng.normal(mean, sd)
    while (x < low) or (x > high):
        x = rng.normal(mean, sd)
    return float(x)


def make_class_angle_map_two_means(
    T_CLASSES, F_CLASSES,
    mean_deg_T=60.0, sd_deg_T=4.0,
    mean_deg_F=30.0, sd_deg_F=12.0,
    low=-90.0, high=90.0, seed=42
):
    rng = np.random.default_rng(seed)
    angle_map = {}
    for c in T_CLASSES:
        angle_map[c] = sample_trunc_normal(mean_deg_T, sd_deg_T, low, high, rng)
    for c in F_CLASSES:
        angle_map[c] = sample_trunc_normal(mean_deg_F, sd_deg_F, low, high, rng)
    return angle_map


def rotate_by_class(images, labels, class_angle_map, fill=0.0):
    out = images.copy()
    for i, (img, c) in enumerate(zip(images, labels)):
        angle = class_angle_map[int(c)]
        tens = torch.from_numpy(img)
        rot = TF.rotate(
            tens, angle=angle,
            interpolation=InterpolationMode.BILINEAR,
            expand=False, fill=fill,
        )
        out[i] = rot.numpy()
    return out


def prepare_data_loader(images, labels, batch_size=64, shuffle=True):
    images = np.array(images)
    if images.ndim == 3:
        images = images[:, None, :, :]
    elif images.ndim != 4:
        raise ValueError(f"Expected images with ndim 3 or 4, got {images.ndim}")
    images_tensor = torch.from_numpy(images).float()
    labels_tensor = torch.LongTensor(np.asarray(labels))
    dataset = TensorDataset(images_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        bs = target.size(0)
        total_loss += loss.detach().item() * bs
        correct += (output.detach().argmax(dim=1) == target).sum().item()
        total += bs
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            bs = target.size(0)
            total_loss += loss.detach().item() * bs
            correct += (output.detach().argmax(dim=1) == target).sum().item()
            total += bs
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


def train_eval_model_vit(X_train, y_train, X_test, y_test, device,
                         epochs=5, lr=1e-4, verbose=True):
    # IMPORTANT: do NOT reseed here; the caller controls randomness via the
    # saved/loaded RNG state so every permutation starts from an identical point.
    model = LightVisionTransformer(num_classes=2, in_chans=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = prepare_data_loader(X_train, y_train, batch_size=64, shuffle=True)
    test_loader = prepare_data_loader(X_test, y_test, batch_size=64, shuffle=False)

    best_test_acc, best_epoch = -1.0, -1
    for epoch in range(1, epochs + 1):
        _, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        _, test_acc = evaluate(model, test_loader, criterion, device)
        if verbose:
            print(f"    Epoch {epoch}: Train Acc = {train_acc:.2f}%, "
                  f"Test Acc = {test_acc:.2f}%", flush=True)
        if test_acc > best_test_acc:
            best_test_acc, best_epoch = test_acc, epoch
    if verbose:
        print(f"[BEST] Epoch {best_epoch}: Test Acc = {best_test_acc:.2f}%", flush=True)
    return best_test_acc / 100.0


def generate_all_unique_permutations(T_CLASSES, t_classes=5, f_classes=5):
    base_pattern = [1] * t_classes + [0] * f_classes
    unique_patterns = list(set(itertools.permutations(base_pattern)))
    original_pattern = tuple([1 if i in T_CLASSES else 0 for i in range(10)])
    if original_pattern not in unique_patterns:
        unique_patterns[0] = original_pattern
        unique_patterns = list(set(unique_patterns))
    unique_patterns.sort()
    return unique_patterns, original_pattern


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--angle", type=float, default=45.0,
                        help="Mean treated-class rotation angle in degrees "
                             "(default: 45). Paper sweeps {0,5,10,15,20,25,30,45,60,90}.")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device id (default: 0).")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--perm-epochs", type=int, default=5)
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    set_seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = f"cuda:{args.gpu}"
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}", flush=True)
    else:
        device = "cpu"
        print("CUDA not available, using CPU", flush=True)
    print(f"Device: {device} | arch=vit | angle={args.angle}", flush=True)

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)

    train_images = np.array([img.numpy() for img, _ in train_dataset])
    train_labels = np.array([label for _, label in train_dataset])
    test_images = np.array([img.numpy() for img, _ in test_dataset])
    test_labels = np.array([label for _, label in test_dataset])
    print(f"Train images shape: {train_images.shape}", flush=True)
    print(f"Test images shape: {test_images.shape}", flush=True)

    T_CLASSES = [0, 1, 2, 3, 4]
    F_CLASSES = [5, 6, 7, 8, 9]
    T_train = np.array([1 if label in T_CLASSES else 0 for label in train_labels])
    T_test = np.array([1 if label in T_CLASSES else 0 for label in test_labels])
    print(f"T_train distribution: {np.bincount(T_train)}", flush=True)
    print(f"T_test distribution: {np.bincount(T_test)}", flush=True)

    all_unique_patterns, original_pattern = generate_all_unique_permutations(T_CLASSES)
    N_PERMUTATIONS = len(all_unique_patterns)
    N_ITERATIONS = args.iterations
    print(f"Number of unique permutations: {N_PERMUTATIONS}", flush=True)

    permutation_results = {i: [] for i in range(N_PERMUTATIONS)}
    true_accuracies = []

    for iter_idx in range(1, N_ITERATIONS + 1):
        print(f"\n{'='*80}\nITERATION {iter_idx}/{N_ITERATIONS}\n{'='*80}", flush=True)
        np.random.seed(42 + iter_idx)
        set_seed(123 + iter_idx)

        # ViT recipe: fixed angle seed (42) across iterations.
        angle_map_iter = make_class_angle_map_two_means(
            T_CLASSES, F_CLASSES,
            mean_deg_T=args.angle, sd_deg_T=2.0,
            mean_deg_F=0.0, sd_deg_F=2.0,
            low=-90.0, high=90.0, seed=42,
        )
        print("\nSampled angles (degrees) for this iteration:", flush=True)
        for c in range(10):
            group = "T" if c in T_CLASSES else "F"
            print(f"  Class {c} ({group}): {angle_map_iter[c]:.2f}", flush=True)

        train_images_rot_iter = rotate_by_class(train_images, train_labels, angle_map_iter)
        test_images_rot_iter = rotate_by_class(test_images, test_labels, angle_map_iter)

        base_rng_state = save_rng_state()

        print(f"\nIteration {iter_idx} - Training on TRUE T labels:", flush=True)
        load_rng_state(base_rng_state)
        true_acc = train_eval_model_vit(
            train_images_rot_iter, T_train,
            test_images_rot_iter, T_test,
            device=device, epochs=args.epochs, lr=1e-4, verbose=True,
        )
        true_accuracies.append(true_acc)

        print(f"\nIteration {iter_idx} - Running {N_PERMUTATIONS} permutations...", flush=True)
        for perm_idx, perm_pattern in enumerate(all_unique_patterns, 1):
            class_to_label = {cid: perm_pattern[cid] for cid in range(10)}
            is_original = (perm_pattern == original_pattern)
            T_train_permuted = np.array([class_to_label[c] for c in train_labels], dtype=int)
            T_test_permuted = np.array([class_to_label[c] for c in test_labels], dtype=int)

            if is_original:
                perm_acc = true_acc
                print(f"\nPermutation {perm_idx} [ORIGINAL] - "
                      f"Using pre-computed true accuracy: {perm_acc:.4f}", flush=True)
            else:
                print(f"\nPermutation {perm_idx} training:", flush=True)
                load_rng_state(base_rng_state)
                perm_acc = train_eval_model_vit(
                    train_images_rot_iter, T_train_permuted,
                    test_images_rot_iter, T_test_permuted,
                    device=device, epochs=args.perm_epochs, lr=1e-4, verbose=True,
                )
            permutation_results[perm_idx - 1].append(perm_acc)
            if (perm_idx % 50 == 0) or (perm_idx == N_PERMUTATIONS):
                print(f"  Completed {perm_idx}/{N_PERMUTATIONS} permutations", flush=True)

    print(f"\n{'='*80}\nALL ITERATIONS COMPLETED\n{'='*80}", flush=True)

    true_accuracies = np.asarray(true_accuracies, dtype=float)
    null_accuracies_all = np.asarray(
        [acc for i in range(N_PERMUTATIONS) for acc in permutation_results[i]], dtype=float)

    p_values = []
    for iter_idx in range(N_ITERATIONS):
        null_iter = np.array(
            [permutation_results[i][iter_idx] for i in range(N_PERMUTATIONS)], dtype=float)
        p_values.append(float(np.mean(null_iter >= true_accuracies[iter_idx])))
    p_values = np.asarray(p_values, dtype=float)

    print("\n" + "=" * 80 + "\nFINAL RESULTS\n" + "=" * 80, flush=True)
    print(f"\nOBSERVED ACCURACY (True T labels, {N_ITERATIONS} iterations):", flush=True)
    print(f"  Mean: {true_accuracies.mean():.4f} ({true_accuracies.mean()*100:.2f}%)", flush=True)
    print(f"  Range: [{true_accuracies.min():.4f}, {true_accuracies.max():.4f}]", flush=True)
    print(f"\nNULL DISTRIBUTION ({len(null_accuracies_all)} permutation results):", flush=True)
    print(f"  Mean: {null_accuracies_all.mean():.4f}  Std: {null_accuracies_all.std():.4f}", flush=True)
    print(f"\nP-VALUES (one-sided, null >= observed):", flush=True)
    print(f"  Per iteration: {[f'{p:.4f}' for p in p_values]}", flush=True)
    print(f"  Mean: {float(np.mean(p_values)):.4f}", flush=True)


if __name__ == "__main__":
    main()
