import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Import project modules
from config import STRAINS, ATCC_GROUPINGS, antibiotics

def set_seed(seed=123):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(123)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = "./data"
print(f"Device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Import project modules
from config import STRAINS, ATCC_GROUPINGS, antibiotics

def set_seed(seed=123):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(123)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = "./data"
print(f"Device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")

# Load the dataset
X_ref = np.load(os.path.join(DATA_DIR, "X_reference.npy"))
y_ref = np.load(os.path.join(DATA_DIR, "y_reference.npy")).astype(int)
X_te = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_te = np.load(os.path.join(DATA_DIR, "y_test.npy")).astype(int)

print("Dataset shapes:")
print(f"X_reference: {X_ref.shape}")
print(f"y_reference: {y_ref.shape}")
print(f"X_test: {X_te.shape}")
print(f"y_test: {y_te.shape}")

# Check strain distribution
print(f"\nStrain distribution in reference set:")
strain_counts = Counter(y_ref)
for strain_id, count in sorted(strain_counts.items()):
    print(f"Strain {strain_id} ({STRAINS[strain_id]}): {count} samples")

def get_binary_strains():
    """Get strain IDs that belong to penicillin or Meropenem classes"""
    binary_strains = []
    for strain_id in range(30):  # All possible strain IDs
        antibiotic_id = ATCC_GROUPINGS[strain_id]
        if antibiotic_id == 0 or antibiotic_id == 5:  # Meropenem or Peniciline
            binary_strains.append(strain_id)
    return sorted(binary_strains)

# Get focused strains for detailed mapping
focused_strains = get_binary_strains()
print("\nTRUE labels: strain -> antibiotic mapping")

for sid in focused_strains:
    true_ab = ATCC_GROUPINGS[sid]
    sname = STRAINS[sid]
    print(f"{sid:2d} | {sname:<20} : [{true_ab}] {antibiotics[true_ab]}")

# Get the focused strains (penicillin and meropenem only)
focused_strains = get_binary_strains()

# Separate into penicillin and meropenem groups
penicillin_strains = [s for s in focused_strains if ATCC_GROUPINGS[s] == 5]
meropenem_strains = [s for s in focused_strains if ATCC_GROUPINGS[s] == 0]

# Create reordering indices - penicillin first, then meropenem
reorder_indices = []
for strain_id in penicillin_strains:
    # Find all indices where y_te equals this strain_id
    strain_indices = np.where(y_te == strain_id)[0]
    reorder_indices.extend(strain_indices)

for strain_id in meropenem_strains:
    # Find all indices where y_te equals this strain_id
    strain_indices = np.where(y_te == strain_id)[0]
    reorder_indices.extend(strain_indices)

# Reorder the data according to focused strains
X_te_reordered = X_te[reorder_indices]
y_te_reordered = y_te[reorder_indices]

# Get antibiotic labels for the reordered data
y_te_ab_reordered = np.array([ATCC_GROUPINGS[strain_id] for strain_id in y_te_reordered])

# # Plot the focused spectra as a heatmap
# plt.figure(figsize=(15, 8))
# plt.imshow(X_te_reordered, cmap='viridis', aspect='auto', interpolation='nearest')
# plt.title('Penicillin and Meropenem Test Spectra - Grouped by Antibiotic Class')
# plt.xlabel('Wavenumber Index (1000 points)')
# plt.ylabel('Spectrum Index - Penicillin first, then Meropenem')

# Add y-axis labels showing strain and antibiotic info
yticks = np.arange(0, len(y_te_ab_reordered), 100)
ytick_labels = []
for i in yticks:
    strain_id = y_te_reordered[i]
    ab_id = y_te_ab_reordered[i]
    strain_name = STRAINS[strain_id]
    ab_name = antibiotics[ab_id]
    ytick_labels.append(f'{i}: {strain_id} ({strain_name}) - {ab_name}')

# plt.yticks(yticks, ytick_labels, rotation=0, fontsize=8)

# plt.colorbar(label='Intensity')
# plt.tight_layout()
# plt.show()

# Print the reordering information
print("Focused strains (Penicillin and Meropenem only):")
print("Penicillin strains =", penicillin_strains)
print("Meropenem strains =", meropenem_strains)
print(f"Total spectra in focused set: {len(reorder_indices)}")
print("First 10 strain IDs in new order:", y_te_reordered[:10])
print("First 10 antibiotic classes in new order:", y_te_ab_reordered[:10])

# Show strain distribution
print("\nStrain distribution in focused test set:")
from collections import Counter
strain_counts = Counter(y_te_reordered)
for strain_id, count in sorted(strain_counts.items()):
    ab_id = ATCC_GROUPINGS[strain_id]
    print(f"Strain {strain_id} ({STRAINS[strain_id]}) - {antibiotics[ab_id]}: {count} samples")

STRAIN_ORDER = np.arange(30)  
sid_ref_all = np.repeat(STRAIN_ORDER, 2000)
sid_te_all  = np.repeat(STRAIN_ORDER, 100)

focused_strains = get_binary_strains()
ref_mask = np.isin(sid_ref_all, focused_strains)
te_mask  = np.isin(sid_te_all,  focused_strains)

sid_ref_filtered = sid_ref_all[ref_mask]
sid_te_filtered  = sid_te_all[te_mask]


X_ref_filtered = X_ref[ref_mask]
X_te_filtered  = X_te[te_mask]

CLASS_TO_BIN = {0: 0, 5: 1}
y_ref_filtered = np.array([CLASS_TO_BIN[ATCC_GROUPINGS[s]] for s in sid_ref_filtered], dtype=int)
y_te_filtered  = np.array([CLASS_TO_BIN[ATCC_GROUPINGS[s]] for s in sid_te_filtered],  dtype=int)

import torch
import torch.nn as nn
import torch.nn.functional as F


class RamanNetResidualBlock(nn.Module):
    """
    Residual block for RamanNet architecture as described in the paper.
    Each residual layer contains a shortcut connection between input and output
    of one convolutional layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(RamanNetResidualBlock, self).__init__()
        
        # Single convolutional layer as per paper description
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        # FIXED: Use out = out + self.shortcut(x) instead of += to avoid in-place operation
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class BinaryClassifier(nn.Module):
    """
    BinaryClassifier model following the exact RamanNet architecture described in the paper.
    
    Architecture:
    1. Initial convolution layer (kernel size 7)
    2. Two residual layers (kernel sizes 7 and 3)
    3. Flatten layer
    4. Final fully connected classification layer (2 classes for binary classification)
    
    All convolution kernel numbers are set to 1 as specified in the paper.
    Uses kaiming normal initialization.
    """
    def __init__(self, input_dim=1000, n_classes=2):
        super(BinaryClassifier, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        # Initial convolution layer (kernel size 7, 1 channel)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(1)
        
        # First residual layer (kernel size 7)
        self.residual1 = RamanNetResidualBlock(1, 1, kernel_size=7)
        
        # Second residual layer (kernel size 3)
        self.residual2 = RamanNetResidualBlock(1, 1, kernel_size=3)
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Final fully connected classification layer
        self.classifier = nn.Linear(input_dim, n_classes)
        
        # Initialize weights using kaiming normal initialization
        self._initialize_weights()

    def forward(self, x):
        # Initial convolution layer
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Two residual layers
        x = self.residual1(x)
        x = self.residual2(x)
        
        # Flatten
        x = self.flatten(x)
        
        # Final classification layer
        x = self.classifier(x)
        
        return x

    def _initialize_weights(self):
        """
        Initialize weights using kaiming normal initialization as specified in the paper.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

def prepare_data(X, y, batch_size=32, shuffle=True):
    """Convert numpy arrays to DataLoader with shape [B,1,L]."""
    X_tensor = torch.as_tensor(X, dtype=torch.float32).unsqueeze(1)  # [N,1,L]
    y_tensor = torch.as_tensor(y, dtype=torch.long)
    dataset  = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # sample-weighted loss to avoid bias from a short last batch
        bs = target.size(0)
        total_loss += loss.detach().item() * bs
        pred = output.detach().argmax(dim=1)
        correct += (pred == target).sum().item()
        total += bs

    avg_loss = total_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc

# Evaluation function
def evaluate(model, test_loader, criterion, device):
    model.eval()
    assert not model.training, "Model must be in eval() during evaluation."

    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            bs = target.size(0)
            total_loss += loss.detach().item() * bs
            pred = output.detach().argmax(dim=1)

            correct += (pred == target).sum().item()
            total += bs
            all_preds.extend(pred.cpu().tolist())
            all_targets.extend(target.cpu().tolist())

    avg_loss = total_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc, all_preds, all_targets

def train_eval_permutation_with_epochs(
    X_train, y_train, X_test, y_test,
    epochs=10, lr=5e-5, batch_size=256, weight_decay=1e-2
):
    
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    
    # Infer spectrum length from data
    input_dim = X_train.shape[1]
    assert X_test.shape[1] == input_dim, "Train/Test spectrum length mismatch."

    # Model (RamanNet) + loaders
    model = BinaryClassifier(input_dim=input_dim, n_classes=2).to(DEVICE)
    train_loader = prepare_data(X_train, y_train, batch_size=batch_size, shuffle=True)
    test_loader  = prepare_data(X_test,  y_test,  batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_test_acc = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        running_correct, running_total, running_loss = 0, 0, 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)  # [B,1,L]
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item() * xb.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == yb).sum().item()
            running_total += xb.size(0)

        train_acc = 100.0 * running_correct / max(1, running_total)

        # ---- Eval ----
        model.eval()
        test_correct, test_total, test_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                logits = model(xb)
                loss = criterion(logits, yb)

                test_loss_sum += float(loss) * xb.size(0)
                preds = logits.argmax(dim=1)
                test_correct += (preds == yb).sum().item()
                test_total += xb.size(0)

        test_acc = 100.0 * test_correct / max(1, test_total)
        print(f"    Epoch {epoch}: Train Acc = {train_acc:.2f}%, Test Acc = {test_acc:.2f}%")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"[BEST] Epoch {best_epoch}: Test Acc = {best_test_acc:.2f}%")
    return best_test_acc / 100.0

# --- binary class mapping (Meropenem=0 -> 0, Penicillin=5 -> 1)
CLASS_TO_BIN = {0: 0, 5: 1}
BIN_TO_CLASS = {0: 0, 1: 5}

LAST_PERM_MAP = None

def _make_strain_perm_map(focused_strains, rng):
    """
    Build ONE strain->binary map per permutation, preserving
    the number of Meropenem (0) and Penicillin (5) strains.
    """
    n_mer = sum(ATCC_GROUPINGS[s] == 0 for s in focused_strains)  # Meropenem
    n_pen = sum(ATCC_GROUPINGS[s] == 5 for s in focused_strains)  # Penicillin

    # build a label list of the correct total length, then shuffle
    perm_bins = np.array([CLASS_TO_BIN[0]] * n_mer + [CLASS_TO_BIN[5]] * n_pen, dtype=int)
    rng.shuffle(perm_bins)

    # safety
    assert len(perm_bins) == len(focused_strains), \
        f"perm_bins({len(perm_bins)}) != focused_strains({len(focused_strains)})"

    # assign one permuted binary label per strain
    return {s: perm_bins[i] for i, s in enumerate(focused_strains)}


def print_permutation_mapping(perm_idx, y_ref_perm, y_te_perm):
    print(f"\nPermutation {perm_idx}: strain -> antibiotic (TRUE -> PERMUTED)")
    if LAST_PERM_MAP is None:
        print("  [warn] LAST_PERM_MAP is not set")
        return
    for sid in focused_strains:
        true_ab = ATCC_GROUPINGS[sid]
        perm_bin = LAST_PERM_MAP[sid]
        perm_ab  = BIN_TO_CLASS[perm_bin]
        sname = STRAINS[sid]
        print(f"{sid:2d} | {sname:<20} : [{true_ab}] {antibiotics[true_ab]}  ->  [{perm_ab}] {antibiotics[perm_ab]}")


# Permutation test parameters
PERM_EPOCHS = 10    # Epochs for each permutation
RANDOM_SEED = 42

from itertools import combinations

def generate_all_unique_permutations(focused_strains):
  # get the number of strains for each binary class based on the true grouping
  n_meropenem = sum(ATCC_GROUPINGS[s] == 0 for s in focused_strains)
  n_penicillin = sum(ATCC_GROUPINGS[s] == 5 for s in focused_strains)

  # we need to choose which strains will be assigned to the 'Meropenem' (binary 0) class in the permutation.
  # The remaining strains will be assigned to the 'Penicillin' (binary 1) class.
  # The number of strains assigned to each class must match the original counts.
  meropenem_combinations = list(combinations(focused_strains, n_meropenem))

  unique_permutations = []
  for meropenem_strains in meropenem_combinations:
    penicillin_strains = [s for s in focused_strains if s not in meropenem_strains]

    # Create strain-to-label mapping for this permutation
    strain_to_label = {}
    for strain in meropenem_strains:
      strain_to_label[strain] = CLASS_TO_BIN[0]  # Meropenem (True class 0) maps to Binary 0
    for strain in penicillin_strains:
      strain_to_label[strain] = CLASS_TO_BIN[5]  # Penicillin (True class 5) maps to Binary 1

    unique_permutations.append(strain_to_label)

  return unique_permutations

all_unique_permutations = generate_all_unique_permutations(focused_strains)
print(f"Number of unique permutations generated: {len(all_unique_permutations)}")

# Do true training with epoch-by-epoch progress
print(f"\nTRUE labels training:")
true_accuracy = train_eval_permutation_with_epochs(
    X_ref_filtered, y_ref_filtered,
    X_te_filtered, y_te_filtered,
    epochs=PERM_EPOCHS
)
print(f"TRUE labels final accuracy: {true_accuracy:.2%}")

import os, csv, time, statistics
from pathlib import Path

# === Config ===
RESULTS_CSV = "./perm_results.csv"   # change path/name if you like

# Ensure output dir exists
Path(os.path.dirname(RESULTS_CSV) or ".").mkdir(parents=True, exist_ok=True)

# Detect existing results (for resume) and seed running timing stats
existing_perm_indices = set()
times_so_far = []

if Path(RESULTS_CSV).exists():
    with open(RESULTS_CSV, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # Skip any non-numeric perm_idx rows if the file was manually edited
            try:
                pidx = int(row["perm_idx"])
                existing_perm_indices.add(pidx)
                # elapsed_s might be empty for malformed lines; guard it
                if row.get("elapsed_s", "").strip():
                    times_so_far.append(float(row["elapsed_s"]))
            except Exception:
                continue

def _running_stats(times_list):
    """Return (count, total, min, max, mean) for a list of floats; None if empty."""
    if not times_list:
        return 0, 0.0, None, None, None
    c = len(times_list)
    tot = sum(times_list)
    mn = min(times_list)
    mx = max(times_list)
    mean = tot / c
    return c, tot, mn, mx, mean

# Prepare CSV writer (append mode; write header if new file)
header = [
    "perm_idx", "epochs", "n_train", "n_test",
    "accuracy", "elapsed_s",
    "cumu_time_s", "min_time_s", "max_time_s", "mean_time_s",
    "timestamp"
]
write_header = not Path(RESULTS_CSV).exists()
csv_f = open(RESULTS_CSV, "a", newline="")
writer = csv.DictWriter(csv_f, fieldnames=header)
if write_header:
    writer.writeheader(); csv_f.flush()

# Initial timing aggregates (in case we're resuming)
_, total_so_far, min_so_far, max_so_far, mean_so_far = _running_stats(times_so_far)

# ========== Main loop ==========
N_PERMUTATIONS = len(all_unique_permutations)
null_accuracies = []

print(f"\nRunning {N_PERMUTATIONS} permutations (exhaustive over unique strain→label maps)...")

ref_strains_order = sid_ref_filtered   # one strain ID per row of X_ref_filtered / y_ref_filtered
te_strains_order  = sid_te_filtered    # one strain ID per row of X_te_filtered  / y_te_filtered

for perm_idx, strain_to_label in enumerate(all_unique_permutations, start=1):
    # Resume support: skip if already in the CSV
    if perm_idx in existing_perm_indices:
        # You can still reconstruct aggregated stats at end; but to keep running aggregates
        # accurate during this session, also rebuild running timers from file:
        continue

    # Apply SAME mapping to BOTH splits, by STRAIN ID per row
    y_ref_permuted = np.fromiter((strain_to_label[s] for s in ref_strains_order),
                                 dtype=int, count=len(ref_strains_order))
    y_te_permuted  = np.fromiter((strain_to_label[s] for s in te_strains_order),
                                 dtype=int, count=len(te_strains_order))

    LAST_PERM_MAP = strain_to_label
    print_permutation_mapping(perm_idx, y_ref_permuted, y_te_permuted)

    # ------- Timed train+eval -------
    print(f"\nPermutation {perm_idx}/{N_PERMUTATIONS} training:")
    t0 = time.perf_counter()
    perm_accuracy = train_eval_permutation_with_epochs(
        X_ref_filtered, y_ref_permuted,
        X_te_filtered,  y_te_permuted,
        epochs=PERM_EPOCHS
    )
    t1 = time.perf_counter()
    elapsed = t1 - t0
    # --------------------------------

    null_accuracies.append(perm_accuracy)
    print(f"Permutation {perm_idx} final accuracy: {perm_accuracy:.2%}  (elapsed {elapsed:.2f}s)")

    # Update running aggregates
    times_so_far.append(elapsed)
    _, total_so_far, min_so_far, max_so_far, mean_so_far = _running_stats(times_so_far)

    # Append row to CSV (write after each perm so you never lose results)
    writer.writerow({
        "perm_idx": perm_idx,
        "epochs": int(PERM_EPOCHS),
        "n_train": len(y_ref_permuted),
        "n_test":  len(y_te_permuted),
        "accuracy": f"{perm_accuracy:.6f}",
        "elapsed_s": f"{elapsed:.6f}",
        "cumu_time_s": f"{total_so_far:.6f}",
        "min_time_s":  "" if min_so_far is None else f"{min_so_far:.6f}",
        "max_time_s":  "" if max_so_far is None else f"{max_so_far:.6f}",
        "mean_time_s": "" if mean_so_far is None else f"{mean_so_far:.6f}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    csv_f.flush()

# Close file handle ASAP
csv_f.close()

# If we skipped completed rows due to resume, rebuild final stats from file to be certain
final_times = []
with open(RESULTS_CSV, "r", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        try:
            if row.get("elapsed_s", "").strip():
                final_times.append(float(row["elapsed_s"]))
        except Exception:
            pass

cnt, tot, mn, mx, mean = _running_stats(final_times)

print("\nPermutation test completed!")
print(f"Timing summary across {cnt} permutations:")
if cnt:
    print(f"  Total (aggregated) time: {tot:.2f} s  ({tot/3600:.2f} h)")
    print(f"  Min / Max / Mean time : {mn:.2f} s  / {mx:.2f} s  / {mean:.2f} s")
else:
    print("  No timings recorded.")

# Create histogram visualization for permutation test (Meropenem vs Penicillin)
null_accuracies = np.array(null_accuracies)

# plt.figure(figsize=(7, 5))
# plt.hist(null_accuracies * 100, bins=20, color='navy', alpha=0.7)

# # Observed performance line
# plt.axvline(true_accuracy * 100, color='red', linewidth=2, label='True Performance')

# # Chance baseline for binary classification = 50%
# plt.axvline(50, color='k', linestyle=':', linewidth=1, label='Chance (50%)')

# plt.xlabel("Accuracy [%] (on the test set)")
# plt.ylabel("Frequency")
# plt.title("Permutation null vs observed (Binary: Meropenem vs Penicillin)")

# Add text annotation with observed accuracy and permutation p-value
ymax = max(np.histogram(null_accuracies * 100, bins=20)[0])
p_value = np.mean(null_accuracies >= true_accuracy)

# Position text to avoid legend overlap - move it lower and to the right
# plt.text(true_accuracy * 100 + 2, ymax * 0.6,  # Reduced y position from 0.9 to 0.6
#          f"Observed = {true_accuracy * 100:.2f}%\nP = {p_value:.4f}",
#          fontsize=12, bbox=dict(facecolor='yellow', alpha=0.7))

# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

import math

def sampling_combinations(arr):
    n = len(null_accuracies)
    print(f"Array size: {n}\n")

    print("Without replacement (unordered combinations):")
    for k in range(1, n + 1):
        c = math.comb(n, k)
        print(f"  k={k}: {c}")

    print("\nWith replacement (unordered combinations):")
    for k in range(1, n + 1):
        c = math.comb(n + k - 1, k)
        print(f"  k={k}: {c}")

from itertools import combinations, combinations_with_replacement

def index_combos_by_k(arr):
    """
    Return two tables (lists) of index-combinations for k=1..n:
      - without_rep[k]: all k-combos of indices without replacement
      - with_rep[k]   : all k-combos of indices with replacement (non-decreasing)
    Each table has length n+1 with index 0 unused (None).
    """
    n = len(arr)
    idx = range(n)

    without_rep = [None] * (n + 1)
    with_rep    = [None] * (n + 1)

    for k in range(1, n + 1):
        without_rep[k] = [list(t) for t in combinations(idx, k)]
        with_rep[k]    = [list(t) for t in combinations_with_replacement(idx, k)]

    return without_rep, with_rep

without_rep, with_rep = index_combos_by_k(null_accuracies)

# Example: all 3-index combos without replacement
print("k=2, without replacement:", without_rep[2])   # [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]
# Example: all 3-index combos with replacement
print("k=2, with replacement:", with_rep[2])         # [[0,0,0],[0,0,1],...,[1,3,3],[2,2,2],[2,2,3],[2,3,3],[3,3,3]]

# If you want the actual element combos from indices:
elem_combos_k2 = [[null_accuracies[i] for i in combo] for combo in without_rep[2]]
print("k=2 element combos (no repl.):", elem_combos_k2)

import numpy as np
import matplotlib.pyplot as plt

def pval_stats_by_k(null_accuracies, true_accuracy, combos_by_k, sample_limit=None, rng=None, ddof=1):
    """
    Compute per-k stats of the 'p-value' for each combo:
      p(combo) = (# elements in combo >= true_accuracy) / k

    combos_by_k may be:
      - dict[int, list[tuple[int]]]
      - list/tuple of length n+1 with index 0 unused (None), i.e., combos_by_k[k] for k>=1
    """
    if rng is None:
        rng = np.random.default_rng(0)

    arr = np.asarray(null_accuracies)
    thr = true_accuracy

    ks, ns, pmins, pmaxs, pmeans, pmeds, pstds = [], [], [], [], [], [], []

    def _iter_k_combos(cbk):
        # Accept dict OR 1-indexed list/tuple
        if isinstance(cbk, dict):
            for k in sorted(cbk.keys()):
                yield k, cbk[k]
        else:
            # assume sequence with index 0 unused
            for k in range(1, len(cbk)):
                yield k, cbk[k]

    for k, combos in _iter_k_combos(combos_by_k):
        if combos is None or len(combos) == 0:
            continue

        # Optional uniform downsampling of combos for large cases
        if (sample_limit is not None) and (len(combos) > sample_limit):
            idxs = rng.choice(len(combos), size=sample_limit, replace=False)
            combos = [combos[i] for i in idxs]

        # Vectorized counting
        idx = np.asarray(combos, dtype=int)  # shape (m, k_expected)
        if idx.ndim != 2 or idx.shape[1] != k:
            raise ValueError(f"All combos for k={k} must be sequences of length {k}.")
        if idx.size:  # bounds check
            if idx.min() < 0 or idx.max() >= arr.size:
                raise IndexError(
                    f"Combo indices out of bounds for null_accuracies of length {arr.size}."
                )

        counts = (arr[idx] >= thr).sum(axis=1)
        pvals = counts / float(k)

        ks.append(k)
        ns.append(pvals.size)
        pmins.append(pvals.min())
        pmaxs.append(pvals.max())
        pmeans.append(pvals.mean())
        pmeds.append(np.median(pvals))
        pstds.append(pvals.std(ddof=ddof))

    return {
        "k": np.array(ks, dtype=int),
        "n": np.array(ns, dtype=int),
        "p_min": np.array(pmins, dtype=float),
        "p_max": np.array(pmaxs, dtype=float),
        "p_mean": np.array(pmeans, dtype=float),
        "p_median": np.array(pmeds, dtype=float),
        "p_std": np.array(pstds, dtype=float),  # SD across combos
    }


import numpy as np
import matplotlib.pyplot as plt

def plot_pval_stats(
    flipped=True,
    stats_without=None,
    stats_with=None,
    save_path=None,
    dpi=300,
    band="se",
    fit="tight",          # 'tight' (auto-fit with padding), 'unit' (clamp to [0,1]), or None (matplotlib auto)
    pad_frac=0.05,        # fractional padding around data limits when fit='tight'
    clip_unit=True,       # clamp p-value bands to [0,1] before computing limits
):
    """
    Plot per-k stats; supports side-by-side plots if both stats_* provided.

    band: 'se' = ±1 standard error around mean (default), 'sd' = ±1 standard deviation.
    fit:
      - 'tight' -> compute limits from the data (zoom to fit) with small padding.
      - 'unit'  -> clamp the p-value axis to [0,1] like before.
      - None    -> let matplotlib autoscale (no manual limits).
    """

    if (stats_without is None) and (stats_with is None):
        raise ValueError("Provide at least one of stats_without or stats_with")

    def _apply_fit(ax, x_all, y_all):
        # flatten and drop NaNs
        xv = np.asarray(x_all).ravel()
        yv = np.asarray(y_all).ravel()
        xv = xv[~np.isnan(xv)]
        yv = yv[~np.isnan(yv)]
        if fit is None:
            ax.relim()
            ax.autoscale(enable=True, axis="both", tight=True)
            ax.margins(pad_frac, pad_frac)
            return

        if fit == "tight":
            # pad both axes
            x_min, x_max = (np.min(xv), np.max(xv)) if xv.size else (0, 1)
            y_min, y_max = (np.min(yv), np.max(yv)) if yv.size else (0, 1)
            dx = max(x_max - x_min, 1e-9)
            dy = max(y_max - y_min, 1e-9)
            ax.set_xlim(x_min - pad_frac * dx, x_max + pad_frac * dx)
            ax.set_ylim(y_min - pad_frac * dy, y_max + pad_frac * dy)
        elif fit == "unit":
            # keep p-value axis in [0,1]; other axis gets tight limits with padding
            # Decide which axis is p-value based on orientation
            pass  # handled inline in each orientation below
        else:
            raise ValueError("fit must be 'tight', 'unit', or None")

    def _one(ax, stats, title):
        if flipped:
            # k on x; p on y
            x_k    = np.asarray(stats["k"])
            p_mean = np.asarray(stats["p_mean"])
            p_med  = np.asarray(stats["p_median"])
            p_max  = np.asarray(stats["p_max"])
            std    = np.asarray(stats["p_std"])
            n      = np.maximum(np.asarray(stats["n"]), 1)

            spread = std / np.sqrt(n) if band == "se" else std
            band_label = "±1 SE (mean)" if band == "se" else "±1 SD (across combos)"

            low  = p_mean - spread
            high = p_mean + spread
            if clip_unit:
                low  = np.clip(low,  0.0, 1.0)
                high = np.clip(high, 0.0, 1.0)

            ax.plot(x_k, p_mean, "-", label="mean")
            ax.plot(x_k, p_max, "--", label="max")
            ax.plot(x_k, p_med, "D", label="median")
            ax.fill_between(x_k, low, high, alpha=0.2, label=band_label)
            ax.set_xlabel("k")
            ax.set_ylabel("p-value")
            ax.set_xticks(x_k)

            if fit == "unit":
                # p-axis fixed to [0,1], k fits tightly with half-step padding
                ax.set_ylim(0, 1)
                ax.set_xlim(np.min(x_k) - 0.5, np.max(x_k) + 0.5)
            else:
                # tight fit on both axes
                _apply_fit(ax, x_k, np.concatenate([p_mean, p_med, p_max, low, high]))
                # keep nice tick coverage on discrete k
                ax.set_xlim(np.min(x_k) - 0.5, np.max(x_k) + 0.5)

        else:
            # p on x; k on y  (fixed a bug: previously x/y were swapped here)
            k      = np.asarray(stats["k"])
            p_mean = np.asarray(stats["p_mean"])
            p_med  = np.asarray(stats["p_median"])
            p_max  = np.asarray(stats["p_max"])
            std    = np.asarray(stats["p_std"])
            n      = np.maximum(np.asarray(stats["n"]), 1)

            spread = std / np.sqrt(n) if band == "se" else std
            band_label = "±1 SE (mean)" if band == "se" else "±1 SD (across combos)"

            low  = p_mean - spread
            high = p_mean + spread
            if clip_unit:
                low  = np.clip(low,  0.0, 1.0)
                high = np.clip(high, 0.0, 1.0)

            ax.plot(p_mean, k, "-", label="mean")
            ax.plot(p_max,  k, "--", label="max")
            ax.plot(p_med,  k, "D", label="median")
            ax.fill_betweenx(k, low, high, alpha=0.2, label=band_label)
            ax.set_ylabel("k")
            ax.set_xlabel("p-value")
            ax.set_yticks(k)

            if fit == "unit":
                ax.set_xlim(0, 1)
                ax.set_ylim(np.min(k) - 0.5, np.max(k) + 0.5)
            else:
                _apply_fit(ax, np.concatenate([p_mean, p_med, p_max, low, high]), k)
                ax.set_ylim(np.min(k) - 0.5, np.max(k) + 0.5)

        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    if (stats_without is not None) and (stats_with is not None):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True if flipped else False)
        _one(axes[0], stats_without, "Without replacement")
        _one(axes[1], stats_with, "With replacement")
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        stats = stats_without if stats_without is not None else stats_with
        which = "Without replacement" if stats_without is not None else "With replacement"
        _one(ax, stats, which)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi)
        plt.close()
    else:
        plt.show()

stats_wout = pval_stats_by_k(null_accuracies, true_accuracy, without_rep)
stats_with = pval_stats_by_k(null_accuracies, true_accuracy, with_rep)
plot_pval_stats(stats_without=stats_wout, stats_with=stats_with, save_path="./pvalues_by_k.png", dpi=300)