#!/usr/bin/env python3
"""
Fine-tune the HVH digit model with labeled real crops.

Workflow:
  1. python3 scanner.py <photo> <layout> --save-crops
  2. python3 label.py           # label pending crops interactively
  3. python3 retrain.py         # fine-tune model
  4. python3 scanner.py <photo> <layout>   # verify improved accuracy

The model is backed up to digit_model.pt.bak before overwriting.

Usage:
    python3 retrain.py [--epochs 5] [--oversample 100] [--from-scratch]
"""

import sys
import shutil
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))
from reader import (
    _build_model, _emnist_fix, _elastic, _add_noise,
    _mixup, _CrossbarSevenDataset, MODEL_PATH, MNIST_DIR,
    NUM_CLASSES, BLANK_CLASS,
)

LABELED_DIR = Path(__file__).parent / "labeled_crops"
NORM        = transforms.Normalize((0.1736,), (0.3317,))

# Folders that map to CNN class indices; "point" is skipped (heuristic detection)
_FOLDER_TO_CLASS = {str(d): d for d in range(10)}
_FOLDER_TO_CLASS["blank"] = BLANK_CLASS   # class 10


class SyntheticBlankDataset(Dataset):
    """Pure near-white images to teach the model what an empty box looks like."""

    def __init__(self, n: int = 3000):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        # Uniform white + tiny noise → normalised
        base = random.uniform(0.85, 1.0)
        t = torch.full((1, 28, 28), base) + torch.randn(1, 28, 28) * 0.02
        t = t.clamp(0.0, 1.0)
        t = (t - 0.1736) / 0.3317
        return t, BLANK_CLASS


class RealCropDataset(Dataset):
    """Labeled digit/blank crops from real answer sheets."""

    def __init__(self, labeled_dir: Path, augment: bool = True):
        self.samples: list[tuple[Path, int]] = []
        self.augment = augment

        for label_dir in sorted(labeled_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            cls = _FOLDER_TO_CLASS.get(label_dir.name)
            if cls is None:
                continue   # skip "point", "pending", etc.
            for png in sorted(label_dir.glob("*.png")):
                self.samples.append((png, cls))

        if not self.samples:
            raise ValueError(f"No labeled crops found in {labeled_dir}. "
                             "Run label.py first.")

    def _to_tensor(self, img_bgr: np.ndarray) -> torch.Tensor:
        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        resized = cv2.resize(inv, (28, 28), interpolation=cv2.INTER_AREA)
        t = torch.tensor(resized / 255.0, dtype=torch.float32).unsqueeze(0)
        return (t - 0.1736) / 0.3317

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torchvision.transforms.functional as TF
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None or img.size == 0:
            return torch.zeros(1, 28, 28), label

        t = self._to_tensor(img)

        if self.augment:
            # Strong augmentation to extract maximum diversity from few samples
            angle = random.uniform(-18, 18)
            t = TF.rotate(t, angle, fill=t.min().item())
            if random.random() < 0.6:
                t = _elastic(t)
            t = _add_noise(t)

        return t, label


def _print_crop_accuracy(model: nn.Module, real_ds: RealCropDataset) -> None:
    """Print model accuracy on every labeled crop (no augmentation)."""
    model.eval()
    from reader import CONFIDENCE_THRESHOLD
    import torch.nn.functional as F

    correct = wrong = uncertain = 0
    errors: list[str] = []

    eval_ds = RealCropDataset(LABELED_DIR, augment=False)
    with torch.no_grad():
        for path, true_label in eval_ds.samples:
            img = cv2.imread(str(path))
            if img is None:
                continue
            t = eval_ds._to_tensor(img).unsqueeze(0)
            probs = F.softmax(model(t), dim=1)[0]
            pred  = probs.argmax().item()
            conf  = probs[pred].item()
            _names = {**{d: str(d) for d in range(10)}, BLANK_CLASS: "blank"}
            true_name = _names.get(true_label, str(true_label))
            pred_name = _names.get(pred, str(pred))
            if conf < CONFIDENCE_THRESHOLD and true_label != BLANK_CLASS:
                uncertain += 1
                errors.append(f"  UNCERTAIN  true={true_name} pred={pred_name}@{conf:.0%}  {path.name}")
            elif pred == true_label:
                correct += 1
            else:
                wrong += 1
                errors.append(f"  WRONG      true={true_name} pred={pred_name}@{conf:.0%}  {path.name}")

    total = correct + wrong + uncertain
    print(f"\nReal-crop accuracy: {correct}/{total} correct, "
          f"{wrong} wrong, {uncertain} uncertain")
    for e in errors:
        print(e)


def retrain(epochs: int = 20, oversample: int = 100,
            from_scratch: bool = False) -> None:

    if not LABELED_DIR.exists():
        sys.exit(f"Directory not found: {LABELED_DIR}\nRun label.py first.")

    # ── Load real crops ──────────────────────────────────────────────────────
    print("Loading labeled crops…")
    try:
        real_ds = RealCropDataset(LABELED_DIR, augment=True)
    except ValueError as e:
        sys.exit(str(e))

    counts: dict[int, int] = {}
    for _, lbl in real_ds.samples:
        counts[lbl] = counts.get(lbl, 0) + 1
    label_names = {**{d: str(d) for d in range(10)}, BLANK_CLASS: "blank"}
    for lbl in sorted(counts):
        print(f"  {label_names.get(lbl, lbl):6s}: {counts[lbl]} sample(s)")
    print(f"  total : {len(real_ds.samples)}")

    # ── Load MNIST + EMNIST ──────────────────────────────────────────────────
    print("\nLoading MNIST + EMNIST…")
    train_aug = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.12, 0.12),
                                scale=(0.85, 1.15), shear=6),
        transforms.ColorJitter(brightness=0.4),
        transforms.ToTensor(),
        NORM,
        transforms.Lambda(_add_noise),
    ])
    train_tf_emnist = transforms.Compose([
        transforms.Lambda(_emnist_fix),
        *train_aug.transforms,
    ])
    val_tf_mnist  = transforms.Compose([transforms.ToTensor(), NORM])
    val_tf_emnist = transforms.Compose([transforms.Lambda(_emnist_fix),
                                         transforms.ToTensor(), NORM])

    mnist_train  = datasets.MNIST(str(MNIST_DIR), train=True,  download=True,
                                   transform=train_aug)
    emnist_train = datasets.EMNIST(str(MNIST_DIR), split="digits", train=True,
                                    download=True, transform=train_tf_emnist)
    mnist_val    = datasets.MNIST(str(MNIST_DIR), train=False, download=True,
                                   transform=val_tf_mnist)
    emnist_val   = datasets.EMNIST(str(MNIST_DIR), split="digits", train=False,
                                    download=True, transform=val_tf_emnist)

    # Oversample real crops: repeat indices so same image is augmented differently
    real_n     = len(real_ds)
    ov_indices = list(range(real_n)) * oversample
    real_over  = torch.utils.data.Subset(real_ds, ov_indices)
    synth_blank = SyntheticBlankDataset(n=3000)
    print(f"  Oversampling real crops {oversample}× → {len(real_over)} virtual samples")
    print(f"  Synthetic blank samples : {len(synth_blank)}")

    combined = _CrossbarSevenDataset(
        ConcatDataset([mnist_train, emnist_train, real_over, synth_blank])
    )
    train_loader = DataLoader(combined, batch_size=256, shuffle=True, num_workers=2)
    val_loader   = DataLoader(ConcatDataset([mnist_val, emnist_val]),
                              batch_size=512, shuffle=False)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # ── Model ────────────────────────────────────────────────────────────────
    model = _build_model().to(device)
    if from_scratch:
        print("Training from scratch…")
        max_lr = 3e-3
    else:
        if MODEL_PATH.exists():
            state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            # Migrate old 10-class checkpoint to 11-class if needed
            w_key, b_key = "head.2.weight", "head.2.bias"
            if state[w_key].shape[0] == 10:
                print("  Migrating 10-class → 11-class (adding blank output)…")
                state[w_key] = torch.cat([state[w_key], torch.zeros(1, 256)], dim=0)
                state[b_key] = torch.cat([state[b_key], torch.tensor([-5.0])],  dim=0)
            model.load_state_dict(state)
            print("Loaded existing weights — fine-tuning…")
        else:
            print("No existing model found — training from scratch…")
        max_lr = 5e-4

    # Back up current model before overwriting
    if MODEL_PATH.exists():
        backup = MODEL_PATH.with_suffix(".pt.bak")
        shutil.copy(MODEL_PATH, backup)
        print(f"Backed up → {backup.name}")

    opt   = optim.AdamW(model.parameters(), lr=max_lr / 10, weight_decay=1e-4)
    sched = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\nTraining {epochs} epoch(s) over "
          f"~{len(combined):,} samples per epoch…")
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            Xm, ya, yb, lam = _mixup(X, y)
            opt.zero_grad()
            out  = model(Xm)
            loss = lam * loss_fn(out, ya) + (1 - lam) * loss_fn(out, yb)
            loss.backward()
            opt.step()
            sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                correct += (model(X).argmax(1) == y).sum().item()
                total   += len(y)
        acc = correct / total
        marker = " ✓ best" if acc > best_acc else ""
        print(f"  Epoch {epoch+1:2d}/{epochs}   MNIST+EMNIST val_acc={acc:.4f}{marker}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.cpu().state_dict(), MODEL_PATH)
            model.to(device)

    print(f"\nBest MNIST+EMNIST val acc : {best_acc:.4f}")
    print(f"Model saved               → {MODEL_PATH}")

    # ── Verify on real crops ─────────────────────────────────────────────────
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu",
                                      weights_only=True))
    _print_crop_accuracy(model, real_ds)

    print("\nDone. Run scanner.py on your test images to verify.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Fine-tune HVH digit model with real crops")
    p.add_argument("--epochs",     type=int,  default=20,
                   help="Training epochs (default 5)")
    p.add_argument("--oversample", type=int,  default=100,
                   help="Repeat real crops this many times per epoch (default 100)")
    p.add_argument("--from-scratch", action="store_true",
                   help="Ignore existing weights, retrain fully from scratch")
    args = p.parse_args()
    retrain(args.epochs, args.oversample, args.from_scratch)
