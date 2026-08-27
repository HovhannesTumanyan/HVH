#!/usr/bin/env python3
"""
HVH Box Reader

CheckboxReader  — CV-based: Otsu threshold + dark-pixel density
DigitReader     — DL-based: small CNN trained on MNIST, weights cached locally

First run downloads MNIST and trains the model (~2 min on CPU).
Subsequent runs load from digit_model.pt in the same directory.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2

MODEL_PATH   = Path(__file__).parent / "digit_model.pt"
MNIST_DIR    = Path(__file__).parent / "mnist_data"

FILLED_THRESHOLD = 0.40   # relative darkness for the winning box (0=white, 1=black)
FILLED_MARGIN    = 0.20   # winner must exceed second-best by at least this much
BLANK_THRESHOLD  = 0.04   # below this → digit box is empty


# ── Region extraction ─────────────────────────────────────────────────────────

def extract_region(
    warped: np.ndarray,
    x_mm: float, y_mm: float, w_mm: float, h_mm: float,
    scale: float,
    inner_frac: float = 0.20,
) -> np.ndarray:
    """Crop a box from the warped image, shrinking by inner_frac to avoid the border."""
    x0 = int(round(x_mm * scale))
    y0 = int(round(y_mm * scale))
    x1 = int(round((x_mm + w_mm) * scale))
    y1 = int(round((y_mm + h_mm) * scale))
    pw = max(1, int((x1 - x0) * inner_frac))
    ph = max(1, int((y1 - y0) * inner_frac))
    crop = warped[y0 + ph: y1 - ph, x0 + pw: x1 - pw]
    if crop.size == 0:
        return np.full((20, 20, 3), 255, dtype=np.uint8)
    return crop


def _to_gray(region: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region


# ── Checkbox detection ────────────────────────────────────────────────────────

def sample_page_white(warped: np.ndarray) -> float:
    """
    Estimate the page white level from a blank area of the warped sheet.
    Uses the 90th-percentile intensity of the top-centre strip (avoids QR corners).
    """
    gray = _to_gray(warped)
    h, w = gray.shape
    strip = gray[int(h * 0.05):int(h * 0.18), int(w * 0.10):int(w * 0.45)]
    white = float(np.percentile(strip, 90))
    return max(white, 150.0)   # guard against sampling a dark area


def relative_darkness(region: np.ndarray, page_white: float) -> float:
    """
    How dark is this region compared to the blank page?
    Returns 0.0 for pure page-white, 1.0 for pure black.
    Uses mean intensity — robust against Otsu splitting uniform gray crops.
    """
    gray = _to_gray(region)
    mean_int = float(np.mean(gray))
    return max(0.0, 1.0 - (mean_int / page_white))


# Legacy alias kept so scanner.py annotation code still compiles
def dark_pixel_ratio(region: np.ndarray, page_white: float = 200.0) -> float:
    return relative_darkness(region, page_white)


def is_filled(region: np.ndarray, page_white: float = 200.0) -> bool:
    return relative_darkness(region, page_white) > FILLED_THRESHOLD


# ── Digit model ───────────────────────────────────────────────────────────────
#
# Target: ≥99.9% on isolated handwritten digits.
#
# Strategy:
#   • ResNet-style CNN with residual shortcuts (deeper without vanishing grads)
#   • Train on MNIST + EMNIST Digits combined (~350k samples)
#   • Elastic distortion + strong affine augmentation (best for handwriting)
#   • MixUp training (reduces overconfident errors)
#   • OneCycleLR scheduler, 20 epochs, save best checkpoint
#   • Test-Time Augmentation (TTA): average 8 predictions at inference
#   • Confidence threshold: uncertain → "?" rather than wrong guess

CONFIDENCE_THRESHOLD = 0.55   # below this → return "?" (flag for review)
TTA_N = 8                      # number of augmented predictions to average

def _build_model():
    import torch.nn as nn

    class _ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
            )
            self.relu = nn.ReLU(inplace=True)
        def forward(self, x):
            return self.relu(self.net(x) + x)

    class _DigitNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            )                                              # 28×28 → 28×28
            self.block1 = nn.Sequential(_ResBlock(64), _ResBlock(64))
            self.down1  = nn.Sequential(
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            )                                              # 28×28 → 14×14
            self.block2 = nn.Sequential(_ResBlock(128), _ResBlock(128))
            self.down2  = nn.Sequential(
                nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            )                                              # 14×14 → 7×7
            self.block3 = nn.Sequential(_ResBlock(256), _ResBlock(256))
            self.pool   = nn.AdaptiveAvgPool2d(1)
            self.head   = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.4),
                nn.Linear(256, 10),
            )
        def forward(self, x):
            x = self.stem(x)
            x = self.block1(x); x = self.down1(x)
            x = self.block2(x); x = self.down2(x)
            x = self.block3(x)
            return self.head(self.pool(x))

    return _DigitNet()


def _emnist_fix(img):
    """EMNIST images are transposed vs MNIST — rotate+flip to correct."""
    from PIL import Image as _I
    return img.rotate(-90).transpose(_I.FLIP_LEFT_RIGHT)


def _elastic(tensor):
    """Random elastic distortion — most effective augmentation for handwriting."""
    import torch
    from scipy.ndimage import map_coordinates, gaussian_filter
    img = tensor.squeeze().numpy()
    h, w = img.shape
    sigma, alpha = 4.0, 20.0
    dx = gaussian_filter(np.random.randn(h, w), sigma) * alpha
    dy = gaussian_filter(np.random.randn(h, w), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    coords = [np.clip(y + dy, 0, h-1), np.clip(x + dx, 0, w-1)]
    distorted = map_coordinates(img, coords, order=1).reshape(h, w)
    return torch.tensor(distorted, dtype=torch.float32).unsqueeze(0)


def _mixup(X, y, alpha=0.2):
    import torch
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(X.size(0))
    return lam * X + (1 - lam) * X[idx], y, y[idx], lam


def _add_noise(tensor):
    """Gaussian noise augmentation to simulate photo/scan grain."""
    import torch
    noise = torch.randn_like(tensor) * 0.15
    return (tensor + noise).clamp(-3.0, 3.0)


def _train_and_save() -> None:
    import torch
    from torch.utils.data import DataLoader, ConcatDataset
    from torchvision import datasets, transforms
    import torch.optim as optim

    EPOCHS = 2
    print(f"Training digit model on MNIST + EMNIST (~350k samples, {EPOCHS} epochs)…")
    print("Augmentations: rotation±10°, brightness, noise, affine, elastic.\n")

    NORM = transforms.Normalize((0.1736,), (0.3317,))

    # Photo-realistic augmentations: rotation, brightness, noise simulate
    # scanned/photographed answer sheets under varying lighting.
    train_aug = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.12, 0.12),
                                scale=(0.85, 1.15), shear=6),
        transforms.ColorJitter(brightness=0.4),   # lighting variation
        transforms.ToTensor(),
        NORM,
        transforms.Lambda(_add_noise),            # sensor/scan noise
    ])
    train_tf_mnist  = train_aug
    train_tf_emnist = transforms.Compose([
        transforms.Lambda(_emnist_fix),
        *train_aug.transforms,
    ])
    val_tf_emnist = transforms.Compose([transforms.Lambda(_emnist_fix),
                                        transforms.ToTensor(), NORM])
    val_tf_mnist  = transforms.Compose([transforms.ToTensor(), NORM])

    mnist_train  = datasets.MNIST(str(MNIST_DIR),  train=True,  download=True, transform=train_tf_mnist)
    emnist_train = datasets.EMNIST(str(MNIST_DIR), split="digits", train=True,  download=True, transform=train_tf_emnist)
    emnist_val   = datasets.EMNIST(str(MNIST_DIR), split="digits", train=False, download=True, transform=val_tf_emnist)
    mnist_val    = datasets.MNIST(str(MNIST_DIR),  train=False, download=True, transform=val_tf_mnist)

    train_loader = DataLoader(ConcatDataset([mnist_train, emnist_train]),
                              batch_size=256, shuffle=True, num_workers=2)
    val_loader   = DataLoader(ConcatDataset([mnist_val, emnist_val]),
                              batch_size=512, shuffle=False)

    model   = _build_model()
    opt     = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched   = optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3,
                                             epochs=EPOCHS,
                                             steps_per_epoch=len(train_loader))
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for X, y in train_loader:
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
                correct += (model(X).argmax(1) == y).sum().item()
                total   += len(y)
        acc = correct / total
        marker = " ✓ best" if acc > best_acc else ""
        print(f"  Epoch {epoch + 1:2d}/{EPOCHS}   val_acc={acc:.4f}{marker}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nBest val acc: {best_acc:.4f} — saved → {MODEL_PATH}")


_model_cache = None

def load_digit_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    import torch
    model = _build_model()
    if not MODEL_PATH.exists():
        _train_and_save()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu",
                                      weights_only=True))
    model.eval()
    _model_cache = model
    return model


def _preprocess(inv: np.ndarray) -> "torch.Tensor":
    """Convert an inverted-binary 28×28 uint8 array to a normalised tensor."""
    import torch
    t = torch.tensor(inv / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return (t - 0.1736) / 0.3317


DIGIT_BLANK_REL = 0.08   # relative-darkness below this → digit box is empty

def read_digit(region: np.ndarray, model,
               page_white: float = 200.0) -> tuple[str, float]:
    """
    Predict the handwritten digit in *region* using TTA.
    Returns:
      ("",  0.0)  — box is empty
      ("?", conf) — model uncertain (conf < CONFIDENCE_THRESHOLD)
      ("3", 0.97) — predicted digit with confidence
    """
    import torch, torch.nn.functional as F
    import torchvision.transforms.functional as TF

    # Relative-darkness check first: avoids Otsu splitting a light/empty box
    # into ~50% "dark" pixels which the CNN then misclassifies as "8".
    if relative_darkness(region, page_white) < DIGIT_BLANK_REL:
        return "", 0.0

    gray   = _to_gray(region)
    _, inv = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    if (inv > 0).mean() < BLANK_THRESHOLD:
        return "", 0.0

    base = cv2.resize(inv, (28, 28), interpolation=cv2.INTER_AREA)
    t0   = _preprocess(base)

    # Build TTA batch: identity + rotations + slight shifts
    angles     = [-8, -4, 0, 4, 8]
    translates = [(-0.05, 0), (0.05, 0), (0, 0)]
    variants   = [t0]
    for ang in angles[:TTA_N - 1]:
        aug = TF.rotate(t0, ang, fill=t0.min().item())
        variants.append(aug)

    batch  = torch.cat(variants[:TTA_N], dim=0)   # (TTA_N, 1, 28, 28)
    with torch.no_grad():
        avg_probs = F.softmax(model(batch), dim=1).mean(dim=0)

    d    = avg_probs.argmax().item()
    conf = avg_probs[d].item()
    if conf < CONFIDENCE_THRESHOLD:
        return "?", conf
    return str(d), float(conf)


# ── High-level box readers ────────────────────────────────────────────────────

def read_all_boxes(
    warped: np.ndarray,
    layout: dict,
    scale: float,
    digit_model=None,
) -> dict:
    """
    Read every box in *layout* and return a structured result dict:

        {
          "student_id": "12345678",
          "answers": {
            "Q01": {"type": "mcq", "answer": "b",
                    "options": {"a": False, "b": True, "c": False, "d": False}},
            "Q21": {"type": "numeric", "digits": ["1","2","","4"],
                    "value": "1 24"},  # blanks shown as space
          }
        }
    """
    # ── Sample page-white level for relative darkness calculation ────────────
    page_white = sample_page_white(warped)

    # ── Student ID ───────────────────────────────────────────────────────────
    id_digits = []
    for b in sorted(layout.get("id_boxes", []), key=lambda x: x["digit"]):
        region = extract_region(warped, b["x_mm"], b["y_mm"],
                                b["w_mm"], b["h_mm"], scale)
        if digit_model is not None:
            d, _ = read_digit(region, digit_model, page_white)
            id_digits.append(d)
        else:
            id_digits.append("?" if relative_darkness(region, page_white) > BLANK_THRESHOLD else "")
    student_id = "".join(id_digits)

    # ── MCQ answers ──────────────────────────────────────────────────────────
    # Collect relative-darkness per option, then decide:
    #   • max_dark < FILLED_THRESHOLD → all empty (no answer)
    #   • max_dark >= FILLED_THRESHOLD AND gap vs second >= FILLED_MARGIN → one answer
    #   • Otherwise → ambiguous ("?")
    mcq_ratios: dict[int, dict[str, float]] = {}
    for b in layout.get("mcq_boxes", []):
        q = b["q"]
        mcq_ratios.setdefault(q, {})
        region = extract_region(warped, b["x_mm"], b["y_mm"],
                                b["w_mm"], b["h_mm"], scale)
        mcq_ratios[q][b["opt"]] = relative_darkness(region, page_white)

    mcq_by_q: dict[int, dict] = {}
    for q, ratios in mcq_ratios.items():
        sorted_ratios = sorted(ratios.values(), reverse=True)
        max_r  = sorted_ratios[0]
        sec_r  = sorted_ratios[1] if len(sorted_ratios) > 1 else 0.0
        if max_r < FILLED_THRESHOLD:
            # Nothing filled
            filled = {o: False for o in ratios}
            answer = ""
        elif max_r - sec_r >= FILLED_MARGIN:
            # One clear winner
            best_opt = max(ratios, key=ratios.get)
            filled   = {o: (o == best_opt) for o in ratios}
            answer   = best_opt
        else:
            # Two or more boxes too close → ambiguous
            filled = {o: (r >= FILLED_THRESHOLD) for o, r in ratios.items()}
            answer = "?"
        mcq_by_q[q] = {"filled": filled, "answer": answer, "ratios": ratios}

    # ── Numeric answers ──────────────────────────────────────────────────────
    # Group by question; separate decimal boxes
    num_by_q: dict[int, dict] = {}
    for b in layout.get("num_boxes", []):
        q = b["q"]
        num_by_q.setdefault(q, {"digits": {}, "decimal_pos": None})
        region = extract_region(warped, b["x_mm"], b["y_mm"],
                                b["w_mm"], b["h_mm"], scale)
        if b.get("is_decimal"):
            num_by_q[q]["decimal_pos"] = b.get("digit", 0)
        else:
            pos = b["digit"]   # 1-based digit position within the question
            if digit_model is not None:
                d, conf = read_digit(region, digit_model, page_white)
            else:
                ratio = relative_darkness(region, page_white)
                d = "?" if ratio > BLANK_THRESHOLD else ""
                conf = ratio
            num_by_q[q]["digits"][pos] = d

    # ── Assemble answers ─────────────────────────────────────────────────────
    answers: dict[str, dict] = {}

    for q, info in sorted(mcq_by_q.items()):
        answers[f"Q{q:02d}"] = {
            "type": "mcq",
            "answer": info["answer"],
            "options": info["filled"],
        }

    for q, info in sorted(num_by_q.items()):
        digs   = info["digits"]
        sorted_digits = [digs.get(i, "") for i in sorted(digs)]
        dp     = info["decimal_pos"]
        if dp is not None:
            # Insert decimal point between digit at decimal_pos and decimal_pos+1
            parts = sorted_digits[:dp] + ["."] + sorted_digits[dp:]
            value = "".join(parts)
        else:
            value = "".join(sorted_digits)
        answers[f"Q{q:02d}"] = {
            "type": "numeric",
            "digits": sorted_digits,
            "value": value,
        }

    return {"student_id": student_id, "answers": answers}
