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


def is_decimal_dot(region: np.ndarray, page_white: float) -> bool:
    """Return True if the region likely contains a handwritten decimal point.

    A decimal dot is a small, roughly circular blob occupying less than 40%
    of the crop in each dimension — clearly smaller than any real digit.
    """
    if relative_darkness(region, page_white) < 0.05:
        return False
    gray = _to_gray(region)
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return False
    h, w = inv.shape
    max_idx = int(areas.argmax()) + 1
    bw = stats[max_idx, cv2.CC_STAT_WIDTH]
    bh = stats[max_idx, cv2.CC_STAT_HEIGHT]
    frac_h = bh / max(h, 1)
    frac_w = bw / max(w, 1)
    aspect = bw / max(bh, 1)
    return frac_h < 0.40 and frac_w < 0.40 and 0.4 < aspect < 2.5


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


class _CrossbarSevenDataset:
    """
    Wraps a digit dataset and randomly adds a horizontal crossbar to '7'
    samples (European-style 7̶), teaching the model that 7-with-crossbar = 7.
    """
    def __init__(self, dataset, prob=0.40):
        self.dataset = dataset
        self.prob    = prob

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        import torch, random
        x, y = self.dataset[idx]
        if y == 7 and random.random() < self.prob:
            x = x.clone()
            _, h, w = x.shape
            row = h // 2 + random.randint(-3, 2)
            row = max(1, min(h - 2, row))
            # Foreground value in normalised MNIST space ≈ (1.0 - 0.1736)/0.3317
            fg = (1.0 - 0.1736) / 0.3317
            x[0, row, w // 4 : 3 * w // 4] = fg
        return x, y


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

    combined_train = _CrossbarSevenDataset(ConcatDataset([mnist_train, emnist_train]))
    train_loader = DataLoader(combined_train,
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

    # Noise rejection via connected components: salt-and-pepper noise produces
    # many tiny disconnected blobs; a real digit has 1-3 large components.
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]   # skip background (label 0)
    if len(areas) == 0:
        return "", 0.0
    max_blob = int(areas.max())
    total_on = int((inv > 0).sum())
    n_blobs  = len(areas)
    # Reject if the largest blob is too small OR noise dominates (many blobs, none big)
    if max_blob < 25 or (n_blobs > 12 and max_blob < total_on * 0.30):
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
    # Track which digit position immediately follows the decimal indicator per Q,
    # so we can apply a larger left extension there (students write the first
    # post-decimal digit close to the printed dot, not at the box edge).
    _post_decimal: dict[int, int] = {}   # q → digit index of first post-decimal box
    _dec_box: dict[int, dict] = {}       # q → decimal indicator box metadata
    prev_was_decimal: dict[int, bool] = {}
    for b in layout.get("num_boxes", []):
        q = b["q"]
        if b.get("is_decimal"):
            prev_was_decimal[q] = True
            _dec_box[q] = b
        elif prev_was_decimal.pop(q, False):
            _post_decimal[q] = b["digit"]

    num_by_q: dict[int, dict] = {}
    _seen_decimal: dict[int, bool] = {}  # q → True once pre-printed decimal box passed
    for b in layout.get("num_boxes", []):
        q = b["q"]
        num_by_q.setdefault(q, {"digits": {}, "decimal_pos": None})
        # Expand left edge so digits written near box borders aren't clipped.
        # Post-decimal first box needs a larger extension (~4 mm) because the
        # decimal-indicator box is full-width but the printed dot is centred,
        # leaving ~3-4 mm of dead space before the student's digit starts.
        # Exception: if the student wrote their own ink in the decimal indicator
        # box (e.g. Q27 where we patch is_decimal=True on a filled box), limit
        # extension to the physical gap between the boxes so we don't pull in
        # the student's handwritten dot and confuse the CNN.
        post_dec_digit = _post_decimal.get(q)
        if not b.get("is_decimal") and b["digit"] == post_dec_digit:
            dec = _dec_box.get(q)
            if dec is not None:
                dec_region = extract_region(warped, dec["x_mm"], dec["y_mm"],
                                            dec["w_mm"], dec["h_mm"], scale, inner_frac=0.05)
                dec_rd = relative_darkness(dec_region, page_white)
                if dec_rd > 0.20:
                    # Student wrote ink here — cap extension at the box gap
                    gap = b["x_mm"] - (dec["x_mm"] + dec["w_mm"])
                    left_ext = max(0.0, gap)
                else:
                    left_ext = 4.0
            else:
                left_ext = 4.0
        else:
            left_ext = 1.0
        x_mm = b["x_mm"] - left_ext
        w_mm = b["w_mm"] + left_ext
        region = extract_region(warped, x_mm, b["y_mm"],
                                w_mm, b["h_mm"], scale, inner_frac=0.05)
        if b.get("is_decimal"):
            _seen_decimal[q] = True
            # Only accept the pre-printed decimal pos if the student hasn't
            # already marked a decimal dot in an earlier integer-part box.
            if num_by_q[q]["decimal_pos"] is None:
                num_by_q[q]["decimal_pos"] = b.get("digit", 0)
        else:
            pos = b["digit"]
            # Before the pre-printed decimal indicator, check whether the student
            # wrote their own decimal dot in this box rather than a digit.
            # Only do this for questions that have a decimal structure at all.
            if not _seen_decimal.get(q, False) and q in _dec_box:
                dot_region = extract_region(warped, b["x_mm"], b["y_mm"],
                                            b["w_mm"], b["h_mm"], scale, inner_frac=0.05)
                if is_decimal_dot(dot_region, page_white):
                    if num_by_q[q]["decimal_pos"] is None:
                        num_by_q[q]["decimal_pos"] = pos
                    continue  # don't add this box to digits dict
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
            int_part  = "".join(sorted_digits[:dp])
            frac_part = "".join(sorted_digits[dp:])
            # Only include decimal dot if there is at least one digit on each side
            if int_part and frac_part:
                value = int_part + "." + frac_part
            elif int_part:
                value = int_part          # no fractional digits written
            elif frac_part:
                value = frac_part         # no integer digits written (leading dot suppressed)
            else:
                value = ""
        else:
            value = "".join(sorted_digits)
        answers[f"Q{q:02d}"] = {
            "type": "numeric",
            "digits": sorted_digits,
            "value": value,
        }

    return {"student_id": student_id, "answers": answers}
