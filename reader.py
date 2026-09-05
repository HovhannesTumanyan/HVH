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


def enhance_contrast(warped: np.ndarray,
                     clip_limit: float = 2.0,
                     tile_grid: int = 8) -> np.ndarray:
    """Apply CLAHE to the L channel of the warped sheet.

    Boosts local contrast so faint ink becomes clearly darker than the
    surrounding paper — without inventing content in genuinely blank regions.
    Operates in LAB colour space so hue and saturation are untouched.
    """
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_grid, tile_grid))
    lab_enhanced = cv2.merge([clahe.apply(l_ch), a_ch, b_ch])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


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

CONFIDENCE_THRESHOLD = 0.45   # below this → return "?" (flag for review)
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


DIGIT_BLANK_REL = 0.05   # relative-darkness below this → digit box is empty

def _count_loops(inv: np.ndarray) -> int:
    """Count topological holes (closed loops) in a binarized digit image.

    Digits with loops: 0, 6, 8, 9 (0/8 have 1-2, 6/9 have 1).
    Digits without loops: 1, 2, 3, 4, 5, 7 → 0.
    Uses RETR_CCOMP: inner contours (holes) have a parent contour.
    """
    contours, hierarchy = cv2.findContours(inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(hierarchy) == 0:
        return 0
    return sum(1 for h in hierarchy[0] if h[3] >= 0)


def read_digit(region: np.ndarray, model,
               page_white: float = 200.0,
               _crops: list | None = None,
               _force_read: bool = False) -> tuple[str, float, str]:
    """
    Predict the handwritten digit in *region* using TTA.
    Returns:
      ("",  0.0,  reason) — box is empty
      ("?", conf, reason) — model uncertain (conf < CONFIDENCE_THRESHOLD)
      ("3", 0.97, reason) — predicted digit with confidence

    If _crops is a list, appends (raw_crop_bgr, bin28_gray, digit, reason) for visualisation.
    Set _force_read=True to skip the relative-darkness blank gate (caller already pre-checked
    using a CLAHE-enhanced crop; the original unmodified region is passed for CNN input).
    """
    import torch, torch.nn.functional as F
    import torchvision.transforms.functional as TF

    rd = relative_darkness(region, page_white)

    # Relative-darkness check first: avoids Otsu splitting a light/empty box
    # into ~50% "dark" pixels which the CNN then misclassifies as "8".
    if not _force_read and rd < DIGIT_BLANK_REL:
        reason = f"blank(rd={rd:.3f})"
        if _crops is not None:
            _crops.append((region.copy(), None, "", reason))
        return "", 0.0, reason

    gray   = _to_gray(region)
    _, inv = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    pixel_ratio = float((inv > 0).mean())
    if pixel_ratio < BLANK_THRESHOLD:
        reason = f"blank(pix={pixel_ratio:.3f})"
        if _crops is not None:
            _crops.append((region.copy(), None, "", reason))
        return "", 0.0, reason

    # Noise rejection via connected components: salt-and-pepper noise produces
    # many tiny disconnected blobs; a real digit has 1-3 large components.
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]   # skip background (label 0)
    if len(areas) == 0:
        reason = "blank(no blobs)"
        if _crops is not None:
            _crops.append((region.copy(), None, "", reason))
        return "", 0.0, reason
    max_blob = int(areas.max())
    total_on = int((inv > 0).sum())
    n_blobs  = len(areas)
    # Reject if the largest blob is too small OR noise dominates (many blobs, none big)
    if max_blob < 25 or (n_blobs > 12 and max_blob < total_on * 0.30):
        reason = f"noise(blobs={n_blobs},max={max_blob})"
        if _crops is not None:
            _crops.append((region.copy(), None, "", reason))
        return "", 0.0, reason

    base = cv2.resize(inv, (28, 28), interpolation=cv2.INTER_AREA)
    t0   = _preprocess(base)

    # Build TTA batch: identity + rotations + slight shifts
    angles   = [-8, -4, 0, 4, 8]
    variants = [t0]
    for ang in angles[:TTA_N - 1]:
        aug = TF.rotate(t0, ang, fill=t0.min().item())
        variants.append(aug)

    batch  = torch.cat(variants[:TTA_N], dim=0)   # (TTA_N, 1, 28, 28)
    with torch.no_grad():
        avg_probs = F.softmax(model(batch), dim=1).mean(dim=0)

    d    = avg_probs.argmax().item()
    conf = avg_probs[d].item()

    # Topology override: "3" and "5" never have a closed loop; "6" and "9" always do.
    # If the model prefers 3/5 but the image has a topological hole, pick 6 or 9
    # (whichever the model rates higher) instead.  Because topology structurally
    # eliminates 3 and 5, we accept the loop-derived answer at a lower threshold.
    if d in (3, 5):
        n_loops = _count_loops(cv2.resize(inv, (28, 28), interpolation=cv2.INTER_AREA))
        if n_loops >= 1:
            alt = max(6, 9, key=lambda x: avg_probs[x].item())
            alt_p = avg_probs[alt].item()
            if alt_p > 0.05:   # any non-trivial mass on 6/9 is enough given topology
                d, conf = alt, alt_p
                reason = f"rd={rd:.3f},conf={conf:.2f},loop→{d}"
                if _crops is not None:
                    _crops.append((region.copy(), base.copy(), str(d), reason))
                return str(d), float(conf), reason

    reason = f"rd={rd:.3f},conf={conf:.2f}"
    if conf < CONFIDENCE_THRESHOLD:
        reason = f"low_conf({d}@{conf:.2f},rd={rd:.3f})"
        if _crops is not None:
            _crops.append((region.copy(), base.copy(), "?", reason))
        return "?", conf, reason

    if _crops is not None:
        _crops.append((region.copy(), base.copy(), str(d), reason))
    return str(d), float(conf), reason


def build_digit_debug_image(results: dict) -> np.ndarray:
    """Build a BGR grid image showing, per numeric question, each digit box:
    left half = raw crop from warped sheet, right half = 28×28 model input.
    """
    CELL_H   = 56          # height for each crop panel
    CELL_W28 = 56          # width for the 28×28 panel (square)
    PAD      = 6           # gap between panels and between digits
    HDR_H    = 22          # header row height (question label + value)
    Q_W      = 90          # left column: question label
    BG       = (245, 245, 245)
    FONT     = cv2.FONT_HERSHEY_SIMPLEX

    # Collect only numeric questions that have debug crops
    rows_data = []
    for key in sorted(results.get("answers", {})):
        val = results["answers"][key]
        if val.get("type") != "numeric":
            continue
        crops = val.get("_crops", [])
        if not crops:
            continue
        rows_data.append((key, val.get("value") or "(blank)", crops))

    if not rows_data:
        blank = np.full((80, 400, 3), 245, dtype=np.uint8)
        cv2.putText(blank, "no digit crops", (10, 50), FONT, 0.6, (100, 100, 100), 1)
        return blank

    # Compute max row width
    max_n = max(len(crops) for _, _, crops in rows_data)
    cell_w_raw = CELL_H * 2   # raw crop scaled to CELL_H height, assume ~2:1 aspect
    col_w = cell_w_raw + PAD + CELL_W28 + PAD   # one digit column
    img_w = Q_W + max_n * col_w + PAD
    row_h = HDR_H + CELL_H + PAD

    strips = []
    for q_label, value, crops in rows_data:
        strip = np.full((row_h, img_w, 3), BG, dtype=np.uint8)

        # Question header
        hdr = f"{q_label} = {value}"
        cv2.putText(strip, hdr, (4, HDR_H - 5), FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        x = Q_W
        for raw_bgr, bin28, digit, reason in crops:
            # ── Raw crop ────────────────────────────────────────────────────
            rh, rw = raw_bgr.shape[:2]
            scale_r = CELL_H / max(rh, 1)
            new_rw  = max(1, int(rw * scale_r))
            raw_s   = cv2.resize(raw_bgr, (new_rw, CELL_H))
            # draw into strip (clip if wider than cell_w_raw)
            draw_w = min(new_rw, cell_w_raw)
            strip[HDR_H: HDR_H + CELL_H, x: x + draw_w] = raw_s[:, :draw_w]
            # thin border
            cv2.rectangle(strip, (x, HDR_H), (x + draw_w - 1, HDR_H + CELL_H - 1),
                          (180, 180, 180), 1)

            # ── 28×28 model input ────────────────────────────────────────────
            x28 = x + cell_w_raw + PAD
            if bin28 is not None:
                # white-on-black → invert to black-on-white for display
                disp28 = cv2.resize(255 - bin28, (CELL_W28, CELL_H),
                                    interpolation=cv2.INTER_NEAREST)
                disp28_bgr = cv2.cvtColor(disp28, cv2.COLOR_GRAY2BGR)
                strip[HDR_H: HDR_H + CELL_H, x28: x28 + CELL_W28] = disp28_bgr
            else:
                # blank box — grey fill
                strip[HDR_H: HDR_H + CELL_H, x28: x28 + CELL_W28] = (210, 210, 210)
            cv2.rectangle(strip, (x28, HDR_H), (x28 + CELL_W28 - 1, HDR_H + CELL_H - 1),
                          (180, 180, 180), 1)

            # ── Digit label (top of cell pair) ───────────────────────────────
            label_color = (0, 140, 0) if digit not in ("", "?") else \
                          (0, 0, 180) if digit == "?" else (120, 120, 120)
            label = digit if digit not in ("",) else "_"
            cv2.putText(strip, label, (x + 2, HDR_H - 3),
                        FONT, 0.55, label_color, 1, cv2.LINE_AA)

            x += col_w

        strips.append(strip)

    # Separator line between questions
    sep = np.full((2, img_w, 3), 200, dtype=np.uint8)
    out_rows = []
    for s in strips:
        out_rows.append(s)
        out_rows.append(sep)
    return np.vstack(out_rows[:-1])   # drop trailing separator


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
    # ── Boost local contrast for blank detection only ───────────────────────
    # CLAHE-enhanced image is used solely for relative_darkness / page_white
    # so faint digits pass the blank threshold.  The original warped image is
    # kept for the CNN crops: Otsu handles local normalisation there, and the
    # model was trained on MNIST-like data closer to the unmodified appearance.
    warped_enh = enhance_contrast(warped)

    # ── Sample page-white level for relative darkness calculation ────────────
    page_white = sample_page_white(warped_enh)

    # ── Student ID ───────────────────────────────────────────────────────────
    id_digits = []
    for b in sorted(layout.get("id_boxes", []), key=lambda x: x["digit"]):
        enh_region  = extract_region(warped_enh, b["x_mm"], b["y_mm"],
                                     b["w_mm"], b["h_mm"], scale)
        orig_region = extract_region(warped, b["x_mm"], b["y_mm"],
                                     b["w_mm"], b["h_mm"], scale)
        if digit_model is not None:
            enh_rd = relative_darkness(enh_region, page_white)
            force = enh_rd >= DIGIT_BLANK_REL
            d, _, _r = read_digit(orig_region, digit_model, page_white,
                                   _force_read=force)
            id_digits.append(d)
        else:
            id_digits.append("?" if relative_darkness(enh_region, page_white) > BLANK_THRESHOLD else "")
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
        region = extract_region(warped_enh, b["x_mm"], b["y_mm"],
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
    _seen_decimal: dict[int, bool] = {}  # q → True once pre-printed decimal box is passed
    for b in layout.get("num_boxes", []):
        q   = b["q"]
        pos = b["digit"]
        num_by_q.setdefault(q, {"digits": {}, "decimal_pos": None, "debug": [], "_crops": []})

        student_dec   = num_by_q[q]["decimal_pos"]           # set if student dot found
        pre_dec_digit = _dec_box[q]["digit"] if q in _dec_box else None

        if b.get("is_decimal"):
            # ── Pre-printed decimal indicator box ────────────────────────────
            _seen_decimal[q] = True
            if student_dec is None:
                # No student dot found yet → use pre-printed position as decimal
                num_by_q[q]["decimal_pos"] = pos
                num_by_q[q]["debug"].append(f"d{pos}=.(pre-printed)")
            else:
                # Student already wrote their decimal earlier.
                # This box may also contain a digit the student wrote over the
                # pre-printed dot.  Try to read it; include it if confident.
                plain_enh  = extract_region(warped_enh, b["x_mm"], b["y_mm"],
                                            b["w_mm"], b["h_mm"], scale, inner_frac=0.05)
                plain_orig = extract_region(warped, b["x_mm"], b["y_mm"],
                                            b["w_mm"], b["h_mm"], scale, inner_frac=0.05)
                enh_rd = relative_darkness(plain_enh, page_white)
                if digit_model is not None:
                    force = enh_rd >= DIGIT_BLANK_REL
                    d, conf, reason = read_digit(plain_orig, digit_model, page_white,
                                                 _crops=num_by_q[q]["_crops"],
                                                 _force_read=force)
                else:
                    rd = relative_darkness(plain_enh, page_white)
                    d = "?" if rd > BLANK_THRESHOLD else ""
                    conf = rd
                    reason = f"rd={rd:.3f}"
                if d and d != "?":
                    num_by_q[q]["digits"][pos] = d
                num_by_q[q]["debug"].append(f"d{pos}=.(+{d or 'skip'},{reason})")

        else:
            # ── Regular digit box ────────────────────────────────────────────
            # Check for a student-written decimal dot before the pre-printed
            # indicator, but only in questions that have a decimal structure.
            if not _seen_decimal.get(q, False) and q in _dec_box:
                dot_crop = extract_region(warped_enh, b["x_mm"], b["y_mm"],
                                          b["w_mm"], b["h_mm"], scale, inner_frac=0.05)
                if is_decimal_dot(dot_crop, page_white):
                    if student_dec is None:
                        num_by_q[q]["decimal_pos"] = pos
                    num_by_q[q]["debug"].append(f"d{pos}=.(student-dot)")
                    continue  # don't add this box to digits dict

            # Left-extension: widen the crop leftward so digits written near
            # the left edge (especially the first post-decimal digit) aren't
            # clipped.  When the student wrote their own dot before the
            # pre-printed decimal, don't extend — the student's digits land
            # naturally in their boxes without the offset seen in normal writing.
            post_dec_digit = _post_decimal.get(q)
            if pos == post_dec_digit:
                if (student_dec is not None
                        and pre_dec_digit is not None
                        and student_dec < pre_dec_digit):
                    left_ext = 1.0          # student decimal earlier → no offset
                else:
                    dec = _dec_box.get(q)
                    if dec is not None:
                        dec_region = extract_region(warped, dec["x_mm"], dec["y_mm"],
                                                    dec["w_mm"], dec["h_mm"],
                                                    scale, inner_frac=0.05)
                        dec_rd = relative_darkness(dec_region, page_white)
                        if dec_rd > 0.20:
                            # Student wrote ink in the decimal box — keep away
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
            # Pre-check with enhanced image; feed original to CNN
            enh_region  = extract_region(warped_enh, x_mm, b["y_mm"],
                                         w_mm, b["h_mm"], scale, inner_frac=0.05)
            orig_region = extract_region(warped, x_mm, b["y_mm"],
                                         w_mm, b["h_mm"], scale, inner_frac=0.05)
            enh_rd = relative_darkness(enh_region, page_white)
            if digit_model is not None:
                force = enh_rd >= DIGIT_BLANK_REL
                d, conf, reason = read_digit(orig_region, digit_model, page_white,
                                             _crops=num_by_q[q]["_crops"],
                                             _force_read=force)
            else:
                ratio = relative_darkness(enh_region, page_white)
                d = "?" if ratio > BLANK_THRESHOLD else ""
                conf = ratio
                reason = f"rd={ratio:.3f}"
            num_by_q[q]["digits"][pos] = d
            num_by_q[q]["debug"].append(f"d{pos}={d or '_'}({reason})")

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
            "debug": info.get("debug", []),
            "_crops": info.get("_crops", []),
        }

    return {"student_id": student_id, "answers": answers}
