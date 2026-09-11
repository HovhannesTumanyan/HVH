#!/usr/bin/env python3
"""
Interactive digit labeler for HVH.

Usage:
    python3 label.py                     # label all pending crops
    python3 label.py --relabel 4         # re-examine already-labeled "4" crops
    python3 scanner.py IMG.jpg layout.json --save-crops   # generate crops first

For each crop:
  0–9  : assign digit label  (→ labeled_crops/<digit>/)
  b    : blank / empty box   (→ labeled_crops/blank/)
  p    : decimal point       (→ labeled_crops/point/)
  s    : skip (leave in pending)
  d    : delete (noise — not worth keeping)
  q    : quit

Labeled crops go into labeled_crops/<label>/.
Run retrain.py afterwards to bake them into the model.
"""

import sys
import shutil
from pathlib import Path

import cv2
import numpy as np

LABELED_BASE = Path("labeled_crops")
PENDING_DIR  = LABELED_BASE / "pending"
DISPLAY_SIZE = 160   # px per panel


def _build_display(img_bgr: np.ndarray) -> np.ndarray:
    """Return a 3-panel BGR image: original | CLAHE | 28×28 model input."""
    S = DISPLAY_SIZE
    if img_bgr is None or img_bgr.size == 0:
        blank = np.full((S, S * 3 + 8, 3), 180, dtype=np.uint8)
        cv2.putText(blank, "missing", (4, S // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (80, 80, 80), 1)
        return blank

    orig = cv2.resize(img_bgr, (S, S))

    lab   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh   = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    enh   = cv2.resize(enh, (S, S))

    gray  = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    bin28 = cv2.resize(inv, (S, S), interpolation=cv2.INTER_NEAREST)
    white_digit = cv2.cvtColor(255 - bin28, cv2.COLOR_GRAY2BGR)

    sep = np.full((S, 4, 3), 160, dtype=np.uint8)
    return np.hstack([orig, sep, enh, sep, white_digit])


def _canvas(display: np.ndarray, info_lines: list[str], hdr_h: int = 80) -> np.ndarray:
    """Add a text header above the display strip."""
    w = display.shape[1]
    canvas = np.full((display.shape[0] + hdr_h, w, 3), 28, dtype=np.uint8)
    canvas[hdr_h:] = display

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    for i, line in enumerate(info_lines):
        cv2.putText(canvas, line, (8, 20 + i * 22),
                    FONT, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    # Column headers
    S = DISPLAY_SIZE
    for txt, hx in [("original", 0), ("CLAHE", S + 4), ("28×28", 2*(S+4))]:
        cv2.putText(canvas, txt, (hx + 4, hdr_h - 6),
                    FONT, 0.38, (120, 180, 120), 1, cv2.LINE_AA)
    return canvas


def _label_batch(files: list[Path], title_prefix: str) -> int:
    """Show each file, collect label from keypress. Returns number labeled."""
    total   = len(files)
    labeled = 0

    for idx, fpath in enumerate(files):
        img = cv2.imread(str(fpath))

        display  = _build_display(img)
        stem     = fpath.stem
        # Extract predicted digit from filename if present
        parts    = stem.split("_pred")
        pred_str = parts[1].split("_")[0] if len(parts) > 1 else "?"

        info = [
            f"[{idx+1}/{total}]  {stem}",
            f"Model predicted: {pred_str}   |   0-9 digit  b blank  p point  s skip  d delete  q quit",
        ]
        frame = _canvas(display, info)

        cv2.imshow(title_prefix, frame)
        cv2.setWindowTitle(title_prefix, f"HVH Labeler — {idx+1}/{total}")

        while True:
            k = cv2.waitKey(0) & 0xFF
            if ord('0') <= k <= ord('9'):
                folder = chr(k)
            elif k == ord('b'):
                folder = "blank"
            elif k == ord('p'):
                folder = "point"
            else:
                folder = None

            if folder is not None:
                dst = LABELED_BASE / folder
                dst.mkdir(parents=True, exist_ok=True)
                shutil.move(str(fpath), str(dst / fpath.name))
                print(f"  {fpath.name}  →  {folder}/")
                labeled += 1
                break
            elif k == ord('s'):
                print(f"  {fpath.name}  →  skipped")
                break
            elif k == ord('d'):
                fpath.unlink()
                print(f"  {fpath.name}  →  deleted")
                break
            elif k in (ord('q'), 27):   # q or Esc
                print("\nQuit.")
                cv2.destroyAllWindows()
                return labeled

    cv2.destroyAllWindows()
    return labeled


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="HVH interactive digit labeler")
    p.add_argument("--relabel", metavar="DIGIT",
                   help="Re-examine already-labeled crops for this digit (0-9)")
    args = p.parse_args()

    if args.relabel is not None:
        if args.relabel not in "0123456789":
            sys.exit("--relabel must be a single digit 0-9")
        src = LABELED_BASE / args.relabel
        if not src.exists() or not list(src.glob("*.png")):
            sys.exit(f"No labeled crops found for digit {args.relabel} in {src}/")
        files = sorted(src.glob("*.png"))
        print(f"Re-examining {len(files)} crops labeled as '{args.relabel}'.")
        print("Press the correct digit to re-label, 's' to keep as-is.\n")
        # For relabeling: 's' keeps in current dir, digit moves to new dir
        labeled = _label_batch(files, "HVH Relabeler")
        print(f"\nDone. Re-labeled {labeled} crop(s).")
        return

    # Default: label pending crops
    if not PENDING_DIR.exists() or not list(PENDING_DIR.glob("*.png")):
        print(f"No pending crops found in {PENDING_DIR}/")
        print("Generate crops with:\n  python3 scanner.py <photo> <layout.json> --save-crops")
        sys.exit(0)

    files = sorted(PENDING_DIR.glob("*.png"))
    print(f"Found {len(files)} pending crops.")
    print("Keys: 0-9 digit  |  b blank  |  p decimal point  |  s skip  |  d delete (noise)  |  q quit\n")

    labeled = _label_batch(files, "HVH Labeler")

    remaining = len(list(PENDING_DIR.glob("*.png")))
    print(f"\nLabeled {labeled} crop(s). {remaining} still pending.")
    if labeled > 0:
        print("Run retrain.py to update the model.")


if __name__ == "__main__":
    main()
