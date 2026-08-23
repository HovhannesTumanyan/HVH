#!/usr/bin/env python3
"""
HVH Scanner — locates all answer boxes in a photo of a filled answer sheet.

Algorithm
---------
1. Detect the 4 corner QR codes (each encodes <CORNER><sheet_id>, e.g. "TLxxxx").
2. Use their known physical positions (mm on A4) to compute a homography that
   perspective-corrects the photo into a canonical top-down A4 view.
3. Map every MCQ checkbox and numeric digit box (from the layout JSON) onto
   the corrected image via simple mm → pixel scaling.
4. Save an annotated image and print pixel coordinates to stdout.

Usage
-----
    python scanner.py <photo.jpg> <layout.json> [--scale 10] [--out annotated.jpg]

    <layout.json> is produced automatically by answer_sheet_gen.generate().

Dependencies
------------
    pip install opencv-python numpy pyzbar
    # On Ubuntu you may also need:  sudo apt install libzbar0
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from pyzbar import pyzbar
except (ImportError, Exception) as _e:
    sys.exit(
        f"pyzbar unavailable ({_e}).\n"
        "Install: sudo apt install libzbar0 && pip install pyzbar"
    )

# ── Physical constants (must match answer_sheet_gen.py) ──────────────────────
A4_W_MM   = 210.0
A4_H_MM   = 297.0
QR_EDGE   =   6.0   # mm from paper edge to QR corner
QR_SIZE   =  22.0   # mm — QR code square side

# Center of each corner QR code in mm (origin: top-left of page, y downward)
QR_CENTERS_MM: dict[str, tuple[float, float]] = {
    "TL": (QR_EDGE + QR_SIZE / 2,            QR_EDGE + QR_SIZE / 2),
    "TR": (A4_W_MM - QR_EDGE - QR_SIZE / 2,  QR_EDGE + QR_SIZE / 2),
    "BL": (QR_EDGE + QR_SIZE / 2,            A4_H_MM - QR_EDGE - QR_SIZE / 2),
    "BR": (A4_W_MM - QR_EDGE - QR_SIZE / 2,  A4_H_MM - QR_EDGE - QR_SIZE / 2),
}

DEFAULT_SCALE = 10.0   # output pixels per mm  → 2100 × 2970 px for A4


# ── QR detection ─────────────────────────────────────────────────────────────

def detect_qr_corners(img: np.ndarray) -> dict[str, tuple[float, float]]:
    """
    Decode QR codes from *img* and return a dict mapping corner label
    ('TL', 'TR', 'BL', 'BR') → (cx_px, cy_px) centre in image pixels.

    Tries grayscale first; falls back to a contrast-enhanced version if
    fewer than 4 codes are found.
    """
    def _decode(image):
        found = {}
        for obj in pyzbar.decode(image):
            data = obj.data.decode("utf-8", errors="ignore")
            label = data[:2] if len(data) >= 2 else ""
            if label in QR_CENTERS_MM:
                pts = np.array([[p.x, p.y] for p in obj.polygon], dtype=np.float64)
                cx, cy = pts.mean(axis=0)
                found[label] = (float(cx), float(cy))
        return found

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    found = _decode(gray)

    if len(found) < 4:
        # Boost local contrast (helps with glare / shadow on photos)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        found.update(_decode(enhanced))   # merge — keeps already-found entries

    return found


# ── Homography & warping ─────────────────────────────────────────────────────

def compute_warp(
    corners_px: dict[str, tuple[float, float]],
    scale: float = DEFAULT_SCALE,
) -> np.ndarray:
    """
    Build a 3×3 homography matrix that maps original image pixels directly
    to the warped output image pixels (scale px/mm).

    H_warp = S @ H_px_to_mm,  where S = diag(scale, scale, 1).
    """
    src, dst = [], []
    for label, (cx, cy) in corners_px.items():
        src.append([cx, cy])
        dst.append(list(QR_CENTERS_MM[label]))

    H_to_mm, _ = cv2.findHomography(
        np.array(src, dtype=np.float32),
        np.array(dst, dtype=np.float32),
    )
    S = np.diag([scale, scale, 1.0])
    return S @ H_to_mm


def warp_image(img: np.ndarray, H: np.ndarray, scale: float = DEFAULT_SCALE) -> np.ndarray:
    out_w = int(round(A4_W_MM * scale))
    out_h = int(round(A4_H_MM * scale))
    return cv2.warpPerspective(img, H, (out_w, out_h))


# ── Annotation ───────────────────────────────────────────────────────────────

def annotate_boxes(warped: np.ndarray, layout: dict, scale: float) -> np.ndarray:
    out = warped.copy()

    for b in layout.get("mcq_boxes", []):
        x0 = int(round(b["x_mm"] * scale))
        y0 = int(round(b["y_mm"] * scale))
        x1 = int(round((b["x_mm"] + b["w_mm"]) * scale))
        y1 = int(round((b["y_mm"] + b["h_mm"]) * scale))
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 50), 1)

    for b in layout.get("num_boxes", []):
        x0 = int(round(b["x_mm"] * scale))
        y0 = int(round(b["y_mm"] * scale))
        x1 = int(round((b["x_mm"] + b["w_mm"]) * scale))
        y1 = int(round((b["y_mm"] + b["h_mm"]) * scale))
        color = (50, 50, 240) if b.get("is_decimal") else (240, 120, 30)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 1)

    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="HVH answer-sheet scanner")
    parser.add_argument("photo",   help="Path to the photo of the answer sheet")
    parser.add_argument("layout",  help="Path to the layout JSON from answer_sheet_gen")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE,
                        help=f"Output pixels per mm (default {DEFAULT_SCALE})")
    parser.add_argument("--out",   default=None,
                        help="Output annotated image path (default: <photo>_annotated.jpg)")
    args = parser.parse_args()

    img = cv2.imread(args.photo)
    if img is None:
        sys.exit(f"Cannot load image: {args.photo}")

    with open(args.layout) as f:
        layout = json.load(f)

    # 1. Detect corner QR codes
    corners = detect_qr_corners(img)
    print(f"Detected QR corners: {sorted(corners.keys())}")
    if len(corners) < 4:
        missing = set(QR_CENTERS_MM) - set(corners)
        print(f"WARNING: missing corners {missing} — homography may be inaccurate.")
        if len(corners) < 2:
            sys.exit("Too few QR codes detected — check lighting and photo focus.")

    # 2. Compute homography and warp
    H = compute_warp(corners, args.scale)
    warped = warp_image(img, H, args.scale)

    # 3. Annotate
    annotated = annotate_boxes(warped, layout, args.scale)

    out_path = args.out or Path(args.photo).stem + "_annotated.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"Annotated image saved: {out_path}")

    # 4. Print coordinates
    scale = args.scale
    print(f"\n{'─'*60}")
    print("MCQ boxes  (green in image)  — warped-image pixel coordinates")
    print(f"{'─'*60}")
    for b in layout.get("mcq_boxes", []):
        x0 = int(round(b["x_mm"] * scale))
        y0 = int(round(b["y_mm"] * scale))
        w  = int(round(b["w_mm"] * scale))
        h  = int(round(b["h_mm"] * scale))
        print(f"  Q{b['q']:02d} opt={b['opt']}  top-left=({x0:4d},{y0:4d})  size={w}×{h}")

    print(f"\n{'─'*60}")
    print("Numeric boxes  (orange=digit, blue=decimal)  — warped-image pixels")
    print(f"{'─'*60}")
    for b in layout.get("num_boxes", []):
        x0   = int(round(b["x_mm"] * scale))
        y0   = int(round(b["y_mm"] * scale))
        w    = int(round(b["w_mm"] * scale))
        h    = int(round(b["h_mm"] * scale))
        kind = "decimal" if b.get("is_decimal") else f"digit{b['digit']}"
        print(f"  Q{b['q']:02d} {kind:8s}  top-left=({x0:4d},{y0:4d})  size={w}×{h}")


if __name__ == "__main__":
    main()
