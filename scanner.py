#!/usr/bin/env python3
"""
HVH Scanner — locates, reads, and annotates all answer boxes in a photo.

Algorithm
---------
1. Detect the 4 corner QR codes; estimate missing one if only 3 found.
2. Compute homography → perspective-correct photo to canonical A4 view.
3. Read each box from the corrected image:
     • MCQ checkboxes  — Otsu threshold + dark-pixel density
     • Digit / ID boxes — CNN trained on MNIST (auto-downloads on first run)
4. Back-project box centres to original photo → draw coloured dots.
5. Write results: annotated photo, warped overview, coords .txt, answers .json.

Usage
-----
    python scanner.py <photo.jpg> <layout.json> [--scale 10]
                      [--out annotated.jpg] [--coords coords.txt]
                      [--results results.json] [--no-read]

    <layout.json> is produced automatically by answer_sheet_gen.generate().

Dependencies
------------
    pip install opencv-python numpy pyzbar torch torchvision
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

    def _merge(found, extra, scale=1.0):
        for label, (cx, cy) in extra.items():
            if label not in found:
                found[label] = (cx * scale, cy * scale)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    found = _decode(gray)

    if len(found) < 4:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        _merge(found, _decode(clahe.apply(gray)))

    # Downscale passes — high-res photos can make QR modules too small for zbar
    if len(found) < 4:
        for scale_factor in (0.5, 0.25):
            small = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor,
                               interpolation=cv2.INTER_AREA)
            _merge(found, _decode(small), scale=1.0 / scale_factor)
            if len(found) == 4:
                break

    # Adaptive threshold — helps with uneven lighting / shadows
    if len(found) < 4:
        for block, C in [(31, 10), (51, 15), (21, 5)]:
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block, C)
            _merge(found, _decode(thresh))
            if len(found) == 4:
                break

    return found


def estimate_missing_corners(
    found: dict[str, tuple[float, float]]
) -> dict[str, tuple[float, float]]:
    """
    If exactly 3 corners are detected, estimate the 4th via the parallelogram
    rule: TL + BR = TR + BL (diagonals share the same midpoint on a rectangle).
    """
    all_corners = {"TL", "TR", "BL", "BR"}
    missing = all_corners - set(found)
    if len(missing) != 1:
        return found

    label = next(iter(missing))
    partners = {"TL": ("TR", "BL", "BR"),
                "TR": ("TL", "BR", "BL"),
                "BL": ("TL", "BR", "TR"),
                "BR": ("TR", "BL", "TL")}
    a, b, diag = partners[label]
    mx = found[a][0] + found[b][0] - found[diag][0]
    my = found[a][1] + found[b][1] - found[diag][1]

    # Sanity check — estimated corner must be within a reasonable image extent
    all_xs = [p[0] for p in found.values()]
    all_ys = [p[1] for p in found.values()]
    margin = max(max(all_xs) - min(all_xs), max(all_ys) - min(all_ys)) * 0.3
    if mx < min(all_xs) - margin or mx > max(all_xs) + margin or \
       my < min(all_ys) - margin or my > max(all_ys) + margin:
        print(f"  WARNING: estimated {label} at ({mx:.0f},{my:.0f}) looks wrong — "
              f"try retaking the photo with all 4 corners visible.")
        return found   # return only 3 corners; caller will error cleanly

    result = dict(found)
    result[label] = (mx, my)
    print(f"  Estimated missing corner {label} at ({mx:.0f}, {my:.0f}) via parallelogram rule.")
    return result


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


def back_project(mm_x: float, mm_y: float, H_inv: np.ndarray, scale: float) -> tuple[int, int]:
    """Map a box-centre (mm_x, mm_y) in warped-image space back to original photo pixels."""
    wp = np.array([mm_x * scale, mm_y * scale, 1.0])
    op = H_inv @ wp
    return int(round(op[0] / op[2])), int(round(op[1] / op[2]))


# ── Coordinate collection ─────────────────────────────────────────────────────

def collect_centres(layout: dict) -> list[dict]:
    """Return list of dicts with box metadata + centre in mm."""
    centres = []
    for b in layout.get("id_boxes", []):
        centres.append({
            "label": f"ID_digit{b['digit']}",
            "kind": "id",
            "cx_mm": b["x_mm"] + b["w_mm"] / 2,
            "cy_mm": b["y_mm"] + b["h_mm"] / 2,
        })
    for b in layout.get("mcq_boxes", []):
        centres.append({
            "label": f"Q{b['q']:02d}_mcq_{b['opt']}",
            "q": b["q"], "kind": "mcq", "opt": b["opt"],
            "cx_mm": b["x_mm"] + b["w_mm"] / 2,
            "cy_mm": b["y_mm"] + b["h_mm"] / 2,
        })
    for b in layout.get("num_boxes", []):
        kind = "decimal" if b.get("is_decimal") else f"digit{b['digit']}"
        centres.append({
            "label": f"Q{b['q']:02d}_num_{kind}",
            "q": b["q"], "kind": kind,
            "cx_mm": b["x_mm"] + b["w_mm"] / 2,
            "cy_mm": b["y_mm"] + b["h_mm"] / 2,
        })
    return centres


# ── Annotation ───────────────────────────────────────────────────────────────

_DOT_COLORS = {
    "id":      (0, 220, 255),   # yellow  — student ID boxes
    "mcq":     (0, 0, 255),     # red     — MCQ checkboxes
    "decimal": (255, 80, 0),    # blue    — decimal point boxes
}

def annotate_original(
    img: np.ndarray,
    centres: list[dict],
    H_inv: np.ndarray,
    scale: float,
    dot_r: int = 6,
) -> np.ndarray:
    """Draw coloured dots at each box centre projected onto the original photo."""
    out = img.copy()
    for entry in centres:
        kind  = entry["kind"]
        color = _DOT_COLORS.get(kind, (0, 0, 255))   # default red for numeric digits
        px, py = back_project(entry["cx_mm"], entry["cy_mm"], H_inv, scale)
        cv2.circle(out, (px, py), dot_r, color, -1)
        cv2.circle(out, (px, py), dot_r + 1, (255, 255, 255), 1)  # white outline
    return out


def annotate_warped(warped: np.ndarray, layout: dict, scale: float) -> np.ndarray:
    """Draw coloured rectangles on the perspective-corrected image."""
    out = warped.copy()
    for b in layout.get("id_boxes", []):
        x0 = int(round(b["x_mm"] * scale)); y0 = int(round(b["y_mm"] * scale))
        x1 = int(round((b["x_mm"] + b["w_mm"]) * scale))
        y1 = int(round((b["y_mm"] + b["h_mm"]) * scale))
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 220, 255), 2)  # yellow — ID
    for b in layout.get("mcq_boxes", []):
        x0 = int(round(b["x_mm"] * scale)); y0 = int(round(b["y_mm"] * scale))
        x1 = int(round((b["x_mm"] + b["w_mm"]) * scale))
        y1 = int(round((b["y_mm"] + b["h_mm"]) * scale))
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 50), 1)   # green — MCQ
    for b in layout.get("num_boxes", []):
        x0 = int(round(b["x_mm"] * scale)); y0 = int(round(b["y_mm"] * scale))
        x1 = int(round((b["x_mm"] + b["w_mm"]) * scale))
        y1 = int(round((b["y_mm"] + b["h_mm"]) * scale))
        color = (50, 50, 240) if b.get("is_decimal") else (240, 120, 30)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 1)
    return out


# ── Coordinate file ───────────────────────────────────────────────────────────

def write_coords(
    path: str,
    centres: list[dict],
    H_inv: np.ndarray,
    scale: float,
) -> None:
    """Write box-centre pixel coordinates (original photo space) to a text file."""
    lines = [
        "# HVH box centres — original photo pixel coordinates",
        "# label  cx_photo  cy_photo  cx_mm  cy_mm",
    ]
    for e in centres:
        px, py = back_project(e["cx_mm"], e["cy_mm"], H_inv, scale)
        lines.append(f"{e['label']:30s}  {px:5d}  {py:5d}  {e['cx_mm']:7.2f}  {e['cy_mm']:7.2f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Coordinates written: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="HVH answer-sheet scanner")
    parser.add_argument("photo",   help="Path to the photo of the answer sheet")
    parser.add_argument("layout",  help="Path to the layout JSON from answer_sheet_gen")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE,
                        help=f"Output pixels per mm (default {DEFAULT_SCALE})")
    parser.add_argument("--out",    default=None,
                        help="Annotated original photo (default: <photo>_annotated.jpg)")
    parser.add_argument("--coords", default=None,
                        help="Box-centre coordinates txt (default: <photo>_coords.txt)")
    parser.add_argument("--results", default=None,
                        help="Answers JSON output (default: <photo>_results.json)")
    parser.add_argument("--no-read", action="store_true",
                        help="Skip box reading — only annotate and write coords")
    parser.add_argument("--save-crops", action="store_true",
                        help="Save non-blank digit crops to labeled_crops/pending/ for labeling")
    args = parser.parse_args()

    img = cv2.imread(args.photo)
    if img is None:
        sys.exit(f"Cannot load image: {args.photo}")

    with open(args.layout) as f:
        layout = json.load(f)

    stem = Path(args.photo).stem

    # 1. Detect corner QR codes
    corners = detect_qr_corners(img)
    print(f"Detected QR corners: {sorted(corners.keys())}")

    if len(corners) == 3:
        corners = estimate_missing_corners(corners)
        print("  (Using estimated 4th corner — result may be slightly off.)")
    elif len(corners) < 3:
        print(f"ERROR: only {len(corners)} QR code(s) found, need at least 3.")
        print("Tips:")
        print("  • Shoot from directly above, not at an angle")
        print("  • Make sure all 4 corners of the paper are in frame")
        print("  • Use better lighting — avoid glare and heavy shadows")
        sys.exit(1)

    # 2. Compute homography
    H      = compute_warp(corners, args.scale)
    H_inv  = np.linalg.inv(H)
    warped = warp_image(img, H, args.scale)

    # 3. Annotate original photo with coloured dots
    centres   = collect_centres(layout)
    out_path  = args.out or stem + "_annotated.jpg"
    cv2.imwrite(out_path, annotate_original(img, centres, H_inv, args.scale))
    print(f"Annotated photo saved : {out_path}")

    # 4. Save warped overview with box outlines
    warped_path = stem + "_warped.jpg"
    cv2.imwrite(warped_path, annotate_warped(warped, layout, args.scale))
    print(f"Warped overview saved : {warped_path}")

    # 5. Write box-centre coordinates
    coords_path = args.coords or stem + "_coords.txt"
    write_coords(coords_path, centres, H_inv, args.scale)

    # 6. Read box contents (unless --no-read)
    if not args.no_read:
        digit_model = None
        try:
            from reader import load_digit_model, read_all_boxes
            print("Loading digit model…")
            digit_model = load_digit_model()
        except ImportError:
            print("WARNING: torch not found — digit boxes will not be read.")
            from reader import read_all_boxes

        results = read_all_boxes(warped, layout, args.scale, digit_model)
        results["sheet_id"] = layout.get("sheet_id", "")

        results_path = args.results or stem + "_results.json"
        # Strip non-serialisable crop arrays before JSON dump
        json_results = {k: v for k, v in results.items() if k != "answers"}
        json_results["answers"] = {
            qk: {ek: ev for ek, ev in qv.items() if ek != "_crops"}
            for qk, qv in results["answers"].items()
        }
        with open(results_path, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"Results saved        : {results_path}")

        # Save digit visualisation image
        from reader import build_digit_debug_image
        dbg_img = build_digit_debug_image(results)
        dbg_path = stem + "_digits.jpg"
        cv2.imwrite(dbg_path, dbg_img)
        print(f"Digit debug image    : {dbg_path}")

        # Save digit crops for labeling
        if args.save_crops:
            import re
            crops_dir = Path("labeled_crops") / "pending"
            crops_dir.mkdir(parents=True, exist_ok=True)
            n_saved = 0
            for qk, qv in results["answers"].items():
                if qv.get("type") != "numeric":
                    continue
                for i, crop_tuple in enumerate(qv.get("_crops", [])):
                    orig_bgr, enh_bgr, bin28, digit, reason = crop_tuple
                    if orig_bgr is None or orig_bgr.size == 0:
                        continue
                    m = re.search(r'conf=(\d+\.\d+)', reason)
                    conf_str = f"_c{int(float(m.group(1))*100):02d}" if m else ""
                    pred_tag = digit if digit not in ("", "?") else \
                               "blank" if digit == "" else "unk"
                    fname = f"{stem}_{qk}_d{i:02d}_pred{pred_tag}{conf_str}.png"
                    cv2.imwrite(str(crops_dir / fname), orig_bgr)
                    n_saved += 1
            # Save decimal indicator crops separately
            from reader import extract_region, enhance_contrast
            warped_enh_local = enhance_contrast(warped)
            with open(args.layout) as _lf:
                _layout_dec = json.load(_lf)
            for b in _layout_dec.get("num_boxes", []):
                if not b.get("is_decimal"):
                    continue
                qk = f"Q{b['q']:02d}"
                reg = extract_region(warped, b["x_mm"], b["y_mm"],
                                     b["w_mm"], b["h_mm"], args.scale, inner_frac=0.05)
                if reg is None or reg.size == 0:
                    continue
                fname = f"{stem}_{qk}_decimal_predpoint.png"
                cv2.imwrite(str(crops_dir / fname), reg)
                n_saved += 1
            print(f"Crops saved          : {n_saved} → {crops_dir}/")

        # Print summary
        print(f"\n  Student ID : {results['student_id'] or '(blank)'}")
        print(f"\n  {'Q':5s}  {'Type':8s}  {'Answer'}")
        print(f"  {'─'*35}")
        for key, val in results["answers"].items():
            if val["type"] == "mcq":
                ans = val["answer"] or "(none)"
                filled = [o for o, f in val["options"].items() if f]
                detail = f"filled={filled}" if len(filled) != 1 else ""
                print(f"  {key:5s}  {val['type']:8s}  {ans}  {detail}")
            else:
                ans    = val["value"] or "(blank)"
                dbg    = "  " + " ".join(val.get("debug", [])) if val.get("debug") else ""
                print(f"  {key:5s}  {val['type']:8s}  {ans}{dbg}")


if __name__ == "__main__":
    main()
