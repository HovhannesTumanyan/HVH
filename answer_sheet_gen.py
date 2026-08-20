#!/usr/bin/env python3
"""
Answer Sheet Generator v2 — HVH Testing System

Layout (matches Armenian exam standard):
  • 4 corner QR codes  (homographic calibration for CV)
  • Student ID boxes   (handwritten, centred BETWEEN the two top QRs)
  • Header             (title, variant, name line)
  • MCQ grid sections  (questions as columns, options a–f as rows, square boxes)
  • Numeric sections   (dotted-border handwriting boxes, 3 per row)
  • Capacity reporter

Coordinate system: ReportLab — origin at page bottom-left, y grows upward.
"""

from __future__ import annotations
import io
from dataclasses import dataclass
from typing import Literal

import qrcode
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


# ══════════════════════════════════════════════════════════════════════════════
# PAGE & ZONE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PAGE_W, PAGE_H = A4           # 595.28 pt × 841.89 pt  (210 × 297 mm)

# ── Corner QR codes ───────────────────────────────────────────────────────────
QR_EDGE = 6 * mm              # gap from paper edge to QR corner
QR_SIZE = 22 * mm             # QR code square side

# ── Content rectangle ─────────────────────────────────────────────────────────
# Left/right: extends to page margin (QR_EDGE) — the side strips between the
#   corner QR codes are free because QRs only occupy the top & bottom corners.
#   Content y-range [CB, CT] never overlaps QR y-ranges [6, 28] and [269, 291].
# Top/bottom: small pad below/above the QR bands.
_VPAD = 2 * mm
CL    = QR_EDGE                                     # left  ≈ 6 mm from edge
CR    = PAGE_W - QR_EDGE                            # right ≈ 204 mm from left
CT    = PAGE_H - QR_EDGE - QR_SIZE - _VPAD          # top   ≈ 267 mm from bottom
CB    = QR_EDGE + QR_SIZE + _VPAD                   # bottom ≈ 30 mm from bottom
CW    = CR - CL               # ≈ 198 mm
CH    = CT - CB               # ≈ 237 mm

# ── Student-ID strip (between the top-left and top-right QR codes) ────────────
ID_BOXES        = 8           # 8-digit student code
ID_BOX_W        = 9.0 * mm
ID_BOX_H        = 11.0 * mm
ID_BOX_GAP      = 1.5 * mm
_ID_STRIP_W     = ID_BOXES * ID_BOX_W + (ID_BOXES - 1) * ID_BOX_GAP  # ≈ 82.5 mm
_ID_ZONE_L      = QR_EDGE + QR_SIZE                 # right edge of TL QR
_ID_ZONE_R      = PAGE_W - QR_EDGE - QR_SIZE        # left edge of TR QR
_ID_ZONE_CY     = PAGE_H - QR_EDGE - QR_SIZE / 2   # vertical centre of top QR band

# ── Header area (top of content rectangle) ────────────────────────────────────
HEADER_H = 17 * mm            # title + name line + separator

# ── MCQ grid ──────────────────────────────────────────────────────────────────
# Target ~7 mm per column; with CW ≈ 198 mm that gives ~27 columns.
GRID_LBL_W    = 9.0 * mm     # left column that holds "a / b / c …"
GRID_NUM_H    = 5.5 * mm     # question-number header row height
GRID_OPT_H    = 6.5 * mm     # height per option row
_TARGET_Q_W   = 7.0 * mm     # desired column width
QUES_PER_ROW  = max(10, int((CW - GRID_LBL_W) / _TARGET_Q_W))  # auto, ≈ 27
GRID_Q_W      = (CW - GRID_LBL_W) / QUES_PER_ROW   # exact column width
GRID_SEC_GAP  = 5.0 * mm     # vertical gap between any two sections

# ── Numeric question boxes ─────────────────────────────────────────────────────
NUM_BOX_W       = 5.8 * mm   # each digit box width
NUM_BOX_H       = 8.0 * mm   # each digit box height
NUM_BOX_GAP     = 0.8 * mm   # gap between boxes
NUM_DOT_W       = 3.8 * mm   # space occupied by decimal dot separator
NUM_Q_LBL_W    = 10.0 * mm  # width of "Q." label
NUM_ROW_H       = NUM_BOX_H + 3.5 * mm   # total height per numeric row
_NUM_Q_GAP      = 5.0 * mm   # horizontal gap between questions in the same row
_NUM_PANEL_GAP  = 4.0 * mm   # gap between MCQ right edge and numeric side panel


# ══════════════════════════════════════════════════════════════════════════════
# QUESTION DATA MODELS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MCQQuestion:
    type: Literal["mcq"] = "mcq"
    n_options: int = 4        # 2 – 6  (maps to a – f)

    def validate(self):
        if not 2 <= self.n_options <= 6:
            raise ValueError(f"MCQ must have 2–6 options, got {self.n_options}")


@dataclass
class NumericQuestion:
    type: Literal["numeric"] = "numeric"
    n_digits:    int  = 4     # total handwriting boxes
    has_decimal: bool = False
    decimal_pos: int  = 2     # digit boxes BEFORE the decimal dot

    def content_width(self) -> float:
        w = self.n_digits * (NUM_BOX_W + NUM_BOX_GAP)
        if self.has_decimal:
            w += NUM_DOT_W + NUM_BOX_GAP
        return w

    def validate(self):
        if self.n_digits < 1:
            raise ValueError("Need at least 1 digit box")
        if self.has_decimal and not (0 < self.decimal_pos < self.n_digits):
            raise ValueError(
                f"decimal_pos={self.decimal_pos} must be 1 … {self.n_digits - 1}"
            )


Question = MCQQuestion | NumericQuestion


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT / CAPACITY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _segment(questions: list[Question]) -> list[tuple[str, list]]:
    """Split into alternating runs of mcq / numeric."""
    if not questions:
        return []
    segs, cur_type, cur = [], questions[0].type, [questions[0]]
    for q in questions[1:]:
        if q.type == cur_type:
            cur.append(q)
        else:
            segs.append((cur_type, cur))
            cur_type, cur = q.type, [q]
    segs.append((cur_type, cur))
    return segs


def _mcq_section_h(max_opts: int) -> float:
    return GRID_NUM_H + max_opts * GRID_OPT_H


def _pack_rows(questions: list[NumericQuestion], width: float) -> list[list[tuple]]:
    """Pack numeric questions left-to-right within `width`. Returns rows of (q, x_offset)."""
    rows: list[list[tuple]] = []
    row:  list[tuple]       = []
    x = 0.0
    for q in questions:
        q_w    = NUM_Q_LBL_W + q.content_width()
        needed = q_w + (_NUM_Q_GAP if row else 0.0)
        if row and x + needed > width:
            rows.append(row)
            row, x = [(q, 0.0)], q_w
        else:
            ox = x + (_NUM_Q_GAP if row else 0.0)
            row.append((q, ox))
            x = ox + q_w
    if row:
        rows.append(row)
    return rows


def _pack_in_panel(
    questions: list[NumericQuestion],
    panel_w: float,
    panel_h: float,
) -> tuple[list[list[tuple]], int]:
    """
    Pack as many questions as fit in a panel of (panel_w × panel_h).
    Returns (rows, n_placed).  Stops when height or width is exhausted.
    """
    max_rows = max(1, int(panel_h / NUM_ROW_H))
    rows: list[list[tuple]] = []
    row:  list[tuple]       = []
    x, placed = 0.0, 0
    for q in questions:
        q_w    = NUM_Q_LBL_W + q.content_width()
        needed = q_w + (_NUM_Q_GAP if row else 0.0)
        if q_w > panel_w:
            break                          # too wide for any row
        if row and x + needed > panel_w:
            if len(rows) + 1 >= max_rows:
                break                      # no more height
            rows.append(row)
            row, x = [(q, 0.0)], q_w
        else:
            ox = x + (_NUM_Q_GAP if row else 0.0)
            row.append((q, ox))
            x = ox + q_w
        placed += 1
    if row:
        rows.append(row)
    return rows, placed


def _plan_layout(questions: list[Question]) -> list[dict]:
    """
    Compute the full visual layout as a list of bands.
    Each band dict contains everything needed for height calculation and drawing.

    Band keys:
      kind        'mcq' | 'num'
      height      float  (excluding trailing GRID_SEC_GAP)

    MCQ band extras:
      mcq         list[MCQQuestion]
      mcq_q0      int  (1-based start)
      mcq_draw_w  float  (actual MCQ grid width; < CW when side panel used)
      side_rows   list[list[tuple(q, x_off)]]  (numeric placed beside MCQ)
      side_q0     int
      side_x      float  (CL-relative left edge of side panel)

    Numeric band extras:
      num_rows    list[list[tuple(q, x_off)]]
      num_q0      int
    """
    bands: list[dict] = []
    i, q_num, n = 0, 1, len(questions)

    while i < n:
        if isinstance(questions[i], MCQQuestion):
            # ── collect MCQ run ───────────────────────────────────────────────
            j = i
            while j < n and isinstance(questions[j], MCQQuestion):
                j += 1
            run     = questions[i:j]
            n_chunks = -(-len(run) // QUES_PER_ROW)
            n_beside = 0                   # numeric placed beside last chunk

            for ci in range(n_chunks):
                chunk    = run[ci * QUES_PER_ROW:(ci + 1) * QUES_PER_ROW]
                max_opts = max(q.n_options for q in chunk)
                mcq_h    = _mcq_section_h(max_opts)
                mcq_w    = GRID_LBL_W + len(chunk) * GRID_Q_W
                is_last  = (ci == n_chunks - 1)

                side_rows, n_side = [], 0
                if is_last and j < n and isinstance(questions[j], NumericQuestion):
                    # collect the numeric run that follows
                    k = j
                    while k < n and isinstance(questions[k], NumericQuestion):
                        k += 1
                    pw = CW - mcq_w - _NUM_PANEL_GAP
                    if pw > NUM_Q_LBL_W:
                        side_rows, n_side = _pack_in_panel(questions[j:k], pw, mcq_h)

                if is_last:
                    n_beside = n_side

                bands.append({
                    'kind':        'mcq',
                    'mcq':         chunk,
                    'mcq_q0':      q_num + ci * QUES_PER_ROW,
                    'mcq_draw_w':  mcq_w if side_rows else CW,
                    'height':      mcq_h,
                    'side_rows':   side_rows,
                    'side_q0':     q_num + len(run),   # numbered after all MCQ in run
                    'side_x':      CL + mcq_w + _NUM_PANEL_GAP,
                })

            q_num += len(run) + n_beside
            i      = j + n_beside

        else:
            # ── collect numeric run ───────────────────────────────────────────
            j = i
            while j < n and isinstance(questions[j], NumericQuestion):
                j += 1
            run  = questions[i:j]
            rows = _pack_rows(run, CW)
            bands.append({
                'kind':     'num',
                'num_rows': rows,
                'num_q0':   q_num,
                'height':   len(rows) * NUM_ROW_H,
            })
            q_num += len(run)
            i      = j

    return bands


def _height_used(questions: list[Question]) -> float:
    return HEADER_H + sum(b['height'] + GRID_SEC_GAP for b in _plan_layout(questions))


def capacity_info(questions: list[Question]) -> dict:
    used  = _height_used(questions)
    avail = CH - used
    rem_mcq = max(0, int(avail / (_mcq_section_h(4) + GRID_SEC_GAP) * QUES_PER_ROW))
    _sw  = NUM_Q_LBL_W + NumericQuestion(n_digits=4).content_width() + _NUM_Q_GAP
    rem_num = max(0, int(avail / NUM_ROW_H) * max(1, int(CW / _sw)))
    return {
        "used_mm":     used  / mm,
        "avail_mm":    avail / mm,
        "fits":        avail >= 0,
        "rem_mcq_4":   rem_mcq,
        "rem_numeric": rem_num,
    }


def report_capacity(questions: list[Question]) -> None:
    cap = capacity_info(questions)
    div = "═" * 58
    print(f"\n{div}")
    print("  HVH Answer Sheet — Capacity Report")
    print(div)
    print(f"  Content area : {CW/mm:.0f} × {CH/mm:.0f} mm")
    print(f"  Used height  : {cap['used_mm']:.1f} mm")
    print(f"  Available    : {cap['avail_mm']:.1f} mm")
    print(f"  Fits on page : {'YES ✓' if cap['fits'] else 'NO — TOO MANY QUESTIONS ⚠'}")
    print(f"  Can still add (approx):")
    print(f"    MCQ 4-opt  : ~{cap['rem_mcq_4']} more questions")
    print(f"    Numeric    : ~{cap['rem_numeric']} more questions")
    print(div)
    for i, q in enumerate(questions, 1):
        if isinstance(q, MCQQuestion):
            print(f"  Q{i:02d}  MCQ  {q.n_options} opts (a–{'abcdef'[q.n_options-1]})")
        else:
            dec = f"  decimal after box {q.decimal_pos}" if q.has_decimal else ""
            print(f"  Q{i:02d}  NUM  {q.n_digits} boxes{dec}")
    print(div + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# QR CODE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _make_qr(data: str) -> ImageReader:
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10, border=1,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return ImageReader(buf)


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def _draw_corner_qrs(c: canvas.Canvas, test_id: str, variant: str) -> None:
    for corner, (x, y) in {
        "TL": (QR_EDGE,                    PAGE_H - QR_EDGE - QR_SIZE),
        "TR": (PAGE_W - QR_EDGE - QR_SIZE, PAGE_H - QR_EDGE - QR_SIZE),
        "BL": (QR_EDGE,                    QR_EDGE),
        "BR": (PAGE_W - QR_EDGE - QR_SIZE, QR_EDGE),
    }.items():
        c.drawImage(_make_qr(f"HVH|{test_id}|{variant}|{corner}"),
                    x, y, width=QR_SIZE, height=QR_SIZE, mask="auto")


def _draw_student_id(c: canvas.Canvas) -> None:
    """Handwriting boxes centred between the two top QR codes."""
    start_x = (_ID_ZONE_L + _ID_ZONE_R - _ID_STRIP_W) / 2
    box_y   = _ID_ZONE_CY - ID_BOX_H / 2

    # Label above
    c.setFont("Helvetica-Bold", 7)
    c.setFillColor(colors.black)
    c.drawCentredString(start_x + _ID_STRIP_W / 2,
                        box_y + ID_BOX_H + 3 * mm, "Student ID")

    # Boxes
    c.setStrokeColor(colors.black)
    c.setLineWidth(1.0)
    for i in range(ID_BOXES):
        bx = start_x + i * (ID_BOX_W + ID_BOX_GAP)
        c.setFillColor(colors.white)
        c.rect(bx, box_y, ID_BOX_W, ID_BOX_H, stroke=1, fill=1)

    # Hint below
    c.setFont("Helvetica", 5.5)
    c.setFillColorRGB(0.45, 0.45, 0.45)
    c.drawCentredString(start_x + _ID_STRIP_W / 2,
                        box_y - 3.5 * mm, "write clearly  ·  use pen")


def _draw_header(c: canvas.Canvas, title: str, variant: str, date: str) -> None:
    y = CT
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(colors.black)
    c.drawString(CL, y - 6 * mm, title)

    c.setFont("Helvetica", 8)
    c.drawRightString(CR, y - 6 * mm, f"Variant {variant}   {date}")

    # Name line
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.25, 0.25, 0.25)
    c.drawString(CL, y - 11 * mm, "Full name:")
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.line(CL + 22 * mm, y - 10.5 * mm, CR - 28 * mm, y - 10.5 * mm)
    c.drawString(CR - 26 * mm, y - 11 * mm, "Class:")
    c.line(CR - 15 * mm, y - 10.5 * mm, CR, y - 10.5 * mm)

    # Bottom border of header
    c.setStrokeColorRGB(0.35, 0.35, 0.35)
    c.setLineWidth(0.6)
    c.line(CL, y - HEADER_H, CR, y - HEADER_H)



def _draw_mcq_section(c: canvas.Canvas, y_top: float,
                      questions: list[MCQQuestion],
                      q_start: int, *, draw_w: float = None) -> float:
    """
    Draw one MCQ grid section. draw_w lets it be narrower than CW when
    a numeric side panel will fill the remaining space.
    Returns height consumed.
    """
    max_opts = max(q.n_options for q in questions)
    sec_h    = _mcq_section_h(max_opts)
    opts     = "abcdef"
    n        = len(questions)
    w        = draw_w if draw_w is not None else CW

    x0 = CL  # left of whole grid (label column starts here)

    # ── Question-number header row ─────────────────────────────────────────────
    c.setFont("Helvetica-Bold", 6)
    c.setFillColor(colors.black)
    for i in range(n):
        cx = x0 + GRID_LBL_W + i * GRID_Q_W + GRID_Q_W / 2
        c.drawCentredString(cx, y_top - GRID_NUM_H / 2 - 2 * mm,
                            str(q_start + i))

    # ── Option rows ────────────────────────────────────────────────────────────
    for opt_i in range(max_opts):
        row_top = y_top - GRID_NUM_H - opt_i * GRID_OPT_H
        row_bot = row_top - GRID_OPT_H

        # Letter label in the leftmost column
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.black)
        c.drawCentredString(x0 + GRID_LBL_W / 2,
                            row_bot + GRID_OPT_H / 2 - 2 * mm,
                            opts[opt_i])

        # One box per question column
        for q_i, q in enumerate(questions):
            bx = x0 + GRID_LBL_W + q_i * GRID_Q_W + 0.7 * mm
            by = row_bot + 0.7 * mm
            bw = GRID_Q_W - 1.4 * mm
            bh = GRID_OPT_H - 1.4 * mm

            if opt_i < q.n_options:
                c.setFillColor(colors.white)
                c.setStrokeColor(colors.black)
                c.setLineWidth(0.5)
            else:
                # option does not exist for this question → grey out
                c.setFillColorRGB(0.87, 0.87, 0.87)
                c.setStrokeColorRGB(0.72, 0.72, 0.72)
                c.setLineWidth(0.3)

            c.rect(bx, by, bw, bh, stroke=1, fill=1)

    # ── Grid lines ─────────────────────────────────────────────────────────────
    # Outer border
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.7)
    c.rect(x0, y_top - sec_h, w, sec_h, stroke=1, fill=0)

    # Divider below num-header row
    c.setLineWidth(0.5)
    c.line(x0, y_top - GRID_NUM_H, x0 + w, y_top - GRID_NUM_H)

    # Horizontal lines between option rows
    c.setStrokeColorRGB(0.55, 0.55, 0.55)
    c.setLineWidth(0.3)
    for opt_i in range(1, max_opts):
        hy = y_top - GRID_NUM_H - opt_i * GRID_OPT_H
        c.line(x0, hy, x0 + w, hy)

    # Vertical divider after label column
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.line(x0 + GRID_LBL_W, y_top - sec_h, x0 + GRID_LBL_W, y_top)

    # Vertical lines between question columns
    c.setStrokeColorRGB(0.55, 0.55, 0.55)
    c.setLineWidth(0.3)
    for q_i in range(1, n):
        vx = x0 + GRID_LBL_W + q_i * GRID_Q_W
        c.line(vx, y_top - sec_h, vx, y_top)

    return sec_h


def _draw_num_rows(c: canvas.Canvas, x_left: float, x_right: float,
                   y_top: float, rows: list, q_start: int) -> float:
    """
    Shared helper: draw pre-packed numeric rows inside a border.
    x_left/x_right define the bounding box (for border + row separators).
    Rows are list of [(question, x_offset_from_x_left), ...].
    Returns height consumed.
    """
    total_h = len(rows) * NUM_ROW_H
    box_w   = x_right - x_left

    c.setStrokeColor(colors.black)
    c.setLineWidth(0.7)
    c.rect(x_left, y_top - total_h, box_w, total_h, stroke=1, fill=0)

    q_counter = 0
    for row_idx, row in enumerate(rows):
        row_y   = y_top - row_idx * NUM_ROW_H
        box_top = row_y - (NUM_ROW_H - NUM_BOX_H) / 2 - NUM_BOX_H

        if row_idx > 0:
            c.setStrokeColorRGB(0.75, 0.75, 0.75)
            c.setLineWidth(0.25)
            c.line(x_left, row_y, x_right, row_y)

        for q, x_off in row:
            col_x = x_left + x_off

            c.setFont("Helvetica-Bold", 7.5)
            c.setFillColor(colors.black)
            c.drawString(col_x + 1.5 * mm,
                         box_top + NUM_BOX_H / 2 - 2.5 * mm,
                         f"{q_start + q_counter}.")

            bx = col_x + NUM_Q_LBL_W
            for d in range(q.n_digits):
                if q.has_decimal and d == q.decimal_pos:
                    c.setFillColor(colors.black)
                    c.circle(bx + NUM_DOT_W / 2,
                             box_top + NUM_BOX_H * 0.18, 1.0 * mm, stroke=0, fill=1)
                    bx += NUM_DOT_W + NUM_BOX_GAP

                c.setFillColor(colors.white)
                c.setStrokeColor(colors.black)
                c.setLineWidth(0.5)
                c.setDash([2, 2])
                c.rect(bx, box_top, NUM_BOX_W, NUM_BOX_H, stroke=1, fill=1)
                c.setDash([])
                bx += NUM_BOX_W + NUM_BOX_GAP

            q_counter += 1

    return total_h


def _draw_numeric_section(c: canvas.Canvas, y_top: float,
                          rows: list, q_start: int) -> float:
    """Full-width numeric section from pre-packed rows."""
    return _draw_num_rows(c, CL, CR, y_top, rows, q_start)


def _draw_numeric_panel(c: canvas.Canvas, x_left: float, y_top: float,
                        rows: list, q_start: int) -> None:
    """Numeric side panel beside an MCQ section (pre-packed rows)."""
    _draw_num_rows(c, x_left, CR, y_top, rows, q_start)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATE
# ══════════════════════════════════════════════════════════════════════════════

def generate(
    questions: list[Question],
    *,
    test_title: str = "Test",
    test_id:    str = "TEST001",
    variant:    str = "A",
    date:       str = "",
    output:     str = "answer_sheet.pdf",
) -> tuple[str, dict]:
    """
    Generate an A4 answer sheet PDF.
    Returns (output_path, capacity_info_dict).
    """
    for q in questions:
        q.validate()

    cap = capacity_info(questions)
    if not cap["fits"]:
        raise ValueError(
            f"Questions overflow by {-cap['avail_mm']:.1f} mm — reduce question count."
        )

    c = canvas.Canvas(output, pagesize=A4)
    c.setTitle(test_title)
    c.setAuthor("HVH Testing System")

    # 1. Corner QR codes
    _draw_corner_qrs(c, test_id, variant)

    # 2. Student ID between top QRs
    _draw_student_id(c)

    # 3. Outer content border (exactly at content edges, no overshoot)
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.8)
    c.rect(CL, CB, CW, CH, stroke=1, fill=0)

    # 4. Header
    _draw_header(c, test_title, variant, date)

    # 5. Question sections — driven by the layout plan
    y = CT - HEADER_H
    for band in _plan_layout(questions):
        if band['kind'] == 'mcq':
            _draw_mcq_section(c, y, band['mcq'], band['mcq_q0'],
                              draw_w=band['mcq_draw_w'])
            if band['side_rows']:
                _draw_numeric_panel(c, band['side_x'], y,
                                    band['side_rows'], band['side_q0'])
        else:
            _draw_numeric_section(c, y, band['num_rows'], band['num_q0'])

        y -= band['height'] + GRID_SEC_GAP

    c.save()
    return output, cap


# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Teacher's mixed question set:
    # Q1–4: mixed option counts, Q5–6: numeric, Q7–10: MCQ, Q11–14: numeric
    questions: list[Question] = [
        # Block 1 — MCQ (mixed options like the user described)
        MCQQuestion(n_options=4),   # Q1
        MCQQuestion(n_options=6),   # Q2
        MCQQuestion(n_options=4),   # Q3
        MCQQuestion(n_options=3),   # Q4
        MCQQuestion(n_options=6),   # Q5
        MCQQuestion(n_options=4),   # Q6
        MCQQuestion(n_options=5),   # Q7
        MCQQuestion(n_options=4),   # Q8
        MCQQuestion(n_options=4),   # Q9
        MCQQuestion(n_options=6),   # Q10
        MCQQuestion(n_options=4),   # Q11
        MCQQuestion(n_options=3),   # Q12
        MCQQuestion(n_options=4),   # Q13
        MCQQuestion(n_options=6),   # Q14
        MCQQuestion(n_options=4),   # Q15
        MCQQuestion(n_options=4),   # Q16
        MCQQuestion(n_options=5),   # Q17
        MCQQuestion(n_options=4),   # Q18
        MCQQuestion(n_options=6),   # Q19
        # Block 2 — Numeric (float with decimal point)
        NumericQuestion(n_digits=5, has_decimal=True,  decimal_pos=2),  # Q20  e.g. "12.345"
        NumericQuestion(n_digits=4, has_decimal=False),                  # Q21  e.g. "2026"
        NumericQuestion(n_digits=6, has_decimal=True,  decimal_pos=3),  # Q22  e.g. "123.456"
        NumericQuestion(n_digits=3, has_decimal=False),                  # Q23  e.g. "314"
        NumericQuestion(n_digits=4, has_decimal=True,  decimal_pos=1),  # Q24  e.g. "3.141"
        NumericQuestion(n_digits=5, has_decimal=False),                  # Q25  e.g. "12345"
        # Block 3 — More MCQ
        MCQQuestion(n_options=4),   # Q26
        MCQQuestion(n_options=4),   # Q27
        MCQQuestion(n_options=4),   # Q28
        MCQQuestion(n_options=4),   # Q29
        MCQQuestion(n_options=3),   # Q30
        MCQQuestion(n_options=4),  # Q2
        MCQQuestion(n_options=4),  # Q3
        MCQQuestion(n_options=3),  # Q4
        MCQQuestion(n_options=4),  # Q5
        MCQQuestion(n_options=4),  # Q6
        MCQQuestion(n_options=5),  # Q7
        MCQQuestion(n_options=4),  # Q8
        MCQQuestion(n_options=4),  # Q9
        MCQQuestion(n_options=4),  # Q10
        MCQQuestion(n_options=4),  # Q11
        NumericQuestion(n_digits=3, has_decimal=False),
        MCQQuestion(n_options=3),  # Q12
        MCQQuestion(n_options=4),  # Q13
        MCQQuestion(n_options=4),  # Q14
        MCQQuestion(n_options=4),  # Q15
        MCQQuestion(n_options=4),  # Q16
        MCQQuestion(n_options=5),  # Q17
        MCQQuestion(n_options=4),  # Q18
        MCQQuestion(n_options=4),  # Q19
        MCQQuestion(n_options=4),  # Q2
        MCQQuestion(n_options=4),  # Q3
        MCQQuestion(n_options=3),  # Q4
        MCQQuestion(n_options=4),  # Q5
        MCQQuestion(n_options=4),  # Q6
        MCQQuestion(n_options=5),  # Q7
        MCQQuestion(n_options=4),  # Q8
        MCQQuestion(n_options=4),  # Q9
        MCQQuestion(n_options=4),  # Q10
        MCQQuestion(n_options=4),  # Q11
        MCQQuestion(n_options=3),  # Q12
        MCQQuestion(n_options=4),  # Q13
        MCQQuestion(n_options=4),  # Q14
        MCQQuestion(n_options=4),  # Q15
        MCQQuestion(n_options=4),  # Q16
        MCQQuestion(n_options=5),  # Q17
        MCQQuestion(n_options=4),  # Q18
        MCQQuestion(n_options=4),  # Q19
    ]

    report_capacity(questions)

    path, cap = generate(
        questions,
        test_title = "Mathematics — Final Exam 2026",
        test_id    = "MATH2026A",
        variant    = "A",
        date       = "18 Aug 2026",
        output     = "answer_sheet.pdf",
    )
    print(f"Generated : {path}")
    print(f"Remaining : ~{cap['rem_mcq_4']} MCQ-4  or  ~{cap['rem_numeric']} numeric\n")
