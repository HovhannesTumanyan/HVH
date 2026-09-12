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
import secrets
from dataclasses import dataclass
from typing import Literal

import qrcode
from PIL import Image as _PILImage, ImageDraw as _PILDraw, ImageFilter as _PILFilter
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
HEADER_H = 20 * mm            # title + name line + separator
GUIDE_H  = 15 * mm            # instruction strip reserved at content-area bottom

# ── MCQ grid ──────────────────────────────────────────────────────────────────
# Target ~7 mm per column; with CW ≈ 198 mm that gives ~27 columns.
GRID_LBL_W    = 9.0 * mm     # left column that holds "a / b / c …"
GRID_NUM_H    = 5.5 * mm     # question-number header row height
GRID_OPT_H    = 5.0 * mm     # height per option row (smaller cells → smaller boxes)
_TARGET_Q_W   = 6.0 * mm     # desired column width (smaller cells → smaller boxes)
QUES_PER_ROW  = max(10, int((CW - GRID_LBL_W) / _TARGET_Q_W))  # auto, ≈ 27
GRID_Q_W      = (CW - GRID_LBL_W) / QUES_PER_ROW   # exact column width
GRID_SEC_GAP  = 5.0 * mm     # vertical gap between any two sections

# ── Numeric question boxes ─────────────────────────────────────────────────────
_CHECKBOX_SIDE  = min(GRID_Q_W, GRID_OPT_H) - 1.4 * mm  # MCQ tick-box square side
NUM_BOX_W       = 2 * _CHECKBOX_SIDE   # digit box = 2× MCQ checkbox
NUM_BOX_H       = 8.0 * mm
NUM_BOX_GAP     = 1.5 * mm   # gap between boxes
NUM_DOT_BOX_W   = NUM_BOX_W  # decimal-point box same size as digit boxes
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
    points:    int  = 1       # max points for this question
    partial:   bool = False   # allow partial credit for multi-answer MCQ

    def validate(self):
        if not 2 <= self.n_options <= 6:
            raise ValueError(f"MCQ must have 2–6 options, got {self.n_options}")


@dataclass
class NumericQuestion:
    type: Literal["numeric"] = "numeric"
    n_digits:    int  = 4     # total handwriting boxes
    has_decimal: bool = False
    decimal_pos: int  = 2     # digit boxes BEFORE the decimal dot
    points:      int  = 1     # max points for this question

    def content_width(self) -> float:
        w = self.n_digits * (NUM_BOX_W + NUM_BOX_GAP)
        if self.has_decimal:
            w += NUM_DOT_BOX_W + NUM_BOX_GAP
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


def _max_row_width(rows: list[list[tuple]]) -> float:
    """Actual maximum width used by any packed row."""
    max_w = 0.0
    for row in rows:
        if row:
            last_q, last_x, _ = row[-1]
            max_w = max(max_w, last_x + NUM_Q_LBL_W + last_q.content_width())
    return max_w


def _fill_rows(
    indexed: list[tuple[int, NumericQuestion]],
    width: float,
    max_rows: int | None = None,
) -> list[list[tuple]]:
    """
    Core row-packing loop shared by _pack_rows and _pack_in_panel.
    Each entry is (q, x_offset, orig_idx).
    """
    remaining = list(indexed)
    rows: list[list[tuple]] = []

    while remaining and (max_rows is None or len(rows) < max_rows):
        row: list[tuple] = []
        x = 0.0
        placed: set[int] = set()

        for list_pos, (orig_idx, q) in enumerate(remaining):
            q_w = NUM_Q_LBL_W + q.content_width()
            gap = _NUM_Q_GAP if row else 0.0
            if x + gap + q_w <= width:
                row.append((q, x + gap, orig_idx))
                x += gap + q_w
                placed.add(list_pos)

        remaining = [item for pos, item in enumerate(remaining) if pos not in placed]
        if not row:
            break
        row.sort(key=lambda t: t[2])
        reordered, x = [], 0.0
        for q, _, orig_idx in row:
            gap = _NUM_Q_GAP if reordered else 0.0
            reordered.append((q, x + gap, orig_idx))
            x += gap + NUM_Q_LBL_W + q.content_width()
        rows.append(reordered)

    return rows


def _pack_rows(questions: list[NumericQuestion], width: float) -> list[list[tuple]]:
    return _fill_rows(list(enumerate(questions)), width)


def _pack_in_panel(
    questions: list[NumericQuestion],
    panel_w: float,
    panel_h: float,
) -> tuple[list[list[tuple]], int]:
    max_rows = max(1, int(panel_h / NUM_ROW_H))
    eligible = [(i, q) for i, q in enumerate(questions)
                if NUM_Q_LBL_W + q.content_width() <= panel_w]
    rows = _fill_rows(eligible, panel_w, max_rows=max_rows)
    placed_ids = {id(q) for r in rows for q, _, _ in r}
    n_placed   = sum(1 for q in questions if id(q) in placed_ids)
    return rows, n_placed


def _optimal_order(questions: list[Question]) -> tuple[list[Question], list[int]]:
    """
    Reorder questions for minimal page height.

    Strategy
    --------
    1. Group MCQ by n_options (ascending).
       Each layout chunk uses height = GRID_NUM_H + max_opts * GRID_OPT_H.
       Mixing option counts inflates every row in a chunk to the tallest option.
       Grouping same-option questions together ensures each chunk is exactly as
       tall as it needs to be.

    2. Sort Numeric by content_width() descending (first-fit-decreasing).
       Wide questions fill complete rows; the narrowest questions fall into the
       last row, leaving the maximum horizontal gap for an MCQ side panel.

    3. All MCQ before all Numeric.
       The last MCQ chunk absorbs numeric questions in its right side panel,
       and any remaining narrow numerics can absorb MCQ in their right side.

    Returns
    -------
    (reordered_questions, original_1based_indices)
    original_1based_indices[k] is the original question number of the k-th
    question in the returned list.
    """
    indexed = list(enumerate(questions, 1))   # (1-based-index, question)

    mcq_idx = [(i, q) for i, q in indexed if isinstance(q, MCQQuestion)]
    num_idx = [(i, q) for i, q in indexed if isinstance(q, NumericQuestion)]

    # MCQ: stable sort by n_options ascending — groups same-height questions
    mcq_sorted = sorted(mcq_idx, key=lambda x: x[1].n_options)

    # Numeric: sort by total box width descending — narrowest questions land last
    num_sorted = sorted(num_idx, key=lambda x: x[1].content_width(), reverse=True)

    combined       = mcq_sorted + num_sorted
    new_questions  = [q for _, q in combined]
    original_order = [i for i, _ in combined]
    return new_questions, original_order


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

                band_h = mcq_h

                # After placing numeric in side panel, check if MCQ fits further right
                side_mcq2, side_mcq2_q0, side_mcq2_x, side_num_nat_w = [], 0, CR, 0
                if side_rows:
                    side_num_nat_w = _max_row_width(side_rows)
                    side_x_pos = CL + mcq_w + _NUM_PANEL_GAP
                    extra_x = side_x_pos + side_num_nat_w + _NUM_PANEL_GAP
                    extra_w = CR - extra_x
                    extra_cols = int((extra_w - GRID_LBL_W) / GRID_Q_W) if extra_w > GRID_LBL_W else 0
                    if extra_cols >= 1:
                        nxt = j + n_side
                        if nxt < n and isinstance(questions[nxt], MCQQuestion):
                            chunk2 = []
                            k2 = nxt
                            while k2 < n and isinstance(questions[k2], MCQQuestion) and len(chunk2) < extra_cols:
                                chunk2.append(questions[k2])
                                k2 += 1
                            if chunk2:
                                side_mcq2     = chunk2
                                side_mcq2_q0  = q_num + len(run) + n_side
                                side_mcq2_x   = extra_x
                                side_num_nat_w = side_num_nat_w  # keep for x_right of numeric panel

                if is_last:
                    n_beside = n_side + len(side_mcq2)

                bands.append({
                    'kind':           'mcq',
                    'mcq':            chunk,
                    'mcq_q0':         q_num + ci * QUES_PER_ROW,
                    'mcq_draw_w':     mcq_w if side_rows else CW,
                    'height':         mcq_h,
                    'side_rows':      side_rows,
                    'side_q0':        q_num + len(run),
                    'side_x':         CL + mcq_w + _NUM_PANEL_GAP,
                    'side_num_nat_w': side_num_nat_w,
                    'side_mcq2':      side_mcq2,
                    'side_mcq2_q0':   side_mcq2_q0,
                    'side_mcq2_x':    side_mcq2_x,
                })

            q_num += len(run) + n_beside
            i      = j + n_beside

        else:
            # ── collect numeric run ───────────────────────────────────────────
            j = i
            while j < n and isinstance(questions[j], NumericQuestion):
                j += 1
            run  = questions[i:j]
            rows  = _pack_rows(run, CW)
            nat_w = _max_row_width(rows)
            num_h = len(rows) * NUM_ROW_H

            # Border fits tightly around content; MCQ fills remaining width
            side_mcq, n_mcq_side, side_mcq_q0, num_draw_w = [], 0, 0, nat_w
            if j < n and isinstance(questions[j], MCQQuestion):
                panel_w = CW - nat_w - _NUM_PANEL_GAP
                mcq_cols = int((panel_w - GRID_LBL_W) / GRID_Q_W) if panel_w > GRID_LBL_W else 0
                if mcq_cols >= 1:
                    # only take consecutive MCQ questions
                    k = j
                    while k < n and isinstance(questions[k], MCQQuestion) and k - j < mcq_cols:
                        k += 1
                    chunk = questions[j:k]
                    if chunk:
                        mcq_h = _mcq_section_h(max(q.n_options for q in chunk))
                        side_mcq     = chunk
                        n_mcq_side   = len(chunk)
                        side_mcq_q0  = q_num + len(run)
                        rows = _pack_rows(run, nat_w)   # repack tighter
                        # band height expands if MCQ is taller than numeric rows
                        num_h = max(num_h, mcq_h)

            bands.append({
                'kind':        'num',
                'num_rows':    rows,
                'num_draw_w':  num_draw_w,
                'num_q0':      q_num,
                'height':      num_h,
                'side_mcq':    side_mcq,
                'side_mcq_q0': side_mcq_q0,
                'side_mcq_x':  CL + num_draw_w + _NUM_PANEL_GAP,
            })
            q_num += len(run) + n_mcq_side
            i      = j + n_mcq_side

    return bands


def _height_used(questions: list[Question], bands: list[dict] | None = None) -> float:
    bands = bands if bands is not None else _plan_layout(questions)
    return HEADER_H + sum(b['height'] + GRID_SEC_GAP for b in bands)


def capacity_info(questions: list[Question], bands: list[dict] | None = None) -> dict:
    used  = _height_used(questions, bands)
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
# QUESTION / ANSWER FILE PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_questions_file(path: str) -> tuple[list[Question], bool]:
    """
    Parse a plain-text question definition file.

    First non-blank, non-comment line must be True or False (shuffle flag).

    Remaining lines (one question each):
      2 / 3 / 4 / 5 / 6   → MCQQuestion with that many options
      n3                   → NumericQuestion(n_digits=3)
      n5.2                 → NumericQuestion(n_digits=5, has_decimal=True, decimal_pos=2)

    Returns (questions, shuffle).
    """
    questions: list[Question] = []
    shuffle: bool = False
    shuffle_read = False

    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if not shuffle_read:
                low = line.lower()
                if low in ("true", "false"):
                    shuffle = low == "true"
                    shuffle_read = True
                    continue
                raise ValueError(
                    f"{path}:{lineno}: first data line must be True or False (shuffle flag), got {line!r}"
                )

            low = line.lower()
            try:
                if low.startswith("n"):
                    rest = low[1:]
                    if "." in rest:
                        n_str, dp_str = rest.split(".", 1)
                        questions.append(NumericQuestion(
                            n_digits=int(n_str),
                            has_decimal=True,
                            decimal_pos=int(dp_str),
                        ))
                    else:
                        questions.append(NumericQuestion(n_digits=int(rest)))
                else:
                    questions.append(MCQQuestion(n_options=int(line)))
            except (ValueError, TypeError) as e:
                raise ValueError(f"{path}:{lineno}: cannot parse {line!r} — {e}") from e

    if not questions:
        raise ValueError(f"{path}: no questions found")
    return questions, shuffle


def parse_answers_file(path: str, questions: list[Question]) -> dict[int, str]:
    """
    Parse a plain-text answer file into {1-based original question number → answer}.

    Format (one answer per non-blank, non-comment line, same order as questions):
      MCQ    : a / b / c / d / e / f  (case-insensitive)
      Numeric: any digit string, e.g. 42  or  3.14
    """
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    answers: dict[int, str] = {}
    for i, (q, ans) in enumerate(zip(questions, lines), 1):
        if isinstance(q, MCQQuestion):
            answers[i] = ans.lower()
        else:
            answers[i] = ans
    return answers


def parse_json_file(path: str) -> tuple[list[Question], bool, dict | None, dict]:
    """
    Parse a JSON test definition file.

    Minimal format:
        { "questions": [ {"type": "mcq", "options": 4}, ... ] }

    Full format:
        {
          "title":     "Exam title",      // optional — passed to generate()
          "variant":   "A",               // optional
          "date":      "2026-09-08",      // optional
          "shuffle":   false,             // optional, default false
          "questions": [
            {"type": "mcq",     "options": 4},
            {"type": "mcq",     "options": 3},
            {"type": "numeric", "digits":  4},
            {"type": "numeric", "digits":  5, "decimal": 2}
          ],
          "answers": ["b", "a", "1234", "12.34"]  // optional; or per-question "answer" field
        }

    Returns (questions, shuffle, answers_dict_or_None, meta_dict).
    meta_dict contains "title", "variant", "date" if present in the file.
    answers_dict maps 1-based question number → answer string.
    """
    import json as _json

    with open(path) as f:
        data = _json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level value must be an object")
    if "questions" not in data:
        raise ValueError(f"{path}: missing required key \"questions\"")

    shuffle: bool = bool(data.get("shuffle", False))
    meta: dict = {k: data[k] for k in ("title", "variant", "date") if k in data}

    questions: list[Question] = []
    inline_answers: list[str | None] = []

    for i, item in enumerate(data["questions"], 1):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: question {i} must be an object")
        qtype = item.get("type", "").lower()
        if qtype == "mcq":
            n = int(item.get("options", 4))
            q = MCQQuestion(n_options=n)
        elif qtype == "numeric":
            digits  = int(item.get("digits", 4))
            decimal = int(item.get("decimal", 0))
            q = NumericQuestion(
                n_digits=digits,
                has_decimal=(decimal > 0),
                decimal_pos=decimal if decimal > 0 else 2,
            )
        else:
            raise ValueError(f"{path}: question {i} has unknown type {item.get('type')!r}"
                             " — use \"mcq\" or \"numeric\"")
        q.points  = int(item.get("points",  1))
        q.partial = bool(item.get("partial", False))
        q.validate()
        questions.append(q)
        inline_answers.append(item.get("answer"))

    if not questions:
        raise ValueError(f"{path}: no questions found")

    # Build answers dict — top-level array takes priority over inline fields
    answers: dict[int, str] | None = None
    top_answers = data.get("answers")
    if top_answers is not None:
        if not isinstance(top_answers, list):
            raise ValueError(f"{path}: \"answers\" must be an array")
        answers = {i: str(a) for i, a in enumerate(top_answers, 1) if a is not None}
    elif any(a is not None for a in inline_answers):
        answers = {i: str(a) for i, a in enumerate(inline_answers, 1) if a is not None}

    return questions, shuffle, answers, meta


def _answer_to_boxes(ans_str: str, q: NumericQuestion) -> dict:
    """
    Map an answer string to per-box characters.

    Returns {'digits': list[str], 'decimal': bool}
      digits[d] — character to draw in digit box d (empty string = blank)
      decimal   — True means draw '.' in the decimal box
    """
    digits = [""] * q.n_digits
    decimal = False
    ans_str = ans_str.strip()
    if not ans_str:
        return {"digits": digits, "decimal": decimal}

    if q.has_decimal:
        decimal = True
        if "." in ans_str:
            int_part, frac_part = ans_str.split(".", 1)
        else:
            int_part, frac_part = ans_str, ""
        # integer part: right-align in [0 .. decimal_pos-1]
        for i, ch in enumerate(int_part[-q.decimal_pos:].rjust(q.decimal_pos)):
            digits[i] = ch if ch != " " else ""
        # fractional part: left-align in [decimal_pos .. n_digits-1]
        frac_slots = q.n_digits - q.decimal_pos
        for i, ch in enumerate((frac_part + " " * frac_slots)[:frac_slots]):
            digits[q.decimal_pos + i] = ch if ch != " " else ""
    else:
        # right-align in all slots
        for i, ch in enumerate(ans_str[-q.n_digits:].rjust(q.n_digits)):
            digits[i] = ch if ch != " " else ""

    return {"digits": digits, "decimal": decimal}


# ══════════════════════════════════════════════════════════════════════════════
# QR CODE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _make_qr(data: str) -> ImageReader:
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=4,
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

def _draw_corner_qrs(c: canvas.Canvas, sheet_id: str) -> None:
    """Each QR encodes its corner prefix + sheet_id: TL<id>, TR<id>, BL<id>, BR<id>."""
    for corner, (x, y) in [
        ("TL", (QR_EDGE,                    PAGE_H - QR_EDGE - QR_SIZE)),
        ("TR", (PAGE_W - QR_EDGE - QR_SIZE, PAGE_H - QR_EDGE - QR_SIZE)),
        ("BL", (QR_EDGE,                    QR_EDGE)),
        ("BR", (PAGE_W - QR_EDGE - QR_SIZE, QR_EDGE)),
    ]:
        c.drawImage(_make_qr(f"{corner}{sheet_id}"),
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

    # Title row — 3 mm breathing room below the top content border
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(colors.black)
    _H_PAD = 3 * mm   # inward padding from content border for header text
    c.drawString(CL + _H_PAD, y - 8 * mm, title)

    c.setFont("Helvetica", 8)
    c.drawRightString(CR - _H_PAD, y - 8 * mm, f"Variant {variant}   {date}")

    # Name row — text first, underline 1.5 mm below its baseline
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.25, 0.25, 0.25)
    c.drawString(CL + _H_PAD, y - 15 * mm, "Full name:")
    c.drawString(CR - 26 * mm, y - 15 * mm, "Class:")
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.line(CL + 22 * mm, y - 16.5 * mm, CR - 28 * mm, y - 16.5 * mm)
    c.line(CR - 15 * mm, y - 16.5 * mm, CR - _H_PAD,   y - 16.5 * mm)

    # Bottom border of header
    c.setStrokeColorRGB(0.35, 0.35, 0.35)
    c.setLineWidth(0.6)
    c.line(CL, y - HEADER_H, CR, y - HEADER_H)



def _draw_mcq_section(c: canvas.Canvas, y_top: float,
                      questions: list[MCQQuestion],
                      q_start: int, *,
                      draw_w: float = None,
                      x_start: float = None,
                      q_answers: dict = None) -> float:
    """
    Draw one MCQ grid section.
    draw_w   : explicit width (defaults to CW or distance to CR from x_start).
    x_start  : left edge (defaults to CL); used when placed beside numeric.
    q_answers: {sheet_q_num → answer_str} — fills the selected option black.
    Returns height consumed.
    """
    max_opts = max(q.n_options for q in questions)
    sec_h    = _mcq_section_h(max_opts)
    opts     = "abcdef"
    n        = len(questions)
    x0       = x_start if x_start is not None else CL
    w        = draw_w if draw_w is not None else (CR - x0)

    # ── Question-number header row ─────────────────────────────────────────────
    c.setFont("Helvetica-Bold", 8)
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
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(colors.black)
        c.drawCentredString(x0 + GRID_LBL_W / 2,
                            row_bot + GRID_OPT_H / 2 - 1.5 * mm,
                            opts[opt_i])

        # One box per question column
        for q_i, q in enumerate(questions):
            cell_x = x0 + GRID_LBL_W + q_i * GRID_Q_W
            cell_y = row_bot
            side = min(GRID_Q_W, GRID_OPT_H) - 1.4 * mm   # square side
            bx = cell_x + (GRID_Q_W   - side) / 2
            by = cell_y + (GRID_OPT_H - side) / 2
            bw = side
            bh = side

            if opt_i < q.n_options:
                selected_raw = (q_answers or {}).get(q_start + q_i, "")
                selected_set = {x.strip().lower()
                                for x in str(selected_raw).replace(",", " ").split()
                                if x.strip()}
                if opts[opt_i] in selected_set:
                    c.setFillColor(colors.black)   # filled = correct answer
                else:
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
                   y_top: float, rows: list, q_start: int,
                   q_answers: dict = None) -> float:
    """
    Shared helper: draw pre-packed numeric rows inside a border.
    q_answers: {sheet_q_num → answer_str} — writes digits into each box.
    Returns height consumed.
    """
    total_h = len(rows) * NUM_ROW_H
    box_w   = x_right - x_left

    c.setStrokeColor(colors.black)
    c.setLineWidth(0.7)
    c.rect(x_left, y_top - total_h, box_w, total_h, stroke=1, fill=0)

    for row_idx, row in enumerate(rows):
        row_y   = y_top - row_idx * NUM_ROW_H
        box_top = row_y - (NUM_ROW_H - NUM_BOX_H) / 2 - NUM_BOX_H

        if row_idx > 0:
            c.setStrokeColorRGB(0.75, 0.75, 0.75)
            c.setLineWidth(0.25)
            c.line(x_left, row_y, x_right, row_y)

        for q, x_off, orig_idx in row:
            q_num = q_start + orig_idx
            col_x = x_left + x_off

            c.setFont("Helvetica-Bold", 7.5)
            c.setFillColor(colors.black)
            c.drawString(col_x + 1.5 * mm,
                         box_top + NUM_BOX_H / 2 - 2.5 * mm,
                         f"{q_num}.")

            box_data = None
            if q_answers and q_num in q_answers:
                box_data = _answer_to_boxes(q_answers[q_num], q)

            bx = col_x + NUM_Q_LBL_W
            char_y = box_top + (NUM_BOX_H - 3.5 * mm) / 2   # baseline for answer chars

            for d in range(q.n_digits):
                if q.has_decimal and d == q.decimal_pos:
                    c.setFillColor(colors.white)
                    c.setStrokeColor(colors.black)
                    c.setLineWidth(0.5)
                    c.setDash([1, 4])
                    c.rect(bx, box_top, NUM_DOT_BOX_W, NUM_BOX_H, stroke=1, fill=1)
                    c.setDash([])
                    if box_data and box_data["decimal"]:
                        c.setFont("Helvetica-Bold", 11)
                        c.setFillColor(colors.black)
                        c.drawCentredString(bx + NUM_DOT_BOX_W / 2, char_y, ".")
                    bx += NUM_DOT_BOX_W + NUM_BOX_GAP

                c.setFillColor(colors.white)
                c.setStrokeColor(colors.black)
                c.setLineWidth(0.5)
                c.setDash([1, 4])
                c.rect(bx, box_top, NUM_BOX_W, NUM_BOX_H, stroke=1, fill=1)
                c.setDash([])
                if box_data:
                    ch = box_data["digits"][d] if d < len(box_data["digits"]) else ""
                    if ch:
                        c.setFont("Helvetica-Bold", 11)
                        c.setFillColor(colors.black)
                        c.drawCentredString(bx + NUM_BOX_W / 2, char_y, ch)
                bx += NUM_BOX_W + NUM_BOX_GAP

    return total_h


def _draw_numeric_section(c: canvas.Canvas, y_top: float,
                          rows: list, q_start: int,
                          x_right: float = None,
                          q_answers: dict = None) -> float:
    """Full-width (or right-bounded) numeric section from pre-packed rows."""
    return _draw_num_rows(c, CL, x_right if x_right is not None else CR,
                          y_top, rows, q_start, q_answers)


def _draw_numeric_panel(c: canvas.Canvas, x_left: float, y_top: float,
                        rows: list, q_start: int,
                        q_answers: dict = None) -> None:
    """Numeric side panel beside an MCQ section (pre-packed rows)."""
    _draw_num_rows(c, x_left, CR, y_top, rows, q_start, q_answers)


def _guide_label(c: canvas.Canvas, cx: float, y: float,
                 text: str, *, bad: bool) -> None:
    c.setFont("Helvetica-Bold", 7.5)
    if bad:
        c.setFillColorRGB(0.75, 0.1, 0.1)
    else:
        c.setFillColorRGB(0.1, 0.55, 0.1)
    c.drawCentredString(cx, y, text)


# Stroke paths for handwritten-style digits 0-9.
# Each digit is a list of strokes; each stroke is (x, y) tuples
# in normalised [0,1]² space where (0,0) = bottom-left, (1,1) = top-right.
_DIGIT_STROKES: dict[int, list[list[tuple[float, float]]]] = {
    0: [[(0.50,0.06),(0.24,0.08),(0.08,0.26),(0.06,0.50),
         (0.08,0.74),(0.24,0.92),(0.50,0.94),
         (0.76,0.92),(0.92,0.74),(0.94,0.50),
         (0.92,0.26),(0.76,0.08),(0.50,0.06)]],

    1: [[(0.32,0.76),(0.50,0.94),(0.50,0.06)]],

    2: [[(0.20,0.74),(0.22,0.88),(0.38,0.94),(0.60,0.92),
         (0.78,0.80),(0.80,0.64),(0.70,0.52),(0.52,0.42),
         (0.18,0.06),(0.82,0.06)]],

    3: [[(0.20,0.88),(0.50,0.94),(0.76,0.82),(0.76,0.64),
         (0.56,0.54),(0.50,0.50),(0.58,0.46),(0.78,0.34),
         (0.76,0.16),(0.50,0.06),(0.22,0.14)]],

    4: [[(0.68,0.94),(0.68,0.06)],
        [(0.68,0.94),(0.14,0.38),(0.84,0.38)]],

    5: [[(0.78,0.94),(0.22,0.94),(0.18,0.54),
         (0.46,0.60),(0.72,0.54),(0.82,0.36),
         (0.76,0.14),(0.50,0.06),(0.20,0.14)]],

    6: [[(0.74,0.90),(0.46,0.96),(0.20,0.78),(0.12,0.50),
         (0.12,0.28),(0.24,0.10),(0.50,0.04),(0.76,0.12),
         (0.86,0.34),(0.78,0.54),(0.50,0.62),
         (0.22,0.54),(0.12,0.34)]],

    7: [[(0.16,0.94),(0.84,0.94),(0.38,0.06)]],

    8: [[(0.50,0.50),(0.24,0.56),(0.14,0.72),(0.20,0.88),
         (0.50,0.94),(0.80,0.88),(0.86,0.72),(0.76,0.56),
         (0.50,0.50),(0.28,0.44),(0.14,0.26),(0.20,0.10),
         (0.50,0.04),(0.80,0.10),(0.86,0.28),(0.74,0.44),(0.50,0.50)]],

    9: [[(0.82,0.60),(0.82,0.06)],
        [(0.82,0.60),(0.70,0.88),(0.46,0.96),(0.20,0.84),
         (0.14,0.62),(0.22,0.40),(0.48,0.34),
         (0.70,0.42),(0.82,0.62)]],
}


def _mnist_digit_image(digit: int, w_mm: float, h_mm: float) -> ImageReader:
    """Render digit as an MNIST-style thick-stroke image and return an ImageReader."""
    buf = io.BytesIO(_mnist_digit_png(digit, w_mm, h_mm))
    return ImageReader(buf)


def _mnist_digit_png(digit: int, w_mm: float, h_mm: float) -> bytes:
    """Return PNG bytes for the digit — cached since the guide always uses the same sizes."""
    key = (digit, round(w_mm, 3), round(h_mm, 3))
    cached = _PNG_CACHE.get(key)
    if cached is not None:
        return cached

    dpi   = 220
    w_px  = max(24, int(w_mm / 25.4 * dpi))
    h_px  = max(32, int(h_mm / 25.4 * dpi))
    sw    = max(3, int(min(w_px, h_px) * 0.14))

    img  = _PILImage.new("L", (w_px, h_px), 255)
    draw = _PILDraw.Draw(img)
    pad_x = w_px * 0.12
    pad_y = h_px * 0.06
    dw    = w_px - 2 * pad_x
    dh    = h_px - 2 * pad_y

    for stroke in _DIGIT_STROKES[digit]:
        pts = [(int(pad_x + x * dw), int(h_px - pad_y - y * dh)) for x, y in stroke]
        if len(pts) >= 2:
            draw.line(pts, fill=30, width=sw)

    img = img.filter(_PILFilter.GaussianBlur(radius=sw * 0.45)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    _PNG_CACHE[key] = data
    return data


_PNG_CACHE: dict[tuple, bytes] = {}


def _draw_guide(c: canvas.Canvas) -> None:
    """
    Instruction strip drawn between the two bottom corner QR codes.

    Available region (portrait A4, origin bottom-left):
      x : QR_EDGE+QR_SIZE … PAGE_W-QR_EDGE-QR_SIZE  ≈ 28 … 182 mm  (154 mm wide)
      y : QR_EDGE          … QR_EDGE+QR_SIZE          ≈  6 … 28 mm  (22 mm tall)
    """
    # Band geometry
    gx_l = QR_EDGE + QR_SIZE                    # ≈ 28 mm from left
    gx_r = PAGE_W - QR_EDGE - QR_SIZE           # ≈ 182 mm from left
    gy_b = QR_EDGE                              # 6 mm from bottom
    gy_t = QR_EDGE + QR_SIZE                   # 28 mm from bottom
    gcy  = (gy_b + gy_t) / 2                   # vertical centre ≈ 17 mm

    _PAD  = 2.5 * mm
    box_s = min(GRID_Q_W, GRID_OPT_H) - 1.4 * mm   # real MCQ box side ≈ 3.6 mm

    # Row positions inside the 22 mm band
    lbl_y = gcy + 4.5 * mm    # label baseline (upper row)
    ex_y  = gcy - box_s - 1.2 * mm  # checkbox baseline (lower row)

    # ── Left section: MCQ marking examples ──────────────────────────────────
    c.setFont("Helvetica-Bold", 6.5)
    c.setFillColorRGB(0.25, 0.25, 0.25)
    c.drawString(gx_l + _PAD, lbl_y, "MARK ANSWER:")

    ex_x = gx_l + _PAD + 28 * mm

    # Wrong ①: dot — barely touched
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.rect(ex_x, ex_y, box_s, box_s, stroke=1, fill=1)
    c.setFillColor(colors.black)
    c.circle(ex_x + box_s / 2, ex_y + box_s / 2, 0.55 * mm, stroke=0, fill=1)
    _guide_label(c, ex_x + box_s / 2, ex_y - 3.0 * mm, "✗", bad=True)

    # Wrong ②: X-mark
    ex_x += box_s + 5 * mm
    c.setFillColor(colors.white)
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.rect(ex_x, ex_y, box_s, box_s, stroke=1, fill=1)
    m = 0.65 * mm
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.9)
    c.line(ex_x + m,         ex_y + m,         ex_x + box_s - m, ex_y + box_s - m)
    c.line(ex_x + box_s - m, ex_y + m,         ex_x + m,         ex_y + box_s - m)
    _guide_label(c, ex_x + box_s / 2, ex_y - 3.0 * mm, "✗", bad=True)

    # Correct: completely filled black
    ex_x += box_s + 5 * mm
    c.setFillColor(colors.black)
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.rect(ex_x, ex_y, box_s, box_s, stroke=1, fill=1)
    _guide_label(c, ex_x + box_s / 2, ex_y - 3.0 * mm, "✓", bad=False)

    # Vertical divider
    div_x = gx_l + 66 * mm
    c.setStrokeColorRGB(0.65, 0.65, 0.65)
    c.setLineWidth(0.3)
    c.line(div_x, gy_b + 2 * mm, div_x, gy_t - 2 * mm)

    # ── Right section: digit writing examples 0–9 ───────────────────────────
    sec_x = div_x + _PAD

    c.setFont("Helvetica-Bold", 6.5)
    c.setFillColorRGB(0.25, 0.25, 0.25)
    c.drawString(sec_x, lbl_y, "WRITE NUMBERS CLEARLY:")

    # Digit boxes: scaled to fit the 22 mm band height
    sc      = 0.82
    dw      = NUM_BOX_W * sc
    dh      = NUM_BOX_H * sc          # ≈ 6.6 mm
    dgap    = 0.5 * mm
    dbox_y  = gcy - dh / 2 - 0.5 * mm

    total_box_w = 10 * dw + 9 * dgap
    digits_x0 = gx_r - total_box_w - 1.5 * mm   # right-align, 1.5 mm gap before QR
    for d in range(10):
        bx = digits_x0 + d * (dw + dgap)
        c.setFillColor(colors.white)
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.45)
        c.setDash([1, 3])
        c.rect(bx, dbox_y, dw, dh, stroke=1, fill=1)
        c.setDash([])
        digit_img = _mnist_digit_image(d, dw / mm, dh / mm)
        c.drawImage(digit_img, bx, dbox_y, width=dw, height=dh, mask="auto")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATE
# ══════════════════════════════════════════════════════════════════════════════

def generate(
    questions: list[Question],
    *,
    test_title: str  = "Test",
    test_id:    str  = "TEST001",
    variant:    str  = "A",
    date:       str  = "",
    output:     str  = "answer_sheet.pdf",
    shuffle:    bool = False,
    answers:    dict = None,   # {original 1-based q# → answer str}; fills key PDF
    is_key:     bool = False,  # True → add "ANSWER KEY" label to header
) -> tuple[str, dict, list[int], str]:
    """
    Generate an A4 answer sheet PDF.

    answers=dict  : if given, every answer is drawn filled in — use for the key.
    is_key=True   : appends "— ANSWER KEY" to the PDF title.
    shuffle=True  : reorders questions for minimal page height.

    Returns (output_path, capacity_info_dict, order, sheet_id).
    """
    order = list(range(1, len(questions) + 1))
    if shuffle:
        questions, order = _optimal_order(questions)

    for q in questions:
        q.validate()

    bands = _plan_layout(questions)          # compute once; reused for cap, draw, json
    cap   = capacity_info(questions, bands)
    if not cap["fits"]:
        raise ValueError(
            f"Questions overflow by {-cap['avail_mm']:.1f} mm — reduce question count."
        )

    # Map original q# → answer/points/partial to sheet q# using the order array.
    sheet_answers: dict = {}
    sheet_points:  dict = {}
    sheet_partial: dict = {}
    for sheet_pos, orig_q in enumerate(order, 1):
        if answers and orig_q in answers:
            sheet_answers[sheet_pos] = answers[orig_q]
        q = questions[orig_q - 1]
        sheet_points[sheet_pos]  = getattr(q, "points",  1)
        sheet_partial[sheet_pos] = getattr(q, "partial", False)

    sheet_id = secrets.token_urlsafe(9)[:9]   # 9-char id → QR = "TL" + 9 = 11 chars

    pdf_title = test_title + (" — ANSWER KEY" if is_key else "")
    c = canvas.Canvas(output, pagesize=A4)
    c.setTitle(pdf_title)
    c.setAuthor("HVH Testing System")

    # 1. Corner QR codes
    _draw_corner_qrs(c, sheet_id)

    # 2. Student ID between top QRs
    _draw_student_id(c)

    # 3. Outer content border
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.8)
    c.rect(CL, CB, CW, CH, stroke=1, fill=0)

    # 4. Header  (answer key gets a red "ANSWER KEY" stamp)
    _draw_header(c, test_title, variant, date)
    if is_key:
        c.setFont("Helvetica-Bold", 9)
        c.setFillColorRGB(0.75, 0.1, 0.1)
        c.drawRightString(CR - 3 * mm, CT - 5 * mm, "ANSWER KEY")

    # 5. Question sections
    qa = (sheet_answers or None) if is_key else None
    y = CT - HEADER_H
    for band in bands:
        if band['kind'] == 'mcq':
            _draw_mcq_section(c, y, band['mcq'], band['mcq_q0'],
                              draw_w=band['mcq_draw_w'], q_answers=qa)
            if band['side_rows']:
                num_x_right = (band['side_x'] + band['side_num_nat_w']
                               if band['side_mcq2'] else CR)
                _draw_num_rows(c, band['side_x'], num_x_right, y,
                               band['side_rows'], band['side_q0'], qa)
                if band['side_mcq2']:
                    _draw_mcq_section(c, y, band['side_mcq2'], band['side_mcq2_q0'],
                                      x_start=band['side_mcq2_x'], q_answers=qa)
        else:
            _draw_numeric_section(c, y, band['num_rows'], band['num_q0'],
                                  x_right=CL + band['num_draw_w'], q_answers=qa)
            if band['side_mcq']:
                _draw_mcq_section(c, y, band['side_mcq'], band['side_mcq_q0'],
                                  x_start=band['side_mcq_x'], q_answers=qa)

        y -= band['height'] + GRID_SEC_GAP

    # Instruction strip between the bottom QR codes
    _draw_guide(c)

    c.save()

    # Export box layout JSON (blank sheet only)
    if not is_key:
        layout_json = output.replace(".pdf", "_layout.json")
        export_layout_json(questions, layout_json, sheet_id=sheet_id, bands=bands,
                           answers=sheet_answers if sheet_answers else None,
                           question_points=sheet_points,
                           question_partial=sheet_partial)

    return output, cap, order, sheet_id


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT EXPORT  (box positions for the scanner)
# ══════════════════════════════════════════════════════════════════════════════

def export_layout_json(questions: list[Question], path: str, sheet_id: str = "",
                       bands: list[dict] | None = None,
                       answers: dict | None = None,
                       question_points: dict | None = None,
                       question_partial: dict | None = None) -> None:
    """
    Write all MCQ checkbox and numeric digit box positions to a JSON file.

    Coordinate system: mm from top-left corner of the A4 page (y increases down).
    This matches what scanner.py expects after perspective-warping the photo.
    """
    import json as _json

    # ReportLab uses points with origin at bottom-left.
    # Convert:  x_mm = x_pt / mm
    #           y_mm_from_top = PAGE_H_mm - y_pt / mm
    PAGE_H_MM = PAGE_H / mm   # 297.0

    def _pt_to_mm(x_pt: float, y_pt: float, w_pt: float, h_pt: float) -> dict:
        """Convert a ReportLab rect (bottom-left origin, points) to top-left-origin mm."""
        return {
            "x_mm": x_pt / mm,
            "y_mm": PAGE_H_MM - (y_pt + h_pt) / mm,
            "w_mm": w_pt / mm,
            "h_mm": h_pt / mm,
        }

    mcq_boxes: list[dict] = []
    num_boxes: list[dict] = []

    y = CT - HEADER_H   # same starting y as generate()
    bands = bands if bands is not None else _plan_layout(questions)

    for band in bands:
        if band["kind"] == "mcq":
            # ── main MCQ grid ──────────────────────────────────────────────
            _collect_mcq_boxes(mcq_boxes, band["mcq"], band["mcq_q0"],
                               x0=CL, y_top=y, _pt_to_mm=_pt_to_mm)

            if band["side_rows"]:
                # numeric side panel
                num_x_right = (band["side_x"] + band["side_num_nat_w"]
                               if band["side_mcq2"] else CR)
                _collect_num_boxes(num_boxes, band["side_rows"], band["side_q0"],
                                   x_left=band["side_x"], y_top=y, _pt_to_mm=_pt_to_mm)
                if band["side_mcq2"]:
                    _collect_mcq_boxes(mcq_boxes, band["side_mcq2"], band["side_mcq2_q0"],
                                       x0=band["side_mcq2_x"], y_top=y, _pt_to_mm=_pt_to_mm)
        else:
            # ── numeric section ────────────────────────────────────────────
            _collect_num_boxes(num_boxes, band["num_rows"], band["num_q0"],
                               x_left=CL, y_top=y, _pt_to_mm=_pt_to_mm)
            if band["side_mcq"]:
                _collect_mcq_boxes(mcq_boxes, band["side_mcq"], band["side_mcq_q0"],
                                   x0=band["side_mcq_x"], y_top=y, _pt_to_mm=_pt_to_mm)

        y -= band["height"] + GRID_SEC_GAP

    # ── Student ID boxes ──────────────────────────────────────────────────────
    id_start_x = (_ID_ZONE_L + _ID_ZONE_R - _ID_STRIP_W) / 2
    id_box_y   = _ID_ZONE_CY - ID_BOX_H / 2
    id_boxes = [
        {
            **_pt_to_mm(id_start_x + i * (ID_BOX_W + ID_BOX_GAP),
                        id_box_y, ID_BOX_W, ID_BOX_H),
            "digit": i + 1,
        }
        for i in range(ID_BOXES)
    ]

    data = {
        "sheet_id":  sheet_id,
        "page_w_mm": PAGE_W / mm,
        "page_h_mm": PAGE_H / mm,
        "mcq_boxes": mcq_boxes,
        "num_boxes": num_boxes,
        "id_boxes":  id_boxes,
    }
    if answers:
        data["answers"] = {f"Q{int(k):02d}": str(v) for k, v in answers.items()}
    if question_points:
        data["question_points"] = {f"Q{int(k):02d}": v
                                   for k, v in question_points.items()}
    if question_partial:
        data["question_partial"] = {f"Q{int(k):02d}": v
                                    for k, v in question_partial.items()
                                    if v}
    with open(path, "w") as f:
        _json.dump(data, f, indent=2)


def _collect_mcq_boxes(out: list, questions: list, q_start: int,
                       x0: float, y_top: float, _pt_to_mm) -> None:
    """Collect MCQ checkbox positions (same geometry as _draw_mcq_section)."""
    max_opts = max(q.n_options for q in questions)
    for opt_i in range(max_opts):
        row_top = y_top - GRID_NUM_H - opt_i * GRID_OPT_H
        row_bot = row_top - GRID_OPT_H
        for q_i, q in enumerate(questions):
            if opt_i >= q.n_options:
                continue   # greyed-out cell — not a real checkbox
            cell_x = x0 + GRID_LBL_W + q_i * GRID_Q_W
            side   = min(GRID_Q_W, GRID_OPT_H) - 1.4 * mm
            bx     = cell_x + (GRID_Q_W   - side) / 2
            by_    = row_bot + (GRID_OPT_H - side) / 2
            box    = _pt_to_mm(bx, by_, side, side)
            box["q"]   = q_start + q_i
            box["opt"] = "abcdef"[opt_i]
            out.append(box)


def _collect_num_boxes(out: list, rows: list, q_start: int,
                       x_left: float, y_top: float, _pt_to_mm) -> None:
    """Collect numeric digit/decimal box positions (same geometry as _draw_num_rows)."""
    for row_idx, row in enumerate(rows):
        row_y   = y_top - row_idx * NUM_ROW_H
        box_top = row_y - (NUM_ROW_H - NUM_BOX_H) / 2 - NUM_BOX_H

        for q, x_off, orig_idx in row:
            q_num = q_start + orig_idx
            bx = x_left + x_off + NUM_Q_LBL_W
            digit_idx = 0
            for d in range(q.n_digits):
                if q.has_decimal and d == q.decimal_pos:
                    box = _pt_to_mm(bx, box_top, NUM_DOT_BOX_W, NUM_BOX_H)
                    box["q"]          = q_num
                    box["digit"]      = digit_idx
                    box["is_decimal"] = True
                    out.append(box)
                    bx += NUM_DOT_BOX_W + NUM_BOX_GAP
                    digit_idx += 1

                box = _pt_to_mm(bx, box_top, NUM_BOX_W, NUM_BOX_H)
                box["q"]          = q_num
                box["digit"]      = digit_idx
                box["is_decimal"] = False
                out.append(box)
                bx += NUM_BOX_W + NUM_BOX_GAP
                digit_idx += 1


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
        NumericQuestion(n_digits=5, has_decimal=True,  decimal_pos=3),  # Q22  e.g. "123.456"
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

    path, cap, order, sheet_id = generate(
        questions,
        test_title = "Mathematics — Final Exam 2026",
        test_id    = "MATH2026A",
        variant    = "A",
        date       = "18 Aug 2026",
        output     = "answer_sheet.pdf",
        shuffle    = True,          # set False to keep original order
    )
    print(f"Generated : {path}")
    print(f"Sheet ID  : {sheet_id}")
    print(f"  TL{sheet_id} / TR{sheet_id} / BL{sheet_id} / BR{sheet_id}")
    print(f"Remaining : ~{cap['rem_mcq_4']} MCQ-4  or  ~{cap['rem_numeric']} numeric")
    print(f"Sheet order (original Q# → sheet position):")
    for pos, orig in enumerate(order, 1):
        print(f"  Sheet Q{pos:02d} ← original Q{orig:02d}")
    print()
