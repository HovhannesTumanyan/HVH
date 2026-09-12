#!/usr/bin/env python3
"""
HVH Testing System — command-line interface.

Generate a blank answer sheet and, optionally, a filled answer-key PDF.

Usage examples
--------------
  # From a JSON file (title/variant/date/answers embedded)
  python3 main.py exam.json

  # From plain-text files
  python3 main.py questions.txt
  python3 main.py questions.txt --answers answers.txt

  # Full options
  python3 main.py questions.txt --answers answers.txt \\
      --title "Mathematics Exam" --variant B --date 2026-08-24 \\
      --output sheet.pdf --key-output key.pdf --shuffle

JSON file format
----------------
  {
    "title":     "Mathematics Exam",      // optional
    "variant":   "A",                     // optional
    "date":      "2026-09-08",            // optional
    "shuffle":   false,                   // optional
    "questions": [
      {"type": "mcq",     "options": 4},
      {"type": "mcq",     "options": 3},
      {"type": "numeric", "digits":  4},
      {"type": "numeric", "digits":  5, "decimal": 2}
    ],
    "answers": ["b", "a", "1234", "12.34"]   // optional
  }

  Answers can also be per-question: {"type": "mcq", "options": 4, "answer": "b"}

Questions file format (one line per question)
---------------------------------------------
  4          → MCQ with 4 options  (a/b/c/d)
  3          → MCQ with 3 options  (a/b/c)
  n4         → Numeric, 4 digit boxes
  n5.2       → Numeric, 5 digit boxes, decimal point after position 2

  Lines starting with # or blank lines are ignored.

Answers file format (one line per question, same order)
-------------------------------------------------------
  b          → MCQ: option b
  a          → MCQ: option a
  1234       → Numeric: 1234
  12.345     → Numeric: 12.345  (for n5.2)

  Lines starting with # or blank lines are ignored.
"""

import argparse
import json
import sys
from pathlib import Path

from answer_sheet_gen import (
    generate,
    parse_questions_file,
    parse_answers_file,
    parse_json_file,
    report_capacity,
)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="hvh",
        description="HVH — generate answer sheets and answer keys",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("questions",
                   help="Questions file: .json (preferred) or plain-text .txt")
    p.add_argument("--answers", "-a", metavar="FILE",
                   help="Answers file (.txt) — also generates a filled answer-key PDF")
    p.add_argument("--title",   "-t", default="Test",
                   help="Test title (default: Test)")
    p.add_argument("--variant", "-v", default="A",
                   help="Variant letter shown on the sheet (default: A)")
    p.add_argument("--date",    "-d", default="",
                   help="Date string shown on the sheet")
    p.add_argument("--output",  "-o", default=None,
                   help="Output PDF for the blank answer sheet (default: <input>_sheet.pdf)")
    p.add_argument("--key-output", "-k", default=None, metavar="FILE",
                   help="Output PDF for the answer key (default: <input>_key.pdf)")
    p.add_argument("--shuffle", "-s", action="store_true",
                   help="Reorder questions for a more compact layout")
    p.add_argument("--capacity", "-c", action="store_true",
                   help="Print a capacity report and exit without generating PDFs")

    args = p.parse_args()

    # ── Parse question file (JSON or plain-text) ────────────────────────────
    qpath = args.questions
    json_meta: dict = {}

    if qpath.lower().endswith(".json"):
        try:
            questions, file_shuffle, json_answers, json_meta = parse_json_file(qpath)
        except Exception as e:
            sys.exit(f"Error reading JSON file: {e}")
        answers_from_json = json_answers  # may be None
    else:
        try:
            questions, file_shuffle = parse_questions_file(qpath)
        except Exception as e:
            sys.exit(f"Error reading questions file: {e}")
        answers_from_json = None

    shuffle = file_shuffle or args.shuffle
    print(f"Loaded {len(questions)} questions from '{qpath}' (shuffle={shuffle})")

    if args.capacity:
        report_capacity(questions)
        return

    # ── Resolve answers: CLI file > JSON embedded > none ─────────────────────
    answers = None
    if args.answers:
        try:
            answers = parse_answers_file(args.answers, questions)
        except Exception as e:
            sys.exit(f"Error reading answers file: {e}")
        print(f"Loaded {len(answers)} answers from '{args.answers}'")
    elif answers_from_json:
        answers = answers_from_json
        print(f"Loaded {len(answers)} answers from JSON")

    # CLI flags override JSON metadata; JSON metadata overrides argparse defaults
    title   = args.title   if args.title   != "Test" else json_meta.get("title",   args.title)
    variant = args.variant if args.variant != "A"    else json_meta.get("variant", args.variant)
    date    = args.date    if args.date    != ""     else json_meta.get("date",    args.date)

    common = dict(
        test_title=title,
        variant=variant,
        date=date,
        shuffle=shuffle,
    )

    # ── Derive output paths from input filename if not specified ────────────
    stem = Path(qpath).stem          # e.g. "test_exam"
    out_pdf = args.output     or f"{stem}_sheet.pdf"
    key_pdf = args.key_output or f"{stem}_key.pdf"

    # ── Generate blank answer sheet ──────────────────────────────────────────
    try:
        path, cap, order, sheet_id = generate(
            questions, output=out_pdf, answers=answers, **common
        )
    except ValueError as e:
        sys.exit(f"Layout error: {e}")

    print(f"\nAnswer sheet : {path}")
    print(f"Sheet ID     : {sheet_id}")
    if not cap["fits"]:
        print(f"WARNING: overflow by {-cap['avail_mm']:.1f} mm")

    # ── Write queue.json (sheet position → original question number) ─────────
    queue_path = f"{stem}_queue.json"
    queue_data = {f"Q{pos:02d}": orig for pos, orig in enumerate(order, 1)}
    with open(queue_path, "w") as f:
        json.dump(queue_data, f, indent=2)
    print(f"Queue        : {queue_path}")

    # ── Generate answer key PDF (auto when answers present) ─────────────────
    if answers:
        try:
            key_path, _, _, _ = generate(
                questions, output=key_pdf,
                answers=answers, is_key=True,
                **common,
            )
        except ValueError as e:
            sys.exit(f"Key layout error: {e}")
        print(f"Answer key   : {key_path}")


if __name__ == "__main__":
    main()
