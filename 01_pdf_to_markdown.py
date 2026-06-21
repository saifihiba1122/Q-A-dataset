"""
01_pdf_to_markdown.py
======================
PDF -> Markdown converter (using pymupdf4llm)

Supports OCR for scanned PDFs (PDFs that have no selectable text-layer,
just page images). If a normal PDF gives 0-length output, it's almost
certainly scanned -- this script auto-detects that and retries with OCR.

REQUIRES (for OCR):
    1. Tesseract OCR installed system-wide: https://github.com/UB-Mannheim/tesseract/wiki
       or:  winget install -e --id UB-Mannheim.TesseractOCR
    2. pip install pymupdf4llm[ocr]

USAGE
-----
Single file:
    python 01_pdf_to_markdown.py --input sources/mydoc.pdf --output sources/mydoc.md

Pure folder (sab PDFs ek saath):
    python 01_pdf_to_markdown.py --input-dir sources --output-dir sources

Force OCR even if text-layer is detected:
    python 01_pdf_to_markdown.py --input-dir sources --output-dir sources --ocr
"""

import os
import glob
import argparse
from pathlib import Path
import pymupdf4llm


def convert_pdf_to_markdown(pdf_path: str, output_path: str, force_ocr: bool = False):
    print(f"Converting: {pdf_path}")

    md_text = ""
    if not force_ocr:
        md_text = pymupdf4llm.to_markdown(pdf_path)

    # Auto-detect scanned PDFs: if normal extraction gave (almost) nothing,
    # the PDF has no text-layer -- retry with OCR.
    if force_ocr or len(md_text.strip()) < 20:
        if not force_ocr:
            print("  No text-layer detected (scanned PDF) -- retrying with OCR...")
        try:
            md_text = pymupdf4llm.to_markdown(pdf_path, use_ocr=True)
        except Exception as e:
            print(f"  [error] OCR extraction failed: {e}")
            print("  Make sure Tesseract OCR is installed and on PATH (see script docstring).")
            return

    if len(md_text.strip()) < 20:
        print(f"  [warn] Still got almost no text from {pdf_path}. "
              f"The PDF may be low-quality scans, password-protected, or corrupted.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    word_count = len(md_text.split())
    print(f"  -> Saved {output_path} ({word_count} words)")


def main():
    parser = argparse.ArgumentParser(description="Convert PDF(s) to Markdown using pymupdf4llm.")
    parser.add_argument("--input", help="Path to a single PDF file")
    parser.add_argument("--output", help="Path for the output .md file (used with --input)")
    parser.add_argument("--input-dir", help="Folder containing multiple PDFs to convert")
    parser.add_argument("--output-dir", help="Folder to write .md files into (used with --input-dir)")
    parser.add_argument("--ocr", action="store_true", help="Force OCR extraction (use for known scanned PDFs)")
    args = parser.parse_args()

    if args.input:
        output = args.output or str(Path(args.input).with_suffix(".md"))
        convert_pdf_to_markdown(args.input, output, force_ocr=args.ocr)

    elif args.input_dir:
        output_dir = args.output_dir or args.input_dir
        pdf_files = glob.glob(os.path.join(args.input_dir, "*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {args.input_dir}")
            return
        for pdf_path in pdf_files:
            filename = Path(pdf_path).stem + ".md"
            output_path = os.path.join(output_dir, filename)
            convert_pdf_to_markdown(pdf_path, output_path, force_ocr=args.ocr)

    else:
        parser.error("Provide either --input <file.pdf> or --input-dir <folder>")


if __name__ == "__main__":
    main()