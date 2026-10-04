from pathlib import Path
import re

import pymupdf
import pytesseract
from pytesseract import Output
from PIL import Image

from ingestion.universal_structure import UniversalStructureDetector
from ingestion.universal_row_extractor import UniversalRowExtractor


PDF_PATH = Path("storage/Canara_2.pdf")

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

DATE_START = re.compile(
    r"^\d{1,2}-[A-Za-z]{3,9}-\d{4}"
)


def main():

    print("=" * 60)
    print("FINORA OCR ROW PARSING TEST")
    print("=" * 60)

    pdf = pymupdf.open(str(PDF_PATH))

    structure_detector = UniversalStructureDetector()
    row_extractor = UniversalRowExtractor()

    for page_number, page in enumerate(pdf, start=1):

        # High-resolution rendering for OCR.
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(4, 4),
            alpha=False,
        )

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples,
        )

        print(
            f"\nPage {page_number}: "
            f"{pix.width}x{pix.height}"
        )

        data = pytesseract.image_to_data(
            image,
            lang="eng",
            config="--psm 6",
            output_type=Output.DICT,
        )

        # ---------------------------------------------------------
        # Build OCR words
        # ---------------------------------------------------------

        words = []

        for i in range(len(data["text"])):

            text = str(data["text"][i]).strip()

            if not text:
                continue

            try:
                confidence = float(data["conf"][i])
            except Exception:
                confidence = -1

            if confidence < 20:
                continue

            left = int(data["left"][i])
            top = int(data["top"][i])
            width = int(data["width"][i])
            height = int(data["height"][i])

            words.append(
                {
                    "text": text,
                    "x0": left,
                    "x1": left + width,
                    "top": top,
                    "bottom": top + height,
                    "block": data["block_num"][i],
                    "paragraph": data["par_num"][i],
                    "line": data["line_num"][i],
                }
            )

        print(f"OCR words: {len(words)}")

        # ---------------------------------------------------------
        # Detect statement structure
        # ---------------------------------------------------------

        structure = structure_detector._detect_page(
            words,
            page_number,
        )

        if not structure:

            print("Structure NOT detected.")
            continue

        print(
            f"Structure detected: "
            f"columns={len(structure['columns'])}, "
            f"score={structure['score']:.2f}"
        )

        for column in structure["columns"]:
            print(
                f"  {column['header']} "
                f"-> {column['semantic_type']}"
            )

        columns = structure["columns"]

        # ---------------------------------------------------------
        # Group OCR words into visual lines
        # ---------------------------------------------------------

        lines = {}

        for word in words:

            key = (
                word["block"],
                word["paragraph"],
                word["line"],
            )

            lines.setdefault(
                key,
                [],
            ).append(word)

        ordered_lines = []

        for line_words in lines.values():

            line_words.sort(
                key=lambda w: w["x0"]
            )

            line_text = " ".join(
                w["text"]
                for w in line_words
            ).strip()

            if not line_text:
                continue

            line_top = min(
                w["top"]
                for w in line_words
            )

            ordered_lines.append(
                (
                    line_top,
                    line_words,
                    line_text,
                )
            )

        ordered_lines.sort(
            key=lambda x: x[0]
        )

        # ---------------------------------------------------------
        # Only parse lines that begin with a transaction date.
        # ---------------------------------------------------------

        transaction_lines = [
            item
            for item in ordered_lines
            if DATE_START.match(item[2])
        ]

        print(
            f"Transaction-start lines: "
            f"{len(transaction_lines)}"
        )

        # ---------------------------------------------------------
        # Parse each line using the existing row parser.
        # ---------------------------------------------------------

        parsed_rows = []

        for index, (
            line_top,
            line_words,
            line_text,
        ) in enumerate(
            transaction_lines,
            start=1,
        ):

            try:
                row = row_extractor._parse_row(
                    line_words,
                    columns,
                )
            except Exception as exc:
                print(
                    f"\nROW {index}: ERROR: {exc}"
                )
                continue

            if not row:
                continue

            if not row_extractor._is_transaction(
                row
            ):
                continue

            parsed_rows.append(row)

            print()
            print(
                f"ROW {index}"
            )
            print(
                f"  OCR: {line_text}"
            )
            print(
                f"  PARSED: {row}"
            )

        print()
        print(
            f"Successfully parsed rows: "
            f"{len(parsed_rows)}"
        )

    pdf.close()

    print()
    print("=" * 60)
    print("OCR ROW PARSING TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()