from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pdfplumber


class PDFReader:
    """
    Universal PDF reader for Finora AI.

    Supports:
    - normal PDFs
    - password-protected PDFs
    - page-by-page text extraction
    """

    def __init__(
        self,
        file_path: str,
        password: Optional[str] = None,
    ):
        self.file_path = Path(file_path)
        self.password = password

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {self.file_path}"
            )

        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError(
                "The supplied file is not a PDF."
            )

    def get_page_count(self) -> int:

        with pdfplumber.open(
            self.file_path,
            password=self.password,
        ) as pdf:

            return len(pdf.pages)

    def extract_pages(self) -> List[Dict]:

        pages = []

        try:

            with pdfplumber.open(
                self.file_path,
                password=self.password,
            ) as pdf:

                for page_number, page in enumerate(
                    pdf.pages,
                    start=1,
                ):

                    text = (
                        page.extract_text()
                        or ""
                    )

                    pages.append(
                        {
                            "page_number": page_number,
                            "text": text,
                            "has_text": bool(
                                text.strip()
                            ),
                        }
                    )

        except Exception as exc:

            message = str(exc).lower()

            if (
                "password" in message
                or "encrypted" in message
                or "decrypt" in message
            ):

                raise ValueError(
                    "The PDF is password protected "
                    "or the supplied password is incorrect."
                ) from exc

            raise

        return pages

    def extract_text(self) -> str:

        pages = self.extract_pages()

        return "\n\n".join(
            f"--- PAGE {page['page_number']} ---\n"
            f"{page['text']}"
            for page in pages
        )