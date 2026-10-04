from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pdfplumber

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


class PDFReader:
    """
    Universal PDF reader for Finora AI.

    Extraction strategy:

        1. pdfplumber
        2. PyMuPDF fallback

    Supports:
        - normal PDFs
        - password-protected PDFs
        - broken PDFMiner/pdfplumber documents
        - documents with unusual PDF structures

    This class is completely bank-agnostic.
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
                "PDFReader only supports PDF files."
            )

    # ============================================================
    # PAGE COUNT
    # ============================================================

    def get_page_count(self) -> int:
        """
        Return the number of pages.

        Tries pdfplumber first and PyMuPDF second.
        """

        try:
            with pdfplumber.open(
                self.file_path,
                password=self.password,
            ) as pdf:
                return len(pdf.pages)

        except Exception:
            if fitz is None:
                raise

            document = self._open_with_pymupdf()

            try:
                return len(document)
            finally:
                document.close()

    # ============================================================
    # TEXT EXTRACTION
    # ============================================================

    def extract_pages(self) -> List[Dict]:
        """
        Extract text page-by-page.

        Strategy:

            pdfplumber
                ↓
            if it fails
                ↓
            PyMuPDF

        The returned structure is identical regardless of the
        extraction engine.
        """

        try:
            pages = self._extract_with_pdfplumber()

            if self._has_useful_text(pages):
                return pages

        except Exception as exc:

            if self._is_password_error(exc):
                raise ValueError(
                    "The PDF is password protected or the supplied "
                    "password is incorrect."
                ) from exc

        # --------------------------------------------------------
        # PyMuPDF fallback
        # --------------------------------------------------------

        if fitz is None:
            raise RuntimeError(
                "pdfplumber could not extract the PDF and PyMuPDF "
                "is not installed."
            )

        try:
            pages = self._extract_with_pymupdf()

            if self._has_useful_text(pages):
                return pages

        except Exception as exc:

            if self._is_password_error(exc):
                raise ValueError(
                    "The PDF is password protected or the supplied "
                    "password is incorrect."
                ) from exc

            raise

        raise ValueError(
            "Could not extract readable text from this PDF."
        )

    # ============================================================
    # PDFPLUMBER
    # ============================================================

    def _extract_with_pdfplumber(self) -> List[Dict]:
        """
        Primary text extraction engine.
        """

        pages: List[Dict] = []

        with pdfplumber.open(
            self.file_path,
            password=self.password,
        ) as pdf:

            for page_number, page in enumerate(
                pdf.pages,
                start=1,
            ):

                text = page.extract_text() or ""

                pages.append(
                    {
                        "page_number": page_number,
                        "text": text,
                        "has_text": bool(
                            text.strip()
                        ),
                        "extraction_engine": "pdfplumber",
                    }
                )

        return pages

    # ============================================================
    # PYMUPDF
    # ============================================================

    def _open_with_pymupdf(self):
        """
        Open the PDF with PyMuPDF, including password handling.
        """

        if fitz is None:
            raise RuntimeError(
                "PyMuPDF is not installed."
            )

        document = fitz.open(
            str(self.file_path)
        )

        if document.needs_pass:

            if not self.password:
                document.close()

                raise ValueError(
                    "The PDF is password protected."
                )

            authenticated = document.authenticate(
                self.password
            )

            if not authenticated:
                document.close()

                raise ValueError(
                    "The supplied PDF password is incorrect."
                )

        return document

    def _extract_with_pymupdf(self) -> List[Dict]:
        """
        Fallback text extraction using PyMuPDF.
        """

        document = self._open_with_pymupdf()

        pages: List[Dict] = []

        try:

            for page_number, page in enumerate(
                document,
                start=1,
            ):

                text = page.get_text(
                    "text"
                ) or ""

                pages.append(
                    {
                        "page_number": page_number,
                        "text": text,
                        "has_text": bool(
                            text.strip()
                        ),
                        "extraction_engine": "pymupdf",
                    }
                )

        finally:
            document.close()

        return pages

    # ============================================================
    # WORD EXTRACTION
    # ============================================================

    def extract_word_pages(self) -> List[Dict]:
        """
        Extract positioned words.

        This is intended for the universal structure and row
        extraction layers.

        Primary:

            pdfplumber.extract_words()

        Fallback:

            PyMuPDF word extraction.

        Returned words use the common structure:

            {
                "text": str,
                "x0": float,
                "x1": float,
                "top": float,
                "bottom": float
            }
        """

        # --------------------------------------------------------
        # pdfplumber
        # --------------------------------------------------------

        try:

            pages = []

            with pdfplumber.open(
                self.file_path,
                password=self.password,
            ) as pdf:

                for page_number, page in enumerate(
                    pdf.pages,
                    start=1,
                ):

                    words = page.extract_words(
                        use_text_flow=False,
                        keep_blank_chars=False,
                    )

                    normalized_words = (
                        self._normalize_pdfplumber_words(
                            words
                        )
                    )

                    pages.append(
                        {
                            "page_number": page_number,
                            "words": normalized_words,
                            "extraction_engine": "pdfplumber",
                        }
                    )

            if self._has_word_data(pages):
                return pages

        except Exception as exc:

            if self._is_password_error(exc):
                raise ValueError(
                    "The PDF is password protected or the supplied "
                    "password is incorrect."
                ) from exc

        # --------------------------------------------------------
        # PyMuPDF fallback
        # --------------------------------------------------------

        if fitz is None:
            raise RuntimeError(
                "Could not extract positioned PDF words with "
                "pdfplumber, and PyMuPDF is not installed."
            )

        pages = self._extract_words_with_pymupdf()

        if self._has_word_data(pages):
            return pages

        raise ValueError(
            "Could not extract positioned words from the PDF."
        )

    # ============================================================
    # PDFPLUMBER WORD NORMALIZATION
    # ============================================================

    def _normalize_pdfplumber_words(
        self,
        words,
    ) -> List[Dict]:

        result = []

        for word in words:

            text = str(
                word.get("text", "")
            ).strip()

            if not text:
                continue

            try:

                result.append(
                    {
                        "text": text,
                        "x0": float(
                            word["x0"]
                        ),
                        "x1": float(
                            word["x1"]
                        ),
                        "top": float(
                            word["top"]
                        ),
                        "bottom": float(
                            word.get(
                                "bottom",
                                word["top"],
                            )
                        ),
                    }
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

        return result

    # ============================================================
    # PYMUPDF WORD EXTRACTION
    # ============================================================

    def _extract_words_with_pymupdf(
        self,
    ) -> List[Dict]:

        document = self._open_with_pymupdf()

        pages = []

        try:

            for page_number, page in enumerate(
                document,
                start=1,
            ):

                words = page.get_text(
                    "words"
                )

                normalized_words = []

                for item in words:

                    if len(item) < 5:
                        continue

                    x0, y0, x1, y1, text = item[:5]

                    text = str(
                        text
                    ).strip()

                    if not text:
                        continue

                    normalized_words.append(
                        {
                            "text": text,
                            "x0": float(x0),
                            "x1": float(x1),
                            "top": float(y0),
                            "bottom": float(y1),
                        }
                    )

                pages.append(
                    {
                        "page_number": page_number,
                        "words": normalized_words,
                        "extraction_engine": "pymupdf",
                    }
                )

        finally:
            document.close()

        return pages

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _has_useful_text(
        pages: List[Dict],
    ) -> bool:

        if not pages:
            return False

        return any(
            page.get("has_text")
            for page in pages
        )

    @staticmethod
    def _has_word_data(
        pages: List[Dict],
    ) -> bool:

        if not pages:
            return False

        return any(
            bool(page.get("words"))
            for page in pages
        )

    @staticmethod
    def _is_password_error(
        exc: Exception,
    ) -> bool:

        message = str(
            exc
        ).lower()

        keywords = (
            "password",
            "encrypted",
            "decrypt",
            "authentication",
            "needs a password",
        )

        return any(
            keyword in message
            for keyword in keywords
        )

    # ============================================================
    # LEGACY API
    # ============================================================

    def extract_text(self) -> str:
        """
        Return the entire document as text.
        """

        pages = self.extract_pages()

        return "\n\n".join(
            (
                f"--- PAGE "
                f"{page['page_number']} ---\n"
                f"{page['text']}"
            )
            for page in pages
        )