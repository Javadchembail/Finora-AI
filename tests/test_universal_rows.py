import pdfplumber

from ingestion.universal_structure import (
    UniversalStructureDetector,
)

from ingestion.row_region_extractor import (
    UniversalRowExtractor,
)


def test_pdf(
    name,
    path,
    password=None,
):

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    detector = UniversalStructureDetector()

    structure = detector.detect(
        path,
        password=password,
    )

    print(
        "Structure score:",
        structure["score"],
    )

    print(
        "Detected columns:",
        len(structure["columns"]),
    )

    print("\nColumns:")

    for column in structure["columns"]:

        print(
            f"  {column['header']} "
            f"-> {column['semantic_type']} "
            f"x0={column['x0']:.2f} "
            f"x1={column['x1']:.2f}"
        )

    with pdfplumber.open(
        path,
        password=password,
    ) as pdf:

        page_number = structure["page"]

        page = pdf.pages[
            page_number - 1
        ]

        words = page.extract_words(
            use_text_flow=False,
            keep_blank_chars=False,
        )

    extractor = UniversalRowExtractor()

    rows = extractor.extract(
        words,
        structure["columns"],
    )

    print(
        "\nTransactions detected:",
        len(rows),
    )

    print("\nFirst 10 rows:")

    for index, row in enumerate(
        rows[:10],
        start=1,
    ):

        print(
            f"\n{index}:"
        )

        for key, value in row.items():

            print(
                f"  {key}: {value}"
            )


# ============================================================
# FEDERAL BANK
# ============================================================

test_pdf(
    "FEDERAL BANK",
    "storage/federal_statement.pdf",
    password="JAVA2301",
)


# ============================================================
# EMIRATES ISLAMIC
# ============================================================

test_pdf(
    "EMIRATES ISLAMIC",
    "storage/sample_statement.pdf",
)