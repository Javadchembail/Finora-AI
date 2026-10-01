import pdfplumber

from ingestion.header_grouping import HeaderGrouper


def inspect_pdf(name, path, password=None):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    with pdfplumber.open(path, password=password) as pdf:

        page = pdf.pages[0]

        words = page.extract_words(
            use_text_flow=False,
            keep_blank_chars=False,
        )

        # Only inspect the upper/middle transaction-header area.
        header_words = [
            word
            for word in words
            if 250 <= word["top"] <= 410
        ]

        grouper = HeaderGrouper()

        groups = grouper.group(
            header_words
        )

        for group in groups:

            print(
                f"{group.text:35}"
                f"x0={group.x0:7.1f} "
                f"x1={group.x1:7.1f} "
                f"top={group.top:7.1f}"
            )


inspect_pdf(
    "FEDERAL BANK",
    "storage/federal_statement.pdf",
    password="JAVA2301",
)

inspect_pdf(
    "EMIRATES ISLAMIC",
    "storage/sample_statement.pdf",
)