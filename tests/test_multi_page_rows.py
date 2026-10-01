from __future__ import annotations

import pdfplumber

from ingestion.universal_structure import (
    UniversalStructureDetector,
)

from ingestion.row_region_extractor import (
    RowRegionExtractor,
)


def print_row(index, row):
    print(f"\n{index}:")

    for key, value in row.items():
        print(f"  {key}: {value}")


def summarize_rows(rows):
    debit_count = 0
    credit_count = 0
    amount_count = 0
    unknown_amount = 0

    for row in rows:

        if row.get("debit") is not None:
            debit_count += 1

        if row.get("credit") is not None:
            credit_count += 1

        if row.get("amount") is not None:
            amount_count += 1

        if (
            row.get("debit") is None
            and row.get("credit") is None
            and row.get("amount") is None
        ):
            unknown_amount += 1

    print("\nSummary:")
    print(f"  Total rows:       {len(rows)}")
    print(f"  Debit rows:       {debit_count}")
    print(f"  Credit rows:      {credit_count}")
    print(f"  Amount rows:      {amount_count}")
    print(f"  No amount:        {unknown_amount}")


def test_pdf(
    name,
    path,
    password=None,
):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    detector = UniversalStructureDetector()
    extractor = RowRegionExtractor()

    all_rows = []

    with pdfplumber.open(
        path,
        password=password,
    ) as pdf:

        print("Total PDF pages:", len(pdf.pages))

        # ----------------------------------------------------------
        # Detect the document structure once.
        # ----------------------------------------------------------

        structure = detector.detect(
            path,
            password=password,
        )

        print("\nDetected structure:")
        print(
            "  Score:",
            structure.get("score"),
        )

        print(
            "  Detection page:",
            structure.get("page"),
        )

        print(
            "  Columns:",
            len(structure.get("columns", [])),
        )

        for column in structure.get("columns", []):

            print(
                f"    {column['header']} "
                f"-> {column['semantic_type']} "
                f"x0={column['x0']:.2f} "
                f"x1={column['x1']:.2f}"
            )

        columns = structure["columns"]

        # ----------------------------------------------------------
        # Process every PDF page.
        # ----------------------------------------------------------

        for page_number, page in enumerate(
            pdf.pages,
            start=1,
        ):

            words = page.extract_words(
                use_text_flow=False,
                keep_blank_chars=False,
            )

            if not words:
                print(
                    f"\nPage {page_number}: "
                    "no words"
                )
                continue

            rows = extractor.extract(
                words,
                columns,
            )

            print(
                f"\nPage {page_number}: "
                f"{len(rows)} transactions"
            )

            all_rows.extend(rows)

            # ------------------------------------------------------
            # Print every extracted row for this page.
            # ------------------------------------------------------

            for local_index, row in enumerate(
                rows,
                start=1,
            ):

                print(
                    f"  Row {local_index}: ",
                    end="",
                )

                transaction_date = (
                    row.get("transaction_date")
                    or row.get("posting_date")
                    or row.get("value_date")
                )

                description = (
                    row.get("description")
                    or ""
                )

                transaction_id = (
                    row.get("transaction_id")
                    or ""
                )

                debit = row.get("debit")
                credit = row.get("credit")
                amount = row.get("amount")
                balance = row.get("balance")

                print(
                    f"date={transaction_date} | "
                    f"description={description[:60]} | "
                    f"id={transaction_id} | "
                    f"debit={debit} | "
                    f"credit={credit} | "
                    f"amount={amount} | "
                    f"balance={balance}"
                )

    # --------------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------------

    print("\n" + "-" * 80)
    print("FINAL RESULT")
    print("-" * 80)

    print(
        f"TOTAL TRANSACTIONS: {len(all_rows)}"
    )

    summarize_rows(all_rows)

    # --------------------------------------------------------------
    # FIRST 10
    # --------------------------------------------------------------

    print("\n" + "-" * 80)
    print("FIRST 10 TRANSACTIONS")
    print("-" * 80)

    for index, row in enumerate(
        all_rows[:10],
        start=1,
    ):
        print_row(index, row)

    # --------------------------------------------------------------
    # LAST 10
    # --------------------------------------------------------------

    print("\n" + "-" * 80)
    print("LAST 10 TRANSACTIONS")
    print("-" * 80)

    start_index = max(
        0,
        len(all_rows) - 10,
    )

    for index, row in enumerate(
        all_rows[start_index:],
        start=start_index + 1,
    ):
        print_row(index, row)

    return all_rows


# ==================================================================
# FEDERAL BANK
# ==================================================================

federal_rows = test_pdf(
    name="FEDERAL BANK",
    path="storage/federal_statement.pdf",
    password="JAVA2301",
)


# ==================================================================
# EMIRATES ISLAMIC
# ==================================================================

emirates_rows = test_pdf(
    name="EMIRATES ISLAMIC",
    path="storage/sample_statement.pdf",
)


# ==================================================================
# GLOBAL SUMMARY
# ==================================================================

print("\n" + "=" * 80)
print("GLOBAL SUMMARY")
print("=" * 80)

print(
    "Federal Bank transactions:",
    len(federal_rows),
)

print(
    "Emirates Islamic transactions:",
    len(emirates_rows),
)

print(
    "Total transactions:",
    len(federal_rows) + len(emirates_rows),
)