from ingestion.pdf_reader import PDFReader
from ingestion.structure_detector import StructureDetector


TEST_DOCUMENTS = [
    {
        "name": "Federal Bank",
        "path": "storage/federal_statement.pdf",
        "password": "JAVA2301",
    },
    {
        "name": "Emirates Islamic",
        "path": "storage/sample_statement.pdf",
        "password": None,
    },
]


detector = StructureDetector()


for document in TEST_DOCUMENTS:

    print()
    print("=" * 70)
    print(
        f"STRUCTURE TEST: {document['name']}"
    )
    print("=" * 70)

    reader = PDFReader(
        document["path"],
        password=document["password"],
    )

    pages = reader.extract_pages()

    structure = detector.detect(
        pages
    )

    print()
    print(
        "Header page:",
        structure.transaction_header_page,
    )

    print(
        "Header:",
        structure.header_text,
    )

    print(
        "Confidence:",
        structure.confidence,
    )

    print()
    print("Columns:")

    for column in structure.columns:

        print(
            f"  {column.name}"
            f" -> {column.semantic_type}"
            f" ({column.confidence})"
        )

    print()
    print(
        "Transaction dates:",
        structure.transaction_date_columns,
    )

    print(
        "Posting dates:",
        structure.posting_date_columns,
    )

    print(
        "Descriptions:",
        structure.description_columns,
    )

    print(
        "Amounts:",
        structure.amount_columns,
    )

    print(
        "Debits:",
        structure.debit_columns,
    )

    print(
        "Credits:",
        structure.credit_columns,
    )

    print(
        "Balances:",
        structure.balance_columns,
    )

    print(
        "Transaction IDs:",
        structure.transaction_id_columns,
    )

    print()
    print(
        "Separate debit/credit columns:",
        structure.has_separate_debit_credit_columns,
    )

    print(
        "Single amount column:",
        structure.has_single_amount_column,
    )

    print(
        "Balance column:",
        structure.has_balance_column,
    )

    print(
        "Transaction ID:",
        structure.has_transaction_id,
    )