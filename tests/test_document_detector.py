from ingestion.pdf_reader import PDFReader
from ingestion.document_detector import DocumentDetector


PDF_PATH = "storage/sample_statement.pdf"


reader = PDFReader(
    PDF_PATH
)

pages = reader.extract_pages()

detector = DocumentDetector()

metadata = detector.detect(
    pages
)


print("\n" + "=" * 60)
print("FINORA DOCUMENT DETECTION")
print("=" * 60)

print(
    "Document type:",
    metadata.document_type,
)

print(
    "Statement type:",
    metadata.statement_type,
)

print(
    "Country:",
    metadata.country,
)

print(
    "Currency:",
    metadata.currency,
)

print(
    "Date format:",
    metadata.date_format,
)

print(
    "Statement start:",
    metadata.statement_start_date,
)

print(
    "Statement end:",
    metadata.statement_end_date,
)

print(
    "Statement year:",
    metadata.statement_year,
)

print(
    "Page count:",
    metadata.page_count,
)

print(
    "Transaction table:",
    metadata.has_transaction_table,
)

print(
    "Posting date:",
    metadata.has_posting_date,
)

print(
    "Transaction date:",
    metadata.has_transaction_date,
)

print(
    "Confidence:",
    metadata.confidence,
)

print("\nDetected keywords:")

for keyword in metadata.detected_keywords:

    print(
        " -",
        keyword,
    )