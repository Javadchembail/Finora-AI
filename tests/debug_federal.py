from ingestion.pdf_reader import PDFReader
from ingestion.document_detector import DocumentDetector

PDF_PATH = "storage/federal_statement.pdf"
PASSWORD = "JAVA2301"

print("=" * 80)
print("FEDERAL BANK PDF DIAGNOSTIC")
print("=" * 80)

reader = PDFReader(
    file_path=PDF_PATH,
    password=PASSWORD,
)

pages = reader.extract_pages()

print(f"\nPage count: {len(pages)}")

detector = DocumentDetector()
metadata = detector.detect(pages)

print("\nDOCUMENT DETECTION")
print("-" * 80)

print(f"Document type       : {metadata.document_type}")
print(f"Statement type      : {metadata.statement_type}")
print(f"Country             : {metadata.country}")
print(f"Currency            : {metadata.currency}")
print(f"Date format         : {metadata.date_format}")
print(f"Statement start     : {metadata.statement_start_date}")
print(f"Statement end       : {metadata.statement_end_date}")
print(f"Statement year      : {metadata.statement_year}")
print(f"Transaction table   : {metadata.has_transaction_table}")
print(f"Posting date        : {metadata.has_posting_date}")
print(f"Transaction date    : {metadata.has_transaction_date}")
print(f"Debit/Credit        : {metadata.has_debit_credit_indicator}")
print(f"Withdrawals         : {metadata.has_withdrawal_column}")
print(f"Deposits            : {metadata.has_deposit_column}")
print(f"Balance             : {metadata.has_balance_column}")
print(f"Transaction ID      : {metadata.has_transaction_id}")
print(f"Confidence          : {metadata.confidence}")

print("\n" + "=" * 80)
print("RAW EXTRACTED TEXT")
print("=" * 80)

for page in pages:
    print("\n")
    print("#" * 80)
    print(f"PAGE {page['page_number']}")
    print("#" * 80)

    lines = page["text"].splitlines()

    for number, line in enumerate(lines, start=1):
        print(f"{number:03d}: {line}")