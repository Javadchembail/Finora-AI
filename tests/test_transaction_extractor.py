from collections import Counter

from ingestion.pdf_reader import PDFReader
from ingestion.transaction_extractor import TransactionExtractor


PDF_PATH = "storage/sample_statement.pdf"


reader = PDFReader(PDF_PATH)
pages = reader.extract_pages()

extractor = TransactionExtractor()
transactions = extractor.extract_from_pages(pages)


print("\n" + "=" * 70)
print("FINORA TRANSACTION EXTRACTION AUDIT")
print("=" * 70)

print("Total candidates:", len(transactions))


# ---------------------------------------------------------
# Transactions by page
# ---------------------------------------------------------

print("\nTransactions by page:")

page_counts = Counter(
    transaction["page_number"]
    for transaction in transactions
)

for page_number in sorted(page_counts):
    print(
        f"  Page {page_number}: "
        f"{page_counts[page_number]}"
    )


# ---------------------------------------------------------
# Search payment / credit-related descriptions
# ---------------------------------------------------------

print("\nPayment / credit-related candidates:")

payment_keywords = [
    "PAYMENT",
    "TRANSFER",
    "REFUND",
    "CREDIT",
    "REVERSAL",
]

payment_matches = []

for transaction in transactions:

    description = transaction["description_raw"].upper()

    if any(
        keyword in description
        for keyword in payment_keywords
    ):
        payment_matches.append(transaction)


for transaction in payment_matches:

    print(
        f"  Page {transaction['page_number']} | "
        f"{transaction['post_date_raw']} | "
        f"{transaction['transaction_date_raw']} | "
        f"{transaction['description_raw']} | "
        f"{transaction['amount_raw']}"
    )


# ---------------------------------------------------------
# Show last 20 transactions
# ---------------------------------------------------------

print("\nLast 20 candidates:")

for transaction in transactions[-20:]:

    print(
        f"  Page {transaction['page_number']} | "
        f"{transaction['post_date_raw']} | "
        f"{transaction['transaction_date_raw']} | "
        f"{transaction['description_raw']} | "
        f"{transaction['amount_raw']}"
    )