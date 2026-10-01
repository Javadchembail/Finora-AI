from collections import Counter

from ingestion.pipeline import FinancialStatementPipeline


PDF_PATH = "storage/federal_statement.pdf"
PASSWORD = "JAVA2301"


print("=" * 70)
print("FINORA BANK STATEMENT TEST")
print("=" * 70)


# ---------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------

pipeline = FinancialStatementPipeline(
    file_path=PDF_PATH,
    password=PASSWORD,
)

transactions = pipeline.run()


print()
print("Canonical transactions:", len(transactions))


# ---------------------------------------------------------------------
# Print first 10 transactions
# ---------------------------------------------------------------------

print()
print("-" * 60)

for transaction in transactions[:10]:

    print("ID:", transaction.transaction_id)

    print(
        "Bank Transaction ID:",
        getattr(
            transaction,
            "bank_transaction_id",
            transaction.transaction_id,
        ),
    )

    print(
        "Date:",
        transaction.transaction_date,
    )

    print(
        "Value Date:",
        transaction.posting_date,
    )

    print(
        "Raw:",
        transaction.description_raw,
    )

    print(
        "Merchant:",
        transaction.merchant,
    )

    print(
        "Amount:",
        transaction.original_amount,
    )

    print(
        "Currency:",
        transaction.original_currency,
    )

    print(
        "Direction:",
        transaction.direction,
    )

    print(
        "Type:",
        transaction.transaction_type,
    )

    print("-" * 60)


# ---------------------------------------------------------------------
# Direction summary
# ---------------------------------------------------------------------

print()
print("=" * 60)
print("DIRECTION SUMMARY")
print("=" * 60)


direction_counts = Counter(
    str(transaction.direction)
    for transaction in transactions
)


for direction, count in direction_counts.items():

    print(
        f"{direction}: {count}"
    )


# ---------------------------------------------------------------------
# Transaction type summary
# ---------------------------------------------------------------------

print()
print("=" * 60)
print("TRANSACTION TYPE SUMMARY")
print("=" * 60)


type_counts = Counter(
    str(transaction.transaction_type)
    for transaction in transactions
)


for transaction_type, count in type_counts.items():

    print(
        f"{transaction_type}: {count}"
    )


# ---------------------------------------------------------------------
# Currency summary
# ---------------------------------------------------------------------

print()
print("=" * 60)
print("CURRENCY SUMMARY")
print("=" * 60)


currency_counts = Counter(
    transaction.original_currency
    for transaction in transactions
)


for currency, count in currency_counts.items():

    print(
        f"{currency}: {count}"
    )


# ---------------------------------------------------------------------
# Basic validation
# ---------------------------------------------------------------------

print()
print("=" * 60)
print("VALIDATION")
print("=" * 60)


print(
    "Total transactions:",
    len(transactions),
)


print(
    "Transactions with unknown direction:",
    sum(
        1
        for transaction in transactions
        if str(transaction.direction).lower()
        == "unknown"
    ),
)


print(
    "Transactions without description:",
    sum(
        1
        for transaction in transactions
        if not transaction.description_raw
        or not transaction.description_raw.strip()
    ),
)


print(
    "Transactions without amount:",
    sum(
        1
        for transaction in transactions
        if transaction.original_amount is None
    ),
)


print()
print("=" * 70)
print("TEST COMPLETED")
print("=" * 70)