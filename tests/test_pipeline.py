from ingestion.pipeline import FinancialStatementPipeline


PDF_PATH = "storage/sample_statement.pdf"


pipeline = FinancialStatementPipeline(
    PDF_PATH
)

transactions = pipeline.run()


print("\n" + "=" * 70)
print("FINORA COMPLETE PIPELINE")
print("=" * 70)

print(
    f"\nCanonical transactions: {len(transactions)}"
)


# =========================================================
# Show first 10 transactions
# =========================================================

for transaction in transactions[:10]:

    print("\n" + "-" * 60)

    print(
        "ID:",
        transaction.transaction_id,
    )

    print(
        "Date:",
        transaction.transaction_date,
    )

    print(
        "Posting:",
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
        "Merchant Confidence:",
        transaction.merchant_confidence,
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

    print(
        "Statement:",
        transaction.statement_type,
    )

    print(
        "Source Page:",
        transaction.source_page,
    )


# =========================================================
# Payment transaction check
# =========================================================

print("\n" + "=" * 70)
print("PAYMENT TRANSACTION CHECK")
print("=" * 70)


payment_transactions = [
    transaction
    for transaction in transactions
    if transaction.transaction_type == "payment"
]


for transaction in payment_transactions:

    print(
        {
            "transaction_id": transaction.transaction_id,
            "date": transaction.transaction_date,
            "raw": transaction.description_raw,
            "merchant": transaction.merchant,
            "merchant_confidence": transaction.merchant_confidence,
            "amount": transaction.original_amount,
            "currency": transaction.original_currency,
            "direction": transaction.direction,
            "type": transaction.transaction_type,
        }
    )


# =========================================================
# Merchant summary
# =========================================================

print("\n" + "=" * 70)
print("MERCHANT NORMALIZATION SUMMARY")
print("=" * 70)


merchant_transactions = [
    transaction
    for transaction in transactions
    if transaction.merchant
]


missing_merchants = [
    transaction
    for transaction in transactions
    if not transaction.merchant
]


print(
    "Transactions with merchant:",
    len(merchant_transactions),
)

print(
    "Transactions without merchant:",
    len(missing_merchants),
)


# =========================================================
# Unique merchants
# =========================================================

unique_merchants = sorted(
    {
        transaction.merchant
        for transaction in merchant_transactions
        if transaction.merchant
    }
)


print(
    "Unique merchants:",
    len(unique_merchants),
)


print("\nFirst 30 merchants:")

for merchant in unique_merchants[:30]:

    print(
        " -",
        merchant,
    )