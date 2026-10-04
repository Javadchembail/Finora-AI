from ingestion.pipeline import FinancialStatementPipeline
from analytics.finance_analyzer import FinanceAnalyzer


PDF_PATH = "storage/federal_statement.pdf"

PASSWORD = input("Enter PDF password: ").strip()


# =========================================================
# RUN PIPELINE
# =========================================================

pipeline = FinancialStatementPipeline(
    PDF_PATH,
    password=PASSWORD,
)

transactions = pipeline.run()


# =========================================================
# ANALYZE
# =========================================================

analyzer = FinanceAnalyzer(transactions)


summary = analyzer.summary()


# =========================================================
# DISPLAY
# =========================================================

print("\n" + "=" * 70)
print("FINORA FINANCIAL ANALYSIS")
print("=" * 70)


print(
    "\nTransactions:",
    summary["transaction_count"],
)

print(
    "Total Income:",
    summary["total_income"],
)

print(
    "Total Expenses:",
    summary["total_expenses"],
)

print(
    "Net Cash Flow:",
    summary["net_cash_flow"],
)

print(
    "Total Transfers:",
    summary["total_transfers"],
)

print(
    "Total Refunds:",
    summary["total_refunds"],
)


# =========================================================
# CATEGORY SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("TOP SPENDING CATEGORIES")
print("=" * 70)


for item in summary["top_categories"]:

    print(
        f"{item['category']}: "
        f"{item['amount']}"
    )


# =========================================================
# MERCHANT SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("TOP SPENDING MERCHANTS")
print("=" * 70)


for item in summary["top_merchants"]:

    print(
        f"{item['merchant']}: "
        f"{item['amount']}"
    )


# =========================================================
# MONTHLY SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("MONTHLY SUMMARY")
print("=" * 70)


for month, data in summary["monthly_summary"].items():

    print(
        f"\n{month}"
    )

    print(
        "  Income:",
        data["income"],
    )

    print(
        "  Expenses:",
        data["expenses"],
    )

    print(
        "  Net:",
        data["net"],
    )