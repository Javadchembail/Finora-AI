from ingestion.structure_validator import StructureValidator


def run_test(name, column_candidates, rows):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    validator = StructureValidator()

    result = validator.validate(
        column_candidates=column_candidates,
        rows=rows,
    )

    print("Valid:", result.valid)
    print("Confidence:", result.confidence)

    print("\nColumns:")

    for semantic_type, validation in result.columns.items():
        print(
            f"  {validation.header}"
            f" -> {semantic_type}"
            f" | score={validation.score:.2f}"
            f" | samples={validation.samples_checked}"
            f" | valid={validation.valid_samples}"
        )

        for evidence in validation.evidence:
            print(f"      {evidence}")

    if result.issues:
        print("\nIssues:")

        for issue in result.issues:
            print("  -", issue)


# ---------------------------------------------------------
# Federal-style structure
# ---------------------------------------------------------

federal_columns = {
    "Date": "transaction_date",
    "Value Date": "value_date",
    "Particulars": "description",
    "Tran ID": "transaction_id",
    "Withdrawals": "debit",
    "Deposits": "credit",
    "Balance": "balance",
}

federal_rows = [
    {
        "Date": "31-AUG-2026",
        "Value Date": "31-AUG-2026",
        "Particulars": "UPIOUT/660945519422",
        "Tran ID": "S63313909",
        "Withdrawals": "260.00",
        "Deposits": "",
        "Balance": "2845.15",
    },
    {
        "Date": "01-SEP-2026",
        "Value Date": "01-SEP-2026",
        "Particulars": "UPI IN/7092886032",
        "Tran ID": "S63313910",
        "Withdrawals": "",
        "Deposits": "300.00",
        "Balance": "3145.15",
    },
    {
        "Date": "02-SEP-2026",
        "Value Date": "02-SEP-2026",
        "Particulars": "IMPS IN/ABC123",
        "Tran ID": "S63313911",
        "Withdrawals": "",
        "Deposits": "500.00",
        "Balance": "3645.15",
    },
]


# ---------------------------------------------------------
# Emirates-style structure
# ---------------------------------------------------------

emirates_columns = {
    "Post Date": "posting_date",
    "Trxn. Date": "transaction_date",
    "Description": "description",
    "Amount": "amount",
    "Balance": "balance",
}

emirates_rows = [
    {
        "Post Date": "03-Aug-2026",
        "Trxn. Date": "02-Aug-2026",
        "Description": "CARREFOUR DUBAI",
        "Amount": "125.50",
        "Balance": "13777.69",
    },
    {
        "Post Date": "04-Aug-2026",
        "Trxn. Date": "04-Aug-2026",
        "Description": "AMAZON UAE",
        "Amount": "250.00",
        "Balance": "13527.69",
    },
    {
        "Post Date": "05-Aug-2026",
        "Trxn. Date": "05-Aug-2026",
        "Description": "TRANSFER PAYMENT RECEIVED",
        "Amount": "14000.00CR",
        "Balance": "13903.19",
    },
]


run_test(
    "FEDERAL-STYLE",
    federal_columns,
    federal_rows,
)

run_test(
    "EMIRATES-STYLE",
    emirates_columns,
    emirates_rows,
)