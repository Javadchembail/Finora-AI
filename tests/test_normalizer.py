from transactions.normalizer import TransactionNormalizer


normalizer = TransactionNormalizer()


print("\n" + "=" * 60)
print("FINORA NORMALIZER TEST")
print("=" * 60)


# UAE-style statement date
date_result = normalizer.parse_month_day(
    "09 AUG",
    2026,
)

print("Date:", date_result)


# Credit-card payment
amount_result = normalizer.parse_amount(
    "14,000.00CR"
)

print("Amount:", amount_result)


# Normal expense
amount_result = normalizer.parse_amount(
    "1,299.50"
)

print("Expense:", amount_result)


# Negative amount
amount_result = normalizer.parse_amount(
    "(500.00)"
)

print("Negative:", amount_result)


# Indian-style numeric date
date_result = normalizer.parse_numeric_date(
    "01/09/2026",
    day_first=True,
)

print("Indian date:", date_result)