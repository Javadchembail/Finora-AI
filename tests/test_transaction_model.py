from decimal import Decimal

from transactions.models import (
    Transaction,
    TransactionDirection,
    TransactionType,
    StatementType,
)


transaction = Transaction(
    transaction_id="TEST-001",
    transaction_date="2026-10-01",
    posting_date="2026-10-01",
    description_raw="FAITH CAFETERIA",
    description_normalized="Faith Cafeteria",
    merchant="Faith Cafeteria",
    original_amount=Decimal("25.50"),
    original_currency="AED",
    statement_amount=Decimal("25.50"),
    statement_currency="AED",
    direction=TransactionDirection.DEBIT,
    transaction_type=TransactionType.PURCHASE,
    statement_type=StatementType.CREDIT_CARD,
    category="Food & Dining",
    subcategory="Restaurants",
    extraction_confidence=0.98,
    merchant_confidence=0.95,
    category_confidence=0.92,
    source_page=1,
)

print(transaction.model_dump())