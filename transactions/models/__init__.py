from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class StatementType(str, Enum):
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"
    DEBIT_CARD = "debit_card"
    WALLET = "wallet"
    LOAN = "loan"
    INVESTMENT = "investment"
    UNKNOWN = "unknown"


class TransactionDirection(str, Enum):
    DEBIT = "debit"
    CREDIT = "credit"


class TransactionType(str, Enum):
    PURCHASE = "purchase"
    PAYMENT = "payment"
    REFUND = "refund"
    TRANSFER = "transfer"
    SALARY = "salary"
    CASH_WITHDRAWAL = "cash_withdrawal"
    FEE = "fee"
    INTEREST = "interest"
    INVESTMENT = "investment"
    LOAN_PAYMENT = "loan_payment"
    BILL_PAYMENT = "bill_payment"
    TAX = "tax"
    OTHER = "other"
    UNKNOWN = "unknown"


class Transaction(BaseModel):
    """
    Canonical transaction model used throughout Finora AI.

    Every bank/card/wallet statement will eventually be
    converted into this structure.
    """

    model_config = ConfigDict(use_enum_values=True)

    # ---------------------------------------------------------
    # Identity
    # ---------------------------------------------------------

    transaction_id: Optional[str] = None

    # ---------------------------------------------------------
    # Dates
    # ---------------------------------------------------------

    transaction_date: Optional[date] = None
    posting_date: Optional[date] = None

    # ---------------------------------------------------------
    # Original statement information
    # ---------------------------------------------------------

    description_raw: str = Field(
        ...,
        description="Original transaction description exactly as found in the statement."
    )

    description_normalized: Optional[str] = None

    merchant: Optional[str] = None

    # ---------------------------------------------------------
    # Money
    # ---------------------------------------------------------

    original_amount: Decimal

    original_currency: str

    statement_amount: Optional[Decimal] = None

    statement_currency: Optional[str] = None

    exchange_rate: Optional[Decimal] = None

    foreign_transaction_fee: Optional[Decimal] = None

    # ---------------------------------------------------------
    # Transaction meaning
    # ---------------------------------------------------------

    direction: TransactionDirection

    transaction_type: TransactionType = TransactionType.UNKNOWN

    statement_type: StatementType = StatementType.UNKNOWN

    # ---------------------------------------------------------
    # Classification
    # ---------------------------------------------------------

    category: Optional[str] = None

    subcategory: Optional[str] = None

    # ---------------------------------------------------------
    # AI confidence
    # ---------------------------------------------------------

    extraction_confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1
    )

    merchant_confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1
    )

    category_confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1
    )

    # ---------------------------------------------------------
    # Account / card
    # ---------------------------------------------------------

    account_identifier: Optional[str] = None

    account_holder: Optional[str] = None

    # ---------------------------------------------------------
    # Source document
    # ---------------------------------------------------------

    source_file: Optional[str] = None

    source_page: Optional[int] = None

    # ---------------------------------------------------------
    # Review / learning
    # ---------------------------------------------------------

    requires_review: bool = False

    user_corrected: bool = False

    notes: Optional[str] = None