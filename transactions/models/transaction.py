from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


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

    model_config = ConfigDict(
        use_enum_values=True
    )

    transaction_id: Optional[str] = None

    transaction_date: Optional[date] = None

    posting_date: Optional[date] = None

    description_raw: str

    description_normalized: Optional[str] = None

    merchant: Optional[str] = None

    original_amount: Decimal

    original_currency: str

    statement_amount: Optional[Decimal] = None

    statement_currency: Optional[str] = None

    exchange_rate: Optional[Decimal] = None

    foreign_transaction_fee: Optional[Decimal] = None

    direction: TransactionDirection

    transaction_type: TransactionType = (
        TransactionType.UNKNOWN
    )

    statement_type: StatementType = (
        StatementType.UNKNOWN
    )

    category: Optional[str] = None

    subcategory: Optional[str] = None

    extraction_confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    merchant_confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    category_confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    account_identifier: Optional[str] = None

    account_holder: Optional[str] = None

    source_file: Optional[str] = None

    source_page: Optional[int] = None

    # ---------------------------------------------
    # Bank-specific metadata that is still useful
    # universally
    # ---------------------------------------------

    bank_transaction_id: Optional[str] = None

    running_balance: Optional[Decimal] = None

    balance_direction: Optional[str] = None

    transaction_channel: Optional[str] = None

    requires_review: bool = False

    user_corrected: bool = False

    notes: Optional[str] = None