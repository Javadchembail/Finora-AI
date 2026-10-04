from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any, Dict, List

from transactions.models.transaction import Transaction


class FinanceAnalyzer:
    """
    Simple financial analysis engine.

    Works directly with the canonical Transaction objects
    produced by the Finora pipeline.
    """

    def __init__(self, transactions: List[Transaction]):
        self.transactions = transactions

    # =========================================================
    # BASIC TOTALS
    # =========================================================

    def total_income(self) -> Decimal:
        """
        Total money received.

        Credit transactions classified as transfers/payments/refunds
        are kept out of normal income.
        """

        excluded_types = {
            "transfer",
            "payment",
            "refund",
        }

        total = Decimal("0")

        for transaction in self.transactions:

            direction = str(transaction.direction)
            transaction_type = str(transaction.transaction_type)

            if (
                direction == "credit"
                and transaction_type not in excluded_types
            ):
                total += transaction.original_amount

        return total

    def total_expenses(self) -> Decimal:
        """
        Total money spent.

        Debit transfers are excluded because moving money between
        accounts is not normally an expense.
        """

        total = Decimal("0")

        for transaction in self.transactions:

            direction = str(transaction.direction)
            transaction_type = str(transaction.transaction_type)

            if (
                direction == "debit"
                and transaction_type != "transfer"
            ):
                total += transaction.original_amount

        return total

    def total_transfers(self) -> Decimal:
        """
        Total value of transfer transactions.
        """

        total = Decimal("0")

        for transaction in self.transactions:

            if str(transaction.transaction_type) == "transfer":
                total += transaction.original_amount

        return total

    def total_refunds(self) -> Decimal:
        """
        Total refunds received.
        """

        total = Decimal("0")

        for transaction in self.transactions:

            if str(transaction.transaction_type) == "refund":
                total += transaction.original_amount

        return total

    def net_cash_flow(self) -> Decimal:
        """
        Income minus expenses.
        """

        return self.total_income() - self.total_expenses()

    # =========================================================
    # TRANSACTION COUNT
    # =========================================================

    def transaction_count(self) -> int:
        return len(self.transactions)

    # =========================================================
    # CATEGORY ANALYSIS
    # =========================================================

    def spending_by_category(self) -> Dict[str, Decimal]:
        """
        Groups debit transactions by category.
        """

        categories = defaultdict(lambda: Decimal("0"))

        for transaction in self.transactions:

            if str(transaction.direction) != "debit":
                continue

            if str(transaction.transaction_type) == "transfer":
                continue

            category = transaction.category or "Other"

            categories[category] += transaction.original_amount

        return dict(
            sorted(
                categories.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    # =========================================================
    # MERCHANT ANALYSIS
    # =========================================================

    def spending_by_merchant(self) -> Dict[str, Decimal]:
        """
        Groups debit transactions by merchant.
        """

        merchants = defaultdict(lambda: Decimal("0"))

        for transaction in self.transactions:

            if str(transaction.direction) != "debit":
                continue

            if str(transaction.transaction_type) == "transfer":
                continue

            merchant = transaction.merchant or "Unknown"

            merchants[merchant] += transaction.original_amount

        return dict(
            sorted(
                merchants.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    # =========================================================
    # MONTHLY ANALYSIS
    # =========================================================

    def monthly_summary(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Creates a simple monthly income/expense summary.
        """

        monthly = defaultdict(
            lambda: {
                "income": Decimal("0"),
                "expenses": Decimal("0"),
                "net": Decimal("0"),
            }
        )

        for transaction in self.transactions:

            transaction_date = transaction.transaction_date

            if not transaction_date:
                continue

            month = transaction_date.strftime("%Y-%m")

            direction = str(transaction.direction)
            transaction_type = str(transaction.transaction_type)

            if transaction_type == "transfer":
                continue

            if direction == "credit":

                if transaction_type not in {
                    "payment",
                    "refund",
                }:
                    monthly[month]["income"] += (
                        transaction.original_amount
                    )

            elif direction == "debit":

                monthly[month]["expenses"] += (
                    transaction.original_amount
                )

        for month in monthly:

            monthly[month]["net"] = (
                monthly[month]["income"]
                - monthly[month]["expenses"]
            )

        return dict(
            sorted(monthly.items())
        )

    # =========================================================
    # TOP MERCHANTS
    # =========================================================

    def top_merchants(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Returns the highest-spending merchants.
        """

        merchant_data = self.spending_by_merchant()

        return [
            {
                "merchant": merchant,
                "amount": amount,
            }
            for merchant, amount
            in list(merchant_data.items())[:limit]
        ]

    # =========================================================
    # TOP CATEGORIES
    # =========================================================

    def top_categories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Returns the highest-spending categories.
        """

        category_data = self.spending_by_category()

        return [
            {
                "category": category,
                "amount": amount,
            }
            for category, amount
            in list(category_data.items())[:limit]
        ]

    # =========================================================
    # COMPLETE SUMMARY
    # =========================================================

    def summary(self) -> Dict[str, Any]:
        """
        Returns the complete financial summary.
        """

        return {
            "transaction_count": self.transaction_count(),
            "total_income": self.total_income(),
            "total_expenses": self.total_expenses(),
            "net_cash_flow": self.net_cash_flow(),
            "total_transfers": self.total_transfers(),
            "total_refunds": self.total_refunds(),
            "spending_by_category": self.spending_by_category(),
            "top_categories": self.top_categories(),
            "top_merchants": self.top_merchants(),
            "monthly_summary": self.monthly_summary(),
        }