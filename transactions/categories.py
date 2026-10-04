from __future__ import annotations

from typing import Dict, List


# ============================================================
# FINORA UNIVERSAL CATEGORY TAXONOMY
# ============================================================
#
# This taxonomy is intentionally bank-independent and
# country-independent.
#
# It is used by:
#   - category engine
#   - review UI
#   - analytics
#   - AI assistant
#   - user learning
#
# Do not add bank names here.
# Do not add statement-format-specific categories here.
# ============================================================


CATEGORIES: Dict[str, List[str]] = {

    # --------------------------------------------------------
    # FOOD & DINING
    # --------------------------------------------------------

    "Food & Dining": [
        "Restaurant",
        "Cafe",
        "Fast Food",
        "Food Delivery",
        "Bakery",
        "Snacks",
        "Cafeteria",
        "Food & Beverage",
    ],

    # --------------------------------------------------------
    # GROCERIES
    # --------------------------------------------------------

    "Groceries": [
        "Supermarket",
        "Grocery Store",
        "Online Grocery",
        "Convenience Store",
        "Fresh Market",
    ],

    # --------------------------------------------------------
    # TRANSPORTATION
    # --------------------------------------------------------

    "Transportation": [
        "Fuel",
        "Taxi",
        "Ride Sharing",
        "Public Transport",
        "Parking",
        "Tolls",
        "Vehicle Maintenance",
        "Vehicle Insurance",
        "Vehicle Purchase",
    ],

    # --------------------------------------------------------
    # HOUSING
    # --------------------------------------------------------

    "Housing": [
        "Rent",
        "Mortgage",
        "Home Maintenance",
        "Furniture",
        "Home Services",
        "Property Charges",
    ],

    # --------------------------------------------------------
    # BILLS & UTILITIES
    # --------------------------------------------------------

    "Bills & Utilities": [
        "Electricity",
        "Water",
        "Internet",
        "Mobile",
        "Telecom",
        "Subscription",
        "Gas",
        "Other Utilities",
    ],

    # --------------------------------------------------------
    # HEALTHCARE
    # --------------------------------------------------------

    "Healthcare": [
        "Pharmacy",
        "Hospital",
        "Doctor",
        "Diagnostics",
        "Dental",
        "Health Insurance",
        "Medical Services",
    ],

    # --------------------------------------------------------
    # SHOPPING
    # --------------------------------------------------------

    "Shopping": [
        "Clothing",
        "Electronics",
        "Home",
        "Personal Care",
        "Beauty",
        "General Shopping",
        "Flowers & Gifts",
        "Jewelry",
    ],

    # --------------------------------------------------------
    # ENTERTAINMENT
    # --------------------------------------------------------

    "Entertainment": [
        "Movies",
        "Streaming",
        "Gaming",
        "Music",
        "Events",
        "Sports & Recreation",
        "Other Entertainment",
    ],

    # --------------------------------------------------------
    # TRAVEL
    # --------------------------------------------------------

    "Travel": [
        "Flights",
        "Hotels",
        "Travel Booking",
        "Visa",
        "Travel Expenses",
        "Car Rental",
        "Travel Insurance",
    ],

    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    "Education": [
        "Courses",
        "Books",
        "Tuition",
        "Training",
        "School",
        "College",
        "Educational Services",
    ],

    # --------------------------------------------------------
    # FINANCIAL
    # --------------------------------------------------------

    "Financial": [
        "Bank Fees",
        "Card Fees",
        "ATM Fees",
        "Interest",
        "Financial Services",
        "Loan Charges",
        "Foreign Exchange",
    ],

    # --------------------------------------------------------
    # TAXES & GOVERNMENT
    # --------------------------------------------------------

    "Taxes & Government": [
        "Income Tax",
        "Sales Tax",
        "Government Fees",
        "Documentation",
        "License & Permit",
        "Other Government",
    ],

    # --------------------------------------------------------
    # INCOME
    # --------------------------------------------------------

    "Income": [
        "Salary",
        "Freelance",
        "Business Income",
        "Interest Income",
        "Rental Income",
        "Refund Income",
        "Other Income",
    ],

    # --------------------------------------------------------
    # TRANSFERS
    # --------------------------------------------------------

    "Transfers": [
        "Bank Transfer",
        "Card Payment",
        "Internal Transfer",
        "UPI Transfer",
        "Wallet Transfer",
        "International Transfer",
    ],

    # --------------------------------------------------------
    # INVESTMENTS
    # --------------------------------------------------------

    "Investments": [
        "Stocks",
        "Mutual Funds",
        "Bonds",
        "Brokerage",
        "Crypto",
        "Fixed Deposit",
        "Other Investments",
    ],

    # --------------------------------------------------------
    # CASH
    # --------------------------------------------------------

    "Cash": [
        "ATM Withdrawal",
        "Cash Deposit",
    ],

    # --------------------------------------------------------
    # OTHER
    # --------------------------------------------------------

    "Other": [
        "General",
        "Uncategorized",
        "Unknown",
    ],
}


# Backward-compatible name used by the existing engine.
BASE_CATEGORIES = CATEGORIES


def all_categories() -> List[str]:
    """
    Return all standard Finora categories.
    """
    return list(CATEGORIES.keys())


def subcategories_for(
    category: str,
) -> List[str]:
    """
    Return subcategories for a standard category.
    """
    return list(
        CATEGORIES.get(
            category,
            [],
        )
    )


def is_valid_category(
    category: str,
) -> bool:
    """
    Check whether a category belongs to
    the standard Finora taxonomy.
    """
    return category in CATEGORIES


def is_valid_subcategory(
    category: str,
    subcategory: str,
) -> bool:
    """
    Check whether a subcategory belongs to
    the supplied category.
    """
    return subcategory in CATEGORIES.get(
        category,
        [],
    )