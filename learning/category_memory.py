from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from sqlalchemy import Boolean, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


BASE_CATEGORIES: Dict[str, List[str]] = {
    "Food & Dining": ["Restaurant", "Cafe", "Fast Food", "Food Delivery", "Bakery", "Snacks", "Cafeteria"],
    "Groceries": ["Supermarket", "Grocery Store", "Online Grocery", "Convenience Store"],
    "Transportation": ["Fuel", "Taxi", "Ride Sharing", "Public Transport", "Parking", "Tolls", "Vehicle Maintenance"],
    "Housing": ["Rent", "Mortgage", "Home Maintenance", "Furniture"],
    "Bills & Utilities": ["Electricity", "Water", "Internet", "Mobile", "Telecom", "Subscription"],
    "Healthcare": ["Pharmacy", "Hospital", "Doctor", "Diagnostics", "Health Insurance"],
    "Shopping": ["Clothing", "Electronics", "Home", "Personal Care", "General Shopping", "Flowers & Gifts"],
    "Entertainment": ["Movies", "Streaming", "Gaming", "Music", "Events", "Other Entertainment"],
    "Travel": ["Flights", "Hotels", "Travel Booking", "Visa", "Travel Expenses"],
    "Education": ["Courses", "Books", "Tuition", "Training"],
    "Financial": ["Bank Fees", "Card Fees", "ATM Fees", "Interest", "Financial Services"],
    "Taxes & Government": ["Income Tax", "Sales Tax", "Government Fees", "Documentation"],
    "Income": ["Salary", "Freelance", "Business Income", "Interest Income", "Other Income"],
    "Transfers": ["Bank Transfer", "Card Payment", "Internal Transfer", "UPI Transfer"],
    "Investments": ["Stocks", "Mutual Funds", "Bonds", "Brokerage", "Crypto"],
    "Cash": ["ATM Withdrawal", "Cash Deposit"],
    "Other": ["General", "Uncategorized", "Unknown"],
}

RULES: List[Tuple[str, str, str, Tuple[str, ...], float]] = [
    ("Swiggy", "Food & Dining", "Food Delivery", ("swiggy",), 0.99),
    ("Zomato", "Food & Dining", "Food Delivery", ("zomato",), 0.99),
    ("KFC", "Food & Dining", "Fast Food", ("kfc",), 0.99),
    ("McDonald's", "Food & Dining", "Fast Food", ("mcdonald", "mcdonalds"), 0.99),
    ("Starbucks", "Food & Dining", "Cafe", ("starbucks",), 0.99),
    ("BookMyShow", "Entertainment", "Movies", ("bookmyshow",), 0.99),
    ("Star Cinemas", "Entertainment", "Movies", ("star cinemas", "starcinemas"), 0.98),
    ("Netflix", "Entertainment", "Streaming", ("netflix",), 0.99),
    ("Amazon Prime", "Entertainment", "Streaming", ("amazon prime",), 0.98),
    ("ADNOC", "Transportation", "Fuel", ("adnoc",), 0.88),
    ("Aster Pharmacy", "Healthcare", "Pharmacy", ("aster pharmacy",), 0.99),
    ("Life Pharmacy", "Healthcare", "Pharmacy", ("life pharmacy",), 0.99),
    ("Medeor", "Healthcare", "Hospital", ("medeor",), 0.96),
    ("Millennium Hospital", "Healthcare", "Hospital", ("millennium hospital",), 0.99),
    ("Lulu", "Groceries", "Supermarket", ("lulu hypermarket", "luluhypermarket", "lulu"), 0.95),
    ("Carrefour", "Groceries", "Supermarket", ("carrefour",), 0.98),
    ("Nesto", "Groceries", "Supermarket", ("nesto hyper", "nesto"), 0.97),
    ("Max", "Shopping", "Clothing", ("max ", "max dubai"), 0.94),
    ("Miniso", "Shopping", "General Shopping", ("miniso",), 0.98),
    ("Zerodha", "Investments", "Brokerage", ("zerodha",), 0.99),
    ("Groww", "Investments", "Brokerage", ("groww",), 0.99),
    ("Upstox", "Investments", "Brokerage", ("upstox",), 0.99),
    ("E&", "Bills & Utilities", "Mobile", ("e& digital app", "e& - dubai", "e& digital"), 0.96),
    ("Q Mobility", "Transportation", "Parking", ("q mobility",), 0.95),
    ("Hala", "Transportation", "Parking", ("hala car park",), 0.96),
]


# Semantic merchant signals used when the exact merchant name is unknown.
# These are intentionally conservative: they classify only strong business
# descriptors and leave ambiguous merchants for review.
SEMANTIC_RULES: List[Tuple[str, str, Tuple[str, ...], float]] = [
    # FOOD & DINING
    ("Food & Dining", "Restaurant", (
        "restaurant", "restauran", "resto", "ristorante", "restaurante",
        "restauracja", "diner", "eatery", "bistro", "brasserie",
        "food court", "canteen", "cafeteria", "grill", "kitchen",
        "steakhouse", "seafood restaurant", "family restaurant",
    ), 0.97),
    ("Food & Dining", "Cafe", (
        "cafe", "coffee shop", "coffeehouse", "coffee house", "espresso bar",
    ), 0.95),
    ("Food & Dining", "Fast Food", (
        "fast food", "burger", "pizza", "fried chicken", "shawarma",
        "sandwich", "taco", "kfc", "mcdonald", "subway",
    ), 0.96),
    ("Food & Dining", "Food Delivery", (
        "food delivery", "food order", "meal delivery", "food delivery app",
    ), 0.97),
    ("Food & Dining", "Bakery", (
        "bakery", "bakeshop", "patisserie", "confectionery",
    ), 0.96),
    ("Food & Dining", "Snacks", (
        "snack bar", "snacks", "juice bar", "dessert shop",
    ), 0.93),

    # GROCERIES
    ("Groceries", "Supermarket", (
        "supermarket", "hypermarket", "super market",
    ), 0.97),
    ("Groceries", "Grocery Store", (
        "grocery store", "grocery", "groceries", "grocer",
    ), 0.95),
    ("Groceries", "Convenience Store", (
        "convenience store", "mini mart", "minimart", "convenience mart",
    ), 0.95),
    ("Groceries", "Fresh Market", (
        "fresh market", "farmers market", "farm market",
    ), 0.94),

    # TRANSPORTATION
    ("Transportation", "Fuel", (
        "fuel station", "petrol station", "gas station", "service station",
        "filling station", "gasoline station", "diesel station",
    ), 0.97),
    ("Transportation", "Taxi", (
        "taxi", "cab fare", "taxicab",
    ), 0.96),
    ("Transportation", "Ride Sharing", (
        "ride sharing", "rideshare", "ride share",
    ), 0.96),
    ("Transportation", "Public Transport", (
        "public transport", "bus fare", "train fare", "metro", "subway",
        "transit",
    ), 0.95),
    ("Transportation", "Parking", (
        "parking", "car park", "parking fee",
    ), 0.96),
    ("Transportation", "Tolls", (
        "toll gate", "road toll", "tollway",
    ), 0.97),

    # HEALTHCARE
    ("Healthcare", "Pharmacy", (
        "pharmacy", "chemist", "drug store", "medical pharmacy",
    ), 0.97),
    ("Healthcare", "Hospital", (
        "hospital", "medical center", "medical centre",
    ), 0.98),
    ("Healthcare", "Doctor", (
        "doctor", "clinic", "medical clinic", "dental clinic", "dentist",
    ), 0.95),
    ("Healthcare", "Diagnostics", (
        "diagnostic center", "diagnostic centre", "diagnostics", "laboratory",
        "pathology",
    ), 0.96),

    # BILLS & UTILITIES
    ("Bills & Utilities", "Electricity", (
        "electricity bill", "electricity payment", "power bill",
    ), 0.98),
    ("Bills & Utilities", "Water", (
        "water bill", "water payment",
    ), 0.98),
    ("Bills & Utilities", "Internet", (
        "internet bill", "broadband", "wifi bill",
    ), 0.97),
    ("Bills & Utilities", "Mobile", (
        "mobile bill", "mobile recharge", "phone bill", "telecom bill",
    ), 0.97),

    # SHOPPING
    ("Shopping", "Clothing", (
        "clothing", "apparel", "fashion store", "garments",
    ), 0.94),
    ("Shopping", "Electronics", (
        "electronics", "electronic store", "computer store", "mobile store",
    ), 0.94),
    ("Shopping", "Home", (
        "home store", "home decor", "household store",
    ), 0.94),

    # TRAVEL
    ("Travel", "Hotels", (
        "hotel", "resort", "inn", "motel", "guest house",
    ), 0.97),
    ("Travel", "Flights", (
        "airline", "airways", "airport", "flight",
    ), 0.96),

    # EDUCATION
    ("Education", "Courses", (
        "training institute", "training center", "training centre",
        "coaching center", "coaching centre",
    ), 0.95),

    # FINANCIAL
    ("Financial", "Bank Fees", (
        "bank fee", "bank charge", "account fee", "service charge",
    ), 0.96),
    ("Financial", "ATM Fees", (
        "atm fee", "cash withdrawal fee",
    ), 0.97),
]


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def merchant_key(transaction) -> str:
    merchant = getattr(transaction, "merchant", None) or getattr(transaction, "description_normalized", None) or getattr(transaction, "description_raw", "")
    key = normalize_text(merchant)
    for suffix in (" abu dhabi are", " abudhabi are", " dubai are", " sharjah are", " are"):
        if key.endswith(suffix):
            key = key[: -len(suffix)].strip()
    key = re.sub(r"\b(llc|l l c|ltd|limited|inc|opc|spc)\b", " ", key)
    return re.sub(r"\s+", " ", key).strip()


class Base(DeclarativeBase):
    pass


class MerchantLearning(Base):
    __tablename__ = "merchant_learning"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    merchant_key: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    merchant_name: Mapped[str] = mapped_column(String(255))
    category: Mapped[str] = mapped_column(String(255))
    subcategory: Mapped[str] = mapped_column(String(255), default="General")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    user_confirmed: Mapped[bool] = mapped_column(Boolean, default=True)
    times_confirmed: Mapped[int] = mapped_column(Integer, default=1)


class CustomCategory(Base):
    __tablename__ = "custom_categories"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    subcategory: Mapped[str] = mapped_column(String(255), default="General")


class CategoryMemory:
    def __init__(self, database_url: Optional[str] = None):
        url = database_url or os.getenv("FINORA_DATABASE_URL") or "sqlite:///finora_learning.db"
        self.engine = create_engine(url, pool_pre_ping=True)
        Base.metadata.create_all(self.engine)

    def get_merchant(self, key: str) -> Optional[MerchantLearning]:
        if not key:
            return None
        with Session(self.engine) as session:
            return session.scalar(select(MerchantLearning).where(MerchantLearning.merchant_key == key))

    def save_merchant(self, key: str, merchant_name: str, category: str, subcategory: str) -> None:
        with Session(self.engine) as session:
            row = session.scalar(select(MerchantLearning).where(MerchantLearning.merchant_key == key))
            if row is None:
                row = MerchantLearning(merchant_key=key, merchant_name=merchant_name, category=category, subcategory=subcategory)
                session.add(row)
            else:
                row.category = category
                row.subcategory = subcategory
                row.confidence = 1.0
                row.user_confirmed = True
                row.times_confirmed += 1
            session.commit()

    def add_category(self, name: str, subcategory: str = "General") -> None:
        with Session(self.engine) as session:
            existing = session.scalar(select(CustomCategory).where(CustomCategory.name == name))
            if existing is None:
                session.add(CustomCategory(name=name, subcategory=subcategory))
                session.commit()

    def list_categories(self) -> List[str]:
        with Session(self.engine) as session:
            return [row.name for row in session.scalars(select(CustomCategory).order_by(CustomCategory.name)).all()]


class HybridCategoryEngine:
    def __init__(self, memory: CategoryMemory):
        self.memory = memory

    def available_categories(self) -> List[str]:
        return list(BASE_CATEGORIES.keys()) + self.memory.list_categories()

    def subcategories_for(self, category: str) -> List[str]:
        if category in BASE_CATEGORIES:
            return BASE_CATEGORIES[category]
        with Session(self.memory.engine) as session:
            rows = session.scalars(select(CustomCategory).where(CustomCategory.name == category)).all()
            return [r.subcategory for r in rows]

    def _transaction_text(self, transaction) -> str:
        values = [
            getattr(transaction, "merchant", None),
            getattr(transaction, "description_normalized", None),
            getattr(transaction, "description_raw", None),
        ]
        return normalize_text(" ".join(str(value or "") for value in values))

    @staticmethod
    def _fuzzy_token_match(text: str, alias: str) -> bool:
        """Catch common PDF/OCR truncations such as 'restauran' -> 'restaurant'."""
        normalized_alias = normalize_text(alias)
        if not normalized_alias or normalized_alias in text:
            return bool(normalized_alias)

        alias_tokens = normalized_alias.split()
        text_tokens = text.split()
        if len(alias_tokens) != 1 or len(normalized_alias) < 6:
            return False

        alias_token = alias_tokens[0]
        for token in text_tokens:
            if abs(len(token) - len(alias_token)) > 2:
                continue
            common = 0
            for left, right in zip(token, alias_token):
                if left != right:
                    break
                common += 1
            if common >= max(6, len(alias_token) - 2):
                return True
        return False

    def _rule_classify(self, transaction):
        text = self._transaction_text(transaction)
        best = None

        # 1) Explicit merchant rules always get first priority.
        for name, category, subcategory, aliases, confidence in RULES:
            for alias in aliases:
                normalized_alias = normalize_text(alias)
                if normalized_alias and normalized_alias in text:
                    score = confidence
                    if best is None or score > best[2]:
                        best = (category, subcategory, score, name)

        # 2) Semantic descriptors handle unknown/global merchants.
        for category, subcategory, aliases, confidence in SEMANTIC_RULES:
            for alias in aliases:
                normalized_alias = normalize_text(alias)
                if normalized_alias and normalized_alias in text:
                    score = confidence
                    if best is None or score > best[2]:
                        best = (category, subcategory, score, f"semantic:{alias}")

        # 3) Fuzzy matching is deliberately limited to long single words so
        # common merchant names do not get incorrectly classified.
        if best is None:
            for category, subcategory, aliases, confidence in SEMANTIC_RULES:
                for alias in aliases:
                    if self._fuzzy_token_match(text, alias):
                        score = max(0.90, confidence - 0.02)
                        if best is None or score > best[2]:
                            best = (category, subcategory, score, f"fuzzy:{alias}")

        return best

    def classify_one(self, transaction):
        key = merchant_key(transaction)
        learned = self.memory.get_merchant(key)
        if learned:
            transaction.category = learned.category
            transaction.subcategory = learned.subcategory
            transaction.category_confidence = 1.0
            transaction.requires_review = False
            return learned.category, learned.subcategory, 1.0, "user_memory"

        rule = self._rule_classify(transaction)
        if rule:
            category, subcategory, confidence, source = rule
            transaction.category = category
            transaction.subcategory = subcategory
            transaction.category_confidence = confidence
            transaction.requires_review = confidence < 0.90
            return category, subcategory, confidence, source

        transaction.category = None
        transaction.subcategory = None
        transaction.category_confidence = 0.0
        transaction.requires_review = True
        return None, None, 0.0, "review"

    def classify_transactions(self, transactions):
        for transaction in transactions:
            self.classify_one(transaction)
        return transactions

    def review_items(self, transactions):
        items = []
        for index, tx in enumerate(transactions):
            if not getattr(tx, "requires_review", False):
                continue
            suggested = getattr(tx, "category", None)
            confidence = float(getattr(tx, "category_confidence", 0.0) or 0.0)
            items.append({
                "index": index,
                "transaction": tx,
                "suggested_category": suggested,
                "confidence": confidence,
            })
        return items

    def learn_from_user(self, transaction, category: str, subcategory: str, apply_all: bool = True, transactions=None):
        key = merchant_key(transaction)
        name = str(getattr(transaction, "merchant", None) or getattr(transaction, "description_raw", "Unknown")).strip()
        self.memory.save_merchant(key, name, category, subcategory)

        transaction.category = category
        transaction.subcategory = subcategory
        transaction.category_confidence = 1.0
        transaction.requires_review = False
        transaction.user_corrected = True

        if apply_all and transactions is not None:
            target_key = key
            for other in transactions:
                if merchant_key(other) != target_key:
                    continue
                other.category = category
                other.subcategory = subcategory
                other.category_confidence = 1.0
                other.requires_review = False
                other.user_corrected = True

    def create_custom_category(self, name: str, subcategory: str = "General"):
        self.memory.add_category(name, subcategory)
