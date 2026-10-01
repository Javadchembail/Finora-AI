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
    # Food & Dining
    ("Swiggy", "Food & Dining", "Food Delivery", ("swiggy",), 0.99),
    ("Zomato", "Food & Dining", "Food Delivery", ("zomato",), 0.99),
    ("Talabat", "Food & Dining", "Food Delivery", ("talabat",), 0.99),
    ("Deliveroo", "Food & Dining", "Food Delivery", ("deliveroo",), 0.99),
    ("Careem Food", "Food & Dining", "Food Delivery", ("careem food",), 0.99),
    ("KFC", "Food & Dining", "Fast Food", ("kfc",), 0.99),
    ("McDonald's", "Food & Dining", "Fast Food", ("mcdonald", "mcdonalds"), 0.99),
    ("Burger King", "Food & Dining", "Fast Food", ("burger king",), 0.99),
    ("Starbucks", "Food & Dining", "Cafe", ("starbucks",), 0.99),
    ("Costa Coffee", "Food & Dining", "Cafe", ("costa coffee",), 0.98),
    ("Restaurant", "Food & Dining", "Restaurant", ("restaurant", "rest.", "rest "), 0.94),
    ("Cafe", "Food & Dining", "Cafe", ("cafe", "coffee shop", "cafeteria"), 0.94),
    ("Bakery", "Food & Dining", "Bakery", ("bakery",), 0.95),
    ("Food Delivery", "Food & Dining", "Food Delivery", ("food delivery", "food order"), 0.94),

    # Entertainment
    ("BookMyShow", "Entertainment", "Movies", ("bookmyshow",), 0.99),
    ("Star Cinemas", "Entertainment", "Movies", ("star cinemas", "starcinemas"), 0.98),
    ("Cinema", "Entertainment", "Movies", ("cinema", "cinemas", "multiplex"), 0.96),
    ("Netflix", "Entertainment", "Streaming", ("netflix",), 0.99),
    ("Amazon Prime", "Entertainment", "Streaming", ("amazon prime",), 0.98),
    ("Disney+", "Entertainment", "Streaming", ("disney plus", "disney+"), 0.98),
    ("Spotify", "Entertainment", "Music", ("spotify",), 0.98),
    ("Gaming", "Entertainment", "Gaming", ("playstation", "xbox", "steam games", "gaming"), 0.95),
    ("Events", "Entertainment", "Events", ("event ticket", "events", "ticketing"), 0.92),

    # Healthcare
    ("Medeor Medicals", "Healthcare", "Pharmacy", ("medeor medicals", "medeor pharmacy"), 0.99),
    ("Aster Pharmacy", "Healthcare", "Pharmacy", ("aster pharmacy",), 0.99),
    ("Life Pharmacy", "Healthcare", "Pharmacy", ("life pharmacy",), 0.99),
    ("Pharmacy", "Healthcare", "Pharmacy", ("pharmacy", "medical store", "medicals", "chemist"), 0.96),
    ("Hospital", "Healthcare", "Hospital", ("hospital", "clinic", "medical center", "medical centre"), 0.96),
    ("Doctor", "Healthcare", "Doctor", ("doctor", "dr.", "physician"), 0.93),
    ("Diagnostics", "Healthcare", "Diagnostics", ("diagnostic", "laboratory", "lab test", "pathology"), 0.94),
    ("Health Insurance", "Healthcare", "Health Insurance", ("health insurance",), 0.97),

    # Groceries
    ("Lulu", "Groceries", "Supermarket", ("lulu hypermarket", "luluhypermarket", "lulu"), 0.95),
    ("Carrefour", "Groceries", "Supermarket", ("carrefour",), 0.98),
    ("Nesto", "Groceries", "Supermarket", ("nesto hyper", "nesto"), 0.97),
    ("Supermarket", "Groceries", "Supermarket", ("supermarket", "hypermarket"), 0.96),
    ("Grocery Store", "Groceries", "Grocery Store", ("grocery", "groceries", "baqala", "mini mart", "minimart"), 0.94),
    ("Online Grocery", "Groceries", "Online Grocery", ("blinkit", "instacart", "zepto", "bigbasket"), 0.97),

    # Transportation
    # Specific ADNOC service descriptions should beat the generic ADNOC fuel rule.
    ("ADNOC Restaurant", "Food & Dining", "Restaurant", ("adnoc samha rest", "adnoc restaurant"), 0.97),
    ("ADNOC Car Care", "Transportation", "Vehicle Maintenance", ("adnoc car care", "adnoc carcare"), 0.98),
    ("ADNOC", "Transportation", "Fuel", ("adnoc",), 0.88),
    ("Fuel", "Transportation", "Fuel", ("fuel station", "petrol", "gas station", "service station", "fuel"), 0.95),
    ("Uber", "Transportation", "Ride Sharing", ("uber",), 0.98),
    ("Careem", "Transportation", "Ride Sharing", ("careem",), 0.98),
    ("Taxi", "Transportation", "Taxi", ("taxi", "cab"), 0.96),
    ("Public Transport", "Transportation", "Public Transport", ("metro", "bus", "tram", "public transport"), 0.94),
    ("Parking", "Transportation", "Parking", ("parking", "car park", "q mobility", "hala car park"), 0.96),
    ("Tolls", "Transportation", "Tolls", ("toll", "salik"), 0.97),
    ("Vehicle Maintenance", "Transportation", "Vehicle Maintenance", ("car care", "auto service", "vehicle maintenance", "tyre", "tire"), 0.95),

    # Shopping
    ("Max", "Shopping", "Clothing", ("max dubai", "max fashion", "max "), 0.94),
    ("Miniso", "Shopping", "General Shopping", ("miniso",), 0.98),
    ("Electronics", "Shopping", "Electronics", ("electronics", "computer store", "computer"), 0.91),
    ("Clothing", "Shopping", "Clothing", ("clothing", "apparel", "fashion"), 0.91),
    ("Personal Care", "Shopping", "Personal Care", ("salon", "beauty", "spa", "personal care"), 0.91),
    ("Flowers & Gifts", "Shopping", "Flowers & Gifts", ("flower", "flowers", "gift shop", "gifts"), 0.91),

    # Travel
    ("Flights", "Travel", "Flights", ("airline", "airways", "flight", "emirates airline", "etihad"), 0.95),
    ("Hotels", "Travel", "Hotels", ("hotel", "resort"), 0.95),
    ("Travel Booking", "Travel", "Travel Booking", ("makemytrip", "booking.com", "agoda", "expedia", "cleartrip"), 0.96),
    ("Visa", "Travel", "Visa", ("visa service", "visa application"), 0.93),

    # Bills & Utilities
    ("E&", "Bills & Utilities", "Mobile", ("e& digital app", "e& - dubai", "e& digital"), 0.96),
    ("Mobile", "Bills & Utilities", "Mobile", ("mobile recharge", "mobile bill", "prepaid recharge", "telecom"), 0.94),
    ("Internet", "Bills & Utilities", "Internet", ("internet bill", "broadband", "fiber internet"), 0.94),
    ("Electricity", "Bills & Utilities", "Electricity", ("electricity bill", "electricity"), 0.94),
    ("Water", "Bills & Utilities", "Water", ("water bill",), 0.94),
    ("Subscription", "Bills & Utilities", "Subscription", ("subscription", "monthly plan"), 0.90),

    # Investments / financial
    ("Zerodha", "Investments", "Brokerage", ("zerodha",), 0.99),
    ("Groww", "Investments", "Brokerage", ("groww",), 0.99),
    ("Upstox", "Investments", "Brokerage", ("upstox",), 0.99),
    ("Brokerage", "Investments", "Brokerage", ("brokerage", "securities", "demat"), 0.92),
    ("Bank Fee", "Financial", "Bank Fees", ("bank fee", "service charge", "account fee"), 0.94),
    ("ATM Fee", "Financial", "ATM Fees", ("atm fee",), 0.95),
    ("Interest", "Financial", "Interest", ("interest charge", "interest"), 0.90),

    # Transfers / cash — these are intentionally lower confidence for anonymous descriptions.
    ("ATM Withdrawal", "Cash", "ATM Withdrawal", ("atm withdrawal", "cash withdrawal"), 0.97),
    ("UPI Transfer", "Transfers", "UPI Transfer", ("upi transfer",), 0.91),
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

    def _rule_classify(self, transaction):
        text = normalize_text(
            " ".join(
                str(x or "")
                for x in [getattr(transaction, "merchant", None), getattr(transaction, "description_raw", "")]
            )
        )
        best = None
        for name, category, subcategory, aliases, confidence in RULES:
            for alias in aliases:
                if normalize_text(alias) and normalize_text(alias) in text:
                    score = confidence
                    if best is None or score > best[2]:
                        best = (category, subcategory, score, name)
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
