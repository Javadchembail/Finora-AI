from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any, Iterable


class StatementRAG:
    """
    Universal, transaction-grounded RAG for Finora AI.

    Design goals:
      - bank/format agnostic
      - deterministic calculations for money questions
      - semantic-ish category inference when the extractor has no category
      - strong merchant/date/amount retrieval
      - compact evidence for the LLM
      - no invented facts
    """

    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "show",
        "tell", "that", "the", "this", "to", "was", "what", "where", "which",
        "who", "with", "you", "did", "do", "much", "many", "spent", "spend",
        "transaction", "transactions", "money", "please", "can", "could", "give",
        "about", "all", "any", "there", "were", "have", "had", "has", "does",
        "mine", "statement", "account", "financial", "finance", "finora", "want",
        "would", "like", "should", "need", "me", "total", "amount", "cost",
    }

    CATEGORY_ALIASES = {
        "food": "Food & Dining", "dining": "Food & Dining", "restaurant": "Food & Dining",
        "restaurants": "Food & Dining", "cafe": "Food & Dining", "cafes": "Food & Dining",
        "grocery": "Groceries", "groceries": "Groceries", "supermarket": "Groceries",
        "shopping": "Shopping", "retail": "Shopping", "store": "Shopping",
        "transport": "Transportation", "transportation": "Transportation", "fuel": "Transportation",
        "petrol": "Transportation", "gas": "Transportation", "taxi": "Transportation",
        "travel": "Travel", "flight": "Travel", "flights": "Travel", "hotel": "Travel",
        "health": "Healthcare", "healthcare": "Healthcare", "medical": "Healthcare",
        "hospital": "Healthcare", "doctor": "Healthcare", "pharmacy": "Healthcare",
        "education": "Education", "school": "Education", "college": "Education",
        "course": "Education", "subscription": "Bills & Utilities", "subscriptions": "Bills & Utilities",
        "utilities": "Bills & Utilities", "electricity": "Bills & Utilities", "internet": "Bills & Utilities",
        "mobile": "Bills & Utilities", "rent": "Housing", "housing": "Housing", "mortgage": "Housing",
        "entertainment": "Entertainment", "movies": "Entertainment", "gaming": "Entertainment",
        "salary": "Income", "income": "Income", "investment": "Investment", "investments": "Investment",
        "fee": "Financial", "fees": "Financial", "tax": "Taxes", "taxes": "Taxes",
    }

    CATEGORY_RULES = {
        "Food & Dining": {"restaurant", "cafe", "cafeteria", "bakery", "biryani", "pizza", "burger", "food", "dining", "grill", "kitchen", "coffee", "meal"},
        "Groceries": {"supermarket", "grocery", "groceries", "hypermarket", "market", "carrefour", "lulu", "waitrose", "spinneys"},
        "Transportation": {"adnoc", "shell", "fuel", "petrol", "gas station", "taxi", "uber", "careem", "transport", "parking", "toll", "metro", "bus"},
        "Healthcare": {"hospital", "clinic", "pharmacy", "medical", "doctor", "dental", "health", "medicine", "diagnostic", "laboratory", "lab", "healthcare"},
        "Travel": {"airline", "airport", "flight", "hotel", "booking", "travel", "visa", "airways"},
        "Education": {"school", "college", "university", "course", "tuition", "training", "education"},
        "Entertainment": {"cinema", "movie", "netflix", "spotify", "gaming", "game", "event", "music"},
        "Shopping": {"amazon", "mall", "store", "fashion", "clothing", "electronics", "retail", "shop"},
        "Bills & Utilities": {"electricity", "water bill", "internet", "telecom", "mobile", "utility", "subscription"},
        "Housing": {"rent", "mortgage", "property", "housing"},
        "Financial": {"bank fee", "service fee", "card fee", "atm fee", "interest charge", "commission"},
        "Taxes": {"tax", "vat", "government fee", "income tax"},
    }

    def __init__(self, transactions: Iterable[Any]):
        self.transactions = list(transactions or [])
        self.index = self._build_index()
        self._merchants = self._unique_merchants()

    @staticmethod
    def _value(tx: Any, field: str, default: Any = None) -> Any:
        if isinstance(tx, dict):
            return tx.get(field, default)
        return getattr(tx, field, default)

    @staticmethod
    def _enum(value: Any) -> str:
        if value is None:
            return ""
        return str(getattr(value, "value", value)).strip().lower()

    @staticmethod
    def _decimal(value: Any) -> Decimal:
        try:
            return abs(Decimal(str(value or 0).replace(",", "")))
        except (InvalidOperation, ValueError, TypeError):
            return Decimal("0")

    @classmethod
    def _money(cls, value: Any) -> str:
        return f"{cls._decimal(value):,.2f}"

    @staticmethod
    def _date_string(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        return str(value)

    @classmethod
    def _clean(cls, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip()) or "Unknown"

    @classmethod
    def _tokens(cls, value: Any) -> set[str]:
        words = re.findall(r"[a-z0-9]+", str(value or "").lower())
        return {w for w in words if len(w) > 1 and w not in cls.STOPWORDS}

    @classmethod
    def _is_uncategorized(cls, value: Any) -> bool:
        return str(value or "").strip().lower() in {"", "unknown", "uncategorized", "other", "none", "null"}

    def _infer_category(self, tx: Any) -> tuple[str | None, float]:
        raw = self._clean(self._value(tx, "category", ""))
        if not self._is_uncategorized(raw):
            return raw, 1.0

        text = " ".join([
            str(self._value(tx, "merchant", "") or ""),
            str(self._value(tx, "description_raw", "") or ""),
            str(self._value(tx, "transaction_type", "") or ""),
        ]).lower()

        best = None
        best_hits = 0
        for category, keywords in self.CATEGORY_RULES.items():
            hits = sum(1 for keyword in keywords if re.search(rf"\b{re.escape(keyword)}\b", text) or keyword in text)
            if hits > best_hits:
                best, best_hits = category, hits

        if best:
            return best, min(0.55 + 0.12 * best_hits, 0.90)
        return None, 0.0

    def _build_index(self) -> list[dict[str, Any]]:
        rows = []
        for position, tx in enumerate(self.transactions):
            merchant = self._clean(self._value(tx, "merchant", ""))
            description = self._clean(self._value(tx, "description_raw", ""))
            category, category_conf = self._infer_category(tx)
            raw_category = self._clean(self._value(tx, "category", ""))
            searchable = " ".join([
                merchant, description, raw_category, category or "",
                str(self._value(tx, "subcategory", "") or ""),
                str(self._value(tx, "transaction_type", "") or ""),
                str(self._value(tx, "transaction_channel", "") or ""),
                str(self._value(tx, "transaction_date", "") or ""),
                str(self._value(tx, "posting_date", "") or ""),
                str(self._value(tx, "original_amount", "") or ""),
                str(self._value(tx, "original_currency", "") or ""),
            ])
            rows.append({
                "position": position,
                "transaction": tx,
                "merchant": merchant,
                "description": description,
                "raw_category": raw_category,
                "category": category,
                "category_confidence": category_conf,
                "tokens": self._tokens(searchable),
                "merchant_tokens": self._tokens(merchant),
                "description_tokens": self._tokens(description),
                "category_tokens": self._tokens(f"{raw_category} {category or ''}"),
                "date": self._date_string(self._value(tx, "transaction_date")),
                "posting_date": self._date_string(self._value(tx, "posting_date")),
                "amount": self._decimal(self._value(tx, "original_amount")),
                "direction": self._enum(self._value(tx, "direction")),
            })
        return rows

    def _unique_merchants(self) -> list[str]:
        seen = set()
        result = []
        for item in self.index:
            name = item["merchant"]
            if name == "Unknown":
                continue
            key = name.casefold()
            if key not in seen:
                seen.add(key)
                result.append(name)
        return result

    def detect_intent(self, question: str) -> dict[str, Any]:
        q = str(question or "").lower().strip()
        category = None
        for alias, canonical in sorted(self.CATEGORY_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", q):
                category = canonical
                break

        if re.search(r"\b(summary|summarize|overview|overall|statement summary)\b", q):
            intent = "summary"
        elif re.search(r"\b(largest|biggest|highest).*(transaction|purchase|payment|expense|spend)|\b(largest|biggest) transaction", q):
            intent = "largest_transaction"
        elif re.search(r"\b(smallest|lowest).*(transaction|purchase|payment|expense|spend)", q):
            intent = "smallest_transaction"
        elif re.search(r"\b(review|flag|attention|concern|unusual|duplicate|duplicates|suspicious|anomal)", q):
            intent = "review"
        elif re.search(r"\b(net|cash flow|remaining|left over|leftover)", q):
            intent = "net"
        elif re.search(r"\b(how much|total).*(income|received|credit|salary|deposit)", q):
            intent = "income"
        elif re.search(r"\b(how much|total).*(spent|expense|expenses|outgoing|debit)", q):
            intent = "expenses"
        elif re.search(r"\b(where|who|merchant).*(spent|spend|most|highest)|\b(top|highest).*(merchant|spend)", q):
            intent = "top_merchant"
        elif category or re.search(r"\b(category|categories|area|areas|healthcare|health|dining|food|grocer)", q):
            intent = "category_spend"
        elif re.search(r"\b(whose|who.*statement|account holder|name on|owner)", q):
            intent = "identity"
        elif re.search(r"\b(merchant|store|shop|company|vendor|at)\b", q):
            intent = "merchant_spend"
        else:
            intent = "general"

        return {"intent": intent, "category": category}

    @staticmethod
    def _similarity(query: str, candidate: str) -> float:
        q = re.sub(r"[^a-z0-9 ]", " ", str(query).lower()).strip()
        c = re.sub(r"[^a-z0-9 ]", " ", str(candidate).lower()).strip()
        if not q or not c:
            return 0.0
        if c in q:
            return 1.0
        return SequenceMatcher(None, q, c).ratio()

    def _retrieve_scored(self, question: str, top_k: int = 12) -> list[tuple[float, dict[str, Any]]]:
        q_tokens = self._tokens(question)
        q_lower = str(question).lower()
        scored = []
        for item in self.index:
            score = 0.0
            score += len(q_tokens & item["tokens"]) * 1.0
            score += len(q_tokens & item["merchant_tokens"]) * 5.0
            score += len(q_tokens & item["description_tokens"]) * 1.5
            score += len(q_tokens & item["category_tokens"]) * 3.0
            merchant = item["merchant"].lower()
            description = item["description"].lower()
            if merchant != "unknown" and merchant in q_lower:
                score += 10.0
            if description != "unknown" and len(description) > 5 and description in q_lower:
                score += 6.0
            amount_text = self._money(item["amount"])
            if amount_text in q_lower or amount_text.replace(",", "") in q_lower:
                score += 6.0
            if score > 0:
                scored.append((score, item))

        if q_tokens:
            q_phrase = " ".join(sorted(q_tokens))
            for item in self.index:
                if item["merchant"] == "Unknown":
                    continue
                similarity = self._similarity(q_phrase, item["merchant"])
                if similarity >= 0.72:
                    scored.append((similarity * 5.0, item))

        scored.sort(key=lambda pair: (pair[0], -pair[1]["position"]), reverse=True)
        result, seen = [], set()
        for score, item in scored:
            if item["position"] in seen:
                continue
            seen.add(item["position"])
            result.append((score, item))
            if len(result) >= top_k:
                break
        return result

    def retrieve(self, question: str, top_k: int = 12) -> list[dict[str, Any]]:
        return [self._serialize_transaction(item["transaction"], score, item) for score, item in self._retrieve_scored(question, top_k)]

    def _serialize_transaction(self, tx: Any, score: float = 0.0, item: dict[str, Any] | None = None) -> dict[str, Any]:
        category, confidence = self._infer_category(tx)
        return {
            "transaction_id": self._value(tx, "transaction_id"),
            "transaction_date": self._date_string(self._value(tx, "transaction_date")),
            "posting_date": self._date_string(self._value(tx, "posting_date")),
            "description": self._value(tx, "description_raw", ""),
            "merchant": self._value(tx, "merchant") or self._value(tx, "description_raw", ""),
            "amount": self._money(self._value(tx, "original_amount")),
            "currency": self._value(tx, "original_currency"),
            "direction": self._enum(self._value(tx, "direction")),
            "transaction_type": self._enum(self._value(tx, "transaction_type")),
            "category": self._value(tx, "category") or None,
            "inferred_category": category,
            "inferred_category_confidence": round(confidence, 2),
            "subcategory": self._value(tx, "subcategory"),
            "balance": self._money(self._value(tx, "running_balance")),
            "source_page": self._value(tx, "source_page"),
            "requires_review": bool(self._value(tx, "requires_review", False)),
        }

    def _spending_rows(self):
        return [x for x in self.index if x["direction"] == "debit"]

    def _income_rows(self):
        return [x for x in self.index if x["direction"] == "credit"]

    def _currency(self):
        currencies = [str(self._value(x["transaction"], "original_currency", "")).upper() for x in self.index]
        currencies = [x for x in currencies if x]
        return Counter(currencies).most_common(1)[0][0] if currencies else "UNKNOWN"

    def _aggregate_merchants(self):
        totals, counts = defaultdict(Decimal), defaultdict(int)
        for item in self._spending_rows():
            merchant = item["merchant"] if item["merchant"] != "Unknown" else item["description"]
            totals[merchant] += item["amount"]
            counts[merchant] += 1
        return sorted([
            {"merchant": k, "amount": float(v), "count": counts[k]}
            for k, v in totals.items()
        ], key=lambda x: x["amount"], reverse=True)

    def _aggregate_inferred_categories(self):
        totals, counts, confs = defaultdict(Decimal), defaultdict(int), defaultdict(list)
        for item in self._spending_rows():
            category = item["category"] or "Uncategorized"
            if not category:
                category = "Uncategorized"
            totals[category] += item["amount"]
            counts[category] += 1
            confs[category].append(item["category_confidence"])
        return sorted([
            {
                "category": k,
                "amount": float(v),
                "count": counts[k],
                "confidence": round(sum(confs[k]) / len(confs[k]), 2) if confs[k] else 0.0,
            }
            for k, v in totals.items()
        ], key=lambda x: x["amount"], reverse=True)

    def analytics(self):
        spending = self._spending_rows()
        income = self._income_rows()
        expense_total = sum((x["amount"] for x in spending), Decimal("0"))
        income_total = sum((x["amount"] for x in income), Decimal("0"))
        merchants = self._aggregate_merchants()
        categories = self._aggregate_inferred_categories()
        largest = sorted(spending, key=lambda x: x["amount"], reverse=True)[:10]
        smallest = sorted([x for x in spending if x["amount"] > 0], key=lambda x: x["amount"])[:5]
        review = [x for x in self.index if bool(self._value(x["transaction"], "requires_review", False))]
        duplicates = [x for x in review if "duplicate" in str(self._value(x["transaction"], "notes", "")).lower()]
        categorized = [x for x in spending if x["category"] and x["category_confidence"] > 0]
        coverage = (len(categorized) / len(spending) * 100) if spending else 0.0
        return {
            "currency": self._currency(),
            "transaction_count": len(self.transactions),
            "income": float(income_total),
            "expenses": float(expense_total),
            "net_cash_flow": float(income_total - expense_total),
            "top_merchants": merchants[:10],
            "top_categories": categories[:10],
            "largest_transactions": [self._serialize_transaction(x["transaction"], 0, x) for x in largest],
            "smallest_transactions": [self._serialize_transaction(x["transaction"], 0, x) for x in smallest],
            "review_count": len(review),
            "duplicate_review_count": len(duplicates),
            "category_coverage_percent": round(coverage, 1),
        }

    def _find_merchant(self, question: str) -> str | None:
        q = str(question).lower()
        direct = [(len(m), m) for m in self._merchants if m.lower() in q]
        if direct:
            return max(direct)[1]
        candidates = [(self._similarity(question, m), m) for m in self._merchants]
        candidates = [x for x in candidates if x[0] >= 0.58]
        return max(candidates)[1] if candidates else None

    def answer_context(self, question: str, top_k: int = 12) -> dict[str, Any]:
        intent_info = self.detect_intent(question)
        intent = intent_info["intent"]
        requested_category = intent_info["category"]
        analytics = self.analytics()
        matches = self.retrieve(question, top_k)
        computed: dict[str, Any] = {"intent": intent}

        if intent == "summary":
            computed["answer"] = {
                "currency": analytics["currency"],
                "transactions": analytics["transaction_count"],
                "income": analytics["income"],
                "expenses": analytics["expenses"],
                "net_cash_flow": analytics["net_cash_flow"],
                "top_merchant": analytics["top_merchants"][0] if analytics["top_merchants"] else None,
                "review_count": analytics["review_count"],
            }
        elif intent == "top_merchant":
            computed["answer"] = analytics["top_merchants"][:5]
        elif intent == "largest_transaction":
            computed["answer"] = analytics["largest_transactions"][:5]
        elif intent == "smallest_transaction":
            computed["answer"] = analytics["smallest_transactions"]
        elif intent == "income":
            computed["answer"] = {"total_income": analytics["income"], "currency": analytics["currency"]}
        elif intent == "expenses":
            computed["answer"] = {"total_expenses": analytics["expenses"], "currency": analytics["currency"]}
        elif intent == "net":
            computed["answer"] = {"net_cash_flow": analytics["net_cash_flow"], "currency": analytics["currency"]}
        elif intent == "review":
            computed["answer"] = {
                "review_count": analytics["review_count"],
                "duplicate_review_count": analytics["duplicate_review_count"],
                "largest_spending_concentration": analytics["top_merchants"][:3],
            }
        elif intent == "category_spend":
            category = requested_category
            if category:
                rows = [x for x in self._spending_rows() if (x["category"] or "").casefold() == category.casefold()]
                total = sum((x["amount"] for x in rows), Decimal("0"))
                computed["category"] = category
                computed["answer"] = {
                    "matched_transactions": len(rows),
                    "total": float(total),
                    "currency": analytics["currency"],
                    "confidence": "inferred" if rows and any(x["category_confidence"] < 1 for x in rows) else "canonical",
                }
                if rows:
                    matches = [self._serialize_transaction(x["transaction"], 2.0, x) for x in sorted(rows, key=lambda x: x["amount"], reverse=True)[:top_k]]
            else:
                computed["answer"] = analytics["top_categories"][:10]
        elif intent == "merchant_spend":
            merchant = self._find_merchant(question)
            if merchant:
                rows = [x for x in self._spending_rows() if x["merchant"].casefold() == merchant.casefold()]
                total = sum((x["amount"] for x in rows), Decimal("0"))
                computed["answer"] = {"merchant": merchant, "total": float(total), "count": len(rows), "currency": analytics["currency"]}
                matches = [self._serialize_transaction(x["transaction"], 3.0, x) for x in sorted(rows, key=lambda x: x["amount"], reverse=True)[:top_k]]
            else:
                computed["answer"] = None
        elif intent == "identity":
            holders = []
            for x in self.index:
                holder = self._value(x["transaction"], "account_holder")
                if holder and str(holder).strip() not in holders:
                    holders.append(str(holder).strip())
            computed["answer"] = {"account_holders": holders[:5], "available": bool(holders)}

        return {
            "query": question,
            "intent": intent_info,
            "computed_evidence": computed,
            "statement_analytics": analytics,
            "match_count": len(matches),
            "matches": matches,
        }

    def context(self, question: str, top_k: int = 12) -> dict[str, Any]:
        return self.answer_context(question, top_k)

    def search(self, question: str, top_k: int = 12):
        return self.retrieve(question, top_k)

    def summary(self):
        return self.analytics()
