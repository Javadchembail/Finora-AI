from __future__ import annotations

import re
from typing import Optional, Tuple


class MerchantNormalizer:
    """
    Rule-based merchant normalization.

    The original description is never modified.

    Example:

        FAITH CAFETERIA LLC ABU DHABI ARE
        ->
        Faith Cafeteria

    The normalized merchant is intended for:
        - dashboards
        - categorization
        - analytics
        - search
        - user corrections
        - AI analysis
    """

    # -----------------------------------------------------
    # Exact known brands
    # -----------------------------------------------------

    KNOWN_MERCHANTS = {
        "ADNOC": "ADNOC",
        "E&": "E&",
        "KEETA": "Keeta",
    }

    # -----------------------------------------------------
    # Payment/system descriptions
    # -----------------------------------------------------

    NON_MERCHANT_PATTERNS = [
        r"\bTRANSFER\s+PAYMENT\b",
        r"\bPAYMENT\s+RECEIVED\b",
        r"\bPAYMENT\s+THANK\s+YOU\b",
        r"\bTHANK\s+YOU\b",
        r"\bMINIMUM\s+PAYMENT\b",
        r"\bTOTAL\s+PAYMENT\b",
        r"\bCASH\s+ADVANCE\b",
        r"\bBALANCE\s+TRANSFER\b",
    ]

    # -----------------------------------------------------
    # Common prefixes
    # -----------------------------------------------------

    PREFIX_PATTERNS = [
        r"^TAP\*\s*",
        r"^TAP\s*\*\s*",
        r"^POS\s+",
        r"^ECOM\s+",
        r"^PURCHASE\s+",
        r"^CARD\s+PURCHASE\s+",
    ]

    # -----------------------------------------------------
    # Business suffixes
    # -----------------------------------------------------

    BUSINESS_SUFFIXES = {
        "LLC",
        "LTD",
        "LIMITED",
        "INC",
        "CORP",
        "CORPORATION",
        "COMPANY",
        "CO",
        "EST",
        "ESTABLISHMENT",
    }

    # -----------------------------------------------------
    # Words that usually indicate location information
    # -----------------------------------------------------

    LOCATION_WORDS = {
        "ABU",
        "DHABI",
        "DUBAI",
        "SHARJAH",
        "AJMAN",
        "RAK",
        "RAS",
        "AIN",
        "FUJAIRAH",
        "UMM",
        "QUWAIN",
        "ARE",
        "UAE",
    }

    # -----------------------------------------------------
    # Words that describe the merchant rather than being
    # the actual merchant name.
    # -----------------------------------------------------

    DESCRIPTOR_PATTERNS = [
        r"\bFOR\s+CAR\s+WASH\b.*$",
        r"\bCAR\s+WASH\b.*$",
        r"\bDIGITAL\s+APP\b.*$",
    ]

    # -----------------------------------------------------
    # Area/location markers
    # -----------------------------------------------------

    LOCATION_MARKERS = {
        "IND",
        "AREA",
        "STREET",
        "ROAD",
        "RD",
        "BRANCH",
        "SHOP",
        "STORE",
        "MALL",
    }

    def normalize(
        self,
        description: str,
    ) -> Tuple[Optional[str], float]:

        if not description:
            return None, 0.0

        raw = description.strip()

        if not raw:
            return None, 0.0

        upper = raw.upper()

        # =================================================
        # 1. Ignore payment/system transactions
        # =================================================

        for pattern in self.NON_MERCHANT_PATTERNS:

            if re.search(
                pattern,
                upper,
                flags=re.IGNORECASE,
            ):
                return None, 0.99

        # =================================================
        # 2. Remove payment prefixes
        # =================================================

        cleaned = upper

        for pattern in self.PREFIX_PATTERNS:

            cleaned = re.sub(
                pattern,
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

        cleaned = re.sub(
            r"\s+",
            " ",
            cleaned,
        ).strip()

        # =================================================
        # 3. Known merchants
        # =================================================

        if cleaned.startswith("ADNOC"):

            return "ADNOC", 0.95

        if cleaned.startswith("E&"):

            return "E&", 0.95

        if cleaned.startswith("KEETA"):

            return "Keeta", 0.95

        # =================================================
        # 4. Remove obvious descriptive suffixes
        # =================================================

        for pattern in self.DESCRIPTOR_PATTERNS:

            cleaned = re.sub(
                pattern,
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()

        # =================================================
        # 5. Tokenize
        # =================================================

        words = cleaned.split()

        if not words:
            return None, 0.0

        # =================================================
        # 6. Remove company suffixes
        # =================================================

        filtered_words = []

        for word in words:

            if word in self.BUSINESS_SUFFIXES:
                continue

            filtered_words.append(word)

        words = filtered_words

        # =================================================
        # 7. Stop at clear geographic information
        # =================================================

        merchant_words = []

        for index, word in enumerate(words):

            # ---------------------------------------------
            # Explicit location
            # ---------------------------------------------

            if word in self.LOCATION_WORDS:

                break

            # ---------------------------------------------
            # Area/branch markers
            # ---------------------------------------------

            if word in self.LOCATION_MARKERS:

                break

            merchant_words.append(word)

        # =================================================
        # 8. Special handling for merchants containing
        #    "AL" as part of their actual name
        # =================================================

        if not merchant_words:

            merchant_words = words

        # =================================================
        # 9. Merchant-specific cleanup
        # =================================================

        merchant_upper = " ".join(
            merchant_words
        ).strip()

        # ADNOC variations
        if merchant_upper.startswith("ADNOC"):
            return "ADNOC", 0.95

        # E& variations
        if merchant_upper.startswith("E&"):
            return "E&", 0.95

        # Keeta variations
        if merchant_upper.startswith("KEETA"):
            return "Keeta", 0.95

        # =================================================
        # 10. Remove unnecessary "FOR"
        # =================================================

        merchant_upper = re.sub(
            r"\s+FOR$",
            "",
            merchant_upper,
        ).strip()

        # =================================================
        # 11. Final cleanup
        # =================================================

        merchant_upper = re.sub(
            r"\s+",
            " ",
            merchant_upper,
        ).strip()

        merchant_upper = merchant_upper.strip(
            " -*_.,"
        )

        if not merchant_upper:
            return None, 0.0

        # =================================================
        # 12. Format merchant name
        # =================================================

        merchant = self._format_merchant(
            merchant_upper
        )

        # =================================================
        # 13. Confidence
        # =================================================

        confidence = self._calculate_confidence(
            raw,
            merchant,
        )

        return merchant, confidence

    # -----------------------------------------------------
    # Formatting
    # -----------------------------------------------------

    def _format_merchant(
        self,
        merchant: str,
    ) -> str:

        upper = merchant.upper()

        # Exact known names
        if upper in self.KNOWN_MERCHANTS:

            return self.KNOWN_MERCHANTS[
                upper
            ]

        words = merchant.split()

        formatted = []

        for word in words:

            if word == "&":

                formatted.append("&")

            elif word.isdigit():

                formatted.append(word)

            else:

                formatted.append(
                    word.capitalize()
                )

        return " ".join(formatted)

    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    def _calculate_confidence(
        self,
        raw: str,
        merchant: str,
    ) -> float:

        if not merchant:

            return 0.0

        confidence = 0.80

        # Raw description contained extra information
        # that was successfully removed.
        if len(raw) > len(merchant):

            confidence += 0.05

        # Short clean merchant names are easier to identify.
        if len(merchant.split()) <= 4:

            confidence += 0.05

        return min(
            round(confidence, 2),
            0.95,
        )