from transactions.merchant_normalizer import (
    MerchantNormalizer,
)


normalizer = MerchantNormalizer()


test_descriptions = [
    "FAITH CAFETERIA LLC ABU DHABI ARE",
    "ADNOC IND AREA 447 SHARJAH ARE",
    "TAWASUL TRANSPORT LLC ABU DHABI ARE",
    "AGHADEER CAFETERIA LLC ABU DHABI ARE",
    "AL DAWRI CAFETERIA SHARJAH ARE",
    "MAWARED AL JOOD MANDI DUBAI ARE",
    "ADNOC CAR CARE CENTER ABU DHABI ARE",
    "E& DIGITAL APP ABU DHABI ARE",
    "AUTONOVA FOR CAR WASH ABU DHABI ARE",
    "TAP*KEETA DUBAI ARE",
    "TRANSFER PAYMENT RECEIVED THANK YOU",
]


print("\n" + "=" * 70)
print("FINORA MERCHANT NORMALIZER")
print("=" * 70)


for description in test_descriptions:

    merchant, confidence = (
        normalizer.normalize(
            description
        )
    )

    print("\nRaw:")
    print(description)

    print("Merchant:")
    print(merchant)

    print("Confidence:")
    print(confidence)