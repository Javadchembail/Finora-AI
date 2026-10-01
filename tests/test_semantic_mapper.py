from ingestion.semantic_mapper import SemanticHeaderMapper


mapper = SemanticHeaderMapper()


headers = [
    "Date",
    "Value Date",
    "Particulars",
    "Tran ID",
    "Withdrawals",
    "Deposits",
    "Balance",
    "Post Date",
    "Trxn. Date",
    "Description",
    "Amount",
]


print("=" * 70)
print("SEMANTIC HEADER MAPPER")
print("=" * 70)

for header in headers:

    result = mapper.map_header(header)

    print(
        f"{header:25} -> {result}"
    )