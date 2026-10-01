from ingestion.universal_structure import (
    UniversalStructureDetector,
)


def inspect(
    name,
    path,
    password=None,
):

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    detector = UniversalStructureDetector()

    result = detector.detect(
        path,
        password=password,
    )

    print("Detected:", result["detected"])
    print("Page:", result["page"])
    print("Score:", result["score"])

    print("\nColumns:")

    for column in result["columns"]:

        print(
            f"  {column['header']:25}"
            f" -> {column['semantic_type']:20}"
            f"x0={column['x0']:7.2f} "
            f"x1={column['x1']:7.2f}"
        )


inspect(
    "FEDERAL BANK",
    "storage/federal_statement.pdf",
    password="JAVA2301",
)


inspect(
    "EMIRATES ISLAMIC",
    "storage/sample_statement.pdf",
)