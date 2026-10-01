import pdfplumber


def show_layout(name, path, password=None):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    with pdfplumber.open(path, password=password) as pdf:
        page = pdf.pages[0]

        print("Page size:", page.width, "x", page.height)

        words = page.extract_words(
            use_text_flow=False,
            keep_blank_chars=False,
        )

        # Group words into visual lines.
        lines = []

        for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
            added = False

            for line in lines:
                if abs(line["top"] - word["top"]) <= 3:
                    line["words"].append(word)
                    added = True
                    break

            if not added:
                lines.append(
                    {
                        "top": word["top"],
                        "words": [word],
                    }
                )

        for line in lines:
            line["words"].sort(key=lambda w: w["x0"])

        keywords = [
            "date",
            "particular",
            "description",
            "withdraw",
            "deposit",
            "balance",
            "amount",
            "tran",
            "post",
            "value",
            "opening",
        ]

        for line in lines:
            text = " ".join(
                word["text"]
                for word in line["words"]
            )

            lower = text.lower()

            if any(keyword in lower for keyword in keywords):
                print()
                print(
                    f"TOP={line['top']:.1f}"
                )
                print("TEXT:", text)

                print("WORDS:")

                for word in line["words"]:
                    print(
                        f"  {word['text']!r:30}"
                        f" x0={word['x0']:.1f}"
                        f" x1={word['x1']:.1f}"
                    )


show_layout(
    "FEDERAL BANK",
    "storage/federal_statement.pdf",
    password="JAVA2301",
)

show_layout(
    "EMIRATES ISLAMIC",
    "storage/sample_statement.pdf",
)