from ingestion.pdf_reader import PDFReader


PDF_PATH = "storage/sample_statement.pdf"


reader = PDFReader(PDF_PATH)

print("Page count:", reader.get_page_count())

pages = reader.extract_pages()

for page in pages:
    print("\n" + "=" * 60)
    print("PAGE:", page["page_number"])
    print("Has text:", page["has_text"])
    print("=" * 60)

    print(page["text"][:1000])
    