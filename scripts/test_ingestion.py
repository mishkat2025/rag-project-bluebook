from src.config.settings import settings
from src.ingestion.pdf_parser import PDFParser


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - PDF INGESTION TEST")
    print("=" * 60)

    print(f"PDF: {settings.pdf_path}")

    parser = PDFParser(settings.pdf_path)
    pages = parser.parse()

    print(f"\nTotal pages extracted: {len(pages)}")

    print("\nFirst page:")
    print("-" * 60)
    print(pages[0].text[:2000])

    print("\nLast page:")
    print("-" * 60)
    print(pages[-1].text[:1000])

    empty_pages = [
        page.page_number
        for page in pages
        if not page.text.strip()
    ]

    print("\nValidation:")
    print(f"  Empty pages: {len(empty_pages)}")

    if empty_pages:
        print(f"  Empty page numbers: {empty_pages}")
    else:
        print("  All pages contain extracted text.")

    print("\nPDF parser test passed.")


if __name__ == "__main__":
    main()