from src.config.settings import settings
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.structure_analyzer import StructureAnalyzer


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - STRUCTURE ANALYSIS TEST")
    print("=" * 60)

    parser = PDFParser(settings.pdf_path)
    pages = parser.parse()

    analyzer = StructureAnalyzer()
    structures = analyzer.analyze(pages)

    print(f"\nPages analyzed: {len(structures)}")

    total_headings = sum(
        len(page.headings)
        for page in structures
    )

    total_subheadings = sum(
        len(page.subheadings)
        for page in structures
    )

    total_sections = sum(
        len(page.sections)
        for page in structures
    )

    total_lists = sum(
        len(page.lists)
        for page in structures
    )

    total_tables = sum(
        len(page.tables)
        for page in structures
    )

    print("\nDetected structure:")
    print(f"  Headings:     {total_headings}")
    print(f"  Subheadings:  {total_subheadings}")
    print(f"  Sections:     {total_sections}")
    print(f"  List items:   {total_lists}")
    print(f"  Table lines:  {total_tables}")

    print("\nSample detected structure:")
    print("-" * 60)

    shown = 0

    for page in structures:
        if (
            page.headings
            or page.subheadings
            or page.sections
        ):
            print(f"\nPage {page.page_number}")

            if page.headings:
                print("  Headings:")
                for heading in page.headings[:5]:
                    print(f"    - {heading}")

            if page.subheadings:
                print("  Subheadings:")
                for subheading in page.subheadings[:5]:
                    print(f"    - {subheading}")

            if page.sections:
                print("  Sections:")
                for section in page.sections[:5]:
                    print(f"    - {section}")

            shown += 1

            if shown >= 10:
                break

    print("\nStructure analysis test passed.")


if __name__ == "__main__":
    main()