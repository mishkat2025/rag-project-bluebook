from collections import Counter

from src.config.settings import settings
from src.ingestion.chunker import StructureAwareChunker
from src.ingestion.metadata_builder import MetadataBuilder
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.structure_analyzer import StructureAnalyzer


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - METADATA TEST")
    print("=" * 60)

    parser = PDFParser(settings.pdf_path)
    pages = parser.parse()

    analyzer = StructureAnalyzer()
    structures = analyzer.analyze(pages)

    chunker = StructureAwareChunker(
        max_characters=1800,
        overlap_characters=250,
    )

    chunks = chunker.chunk(
        pages=pages,
        structures=structures,
    )

    builder = MetadataBuilder(
        source_name=settings.pdf_path.name,
    )

    records = builder.build(chunks)

    print(f"\nPages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Metadata records: {len(records)}")

    print("\nFirst metadata record:")
    print("-" * 60)

    for key, value in records[0].items():
        print(f"{key}: {value}")

    content_types = Counter(
        record["content_type"]
        for record in records
    )

    programs = [
        record
        for record in records
        if record["program"] is not None
    ]

    print("\nContent types:")

    for content_type, count in content_types.items():
        print(f"  {content_type}: {count}")

    print(f"\nChunks with detected program: {len(programs)}")

    print("\nSample program metadata:")

    for record in programs[:10]:
        print(
            f"  Page {record['page']}: "
            f"{record['program']}"
        )

    print("\nMetadata test passed.")


if __name__ == "__main__":
    main()