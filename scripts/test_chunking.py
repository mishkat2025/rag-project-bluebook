from collections import Counter

from src.config.settings import settings
from src.ingestion.chunker import StructureAwareChunker
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.structure_analyzer import StructureAnalyzer


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - CHUNKING TEST")
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

    print(f"\nPages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")

    print("\nContent types:")
    counts = Counter(chunk.content_type for chunk in chunks)

    for content_type, count in counts.items():
        print(f"  {content_type}: {count}")

    print("\nFirst 5 chunks:")
    print("-" * 60)

    for chunk in chunks[:5]:
        print(f"\nChunk position: {chunk.chunk_position}")
        print(f"Page: {chunk.page}")
        print(f"Section: {chunk.section}")
        print(f"Heading: {chunk.heading}")
        print(f"Content type: {chunk.content_type}")
        print(f"Characters: {len(chunk.text)}")
        print(f"Text:\n{chunk.text[:500]}")

    print("\nChunking test passed.")


if __name__ == "__main__":
    main()